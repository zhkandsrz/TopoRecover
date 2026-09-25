"""Portable entry point for the frozen TopoRecover policy.

Archived-attempt mode reruns finalization, not model inference. Model mode runs
the public-input policy with grammar-constrained local Hugging Face inference.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from io_utils import indexed, write_rows
from toporeward.stage_a_public_parameter_policy import prepare_public_parameter_intent, submit_public_parameter_intent
from toporeward.stage_a_geometry_constrained_fallback import finalize_with_geometry_constraints
from toporeward.packet_localization_ablation import region_selection, SELECTORS


class LocalModel:
    def __init__(self, model_name):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        if not torch.cuda.is_available():
            raise RuntimeError('Fresh paper-style inference requires a CUDA GPU.')
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                        attn_implementation='sdpa').to('cuda').eval()
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)

    def __call__(self, requests, *, grammar_fn=None, max_new_tokens=2048):
        from lmformatenforcer import RegexParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        from grammar_source import request_grammar
        tok, torch = self.tokenizer, self.torch
        inputs = [tok.apply_chat_template([{'role': 'user', 'content': q['prompt']}],
                  tokenize=True, add_generation_prompt=True) for q in requests]
        blocked = {q['case_id'] for q, ids in zip(requests, inputs) if len(ids) > 8192}
        result = []
        with torch.inference_mode():
            for q, ids in zip(requests, inputs):
                r = dict(source_uid=q['source_uid'], case_id=q['case_id'], input_tokens=len(ids))
                if q['case_id'] in blocked:
                    r.update(response='', output_tokens=0, model_call_executed=False,
                             transport_failure='input_token_budget_exceeded')
                else:
                    regex = (grammar_fn or request_grammar)(q)
                    constraint = build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, RegexParser(regex))
                    x = torch.tensor([ids], device='cuda')
                    y = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                        temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new_tokens, use_cache=True,
                        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                        prefix_allowed_tokens_fn=constraint)[0, len(ids):]
                    response = tok.decode(y, skip_special_tokens=True).strip()
                    r.update(response=response, output_tokens=len(y), model_call_executed=True,
                             grammar_match=bool(re.fullmatch(regex, response)),
                             ended_with_eos=bool(len(y) and int(y[-1]) == tok.eos_token_id))
                result.append(r)
        return result


def candidate(row, model, selector, trace):
    audit = []
    with region_selection(selector, audit):
        state = prepare_public_parameter_intent(row)
        for _ in range(6):
            if state['phase'] == 'complete':
                break
            requests = state['requests']
            raw = model(requests)
            trace.append({'case_id': row['case_id'], 'requests': requests, 'responses': raw})
            failures = {r.get('transport_failure') for r in raw if r.get('transport_failure')}
            if failures:
                if failures != {'input_token_budget_exceeded'}:
                    raise RuntimeError('Inference transport failed; no repair outcome recorded.')
                return dict(case_id=row['case_id'], final_actions=[], failure='input_token_budget_exceeded',
                            llm_calls=state.get('_actual_calls', 0) + sum(r.get('model_call_executed', True) for r in raw),
                            output_tokens=state.get('_actual_tokens', 0) + sum(r.get('output_tokens', 0) for r in raw))
            state = submit_public_parameter_intent(row, deepcopy(state), raw)
        if state['phase'] != 'complete':
            raise RuntimeError('Fixed six-round orchestration budget exhausted.')
        return state['result']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('data/main/cases.jsonl'))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model', help='Local checkpoint directory or Hugging Face model identifier.')
    mode.add_argument('--archived-attempts', type=Path)
    mode.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--selector', choices=SELECTORS, default='topology')
    parser.add_argument('--case-id', action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = indexed(args.cases)
    selected = list(cases) if args.case_id is None else args.case_id
    if len(selected) != len(set(selected)) or not set(selected) <= cases.keys():
        parser.error('Case IDs must be unique and present in the input.')
    if args.archived_attempts and args.selector != 'topology':
        parser.error('Archived main-method attempts cannot be reused for a different region selector.')
    cached = indexed(args.archived_attempts) if args.archived_attempts else None
    model = LocalModel(args.model) if args.model else None
    trace = []

    def results():
        for cid in selected:
            row = cases[cid]
            if args.prepare_only:
                with region_selection(args.selector, []):
                    state = prepare_public_parameter_intent(row)
                yield {'case_id': cid, 'phase': state['phase'], 'requests': state['requests']}
                continue
            attempt = cached[cid] if cached is not None else candidate(row, model, args.selector, trace)
            result = finalize_with_geometry_constraints(row, attempt)
            yield {**result, 'table_cohort': row.get('table_cohort'),
                   'inference_mode': 'archived_attempt_finalization' if cached is not None else 'fresh_model',
                   'selector': args.selector}
            print(json.dumps({'case_id': cid, 'finished': True}), flush=True)

    write_rows(args.output, results())
    if trace:
        write_rows(args.output.with_suffix('.trace.jsonl'), trace)


if __name__ == '__main__':
    main()
