"""Portable sequential runner for the two public-input 7B regeneration baselines."""
import argparse
import json
from pathlib import Path

from io_utils import indexed, write_rows
from repair import LocalModel
from llm_regeneration_protocol import prompt, full_program_grammar, parse_program, check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('data/main/cases.jsonl'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--method', choices=('plan_generation', 'feedback_regeneration'), required=True)
    parser.add_argument('--case-id', action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = indexed(args.cases)
    selected = list(cases) if args.case_id is None else args.case_id
    if len(selected) != len(set(selected)) or not set(selected) <= cases.keys():
        parser.error('Case IDs must be unique and present.')
    model = LocalModel(args.model)
    rounds, tokens = (1, 12288) if args.method == 'plan_generation' else (2, 6144)

    def results():
        for cid in selected:
            row, previous, attempts, accepted = cases[cid], None, [], False
            for index in range(rounds):
                request = {'case_id': cid, 'source_uid': row['source_uid'],
                           'prompt': prompt(row, args.method, previous=previous)}
                raw = model([request], grammar_fn=lambda _: full_program_grammar(), max_new_tokens=tokens)[0]
                if raw.get('transport_failure'):
                    raise RuntimeError('Regeneration input exceeded the fixed token budget; do not truncate.')
                previous, error = parse_program(raw['response'])
                checked = {'accepted': False, 'reason': error} if error else check_program(row, previous)
                accepted = checked['accepted']
                attempts.append({'round': index + 1, 'request': request, 'response': raw, 'validation': checked})
                if accepted:
                    break
            yield {'case_id': cid, 'table_cohort': row['table_cohort'], 'method': args.method,
                   'final_actions': previous if accepted else [], 'attempts': attempts,
                   'inference_mode': 'fresh_model_sequential'}
            print(json.dumps({'case_id': cid, 'accepted': accepted}), flush=True)

    write_rows(args.output, results())


if __name__ == '__main__':
    main()
