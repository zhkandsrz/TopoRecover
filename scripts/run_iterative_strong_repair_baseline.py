from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from toporeward.strong_repair_baselines import (
    cadcodeverify_style_prompt,
    caddesigner_style_prompt,
    parse_full_history,
    parse_recad_feedback,
    prompt_input_audit,
    recad_style_editor_prompt,
    recad_style_feedback_prompt,
    requirement_candidate_key,
    requirement_validation_feedback,
)


METHODS = (
    "cadcodeverify_iterative",
    "caddesigner_iterative",
    "recad_iterative",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--frozen-plans", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-input-length", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--feedback-max-new-tokens", type=int, default=768)
    parser.add_argument("--editor-max-new-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument("--load-on-device", action="store_true",
                        help="Load weights directly on the selected device instead of staging the full model on CPU")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-design-brief", action="store_true",
                        help="Give binary-feedback refinement the same public brief as other methods")
    parser.add_argument("--max-memory-per-gpu", default="30GiB")
    parser.add_argument("--dtype", choices=('auto', 'float16'), default='auto')
    args = parser.parse_args()

    if args.rounds < 1 or args.num_samples < 1:
        raise ValueError("rounds and num-samples must be positive")
    if args.device_map_auto and args.load_on_device:
        parser.error("Choose either automatic sharding or direct single-device loading")

    public = read_jsonl(args.public)
    end = args.end_index if args.end_index > 0 else len(public)
    public = public[args.start_index : end]
    plans = {str(row["source_uid"]): row for row in read_jsonl(args.frozen_plans)}

    completed: set[str] = set()
    if args.resume and args.output.exists():
        completed = {
            str(row["case_id"])
            for row in read_jsonl(args.output)
        }

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.dtype == 'float16' else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if args.device_map_auto:
        model_kwargs.update(
            {
                "device_map": "balanced",
                "max_memory": {
                    index: args.max_memory_per_gpu for index in range(torch.cuda.device_count())
                },
            }
        )
    elif args.load_on_device:
        model_kwargs["device_map"] = {"": args.device}
    print(json.dumps({"event": "model_loading", "direct_device_loading": args.load_on_device,
                      "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "dtype": str(dtype)}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map_auto and not args.load_on_device:
        model.to(args.device)
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    if args.device_map_auto and any(str(v) in ('cpu', 'disk') for v in model.hf_device_map.values()):
        raise RuntimeError('CPU/disk offload is outside the requested GPU protocol')
    cost = {}

    def account(encoded, generated, input_width, started, samples):
        cost['generation_calls'] += 1
        cost['generated_sequences'] += len(generated)
        cost['input_tokens'] += int(encoded['attention_mask'].sum()) * samples
        for seq in generated[:, input_width:]:
            eos = (seq == tokenizer.eos_token_id).nonzero()
            cost['output_tokens'] += int(eos[0, 0])+1 if len(eos) else len(seq)
        cost['generation_seconds'] += time.monotonic()-started
    print(json.dumps({"event": "model_ready", "input_device": str(input_device),
                      "allocated_gib": torch.cuda.memory_allocated() / 2**30}), flush=True)

    def generate_texts(
        prompt: str, *, num_samples: int, max_new_tokens: int | None = None
    ) -> list[str]:
        template_kwargs = (
            {"enable_thinking": False} if args.disable_thinking else {}
        )
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        encoded = tokenizer(
            [chat],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        )
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        started = time.monotonic()
        generated = model.generate(
            **encoded,
            do_sample=num_samples > 1,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=num_samples,
            max_new_tokens=(
                args.max_new_tokens if max_new_tokens is None else max_new_tokens
            ),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        input_width = encoded["input_ids"].shape[1]
        account(encoded, generated, input_width, started, num_samples)
        return tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )

    def generate_batch_texts(
        prompts: list[str], *, max_new_tokens: int
    ) -> list[str]:
        if not prompts:
            return []
        template_kwargs = (
            {"enable_thinking": False} if args.disable_thinking else {}
        )
        chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            for prompt in prompts
        ]
        encoded = tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        started = time.monotonic()
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        input_width = encoded["input_ids"].shape[1]
        account(encoded, generated, input_width, started, 1)
        return tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle, torch.no_grad():
        for case_index, source in enumerate(public, start=args.start_index):
            case_id = str(source["case_id"])
            if case_id in completed:
                continue
            cost = {'generation_calls': 0, 'generated_sequences': 0, 'input_tokens': 0,
                    'output_tokens': 0, 'generation_seconds': 0.}
            case_seed = int.from_bytes(
                hashlib.sha256(f"{args.seed}:{case_id}".encode("utf-8")).digest()[:8],
                "big",
            ) % (2**31)
            torch.manual_seed(case_seed)
            torch.cuda.manual_seed_all(case_seed)
            plan_row = plans[str(source["source_uid"])]
            feature_plan = plan_row["feature_plan"]
            original = [str(line) for line in source["observed_actions"]]
            current = list(original)
            round_records: list[dict[str, Any]] = []

            for round_index in range(1, args.rounds + 1):
                review_records: list[dict[str, Any]] = []
                if args.method == "cadcodeverify_iterative":
                    prompt, _ = cadcodeverify_style_prompt(
                        observed_lines=current,
                        feature_plan=feature_plan,
                        round_index=round_index,
                        max_rounds=args.rounds,
                    )
                    if args.include_design_brief:
                        prompt = '[DESIGN_REQUIREMENT]\n'+str(plan_row.get('design_brief') or '')+'\n'+prompt
                elif args.method == "caddesigner_iterative":
                    prompt, _ = caddesigner_style_prompt(
                        observed_lines=current,
                        feature_plan=feature_plan,
                        design_brief=str(plan_row.get("design_brief") or ""),
                        round_index=round_index,
                        max_rounds=args.rounds,
                    )
                else:
                    prompt, _ = recad_style_feedback_prompt(
                        observed_lines=current,
                        feature_plan=feature_plan,
                        design_brief=str(plan_row.get("design_brief") or ""),
                        round_index=round_index,
                        max_rounds=args.rounds,
                    )
                audit = prompt_input_audit(prompt)
                if any(audit.values()):
                    raise ValueError(f"forbidden input leaked for {case_id}: {audit}")

                if args.method == "recad_iterative":
                    feedback_texts = generate_texts(
                        prompt,
                        num_samples=args.num_samples,
                        max_new_tokens=args.feedback_max_new_tokens,
                    )
                    editor_prompts: list[str] = []
                    parsed_feedback: list[dict[str, Any]] = []
                    for feedback_text in feedback_texts:
                        feedback = parse_recad_feedback(
                            feedback_text, observed_length=len(current)
                        )
                        review_records.append(
                            {
                                "raw": feedback_text,
                                "parseable": feedback is not None,
                                "feedback": feedback,
                            }
                        )
                        if feedback is None:
                            continue
                        editor_prompt = recad_style_editor_prompt(
                            observed_lines=current,
                            feature_plan=feature_plan,
                            design_brief=str(plan_row.get("design_brief") or ""),
                            generated_feedback=feedback,
                            round_index=round_index,
                            max_rounds=args.rounds,
                        )
                        editor_audit = prompt_input_audit(editor_prompt)
                        if any(editor_audit.values()):
                            raise ValueError(
                                f"forbidden editor input leaked for {case_id}: {editor_audit}"
                            )
                        editor_prompts.append(editor_prompt)
                        parsed_feedback.append(feedback)
                    editor_budget = min(
                        args.editor_max_new_tokens,
                        max(768, 24 * len(current) + 256),
                    )
                    texts = generate_batch_texts(
                        editor_prompts,
                        max_new_tokens=editor_budget,
                    )
                else:
                    full_history_budget = min(
                        args.max_new_tokens,
                        max(768, 24 * len(current) + 256),
                    )
                    texts = generate_texts(
                        prompt,
                        num_samples=args.num_samples,
                        max_new_tokens=full_history_budget,
                    )
                    parsed_feedback = []
                    editor_budget = None

                candidates: list[dict[str, Any]] = []
                for text_index, text in enumerate(texts):
                    parsed = parse_full_history(text)
                    if parsed is None:
                        candidate = {"parseable": False, "raw": text}
                        if parsed_feedback:
                            candidate["review_feedback"] = parsed_feedback[text_index]
                        candidates.append(candidate)
                        continue
                    feedback = requirement_validation_feedback(
                        history_lines=parsed, feature_plan=feature_plan
                    )
                    key = requirement_candidate_key(
                        candidate_lines=parsed,
                        feature_plan=feature_plan,
                        original_lines=original,
                    )
                    candidate = {
                        "parseable": True,
                        "raw": text,
                        "actions": parsed,
                        "feedback": feedback,
                        "selection_key": list(key),
                    }
                    if parsed_feedback:
                        candidate["review_feedback"] = parsed_feedback[text_index]
                    candidates.append(candidate)

                parseable = [row for row in candidates if row["parseable"]]
                if parseable:
                    selected = max(
                        parseable,
                        key=lambda row: tuple(row["selection_key"]),
                    )
                    current = list(selected["actions"])
                    selected_index = candidates.index(selected)
                else:
                    selected_index = None
                round_records.append(
                    {
                        "round": round_index,
                        "input_feedback": requirement_validation_feedback(
                            history_lines=(
                                original
                                if round_index == 1
                                else round_records[-1]["selected_actions"]
                            ),
                            feature_plan=feature_plan,
                        ),
                        "selected_index": selected_index,
                        "selected_actions": current,
                        "generated_reviews": review_records,
                        "editor_max_new_tokens_effective": editor_budget,
                        "full_history_max_new_tokens_effective": (
                            full_history_budget
                            if args.method != "recad_iterative"
                            else None
                        ),
                        "candidates": candidates,
                    }
                )

            final_feedback = requirement_validation_feedback(
                history_lines=current, feature_plan=feature_plan
            )
            payload = {
                "case_id": case_id,
                "program_id": str(source.get("program_id") or ""),
                "source_uid": str(source["source_uid"]),
                "failure_scope": source.get("failure_scope"),
                "method": args.method,
                "output_mode": "full_history",
                "outputs": [json.dumps({"actions": current}, separators=(",", ":"))],
                "rounds": round_records,
                "final_feedback": final_feedback,
                "model": args.model,
                "inference_cost": dict(cost),
                "public_brief_shared": args.include_design_brief or args.method != 'cadcodeverify_iterative',
                "seed": args.seed,
                "case_seed": case_seed,
                "budget": {
                    "rounds": args.rounds,
                    "samples_per_round": args.num_samples,
                    "feedback_max_new_tokens": args.feedback_max_new_tokens,
                    "editor_max_new_tokens": args.editor_max_new_tokens,
                },
                "input_audit": {
                    "stage_a_discrepancy_locus_visible": False,
                    "target_actions_visible": False,
                    "target_geometry_visible": False,
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "method": args.method,
                        "completed_index": case_index + 1,
                        "total": end,
                        "case_id": case_id,
                        "checks_passed": final_feedback["passed"],
                        "checks_total": final_feedback["total"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
