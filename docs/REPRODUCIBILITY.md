# Reproducibility Guide

## What This Release Reproduces

There are three distinct operations:

1. `report_tables.py --check` aggregates archived per-case measurements and checks
   every main-table number. It needs no GPU or CAD execution.
2. `repair.py --archived-attempts ...`, followed by `evaluate.py`, replays the
   frozen method's final acceptance/fallback and re-executes the resulting programs.
3. `repair.py --model ...` also runs the model-request policy and fresh inference.
   Its outputs are new runs, not byte-identical claims about the archived table.

Release validation covers CPU tests, table reconstruction, public-request
preparation, and representative finalization/kernel smoke tests. A new full GPU
experiment was not run as part of packaging. Archived results are not newly
generated or independently sealed experiments.

## Dataset Construction

The main generation study sampled 500 Text2CAD briefs and obtained 487 evaluable
command sequences. The frozen planner supplied a feature plan before history
generation. The repair cohort selects every evaluable sequence for which verifier
execution/termination fails or the supported plan requirements are not satisfied.
`data/construction/evaluable_inputs.jsonl` exposes the audit for all 487, including
the non-selected inputs. `verify_release.py` checks that this selection reproduces
exactly the 347 released repair UIDs. No repaired output participates in selection.

The main structural requirements are model-generated specifications, not manually
certified topology ground truth. The metric is explicitly plan-conditioned.
The geometry cohort contains 23 distinct inputs and 22 executable references;
it is not a geometry measurement on all 347 repair cases.

For reference-derived structural plans, obtain native Text2CAD JSON through the
upstream source and run:

```bash
python scripts/build_reference_plan.py --native-json example.json --output reference-plan.json
```

This parses JSON objects, not regular expressions: part -> face -> loop -> curve.
The native boundary audit checks closure/validity, the outer boundary, inner-loop
containment, and disjointness. It extracts profile roles and extrusion operations,
then checks agreement with command-sequence replay and valid CAD execution.
Unsupported cases are rejected, not silently inferred. This supports structural
agreement with the reference representation, not a universal proof of design
intent or geometric equivalence to a native STEP file. The utility does not
replace the main cohort's plans. Do not mix the two evaluation protocols.

`examples/native_ring.json` is a small synthetic fixture for exercising this
utility without downloading the upstream corpus; it is not a benchmark case.

## Method and Configuration

The state-machine interpreter is `toporeward.verifier.TopoVerifier`; action types
are in `actions.py`, and parsing is in `lm/parsing.py`. Profile diagnostics map
saved construction to source regions. Profile packet prompts include localized
commands, existing curve geometry, protected geometry and plan requirements.
They request bounded structured proposals, not unrestricted arbitrary code.

The portable runner retains the frozen public-parameter policy, six orchestration
rounds, grammar-constrained greedy 7B decoding, 8192 input tokens and 2048 output
tokens per local request. It does not truncate oversized inputs. Finalization
uses one bounded fallback attempt from the original public input when the model
path is rejected, with horizon 6, rollback 32 and at most 240 replacement actions.
Both paths undergo public-requirement and geometry checks; evaluation references
never select the repair. A failure remains a failure in the reported denominator.

`configs/main.json` records checkpoint identity evidence and the actual archived
budgets. The retained source namespace and internal `stage_a` names are for code
compatibility, not additional paper methods. Some dependency modules contain
optional legacy interfaces; the documented entry point selects the paper method.

For CPU evaluation the release was checked with Python 3.13, CadQuery 2.7.0,
NumPy 2.3.1, SciPy 1.16.1 and trimesh 4.11.2. Python 3.11/3.12 is recommended for
compatibility with older inference wheels. The recorded regeneration environment
used torch 2.3.1+cu121, Transformers 4.51.3 and lm-format-enforcer 0.11.3.
Fresh model inference needs CUDA and enough memory for FP16 7B weights plus
the attention cache; full-program regeneration has a larger output budget.
No model weights, fine-tuning checkpoints or CUDA libraries are bundled.

## Metrics

- **Rep.**: verifier-valid terminated output matching the supported plan topology
  and an independently executable, valid positive-volume CAD solid; divide by 347.
- **Exe.**: CAD kernel execution validity alone, over the same 347 inputs.
- **PM**: terminated verifier replay and agreement with the implemented topology
  requirements, over 347 inputs. The checker uses supported counts, boundary-role
  multisets and profile/extrusion incidence; it is not full B-rep equivalence.
- **His.**: exact-command preservation over successful repairs only. The archived
  `action_lcs` field uses `SequenceMatcher` matching blocks; it is not a claim
  of globally optimal edit-distance minimization.
- **Rec.**: the same preservation score, with failures assigned zero, over 347.
- **CD/HD**: mean bidirectional nearest-surface distance and maximum sampled
  surface distance, normalized by reference bounding-box diagonal. 4096 surface
  samples are used. These are conditional on measurable outputs; `n_g` gives
  the count and differs by method.
- **IoU**: volume intersection-over-union, assigning unmeasurable outputs zero,
  averaged over all 22 executable references.
- **Euler agreement**: boundary meshes must be watertight and stable at two
  tessellation resolutions. Equality with reference Euler characteristic is
  counted over all 22 eligible references; failures are incorrect. This is a
  topology proxy, not a complete topological-equivalence test.

Geometry metrics stay in original coordinates, with no post-hoc alignment.
Only mesh measurement normalizes the solid before tessellation. See the released
metric functions for exact tolerances. Preserve missing values and denominators.

## Matched Localization

```bash
python scripts/repair.py --model Qwen/Qwen2.5-7B-Instruct --selector rejected_command --output outputs/rejected.jsonl
```

Repeat for `active_block`, `runtime_dependency`, and `topology`. Do not reuse the
main method's archived model attempts for other selectors. The same public inputs,
patcher, validation, fallback and budgets are used. `region_selection` changes
the profile-region selector; it is not thread-safe, so use one process per run.
The archived four-arm results include all 370 inputs and all failures. Runtime-only
selectors have no trigger on programs accepted by the interpreter but inconsistent
with the plan; this distinction is part of the ablation, not a dropped case.

## Regeneration Baselines

```bash
python scripts/regenerate.py --model Qwen/Qwen2.5-7B-Instruct --method plan_generation --output outputs/plan-generation.jsonl
python scripts/regenerate.py --model Qwen/Qwen2.5-7B-Instruct --method feedback_regeneration --output outputs/feedback-generation.jsonl
```

Plan-only generation has one 12288-token output; feedback regeneration has up to
two 6144-token outputs. The public-only prompts and JSON grammar are in
`llm_regeneration_protocol.py`. The portable runner is sequential; the archived
run used batch size four, so identical floating-point outputs are not guaranteed.
Evaluate either resulting file with `scripts/evaluate.py`.

## Adapted Repair Baselines

The three published adapted-system rows used Qwen3-30B-A3B, not the 7B checkpoint.
Full historical prompt bytes/complete prompt parity were not archived; current
prompt builders are supplied, but are not evidence of identical historical inputs.
The released per-case final programs and measurements permit direct reevaluation.

For a new adapted-system run, provide a local Qwen checkpoint:

```bash
python scripts/run_iterative_strong_repair_baseline.py --public data/main/cases.jsonl --frozen-plans data/main/cases.jsonl --model /path/to/checkpoint --method recad_iterative --rounds 2 --num-samples 4 --feedback-max-new-tokens 768 --editor-max-new-tokens 3072 --output outputs/recad-raw.jsonl
python scripts/convert_baseline_outputs.py --input outputs/recad-raw.jsonl --output outputs/recad.jsonl
python scripts/evaluate.py --outputs outputs/recad.jsonl --output outputs/recad-evaluated.jsonl
```

Other method names are `cadcodeverify_iterative` and `caddesigner_iterative`.
Review the runner's options for sampling, dtype, context limit and design-brief
sharing; do not describe a fresh run as an exact archived reproduction without
checking those settings. The main table is a system comparison, whereas the
region-selector study is the controlled localization comparison.
