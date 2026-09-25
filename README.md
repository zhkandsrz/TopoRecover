# TopoRecover

Code and evaluation artifacts for topology-aware localization and repair of CAD
command sequences. TopoVerifier interprets commands while retaining construction
and source records. A saved profile that disagrees with the feature plan points
back to its producing commands, which need not coincide with a later runtime error.
Localized model proposals are checked before execution repair and final CAD validation.

This is a review release. It contains no model weights, author contact information,
server configuration, or upstream project history. The Python namespace remains
`toporeward` to preserve compatibility with the frozen implementation.

## Quick Start

Use Python 3.11 or 3.12 in a separate environment. CPU-only evaluation does not
require model downloads; fresh 7B inference requires a CUDA GPU and the optional
LLM dependencies.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/report_tables.py --check
```

The reporter recomputes the six main-table rows from per-case records, not from
hard-coded percentages. Expected repair success is **78.4%** for TopoRecover and
**64.3%** for the strongest archived adapted baseline. It also reports the matched
localization ablation. See [metric definitions and scope](docs/REPRODUCIBILITY.md).

## Run Repair

Inspect the actual public-input requests without loading a model:

```bash
python scripts/repair.py --prepare-only --case-id 3878a8e46955707f58c8 --output outputs/requests.jsonl
```

Replay finalization using the archived model-path attempts:

```bash
python scripts/repair.py --archived-attempts data/main/llm_attempts.jsonl --output outputs/repaired.jsonl
python scripts/evaluate.py --outputs outputs/repaired.jsonl --output outputs/evaluated.jsonl
```

This reruns acceptance, fallback when needed, and CAD execution. It does **not**
rerun model inference. For a fresh run:

```bash
python -m pip install -e ".[llm]"
python scripts/repair.py --model Qwen/Qwen2.5-7B-Instruct --output outputs/fresh.jsonl
```

Use a new output filename for each run; files are never silently overwritten.
Pass repeated `--case-id` flags for a smoke-test subset. Evaluation of a subset
requires `--allow-subset` and must not be reported as the complete main table.

## Contents

| Path | Content |
|---|---|
| `src/toporeward/verifier/` | State-machine interpreter and construction checks |
| `src/toporeward/llm_stage_a.py` | Profile diagnosis and source-region patch interface |
| `src/toporeward/stage_a_public_parameter_policy.py` | Public requirements and model-request orchestration |
| `src/toporeward/stage_a_geometry_constrained_fallback.py` | Final acceptance and bounded fallback |
| `src/toporeward/stage_a_profile_packets.py` | Localized commands, existing geometry, requirements and output schema |
| `scripts/` | Repair, adapted baseline, regeneration, evaluation and reporting entry points |
| `configs/` | Actual model identities and decoding/search budgets |
| `examples/native_ring.json` | Synthetic native-reference fixture for the plan utility |
| `data/main/` | 347 repair inputs, 23 separate geometry inputs, archived model-path attempts |
| `data/construction/` | 487 evaluable source inputs and recorded selection audits |
| `data/evaluation/` | Reference commands and topology, for evaluation only |
| `results/main/` | All six methods' outputs and per-case measurements, including failures |
| `results/localization/` | Four matched region-selector arms |
| `tests/` | Verifier, geometry protection, localization and release checks |

## Reproduction and Limitations

The main repair cohort uses frozen, model-generated feature plans. Repair success
means successful execution **and agreement with those specified requirements**;
it is not proof that every plan is the unique correct interpretation of the brief.
The separate reference-derived plan utility does not retroactively change the main
cohort or its reported results. Reference geometry is not an input to repair.

The three adapted repair baselines are archived 30B-system comparisons; TopoRecover
and the two regeneration baselines use a 7B model. They are not model/budget-matched
comparisons. The region-selector ablation holds the repair machinery fixed.
Fresh GPU runs may differ across software/hardware versions. No training is needed.

Detailed commands, dataset construction, denominators and reproduction boundaries
are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## License

Original code is released under [MIT](LICENSE). Text2CAD-derived data and program
artifacts have separate terms: [data attribution and license](data/README.md).
Third-party packages and model weights retain their upstream licenses. See
[NOTICE.md](NOTICE.md). No model weights or third-party source trees are bundled.
