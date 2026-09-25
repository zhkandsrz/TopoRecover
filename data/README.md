# Evaluation Data

## License and Attribution

These are derived Text2CAD evaluation artifacts, not a redistribution of the full
native dataset. Text2CAD is by Sadil Khan and collaborators; consult the
[official dataset card](https://huggingface.co/datasets/SadilKhan/Text2CAD) for
the complete upstream attribution. Text2CAD derives CAD assets from DeepCAD.

Files in `data/` and `results/` are distributed under **CC BY-NC-SA 4.0**.
The complete license is in [LICENSE](LICENSE). Original code elsewhere uses MIT.
Changes relative to upstream include sampling design briefs, translating generated
sequences to the supported command representation, generating structural plans,
selecting repair inputs, running repairs, and recording evaluation results.
Upstream `source_uid` values are retained so samples can be traced to their source.

## Files and Boundaries

- `main/cases.jsonl`: 370 public repair inputs, split by `table_cohort` into
  `repair347` and a separate `geometry23` cohort. Do not pool their denominators.
- `main/llm_attempts.jsonl`: archived completed model-path results supplied to
  finalization. These are not raw model transcripts and are not new inference.
- `construction/evaluable_inputs.jsonl`: the 487 evaluable generation inputs and
  original selection audit fields. All 347 selected repair UIDs are retained.
- `evaluation/reference_actions.jsonl`: 23 reference sequences for evaluation
  only. 22 execute successfully; the failed reference is retained in the file.
- `evaluation/reference_topology.jsonl`: reference kernel and mesh measurements
  used for the failure-aware Euler-characteristic comparison.

The feature plan records supported profiles, their boundary roles, and extrusion
references/operations. `topology_contract` is the normalized implementation schema
for checking these requirements, not an additional ground-truth shape. Main-cohort
plans were generated from public briefs before the corresponding histories and
kept fixed during repair. Programmatic agreement with a plan does not certify
the plan's semantic correctness. Reference-derived plans are a separate protocol.

Reference geometry and target commands must not be passed to a repair method or
used to choose among its outputs. They belong only to post-repair evaluation.
The published cohort is for reproduction and analysis, not a fresh held-out set
for tuning and then claiming independent generalization.
