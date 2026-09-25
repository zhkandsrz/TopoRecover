"""Re-execute fixed output programs; reference files are evaluation-only inputs."""
import argparse
import json
from pathlib import Path

from io_utils import indexed, write_rows
from geometry_metrics import capture_execute, metrics
from preservation_metrics import preservation_metrics
from solid_metrics import shape_record
from toporeward.natural_repair_recall import evaluate_repaired_history


def evaluate(row, result, target_actions=None):
    actions = result['final_actions']
    kernel, shape = capture_execute(actions, 4096 if target_actions is not None else 32)
    try:
        topology = bool(actions) and evaluate_repaired_history(actions, row['topology_contract'])['valid_and_intent_satisfied']
    except ValueError:
        topology = False
    success = bool(kernel.get('success'))
    record = dict(case_id=row['case_id'], table_cohort=row.get('table_cohort'), final_actions=actions,
                  execution_success=success, contract_success=bool(topology), repair_success=bool(success and topology),
                  occ={k: v for k,v in kernel.items() if k != 'surface_evidence'},
                  **preservation_metrics(row['observed_actions'], actions))
    if target_actions is None:
        return record
    target, target_shape = capture_execute(target_actions, 4096)
    ref_ok = bool(target.get('success') and target_shape is not None and target.get('surface_evidence'))
    record.update(reference_executable=ref_ok, geometry=None,
                  solid_topology={'eligible': False, 'euler': None})
    if success and shape is not None:
        record['solid_topology'] = shape_record(shape)
        if ref_ok:
            record['geometry'] = metrics(kernel, shape, target, target_shape)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('data/main/cases.jsonl'))
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--references', type=Path, default=Path('data/evaluation/reference_actions.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-subset', action='store_true', help='Explicitly allow a smoke-test subset; not a main-table run.')
    args = parser.parse_args()
    cases, outputs, refs = indexed(args.cases), indexed(args.outputs), indexed(args.references)
    if not outputs.keys() <= cases.keys() or (not args.allow_subset and cases.keys() != outputs.keys()):
        parser.error('Output coverage differs from the input cohort. Partial smoke tests require --allow-subset.')

    def results():
        for cid, result in outputs.items():
            record = evaluate(cases[cid], result, refs.get(cid, {}).get('target_actions'))
            print(json.dumps({'case_id': cid, 'repair_success': record['repair_success']}), flush=True)
            yield record

    write_rows(args.output, results())


if __name__ == '__main__':
    main()
