"""Check artifact integrity, cohort coverage, selection and optional verifier replay."""
import argparse
import hashlib
import json
from pathlib import Path

from io_utils import indexed, read_rows


def check(root, replay=False):
    manifest = json.loads((root / 'release_manifest.json').read_text())
    for name, record in manifest['files'].items():
        data = (root / name).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Artifact changed: ' + name)
    cases = indexed(root / 'data/main/cases.jsonl')
    if len(cases) != 370:
        raise ValueError('Expected exactly 370 main input records.')
    repair_uids = {r['source_uid'] for r in cases.values() if r['table_cohort'] == 'repair347'}
    geometry_uids = {r['source_uid'] for r in cases.values() if r['table_cohort'] == 'geometry23'}
    if len(repair_uids) != 347 or len(geometry_uids) != 23 or repair_uids & geometry_uids:
        raise ValueError('Main cohort counts or UID disjointness changed.')
    source = read_rows(root / 'data/construction/evaluable_inputs.jsonl')
    selected = {r['source_uid'] for r in source if r['recorded_audit']['goal_failure'] or
                not r['recorded_audit']['execution_valid_ended']}
    if len(source) != 487 or len({r['source_uid'] for r in source}) != 487 or selected != repair_uids:
        raise ValueError('Recorded input selection does not reproduce the repair cohort.')
    attempts = indexed(root / 'data/main/llm_attempts.jsonl')
    if attempts.keys() != cases.keys():
        raise ValueError('Archived model attempts do not cover the complete cohort.')
    references = indexed(root / 'data/evaluation/reference_actions.jsonl')
    if references.keys() != {cid for cid, r in cases.items() if r['table_cohort'] == 'geometry23'}:
        raise ValueError('Geometry references do not cover the full geometry cohort.')
    checked = 0
    for path in sorted((root / 'results/main').glob('*.jsonl')):
        values = indexed(path)
        if values.keys() != cases.keys():
            raise ValueError('Incomplete result file: ' + path.name)
        for cid, result in values.items():
            if result['repair_success'] != bool(result['execution_success'] and result['contract_success']):
                raise ValueError('Inconsistent joint success: ' + cid)
            if replay:
                from toporeward.natural_repair_recall import evaluate_repaired_history
                actions = result['final_actions']
                try:
                    matched = bool(actions) and evaluate_repaired_history(actions, cases[cid]['topology_contract'])['valid_and_intent_satisfied']
                except ValueError:
                    matched = False
                if bool(matched) != result['contract_success']:
                    raise ValueError(f'Verifier outcome differs: {path.name}, {cid}')
                checked += 1
    return {'manifest_files_verified': len(manifest['files']), 'main_cases': len(cases),
            'selection_reproduced': True, 'verifier_outputs_replayed': checked}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--replay-checks', action='store_true')
    a = p.parse_args()
    print(json.dumps(check(a.root, a.replay_checks), indent=2))


if __name__ == '__main__':
    main()
