"""Recompute main-table and localization aggregates from all released per-case rows."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from io_utils import indexed


def summarize(values, cases, reference_topology):
    if values.keys() != cases.keys():
        raise ValueError('Missing, extra or duplicate case IDs: refusing a changed denominator.')
    repair = [values[cid] for cid, r in cases.items() if r['table_cohort'] == 'repair347']
    geometry = [values[cid] for cid, r in cases.items() if r['table_cohort'] == 'geometry23']
    if len(repair) != 347 or len(geometry) != 23:
        raise ValueError('This reporter is for the frozen 347/23 main cohorts.')
    good = [r for r in repair if r['repair_success']]
    scores = [100 * mean(r['execution_success'] for r in repair),
              100 * mean(r['contract_success'] for r in repair), 100 * len(good) / len(repair),
              mean(r['action_lcs'] for r in good) if good else None,
              sum(r['action_lcs'] for r in good) / len(repair)]
    refs = [r for r in geometry if reference_topology[r['case_id']]['eligible']]
    if len(refs) != 22:
        raise ValueError('Reference denominator changed.')
    measured = [r for r in refs if r.get('geometry') is not None]
    correct = sum(bool(r.get('solid_topology', {}).get('eligible')) and
                  r['solid_topology']['euler'] == reference_topology[r['case_id']]['euler'] for r in refs)
    geo = [len(measured), mean(r['geometry']['chamfer_l2_normalized'] for r in measured) if measured else None,
           mean(r['geometry']['hausdorff_l2_normalized'] for r in measured) if measured else None,
           sum(r['geometry']['volume_iou'] for r in measured) / len(refs), 100 * correct / len(refs)]
    return {'repair_cells': scores, 'geometry_cells': geo}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--check', action='store_true', help='Check every main-table metric against the frozen table.')
    args = parser.parse_args()
    root = args.root
    cases = indexed(root / 'data/main/cases.jsonl')
    refs = indexed(root / 'data/evaluation/reference_topology.jsonl')
    expected = json.loads((root / 'results/expected_main_table.json').read_text())
    print('| Method | Rep.% | Exe.% | PM% | His. | Rec. | n_g | CD | HD | IoU | Euler% |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for method in expected['repair_cells']:
        result = summarize(indexed(root / f'results/main/{method}.jsonl'), cases, refs)
        r, g = result['repair_cells'], result['geometry_cells']
        if args.check:
            er, eg = expected['repair_cells'][method], expected['geometry_cells'][method]
            wanted = er + eg[:4] + [eg[6]]
            actual = r + g
            for index, (a, b) in enumerate(zip(actual, wanted)):
                if a is None or b is None:
                    if a != b:
                        raise ValueError((method, index, a, b))
                elif not math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10):
                    raise ValueError((method, index, a, b))
        line = [r[2], r[0], r[1], r[3], r[4], *g]
        print('| ' + method + ' | ' + ' | '.join('--' if x is None else f'{x:.6g}' for x in line) + ' |')
    print('\nMatched localization (347 repair cases):')
    for path in sorted((root / 'results/localization').glob('*.jsonl')):
        values = indexed(path)
        if values.keys() != cases.keys():
            raise ValueError('Localization cohort differs from the main cohort.')
        rows = [r for r in values.values() if r['table_cohort'] == 'repair347']
        print(f'{path.stem}: {sum(r["repair_success"] for r in rows)}/{len(rows)} = {100*mean(r["repair_success"] for r in rows):.3f}%')
    if args.check:
        print('\nPASS: all 60 main-table aggregate values match the frozen records.')


if __name__ == '__main__':
    main()
