"""Convert frozen adapted-baseline outputs without candidate reselection."""
import argparse
from pathlib import Path

from io_utils import indexed, write_rows
from toporeward.strong_repair_baselines import parse_full_history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--cases', type=Path, default=Path('data/main/cases.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cases, raw = indexed(a.cases), indexed(a.input)
    if cases.keys() != raw.keys():
        p.error('Baseline output coverage differs from cases.')
    converted = []
    for cid, r in raw.items():
        if len(r['outputs']) != 1:
            raise ValueError('Exactly one frozen output required; this converter never selects a candidate.')
        converted.append({'case_id': cid, 'table_cohort': cases[cid]['table_cohort'],
                          'final_actions': parse_full_history(r['outputs'][0]) or []})
    write_rows(a.output, converted)


if __name__ == '__main__':
    main()
