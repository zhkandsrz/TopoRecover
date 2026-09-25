"""JSONL inputs with explicit case identity checks."""
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def indexed(path):
    rows = read_rows(path)
    result = {r['case_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate case IDs: {path}')
    return result


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=True) + '\n')
            output.flush()
