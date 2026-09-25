import json
from pathlib import Path

import pytest

from io_utils import indexed
from report_tables import summarize
from repair import candidate


ROOT = Path(__file__).resolve().parents[1]


def test_all_table_metrics_reproduce():
    cases = indexed(ROOT / 'data/main/cases.jsonl')
    refs = indexed(ROOT / 'data/evaluation/reference_topology.jsonl')
    expected = json.loads((ROOT / 'results/expected_main_table.json').read_text())
    for method in expected['repair_cells']:
        measured = summarize(indexed(ROOT / f'results/main/{method}.jsonl'), cases, refs)
        assert measured['repair_cells'] == pytest.approx(expected['repair_cells'][method])
        geo = expected['geometry_cells'][method]
        assert measured['geometry_cells'] == pytest.approx(geo[:4] + [geo[6]])


def test_reporter_rejects_missing_case():
    cases = indexed(ROOT / 'data/main/cases.jsonl')
    values = indexed(ROOT / 'results/main/toporecover.jsonl')
    del values[next(iter(values))]
    with pytest.raises(ValueError, match='denominator'):
        summarize(values, cases, {})


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / 'duplicate.jsonl'
    path.write_text('{"case_id":"one"}\n{"case_id":"one"}\n')
    with pytest.raises(ValueError, match='Duplicate'):
        indexed(path)


def test_transport_failure_not_counted_as_repair_failure(monkeypatch):
    row = {'case_id': 'transport-test'}
    monkeypatch.setattr('repair.prepare_public_parameter_intent', lambda _: {
        'phase': 'awaiting', 'requests': [{'case_id': 'transport-test'}]})
    with pytest.raises(RuntimeError, match='transport'):
        candidate(row, lambda _: [{'transport_failure': 'network_error'}], 'topology', [])


def test_input_budget_abstention_is_explicit(monkeypatch):
    row = {'case_id': 'budget-test'}
    monkeypatch.setattr('repair.prepare_public_parameter_intent', lambda _: {
        'phase': 'awaiting', 'requests': [{'case_id': 'budget-test'}], '_actual_calls': 2})
    result = candidate(row, lambda _: [{'transport_failure': 'input_token_budget_exceeded',
                        'model_call_executed': False}], 'topology', [])
    assert result['failure'] == 'input_token_budget_exceeded'
    assert result['llm_calls'] == 2
    assert result['final_actions'] == []


def test_native_reference_plan_checks_geometry():
    from build_reference_plan import build
    native = json.loads((ROOT / 'examples/native_ring.json').read_text())
    result = build(native)
    assert result['feature_plan']['profiles'][0]['loop_roles'] == ['outer', 'inner']
    assert result['source_provenance'][0]['containment_checked']
    assert result['reference_geometry_faithful']


def test_native_reference_plan_rejects_uncontained_hole():
    from build_reference_plan import build
    native = json.loads((ROOT / 'examples/native_ring.json').read_text())
    native['parts']['part_1']['sketch']['face_1']['loop_2']['circle_1']['Center'] = [10, 0]
    with pytest.raises(ValueError, match='contained'):
        build(native)
