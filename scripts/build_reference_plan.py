"""Extract a reference-faithful structural plan for the supported native JSON subset.

Checks native loop containment, agreement with DSL replay, and valid DSL execution.
This is not a claim that a generated language-model plan is ground truth.
"""
import argparse
import json
from pathlib import Path

from native_plan import native_boundary_audit
from geometry_metrics import capture_execute
from toporeward.actions import action_to_text
from toporeward.feature_plan import feature_plan_to_topology_contract
from toporeward.prompt_contract import normalize_topology_contract
from toporeward.text2cad_bridge import text2cad_minimal_json_to_actions
from toporeward.topology_history_repair import topology_intent_contract


def build(native):
    plan, provenance = native_boundary_audit(native)
    contract = feature_plan_to_topology_contract(plan)
    bridge = text2cad_minimal_json_to_actions(native)
    actions = [action_to_text(a) for a in bridge.actions]
    replay = topology_intent_contract(actions, level='topology')
    if replay is None or normalize_topology_contract(replay) != normalize_topology_contract(contract):
        raise ValueError('Native and DSL structural requirements disagree.')
    kernel, _ = capture_execute(actions, 32)
    if not kernel.get('success'):
        raise ValueError('Reference DSL kernel execution failed.')
    return {'feature_plan': plan, 'topology_contract': contract, 'source_provenance': provenance,
            'reference_geometry_faithful': bridge.geometry_faithful, 'bridge_issues': list(bridge.issues),
            'verification_scope': 'supported native boundaries, structural DSL agreement, DSL kernel execution; not full native STEP equivalence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.native_json.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
