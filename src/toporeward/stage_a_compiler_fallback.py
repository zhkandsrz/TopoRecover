"""LLM-first recovery with one explicitly attributed compiler fallback.

Selection uses only public topology and kernel execution, never reference-solid
geometry. This gate does not certify agreement with all design-brief dimensions.
"""
from copy import deepcopy

from .cadquery_executor import execute_actions
from .lm.parsing import parse_action_line
from .natural_repair_recall import evaluate_repaired_history
from .stage_a_packet_policy import PUBLIC_SOURCE_FIELDS
from .strong_repair_baselines import runtime_trace
from .topology_transaction_repair import repair_topology_history


def validate_public_output(lines, contract):
    if not lines:
        return dict(accepted=False, reason='no_final_output')
    parsed = [parse_action_line(line) for line in lines]
    if any(action is None for action in parsed):
        return dict(accepted=False, reason='unparseable_output')
    replay = evaluate_repaired_history(lines, contract)
    if not replay['valid_and_intent_satisfied']:
        return dict(accepted=False, reason='public_topology_or_execution_rejected')
    kernel = execute_actions(parsed).to_dict()
    return dict(accepted=bool(kernel['success']),
                reason=None if kernel['success'] else 'kernel_execution_rejected',
                kernel=kernel)


def compile_public_fallback(source):
    lines = source['observed_actions']
    if not lines:
        return []
    step = runtime_trace(lines).first_rejected_step
    plan = repair_topology_history(
        lines, failure_step=step if step is not None else len(lines)-1,
        contract=source['topology_contract'], horizon=6,
        max_rollback=32, max_replacement_actions=160)
    return list(plan.final_actions) if plan and plan.valid_and_intent_satisfied else []


def finalize_with_compiler_fallback(row, llm_result):
    """Call only after the fixed LLM attempt/retry budget has finished.

Infrastructure exceptions propagate rather than being disguised as refusals.
Rejected partial patches are never passed to the compiler.
"""
    source = {key: deepcopy(row[key]) for key in PUBLIC_SOURCE_FIELDS}
    if llm_result['case_id'] != source['case_id']:
        raise ValueError('fallback_case_identity_mismatch')
    result = dict(
        case_id=source['case_id'], final_actions=[],
        llm_attempt=deepcopy(llm_result),
        llm_calls=llm_result.get('llm_calls', 0),
        output_tokens=llm_result.get('output_tokens', 0),
        compiler_calls=0, fallback_attempted=False, fallback_used=False,
        symbolic_stage_a_fallback=False, topology_success=False,
        selection_uses_private_reference=False,
        explicit_geometry_fidelity_certified=False)
    # Inconsistent public requirements need clarification, not a bypass.
    if llm_result.get('route') == 'public_parameter_evidence_rejected':
        return dict(result, route='unresolved_public_input',
                    failure=llm_result.get('failure'))
    lines = list(llm_result.get('final_actions') or [])
    checked = validate_public_output(lines, source['topology_contract'])
    result['llm_validation'] = checked
    if checked['accepted'] and not llm_result.get('failure'):
        return dict(result, final_actions=lines, route='llm_path_accepted',
                    topology_success=True, failure=None)
    result.update(fallback_attempted=True, compiler_calls=1,
                  fallback_trigger=llm_result.get('failure') or checked['reason'])
    lines = compile_public_fallback(source)
    checked = validate_public_output(lines, source['topology_contract'])
    result['compiler_validation'] = checked
    if checked['accepted']:
        return dict(result, final_actions=lines, route='compiler_fallback_accepted',
                    topology_success=True, fallback_used=True,
                    symbolic_stage_a_fallback=True, failure=None)
    return dict(result, route='both_paths_failed', failure=checked['reason'])
