"""Versioned development integration of the paper's public source-frame prior.

The prior realizes wholly new compiler templates, not accepted LLM patches. It
does not infer unknown design intent or use reference geometry for selection.
"""
from copy import deepcopy

from .geometry_source_frame import source_frame_realization
from .stage_a_compiler_fallback import validate_public_output
from .stage_a_packet_policy import PUBLIC_SOURCE_FIELDS
from .strong_repair_baselines import runtime_trace
from .topology_transaction_repair import repair_topology_history

COMPILER_BUDGET = dict(horizon=6, max_rollback=32, max_replacement_actions=240)


def compile_source_frame_fallback(source):
    public = {key:deepcopy(source[key]) for key in PUBLIC_SOURCE_FIELDS}
    lines, contract = public['observed_actions'], public['topology_contract']
    record = dict(final_actions=[], source_frame_used=False,
                  compiler_budget=dict(COMPILER_BUDGET), reference_geometry_access=False,
                  explicit_geometry_fidelity_certified=False)
    if not lines:
        return dict(record, validation=dict(accepted=False,reason='no_original_input'))
    step = runtime_trace(lines).first_rejected_step
    plan = repair_topology_history(lines, failure_step=step if step is not None else len(lines)-1,
                                   contract=contract, **COMPILER_BUDGET)
    raw = list(plan.final_actions) if plan is not None else []
    record['raw_compiler_actions'] = raw
    if not raw:
        return dict(record,validation=dict(accepted=False,reason='no_compiler_output'))
    prior, evidence = source_frame_realization(lines,raw,contract)
    record['source_frame_evidence'] = evidence
    if prior is not None:
        checked = validate_public_output(prior,contract)
        record['source_frame_validation'] = checked
        if checked['accepted']:
            return dict(record,final_actions=prior,source_frame_used=True,validation=checked)
    checked = validate_public_output(raw,contract)
    return dict(record,final_actions=raw if checked['accepted'] else [],validation=checked)


def finalize_with_source_frame_fallback(row,llm_result):
    """Keep the frozen LLM path; otherwise invoke the complete public pipeline."""
    public = {key:deepcopy(row[key]) for key in PUBLIC_SOURCE_FIELDS}
    if llm_result['case_id'] != public['case_id']:
        raise ValueError('fallback_case_identity_mismatch')
    result = dict(case_id=public['case_id'],final_actions=[],llm_attempt=deepcopy(llm_result),
        llm_calls=llm_result.get('llm_calls',0),output_tokens=llm_result.get('output_tokens',0),
        compiler_calls=0,fallback_attempted=False,fallback_used=False,
        symbolic_stage_a_fallback=False,topology_success=False,
        selection_uses_private_reference=False,explicit_geometry_fidelity_certified=False)
    if llm_result.get('route') == 'public_parameter_evidence_rejected':
        return dict(result,route='unresolved_public_input',failure=llm_result.get('failure'))
    lines = list(llm_result.get('final_actions') or [])
    checked = validate_public_output(lines,public['topology_contract'])
    result['llm_validation'] = checked
    if checked['accepted'] and not llm_result.get('failure'):
        return dict(result,final_actions=lines,route='llm_path_accepted',
                    topology_success=True,failure=None)
    compiled = compile_source_frame_fallback(public)
    result.update(compiler_calls=1,fallback_attempted=True,compiler_result=compiled,
                  fallback_trigger=llm_result.get('failure') or checked['reason'],
                  compiler_validation=compiled['validation'])
    if compiled['validation']['accepted']:
        return dict(result,final_actions=compiled['final_actions'],route='compiler_fallback_accepted',
                    topology_success=True,fallback_used=True,symbolic_stage_a_fallback=True,failure=None)
    return dict(result,route='both_paths_failed',failure=compiled['validation']['reason'])
