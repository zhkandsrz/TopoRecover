"""Development hybrid: explicit public geometry outranks a source-scale prior."""
from copy import deepcopy

from .public_profile_geometry import check_public_geometry, realize_explicit_new_profiles
from .stage_a_compiler_fallback import validate_public_output
from .stage_a_packet_policy import PUBLIC_SOURCE_FIELDS
from .stage_a_source_frame_fallback import compile_source_frame_fallback
from .stage_b_preserving_geometry import bind_plan_dimensions


def compile_geometry_constrained_fallback(source):
    public = {k:deepcopy(source[k]) for k in PUBLIC_SOURCE_FIELDS}
    record = dict(final_actions=[],reference_geometry_access=False,
                  explicit_geometry_fidelity_certified=False)
    try:
        public['topology_contract'], depths = bind_plan_dimensions(
            public['topology_contract'],public['feature_plan'])
    except (ValueError, KeyError, TypeError) as error:
        return dict(record,validation=dict(accepted=False,reason=str(error)))
    compiled = compile_source_frame_fallback(public)
    record.update(compiler_result=compiled,depth_binding=depths)
    # If the scale prior failed, raw compiler geometry still defines the slots.
    proposed = compiled['final_actions'] or compiled.get('raw_compiler_actions',[])
    if not proposed:
        return dict(record,validation=compiled['validation'])
    try:
        proposed, evidence = realize_explicit_new_profiles(public,proposed)
    except (ValueError, KeyError, TypeError) as error:
        return dict(record,validation=dict(accepted=False,reason=str(error)))
    geometry = check_public_geometry(public,proposed)
    record.update(public_geometry_validation=geometry,explicit_geometry_binding=evidence)
    if not geometry['accepted']:
        return dict(record,validation=geometry)
    checked = validate_public_output(proposed,public['topology_contract'])
    return dict(record,final_actions=proposed if checked['accepted'] else [],validation=checked)


def finalize_with_geometry_constraints(row,llm_result):
    public = {k:deepcopy(row[k]) for k in PUBLIC_SOURCE_FIELDS}
    if llm_result['case_id'] != public['case_id']:
        raise ValueError('fallback_case_identity_mismatch')
    result = dict(case_id=public['case_id'],final_actions=[],llm_attempt=deepcopy(llm_result),
        llm_calls=llm_result.get('llm_calls',0),output_tokens=llm_result.get('output_tokens',0),
        compiler_calls=0,fallback_attempted=False,fallback_used=False,
        symbolic_stage_a_fallback=False,topology_success=False,
        selection_uses_private_reference=False,explicit_geometry_fidelity_certified=False)
    if llm_result.get('route')=='public_parameter_evidence_rejected':
        return dict(result,route='unresolved_public_input',failure=llm_result.get('failure'))
    lines = list(llm_result.get('final_actions') or [])
    checked = validate_public_output(lines,public['topology_contract'])
    geometry = check_public_geometry(public,lines)
    result.update(llm_validation=checked,llm_public_geometry_validation=geometry)
    if checked['accepted'] and geometry['accepted'] and not llm_result.get('failure'):
        return dict(result,final_actions=lines,route='llm_path_accepted',topology_success=True,failure=None)
    compiled = compile_geometry_constrained_fallback(public)
    result.update(compiler_calls=1,fallback_attempted=True,compiler_result=compiled,
        fallback_trigger=llm_result.get('failure') or
            (geometry['reason'] if not geometry['accepted'] else checked['reason']),
        compiler_validation=compiled['validation'])
    if compiled['validation']['accepted']:
        return dict(result,final_actions=compiled['final_actions'],route='compiler_fallback_accepted',
                    topology_success=True,fallback_used=True,symbolic_stage_a_fallback=True,failure=None)
    return dict(result,route='both_paths_failed',failure=compiled['validation']['reason'])
