from copy import deepcopy
import pytest

from toporeward.feature_plan import feature_plan_to_topology_contract
from toporeward.lm.parsing import parse_action_line
from toporeward.actions import AddCircle
from toporeward.stage_a_geometry_constrained_fallback import compile_geometry_constrained_fallback
from toporeward.stage_a_geometry_constrained_fallback import finalize_with_geometry_constraints
from toporeward.stage_a_source_frame_fallback import compile_source_frame_fallback
from toporeward.public_profile_geometry import (
    check_public_geometry, explicit_curves, realize_explicit_new_profiles)
from toporeward.actions import action_to_text


def example():
    plan = dict(profiles=[dict(profile_id='p0',loop_roles=['outer']),
                          dict(profile_id='p1',loop_roles=['outer'],
                               geometry=dict(shape='circle',center=[4.,2.],radius=.4))],
                features=[dict(feature_id=f'f{i}',type='extrude',profile_id=f'p{i}',
                               operation='add',depth=.5) for i in range(2)])
    return dict(case_id='public_geometry_separate_boss',feature_plan=plan,
        topology_contract=feature_plan_to_topology_contract(plan),
        design_brief='Keep p0. Add p1 as explicitly specified in the feature plan.',
        observed_actions=['StartSketch','StartFace','StartLoop(kind=outer)',
            'AddCircle(center=(0,0), radius=1)','EndLoop','EndFace',
            'RegisterProfile(profile_id=p0)','EndSketch',
            'Extrude(profile_id=p0, depth=0.5, op=add)','End'])


def test_compiler_must_honor_explicit_new_profile_geometry():
    row = example()
    before = deepcopy(row)
    result = compile_geometry_constrained_fallback(row)
    assert result['validation']['accepted']
    curves = [parse_action_line(s) for s in result['final_actions']]
    assert AddCircle((4.,2.),.4) in curves
    assert row == before


@pytest.mark.parametrize('geometry',[
    dict(shape='circle',center=[-.01,.005],radius=.001),
    dict(shape='rectangle',center=[35.,-8.],width=4.,height=1.),
    dict(shape='rectangle',center=[.035,-.008],width=.004,height=.001),
])
def test_multiple_shapes_scales_and_placements(geometry):
    row = example()
    row['feature_plan']['profiles'][1]['geometry'] = geometry
    result = compile_geometry_constrained_fallback(row)
    assert result['validation']['accepted']
    assert check_public_geometry(row,result['final_actions'])['accepted']
    parsed = [parse_action_line(s) for s in result['final_actions']]
    assert all(parse_action_line(action_to_text(a)) in parsed for a in explicit_curves(geometry))
    assert AddCircle((0.,0.),1.) in parsed


def test_topology_valid_but_wrong_llm_geometry_triggers_original_input_fallback():
    row = example()
    old = compile_source_frame_fallback(row)
    assert old['validation']['accepted']
    assert not check_public_geometry(row,old['final_actions'])['accepted']
    attempt = dict(case_id=row['case_id'],final_actions=old['final_actions'],llm_calls=2)
    result = finalize_with_geometry_constraints(row,attempt)
    assert result['fallback_used'] and result['llm_calls']==2
    assert result['fallback_trigger']=='explicit_profile_geometry_mismatch'
    assert result['llm_attempt']==attempt
    assert check_public_geometry(row,result['final_actions'])['accepted']


def test_valid_llm_is_not_replaced_or_called_again():
    row = example()
    lines = compile_geometry_constrained_fallback(row)['final_actions']
    attempt = dict(case_id=row['case_id'],final_actions=lines,llm_calls=1)
    result = finalize_with_geometry_constraints(row,attempt)
    assert result['route']=='llm_path_accepted' and result['compiler_calls']==0
    assert result['final_actions']==lines and not result['explicit_geometry_fidelity_certified']


def test_exact_rectangle_geometry_is_independent_of_start_vertex_and_orientation():
    row = example()
    row['feature_plan']['profiles'][1]['geometry']=dict(shape='rectangle',center=[4.,2.],width=2.,height=1.)
    result = compile_geometry_constrained_fallback(row)
    from toporeward.topology_transaction_repair import _profile_transaction_spans
    span = sorted(_profile_transaction_spans(result['final_actions'],len(result['final_actions'])),
                  key=lambda s:s.profile_id)[1]
    loop = span.loops[0]
    edges = [parse_action_line(s) for s in result['final_actions'][loop.start+1:loop.end-1]]
    from toporeward.actions import AddLine
    reverse = [AddLine(e.end,e.start) for e in reversed(edges)]
    reverse = reverse[2:]+reverse[:2]
    changed = list(result['final_actions'])
    changed[loop.start+1:loop.end-1]=[action_to_text(a) for a in reverse]
    assert check_public_geometry(row,changed)['accepted']


@pytest.mark.parametrize('geometry',[
    dict(shape='ellipse',center=[4.,2.],radius=.4),
    dict(shape='circle',center=[4.,2.]),
    dict(shape='circle',center=[4.,2.],radius=-1.),
    dict(shape='circle',center=[4.,2.],radius=float('nan')),
    dict(shape='circle',center=[4.,2.],radius=True),
    dict(shape='rectangle',center=[4.,2.],width=2.,height=1.,rotation=20),
])
def test_unsupported_explicit_requirements_are_not_silently_ignored(geometry):
    row=example()
    row['feature_plan']['profiles'][1]['geometry']=geometry
    result=compile_geometry_constrained_fallback(row)
    assert not result['validation']['accepted'] and not result['final_actions']


def test_matching_observed_geometry_stays_unchanged_but_conflicts_do_not_get_overwritten():
    row=example()
    raw=compile_source_frame_fallback(row)['final_actions']
    row['feature_plan']['profiles'][0]['geometry']=dict(shape='circle',center=[0.,0.],radius=1.)
    changed,_=realize_explicit_new_profiles(row,raw)
    assert AddCircle((0.,0.),1.) in [parse_action_line(s) for s in changed]
    row['feature_plan']['profiles'][0]['geometry']['radius']=2.
    with pytest.raises(ValueError,match='overwrite_observed'):
        realize_explicit_new_profiles(row,raw)


def test_ambiguous_profile_id_does_not_bind_geometry_by_arbitrary_order():
    row=example()
    raw=compile_source_frame_fallback(row)['final_actions']
    raw=[s.replace('zz_repair_profile_0001','unidentified_profile') for s in raw]
    with pytest.raises(ValueError,match='correspondence'):
        realize_explicit_new_profiles(row,raw)


def test_unknown_geometry_is_not_falsely_certified():
    row=example()
    row['feature_plan']['profiles'][1].pop('geometry')
    result=compile_geometry_constrained_fallback(row)
    assert result['validation']['accepted']
    assert not result['explicit_geometry_fidelity_certified']
    assert result['public_geometry_validation']['checked_profiles']==[]
    assert result['public_geometry_validation']['checked_depths']==[0,1]


def test_explicit_depth_is_verified_and_missing_depth_bound_to_compiler():
    row=example()
    row['feature_plan']['features'][1]['depth']=2.5
    result=compile_geometry_constrained_fallback(row)
    assert result['validation']['accepted']
    assert check_public_geometry(row,result['final_actions'])['accepted']
    bad=[s.replace('depth=2.5','depth=1.5') for s in result['final_actions']]
    assert check_public_geometry(row,bad)['reason']=='explicit_extrusion_depth_mismatch'


def test_private_target_is_ignored_and_original_input_is_immutable():
    row=example()
    baseline=compile_geometry_constrained_fallback(row)
    row['target_actions']=['PRIVATE_NONEXISTENT_GEOMETRY']
    before=deepcopy(row)
    actual=compile_geometry_constrained_fallback(row)
    assert actual['final_actions']==baseline['final_actions'] and row==before


def test_kernel_infrastructure_error_is_not_a_refusal(monkeypatch):
    import toporeward.stage_a_geometry_constrained_fallback as policy
    def unavailable(*a):
        raise RuntimeError('kernel_unavailable')
    monkeypatch.setattr(policy,'validate_public_output',unavailable)
    row=example()
    with pytest.raises(RuntimeError,match='kernel_unavailable'):
        policy.finalize_with_geometry_constraints(row,dict(case_id=row['case_id'],final_actions=[]))
