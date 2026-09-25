"""Check explicit upstream geometry separately from anonymous topology labels.

Only fully specified XY circles and axis-aligned rectangles are supported here.
Missing geometry is unknown, not evidence for a target shape. Unsupported or
ambiguous explicit specifications are reported, never silently certified.
"""
from math import isclose

from shapely.geometry import Polygon

from .actions import AddCircle, AddLine, Extrude, action_to_text
from .geometry_patch_slots import geometry_slots
from .llm_stage_a_geometry import number, positive
from .lm.parsing import parse_action_line
from .natural_repair_recall import apply_patch_operations
from .stage_b_preserving_geometry import bind_plan_dimensions
from .topology_transaction_repair import _profile_transaction_spans


def explicit_curves(geometry):
    if not isinstance(geometry, dict):
        raise ValueError('invalid_public_profile_geometry')
    shape = geometry.get('shape')
    fields = {'circle': {'shape', 'center', 'radius'},
              'rectangle': {'shape', 'center', 'width', 'height'}}
    if shape not in fields or set(geometry) != fields[shape]:
        raise ValueError('unsupported_or_incomplete_public_profile_geometry')
    center = geometry['center']
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError('invalid_public_geometry_center')
    x, y = map(number, center)
    if shape == 'circle':
        return [AddCircle((x, y), positive(geometry['radius']))]
    w, h = positive(geometry['width']), positive(geometry['height'])
    points = [(x-w/2,y-h/2),(x+w/2,y-h/2),(x+w/2,y+h/2),(x-w/2,y+h/2)]
    return [AddLine(a,b) for a,b in zip(points,points[1:]+points[:1])]


def _requirements(source, proposed):
    raw = source['feature_plan']
    if not raw:
        return [], []
    plan = raw.get('feature_plan', raw)
    bound, depths = bind_plan_dimensions(source['topology_contract'], raw)
    profiles = plan['profiles']
    specifications = [(i,p) for i,p in enumerate(profiles) if 'geometry' in p]
    if not specifications:
        return [], depths
    spans = sorted(_profile_transaction_spans(proposed, len(proposed)),key=lambda s:s.profile_id)
    if len(spans) != len(profiles) or len({s.profile_id for s in spans}) != len(spans):
        raise ValueError('public_geometry_profile_correspondence_unresolved')
    by_id = {s.profile_id:s for s in spans}
    used, requirements = set(), []
    for index, profile in specifications:
        pid = profile['profile_id']
        if profile.get('sketch_plane','XY') != 'XY':
            raise ValueError('unsupported_public_geometry_plane')
        span = by_id.get(pid)
        if span is None:
            # Only the compiler's documented contract-index names are aliases.
            aliases = {f'zz_repair_profile_{index:04d}', f'repair_profile_{index}',f'profile_{index}'}
            candidates = [s for s in spans if s.profile_id in aliases
                          and s.profile_id not in {p['profile_id'] for p in profiles}]
            if len(candidates) != 1:
                raise ValueError('public_geometry_profile_correspondence_unresolved')
            span = candidates[0]
        if span.profile_id in used or spans[index] != span:
            raise ValueError('public_geometry_profile_correspondence_conflict')
        used.add(span.profile_id)
        outer = [loop for loop in span.loops if loop.kind == 'outer']
        if len(outer) != 1:
            raise ValueError('public_geometry_outer_loop_unresolved')
        requirements.append(dict(profile_id=pid,output_profile_id=span.profile_id,
            start=outer[0].start+1,end=outer[0].end-1,
            expected=explicit_curves(profile['geometry'])))
    return requirements, depths


def _matches(expected, actual):
    if isinstance(expected[0], AddCircle):
        if len(actual) != 1 or not isinstance(actual[0], AddCircle):
            return False
        a, b = expected[0], actual[0]
        tol = max(1e-9, a.radius*1e-8)
        return all(abs(x-y) <= tol for x,y in zip((*a.center,a.radius),(*b.center,b.radius)))
    if len(actual) < 3 or not all(isinstance(a, AddLine) for a in actual):
        return False
    target = Polygon([a.start for a in expected])
    span = max(target.bounds[2]-target.bounds[0],target.bounds[3]-target.bounds[1])
    tol = max(1e-9,span*1e-8)
    for a,b in zip(actual,actual[1:]+actual[:1]):
        if any(abs(x-y) > tol for x,y in zip(a.end,b.start)):
            return False
    polygon = Polygon([a.start for a in actual])
    return (polygon.is_valid and not polygon.is_empty
            and target.hausdorff_distance(polygon) <= tol
            and target.symmetric_difference(polygon).area <= tol*target.length)


def check_public_geometry(source, proposed):
    """Validate only explicit supported requirements, not full design intent."""
    audit = dict(accepted=False,checked_profiles=[],checked_depths=[],
                 private_reference_access=False,full_geometry_certified=False)
    try:
        requirements, depths = _requirements(source,proposed)
        for spec in requirements:
            actual = [parse_action_line(s) for s in proposed[spec['start']:spec['end']]]
            if not _matches(spec['expected'],actual):
                return dict(audit,reason='explicit_profile_geometry_mismatch',
                            mismatch_profile=spec['profile_id'])
            audit['checked_profiles'].append(spec['profile_id'])
        extrudes = [parse_action_line(s) for s in proposed
                    if isinstance(parse_action_line(s),Extrude)]
        for spec in depths:
            index = spec['operation_index']
            if index >= len(extrudes) or not isclose(extrudes[index].depth,spec['depth'],
                                                    rel_tol=1e-8,abs_tol=1e-9):
                return dict(audit,reason='explicit_extrusion_depth_mismatch')
            audit['checked_depths'].append(index)
    except (ValueError, KeyError, TypeError) as error:
        return dict(audit,reason=str(error))
    return dict(audit,accepted=True,reason='explicit_requirements_checked',
                explicit_requirement_count=len(requirements)+len(depths))


def realize_explicit_new_profiles(source, proposed):
    """Bind supplied geometry only in wholly unobserved compiler loop slots."""
    requirements, _ = _requirements(source,proposed)
    slots = geometry_slots(source['observed_actions'],proposed) if requirements else []
    edits, evidence = [], []
    for spec in requirements:
        actual = [parse_action_line(s) for s in proposed[spec['start']:spec['end']]]
        if _matches(spec['expected'],actual):
            continue
        if not any(s['kind']=='loop_geometry' and s['start']==spec['start']
                   and s['end']==spec['end'] for s in slots):
            raise ValueError('explicit_geometry_would_overwrite_observed_loop')
        edits.append(dict(start=spec['start'],end=spec['end'],
                          replacement=[action_to_text(a) for a in spec['expected']]))
        evidence.append(dict(profile_id=spec['profile_id'],output_profile_id=spec['output_profile_id'],
            source='public_feature_plan_geometry',start=spec['start'],end=spec['end']))
    return apply_patch_operations(proposed,edits) if edits else list(proposed), evidence
