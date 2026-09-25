"""Source-scale initialization for wholly new compiler-template profiles.

This is an explicit geometric prior, not recovery of an unknown target shape.
Only unmatched canonical profiles change; observed actions and topology stay fixed.
"""
from math import isfinite

from .actions import AddArc, AddCircle, AddLine, action_to_text
from .geometry_patch_slots import geometry_slots
from .natural_repair_recall import evaluate_repaired_history
from .lm.parsing import parse_action_line
from .topology_transaction_repair import _canonical_outer_actions, _profile_transaction_spans
from .verifier.geometry import arc_as_polyline, circle_as_polygon


def curve_points(action):
    if isinstance(action, AddLine):
        return [action.start, action.end]
    if isinstance(action, AddCircle):
        return circle_as_polygon(action.center, action.radius)
    if isinstance(action, AddArc):
        try:
            return arc_as_polyline(action.start, action.mid, action.end)
        except ValueError:
            return [action.start, action.mid, action.end]
    return []


def source_frame_realization(observed, proposed, contract):
    """Uniformly fit the new template bundle to the observed XY envelope.

    The bundle's relative positions and aspect ratio are retained. Its overall
    center is the observed envelope center; no target alignment is used. Existing
    profiles, extrusion parameters and action ordering are not transformed.
    """
    evidence = {'geometry_origin': 'source_scale_prior_not_intent_certified',
                'reference_geometry_access': False}
    points = [p for line in observed for p in curve_points(parse_action_line(line))]
    if not points or not all(isfinite(x) for p in points for x in p):
        return None, {**evidence, 'reason': 'no_finite_source_frame'}
    low = [min(p[d] for p in points) for d in (0, 1)]
    high = [max(p[d] for p in points) for d in (0, 1)]
    if any(high[d] <= low[d] for d in (0, 1)):
        return None, {**evidence, 'reason': 'degenerate_source_frame'}
    slots = geometry_slots(observed, proposed)
    mutable = {i for s in slots for i in range(s['start'], s['end'])}
    spans = _profile_transaction_spans(proposed, len(proposed))
    templates = [[action_to_text(a) for a in _canonical_outer_actions(i)] for i in range(len(spans))]
    selected, template_points = [], []
    for span in spans:
        outer = [loop for loop in span.loops if loop.kind == 'outer']
        if len(outer) != 1:
            continue
        loop = outer[0]
        body = [action_to_text(parse_action_line(s)) for s in proposed[loop.start + 1:loop.end - 1]]
        indices = [i for part in span.loops for i in range(part.start + 1, part.end - 1)]
        if body not in templates or not all(i in mutable for i in indices):
            continue
        selected.extend(indices)
        template_points.extend(p for s in body for p in curve_points(parse_action_line(s)))
    if not selected:
        return None, {**evidence, 'reason': 'no_unprotected_canonical_profile'}
    template_low = [min(p[d] for p in template_points) for d in (0, 1)]
    template_high = [max(p[d] for p in template_points) for d in (0, 1)]
    scale = min((high[d] - low[d]) / (template_high[d] - template_low[d]) for d in (0, 1))
    offset = [(high[d] + low[d] - scale * (template_high[d] + template_low[d])) / 2 for d in (0, 1)]

    def point(p):
        return tuple(scale * p[d] + offset[d] for d in (0, 1))

    output = list(proposed)
    for i in selected:
        action = parse_action_line(proposed[i])
        if isinstance(action, AddLine):
            updated = AddLine(point(action.start), point(action.end))
        elif isinstance(action, AddArc):
            updated = AddArc(point(action.start), point(action.mid), point(action.end))
        elif isinstance(action, AddCircle):
            updated = AddCircle(point(action.center), action.radius * scale)
        else:
            raise ValueError('Unexpected non-curve in a template loop')
        output[i] = action_to_text(updated)
    evaluation = evaluate_repaired_history(output, contract)
    evidence.update(source_xy_bounds=[low, high], template_xy_bounds=[template_low, template_high],
                    scale=scale, translation=offset, changed_action_indices=selected,
                    public_topology_evaluation=evaluation)
    if not evaluation['valid_and_intent_satisfied']:
        return None, {**evidence, 'reason': 'source_frame_topology_invalid'}
    return output, {**evidence, 'reason': 'source_frame_prior_occ_pending'}
