"""Geometry-only edits to frozen topology plans, with original actions locked."""
from __future__ import annotations

from difflib import SequenceMatcher
import ast
import json

from .actions import AddArc, AddCircle, AddLine, Extrude, action_from_dict, action_to_text
from .lm.parsing import parse_action_line
from .natural_repair_recall import apply_patch_operations
from .topology_transaction_repair import _profile_transaction_spans
from .verifier.geometry import arc_as_polyline, circle_as_polygon

CURVES = (AddLine, AddArc, AddCircle)


def _decode_explicit_actions(items):
    """Accept equivalent explicit constructors, without filling any argument."""
    fields = {"AddLine": {"start", "end"}, "AddArc": {"start", "mid", "end"},
              "AddCircle": {"center", "radius"}, "Extrude": {"profile_id", "depth", "op"}}
    result = []
    i = 0
    while i < len(items):
        item = items[i]
        if not isinstance(item, str):
            return None
        if item not in fields:
            action = parse_action_line(item)
            if action is None:
                return None
            result.append(action)
            i += 1
            continue
        name = item
        i += 1
        if i < len(items) and isinstance(items[i], dict):
            values = dict(items[i])
            i += 1
            if set(values) != fields[name]:
                return None
            try:
                for key in ("start", "mid", "end", "center"):
                    if key in values and isinstance(values[key], str):
                        values[key] = ast.literal_eval(values[key])
                action = action_from_dict({"type": name, **values})
            except (ValueError, SyntaxError, TypeError):
                return None
        else:
            arguments = []
            while i < len(items) and isinstance(items[i], str) and "=" in items[i] and items[i].split("=", 1)[0] in fields[name]:
                arguments.append(items[i])
                i += 1
            keys = [a.split("=", 1)[0] for a in arguments]
            if set(keys) != fields[name] or len(keys) != len(set(keys)):
                return None
            action = parse_action_line(f"{name}({','.join(arguments)})")
        if action is None:
            return None
        result.append(action)
    return result


def geometry_slots(observed, proposed):
    original = [action_to_text(parse_action_line(s)) for s in observed]
    final = [action_to_text(parse_action_line(s)) for s in proposed]
    protected = set()
    for block in SequenceMatcher(a=original, b=final, autojunk=False).get_matching_blocks():
        protected.update(range(block.b, block.b + block.size))
    slots = []
    for span in _profile_transaction_spans(final, len(final)):
        for loop in span.loops:
            start, end = loop.start + 1, loop.end - 1
            if start < end and not any(i in protected for i in range(start, end)):
                slots.append({"start": start, "end": end, "kind": "loop_geometry", "role": loop.kind})
            else:
                for i in range(start, end):
                    if i not in protected and isinstance(parse_action_line(final[i]), CURVES):
                        slots.append({"start": i, "end": i + 1, "kind": "curve_geometry", "role": loop.kind})
    for i, line in enumerate(final):
        if i not in protected and isinstance(parse_action_line(line), Extrude):
            slots.append({"start": i, "end": i + 1, "kind": "extrusion_depth"})
    return sorted(slots, key=lambda s: s["start"])


def geometry_prompt(observed, proposed, contract, slots):
    return "\n".join([
        "Fill only the permitted geometric slots in a CAD topology repair.",
        "The topology repair location, loop roles, profile references and operation order are fixed.",
        "The observed history is incomplete or erroneous. Preserve its supported dimensions, scale and construction context.",
        "The compiler may have inserted arbitrary large rectangles or default depths. Replace these placeholders with locally plausible geometry.",
        "Do not treat template coordinates as intended dimensions. Do not create remote disconnected blocks just to satisfy profile counts.",
        "For loop slots output only AddLine(start=(x,y), end=(x,y)), AddArc(start=(x,y), mid=(x,y), end=(x,y)), or AddCircle(center=(x,y), radius=r).",
        "A loop must be closed, non-self-intersecting and have its prescribed role. Inner loops must lie inside the outer loop without touching it.",
        "For an extrusion slot retain the same profile_id and op, changing only depth. Never change a non-slot action.",
        "You have no reference CAD or exact geometric target. These are feasible completion proposals, not certified recovery of unknown intent.",
        'Return only JSON: {"edits":[{"start":0,"end":1,"replacement":["Action",...]}]}.',
        "Each edit must use one exact listed [start,end) range. Use at most 32 curve primitives per slot. Indices refer to the fixed proposed history.",
        "If no reasonable completion is supported, return {\"edits\":[]}.",
        "OBSERVED HISTORY:", *[f"{i}: {s}" for i, s in enumerate(observed)],
        "TOPOLOGY REQUIREMENTS:", json.dumps(contract, sort_keys=True),
        "FIXED TOPOLOGY REPAIR:", *[f"{i}: {s}" for i, s in enumerate(proposed)],
        "PERMITTED SLOTS:", json.dumps(slots, sort_keys=True),
    ])


def geometry_prompt_masked(observed, proposed, contract, slots):
    """Expose geometric evidence, never numeric values of compiler placeholders."""
    points, depths, warnings = [], [], []
    for index, line in enumerate(observed):
        action = parse_action_line(line)
        if isinstance(action, AddLine):
            points.extend((action.start, action.end))
        elif isinstance(action, AddCircle):
            points.extend(circle_as_polygon(action.center, action.radius))
        elif isinstance(action, AddArc):
            try:
                points.extend(arc_as_polyline(action.start, action.mid, action.end))
            except ValueError:
                warnings.append({'action_index': index, 'issue': 'collinear observed arc; not usable as an arc'})
                points.extend((action.start, action.mid, action.end))
        elif isinstance(action, Extrude):
            depths.append({'action_index': index, 'profile_id': action.profile_id,
                           'operation': action.op, 'depth': action.depth})
    evidence = {'observed_extrusions': depths, 'warnings': warnings}
    if points:
        evidence['observed_sampled_xy_bounds'] = {
            'min': [min(p[d] for p in points) for d in (0, 1)],
            'max': [max(p[d] for p in points) for d in (0, 1)]}

    by_start = {s['start']: s for s in slots}
    hidden = {i for s in slots for i in range(s['start'], s['end'])}
    masked, constraints = [], []
    for i, line in enumerate(proposed):
        slot = by_start.get(i)
        if slot is None:
            if i not in hidden:
                masked.append(f'{i}: {line}')
            continue
        row = dict(slot)
        if slot['kind'] == 'extrusion_depth':
            action = parse_action_line(line)
            marker = f'Extrude(profile_id={action.profile_id}, depth=<unknown>, op={action.op})'
        else:
            marker = f"<unknown {slot['role']} {slot['kind']}; coordinates and primitive count are unspecified>"
            if slot['kind'] == 'curve_geometry':
                # Neighboring unknown slots cannot supply endpoint constraints.
                for neighbor, field, attribute in ((i - 1, 'required_start', 'end'),
                                                   (slot['end'], 'required_end', 'start')):
                    if 0 <= neighbor < len(proposed) and neighbor not in hidden:
                        action = parse_action_line(proposed[neighbor])
                        if isinstance(action, (AddLine, AddArc)):
                            row[field] = getattr(action, attribute)
        masked.append(f"[{i},{slot['end']}): {marker}")
        constraints.append(row)
    return '\n'.join([
        'Complete the unknown geometry in this fixed CAD topology repair.',
        'Return ONLY one JSON object with edits. Every replacement action is ONE complete string, not a constructor name and separate arguments.',
        'Example format: {"edits":[{"start":3,"end":4,"replacement":["AddCircle(center=(0.2,0.3), radius=0.05)"]}]}. The example numbers are NOT this design.',
        'Fill EVERY listed slot using its exact [start,end) indices in the fixed history. Do not use observed-history indices.',
        'Use only AddLine(start=(x,y), end=(x,y)), AddArc(start=(x,y), mid=(x,y), end=(x,y)), AddCircle(center=(x,y), radius=r) inside a curve slot.',
        'Keep Extrude profile_id and op EXACTLY as specified; only depth is unknown. All other actions, loop roles and references are locked.',
        'Infer dimensions from the observed history and explicit requirements. Observed bounds summarize incomplete input, not a target bounding box or a mandatory constraint.',
        'Preserve design scale. Do not invent remote blocks, tiny duplicate profiles, or extra features merely to match counts. Missing dimensions remain uncertain.',
        'A line chain must join exactly and close; a circular arc must pass through a noncollinear midpoint. Inner loops must be strictly inside their outer boundary and disjoint.',
        'When a curve gap has required_start/required_end, join those endpoints. A new outer loop and its inner loops must be designed together.',
        'New extrusion depths should use compatible observed operation depths when supported; do not assume depth=1.',
        'At most 32 curves per slot. If the entire completion is not supported, return {"edits":[]}. Do not omit individual slots or retain hidden numeric defaults.',
        'OBSERVED HISTORY:', *[f'{i}: {line}' for i, line in enumerate(observed)],
        'GEOMETRIC EVIDENCE (observation only, not target truth):', json.dumps(evidence, sort_keys=True),
        'TOPOLOGY REQUIREMENTS:', json.dumps(contract, sort_keys=True),
        'FIXED REPAIR WITH UNKNOWN GEOMETRY:', *masked,
        'REQUIRED EDIT SLOTS:', json.dumps(constraints, sort_keys=True),
    ])


def apply_geometry_patch(proposed, slots, text):
    # Match the repository's patch protocol: extract the first JSON patch from
    # prose/fences, but retain raw types for strict range/schema validation.
    value = None
    decoder = json.JSONDecoder()
    for i, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "edits" in candidate:
            value = candidate
            break
    if value is None:
        return None, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"edits"} or not isinstance(value["edits"], list):
        return None, "invalid_schema"
    if not value["edits"]:
        return None, "model_abstain"
    allowed = {(s["start"], s["end"]): s for s in slots}
    used = set()
    edits = []
    for edit in value["edits"]:
        if not isinstance(edit, dict) or set(edit) != {"start", "end", "replacement"}:
            return None, "invalid_edit"
        if type(edit["start"]) is not int or type(edit["end"]) is not int:
            return None, "invalid_index"
        key = (edit["start"], edit["end"])
        if key not in allowed or key in used:
            return None, "outside_or_duplicate_slot"
        used.add(key)
        replacement = edit["replacement"]
        if not isinstance(replacement, list) or not 1 <= len(replacement) <= 160:
            return None, "invalid_replacement"
        actions = _decode_explicit_actions(replacement)
        if actions is None or not 1 <= len(actions) <= 32:
            return None, "invalid_explicit_constructor"
        if allowed[key]["kind"] == "extrusion_depth":
            old = parse_action_line(proposed[key[0]])
            if len(actions) != 1 or not isinstance(actions[0], Extrude):
                return None, "non_extrusion_in_depth_slot"
            if actions[0].profile_id != old.profile_id or actions[0].op != old.op:
                return None, "topology_change_in_depth_slot"
        elif not all(isinstance(a, CURVES) for a in actions):
            return None, "non_geometry_in_curve_slot"
        edits.append({"start": key[0], "end": key[1], "replacement": [action_to_text(a) for a in actions]})
    # Require every placeholder to be addressed, rather than silently retain an
    # invented default in a partially edited history.
    if used != set(allowed):
        return None, "unfilled_geometry_slots"
    return apply_patch_operations(proposed, edits), "geometry_only_patch"
