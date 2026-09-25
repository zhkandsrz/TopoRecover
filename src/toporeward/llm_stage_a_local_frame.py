"""Observed-profile coordinate grounding for a bounded Stage A geometry proposal."""
from __future__ import annotations

import json
import math

from .actions import AddLine
from .lm.parsing import parse_action_line
from .llm_stage_a_geometry import (
    evaluate_geometry_response, number, positive, prepare_geometry_request, scope_for_case,
)

MODES = ("world_unassisted", "world_grounded", "local_grounded")


def observed_rectangle_frame(row):
    scope = scope_for_case(row)
    lines = row["observed_actions"][scope["source_start"]:scope["source_end"]]
    starts = [i for i, line in enumerate(lines) if line.strip() == "StartLoop(kind=outer)"]
    if len(starts) != 1:
        raise ValueError("requires_one_observed_outer_loop")
    start = starts[0] + 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "EndLoop")
    curves = [parse_action_line(line) for line in lines[start:end]]
    if len(curves) != 4 or not all(isinstance(a, AddLine) for a in curves):
        raise ValueError("only_axis_aligned_rectangle_frame_supported")
    points = [tuple(map(number, a.start)) for a in curves]
    if any(tuple(a.end) != b for a, b in zip(curves, points[1:]+points[:1])):
        raise ValueError("outer_loop_not_connected")
    xs, ys = sorted(set(x for x, _ in points)), sorted(set(y for _, y in points))
    if len(xs) != 2 or len(ys) != 2 or set(points) != {(x, y) for x in xs for y in ys}:
        raise ValueError("outer_loop_not_rectangle")
    if any(a.start[0] != a.end[0] and a.start[1] != a.end[1] for a in curves):
        raise ValueError("diagonal_outer_edge")
    width, height = positive(xs[1]-xs[0]), positive(ys[1]-ys[0])
    return {"origin": [xs[0], ys[0]], "width": width, "height": height,
            "shorter_side": min(width, height), "bounds": [xs[0], ys[0], xs[1], ys[1]],
            "provenance": "observed_outer_curves_only"}


def prepare_local_frame_request(row, mode):
    if mode not in MODES:
        raise ValueError("unknown_local_frame_mode")
    request = prepare_geometry_request(row, "loop_parameters")
    request.update(source_uid=f"{row['case_id']}::{mode}", mode=mode)
    if mode == "world_unassisted":
        return request
    frame = observed_rectangle_frame(row)
    request["observed_frame"] = frame
    request["input_audit"]["frame_from_observed_geometry_only"] = True
    context = "\nOBSERVED OUTER FRAME (computed from input, not target geometry):\n" + json.dumps(frame)
    if mode == "world_grounded":
        request["prompt"] += context + (
            "\nKeep world-coordinate output. The missing hole must lie strictly inside this outer frame. "
            "Do not copy the outer boundary as a hole. Use the brief for shape, position and size."
        )
        return request
    visible = {
        "design_brief": row["design_brief"], "feature_plan": row["feature_plan"],
        "profile_id": request["scope"]["profile_id"], "observed_frame": frame,
        "localized_profile": row["observed_actions"][request["scope"]["source_start"]:request["scope"]["source_end"]],
    }
    request["prompt"] = (
        "Propose the ONE missing inner loop after verifier localization. The existing outer rectangle is fixed. "
        "Read the design brief to determine shape, position and size. Output local fractions, NOT world coordinates. "
        "u is the fraction of outer width measured from LEFT to the hole CENTER; v is the fraction of outer height "
        "measured from BOTTOM to the hole CENTER. For rectangles, width_fraction and height_fraction are FULL hole "
        "width/outer width and FULL hole height/outer height. For circles, radius_fraction is hole RADIUS divided "
        "by the outer SHORTER SIDE; convert diameter to radius when needed. "
        "All fractions must be positive; the hole must be strictly contained. Do not duplicate the outer boundary. "
        "The compiler converts fractions exactly, inserts the loop and repairs later references; it will not "
        "clip, shrink or replace your geometry. Return exactly one JSON object, without Markdown or explanation: "
        '{"profile_id":"<given>","loop":{"shape":"circle","center_fraction":[u,v],"radius_fraction":r}} '
        'or {"profile_id":"<given>","loop":{"shape":"rectangle","center_fraction":[u,v],'
        '"width_fraction":w,"height_fraction":h}}.\nINPUT:\n' + json.dumps(visible, sort_keys=True)
    )
    return request


def world_parameters(row, response):
    data = json.loads(response)
    scope = scope_for_case(row)
    if data["profile_id"] != scope["profile_id"]:
        raise ValueError("wrong_profile_id")
    frame = observed_rectangle_frame(row)
    loop = data["loop"]
    uv = loop["center_fraction"]
    if not isinstance(uv, list) or len(uv) != 2:
        raise ValueError("invalid_local_center")
    u, v = map(number, uv)
    if not (0 < u < 1 and 0 < v < 1):
        raise ValueError("local_center_outside")
    w, h = frame["width"], frame["height"]
    x0, y0 = frame["origin"]
    result = {"shape": loop["shape"], "center": [x0+u*w, y0+v*h]}
    if loop["shape"] == "circle":
        r = positive(loop["radius_fraction"]) * min(w, h)
        result["radius"] = r
        dx = dy = r
    elif loop["shape"] == "rectangle":
        result.update(width=positive(loop["width_fraction"])*w,
                      height=positive(loop["height_fraction"])*h)
        dx, dy = result["width"]/2, result["height"]/2
    else:
        raise ValueError("unsupported_loop_shape")
    values = [*result["center"], dx, dy]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite_transformed_geometry")
    if not (dx < min(u, 1-u)*w and dy < min(v, 1-v)*h):
        raise ValueError("local_hole_not_strictly_contained")
    return {"profile_id": data["profile_id"], "loop": result}


def evaluate_local_frame_response(row, request, response):
    mode = request["mode"]
    if mode == "local_grounded":
        try:
            response = json.dumps(world_parameters(row, response))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return {"case_id": row["case_id"], "mode": mode, "stage_a_success": False,
                    "topology_success": False, "final_actions": [], "action_lcs": 0.0,
                    "fallback_used": False, "failure": f"local_geometry_output:{error}"}
    result = evaluate_geometry_response(row, {**request, "mode": "loop_parameters"}, response)
    result["mode"] = mode
    result["geometry_clipped_or_replaced"] = False
    return result
