"""LM geometry instantiation after verifier localization, with no patch fallback."""
from __future__ import annotations

from collections import Counter
import json
import math

from .actions import AddArc, AddCircle, AddLine
from .lm.parsing import parse_action_line
from .llm_stage_a import evaluate_response, prepare_request, profile_diagnostics

MODES = ("loop_dsl", "loop_parameters")


def scope_for_case(row):
    diagnostics = profile_diagnostics(row["observed_actions"], row["topology_contract"])
    if len(diagnostics) != 1:
        raise ValueError("pilot_requires_one_localized_profile")
    diagnostic = diagnostics[0]
    actual, required = Counter(diagnostic["observed_roles"]), Counter(diagnostic["required_roles"])
    if required - actual != Counter({"inner": 1}) or actual - required:
        raise ValueError("pilot_supports_one_missing_inner_loop_only")
    end_faces = [i for i in range(diagnostic["source_start"], diagnostic["source_end"])
                 if row["observed_actions"][i].strip() == "EndFace"]
    if len(end_faces) != 1:
        raise ValueError("ambiguous_profile_commit")
    return {**diagnostic, "insertion_index": end_faces[0]}


def prepare_geometry_request(row, mode):
    if mode not in MODES:
        raise ValueError("unknown_geometry_interface")
    scope = scope_for_case(row)
    visible = {
        "design_brief": row["design_brief"], "feature_plan": row["feature_plan"],
        "profile_id": scope["profile_id"], "observed_roles": scope["observed_roles"],
        "required_roles": scope["required_roles"],
        "localized_profile": row["observed_actions"][scope["source_start"]:scope["source_end"]],
    }
    if mode == "loop_dsl":
        output = (
            'Return {"profile_id":"<given profile>","curves":["<curve action>", ...]}. '
            "Curves must be AddCircle(center=(x,y), radius=r), or a connected closed sequence of "
            "AddLine(start=(x,y), end=(x,y)) / AddArc(start=(x,y), mid=(x,y), end=(x,y)). "
            "Do not emit StartLoop, EndLoop, EndFace, registration, extrusion, or edit indices. "
            "For a rectangle, compute its four corners and emit four connected lines."
        )
    else:
        output = (
            'Return {"profile_id":"<given profile>","loop":{"shape":"circle","center":[x,y],"radius":r}} '
            'or {"profile_id":"<given profile>","loop":{"shape":"rectangle","center":[x,y],"width":w,"height":h}}. '
            "Use the shape required by the brief. Convert diameter to radius if necessary. "
            "If rectangle bounds are given, compute center, width and height."
        )
    prompt = (
        "TASK: construct the one missing INNER LOOP in the given CAD profile. "
        "The verifier has already located the profile and established that this loop is missing. "
        "The existing outer loop is correct and must remain unchanged. "
        "Even if the history executes, it does not satisfy the requested hole. "
        "Your only responsibility is the missing loop's shape and geometry from the design brief, "
        "in the same coordinate frame as the profile. "
        "The patch compiler will insert your loop before EndFace and a downstream stage handles references. "
        "Do not repair extrusion or reproduce existing curves. " + output + "\n"
        "Output one valid JSON object only: double-quoted keys and strings, numeric JSON values, "
        "no trailing commas, Markdown fences, or explanations.\nINPUT:\n" + json.dumps(visible, sort_keys=True)
    )
    return {"source_uid": f"{row['case_id']}::{mode}", "case_id": row["case_id"], "mode": mode,
            "prompt": prompt, "scope": scope,
            "input_audit": {"reference_cad_visible": False, "private_requirement_visible": False,
                            "teacher_geometry_visible": False, "oracle_patch_visible": False,
                            "design_brief_visible": True, "verifier_localization_visible": True}}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("nonfinite_or_nonnumeric_geometry")
    return float(value)


def positive(value):
    result = number(value)
    if result <= 0:
        raise ValueError("nonpositive_dimension")
    return result


def compile_geometry_response(row, request, response):
    data = json.loads(response)
    scope = scope_for_case(row)
    if data["profile_id"] != scope["profile_id"]:
        raise ValueError("wrong_profile_id")
    if request["mode"] == "loop_dsl":
        curves = data["curves"]
        if not isinstance(curves, list) or not 1 <= len(curves) <= 30:
            raise ValueError("invalid_curve_count")
        if any(not isinstance(line, str) or not isinstance(parse_action_line(line), (AddCircle, AddLine, AddArc)) for line in curves):
            raise ValueError("only_curve_actions_allowed")
    else:
        loop = data["loop"]
        if not isinstance(loop["center"], list) or len(loop["center"]) != 2:
            raise ValueError("invalid_center")
        cx, cy = map(number, loop["center"])
        if loop["shape"] == "circle":
            radius = positive(loop["radius"])
            curves = [f"AddCircle(center=({cx},{cy}), radius={radius})"]
        elif loop["shape"] == "rectangle":
            width, height = positive(loop["width"]), positive(loop["height"])
            vertices = [(cx-width/2, cy-height/2), (cx+width/2, cy-height/2),
                        (cx+width/2, cy+height/2), (cx-width/2, cy+height/2)]
            curves = [f"AddLine(start={a}, end={b})" for a, b in zip(vertices, vertices[1:]+vertices[:1])]
        else:
            raise ValueError("unsupported_loop_shape")
    at = scope["insertion_index"]
    return {"edits": [{"start": at, "end": at, "replacement": ["StartLoop(kind=inner)", *curves, "EndLoop"]}]}


def evaluate_geometry_response(row, request, response):
    try:
        patch = compile_geometry_response(row, request, response)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return {"case_id": row["case_id"], "mode": request["mode"], "patch_accepted": False,
                "stage_a_success": False, "topology_success": False, "fallback_used": False,
                "final_actions": [], "action_lcs": 0.0, "failure": f"geometry_output:{error}"}
    base = prepare_request(row, "obligations")
    base["mode"] = request["mode"]
    result = evaluate_response(row, base, json.dumps(patch))
    result["compiled_lm_patch"] = patch
    return result
