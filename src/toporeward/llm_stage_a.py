"""Bounded LM proposals for Stage A, separate from deterministic fallback."""
from __future__ import annotations

import json
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

from .actions import Extrude, action_to_text
from .lm.parsing import parse_action_line
from .natural_repair_recall import apply_patch_operations, evaluate_repaired_history
from .strong_repair_baselines import (
    PATCH_SCHEMA, _dsl_text, parse_unrestricted_bounded_patch, runtime_trace,
)
from .topology_transaction_repair import (
    _profile_transaction_spans, _replay_prefix, compile_typed_transactions,
    evaluate_transaction, select_transaction,
)
from .verifier import TopoVerifier


MODES = ("runtime_only", "localization_only", "obligations")


def profile_diagnostics(lines: Sequence[str], contract: Mapping[str, Any]) -> list[dict]:
    """Inspect committed source spans without constructing a teacher patch."""
    trace = runtime_trace(lines)
    stop = trace.first_rejected_step
    stop = len(lines) if stop is None else stop
    spans = sorted(_profile_transaction_spans(lines, stop), key=lambda s: s.profile_id)
    desired = list(contract.get("profiles", []))
    diagnostics = []
    for index, span in enumerate(spans):
        actual = [loop.kind for loop in span.loops]
        expected = list(desired[index]["loop_roles"]) if index < len(desired) else []
        if Counter(actual) != Counter(expected):
            diagnostics.append({
                "profile_id": span.profile_id,
                "source_start": span.face_start,
                "source_end": span.register_end,
                "observed_roles": actual,
                "required_roles": expected,
            })
    return diagnostics


def prepare_request(row: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in (*MODES, "runtime_unrestricted"):
        raise ValueError(f"Unknown feedback mode: {mode}")
    # Explicit projection: private target actions/geometry never enter prompts.
    lines = list(row["observed_actions"])
    contract = dict(row["topology_contract"])
    trace = runtime_trace(lines)
    diagnostics = profile_diagnostics(lines, contract)
    if mode == "runtime_unrestricted":
        regions = [{"start": 0, "end": len(lines)}]
    elif mode == "runtime_only":
        start = trace.first_rejected_step
        start = len(lines) - 1 if start is None else start
        regions = [{"start": start, "end": len(lines)}]
    else:
        regions = [
            {"start": d["source_start"], "end": d["source_end"]}
            for d in diagnostics
        ]
    visible = {
        "design_brief": str(row["design_brief"]),
        "feature_plan": row["feature_plan"],
        "history": [{"step": i, "action": line} for i, line in enumerate(lines)],
        "runtime": {
            "status": trace.execution_status,
            "first_rejected_step": trace.first_rejected_step,
            "failure_type": trace.failure_type,
        },
        "editable_regions": regions,
    }
    if mode == "obligations":
        visible["profile_obligations"] = diagnostics
    prompt = (
        "Repair the CAD history with a bounded local patch. Follow the design brief "
        "and frozen feature plan. Keep unaffected curves, profile identifiers, and "
        "feature parameters unchanged. Do not modify the plan. A downstream compiler "
        "can complete execution after your patch, but cannot repair earlier topology "
        "on your behalf. Return only JSON: " + PATCH_SCHEMA + "\n"
        "Indices are zero-based and refer to the ORIGINAL history. Each edit replaces "
        "[start,end); start=end inserts before that action. Each edit must fit in one "
        "editable region. At most 2 non-overlapping edits, 32 total replacement actions. "
        "Return {\"edits\":[]} to abstain. No code fences or explanations.\n"
        "SUPPORTED DSL:\n" + _dsl_text() + "\nINPUT:\n"
        + json.dumps(visible, sort_keys=True)
    )
    return {
        "source_uid": f"{row['case_id']}::{mode}", "case_id": row["case_id"],
        "mode": mode, "prompt": prompt, "editable_regions": regions,
        "input_audit": {
            "reference_cad_visible": False, "oracle_patch_visible": False,
            "target_actions_visible": False, "teacher_patch_visible": False,
            "design_brief_visible": True, "frozen_plan_visible": True,
            "structured_obligations_visible": mode == "obligations",
        },
    }


def apply_lm_patch(lines: Sequence[str], response: str, regions: Sequence[Mapping]) -> tuple[list[str], list[dict]]:
    patch = parse_unrestricted_bounded_patch(
        response, observed_length=len(lines), max_edits=2, max_replacement_actions=32,
    )
    if patch is None:
        raise ValueError("invalid_patch_or_abstention")
    edits = patch["edits"]
    if sum(len(edit["replacement"]) for edit in edits) > 32:
        raise ValueError("total_action_budget_exceeded")
    for edit in edits:
        if not any(r["start"] <= edit["start"] <= edit["end"] <= r["end"] for r in regions):
            raise ValueError("edit_outside_localized_region")
    if len({e["start"] for e in edits}) != len(edits):
        raise ValueError("ambiguous_same_offset_edits")
    return apply_patch_operations(lines, edits, preserve_source_text=True), edits


def complete_after_patch(lines: Sequence[str], contract: Mapping[str, Any], *, protected_end: int,
                         preserve_reference_parameters: bool = False) -> tuple[list[str] | None, str]:
    """Run existing Stage B without rolling back over the accepted LM edit."""
    evaluation = evaluate_repaired_history(lines, contract)
    if evaluation["valid_ended"] and evaluation["intent_satisfied"]:
        return list(lines), "unchanged_already_complete"
    if profile_diagnostics(lines, contract):
        return None, "committed_topology_unresolved"
    trace = runtime_trace(lines)
    failure = trace.first_rejected_step
    failure = len(lines) - 1 if failure is None else failure
    if failure < protected_end:
        return None, "rejection_inside_protected_patch"
    if preserve_reference_parameters:
        action = parse_action_line(lines[failure])
        replayed = _replay_prefix(lines[:failure], TopoVerifier())
        if isinstance(action, Extrude) and replayed is not None:
            state, _ = replayed
            if action.profile_id not in state.profiles:
                # The contract identifies the reference; observed depth/op remain
                # authoritative. Do not regenerate the unaffected suffix.
                graph = list(contract.get("extrusion_graph") or [])
                index = len(state.extrusions)
                profiles = sorted(state.profiles)
                if index < len(graph):
                    obligation = graph[index]
                    target = int(obligation.get("profile_index", -1))
                    if 0 <= target < len(profiles) and obligation.get("operation") == action.op:
                        repaired = list(lines)
                        repaired[failure] = action_to_text(Extrude(profiles[target], action.depth, action.op))
                        checked = evaluate_repaired_history(repaired, contract)
                        if checked["valid_ended"] and checked["intent_satisfied"]:
                            return repaired, "reference_only_preserving_parameters"
                return None, "reference_only_completion_failed"
    _, transactions = compile_typed_transactions(lines, failure_step=failure, contract=contract)
    allowed = [t for t in transactions if t.repair_start >= protected_end
               and t.operator != "contract_divergence_repair"]
    selected = select_transaction([
        evaluate_transaction(lines, t, contract, failure_step=failure, horizon=6)
        for t in allowed
    ])
    if selected is None or not selected.valid_and_intent_satisfied:
        return None, "stage_b_no_valid_completion"
    return list(selected.repaired_actions), selected.transaction.operator


def evaluate_response(row: Mapping[str, Any], request: Mapping[str, Any], response: str,
                      *, preserve_reference_parameters: bool = False) -> dict:
    original = list(row["observed_actions"])
    result = {
        "case_id": row["case_id"], "mode": request["mode"],
        "patch_accepted": False, "stage_a_success": False,
        "topology_success": False, "fallback_used": False,
        "final_actions": [], "action_lcs": 0.0,
    }
    try:
        patched, edits = apply_lm_patch(original, response, request["editable_regions"])
    except (ValueError, TypeError, KeyError) as error:
        return {**result, "failure": str(error)}
    result.update(patch_accepted=True, edits=edits, patched_actions=patched)
    # End of the last replacement in patched coordinates, including deletions.
    shift = 0
    protected_end = 0
    for edit in edits:
        protected_end = max(protected_end, edit["start"] + shift + len(edit["replacement"]))
        shift += len(edit["replacement"]) - (edit["end"] - edit["start"])
    if _replay_prefix(patched[:protected_end], TopoVerifier()) is None:
        return {**result, "failure": "patch_prefix_invalid"}
    before_ids = [d["profile_id"] for d in profile_diagnostics(original, row["topology_contract"])]
    trace = runtime_trace(patched)
    stop = trace.first_rejected_step
    stop = len(patched) if stop is None else stop
    committed_ids = {s.profile_id for s in _profile_transaction_spans(patched, stop)}
    if not set(before_ids).issubset(committed_ids) or profile_diagnostics(patched, row["topology_contract"]):
        return {**result, "failure": "stage_a_obligations_unresolved"}
    original_trace = runtime_trace(original)
    original_stop = original_trace.first_rejected_step
    original_stop = len(original) if original_stop is None else original_stop
    def outer_blocks(history, limit):
        return {
            span.profile_id: [list(history[loop.start:loop.end]) for loop in span.loops if loop.kind == "outer"]
            for span in _profile_transaction_spans(history, limit)
        }
    old_outer, new_outer = outer_blocks(original, original_stop), outer_blocks(patched, stop)
    if any(new_outer.get(key) != value for key, value in old_outer.items()):
        return {**result, "failure": "protected_outer_boundary_changed"}
    result["stage_a_success"] = True
    final, reason = complete_after_patch(patched, row["topology_contract"], protected_end=protected_end,
                                        preserve_reference_parameters=preserve_reference_parameters)
    result["completion_status"] = reason
    if final is None:
        return {**result, "failure": reason}
    result.update(topology_success=True, final_actions=final)
    result["action_lcs"] = sum(
        block.size for block in SequenceMatcher(a=original, b=final, autojunk=False).get_matching_blocks()
    ) / max(1, len(original))
    return result
