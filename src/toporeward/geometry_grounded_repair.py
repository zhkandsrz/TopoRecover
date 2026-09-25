"""Opt-in realization of a topology repair using observed geometry only.

The topology compiler and its localization remain unchanged. This adapter
replaces geometry in its proposed profile transactions using explicit source
correspondences, or abstains. Source geometry is evidence, not geometric truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .actions import AddLine, Extrude, action_to_text
from .lm.parsing import parse_action_line
from .natural_repair_recall import apply_patch_operations, evaluate_repaired_history
from .topology_transaction_repair import _profile_transaction_spans


@dataclass(frozen=True)
class GroundedRepair:
    status: str
    actions: tuple[str, ...]
    reason: str
    evidence: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ground_repair_geometry(
    observed_lines: Sequence[str],
    proposed_lines: Sequence[str],
    contract: Mapping[str, Any],
) -> GroundedRepair:
    """Keep proposed topology, grounding loop geometry and extrusion depths.

    Existing profile IDs are authoritative. Compiler-created profile IDs may
    map to the source profile at the same sorted contract index. Missing profile
    evidence, missing loops and ambiguous correspondences cause abstention.
    No reference solid, target history or geometry scoring function is accepted.
    Kernel execution must still be checked by the caller before acceptance.
    """
    evidence: list[dict[str, Any]] = []

    def stop(reason: str) -> GroundedRepair:
        return GroundedRepair("abstain", (), reason, tuple(evidence))

    original_actions = [parse_action_line(str(s)) for s in observed_lines]
    proposed_actions = [parse_action_line(str(s)) for s in proposed_lines]
    if not original_actions or any(a is None for a in original_actions):
        return stop("unparseable_observed_history")
    if not proposed_actions or any(a is None for a in proposed_actions):
        return stop("unparseable_proposed_history")
    original = [action_to_text(a) for a in original_actions]
    proposed = [action_to_text(a) for a in proposed_actions]
    source_spans = sorted(_profile_transaction_spans(original, len(original)), key=lambda s: s.profile_id)
    final_spans = sorted(_profile_transaction_spans(proposed, len(proposed)), key=lambda s: s.profile_id)
    source_by_id = {s.profile_id: s for s in source_spans}
    if len(source_by_id) != len(source_spans) or len({s.profile_id for s in final_spans}) != len(final_spans):
        return stop("ambiguous_duplicate_profile_id")
    if len(final_spans) != len(contract.get("profiles") or []):
        return stop("proposed_profile_count_mismatch")
    edits: list[dict[str, Any]] = []
    mapped: dict[str, str] = {}
    used_profiles: set[str] = set()
    for index, span in enumerate(final_spans):
        source = source_by_id.get(span.profile_id)
        correspondence = "profile_identity"
        if source is None:
            # Only known compiler-generated IDs carry this ordinal meaning.
            aliases = {f"zz_repair_profile_{index:04d}", f"repair_profile_{index}", f"profile_{index}"}
            if span.profile_id not in aliases or index >= len(source_spans):
                return stop("missing_profile_geometry")
            source = source_spans[index]
            correspondence = "compiler_contract_index"
        if source.profile_id in used_profiles:
            return stop("ambiguous_profile_correspondence")
        used_profiles.add(source.profile_id)
        mapped[span.profile_id] = source.profile_id
        evidence.append({"kind": correspondence, "profile": span.profile_id,
                         "source_profile": source.profile_id,
                         "source_span": [source.face_start, source.register_end]})
        for role in ("outer", "inner"):
            src_loops = [s for s in source.loops if s.kind == role]
            dst_loops = [s for s in span.loops if s.kind == role]
            if len(dst_loops) > len(src_loops):
                return stop(f"missing_{role}_geometry")
            used_loops: set[int] = set()
            for ordinal, dest in enumerate(dst_loops):
                exact = [j for j, s in enumerate(src_loops) if j not in used_loops
                         and original[s.start:s.end] == proposed[dest.start:dest.end]]
                if exact:
                    selected = exact[0]
                elif len(src_loops) == len(dst_loops):
                    selected = ordinal
                    if selected in used_loops:
                        return stop("ambiguous_loop_correspondence")
                else:
                    return stop("ambiguous_surviving_loop")
                used_loops.add(selected)
                src = src_loops[selected]
                replacement = original[src.start:src.end]
                primitives = [parse_action_line(s) for s in replacement[1:-1]]
                # Closing an existing polyline reuses endpoints; a one-edge
                # sketch does not determine a missing rectangle width.
                if len(primitives) >= 2 and all(isinstance(a, AddLine) for a in primitives):
                    if primitives[-1].end != primitives[0].start:
                        replacement = [*replacement[:-1], action_to_text(AddLine(primitives[-1].end, primitives[0].start)), replacement[-1]]
                if replacement != proposed[dest.start:dest.end]:
                    edits.append({"start": dest.start, "end": dest.end, "replacement": replacement})
                evidence.append({"kind": "observed_loop", "role": role,
                                 "source_span": [src.start, src.end], "proposed_span": [dest.start, dest.end]})

    source_extrudes = [(i, a) for i, a in enumerate(original_actions) if isinstance(a, Extrude)]
    final_extrudes = [(i, a) for i, a in enumerate(proposed_actions) if isinstance(a, Extrude)]
    graph = list(contract.get("extrusion_graph") or [])
    if len(final_extrudes) != len(graph):
        return stop("proposed_extrusion_count_mismatch")
    used_extrudes: set[int] = set()
    for ordinal, (dest_index, dest) in enumerate(final_extrudes):
        specification = graph[ordinal]
        source_profile = mapped.get(dest.profile_id)
        if source_profile is None:
            return stop("unmapped_extrusion_profile")
        if specification.get("depth") is not None:
            depth = float(specification["depth"])
            anchor = {"kind": "explicit_contract_depth", "operation_index": ordinal}
        elif len(source_extrudes) == len(final_extrudes):
            source_index, action = source_extrudes[ordinal]
            # Equal counts preserve operation occurrence even when the repair
            # fixes a wrong reference or Boolean operation at that occurrence.
            depth = action.depth
            used_extrudes.add(source_index)
            anchor = {"kind": "observed_operation_occurrence", "source_action_index": source_index}
        else:
            matches = [(i, a) for i, a in source_extrudes if i not in used_extrudes
                       and a.profile_id == source_profile and a.op == dest.op]
            if len(matches) != 1:
                return stop("missing_or_ambiguous_extrusion_depth")
            source_index, action = matches[0]
            depth = action.depth
            used_extrudes.add(source_index)
            anchor = {"kind": "observed_profile_operation", "source_action_index": source_index}
        if not isfinite(depth) or depth <= 0:
            return stop("invalid_observed_depth")
        replacement = action_to_text(Extrude(dest.profile_id, depth, dest.op))
        if replacement != proposed[dest_index]:
            edits.append({"start": dest_index, "end": dest_index + 1, "replacement": [replacement]})
        evidence.append({**anchor, "proposed_action_index": dest_index, "depth": depth})
    grounded = apply_patch_operations(proposed, edits)
    outcome = evaluate_repaired_history(grounded, contract)
    if not outcome["valid_and_intent_satisfied"]:
        return GroundedRepair("verification_failed", (), "observed_geometry_not_topology_feasible", tuple(evidence))
    return GroundedRepair("candidate", tuple(grounded), "observed_geometry_grounded_occ_pending", tuple(evidence))
