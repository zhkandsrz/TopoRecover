"""Geometry evaluator, extracted without metric changes from the paper evaluator."""
from __future__ import annotations
import sys
from typing import Any
import numpy as np
from scipy.spatial import cKDTree

def capture_execute(lines: list[str], samples: int) -> tuple[dict[str, Any], Any | None]:
    from toporeward.cadquery_executor import execute_actions
    from toporeward.lm.parsing import parse_action_line

    actions = [parse_action_line(line) for line in lines]
    if not actions or any(action is None for action in actions):
        return {"success": False, "failure_type": "parse_error"}, None
    captured: dict[str, Any] = {}
    code = execute_actions.__code__

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "return" and frame.f_code is code:
            captured["shape"] = frame.f_locals.get("model")

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = execute_actions(
            [action for action in actions if action is not None],
            include_surface_evidence=True,
            surface_sample_count=samples,
        )
    finally:
        sys.setprofile(previous)
    return result.to_dict(), captured.get("shape")


def metrics(candidate: dict[str, Any], candidate_shape: Any, target: dict[str, Any], target_shape: Any) -> dict[str, float]:
    cp = np.asarray(candidate["surface_evidence"]["point_samples"], dtype=np.float64)
    tp = np.asarray(target["surface_evidence"]["point_samples"], dtype=np.float64)
    c_to_t = cKDTree(tp).query(cp, k=1)[0]
    t_to_c = cKDTree(cp).query(tp, k=1)[0]
    bbox = np.asarray(target["surface_evidence"]["bbox"], dtype=np.float64)
    diagonal = float(np.linalg.norm(bbox[3:] - bbox[:3]))
    if not np.isfinite(diagonal) or diagonal <= 1e-12:
        raise ValueError("nonpositive target bbox diagonal")
    intersection = float(candidate_shape.intersect(target_shape).Volume())
    union = float(candidate_shape.Volume()) + float(target_shape.Volume()) - intersection
    if union <= 1e-12:
        raise ValueError("nonpositive OCC union volume")
    return {
        "chamfer_l2_normalized": float(0.5 * (c_to_t.mean() + t_to_c.mean()) / diagonal),
        "hausdorff_l2_normalized": float(max(c_to_t.max(), t_to_c.max()) / diagonal),
        "volume_iou": float(max(0.0, min(1.0, intersection / union))),
        "target_bbox_diagonal": diagonal,
    }
