"""Command preservation measures used in the paper."""
from __future__ import annotations
from typing import Any, Sequence
from difflib import SequenceMatcher
from toporeward.actions import AddArc, AddCircle, AddLine, Extrude
from toporeward.lm.parsing import parse_action_line

def lcs_ratio(original: Sequence[str], repaired: Sequence[str]) -> float:
    if not original:
        return 1.0
    matcher = SequenceMatcher(a=list(original), b=list(repaired), autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(original)


def preservation_metrics(original: Sequence[str], repaired: Sequence[str]) -> dict[str, Any]:
    original_rows = list(original)
    repaired_rows = list(repaired)
    matcher = SequenceMatcher(a=original_rows, b=repaired_rows, autojunk=False)
    matched_original: set[int] = set()
    for block in matcher.get_matching_blocks():
        matched_original.update(range(block.a, block.a + block.size))

    curves: list[int] = []
    extrusions: list[int] = []
    for index, line in enumerate(original_rows):
        action = parse_action_line(line)
        if isinstance(action, (AddLine, AddArc, AddCircle)):
            curves.append(index)
        elif isinstance(action, Extrude):
            extrusions.append(index)

    def retention(indices: list[int]) -> float:
        return sum(index in matched_original for index in indices) / len(indices) if indices else 1.0

    prefix = 0
    for left, right in zip(original_rows, repaired_rows):
        if left != right:
            break
        prefix += 1
    edit_hunks = sum(tag != "equal" for tag, *_ in matcher.get_opcodes())
    return {
        "action_lcs": lcs_ratio(original_rows, repaired_rows),
        "curve_parameter_retention": retention(curves),
        "extrusion_parameter_retention": retention(extrusions),
        "preserved_prefix_rate": prefix / len(original_rows) if original_rows else 1.0,
        "edit_hunks": edit_hunks,
    }
