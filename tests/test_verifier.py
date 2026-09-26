from __future__ import annotations

import unittest

from toporeward.actions import (
    AddArc,
    AddCircle,
    AddLine,
    End,
    EndFace,
    EndLoop,
    EndSketch,
    Extrude,
    RegisterProfile,
    StartFace,
    StartLoop,
    StartSketch,
)
from toporeward.verifier import TopoVerifier


class TopoVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = TopoVerifier()

    def run_prefix(self, actions):
        state = self.verifier.initial_state()
        for action in actions:
            result = self.verifier.step(state, action)
            self.assertTrue(result.valid, result)
            state = result.next_state
        return state

    def test_valid_rectangle_program(self) -> None:
        actions = [
            StartSketch(), StartFace(), StartLoop("outer"),
            AddLine((0.0, 0.0), (1.0, 0.0)),
            AddLine((1.0, 0.0), (1.0, 1.0)),
            AddLine((1.0, 1.0), (0.0, 1.0)),
            AddLine((0.0, 1.0), (0.0, 0.0)),
            EndLoop(), EndFace(), RegisterProfile("profile_0"),
            EndSketch(), Extrude("profile_0", depth=0.5, op="add"), End(),
        ]
        result = self.verifier.verify_program(actions)
        self.assertTrue(result.valid, result)
        self.assertTrue(result.next_state.ended)

    def test_endpoint_discontinuity(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddLine((0, 0), (1, 0)),
            ]
        )
        result = self.verifier.step(state, AddLine((0.5, 0.5), (1, 1)))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "endpoint_discontinuity")

    def test_unclosed_loop(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddLine((0, 0), (1, 0)),
                AddLine((1, 0), (1, 1)),
            ]
        )
        result = self.verifier.step(state, EndLoop())
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "unclosed_loop")

    def test_self_intersection(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddLine((0, 0), (1, 1)),
                AddLine((1, 1), (0, 1)),
            ]
        )
        result = self.verifier.step(state, AddLine((0, 1), (1, 0)))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "self_intersection")

    def test_semicircular_arc_and_diameter_form_valid_loop(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddArc((0, 0), (1, 1), (2, 0)),
                AddLine((2, 0), (0, 0)),
            ]
        )
        result = self.verifier.step(state, EndLoop())
        self.assertTrue(result.valid, result)

    def test_invalid_profile_reference(self) -> None:
        state = self.verifier.initial_state()
        result = self.verifier.step(state, Extrude("missing", 1.0, "add"))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "invalid_profile_reference")

    def test_invalid_hole_containment(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddLine((0, 0), (1, 0)),
                AddLine((1, 0), (1, 1)),
                AddLine((1, 1), (0, 1)),
                AddLine((0, 1), (0, 0)),
                EndLoop(),
                StartLoop("inner"),
            ]
        )
        result = self.verifier.step(state, AddLine((1.5, 1.5), (1.7, 1.5)))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "invalid_hole_containment")

    def test_overlapping_inner_loop_is_invalid_at_end_loop(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddLine((0, 0), (4, 0)),
                AddLine((4, 0), (4, 4)),
                AddLine((4, 4), (0, 4)),
                AddLine((0, 4), (0, 0)),
                EndLoop(),
                StartLoop("inner"),
                AddLine((1, 1), (2, 1)),
                AddLine((2, 1), (2, 2)),
                AddLine((2, 2), (1, 2)),
                AddLine((1, 2), (1, 1)),
                EndLoop(),
                StartLoop("inner"),
                AddLine((1, 1), (2, 1)),
                AddLine((2, 1), (2, 2)),
                AddLine((2, 2), (1, 2)),
                AddLine((1, 2), (1, 1)),
            ]
        )
        result = self.verifier.step(state, EndLoop())
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "invalid_hole_containment")

    def test_inner_loop_line_on_outer_boundary_is_invalid(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddLine((0, 0), (4, 0)),
                AddLine((4, 0), (4, 4)),
                AddLine((4, 4), (0, 4)),
                AddLine((0, 4), (0, 0)),
                EndLoop(),
                StartLoop("inner"),
            ]
        )
        result = self.verifier.step(state, AddLine((0, 0), (1, 0)))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "invalid_hole_containment")

    def test_overlapping_inner_circle_is_invalid_at_add_circle(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddLine((0, 0), (4, 0)),
                AddLine((4, 0), (4, 4)),
                AddLine((4, 4), (0, 4)),
                AddLine((0, 4), (0, 0)),
                EndLoop(),
                StartLoop("inner"),
                AddCircle((2, 2), 0.4),
                EndLoop(),
                StartLoop("inner"),
            ]
        )
        result = self.verifier.step(state, AddCircle((2, 2), 0.4))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "invalid_hole_containment")

    def test_degenerate_curve(self) -> None:
        state = self.run_prefix([StartSketch(), StartFace(), StartLoop("outer")])
        result = self.verifier.step(state, AddLine((0, 0), (0, 0)))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "degenerate_curve")

    def test_hierarchy_error(self) -> None:
        state = self.verifier.initial_state()
        result = self.verifier.step(state, End())
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "hierarchy_error")

    def test_empty_sketch_is_invalid(self) -> None:
        state = self.run_prefix([StartSketch()])
        result = self.verifier.step(state, EndSketch())
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_type, "hierarchy_error")

    def test_register_then_extrude(self) -> None:
        state = self.run_prefix(
            [
                StartSketch(),
                StartFace(),
                StartLoop("outer"),
                AddLine((0, 0), (1, 0)),
                AddLine((1, 0), (1, 1)),
                AddLine((1, 1), (0, 1)),
                AddLine((0, 1), (0, 0)),
                EndLoop(),
                EndFace(),
                RegisterProfile("p0"),
                EndSketch(),
            ]
        )
        result = self.verifier.step(state, Extrude("p0", 1.0, "add"))
        self.assertTrue(result.valid, result)


if __name__ == "__main__":
    unittest.main()
