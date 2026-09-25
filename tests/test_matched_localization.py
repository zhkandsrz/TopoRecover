import json
import re
import unittest

from toporeward.matched_localization import (
    SELECTORS, assert_only_regions_change, request, select_regions, syntax_grammar)
from toporeward.llm_stage_a import apply_lm_patch


def case():
    return dict(case_id='missing_hole', design_brief='Plate with a round hole',
                feature_plan={}, topology_contract={'profiles': [{'loop_roles': ['outer', 'inner']}]},
                observed_actions=['StartSketch', 'StartFace', 'StartLoop(kind=outer)',
                    'AddLine(start=(0,0), end=(10,0))', 'AddLine(start=(10,0), end=(10,10))',
                    'AddLine(start=(10,10), end=(0,10))', 'AddLine(start=(0,10), end=(0,0))',
                    'EndLoop', 'EndFace', 'RegisterProfile(profile_id=plate)', 'EndSketch',
                    'Extrude(profile_id=unknown, depth=1, op=add)', 'End'])


class MatchedLocalizationTests(unittest.TestCase):
    def test_identical_prompt_except_region(self):
        rows = [request(case(), s) for s in SELECTORS]
        assert_only_regions_change(rows)
        rows[0]['prompt'] += 'changed'
        with self.assertRaises((AssertionError, json.JSONDecodeError)):
            assert_only_regions_change(rows)

    def test_earlier_profile_and_later_rejection(self):
        row = case()
        self.assertEqual(select_regions(row, 'topology'), [{'start': 1, 'end': 10}])
        self.assertEqual(select_regions(row, 'rejected_command'), [{'start': 11, 'end': 12}])
        self.assertEqual(select_regions(row, 'runtime_dependency'), [{'start': 11, 'end': 12}])

    def test_executable_but_wrong_retained(self):
        row = case()
        row['observed_actions'][11] = 'Extrude(profile_id=plate, depth=1, op=add)'
        self.assertEqual(select_regions(row, 'rejected_command'), [{'start': 12, 'end': 13}])
        self.assertEqual(select_regions(row, 'topology'), [{'start': 1, 'end': 10}])

    def test_scope_enforced_not_just_prompted(self):
        row = case()
        patch = json.dumps({'edits': [{'start': 8, 'end': 8, 'replacement': ['StartLoop(kind=inner)',
            'AddCircle(center=(5,5), radius=1)', 'EndLoop']}]})
        self.assertRegex(patch, re.compile(syntax_grammar()))
        with self.assertRaisesRegex(ValueError, 'outside_localized'):
            apply_lm_patch(row['observed_actions'], patch, select_regions(row, 'rejected_command'))
        lines, _ = apply_lm_patch(row['observed_actions'], patch, select_regions(row, 'topology'))
        self.assertEqual(len(lines), 16)

    def test_common_grammar_abstention_and_budget(self):
        self.assertIsNotNone(re.fullmatch(syntax_grammar(), '{"edits":[]}'))
        edit = {'start': 1, 'end': 1, 'replacement': ['End']}
        with self.assertRaises(ValueError):
            apply_lm_patch(case()['observed_actions'], json.dumps({'edits': [edit] * 3}),
                           select_regions(case(), 'full_history'))


if __name__ == '__main__':
    unittest.main()
