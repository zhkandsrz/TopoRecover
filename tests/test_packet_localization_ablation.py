import json
import unittest

from test_matched_localization import case
from toporeward.packet_localization_ablation import region_selection, selected_scopes
from toporeward.stage_a_multi_transaction import transaction_scopes
from toporeward.stage_a_profile_packets import prepare_profile_packets
from toporeward.llm_stage_a import profile_diagnostics


class PacketLocalizationTests(unittest.TestCase):
    def test_production_packet_unchanged(self):
        row = case()
        expected = prepare_profile_packets(row)
        audit = []
        with region_selection('topology', audit):
            self.assertEqual(prepare_profile_packets(row), expected)
        self.assertEqual(audit[0]['profiles'], ['plate'])

    def test_runtime_region_cannot_reach_earlier_profile(self):
        row = case()
        for selector in ('rejected_command', 'active_block', 'runtime_dependency'):
            with self.assertRaisesRegex(ValueError, 'reaches_no_committed_profile'):
                selected_scopes(row, selector, transaction_scopes)

    def test_executable_source_has_no_fictitious_runtime_error(self):
        row = case()
        row['observed_actions'][11] = 'Extrude(profile_id=plate, depth=1, op=add)'
        with self.assertRaisesRegex(ValueError, 'no_runtime_trigger'):
            selected_scopes(row, 'active_block', transaction_scopes)
        self.assertEqual(len(selected_scopes(row, 'topology', transaction_scopes)), 1)

    def test_validation_diagnostics_never_filtered_and_binding_restored(self):
        row = case()
        expected = profile_diagnostics(row['observed_actions'], row['topology_contract'])
        with region_selection('rejected_command', []):
            with self.assertRaises(ValueError):
                prepare_profile_packets(row)
            self.assertEqual(profile_diagnostics(row['observed_actions'], row['topology_contract']), expected)
        self.assertEqual(len(prepare_profile_packets(row)), 1)

    def test_dependency_can_reach_registered_profile(self):
        row = case()
        row['observed_actions'][11] = 'Extrude(profile_id=plate, depth=0, op=add)'
        selected = selected_scopes(row, 'runtime_dependency', transaction_scopes)
        self.assertEqual(selected[0]['profile_id'], 'plate')


if __name__ == '__main__':
    unittest.main()
