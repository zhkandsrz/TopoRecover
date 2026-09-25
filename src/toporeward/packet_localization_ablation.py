"""Experimental region selection around the unchanged production packet patcher.

Run in one process per case/arm: the temporary scope binding is not thread-safe.
Validation, non-profile handlers and the final fallback are never rebound.
"""
from contextlib import contextmanager
from unittest.mock import patch

from . import stage_a_multi_transaction as transactions
from .matched_localization import select_regions
from .strong_repair_baselines import runtime_trace


SELECTORS = ('rejected_command', 'active_block', 'runtime_dependency', 'topology')


def selected_scopes(row, selector, original):
    scopes = original(row)
    if selector == 'topology':
        return scopes
    if selector not in SELECTORS:
        raise ValueError('unknown_packet_region_selector')
    # A runtime-only selector has no trigger on executable-but-wrong programs.
    # Do not invent a rejection at End, as the earlier generic diagnostic did.
    if runtime_trace(row['observed_actions']).first_rejected_step is None:
        raise ValueError('selector_has_no_runtime_trigger')
    regions = select_regions(row, selector)
    selected = [s for s in scopes if any(
        s['source_start'] < r['end'] and r['start'] < s['source_end']
        for r in regions)]
    if not selected:
        raise ValueError('selector_reaches_no_committed_profile')
    return selected


@contextmanager
def region_selection(selector, audit):
    original = transactions.transaction_scopes

    def select(row):
        try:
            scopes = selected_scopes(row, selector, original)
        except ValueError as error:
            audit.append(dict(case_id=row['case_id'], selector=selector,
                              status=str(error), profiles=[]))
            raise
        audit.append(dict(case_id=row['case_id'], selector=selector, status='selected',
                          profiles=[s['profile_id'] for s in scopes],
                          intervals=[[s['source_start'], s['source_end']] for s in scopes]))
        return scopes

    with patch.object(transactions, 'transaction_scopes', select):
        yield
