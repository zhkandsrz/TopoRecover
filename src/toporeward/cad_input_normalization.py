"""Lossless spelling adapter; not geometry repair or execution of input code."""
import ast
import re

from .actions import Extrude, RegisterProfile
from .lm.parsing import parse_action_line


def normalize_history(lines):
    output, audit = [], []
    for index, raw in enumerate(lines):
        line = raw.strip()
        if parse_action_line(line) is not None:
            output.append(line)
            continue
        try:
            call = ast.parse(line, mode='eval').body
        except SyntaxError as error:
            raise ValueError(f'unparseable_action:{index}') from error
        if (not line.isascii() or not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id not in ('RegisterProfile', 'Extrude') or call.args):
            raise ValueError(f'unsupported_action_spelling:{index}')
        values = [k.value for k in call.keywords if k.arg == 'profile_id']
        if (len(values) != 1 or not isinstance(values[0], ast.Constant) or type(values[0].value) is not str
                or re.fullmatch(r'[A-Za-z0-9_\-]+', values[0].value) is None):
            raise ValueError(f'unsupported_profile_identifier:{index}')
        value = values[0]
        normalized = line[:value.col_offset] + value.value + line[value.end_col_offset:]
        action = parse_action_line(normalized)
        if not isinstance(action, (RegisterProfile, Extrude)) or action.profile_id != value.value:
            raise ValueError(f'unparseable_action:{index}')
        output.append(normalized)
        audit.append({'action_index': index, 'kind': 'quoted_profile_id', 'before': line, 'after': normalized})
    if not output:
        raise ValueError('empty_history')
    return output, audit
