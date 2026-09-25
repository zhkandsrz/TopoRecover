"""Extends the frozen packet grammar with bounded terminal-deletion indices."""
import re


def request_grammar(request):
    if request.get('structure_repair'):
        ws = r'\s*'
        positions = '(?:'+'|'.join(re.escape(str(i)) for i in request['insertion_positions'])+')'
        actions = '|'.join(re.escape(s) for s in ('StartSketch','StartFace','StartLoop(kind=outer)',
                   'StartLoop(kind=inner)','EndLoop','EndFace','EndSketch','End'))
        if request.get('structure_edit_operations'):
            insert = rf'\{{{ws}"op"{ws}:{ws}"insert"{ws},{ws}"before_step"{ws}:{ws}{positions}{ws},{ws}"action"{ws}:{ws}"(?:{actions})"{ws}\}}'
            variants = [insert]
            if request['movable_steps']:
                movable = '(?:'+'|'.join(re.escape(str(i)) for i in request['movable_steps'])+')'
                variants.append(rf'\{{{ws}"op"{ws}:{ws}"move"{ws},{ws}"step"{ws}:{ws}{movable}{ws},{ws}"before_step"{ws}:{ws}{positions}{ws}\}}')
            if request['deletable_steps']:
                deletable = '(?:'+'|'.join(re.escape(str(i)) for i in request['deletable_steps'])+')'
                variants.append(rf'\{{{ws}"op"{ws}:{ws}"delete"{ws},{ws}"start_step"{ws}:{ws}{deletable}{ws},{ws}"end_step"{ws}:{ws}{positions}{ws}\}}')
            operation = '(?:'+'|'.join(variants)+')'
            # The host enforces the expanded 32-command budget and protected ranges.
            operations = rf'(?:{operation}(?:{ws},{ws}{operation})*)?'
            return rf'\{{{ws}"edits"{ws}:{ws}\[{ws}{operations}{ws}\]{ws}\}}'
        item = rf'\{{{ws}"before_step"{ws}:{ws}{positions}{ws},{ws}"action"{ws}:{ws}"(?:{actions})"{ws}\}}'
        insertions = rf'(?:{item}(?:{ws},{ws}{item}){{0,31}})?'
        removable = request['deletable_steps']
        if removable:
            ids = '(?:'+'|'.join(re.escape(str(i)) for i in removable)+')'
            removals = rf'(?:{ids}(?:{ws},{ws}{ids}){{0,31}})?'
        else:
            removals = ''
        prefix = ''
        if request.get('structure_diagnosis'):
            diagnosis = r'"[A-Za-z0-9 .,;:()=+*/<>_%!?\[\]-]{1,400}"'
            prefix = rf'"diagnosis"{ws}:{ws}{diagnosis}{ws},{ws}'
        return rf'\{{{ws}{prefix}"insertions"{ws}:{ws}\[{ws}{insertions}{ws}\]{ws},{ws}"remove_steps"{ws}:{ws}\[{ws}{removals}{ws}\]{ws}\}}'
    if not request.get('terminal_boundary'):
        from intent_grammar_base import request_grammar as base
        return base(request)
    ids = '(?:'+'|'.join(re.escape(str(i)) for i in request['candidate_terminal_steps'])+')'
    ws = r'\s*'
    body = rf'(?:{ids}(?:{ws},{ws}{ids}){{0,7}})?'
    return rf'\{{{ws}"remove_terminal_steps"{ws}:{ws}\[{ws}{body}{ws}\]{ws}\}}'
