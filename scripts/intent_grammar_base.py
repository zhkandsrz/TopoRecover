"""Grammar for evidence IDs; every available sentence remains selectable."""
import re


def request_grammar(request):
    if not request.get('intent_grounding'):
        from clearance_grammar_source import request_grammar as clearance
        return clearance(request)
    ws=r'[ \n\t\r]*'
    choice='(?:'+'|'.join(re.escape(str(i)) for i in range(len(request['sentences'])))+')'
    items='(?:'+choice+'(?:'+ws+','+ws+choice+'){0,3})?'
    return ws+r'\{'+ws+r'"sentence_ids"'+ws+':'+ws+r'\['+ws+items+ws+r'\]'+ws+r'\}'+ws
