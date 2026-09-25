"""Standalone serialization grammar; no geometry selection on the inference host."""
import json
import re


def request_grammar(request):
    ws = r'[ \n\t\r]*'
    ids = '(?:'+'|'.join(re.escape(json.dumps(p['profile_id'])) for p in request['profiles'])+')'
    nonempty = (ws+','+ws).join([ids]*request['keep_count'])
    return ws+r'\{'+ws+'"keep_profile_ids"'+ws+':'+ws+r'\['+ws+'(?:'+nonempty+'|)'+ws+r'\]'+ws+r'\}'+ws
