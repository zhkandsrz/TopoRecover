"""LLM-selected brief evidence for existing-loop retention, without target access."""
from copy import deepcopy
import hashlib
import json
import re


def public_packet(packet):
    return json.loads(packet['prompt'].split('\nINPUT:\n',1)[1])


def prepare_grounding(packet):
    public=public_packet(packet)
    scope=packet['scopes'][0]
    keep_count=len(scope['inner_loops'])-scope['remove_count']
    if scope['add_count']!=0 or keep_count<=0 or scope['remove_count']<=0:
        raise ValueError('not_retention_only_scope')
    brief=public['design_brief']
    sentences=[s for s in re.split(r'(?<=[.!?])\s+',brief.strip()) if s]
    if not sentences:
        raise ValueError('empty_design_brief')
    # IDs are rotated independently of content so a fixed sentence index is not an answer.
    shift=int(hashlib.sha256(packet['source_uid'].encode()).hexdigest(),16)%len(sentences)
    sentences=sentences[shift:]+sentences[:shift]
    evidence=[{'sentence_id':i,'text':s} for i,s in enumerate(sentences)]
    current=public.get('current_profile',{})
    context={'current_profile_id':packet['profile_id'],'design_brief':brief,
             'brief_sentences':evidence,'keep_inner_loop_count':keep_count,
             'current_profile_history':current.get('localized_profile',[])}
    for name in ('retained_profiles_in_history_order','current_profile_position_one_based'):
        if name in public: context[name]=public[name]
    prompt=(
        'Identify the original design-brief evidence for deciding which EXISTING INNER '
        'loops must be KEPT in the current CAD profile. Select only sentences describing '
        'the requested existing openings in this profile, including qualifiers needed '
        'to resolve it. Do not confuse the OUTER profile center with an INNER opening '
        'center. A sentence saying which whole profiles to preserve is not an instruction '
        'to preserve a hole at that center. Other profiles and new openings are handled '
        'separately. You do not repair geometry in this step. Return JSON only: '
        '{"sentence_ids":[0]}. Select one to four distinct available IDs, or [] if no '
        'applicable evidence exists. Every ID refers to the exact supplied sentence.\nINPUT:\n'
    )+json.dumps(context,sort_keys=True)
    return {'case_id':packet['case_id'],'source_uid':packet['source_uid']+'::intent',
            'profile_id':packet['profile_id'],'intent_grounding':True,'prompt':prompt,
            'sentences':evidence,'parent_source_uid':packet['source_uid']}


def bind_evidence(packet,request,response):
    if request['parent_source_uid']!=packet['source_uid'] or request!=prepare_grounding(packet):
        raise ValueError('stale_grounding_request')
    data=json.loads(response)
    if not isinstance(data,dict) or set(data)!={'sentence_ids'}:
        raise ValueError('intent_schema')
    ids=data['sentence_ids']
    if (not isinstance(ids,list) or not 1<=len(ids)<=4 or
        any(type(i) is not int or not 0<=i<len(request['sentences']) for i in ids) or len(set(ids))!=len(ids)):
        raise ValueError('invalid_or_abstained_evidence')
    selected=[request['sentences'][i]['text'] for i in ids]
    old=public_packet(packet)
    loops=old.get('current_profile',{}).get('inner_loops',old.get('existing_inner_loops',[]))
    scope=packet['scopes'][0]
    public={'current_profile_id':packet['profile_id'],
            'selected_design_brief_evidence':selected,
            'keep_count':len(scope['inner_loops'])-scope['remove_count'],
            'existing_inner_loops':[{'loop_id':l['loop_id'],'world_geometry':l['world_geometry']} for l in loops]}
    for name in ('retained_profiles_in_history_order','current_profile_position_one_based','last_rejection'):
        if name in old: public[name]=old[name]
    q=deepcopy(packet)
    q['prompt']=(
        'Repair the existing INNER-loop selection for this CAD profile using the supplied '
        'design-brief evidence. Match the requested existing opening by its world center, '
        'shape and dimensions. Return exactly keep_count distinct keep_loop_ids; unselected '
        'inner loops will be deleted and kept geometry must be unchanged. No geometry is '
        'added in this request. Do not treat an OUTER profile center as an INNER opening '
        'requirement. If the evidence does not specify a defensible choice, abstain. '
        'Return JSON only: {"profiles":[{"profile_id":"<current profile>",'
        '"keep_loop_ids":["<existing loop id>"],"add_loops":[]}]}, or {"profiles":[]}.\nINPUT:\n'
    )+json.dumps(public,sort_keys=True)
    return q,{'raw_grounding_response':response,'selected_sentence_ids':ids,
              'selected_evidence':selected,'host_selected_target':False}
