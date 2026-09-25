"""Observed scope -> LLM evidence -> LLM local patch -> protected Stage B."""
from copy import deepcopy
import json

from .stage_a_clearance_policy import prepare_clearance,submit_clearance
from .stage_a_intent_grounding import prepare_grounding,bind_evidence,public_packet
from .stage_a_profile_packets import prepare_profile_packets
from .stage_a_multi_transaction import committed_spans
from .stage_a_packet_policy import PUBLIC_SOURCE_FIELDS


def _full_packet(state,packet):
    source=state['working_source']
    fresh=next(p for p in prepare_profile_packets(source) if p['profile_id']==packet['profile_id'])
    if fresh['scopes']!=packet['scopes']: raise ValueError('stale_intent_scope')
    header,payload=fresh['prompt'].split('\nINPUT:\n',1)
    public=json.loads(payload)
    order=[p.profile_id for p in committed_spans(source['observed_actions'])]
    public.update(retained_profiles_in_history_order=order,
                  current_profile_position_one_based=order.index(packet['profile_id'])+1)
    old=public_packet(packet)
    if 'last_rejection' in old: public['last_rejection']=old['last_rejection']
    q=deepcopy(packet);q['prompt']=header+'\nINPUT:\n'+json.dumps(public,sort_keys=True)
    return q


def _open_intent(state):
    if state['phase']!='inner_first' or state.get('_intent_grounded'):
        return state
    packets={}
    for q in state['requests']:
        s=q['scopes'][0]
        if s['add_count']==0 and 0<s['remove_count']<len(s['inner_loops']):
            packets[q['source_uid']]=_full_packet(state,q)
    if not packets: return {**state,'_intent_grounded':True}
    return {**state,'phase':'intent_selection','_intent_base':deepcopy(state),
            '_intent_packets':packets,'requests':[prepare_grounding(q) for q in packets.values()]}


def _finish_counts(state):
    if state['phase']=='complete':
        state['result'].update(llm_calls=state['_actual_calls'],calls=state['_actual_calls'],
            output_tokens=state['_actual_tokens'],intent_audit=deepcopy(state.get('intent_audit',[])),
            intent_grounding_calls=state.get('_intent_calls',0),
            intent_evidence_used_for_private_selection=False)
    return state


def prepare_intent_policy(row):
    state=prepare_clearance(row)
    state.update(_actual_calls=0,_actual_tokens=0,_intent_calls=0,intent_audit=[])
    return _finish_counts(_open_intent(state))


def submit_intent_policy(row,state,raw):
    if state['phase']=='complete': raise ValueError('intent_session_complete')
    if any(row[k]!=state['source'][k] for k in PUBLIC_SOURCE_FIELDS):
        raise ValueError('stale_intent_source')
    ids=[r['source_uid'] for r in raw]
    if len(ids)!=len(set(ids)) or set(ids)!={q['source_uid'] for q in state['requests']}:
        raise ValueError('intent_response_coverage')
    state=deepcopy(state)
    calls=state['_actual_calls']+len(raw)
    tokens=state['_actual_tokens']+sum(r.get('output_tokens',0) for r in raw)
    if state['phase']=='intent_selection':
        base=state['_intent_base']; byid={r['source_uid']:r for r in raw}
        changed={}; audit=[]; evidence={}
        for request in state['requests']:
            packet=state['_intent_packets'][request['parent_source_uid']]
            response=byid[request['source_uid']]['response']
            try:
                bound,record=bind_evidence(packet,request,response)
                changed[packet['source_uid']]=bound
                evidence[packet['profile_id']]=record['selected_evidence']
                audit.append({'profile_id':packet['profile_id'],'source_uid':request['source_uid'],**record})
            except (ValueError,TypeError,KeyError) as error:
                audit.append({'profile_id':packet['profile_id'],'raw_response':response,'error':str(error)})
        base.update(_actual_calls=calls,_actual_tokens=tokens,
                    _intent_calls=state['_intent_calls']+len(raw),intent_audit=audit,
                    _intent_evidence=evidence,_intent_grounded=True)
        if len(changed)!=len(state['requests']):
            base.update(phase='complete',requests=[],result={
                'case_id':row['case_id'],'failure':'invalid_intent_evidence',
                'topology_success':False,'stage_a_success':False,'final_actions':[],
                'action_lcs':0.,'fallback_used':False,'route':'llm_intent_grounding'})
            return _finish_counts(base)
        base['requests']=[changed.get(q['source_uid'],q) for q in base['requests']]
        base['inner_session']['requests']=deepcopy(base['requests'])
        return base
    state.update(_actual_calls=calls,_actual_tokens=tokens)
    # The legacy feedback builder expects a brief. Its host-side view receives
    # only already selected evidence, never the unfiltered original brief.
    # Actual model requests remain in the frozen inference shards.
    for key in ('requests','feedback_requests'):
        for q in state.get('inner_session',{}).get(key,[]):
            public=public_packet(q)
            if 'selected_design_brief_evidence' in public and 'design_brief' not in public:
                header=q['prompt'].split('\nINPUT:\n',1)[0]
                public['design_brief']=' '.join(public['selected_design_brief_evidence'])
                q['prompt']=header+'\nINPUT:\n'+json.dumps(public,sort_keys=True)
    updated=submit_clearance(row,state,raw)
    if updated['phase']=='inner_feedback' and updated.get('_intent_evidence'):
        rebound=[]
        for q in updated['requests']:
            if q['profile_id'] in updated['_intent_evidence']:
                full=_full_packet(updated,q); request=prepare_grounding(full)
                bytext={s['text']:s['sentence_id'] for s in request['sentences']}
                chosen=[bytext[t] for t in updated['_intent_evidence'][q['profile_id']]]
                q,_=bind_evidence(full,request,json.dumps({'sentence_ids':chosen}))
            rebound.append(q)
        updated['requests']=rebound
        updated['inner_session']['feedback_requests']=deepcopy(rebound)
    return _finish_counts(_open_intent(updated))
