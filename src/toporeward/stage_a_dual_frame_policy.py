"""LLM-authored world/local parameters and bounded verifier rejection feedback."""
from copy import deepcopy
import json

from .actions import AddArc, AddCircle, AddLine, StartLoop, EndLoop
from .llm_stage_a_local_frame import observed_rectangle_frame, world_parameters
from .lm.parsing import parse_action_line
from .stage_a_parameter_policy import prepare_parameter_request, evaluate_parameter_response
from .strong_repair_baselines import runtime_trace


def normalized_loops(lines, frame):
    def point(p):
        return [(p[0]-frame['origin'][0])/frame['width'],
                (p[1]-frame['origin'][1])/frame['height']]
    loops=[];current=None
    for line in lines:
        a=parse_action_line(line)
        if isinstance(a,StartLoop):
            current={'kind':a.kind,'curves':[]}
        elif isinstance(a,EndLoop) and current is not None:
            loops.append(current);current=None
        elif current is not None:
            if isinstance(a,AddCircle):
                curve={'type':'circle','center_fraction':point(a.center),
                       'radius_over_width':a.radius/frame['width'],
                       'radius_over_height':a.radius/frame['height']}
            elif isinstance(a,(AddLine,AddArc)):
                curve={'type':a.type,'start':point(a.start),'end':point(a.end)}
                if isinstance(a,AddArc):curve['mid']=point(a.mid)
            else:
                continue
            current['curves'].append(curve)
    return loops


def prepare_dual_frame_request(row):
    q=prepare_parameter_request(row)
    try:
        frame=observed_rectangle_frame(row)
    except ValueError:
        frame=None
    scope=q['scope']
    lines=row['observed_actions'][scope['source_start']:scope['source_end']]
    visible={'design_brief':row['design_brief'],'feature_plan':row['feature_plan'],
        'profile_id':scope['profile_id'],'observed_roles':scope['observed_roles'],
        'required_roles':scope['required_roles'],'localized_profile':lines,
        'observed_outer_frame':frame,
        'observed_loops_in_frame':normalized_loops(lines,frame) if frame else None}
    q['prompt']=(
        'Repair the ONE missing inner loop in the verifier-localized CAD profile. '
        'Keep all existing curves and extrusion parameters unchanged. The new opening '
        'must lie strictly inside the outer loop, with positive clearance, and must not '
        'touch or overlap any existing inner loop. Do not copy the outer outline as a hole. '
        'Respect specified geometric dimensions and shape. If dimensions are unspecified, '
        'choose a feasible opening; unspecified intent cannot be inferred as ground truth. '
        'You may output WORLD coordinates, especially when explicit world dimensions are '
        'given, or PROFILE_FRACTION coordinates to avoid world-coordinate arithmetic. '
        'The latter is available only when observed_outer_frame is not null. '
        'For profile_fraction, center=[u,v] means fractions of outer width and height from '
        'the lower-left origin; width and height are fractions of outer width and height; '
        'radius is a fraction of the SHORTER outer side. Circle diameter is twice radius. '
        'Normalized observed curves describe existing geometry, not a suggested new hole. '
        'The host converts coordinates exactly; it does not shrink, clip, move, search '
        'for or replace your proposal. A rejected proposal may receive one verifier '
        'diagnosis; then return a revised JSON patch within the same localized profile. '
        'Return only one JSON object with exactly these fields: '
        '{"profile_id":"<given>","coordinate_frame":"world" or "profile_fraction",'
        '"loop":{"shape":"circle","center":[x,y],"radius":r}} '
        'or the same outer fields and '
        '"loop":{"shape":"rectangle","center":[x,y],"width":w,"height":h}. '
        'Use finite numeric JSON values, no Markdown or explanations.\nINPUT:\n'
    )+json.dumps(visible,sort_keys=True)
    q.update(attempt=0)
    q['input_audit'].update(normalized_geometry_from_observed_only=True,
        host_selects_hole_geometry=False,host_clips_or_searches=False,
        model_selects_coordinate_frame=True,max_attempts=2)
    return q


def decode_world(row,response):
    data=json.loads(response)
    if not isinstance(data,dict) or set(data)!={'profile_id','coordinate_frame','loop'}:
        raise ValueError('dual_frame_schema')
    loop=data['loop']
    fields={'circle':{'shape','center','radius'},'rectangle':{'shape','center','width','height'}}
    if not isinstance(loop,dict) or not isinstance(loop.get('shape'),str) or loop['shape'] not in fields or set(loop)!=fields[loop['shape']]:
        raise ValueError('dual_frame_loop_schema')
    if data['coordinate_frame']=='world':
        return {'profile_id':data['profile_id'],'loop':loop}
    if data['coordinate_frame']!='profile_fraction':
        raise ValueError('unknown_coordinate_frame')
    local={'shape':loop['shape'],'center_fraction':loop['center']}
    if loop['shape']=='circle':local['radius_fraction']=loop['radius']
    else:local.update(width_fraction=loop['width'],height_fraction=loop['height'])
    return world_parameters(row,json.dumps({'profile_id':data['profile_id'],'loop':local}))


def evaluate_dual_frame_response(row,q,response):
    try:
        world=decode_world(row,response)
    except (ValueError,TypeError,KeyError,AttributeError,OverflowError) as error:
        return {'case_id':row['case_id'],'mode':q['mode'],'stage_a_success':False,
            'topology_success':False,'patch_accepted':False,'final_actions':[],
            'action_lcs':0.0,'failure':str(error),'fallback_used':False}
    result=evaluate_parameter_response(row,q,json.dumps(world))
    result.update(model_parameter_response=response,model_world_parameters=world,
                  geometry_clipped_or_replaced=False,attempt=q['attempt'])
    return result


def feedback_request(row,q,response):
    if q['attempt']!=0:
        raise ValueError('feedback_budget_exhausted')
    result=evaluate_dual_frame_response(row,q,response)
    if result['stage_a_success']:
        return None
    trace=runtime_trace(result['patched_actions']).to_dict() if result.get('patched_actions') else None
    feedback={'rejected_response':response,'failure':result['failure']}
    if trace:
        feedback.update(failure_type=trace['failure_type'],repair_hint=trace['repair_hint'])
    revised=deepcopy(q)
    instructions,payload=q['prompt'].split('\nINPUT:\n',1)
    visible=json.loads(payload);visible['last_rejection']=feedback
    revised.update(attempt=1,source_uid=q['source_uid']+'::feedback',
        prompt=instructions+'\nINPUT:\n'+json.dumps(visible,sort_keys=True))
    return revised
