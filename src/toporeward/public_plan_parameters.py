"""Bind missing feature parameters from an explicit public JSON specification.

This does not parse unrestricted natural language, infer dimensions from a
reference solid, change supplied plan values, or generate a repair itself.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
from math import isclose, isfinite

from .stage_a_packet_policy import PUBLIC_SOURCE_FIELDS

UNIT_METRES = {'mm':'0.001', 'cm':'0.01', 'm':'1', 'inch':'0.0254'}


def explicit_specification(text):
    def unique_fields(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate_public_specification_field')
            result[key] = value
        return result
    decoder = json.JSONDecoder(object_pairs_hook=unique_fields)
    documents = []
    offset = 0
    while offset < len(text):
        start = text.find('{',offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start+1
            continue
        offset = start+end
        if isinstance(value,dict) and isinstance(value.get('profiles'),list) and isinstance(value.get('features'),list):
            documents.append(value)
    if len(documents) > 1:
        raise ValueError('ambiguous_public_specification')
    return documents[0] if documents else None


def bind_public_plan_parameters(row):
    public = {k:deepcopy(row[k]) for k in PUBLIC_SOURCE_FIELDS}
    document = explicit_specification(public['design_brief'])
    if document is None:
        return public, []
    plan = public['feature_plan'].get('feature_plan',public['feature_plan'])
    feature_rows = document['features']
    identities = [f.get('feature_id') for f in feature_rows]
    if any(not isinstance(i,str) or not i for i in identities) or len(set(identities)) != len(identities):
        raise ValueError('public_specification_requires_unique_feature_ids')
    by_id = {f['feature_id']:(i,f) for i,f in enumerate(feature_rows)}
    supplied = plan.get('features',[])
    plan_ids = [f.get('feature_id') for f in supplied]
    # No positional fallback: a missing name is not evidence of correspondence.
    if any(not isinstance(i,str) or not i for i in plan_ids):
        return public, []
    if len(set(plan_ids)) != len(plan_ids) or set(plan_ids) != set(identities):
        raise ValueError('public_specification_feature_identity_conflict')
    source_unit, target_unit = document.get('length_unit'), plan.get('length_unit')
    scale = Decimal(1)
    if source_unit is not None or target_unit is not None:
        if source_unit not in UNIT_METRES or target_unit not in UNIT_METRES:
            raise ValueError('public_specification_units_unresolved')
        scale = Decimal(UNIT_METRES[source_unit])/Decimal(UNIT_METRES[target_unit])
    audit = []
    for feature in supplied:
        index, evidence = by_id[feature['feature_id']]
        if any(feature.get(k) != evidence.get(k) for k in ('type','profile_id','operation')):
            raise ValueError('public_specification_feature_reference_conflict')
        if evidence.get('depth') is None:
            continue
        if isinstance(evidence['depth'],bool):
            raise ValueError('invalid_public_specification_depth')
        try:
            depth = float(Decimal(str(evidence['depth']))*scale)
        except (InvalidOperation,OverflowError) as error:
            raise ValueError('invalid_public_specification_depth') from error
        if not isfinite(depth) or depth <= 0:
            raise ValueError('invalid_public_specification_depth')
        if feature.get('depth') is not None:
            if not isclose(float(feature['depth']),depth,rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError('public_specification_depth_conflict')
            continue
        feature['depth'] = depth
        audit.append(dict(feature_id=feature['feature_id'], profile_id=feature['profile_id'],
                          operation=feature['operation'], source='public_design_brief_json',
                          json_pointer=f'/features/{index}/depth', source_depth=evidence['depth'],
                          source_unit=source_unit, plan_unit=target_unit, depth=depth,
                          private_reference_visible=False))
    return public, audit
