"""Preserve accepted profile transactions, not unrelated earlier executions.

Stage A geometry remains immutable. Stage B may correct public-contract extrusion
occurrences and close incomplete structure, including redundant closing markers.
It cannot synthesize a profile, hole, or unspecified extrusion depth.
"""
from math import isfinite

from .actions import (AddLine, End, EndFace, EndLoop, EndSketch, Extrude,
                      RegisterProfile, action_to_text)
from .lm.parsing import parse_action_line
from .natural_repair_recall import evaluate_repaired_history
from .stage_a_multi_transaction import committed_spans
from .stage_b_preserving_geometry import PreservedCompletion
from .verifier import TopoVerifier


def exact_action_text(action):
    if isinstance(action, Extrude):
        return f'Extrude(profile_id={action.profile_id}, depth={action.depth!r}, op={action.op})'
    return action_to_text(action)


def unique_feature_alignment(observed, expected):
    """Monotone matching with missing planned features, never observed deletions.

    Cost is one per missing feature or changed explicit field. Tied mappings
    abstain; coordinates and private geometry do not participate in alignment.
    """
    previous = [(j, 1, ()) for j in range(len(expected)+1)]
    for action in observed:
        current = [(float('inf'), 0, ())]
        for j, (identity, depth, operation) in enumerate(expected, 1):
            cost = int(action.profile_id != identity) + int(action.op != operation)
            if depth is not None:
                cost += int(action.depth != depth)
            match = previous[j-1]
            skip = current[j-1]
            choices = [(match[0]+cost, match[1], match[2]+(j-1,)),
                       (skip[0]+1, skip[1], skip[2])]
            best = min(c[0] for c in choices)
            tied = [c for c in choices if c[0] == best and c[1]]
            current.append((best, min(2,sum(c[1] for c in tied)), tied[0][2] if tied else ()))
        previous = current
    return previous[-1][2] if previous[-1][1] == 1 else None


def complete_transaction_preserving(lines, contract, *, protected_end, max_edits=8,
                                    align_missing_features=False):
    """protected_end bounds accepted profile blocks, not all execution actions."""
    original = list(lines)
    if not 0 <= protected_end <= len(original):
        raise ValueError('invalid_protected_end')
    parsed = [parse_action_line(s) for s in original]
    if any(a is None for a in parsed):
        return PreservedCompletion((), 'unparseable_observed_history')
    protected = [s for s in committed_spans(original) if s.register_end <= protected_end]
    protected_indices = {i for s in protected for i in range(s.face_start, s.register_end)}
    starts = {s.face_start for s in protected}
    graph, required_profiles = contract.get('extrusion_graph', []), contract.get('profiles', [])
    registrations = [a.profile_id for a in parsed if isinstance(a, RegisterProfile)]
    if len(set(registrations)) != len(registrations):
        return PreservedCompletion((), 'ambiguous_profile_identity')
    extrudes = [(i, a) for i, a in enumerate(parsed) if isinstance(a, Extrude)]
    identities = sorted(registrations)
    if len(identities) < len(required_profiles):
        referenced = {a.profile_id for _, a in extrudes} | set(identities)
        if len(referenced) == len(required_profiles):
            identities = sorted(referenced)
    working, origins, evidence = list(original), list(range(len(original))), []

    def stop(reason):
        return PreservedCompletion((), reason, tuple(evidence))

    def profile_for(spec):
        index = spec.get('profile_index')
        if len(identities) == len(required_profiles) and type(index) is int and 0 <= index < len(identities):
            return identities[index]
        return None

    def touches_profile(position):
        origin = next((i for i in origins[position:] if i is not None), None)
        return origin in protected_indices and origin not in starts

    def protected_blocks_unchanged():
        positions = {origin: i for i, origin in enumerate(origins) if origin is not None}
        for span in protected:
            old_indices = list(range(span.face_start, span.register_end))
            if any(i not in positions for i in old_indices):
                return False
            locations = [positions[i] for i in old_indices]
            if locations != list(range(locations[0], locations[0]+len(old_indices))):
                return False
            if working[locations[0]:locations[-1]+1] != original[span.face_start:span.register_end]:
                return False
        return True

    if align_missing_features and 0 < len(extrudes) < len(graph):
        expected = [(profile_for(s), float(s['depth']) if 'depth' in s else None, s.get('operation'))
                    for s in graph]
        if any(pid is None for pid,_,_ in expected):
            return stop('unresolved_profile_reference')
        if any(depth is not None and (not isfinite(depth) or depth <= 0) for _,depth,_ in expected):
            return stop('invalid_supported_depth')
        if any(op not in ('add','cut','intersect') for _,_,op in expected):
            return stop('invalid_required_operation')
        mapping = unique_feature_alignment([a for _,a in extrudes],expected)
        if mapping is None:
            return stop('ambiguous_feature_sequence_alignment')
        matched = {ordinal:(index,action) for ordinal,(index,action) in zip(mapping,extrudes)}
        additions = {}
        for ordinal,(identity,depth,operation) in enumerate(expected):
            if ordinal in matched:
                index,action = matched[ordinal]
                fixed = Extrude(identity,action.depth if depth is None else depth,operation)
                if fixed == action:
                    continue
                if index in protected_indices:
                    return stop('would_change_accepted_profile')
                working[index] = exact_action_text(fixed)
                evidence.append(dict(kind='aligned_extrusion_occurrence',action_index=index,
                                     operation_index=ordinal,source='public_feature_sequence'))
            else:
                if depth is None:
                    return stop('missing_or_ambiguous_extrusion_depth')
                following = [matched[j][0] for j in mapping if j > ordinal]
                index = following[0] if following else (len(original)-1 if parsed and isinstance(parsed[-1],End) else len(original))
                if index in protected_indices and index not in starts:
                    return stop('would_change_accepted_profile')
                fixed = Extrude(identity,depth,operation)
                additions.setdefault(index,[]).append(exact_action_text(fixed))
                evidence.append(dict(kind='missing_aligned_extrusion',action_index=index,
                                     operation_index=ordinal,source='explicit_public_feature_sequence'))
            if len(evidence) > max_edits:
                return stop('local_completion_budget_exceeded')
        expanded, expanded_origins = [], []
        for index in range(len(working)+1):
            insertions = additions.get(index,[])
            expanded.extend(insertions)
            expanded_origins.extend([None]*len(insertions))
            if index < len(working):
                expanded.append(working[index])
                expanded_origins.append(index)
        working, origins = expanded, expanded_origins

    if len(extrudes) == len(graph):
        for ordinal, ((index, action), spec) in enumerate(zip(extrudes, graph)):
            identity = profile_for(spec)
            if identity is None:
                continue
            depth, operation = float(spec.get('depth', action.depth)), spec.get('operation')
            if not isfinite(depth) or depth <= 0:
                return stop('invalid_supported_depth')
            if operation not in ('add', 'cut', 'intersect'):
                return stop('invalid_required_operation')
            fixed = Extrude(identity, depth, operation)
            if fixed != action:
                if index in protected_indices:
                    return stop('would_change_accepted_profile')
                if len(evidence) >= max_edits:
                    return stop('local_completion_budget_exceeded')
                working[index] = exact_action_text(fixed)
                evidence.append(dict(kind='extrusion_occurrence', action_index=index,
                                     operation_index=ordinal, depth=depth,
                                     depth_source='explicit_contract' if 'depth' in spec else 'observed_occurrence'))

    verifier = TopoVerifier()
    for _ in range(max_edits+1):
        state, position, rejected = verifier.initial_state(), len(working), False
        for index, line in enumerate(working):
            step = verifier.step(state, parse_action_line(line))
            if not step.valid:
                position, rejected = index, True
                break
            state = step.next_state
        else:
            outcome = evaluate_repaired_history(working, contract)
            if outcome['valid_and_intent_satisfied']:
                if any('depth' in spec and float(spec['depth']) != operation['depth']
                       for spec, operation in zip(graph, state.extrusions)):
                    return stop('explicit_parameter_contract_unresolved')
                if not protected_blocks_unchanged():
                    return stop('stage_b_changed_accepted_profile')
                return PreservedCompletion(tuple(working), 'transaction_preserving_completion', tuple(evidence))
            if state.ended and working and isinstance(parse_action_line(working[-1]), End):
                position, state = len(working)-1, verifier.initial_state()
                for line in working[:position]:
                    state = verifier.step(state, parse_action_line(line)).next_state
        if len(evidence) >= max_edits:
            return stop('local_completion_budget_exceeded')

        # Only adjacent, identical closing markers can be removed; rejected
        # geometry and nonredundant operations are never silently discarded.
        if rejected and position > 0:
            current = parse_action_line(working[position])
            previous = parse_action_line(working[position-1])
            if isinstance(current, (EndLoop, EndFace, EndSketch, End)) and current == previous:
                if origins[position] in protected_indices:
                    return stop('would_change_accepted_profile')
                evidence.append(dict(kind='redundant_closing_marker', action_index=position,
                                     deleted_action=working[position]))
                del working[position], origins[position]
                continue
        if touches_profile(position):
            return stop('rejection_inside_accepted_profile')
        for identity, face in state.profiles.items():
            if identity not in identities:
                return stop('unresolved_profile_reference')
            ordinal = identities.index(identity)
            if ordinal >= len(required_profiles) or sorted(loop.kind for loop in face.loops) != sorted(required_profiles[ordinal]['loop_roles']):
                return stop('committed_topology_unresolved')
        insertion, kind = None, 'structural_completion'
        top = state.stack[-1] if state.stack else None
        if top == 'loop':
            if verifier.step(state, EndLoop()).valid:
                insertion = EndLoop()
            elif (state.current_loop is not None and len(state.current_loop.segments) >= 2
                  and state.current_loop.tail != state.current_loop.start):
                insertion = AddLine(state.current_loop.tail, state.current_loop.start)
                kind = 'existing_endpoint_closure'
        elif top == 'face':
            insertion = EndFace()
        elif state.pending_face is not None:
            available = [p for p in identities if p not in state.profiles and p not in registrations]
            if len(available) == 1:
                insertion, kind = RegisterProfile(available[0]), 'unique_observed_profile_reference'
        elif top == 'sketch':
            insertion = EndSketch()
        elif len(state.extrusions) < len(graph):
            spec = graph[len(state.extrusions)]
            identity = profile_for(spec)
            if identity is None or identity not in state.profiles:
                return stop('missing_profile_geometry')
            if spec.get('depth') is None:
                return stop('missing_or_ambiguous_extrusion_depth')
            depth = float(spec['depth'])
            if not isfinite(depth) or depth <= 0:
                return stop('invalid_supported_depth')
            insertion, kind = Extrude(identity, depth, spec['operation']), 'explicit_contract_depth'
        elif not state.ended:
            insertion = End()
        if insertion is None or not verifier.step(state, insertion).valid:
            return stop('no_evidence_supported_local_completion')
        if position < len(working) and parse_action_line(working[position]) == insertion:
            return stop('local_completion_no_progress')
        working.insert(position, exact_action_text(insertion))
        origins.insert(position, None)
        evidence.append(dict(kind=kind, action_index=position, inserted_action=exact_action_text(insertion)))
    return stop('local_completion_budget_exceeded')
