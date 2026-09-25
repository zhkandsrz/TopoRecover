"""Native JSON boundary audit for supported reference-derived plans."""

def ordered(value):
    return sorted(value.items(), key=lambda item: int(item[0].rsplit("_", 1)[1]))


def native_boundary_audit(native):
    import cadquery as cq
    profiles, features, provenance = [], [], []
    for pi, (part_key, part) in enumerate(ordered(native["parts"])):
        op = part["extrusion"]["operation"]
        if pi > 0 and op == "NewBodyFeatureOperation":
            raise ValueError("multiple_new_bodies_unsupported")
        mapping = {"NewBodyFeatureOperation": "add", "JoinFeatureOperation": "add",
                   "CutFeatureOperation": "cut", "IntersectFeatureOperation": "intersect"}
        if op not in mapping:
            raise ValueError("unsupported_native_operation")
        for fi, (face_key, face) in enumerate(ordered(part["sketch"])):
            wires, solids = [], []
            for loop_key, loop in ordered(face):
                edges = []
                for curve_key, curve in loop.items():
                    def point(key):
                        xy = curve[key]
                        return cq.Vector(float(xy[0]), float(xy[1]), 0)
                    if curve_key.startswith("line_"):
                        edge = cq.Edge.makeLine(point("Start Point"), point("End Point"))
                    elif curve_key.startswith("arc_"):
                        edge = cq.Edge.makeThreePointArc(point("Start Point"), point("Mid Point"), point("End Point"))
                    elif curve_key.startswith("circle_"):
                        edge = cq.Edge.makeCircle(float(curve["Radius"]), point("Center"))
                    else:
                        raise ValueError("unsupported_native_curve")
                    edges.append(edge)
                wire = cq.Wire.assembleEdges(edges)
                if not wire.IsClosed() or not wire.isValid():
                    raise ValueError("native_wire_not_closed_or_invalid")
                solid = cq.Solid.extrudeLinear(wire, [], cq.Vector(0, 0, 1))
                if not solid.isValid() or solid.Volume() <= 1e-12:
                    raise ValueError("native_boundary_nonpositive_or_invalid")
                wires.append(wire)
                solids.append(solid)
            if not solids:
                raise ValueError("empty_native_profile")
            outer = max(range(len(solids)), key=lambda j: solids[j].Volume())
            if outer != 0:
                raise ValueError("first_loop_not_geometric_outer_boundary")
            for j in range(1, len(solids)):
                if solids[j].cut(solids[0]).Volume() > max(1e-10, solids[j].Volume() * 1e-6):
                    raise ValueError("inner_boundary_not_contained")
                if wires[j].distance(wires[0]) <= 1e-8:
                    raise ValueError("inner_boundary_touches_outer")
                for k in range(1, j):
                    if solids[j].intersect(solids[k]).Volume() > 1e-10 or wires[j].distance(wires[k]) <= 1e-8:
                        raise ValueError("inner_boundaries_overlap_or_touch")
            combined = cq.Face.makeFromWires(wires[0], wires[1:])
            if not combined.isValid():
                raise ValueError("native_profile_face_invalid")
            profile_id = f"profile_{pi}_{fi}"
            profiles.append({"profile_id": profile_id, "loop_roles": ["outer"] + ["inner"] * (len(wires) - 1)})
            features.append({"profile_id": profile_id, "operation": mapping[op]})
            provenance.append({"profile_id": profile_id, "native_pointer": f"/parts/{part_key}/sketch/{face_key}",
                               "boundary_count": len(wires), "boundary_areas": [s.Volume() for s in solids],
                               "native_operation": op, "containment_checked": True})
    profiles.sort(key=lambda x: x["profile_id"])
    rename = {p["profile_id"]: f"p{i}" for i, p in enumerate(profiles)}
    plan = {"profiles": [{"profile_id": rename[p["profile_id"]], "loop_roles": p["loop_roles"]} for p in profiles],
            "features": [{"feature_id": f"f{i}", "type": "extrude", "profile_id": rename[f["profile_id"]], "operation": f["operation"]} for i, f in enumerate(features)]}
    return plan, provenance
