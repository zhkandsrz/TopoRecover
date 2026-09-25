"""Boundary-mesh topology evaluator used in the paper."""
import numpy as np
import trimesh

PROTOCOL = {
    "metric": "boundary-mesh Euler chi=V-E+F; absolute reference error and exact equality",
    "normalization": "center each final shape and scale bounding-box diagonal to one",
    "meshing": [{"linear": 0.001, "angular": 0.1}, {"linear": 0.0005, "angular": 0.05}],
    "seam_merge_decimal_places": 8,
    "mesh_processing": "merge seam vertices; remove unreferenced vertices only; no repair",
    "eligibility": "positive-volume valid OCC shape; both meshes watertight with equal Euler",
    "conditional": "intersection of reference and all six method eligibility sets",
    "failure_aware": "all eligible references; missing/nonwatertight/unstable outputs count incorrect",
    "reference_boundary": "reference used for evaluation only, not repair or selection",
    "trimesh_version": trimesh.__version__,
}


def mesh_record(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices(digits_vertex=PROTOCOL["seam_merge_decimal_places"])
    mesh.remove_unreferenced_vertices()
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    degenerate = int(np.count_nonzero(~mesh.nondegenerate_faces(height=1e-12)))
    watertight = bool(len(mesh.faces) and mesh.is_watertight and not degenerate)
    return {
        "vertices": len(mesh.vertices), "edges": len(edges), "faces": len(mesh.faces),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "degenerate_triangles": degenerate, "watertight": watertight,
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_components": int(mesh.body_count),
        "euler": int(mesh.euler_number) if watertight else None,
    }


def shape_record(shape):
    bbox = shape.BoundingBox()
    diagonal = float(np.linalg.norm([bbox.xlen, bbox.ylen, bbox.zlen]))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Nonpositive shape diagonal")
    center = tuple(-(getattr(bbox, axis + "min") + getattr(bbox, axis + "max")) / 2 for axis in "xyz")
    normalized = shape.translate(center).scale(1.0 / diagonal)
    meshes = []
    for setting in PROTOCOL["meshing"]:
        vertices, triangles = normalized.tessellate(setting["linear"], setting["angular"])
        meshes.append(mesh_record([v.toTuple() for v in vertices], triangles))
    eligible = all(m["watertight"] for m in meshes) and meshes[0]["euler"] == meshes[1]["euler"]
    return {"meshes": meshes, "eligible": eligible,
            "euler": meshes[0]["euler"] if eligible else None,
            "failure": None if eligible else "nonwatertight_degenerate_or_resolution_unstable"}
