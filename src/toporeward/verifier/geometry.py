from __future__ import annotations

import math

Point = tuple[float, float]
Segment = tuple[Point, Point]
EPS = 1e-7


def close_points(a: Point, b: Point, eps: float = EPS) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def signed_area(points: list[Point] | tuple[Point, ...]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for a, b in zip(points, points[1:] + points[:1]):
        area += a[0] * b[1] - b[0] * a[1]
    return 0.5 * area


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def collinear(a: Point, b: Point, c: Point, eps: float = EPS) -> bool:
    return abs(orientation(a, b, c)) <= eps


def on_segment(a: Point, b: Point, c: Point, eps: float = EPS) -> bool:
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
        and collinear(a, b, c, eps)
    )


def segments_intersect(s1: Segment, s2: Segment, eps: float = EPS) -> bool:
    p1, q1 = s1
    p2, q2 = s2
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True

    if abs(o1) <= eps and on_segment(p1, p2, q1, eps):
        return True
    if abs(o2) <= eps and on_segment(p1, q2, q1, eps):
        return True
    if abs(o3) <= eps and on_segment(p2, p1, q2, eps):
        return True
    if abs(o4) <= eps and on_segment(p2, q1, q2, eps):
        return True
    return False


def _shares_only_allowed_endpoint(new_seg: Segment, old_seg: Segment, allowed_points: list[Point]) -> bool:
    points = [new_seg[0], new_seg[1]]
    old_points = [old_seg[0], old_seg[1]]
    shared = [p for p in points for q in old_points if close_points(p, q)]
    if len(shared) != 1:
        return False
    return any(close_points(shared[0], allowed) for allowed in allowed_points)


def would_self_intersect(existing: list[Segment], new_seg: Segment, closing_to_start: bool) -> bool:
    if not existing:
        return False
    allowed: list[Point] = [new_seg[0]]
    if closing_to_start:
        allowed.append(new_seg[1])

    for idx, old_seg in enumerate(existing):
        adjacent_to_tail = idx == len(existing) - 1
        adjacent_to_start = closing_to_start and idx == 0
        if not segments_intersect(old_seg, new_seg):
            continue
        if adjacent_to_tail and _shares_only_allowed_endpoint(new_seg, old_seg, [new_seg[0]]):
            continue
        if adjacent_to_start and _shares_only_allowed_endpoint(new_seg, old_seg, [new_seg[1]]):
            continue
        return True
    return False


def point_in_polygon(point: Point, polygon: list[Point] | tuple[Point, ...]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        pj = polygon[j]
        if on_segment(pj, point, pi):
            return True
        intersects = (pi[1] > y) != (pj[1] > y)
        if intersects:
            x_cross = (pj[0] - pi[0]) * (y - pi[1]) / (pj[1] - pi[1] + 0.0) + pi[0]
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def polygon_segments(points: list[Point] | tuple[Point, ...]) -> list[Segment]:
    if len(points) < 2:
        return []
    return list(zip(points, points[1:] + points[:1]))


def polygons_cross(a: list[Point] | tuple[Point, ...], b: list[Point] | tuple[Point, ...]) -> bool:
    for sa in polygon_segments(a):
        for sb in polygon_segments(b):
            if segments_intersect(sa, sb):
                return True
    return False


def circle_as_polygon(center: Point, radius: float, n: int = 24) -> list[Point]:
    return [
        (
            center[0] + radius * math.cos(2.0 * math.pi * i / n),
            center[1] + radius * math.sin(2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def arc_as_polyline(start: Point, mid: Point, end: Point) -> list[Point]:
    """Sample the circular arc from ``start`` through ``mid`` to ``end``."""

    x1, y1 = start
    x2, y2 = mid
    x3, y3 = end
    determinant = 2.0 * (
        x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    )
    if abs(determinant) <= EPS:
        raise ValueError("Cannot sample a collinear arc")
    q1 = x1 * x1 + y1 * y1
    q2 = x2 * x2 + y2 * y2
    q3 = x3 * x3 + y3 * y3
    center_x = (
        q1 * (y2 - y3) + q2 * (y3 - y1) + q3 * (y1 - y2)
    ) / determinant
    center_y = (
        q1 * (x3 - x2) + q2 * (x1 - x3) + q3 * (x2 - x1)
    ) / determinant
    radius = distance(start, (center_x, center_y))

    tau = 2.0 * math.pi
    start_angle = math.atan2(y1 - center_y, x1 - center_x) % tau
    mid_angle = math.atan2(y2 - center_y, x2 - center_x) % tau
    end_angle = math.atan2(y3 - center_y, x3 - center_x) % tau
    ccw_sweep = (end_angle - start_angle) % tau
    mid_ccw = (mid_angle - start_angle) % tau
    sweep = ccw_sweep if mid_ccw <= ccw_sweep + EPS else ccw_sweep - tau
    segments = max(2, int(math.ceil(abs(sweep) / (math.pi / 12.0))))
    points = [
        (
            center_x + radius * math.cos(start_angle + sweep * index / segments),
            center_y + radius * math.sin(start_angle + sweep * index / segments),
        )
        for index in range(segments + 1)
    ]
    points[0] = start
    points[-1] = end
    return points
