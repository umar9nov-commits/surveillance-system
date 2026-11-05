"""Prune grid paths using conservative line-of-sight checks."""

import math


def segment_is_clear(start, end, blocked, width, height):
    """Check every grid cell touched by a line between cell centres."""
    x, y = start
    end_x, end_y = end

    def free(cx, cy):
        return (
            0 <= cx < width
            and 0 <= cy < height
            and not blocked[cy * width + cx]
        )

    if not free(x, y) or not free(end_x, end_y):
        return False

    dx = end_x - x
    dy = end_y - y
    step_x = (dx > 0) - (dx < 0)
    step_y = (dy > 0) - (dy < 0)

    delta_x = 1.0 / abs(dx) if dx else math.inf
    delta_y = 1.0 / abs(dy) if dy else math.inf
    next_x = 0.5 * delta_x
    next_y = 0.5 * delta_y

    while (x, y) != (end_x, end_y):
        if math.isclose(next_x, next_y, rel_tol=0.0, abs_tol=1e-12):
            # At a corner, both neighbouring cells must be free.
            if not free(x + step_x, y) or not free(x, y + step_y):
                return False
            x += step_x
            y += step_y
            next_x += delta_x
            next_y += delta_y
        elif next_x < next_y:
            x += step_x
            next_x += delta_x
        else:
            y += step_y
            next_y += delta_y

        if not free(x, y):
            return False

    return True


def prune_path(cells, blocked, width, height):
    """Keep the farthest visible waypoint from each retained waypoint."""
    if width <= 0 or height <= 0 or len(blocked) != width * height:
        raise ValueError("Invalid blocked grid.")

    cells = list(cells)
    if not cells:
        return []

    # Validate the original route before attempting shortcuts.
    for cell in cells:
        if not segment_is_clear(cell, cell, blocked, width, height):
            raise ValueError("Path contains a blocked or outside cell.")

    for start, end in zip(cells, cells[1:]):
        if not segment_is_clear(start, end, blocked, width, height):
            raise ValueError("Original path crosses blocked space.")

    result = [cells[0]]
    index = 0

    while index < len(cells) - 1:
        candidate = len(cells) - 1

        while not segment_is_clear(
            cells[index], cells[candidate], blocked, width, height
        ):
            candidate -= 1

        result.append(cells[candidate])
        index = candidate

    return result


def smooth_path(cells, blocked, width, height):
    """Return grid-coordinate points and whether spline smoothing passed.

    Validate cubic Bezier control-point bounding boxes recursively.
    Each accepted box must lie entirely in unblocked grid cells.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    cells = list(cells)

    # Validate the fallback route first.
    if width <= 0 or height <= 0 or len(blocked) != width * height:
        raise ValueError("Invalid blocked grid.")

    for cell in cells:
        if not segment_is_clear(cell, cell, blocked, width, height):
            raise ValueError("Invalid route point.")
    for a, b in zip(cells, cells[1:]):
        if not segment_is_clear(a, b, blocked, width, height):
            raise ValueError("Invalid fallback route.")

    fallback = [(float(x), float(y)) for x, y in cells]
    if len(cells) < 3:
        return fallback, False

    points = np.asarray(cells, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Duplicate consecutive route points.")

    parameter = np.concatenate(([0.0], np.cumsum(lengths)))
    spline = CubicSpline(
        parameter, points, axis=0, bc_type='natural'
    )
    derivative = spline.derivative()
    output = [tuple(points[0])]

    def box_is_clear(control):
        # Coordinates refer to cell centres. Cell edges are at +/- 0.5.
        # Include both sides when a box touches a cell boundary.
        epsilon = 1e-9
        lower = np.floor(
            np.min(control, axis=0) + 0.5 - epsilon
        ).astype(int)
        upper = np.floor(
            np.max(control, axis=0) + 0.5 + epsilon
        ).astype(int)

        x0, y0 = lower
        x1, y1 = upper
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
            return False

        return all(
            not blocked[y * width + x]
            for y in range(y0, y1 + 1)
            for x in range(x0, x1 + 1)
        )

    def subdivide(control, depth=0):
        if not np.all(np.isfinite(control)):
            return False

        polygon_length = np.linalg.norm(
            np.diff(control, axis=0), axis=1
        ).sum()

        # Both the curve and its displayed chord lie inside this box.
        if box_is_clear(control) and polygon_length <= 0.4:
            output.append(tuple(control[-1]))
            return len(output) <= 20000

        if depth >= 18:
            return False

        a = (control[0] + control[1]) / 2.0
        b = (control[1] + control[2]) / 2.0
        c = (control[2] + control[3]) / 2.0
        d = (a + b) / 2.0
        e = (b + c) / 2.0
        middle = (d + e) / 2.0

        left = np.array([control[0], a, d, middle])
        right = np.array([middle, e, c, control[3]])
        return (
            subdivide(left, depth + 1)
            and subdivide(right, depth + 1)
        )

    for index, interval in enumerate(np.diff(parameter)):
        control = np.array([
            points[index],
            points[index] + derivative(parameter[index]) * interval / 3.0,
            points[index + 1]
            - derivative(parameter[index + 1]) * interval / 3.0,
            points[index + 1],
        ])
        if not subdivide(control):
            return fallback, False

    return output, True
