"""Validate a short connector out of the extra planning buffer."""

import math

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from custom_navigation.IAstar import astar


def find_recovery_start(
    data, width, height, resolution, blocked, robot_grid, goal,
    clearance, person=None, max_distance=0.20,
):
    """Return a buffered A* suffix reachable by a checked straight connector.

    robot_grid uses cell-centre coordinates: cell (0, 0) has centre (0, 0).
    person, if supplied, is (grid_x, grid_y, radius_in_metres).
    """
    rx, ry = robot_grid
    sx, sy = math.floor(rx + 0.5), math.floor(ry + 0.5)
    gx, gy = goal

    def inside(x, y):
        return 0 <= x < width and 0 <= y < height

    if (
        not inside(sx, sy)
        or not inside(gx, gy)
        or not blocked[sy * width + sx]
        or blocked[gy * width + gx]
    ):
        return []

    grid = np.asarray(data).reshape(height, width)
    free = (grid >= 0) & (grid < 50)
    distances = distance_transform_edt(
        np.pad(free, 1, constant_values=False),
        sampling=resolution,
    )[1:-1, 1:-1] - math.sqrt(2.0) * resolution

    def connector_clear(cx, cy):
        length = math.hypot(cx-rx, cy-ry) * resolution
        intervals = max(1, math.ceil(length / 0.005))
        fractions = np.linspace(0.0, 1.0, intervals + 1)
        xs = rx + fractions * (cx-rx)
        ys = ry + fractions * (cy-ry)

        # Stay inside the interpolation domain.
        if (
            np.any(xs < 0) or np.any(xs > width-1)
            or np.any(ys < 0) or np.any(ys > height-1)
        ):
            return False

        values = map_coordinates(
            distances, [ys, xs], order=1,
            mode='constant', cval=-1.0, prefilter=False,
        )

        # Bound variation between samples of the interpolated distance field.
        allowance = math.sqrt(2.0) * length / (2 * intervals)
        if np.any(values < clearance + allowance + 1e-9):
            return False

        if person is not None:
            px, py, radius = person
            dx, dy = cx-rx, cy-ry
            squared = dx*dx + dy*dy
            fraction = (
                0.0 if squared == 0
                else max(0.0, min(
                    1.0, ((px-rx)*dx + (py-ry)*dy) / squared
                ))
            )
            separation = math.hypot(
                rx + fraction*dx - px,
                ry + fraction*dy - py,
            ) * resolution
            if separation <= radius:
                return False

        return True

    radius_cells = math.ceil(max_distance / resolution)
    candidates = []
    for cy in range(max(0, sy-radius_cells), min(height, sy+radius_cells+1)):
        for cx in range(max(0, sx-radius_cells), min(width, sx+radius_cells+1)):
            distance = math.hypot(cx-rx, cy-ry) * resolution
            if distance <= max_distance and not blocked[cy*width+cx]:
                candidates.append((distance, cx, cy))

    for _, cx, cy in sorted(candidates):
        if not connector_clear(cx, cy):
            continue
        route = astar(blocked, width, height, (cx, cy), goal)
        if route:
            return route

    return []
