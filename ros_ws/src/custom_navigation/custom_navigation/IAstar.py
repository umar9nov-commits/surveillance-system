"""A* planning on an occupancy grid."""

import heapq
import math


def build_blocked_grid(data, width, height, resolution, clearance=0.22):
    """Use the controller's clearance calculation plus a tracking buffer."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    if (
        resolution <= 0
        or width <= 0
        or height <= 0
        or len(data) != width * height
    ):
        raise ValueError("Invalid occupancy grid.")

    grid = np.asarray(data).reshape(height, width)
    free = (grid >= 0) & (grid < 50)

    # Unknown cells and space outside the map are blocked.
    padded = np.pad(free, 1, constant_values=False)
    distances = distance_transform_edt(
        padded, sampling=resolution
    )[1:-1, 1:-1]

    # Same cell allowance used by the MPC controller.
    conservative = distances - math.sqrt(2.0) * resolution

    # Additional space for tracking error during movement.
    tracking_buffer = 0.08
    blocked = (~free) | (
        conservative < clearance + tracking_buffer
    )

    return blocked.ravel().tolist()


def astar(blocked, width, height, start, goal):
    """Return a grid path, or an empty list if no path exists."""

    def is_free(cell):
        x, y = cell
        return (
            0 <= x < width
            and 0 <= y < height
            and not blocked[y * width + x]
        )

    def heuristic(cell):
        dx = abs(cell[0] - goal[0])
        dy = abs(cell[1] - goal[1])
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    if not is_free(start) or not is_free(goal):
        return []

    moves = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (1, -1),
        (-1, 1), (-1, -1),
    ]

    queue = [(heuristic(start), 0.0, start)]
    costs = {start: 0.0}
    parents = {}

    while queue:
        _, current_cost, current = heapq.heappop(queue)

        if current_cost > costs.get(current, math.inf):
            continue

        if current == goal:
            path = [current]

            while current in parents:
                current = parents[current]
                path.append(current)

            return path[::-1]

        x, y = current

        for dx, dy in moves:
            neighbor = (x + dx, y + dy)

            if not is_free(neighbor):
                continue

            # Prevent diagonal movement through blocked corners.
            if dx != 0 and dy != 0:
                if (
                    not is_free((x + dx, y))
                    or not is_free((x, y + dy))
                ):
                    continue

            new_cost = current_cost + math.hypot(dx, dy)

            if new_cost < costs.get(neighbor, math.inf):
                costs[neighbor] = new_cost
                parents[neighbor] = current
                priority = new_cost + heuristic(neighbor)

                heapq.heappush(
                    queue,
                    (priority, new_cost, neighbor),
                )

    return []
