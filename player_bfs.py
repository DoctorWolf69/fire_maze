# player_bfs.py
import numpy as np
from collections import deque
from utils import INF, DIRECTIONS, in_bounds, BFSStats


def bfs_player_with_fire(maze, start, goal, fire_time, start_time=0):
    """
    BFS from start to goal with constraint:
        at time t, player cannot be on a cell where fire_time <= t.

    start_time = current simulation time (player has already taken t steps).

    Returns:
        path: list of (r, c) from start to goal (inclusive), or None
        stats: BFSStats object
        dist: distance grid (for analysis)
    """
    rows, cols = maze.shape
    stats = BFSStats()

    dist = np.full((rows, cols), INF, dtype=int)
    parent = [[None for _ in range(cols)] for _ in range(rows)]

    sr, sc = start
    gr, gc = goal

    # If start is already on fire at current time, impossible
    if fire_time[sr, sc] <= start_time:
        return None, stats, dist

    q = deque()
    q.append((sr, sc))
    dist[sr, sc] = 0
    stats.queue_pushes += 1

    while q:
        r, c = q.popleft()
        stats.node_expansions += 1
        t_here = dist[r, c] + start_time

        # If fire reaches or has reached here by this time, discard this node
        if fire_time[r, c] <= t_here:
            continue

        if (r, c) == (gr, gc):
            break

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, rows, cols):
                continue
            if maze[nr, nc] == 1:  # wall
                continue
            if dist[nr, nc] != INF:
                continue

            # Time when player would reach neighbor
            t_next = dist[r, c] + 1 + start_time

            # Player must reach neighbor STRICTLY before fire
            if fire_time[nr, nc] <= t_next:
                continue

            dist[nr, nc] = dist[r, c] + 1
            parent[nr][nc] = (r, c)
            q.append((nr, nc))
            stats.queue_pushes += 1

    if dist[gr, gc] == INF:
        return None, stats, dist

    # Reconstruct path from goal to start
    path = []
    cur = (gr, gc)
    while cur is not None:
        path.append(cur)
        r, c = cur
        cur = parent[r][c]
    path.reverse()

    return path, stats, dist


def bfs_player_no_fire(maze, start):
    """
    Simple BFS ignoring fire – used for distance heatmap.
    Returns dist matrix.
    """
    rows, cols = maze.shape
    dist = np.full((rows, cols), INF, dtype=int)
    from collections import deque
    q = deque()
    sr, sc = start
    if maze[sr, sc] == 1:
        return dist

    dist[sr, sc] = 0
    q.append((sr, sc))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, rows, cols):
                continue
            if maze[nr, nc] == 1:
                continue
            if dist[nr, nc] != INF:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return dist
