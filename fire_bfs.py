# fire_bfs.py
import numpy as np
from collections import deque
from utils import INF, DIRECTIONS, in_bounds


def compute_fire_times(maze, fire_sources):
    """
    Multi-source BFS from all fire sources.
    Returns a matrix fire_time where:
        fire_time[r, c] = earliest time step when fire reaches (r, c)
        INF if fire never reaches.
    Walls remain INF.
    """
    rows, cols = maze.shape
    fire_time = np.full((rows, cols), INF, dtype=int)
    q = deque()

    # Initialize queue with all fire sources
    for (r, c) in fire_sources:
        if maze[r, c] == 1:
            continue  # ignore fires spawned inside walls
        fire_time[r, c] = 0
        q.append((r, c))

    # BFS
    while q:
        r, c = q.popleft()
        t = fire_time[r, c]

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, rows, cols):
                continue
            if maze[nr, nc] == 1:  # wall
                continue
            if fire_time[nr, nc] > t + 1:
                fire_time[nr, nc] = t + 1
                q.append((nr, nc))

    return fire_time
