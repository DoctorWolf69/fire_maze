# utils.py
from collections import deque

INF = 10**9

# 4-directional movement
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols


class BFSStats:
    """Tracks BFS operations for analysis."""
    def __init__(self):
        self.node_expansions = 0
        self.queue_pushes = 0

    def to_dict(self):
        return {
            "node_expansions": self.node_expansions,
            "queue_pushes": self.queue_pushes,
        }
