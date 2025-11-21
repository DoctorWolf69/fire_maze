# maze.py
import numpy as np
from numpy.random import default_rng


def generate_maze(rows, cols, wall_prob=0.3, seed=None):
    """
    Generate a random maze.
    0 = empty cell
    1 = wall
    """
    rng = default_rng(seed)
    maze = (rng.random((rows, cols)) < wall_prob).astype(int)
    return maze


def choose_positions(maze, num_fires=1, rng_seed=None):
    """
    Choose positions for:
    - player_start
    - exit_pos
    - fire_sources (list)
    All on empty cells (0).
    """
    rng = default_rng(rng_seed)
    rows, cols = maze.shape
    empty_cells = [(r, c) for r in range(rows) for c in range(cols) if maze[r, c] == 0]

    if len(empty_cells) < num_fires + 2:
        raise ValueError("Not enough empty cells to place player, exit, and fires.")

    rng.shuffle(empty_cells)

    player_start = empty_cells[0]
    exit_pos = empty_cells[1]
    fire_sources = empty_cells[2:2 + num_fires]

    return player_start, exit_pos, fire_sources
