# analytics.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from player_bfs import bfs_player_no_fire
from utils import INF


def build_category_grid(maze, fire_time, current_time, player_pos, exit_pos, path=None):
    """
    Build an integer grid for visualization:
        0 = empty
        1 = wall
        2 = fire (at or before current_time)
        3 = player
        4 = exit
        5 = path (optional, override empty cells)
    """
    rows, cols = maze.shape
    grid = np.zeros((rows, cols), dtype=int)

    # walls
    grid[maze == 1] = 1

    # fire (cells that have caught fire up to current_time)
    burning = fire_time <= current_time
    grid[burning & (maze == 0)] = 2

    # path
    if path:
        for (r, c) in path:
            # don't overwrite walls, exit, player
            if grid[r, c] == 0:
                grid[r, c] = 5

    # exit
    er, ec = exit_pos
    grid[er, ec] = 4

    # player
    pr, pc = player_pos
    grid[pr, pc] = 3

    return grid


def plot_maze_state(maze, fire_time, current_time, player_pos, exit_pos, path=None):
    """
    High-contrast 'game style' visualization with pixel-grid look.
    """
    grid = build_category_grid(maze, fire_time, current_time, player_pos, exit_pos, path)
    rows, cols = grid.shape

    # Game-style neon palette:
    # 0 empty  -> light metal gray
    # 1 wall   -> near-black
    # 2 fire   -> neon red
    # 3 player -> cyan glow
    # 4 exit   -> gold
    # 5 path   -> neon magenta
    colors = [
        "#C8C8C8",  # empty
        "#1A1A1A",  # wall
        "#FF2B2B",  # fire
        "#00E5FF",  # player
        "#FFD700",  # exit
        "#FF00FF",  # path
    ]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap, interpolation="nearest")

    # Pixel gridlines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(color="#444444", linewidth=0.5)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_title(f"Maze State at t = {current_time}")
    return fig


def plot_heatmap(matrix, title, invalid_value=INF):
    """
    Generic heatmap for fire_time / distances / safety margin.
    Replaces INF with NaN so they show as blank.
    """
    mat = matrix.astype(float).copy()
    mat[mat >= invalid_value] = np.nan

    fig, ax = plt.subplots()
    im = ax.imshow(mat, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def compute_player_distance_and_safety(maze, fire_time, player_start):
    """
    Returns:
      dist_player: distance from player ignoring fire (BFS)
      safety_margin: fire_time - dist_player (where both finite)
    """
    dist_player = bfs_player_no_fire(maze, player_start)
    safety_margin = fire_time.astype(float) - dist_player.astype(float)

    # Where player can't reach or fire never reaches, mark as INF
    mask = (dist_player >= INF) | (fire_time >= INF)
    safety_margin[mask] = INF

    return dist_player, safety_margin
