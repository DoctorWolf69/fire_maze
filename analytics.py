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

    # fire
    burning = fire_time <= current_time
    grid[burning & (maze == 0)] = 2

    # path
    if path:
        for (r, c) in path:
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
    High-contrast 'game style' visualization.
    """
    grid = build_category_grid(maze, fire_time, current_time, player_pos, exit_pos, path)

    # Define colors:
    # 0 empty  -> light gray
    # 1 wall   -> black
    # 2 fire   -> bright red
    # 3 player -> cyan
    # 4 exit   -> yellow
    # 5 path   -> magenta
    colors = [
        "#E0E0E0",  # empty
        "#000000",  # wall
        "#FF0000",  # fire
        "#00FFFF",  # player
        "#FFFF00",  # exit
        "#FF00FF",  # path
    ]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap, interpolation="nearest")

    ax.set_xticks([])
    ax.set_yticks([])
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
      dist_player: distance from player ignoring fire
      safety_margin: fire_time - dist_player (where both finite)
    """
    dist_player = bfs_player_no_fire(maze, player_start)
    safety_margin = fire_time.astype(float) - dist_player.astype(float)

    # Where dist_player is INF, set safety_margin to INF for clarity
    mask = (dist_player >= INF) | (fire_time >= INF)
    safety_margin[mask] = INF

    return dist_player, safety_margin
