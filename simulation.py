from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np

from fire_bfs import compute_fire_times
from player_bfs import bfs_player_with_fire
from utils import INF


@dataclass
class SimulationStep:
    time: int
    player_pos: Tuple[int, int]
    path: Optional[List[Tuple[int, int]]] = None
    status: str = "RUNNING"  # RUNNING / ESCAPED / TRAPPED


@dataclass
class SimulationResult:
    steps: List[SimulationStep] = field(default_factory=list)
    status: str = "TRAPPED"
    total_time: int = 0
    bfs_op_counts: Dict[str, int] = field(default_factory=dict)
    fire_time: Optional[np.ndarray] = None


def run_dynamic_simulation(maze,
                           player_start,
                           exit_pos,
                           fire_sources,
                           max_steps=500):
    """
    Runs the simulation step-by-step.
    At each step:
      - Player recomputes BFS path to exit
      - Moves 1 step along that path
      - Fire constraints via fire_time matrix
    """
    fire_time = compute_fire_times(maze, fire_sources)

    steps: List[SimulationStep] = []
    status = "RUNNING"
    time = 0
    player_pos = player_start

    total_node_expansions = 0
    total_queue_pushes = 0

    for _ in range(max_steps):
        # If already at exit
        if player_pos == exit_pos:
            status = "ESCAPED"
            steps.append(SimulationStep(time=time, player_pos=player_pos, status=status))
            break

        # Recompute path at current time
        path, stats, _ = bfs_player_with_fire(
            maze,
            start=player_pos,
            goal=exit_pos,
            fire_time=fire_time,
            start_time=time,
        )

        total_node_expansions += stats.node_expansions
        total_queue_pushes += stats.queue_pushes

        if path is None or len(path) <= 1:
            status = "TRAPPED"
            steps.append(SimulationStep(time=time, player_pos=player_pos, path=None, status=status))
            break

        # Move to next cell on path
        next_pos = path[1]
        time += 1
        player_pos = next_pos

        # Check if fire reaches player at this time
        pr, pc = player_pos
        if fire_time[pr, pc] <= time:
            status = "TRAPPED"
            steps.append(SimulationStep(time=time, player_pos=player_pos, path=path, status=status))
            break

        # Check for escape
        if player_pos == exit_pos:
            status = "ESCAPED"
            steps.append(SimulationStep(time=time, player_pos=player_pos, path=path, status=status))
            break

        steps.append(SimulationStep(time=time, player_pos=player_pos, path=path, status="RUNNING"))

    result = SimulationResult(
        steps=steps,
        status=status,
        total_time=time,
        bfs_op_counts={
            "total_node_expansions": total_node_expansions,
            "total_queue_pushes": total_queue_pushes,
        },
        fire_time=fire_time,
    )
    return result
