import numpy as np
import streamlit as st

from maze import generate_maze, choose_positions
from simulation import run_dynamic_simulation
from analytics import (
    plot_maze_state,
    plot_heatmap,
    compute_player_distance_and_safety,
)
from fire_bfs import compute_fire_times
from utils import INF


def init_session_state():
    if "maze" not in st.session_state:
        st.session_state.maze = None
    if "player_start" not in st.session_state:
        st.session_state.player_start = None
    if "exit_pos" not in st.session_state:
        st.session_state.exit_pos = None
    if "fire_sources" not in st.session_state:
        st.session_state.fire_sources = None
    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None
    if "current_t" not in st.session_state:
        st.session_state.current_t = 0
    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False
    if "time_slider_key" not in st.session_state:
        st.session_state.time_slider_key = "time_slider_0"


def main():
    st.set_page_config(page_title="Fire in the Maze", layout="wide")
    st.title("Fire in the Maze - Dynamic Shortest Path (DAA Project)")

    init_session_state()

    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("Configuration")

        rows = st.slider("Rows", 10, 50, 20)
        cols = st.slider("Cols", 10, 50, 20)
        wall_prob = st.slider("Wall density", 0.0, 0.6, 0.3, 0.05)
        num_fires = st.slider("Number of fire sources", 1, 5, 2)
        max_steps = st.slider("Max simulation steps", 10, 300, 100)
        seed = st.number_input(
            "Random seed (for reproducible maze)",
            min_value=0,
            max_value=10_000,
            value=0,
            step=1,
        )

        if st.button("Generate New Maze"):
            maze = generate_maze(rows, cols, wall_prob=wall_prob, seed=seed)
            try:
                player_start, exit_pos, fire_sources = choose_positions(
                    maze, num_fires=num_fires, rng_seed=seed + 1
                )
            except ValueError:
                st.error("Maze too dense to place player/exit/fires. Reduce wall density.")
                st.session_state.maze = None
                st.session_state.sim_result = None
            else:
                st.session_state.maze = maze
                st.session_state.player_start = player_start
                st.session_state.exit_pos = exit_pos
                st.session_state.fire_sources = fire_sources
                st.session_state.sim_result = None
                st.session_state.current_t = 0
                st.session_state.auto_play = False
                st.session_state.time_slider_key = f"time_slider_{np.random.randint(0, 1_000_000)}"

        if st.button("Run Dynamic Simulation"):
            if st.session_state.maze is None:
                st.warning("Generate a maze first.")
            elif not st.session_state.fire_sources:
                st.warning("No fire sources. Generate a new maze.")
            else:
                sim_result = run_dynamic_simulation(
                    st.session_state.maze,
                    st.session_state.player_start,
                    st.session_state.exit_pos,
                    st.session_state.fire_sources,
                    max_steps=max_steps,
                )
                st.session_state.sim_result = sim_result
                st.session_state.current_t = 0
                st.session_state.auto_play = False
                st.session_state.time_slider_key = f"time_slider_{np.random.randint(0, 1_000_000)}"

    # ================= MAIN LAYOUT =================
    col1, col2 = st.columns([2, 1])

    # ---------- LEFT: Maze & Simulation ----------
    with col1:
        st.subheader("Maze / Simulation")

        if st.session_state.maze is None:
            st.info("Generate a maze from the sidebar to begin.")
        else:
            maze = st.session_state.maze
            player_start = st.session_state.player_start
            exit_pos = st.session_state.exit_pos

            if st.session_state.sim_result is None:
                fire_time = compute_fire_times(maze, st.session_state.fire_sources)
                fig = plot_maze_state(
                    maze,
                    fire_time,
                    current_time=0,
                    player_pos=player_start,
                    exit_pos=exit_pos,
                    path=None,
                )
                st.pyplot(fig)

            else:
                sim_res = st.session_state.sim_result
                steps = sim_res.steps

                if not steps:
                    st.warning("Simulation produced no steps.")
                else:
                    max_t = steps[-1].time
                    st.session_state.current_t = min(
                        st.session_state.current_t, max_t
                    )

                    st.markdown("### Time Control")

                    t = st.slider(
                        "Time step",
                        0,
                        max_t,
                        st.session_state.current_t,
                        key=st.session_state.time_slider_key,
                    )
                    st.session_state.current_t = t

                    b1, b2, b3, b4 = st.columns(4)

                    with b1:
                        if st.button("Prev"):
                            st.session_state.current_t = max(t - 1, 0)

                    with b2:
                        if st.button("Next"):
                            st.session_state.current_t = min(t + 1, max_t)

                    with b3:
                        if st.button("Play"):
                            st.session_state.current_t = 0
                            st.session_state.auto_play = True

                    with b4:
                        if st.button("Pause"):
                            st.session_state.auto_play = False

                    t = st.session_state.current_t

                    valid_steps = [s for s in steps if s.time <= t]
                    current_step = valid_steps[-1]

                    fig = plot_maze_state(
                        maze,
                        sim_res.fire_time,
                        current_time=t,
                        player_pos=current_step.player_pos,
                        exit_pos=exit_pos,
                        path=current_step.path,
                    )
                    st.pyplot(fig)

                    st.markdown(f"**Status at t={t}:** `{current_step.status}`")
                    st.markdown(
                        f"**Final Status:** `{sim_res.status}` | **Time:** `{sim_res.total_time}`"
                    )

                    if st.session_state.auto_play and t < max_t:
                        import time as _time
                        _time.sleep(0.18)
                        st.session_state.current_t += 1
                        st.rerun()   # <-- FIXED

    # ---------- RIGHT: Analytics ----------
    with col2:
        st.subheader("Analytics")

        if st.session_state.maze is None:
            st.info("Run a simulation to see analytics.")
        else:
            maze = st.session_state.maze
            player_start = st.session_state.player_start

            if st.session_state.sim_result:
                fire_time = st.session_state.sim_result.fire_time
                st.write("**BFS Operation Counts:**")
                st.json(st.session_state.sim_result.bfs_op_counts)
            else:
                fire_time = compute_fire_times(maze, st.session_state.fire_sources)

            dist_player, safety_margin = compute_player_distance_and_safety(
                maze, fire_time, player_start
            )

            with st.expander("Fire Reach Time Heatmap"):
                st.pyplot(plot_heatmap(fire_time, "Fire Reach Time"))

            with st.expander("Player Distance Heatmap"):
                st.pyplot(plot_heatmap(dist_player, "Distance from Player"))

            with st.expander("Safety Margin Heatmap"):
                st.pyplot(plot_heatmap(safety_margin, "Safety Margin"))

    st.markdown("---")
    st.caption("DAA Project - Fire in the Maze (Dynamic BFS under spreading fire)")


if __name__ == "__main__":
    main()
