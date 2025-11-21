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
    defaults = {
        "maze": None,
        "player_start": None,
        "exit_pos": None,
        "fire_sources": None,
        "sim_result": None,
        "current_t": 0,
        "auto_play": False,
        "time_slider_key": "time_slider_0",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def safely_get_last_step(steps):
    """Return the last step or None, without crashing."""
    return steps[-1] if steps else None


def main():
    st.set_page_config(page_title="Fire in the Maze", layout="wide")
    st.title("🔥 Fire in the Maze — Dynamic BFS Simulation (DAA Project)")

    init_session_state()

    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("Configuration")

        rows = st.slider("Rows", 10, 50, 20)
        cols = st.slider("Cols", 10, 50, 20)
        wall_prob = st.slider("Wall Density", 0.0, 0.6, 0.3, 0.05)
        num_fires = st.slider("Fire Sources", 1, 5, 2)
        max_steps = st.slider("Max Simulation Steps", 10, 300, 100)
        seed = st.number_input("Random Seed", 0, 10000, 0)

        if st.button("Generate New Maze"):
            maze = generate_maze(rows, cols, wall_prob=wall_prob, seed=seed)
            try:
                player_start, exit_pos, fire_sources = choose_positions(
                    maze, num_fires=num_fires, rng_seed=seed + 1
                )
            except ValueError:
                st.error("Maze too dense. Reduce wall density.")
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
                st.session_state.time_slider_key = f"time_slider_{np.random.randint(1000000)}"

        if st.button("Run Dynamic Simulation"):
            if st.session_state.maze is None:
                st.warning("Generate a maze first.")
            elif not st.session_state.fire_sources:
                st.warning("No fire sources. Generate another maze.")
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
                st.session_state.time_slider_key = f"time_slider_{np.random.randint(1000000)}"

    # ================= LAYOUT =================
    col1, col2 = st.columns([2, 1])

    # ---------- LEFT PANEL ----------
    with col1:
        st.subheader("Simulation View")

        if st.session_state.maze is None:
            st.info("Generate a maze to begin.")
            return

        maze = st.session_state.maze
        player_start = st.session_state.player_start
        exit_pos = st.session_state.exit_pos

        # If no simulation yet
        if st.session_state.sim_result is None:
            fire_time = compute_fire_times(maze, st.session_state.fire_sources)
            fig = plot_maze_state(maze, fire_time, 0, player_start, exit_pos, None)
            st.pyplot(fig)
            return

        # Simulation exists
        sim = st.session_state.sim_result
        steps = sim.steps

        # SAFETY CHECK: steps empty
        if not steps:
            st.error("Simulation returned 0 steps. Try lowering wall density.")
            return

        last_step = steps[-1]
        max_t = last_step.time

        # Time slider
        st.markdown("### Time Control")
        t = st.slider(
            "Time Step",
            0,
            max_t,
            st.session_state.current_t,
            key=st.session_state.time_slider_key,
        )

        st.session_state.current_t = t

        # Navigation buttons
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

        # SAFELY GET CURRENT STEP
        valid_steps = [s for s in steps if s.time <= t]
        current_step = valid_steps[-1] if valid_steps else steps[0]  # safe fallback

        fig = plot_maze_state(
            maze,
            sim.fire_time,
            t,
            current_step.player_pos,
            exit_pos,
            current_step.path,
        )
        st.pyplot(fig)

        st.markdown(f"**Status at t={t}:** `{current_step.status}`")
        st.markdown(f"**Final Status:** `{sim.status}` | **Total Time:** `{sim.total_time}`")

        # Auto-play logic
        if st.session_state.auto_play and t < max_t:
            import time as _time
            _time.sleep(0.18)
            st.session_state.current_t += 1
            st.rerun()

    # ---------- RIGHT PANEL ----------
    with col2:
        st.subheader("Analytics")

        sim = st.session_state.sim_result
        fire_time = sim.fire_time

        st.write("### BFS Stats")
        st.json(sim.bfs_op_counts)

        dist_player, safety_margin = compute_player_distance_and_safety(
            maze, fire_time, st.session_state.player_start
        )

        with st.expander("Fire Reach Heatmap"):
            st.pyplot(plot_heatmap(fire_time, "Fire Reach Time"))

        with st.expander("Player Distance Heatmap"):
            st.pyplot(plot_heatmap(dist_player, "Player Distance", INF))

        with st.expander("Safety Margin Heatmap"):
            st.pyplot(plot_heatmap(safety_margin, "Safety Margin", INF))

    st.markdown("---")
    st.caption("DAA Project — Fire in the Maze (Dynamic BFS Avoiding Fire)")


if __name__ == "__main__":
    main()
