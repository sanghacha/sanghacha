"""uv run python scripts/run_simulation.py — 실제 SO-101 MuJoCo 공통 개발 환경."""

import argparse
from pathlib import Path
import time

import numpy as np

from hssg_generator import generate_scene_graph
from planner import plan_next_subgoal
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.sample_data import Sample, save_sample
from pipeline.viewer import JointJogControl, SimulationViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 MuJoCo development environment; no physical robot")
    parser.add_argument("--case", choices=("shelves", "blocked", "accessible", "after_move"), default="shelves")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", help="windowless RGB-D rendering")
    parser.add_argument("--physics-only", action="store_true", help="no OpenGL; rgb_valid=False")
    parser.add_argument("--seconds", type=float, default=0, help="0: interactive until viewer closes; headless defaults to 2s")
    parser.add_argument("--demo", choices=("hold", "motion"), default="hold")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--joint-speed", type=float, default=60, help="held-key arm jog speed, degrees/second")
    parser.add_argument("--gripper-speed", type=float, default=.8, help="held-key gripper speed, normalized/second")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    if args.physics_only:
        args.headless = True
    env = MujocoEnvironment(args.case, render=not args.physics_only)
    viewer = None
    try:
        current_seed = args.seed
        observation = env.reset(current_seed)
        initial = observation.robot_state.copy()
        control = JointJogControl(initial, env.lower, env.upper, env.config["max_delta"],
                                  joint_speed=args.joint_speed, gripper_speed=args.gripper_speed)
        print("SO-101 / MuJoCo ready. No hardware connected.")
        print("State: five arm angles [rad], gripper [0=open, 1=closed]")
        if not args.headless:
            viewer = SimulationViewer(env.model, env.data)
            print("Keys: 1..6 select joint, HOLD Up/Down to move, RELEASE to stop, Space home, R new layout.")
            viewer.sync(observation, control, env.joints)
        duration = args.seconds or (2. if args.headless else float("inf"))
        start = time.perf_counter()
        steps = 0
        while observation.timestamp < duration - 1e-9 and (viewer is None or viewer.is_running()):
            direction = viewer.poll_direction() if viewer else 0
            if viewer and not viewer.is_running():
                break
            if viewer and viewer.reset_requested:
                current_seed += 1
                observation = env.reset(current_seed)
                initial = observation.robot_state.copy()
                control.reset(initial)
                viewer.reset_requested = False
                print("new layout seed:", current_seed)
            if viewer and viewer.home_requested:
                control.home()
                viewer.home_requested = False
            if args.demo == "motion":
                control.target[0] = initial[0] + .12 * np.sin(2*np.pi*observation.timestamp/3)
            command = control.command(observation.robot_state, observation.control_dt,
                                      viewer.joint if viewer else 0, direction)
            observation = env.step(command)
            if viewer:
                viewer.sync(observation, control, env.joints)
                time.sleep(max(0., start + (steps+1)*observation.control_dt - time.perf_counter()))
            steps += 1
        graph = generate_scene_graph(observation)
        subgoal = plan_next_subgoal(graph, env.task, observation.robot_state)
        snapshot = args.snapshot or Path("outputs/so101") / args.case / f"seed_{current_seed}"
        save_sample(snapshot, Sample(observation, graph, env.task, subgoal), config=env.config)
        if viewer:
            viewer.sync(observation, control, env.joints, screenshot=snapshot / "viewer.png")
        print("joint delta:", np.round(observation.robot_state - initial, 5).tolist())
        print("graph:", len(graph.nodes), "objects,", sum(e.relation == "supports" for e in graph.edges), "support relations")
        print("objects:", len(observation.ground_truth), "seed:", current_seed)
        print("sample:", snapshot, "rgb_valid:", observation.rgb_valid)
        print("Development run complete; target-retrieval success was not evaluated.")
    finally:
        if viewer:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
