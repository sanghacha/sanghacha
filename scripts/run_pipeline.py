"""uv run python scripts/run_pipeline.py"""

import argparse
from importlib import import_module
from pathlib import Path

from pipeline.environment import MockEnvironment
from pipeline.runner import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock module integration; no real robot / ACT")
    parser.add_argument("--case", choices=("shelves", "blocked", "accessible", "after_move", "missing_target"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("mock", "mujoco"), default="mock")
    parser.add_argument("--hssg-module", default="hssg_generator")
    parser.add_argument("--planner-module", default="planner")
    parser.add_argument("--action-module", default="action_generator")
    args = parser.parse_args()
    args.case = args.case or ("shelves" if args.backend == "mujoco" else "blocked")
    if args.backend == "mujoco" and args.case == "missing_target":
        parser.error("missing_target is an offline/mock error fixture; use shelves, blocked, accessible or after_move in MuJoCo")
    if args.backend == "mock" and args.case == "shelves":
        parser.error("shelves is a MuJoCo scene; select --backend mujoco")
    if args.backend == "mujoco":
        from pipeline.mujoco_env import MujocoEnvironment
        env = MujocoEnvironment(args.case, args.config) if args.config else MujocoEnvironment(args.case)
        if args.action_module == "action_generator":
            print("SO-101 starter C holds the current joints; it does not execute PICK/PLACE. Supply --action-module with your policy.")
    else:
        env = MockEnvironment(args.case, args.config) if args.config else MockEnvironment(args.case)
    try:
        subgoals = run_pipeline(env, env.task, seed=args.seed,
                               generate_graph=import_module(args.hssg_module).generate_scene_graph,
                               plan_subgoal=import_module(args.planner_module).plan_next_subgoal,
                               predict_action=import_module(args.action_module).predict_action_chunk)
        for goal in subgoals:
            print(f"{goal.primitive}({goal.object_id})")
        print(f"{args.backend} succeeded: {len(subgoals)} subgoals")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"{args.backend} failed: {exc}")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
