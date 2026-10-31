"""uv run python scripts/check_module.py a [--module my_module]"""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np

from pipeline.mock_data import make_mock_data, make_policy_input
from pipeline.types import ActionChunk, SceneGraph, Subgoal, check
from pipeline.sample_data import load_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("team", choices=("a", "b", "c"))
    parser.add_argument("--module", help="공개 함수가 있는 import 가능한 Python 모듈")
    parser.add_argument("--backend", choices=("mock", "mujoco"), default="mujoco")
    parser.add_argument("--sample", type=Path, help="저장된 SO-101 입력 폴더; 시뮬레이터 재실행 없이 검사")
    args = parser.parse_args()
    module = import_module(args.module or {"a": "hssg_generator", "b": "planner", "c": "action_generator"}[args.team])
    cases = ("shelves", "blocked", "accessible", "after_move") if args.backend == "mujoco" else ("blocked", "accessible", "after_move")
    for case in ((str(args.sample),) if args.sample else cases):
        env = None
        if args.sample:
            sample = load_sample(args.sample)
            observation, reference, task = sample.observation, sample.graph, sample.task
            expected = (sample.subgoal.primitive, sample.subgoal.object_id)
            goal_input = sample.subgoal
        elif args.backend == "mujoco":
            from pipeline.mujoco_env import MujocoEnvironment
            from hssg_generator import generate_scene_graph
            from planner import plan_next_subgoal

            env = MujocoEnvironment(case)
            try:
                observation = env.reset(seed=42)
                reference = generate_scene_graph(observation)
                task = env.task
                goal_input = plan_next_subgoal(reference, task, observation.robot_state)
                expected = ((goal_input.primitive, goal_input.object_id) if case == "shelves" else
                            ("PICK", "A" if case == "blocked" else "B"))
            finally:
                env.close()
        else:
            data = make_mock_data(case)
            observation, reference, task = data.observation, data.graph, data.task
            expected = data.expected_order[0]
            _, goal_input = make_policy_input("PICK", case)
        if args.team == "a":
            graph = module.generate_scene_graph(observation)
            check(isinstance(graph, SceneGraph), "expected SceneGraph")
            check(graph.episode_id == observation.episode_id and graph.timestamp == observation.timestamp and
                  graph.source == observation.source, "graph metadata mismatch")
            check({obj.object_id for obj in graph.nodes} == {obj.object_id for obj in reference.nodes}, "object IDs changed")
            check({e for e in graph.edges if e.relation in ("on_top_of", "supports")} == set(reference.edges),
                  "unexpected support relations")
        elif args.team == "b":
            goal = module.plan_next_subgoal(reference, task, observation.robot_state)
            check(isinstance(goal, Subgoal) and (goal.primitive, goal.object_id) == expected,
                  "unexpected first subgoal")
        else:
            inputs = ((observation, goal_input),)
            if observation.source == "mock_gt":
                inputs = tuple(make_policy_input(primitive, case) for primitive in ("PICK", "PLACE"))
            for obs, goal in inputs:
                chunk = module.predict_action_chunk(obs, goal)
                check(isinstance(chunk, ActionChunk), "expected ActionChunk")
                check(chunk.subgoal_id == goal.subgoal_id and chunk.robot_id == obs.robot_id and
                      np.isclose(chunk.command_dt, obs.control_dt), "action metadata mismatch")
        print(f"{case}: passed")
    print("Input/output checks only; policy grasp success is not evaluated.")


if __name__ == "__main__":
    main()
