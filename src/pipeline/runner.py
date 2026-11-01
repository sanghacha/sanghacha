"""세 개의 공개 함수를 그대로 호출하는 오케스트레이터. mock/MuJoCo 환경 공용."""

import numpy as np

from action_generator import predict_action_chunk
from hssg_generator import generate_scene_graph
from planner import plan_next_subgoal

from .environment import MockEnvironment
from .mujoco_env import MujocoEnvironment
from .types import ActionChunk, SceneGraph, Subgoal, TaskRequest, check
from .target import TargetResolver


def run_pipeline(env: MockEnvironment | MujocoEnvironment, task: TaskRequest, *, seed: int = 0,
                 generate_graph=generate_scene_graph, plan_subgoal=plan_next_subgoal,
                 predict_action=predict_action_chunk, on_step=None, refresh_observation=None, on_event=None) -> list[Subgoal]:
    """자기 팀 함수 하나만 keyword argument로 교체해서 통합할 수 있다."""
    completed = []
    seen_ids = set()
    resolver = TargetResolver()
    def emit(event, **payload):
        if on_event is not None:
            on_event(event, **payload)
    try:
        observation = env.reset(seed)
        check(env.contains(task.destination.position), "destination outside workspace")
        for _ in range(env.config["max_subgoals"] + 1):
            if refresh_observation is not None:
                refreshed = refresh_observation()
                check(refreshed.episode_id == observation.episode_id and
                      np.isclose(refreshed.timestamp, observation.timestamp, rtol=0, atol=1e-9),
                      "scene refresh must not advance physics")
                observation = refreshed
            graph = generate_graph(observation)
            check(isinstance(graph, SceneGraph), "A must return SceneGraph")
            check(graph.episode_id == observation.episode_id and graph.source == observation.source and
                  np.isclose(graph.timestamp, observation.timestamp, rtol=0, atol=1e-9), "stale/mismatched graph")
            task = resolver.resolve(graph, task)
            emit('scene', observation=observation, graph=graph, task=task)
            if env.task_complete(task):
                emit('finished', observation=observation, completed=tuple(completed))
                return completed
            check(len(completed) < env.config["max_subgoals"], "subgoal count budget exhausted")
            subgoal = plan_subgoal(graph, task, observation.robot_state)
            check(isinstance(subgoal, Subgoal), "B must return Subgoal")
            check(subgoal.subgoal_id not in seen_ids, "repeated subgoal ID")
            seen_ids.add(subgoal.subgoal_id)
            check(np.allclose(graph.object(subgoal.object_id).pose.position, subgoal.object_pose.position,
                              rtol=0, atol=1e-5), "stale subgoal object position")
            check(env.contains(subgoal.target_ee_pose.position), "EE target outside workspace")
            if subgoal.place_pose:
                check(env.contains(subgoal.place_pose.position), "placement outside workspace")
            emit('planned', observation=observation, graph=graph, task=task, subgoal=subgoal,
                 index=len(completed)+1, budget=env.config['max_steps_per_subgoal'])
            steps = 0
            while not env.subgoal_complete(subgoal):
                check(steps < env.config["max_steps_per_subgoal"], "subgoal step budget exhausted")
                chunk = predict_action(observation, subgoal)
                check(isinstance(chunk, ActionChunk), "C must return ActionChunk")
                check(chunk.subgoal_id == subgoal.subgoal_id and chunk.robot_id == observation.robot_id,
                      "stale/mismatched action chunk")
                check(np.isclose(chunk.command_dt, observation.control_dt, rtol=1e-6, atol=1e-9), "wrong command_dt")
                emit('chunk', chunk=chunk)
                count = min(env.config["n_action_steps"], env.config["max_steps_per_subgoal"] - steps)
                for command in chunk.commands[:count]:
                    previous = observation
                    observation = env.step(command)
                    check(observation.episode_id == previous.episode_id and
                          np.isclose(observation.timestamp - previous.timestamp, previous.control_dt), "invalid step timing")
                    steps += 1
                    if on_step is not None:
                        on_step(previous, command, observation, subgoal)
                    if env.subgoal_complete(subgoal):
                        break
            completed.append(subgoal)
            emit('completed', observation=observation, subgoal=subgoal, steps=steps)
        raise RuntimeError("pipeline did not terminate")
    except Exception as exc:
        emit('failed', error=str(exc))
        raise
    finally:
        env.stop()
