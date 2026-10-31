"""단순한 mock 환경. 실제 MuJoCo/SO-101 제어는 포함하지 않는다."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .mock_data import ROOT, make_mock_data
from .types import MOCK_GRASP_OFFSET, Observation, Pose, Subgoal, TaskRequest, check


class MockEnvironment:
    def __init__(self, case: str = "blocked", config_path: str | Path = ROOT / "configs/envs/mock.json"):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        check(self.config["backend"] == "mock" and self.config["robot_id"] == "mock", "only mock backend is available")
        for key, size in (("joint_lower", 6), ("joint_upper", 6), ("max_delta", 6),
                          ("workspace_min", 3), ("workspace_max", 3)):
            value = np.array(self.config[key])
            check(value.shape == (size,) and bool(np.isfinite(value).all()), f"invalid {key}")
        check(np.all(np.array(self.config["joint_lower"]) < self.config["joint_upper"]), "invalid joint bounds")
        check(np.all(np.array(self.config["max_delta"]) > 0), "invalid delta limits")
        check(np.all(np.array(self.config["workspace_min"]) < self.config["workspace_max"]), "invalid workspace")
        check(np.isfinite(self.config["control_dt"]) and self.config["control_dt"] > 0, "invalid control_dt")
        for key in ("n_action_steps", "max_steps_per_subgoal", "max_subgoals"):
            check(type(self.config[key]) is int and self.config[key] > 0, f"invalid {key}")
        self.fixture = make_mock_data(case)
        self.task = self.fixture.task
        self.stopped = False
        self.closed = False
        self._observation: Observation | None = None

    def reset(self, seed: int = 0) -> Observation:
        check(not self.closed, "environment is closed")
        check(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
        self._observation = replace(self.fixture.observation, episode_id=f"mock-{seed}",
                                    control_dt=self.config["control_dt"])
        self.stopped = False
        return self.observe()

    def observe(self) -> Observation:
        check(not self.closed and self._observation is not None, "call reset first")
        return self._observation

    def contains(self, position) -> bool:
        check(len(position) == 3 and bool(np.isfinite(position).all()), "invalid xyz")
        return all(lo <= p <= hi for lo, p, hi in zip(
            self.config["workspace_min"], position, self.config["workspace_max"]))

    def validate_command(self, command: np.ndarray) -> None:
        check(isinstance(command, np.ndarray) and command.shape == (6,) and command.dtype == np.float32 and
              bool(np.isfinite(command).all()), "command must be finite float32[6]")
        check(np.all(command >= self.config["joint_lower"]) and np.all(command <= self.config["joint_upper"]),
              "command outside joint bounds")
        check(np.all(np.abs(command.astype(float) - self.observe().robot_state) <=
                     np.array(self.config["max_delta"]) + 1e-6), "command exceeds delta limit")

    def step(self, command: np.ndarray) -> Observation:
        observation = self.observe()
        check(not self.stopped, "environment is stopped")
        self.validate_command(command)
        ee = command[:3].astype(float) * .1
        check(self.contains(ee), "EE outside workspace")
        center = ee - [0., 0., MOCK_GRASP_OFFSET]
        objects = observation.ground_truth
        held = observation.held_object_id
        if held is None and observation.robot_state[5] < .5 <= command[5]:
            for obj in objects:
                blocked = any(other.pose.position[2] > obj.pose.position[2] and
                              all(abs(other.pose.position[i] - obj.pose.position[i]) <
                                  (other.size[i] + obj.size[i])/2 for i in (0, 1))
                              for other in objects)
                if not blocked and np.linalg.norm(np.array(obj.pose.position) - center) <= .005:
                    held = obj.object_id
                    break
        if held is not None:
            check(self.contains(center), "held object outside workspace")
        objects = tuple(replace(obj, pose=Pose(tuple(center))) if obj.object_id == held else obj for obj in objects)
        self._observation = replace(observation, robot_state=command, ground_truth=objects,
                                    held_object_id=None if command[5] < .5 else held,
                                    timestamp=observation.timestamp + observation.control_dt)
        return self.observe()

    def subgoal_complete(self, subgoal: Subgoal) -> bool:
        observation = self.observe()
        if subgoal.primitive == "PICK":
            return observation.held_object_id == subgoal.object_id
        obj = next(obj for obj in observation.ground_truth if obj.object_id == subgoal.object_id)
        return observation.held_object_id is None and np.linalg.norm(
            np.array(obj.pose.position) - subgoal.place_pose.position) <= .005

    def task_complete(self, task: TaskRequest) -> bool:
        observation = self.observe()
        obj = next((obj for obj in observation.ground_truth if obj.object_id == task.target_object_id), None)
        check(obj is not None, f"unknown target: {task.target_object_id}")
        return observation.held_object_id is None and np.linalg.norm(
            np.array(obj.pose.position) - task.destination.position) <= .005

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.stop()
        self.closed = True
