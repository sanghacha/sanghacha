"""각 모듈이 독립적으로 불러 쓰는 고정 예시. 실제 학습 데이터가 아니다."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from .types import Edge, MOCK_GRASP_OFFSET, Observation, Pose, SceneGraph, SceneObject, Subgoal, TaskRequest, check

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MockData:
    observation: Observation
    graph: SceneGraph
    task: TaskRequest
    expected_order: tuple[tuple[str, str], ...]


def make_mock_data(case: str = "blocked") -> MockData:
    data = json.loads((ROOT / "configs/io/mock_cases.json").read_text(encoding="utf-8"))
    check(data["schema_version"] == "0.2", "unsupported mock schema")
    check(case in data["cases"], f"unknown case: {case}")
    scene = data["cases"][case]
    objects = tuple(SceneObject(key, Pose(tuple(position)), tuple(data["object_size"]))
                    for key, position in scene["objects"].items())
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:8, :, 0], rgb[8:, :, 2] = 180, 180
    observation = Observation(rgb, np.array([0, 0, 3, 0, 0, 0], dtype=np.float32), objects, rgb_valid=False)
    groups = {"workspace": None, "bin": "workspace"}
    groups.update({parent: "bin" for parent in scene["parents"].values()})
    nodes = tuple(replace(obj, parent_id=scene["parents"][obj.object_id]) for obj in objects)
    graph = SceneGraph(nodes, tuple(Edge(*edge) for edge in scene["edges"]), groups)
    task = TaskRequest(scene.get("target_object_id", "B"), Pose(tuple(data["destination"])),
                       Pose(tuple(data["buffer_position"])))
    return MockData(observation, graph, task, tuple(tuple(pair) for pair in scene["order"]))


def make_policy_input(primitive: str = "PICK", case: str = "blocked") -> tuple[Observation, Subgoal]:
    """PLACE에는 실제로 이미 물체를 잡고 있는 mock 관측을 제공한다."""
    check(primitive in ("PICK", "PLACE"), "unknown primitive")
    fixture = make_mock_data(case)
    check(bool(fixture.expected_order), "missing target has no valid policy input")
    obj = fixture.graph.object(fixture.expected_order[0][1])
    destination = fixture.task.destination if obj.object_id == fixture.task.target_object_id else fixture.task.buffer_pose
    target = obj.pose if primitive == "PICK" else destination
    x, y, z = target.position
    goal = Subgoal(f"example-{case}-{primitive}", primitive, obj.object_id, obj.pose,
                   Pose((x, y, z + MOCK_GRASP_OFFSET)), destination if primitive == "PLACE" else None)
    observation = fixture.observation
    if primitive == "PLACE":
        state = observation.robot_state.copy()
        state[:3] = np.array(obj.pose.position) / 0.1 + [0, 0, MOCK_GRASP_OFFSET / 0.1]
        state[5] = 1
        observation = replace(observation, robot_state=state, held_object_id=obj.object_id)
    return observation, goal
