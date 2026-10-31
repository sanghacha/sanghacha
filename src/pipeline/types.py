"""공통 입출력 v0.3. robot_base / m / quaternion xyzw."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FRAME = "robot_base"
MOCK_GRASP_OFFSET = 0.08
Primitive = Literal["PICK", "PLACE"]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _vector(value, length: int) -> None:
    check(len(value) == length and bool(np.isfinite(value).all()), f"expected {length} finite values")


def _array(value: NDArray, shape: tuple[int, ...], dtype: str) -> NDArray:
    check(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == np.dtype(dtype),
          f"expected {dtype} array with shape {shape}")
    check(bool(np.isfinite(value).all()), "array contains NaN/Inf")
    result = value.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float] = (0., 0., 0., 1.)

    def __post_init__(self):
        _vector(self.position, 3)
        _vector(self.quaternion_xyzw, 4)
        check(bool(np.isclose(np.linalg.norm(self.quaternion_xyzw), 1.)), "quaternion must be unit xyzw")


def wxyz_to_xyzw(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = value
    return Pose((0., 0., 0.), (x, y, z, w)).quaternion_xyzw


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    pose: Pose
    size: tuple[float, float, float] = (0.06, 0.06, 0.06)
    label: str = "box"
    parent_id: str | None = None

    def __post_init__(self):
        check(bool(self.object_id) and bool(self.label), "missing object ID/label")
        _vector(self.size, 3)
        check(all(v > 0 for v in self.size), "object size must be positive")


@dataclass(frozen=True)
class Contact:
    entity_a: str
    entity_b: str
    position: tuple[float, float, float]
    normal_a_to_b: tuple[float, float, float]
    normal_force: float

    def __post_init__(self):
        check(bool(self.entity_a) and bool(self.entity_b), "missing contact entities")
        _vector(self.position, 3)
        _vector(self.normal_a_to_b, 3)
        check(np.isclose(np.linalg.norm(self.normal_a_to_b), 1.), "contact normal must be unit length")
        check(np.isfinite(self.normal_force) and self.normal_force >= 0, "invalid contact force")


@dataclass(frozen=True)
class Observation:
    rgb: NDArray[np.uint8]
    robot_state: NDArray[np.float32]
    ground_truth: tuple[SceneObject, ...] | None = None
    held_object_id: str | None = None
    timestamp: float = 0.
    episode_id: str = "mock-0"
    frame_id: str = FRAME
    source: str = "mock_gt"
    robot_id: str = "mock"
    control_dt: float = 1 / 30
    depth: NDArray[np.float32] | None = None
    camera_intrinsics: NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    camera_to_robot_base: NDArray[np.float64] = field(default_factory=lambda: np.eye(4))
    contacts: tuple[Contact, ...] = ()
    ee_pose: Pose | None = None
    rgb_valid: bool = True  # False: 물리/GT 전용 실행. RGB를 정책/인식 입력으로 쓰지 않는다.

    def __post_init__(self):
        check(self.frame_id == FRAME, "observation must use robot_base")
        check(bool(self.episode_id) and bool(self.robot_id), "missing episode/robot ID")
        check(np.isfinite(self.timestamp) and self.timestamp >= 0, "invalid timestamp")
        check(np.isfinite(self.control_dt) and self.control_dt > 0, "invalid control_dt")
        check(self.source in ("mock_gt", "mujoco_gt", "rgbd"), "invalid observation source")
        check(isinstance(self.rgb, np.ndarray) and self.rgb.ndim == 3 and
              self.rgb.shape[2] == 3 and min(self.rgb.shape[:2]) > 0, "RGB must be HWC")
        object.__setattr__(self, "rgb", _array(self.rgb, self.rgb.shape, "uint8"))
        object.__setattr__(self, "robot_state", _array(self.robot_state, (6,), "float32"))
        k = _array(self.camera_intrinsics, (3, 3), "float64")
        t = _array(self.camera_to_robot_base, (4, 4), "float64")
        check(k[0, 0] > 0 and k[1, 1] > 0 and np.allclose(k[2], [0, 0, 1]), "invalid intrinsics")
        check(np.allclose(t[3], [0, 0, 0, 1]) and
              np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3)) and
              np.isclose(np.linalg.det(t[:3, :3]), 1.), "invalid camera transform")
        object.__setattr__(self, "camera_intrinsics", k)
        object.__setattr__(self, "camera_to_robot_base", t)
        if self.depth is not None:
            d = _array(self.depth, self.rgb.shape[:2], "float32")
            check(bool((d >= 0).all()), "depth must be nonnegative meters")
            object.__setattr__(self, "depth", d)
        if self.ground_truth is not None:
            ids = [obj.object_id for obj in self.ground_truth]
            check(len(ids) == len(set(ids)), "duplicate ground-truth object IDs")
            check(self.source != "rgbd", "RGB-D observations must not expose GT")
            check(self.held_object_id is None or self.held_object_id in ids, "unknown held object")


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str

    def __post_init__(self):
        check(self.source != self.target, "self edge is not allowed")
        check(self.relation in ("on_top_of", "supports", "occludes", "inside"), "unknown relation")


@dataclass(frozen=True)
class SceneGraph:
    nodes: tuple[SceneObject, ...]
    edges: tuple[Edge, ...]
    groups: dict[str, str | None]
    held_object_id: str | None = None
    timestamp: float = 0.
    episode_id: str = "mock-0"
    frame_id: str = FRAME
    source: str = "mock_gt"

    def __post_init__(self):
        check(self.frame_id == FRAME, "graph must use robot_base")
        check(self.source in ("mock_gt", "mujoco_gt", "rgbd"), "invalid graph source")
        check(bool(self.episode_id) and np.isfinite(self.timestamp) and self.timestamp >= 0, "invalid graph metadata")
        ids = [obj.object_id for obj in self.nodes]
        check(len(ids) == len(set(ids)) and not set(ids) & self.groups.keys(), "duplicate graph IDs")
        check(all(obj.parent_id in self.groups for obj in self.nodes), "missing object parent")
        check(self.held_object_id is None or self.held_object_id in ids, "unknown held object")
        for group_id in self.groups:
            seen = set()
            current = group_id
            while current is not None:
                check(current in self.groups and current not in seen, "missing parent or hierarchy cycle")
                seen.add(current)
                current = self.groups[current]
        endpoints = set(ids) | self.groups.keys()
        check(all(e.source in endpoints and e.target in endpoints for e in self.edges), "unknown edge endpoint")
        check(len(set(self.edges)) == len(self.edges), "duplicate edge")

    def object(self, object_id: str) -> SceneObject:
        for obj in self.nodes:
            if obj.object_id == object_id:
                return obj
        raise ValueError(f"unknown object: {object_id}")


@dataclass(frozen=True)
class TaskRequest:
    target_object_id: str
    destination: Pose
    buffer_pose: Pose

    def __post_init__(self):
        check(bool(self.target_object_id), "missing target object ID")


@dataclass(frozen=True)
class Subgoal:
    subgoal_id: str
    primitive: Primitive
    object_id: str
    object_pose: Pose
    target_ee_pose: Pose
    place_pose: Pose | None = None
    frame_id: str = FRAME

    def __post_init__(self):
        check(bool(self.subgoal_id) and bool(self.object_id), "missing subgoal/object ID")
        check(self.primitive in ("PICK", "PLACE"), "unknown primitive")
        check(self.frame_id == FRAME, "subgoal must use robot_base")
        check((self.primitive == "PLACE") == (self.place_pose is not None), "PLACE alone requires place_pose")


@dataclass(frozen=True)
class ActionChunk:
    subgoal_id: str
    commands: NDArray[np.float32]
    command_dt: float = 1 / 30
    robot_id: str = "mock"

    def __post_init__(self):
        check(bool(self.subgoal_id) and bool(self.robot_id), "missing action IDs")
        check(np.isfinite(self.command_dt) and self.command_dt > 0, "invalid command_dt")
        check(isinstance(self.commands, np.ndarray) and self.commands.ndim == 2 and
              self.commands.shape[0] > 0 and self.commands.shape[1] == 6, "commands must be (K, 6), K > 0")
        object.__setattr__(self, "commands", _array(self.commands, self.commands.shape, "float32"))
