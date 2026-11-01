"""실제 시뮬레이터에서 캡처한 한 시점의 공용 개발 입력 저장/로드."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
from PIL import Image

from .mock_data import ROOT
from .types import Contact, Edge, Observation, Pose, SceneGraph, SceneObject, Subgoal, TargetSpec, TaskRequest, check


@dataclass
class Sample:
    observation: Observation
    graph: SceneGraph
    task: TaskRequest
    subgoal: Subgoal


def save_sample(directory: str | Path, sample: Sample, *, config: dict | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    obs = sample.observation
    np.savez_compressed(directory / "arrays.npz", rgb=obs.rgb, robot_state=obs.robot_state,
                        depth=obs.depth if obs.depth is not None else np.empty(0, dtype=np.float32),
                        camera_intrinsics=obs.camera_intrinsics, camera_to_robot_base=obs.camera_to_robot_base)
    if obs.rgb_valid:
        Image.fromarray(obs.rgb).save(directory / "rgb.png")
    metadata = {"schema_version": "0.3", "timestamp": obs.timestamp, "episode_id": obs.episode_id,
                "source": obs.source, "frame_id": obs.frame_id, "robot_id": obs.robot_id,
                "control_dt": obs.control_dt, "rgb_valid": obs.rgb_valid,
                "held_object_id": obs.held_object_id,
                "objects": [asdict(obj) for obj in obs.ground_truth] if obs.ground_truth is not None else None,
                "contacts": [asdict(contact) for contact in obs.contacts],
                "perceived_objects": [asdict(obj) for obj in obs.perceived_objects],
                "finger_contacts": list(obs.finger_contacts),
                "ee_pose": asdict(obs.ee_pose) if obs.ee_pose else None,
                "graph": asdict(sample.graph), "task": asdict(sample.task), "subgoal": asdict(sample.subgoal),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
                "config": config,
                "model_commit": json.loads((ROOT / "assets/so101/source.json").read_text(encoding="utf-8"))["commit"]}
    (directory / "sample.json").write_text(json.dumps(metadata, indent=2, allow_nan=False,
                                                    default=lambda value: value.item()) + "\n", encoding="utf-8")


def load_sample(directory: str | Path) -> Sample:
    directory = Path(directory)
    d = json.loads((directory / "sample.json").read_text(encoding="utf-8"))
    check(d["schema_version"] == "0.3", "unsupported sample schema")

    def pose(value):
        return Pose(tuple(value["position"]), tuple(value["quaternion_xyzw"]))

    def obj(value):
        return SceneObject(value["object_id"], pose(value["pose"]), tuple(value["size"]),
                           value["label"], value["parent_id"],
                           tuple(value['color_rgb']) if value.get('color_rgb') is not None else None)

    contacts = tuple(Contact(c["entity_a"], c["entity_b"], tuple(c["position"]), tuple(c["normal_a_to_b"]),
                             c["normal_force"]) for c in d["contacts"])
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        observation = Observation(arrays["rgb"], arrays["robot_state"],
                                  tuple(obj(o) for o in d["objects"]) if d["objects"] is not None else None,
                                  d["held_object_id"], d["timestamp"], d["episode_id"], d["frame_id"],
                                  d["source"], d["robot_id"], d["control_dt"],
                                  arrays["depth"] if arrays["depth"].size else None,
                                  arrays["camera_intrinsics"], arrays["camera_to_robot_base"], contacts,
                                  pose(d["ee_pose"]) if d["ee_pose"] else None, d["rgb_valid"],
                                  tuple(obj(o) for o in d.get('perceived_objects', [])),
                                  tuple(d.get('finger_contacts', [False, False])))
    g, task, goal = d["graph"], d["task"], d["subgoal"]
    graph = SceneGraph(tuple(obj(o) for o in g["nodes"]), tuple(Edge(**e) for e in g["edges"]),
                       g["groups"], g["held_object_id"], g["timestamp"], g["episode_id"], g["frame_id"], g["source"],
                       g.get('regions', {}))
    return Sample(observation, graph, TaskRequest(task["target_object_id"], pose(task["destination"]), pose(task["buffer_pose"]),
                  TargetSpec(**task['target_spec']) if task.get('target_spec') else None),
                  Subgoal(goal["subgoal_id"], goal["primitive"], goal["object_id"], pose(goal["object_pose"]),
                          pose(goal["target_ee_pose"]), pose(goal["place_pose"]) if goal["place_pose"] else None, goal["frame_id"]))
