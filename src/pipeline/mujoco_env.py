"""공식 SO-101 MJCF로 구성한 공통 MuJoCo 개발 환경. 실제 하드웨어 접속 없음."""

import json
from pathlib import Path

import mujoco
import numpy as np

from .mock_data import ROOT
from .shelf_scene import ShelfScene, add_empty_tray
from .types import Contact, Observation, Pose, SceneObject, Subgoal, TaskRequest, check, wxyz_to_xyzw


class MujocoEnvironment:
    def __init__(self, case: str = "shelves", config_path: str | Path = ROOT / "configs/envs/so101.json",
                 *, render: bool = True):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        check(case in ("shelves", "blocked", "accessible", "after_move"), "unknown MuJoCo case")
        check(self.config["backend"] == "mujoco" and self.config["robot_id"] == "so101", "invalid backend")
        self.case, self.render_enabled = case, render
        self.renderer = None
        self.closed, self.stopped = False, False
        self._reset_done = False
        self._episode = ""
        self._start_time = 0.
        tray_config = self.config["placement_trays"]
        self.placement_trays = {
            tray_id: {"center": self.config[position][:2], "size": tray_config["size"]}
            for tray_id, position in (("buffer_tray", "buffer_position"), ("output_tray", "destination"))}
        self.shelf_scene = ShelfScene(self.config["shelf_scene"]) if case == "shelves" else None
        colors = ([.95, .18, .15, 1], [.15, .35, .95, 1], [.95, .8, .15, 1])
        self.object_definitions = self.shelf_scene.objects if self.shelf_scene else {
            key: {"position": position, "size": (self.config["cube_size"],)*3,
                  "mass": self.config["cube_mass"], "color": color}
            for (key, position), color in zip(self.config["objects"].items(), colors)}
        if self.shelf_scene:
            for key in ("home", "camera_position", "camera_target", "target_object_id"):
                self.config[key] = self.config["shelf_scene"][key]
        model_path = ROOT / "assets/so101/upstream/so101.xml"
        if not model_path.exists():
            raise FileNotFoundError("SO-101 assets missing. Run: uv run python scripts/setup_assets.py")
        self.spec = mujoco.MjSpec.from_file(str(model_path))
        self._build_scene()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.joints = tuple(self.config["joint_names"])
        check(len(self.joints) == 6, "SO-101 needs five arm joints and gripper")
        self.qadr = np.array([self.model.joint(name).qposadr[0] for name in self.joints])
        self.actuators = np.array([self.model.actuator(name).id for name in self.joints])
        limits = self.model.actuator_ctrlrange[self.actuators].copy()
        self.raw_lower, self.raw_upper = limits[:, 0], limits[:, 1]
        self.lower, self.upper = self.raw_lower.copy(), self.raw_upper.copy()
        self.lower[5], self.upper[5] = 0., 1.
        self.object_bodies = {key: self.model.body(f"object_{key}").id for key in self.object_definitions}
        self.object_qadr = {key: int(self.model.jnt_qposadr[self.model.body_jntadr[body]])
                            for key, body in self.object_bodies.items()}
        self.object_geoms = {self.model.geom(f"object_{key}").id: key for key in self.object_bodies}
        shelf_ids = [s["id"] for s in self.shelf_scene.shelves] if self.shelf_scene else []
        shelf_ids += list(self.placement_trays)
        self.shelf_geoms = {g: shelf_id for shelf_id in shelf_ids for g in range(self.model.ngeom)
                            if self.model.geom(g).name.startswith(shelf_id + "_")}
        self.ee_site = self.model.site("gripperframe").id
        self.camera_id = self.model.camera(self.config["camera_name"]).id
        self.fixed_jaw_body = self.model.body("gripper").id
        self.moving_jaw_body = self.model.body("moving_jaw_so101_v1").id
        destination, buffer = list(self.config["destination"]), list(self.config["buffer_position"])
        # 목표는 선반 바닥 위에 놓인 물체의 중심이다.
        destination[2] = buffer[2] = (tray_config["floor_thickness"] +
                                    self.object_definitions[self.config["target_object_id"]]["size"][2]/2)
        self.task = TaskRequest(self.config["target_object_id"], Pose(tuple(destination)), Pose(tuple(buffer)))
        self._validate_config()

    def _validate_config(self):
        for name in ("physics_substeps", "image_width", "image_height", "n_action_steps",
                     "max_subgoals", "max_steps_per_subgoal"):
            check(type(self.config[name]) is int and self.config[name] > 0, f"invalid {name}")
        check(self.config["control_dt"] > 0 and np.isfinite(self.config["control_dt"]), "invalid control_dt")
        check(np.asarray(self.config["max_delta"]).shape == (6,) and
              np.all(np.array(self.config["max_delta"]) > 0), "invalid max_delta")
        home = np.asarray(self.config["home"])
        check(home.shape == (6,) and np.isfinite(home).all() and
              np.all(home >= self.lower) and np.all(home <= self.upper), "invalid home")

    def _build_scene(self):
        c = self.config
        self.spec.option.timestep = c["control_dt"] / c["physics_substeps"]
        self.spec.visual.global_.offwidth = c["image_width"]
        self.spec.visual.global_.offheight = c["image_height"]
        world = self.spec.worldbody
        world.add_light(name="overhead", pos=[0.2, -0.2, 1.5], dir=[0, 0, -1],
                        diffuse=[.8, .8, .8])
        world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[.15, 0, -.025],
                       size=[.4, .35, .025], rgba=[.72, .71, .66, 1], friction=[1, .005, .0001])
        tray_config = c["placement_trays"]
        for tray_id, color_key in (("buffer_tray", "buffer_color"), ("output_tray", "destination_color")):
            tray = self.placement_trays[tray_id]
            add_empty_tray(world, tray_id, tray["center"], tray["size"],
                           tray_config["floor_thickness"], tray_config["wall_thickness"], tray_config["wall_height"],
                           tray_config[color_key], tray_config["floor_color"])
        if self.shelf_scene:
            self.shelf_scene.add_shelves(world)
        for key, obj in self.object_definitions.items():
            body = world.add_body(name=f"object_{key}", pos=obj["position"], quat=obj.get("quaternion", [1, 0, 0, 0]))
            body.add_freejoint(name=f"object_{key}_free")
            body.add_geom(name=f"object_{key}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=np.array(obj["size"])/2, mass=obj["mass"], rgba=obj["color"],
                          friction=[1, .01, .001], condim=4)
        position, target = np.array(c["camera_position"]), np.array(c["camera_target"])
        z = position - target
        z = z / np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z)
        x /= np.linalg.norm(x)
        rotation = np.column_stack((x, np.cross(z, x), z))
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        world.add_camera(name=c["camera_name"], pos=position, quat=quaternion, fovy=c["camera_fovy"])
        # RGB-D acquisition cameras remain fixed independently of arm movement.
        for name, position, target, fovy in (
            ("fixed_top", [.24, 0., .75], [.24, 0., .02], 50.),
            ("fixed_oblique", [.58, .48, .48], [.22, 0., .04], 47.),
        ):
            z = np.asarray(position) - target
            z /= np.linalg.norm(z)
            x = np.cross([0., 1., 0.], z)
            x /= np.linalg.norm(x)
            rotation = np.column_stack((x, np.cross(z, x), z))
            mujoco.mju_mat2Quat(quaternion, rotation.ravel())
            world.add_camera(name=name, pos=position, quat=quaternion, fovy=fovy)

    def _to_ctrl(self, command):
        result = np.array(command, dtype=np.float64)
        # 공식 MJCF의 gripper hinge: 작은 각도=닫힘, 큰 각도=열림.
        result[5] = self.raw_upper[5] - command[5] * (self.raw_upper[5] - self.raw_lower[5])
        return result

    def joint_state(self):
        result = self.data.qpos[self.qadr].copy()
        result[5] = np.clip((self.raw_upper[5] - result[5]) /
                            (self.raw_upper[5] - self.raw_lower[5]), 0, 1)
        return result.astype(np.float32)

    def reset(self, seed: int = 0, *, privileged: bool = True) -> Observation:
        check(not self.closed and type(seed) is int and seed >= 0, "invalid reset")
        mujoco.mj_resetData(self.model, self.data)
        self._episode, self.stopped = f"so101-{self.case}-{seed}", False
        self.data.qpos[self.qadr] = self._to_ctrl(self.config["home"])
        self.data.ctrl[self.actuators] = self.data.qpos[self.qadr]
        rng = np.random.default_rng(seed)
        if self.shelf_scene:
            layout = self.shelf_scene.sample_layout(rng)
        else:
            layout = {}
        shared_offset = rng.uniform(-self.config["random_xy"], self.config["random_xy"], 2)
        for key, address in self.object_qadr.items():
            if self.shelf_scene:
                position, quaternion = layout[key]
            else:
                position = np.array(self.config["objects"][key], dtype=float)
                if key == "A" and self.case != "blocked":
                    position = np.array(self.task.buffer_pose.position, dtype=float)
                    if self.case == "accessible":
                        # 소형 회귀 장면에서 A는 빈 선반 테두리와 겹치지 않는 테이블에 둔다.
                        position[1], position[2] = -.07, self.object_definitions[key]["size"][2]/2
                else:
                    position[:2] += shared_offset
                quaternion = [1, 0, 0, 0]
            self.data.qpos[address:address+3] = position
            self.data.qpos[address+3:address+7] = quaternion
        mujoco.mj_forward(self.model, self.data)
        for _ in range(round(self.config["settle_seconds"] / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self._start_time, self._reset_done = self.data.time, True
        return self.observe(privileged=privileged)

    def _contacts(self) -> tuple[Contact, ...]:
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            entities = []
            for geom in contact.geom:
                entities.append(self.object_geoms.get(int(geom)) or self.shelf_geoms.get(int(geom)) or
                                self.model.geom(int(geom)).name or f"geom_{geom}")
            if not any(int(geom) in self.object_geoms for geom in contact.geom):
                continue
            force = np.empty(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            contacts.append(Contact(*entities, tuple(contact.pos), tuple(contact.frame[:3]), max(0., float(force[0]))))
        return tuple(contacts)

    def _held_object(self):
        for key, object_body in self.object_bodies.items():
            fingers = set()
            for contact in self.data.contact:
                bodies = {int(self.model.geom_bodyid[int(g)]) for g in contact.geom}
                if object_body in bodies:
                    fingers.update(bodies & {self.fixed_jaw_body, self.moving_jaw_body})
            lifted = self.data.xpos[object_body, 2] > self.object_definitions[key]["size"][2]/2 + .01
            if len(fingers) == 2 and lifted and not self._supported(key):
                return key
        return None

    def _supported(self, key, required_entity=None):
        scene_entities = ({required_entity} if required_entity else
                          set(self.object_bodies) | {"table"} | set(self.shelf_geoms.values()))
        return any(c.normal_force > 1e-5 and (
            (c.entity_a == key and c.entity_b in scene_entities and c.normal_a_to_b[2] < -.7) or
            (c.entity_b == key and c.entity_a in scene_entities and c.normal_a_to_b[2] > .7)
        ) for c in self._contacts())

    def _placement_region(self, position):
        for tray_id, tray in self.placement_trays.items():
            if all(abs(p-c) <= s/2 for p, c, s in zip(position[:2], tray["center"], tray["size"])):
                return tray_id
        return None

    def _object_region(self, position):
        return self._placement_region(position) or (self.shelf_scene.region_of(position) if self.shelf_scene else None)

    def observe(self, *, privileged: bool = True) -> Observation:
        check(self._reset_done and not self.closed, "call reset first")
        c = self.config
        rgb = np.zeros((c["image_height"], c["image_width"], 3), dtype=np.uint8)
        depth = None
        if self.render_enabled:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=c["image_height"], width=c["image_width"])
            self.renderer.update_scene(self.data, camera=c["camera_name"])
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            try:
                depth = self.renderer.render().astype(np.float32)
            finally:
                self.renderer.disable_depth_rendering()
        objects = tuple(SceneObject(key, Pose(tuple(self.data.xpos[body]),
                                              wxyz_to_xyzw(tuple(self.data.xquat[body]))),
                                    tuple(self.object_definitions[key]["size"]),
                                    parent_id=self._object_region(self.data.xpos[body]))
                        for key, body in self.object_bodies.items()) if privileged else None
        ee = Pose(tuple(self.data.site_xpos[self.ee_site]), self._site_quaternion())
        fy = c["image_height"] / (2 * np.tan(np.deg2rad(self.model.cam_fovy[self.camera_id])/2))
        k = np.array([[fy, 0, c["image_width"]/2], [0, fy, c["image_height"]/2], [0, 0, 1.]])
        transform = np.eye(4)
        transform[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        transform[:3, 3] = self.data.cam_xpos[self.camera_id]
        return Observation(rgb, self.joint_state(), objects, self._held_object() if privileged else None,
                           timestamp=float(self.data.time - self._start_time), episode_id=self._episode,
                           source="mujoco_gt" if privileged else 'rgbd', robot_id="so101", control_dt=c["control_dt"], depth=depth,
                           camera_intrinsics=k, camera_to_robot_base=transform,
                           contacts=self._contacts() if privileged else (), ee_pose=ee, rgb_valid=self.render_enabled)

    def _site_quaternion(self):
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.ee_site])
        return wxyz_to_xyzw(tuple(q))

    def contains(self, position):
        check(len(position) == 3 and np.isfinite(position).all(), "invalid position")
        return all(lo <= p <= hi for lo, p, hi in zip(self.config["workspace_min"], position, self.config["workspace_max"]))

    def validate_command(self, command):
        check(isinstance(command, np.ndarray) and command.shape == (6,) and command.dtype == np.float32 and
              np.isfinite(command).all(), "command must be finite float32[6]")
        check(np.all(command >= self.lower) and np.all(command <= self.upper), "command outside joint limits")
        check(np.all(np.abs(command.astype(float) - self.joint_state()) <= np.array(self.config["max_delta"]) + 1e-6),
              "command exceeds per-step delta limit")

    def step(self, command: np.ndarray, *, privileged: bool = True) -> Observation:
        check(self._reset_done and not self.closed and not self.stopped, "reset environment before stepping")
        self.validate_command(command)
        self.data.ctrl[self.actuators] = self._to_ctrl(command)
        for _ in range(self.config["physics_substeps"]):
            mujoco.mj_step(self.model, self.data)
        check(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all(), "nonfinite simulation state")
        return self.observe(privileged=privileged)

    def subgoal_complete(self, subgoal: Subgoal) -> bool:
        if subgoal.primitive == "PICK":
            body = self.object_bodies[subgoal.object_id]
            return self._held_object() == subgoal.object_id and self.data.xpos[body, 2] > subgoal.object_pose.position[2] + .01
        body = self.object_bodies[subgoal.object_id]
        slow = np.linalg.norm(self.data.cvel[body]) < .08
        return self._held_object() is None and self._supported(subgoal.object_id, self._placement_region(subgoal.place_pose.position)) and self.joint_state()[5] < .25 and slow and np.linalg.norm(
            self.data.xpos[body] - subgoal.place_pose.position) < .01

    def task_complete(self, task: TaskRequest) -> bool:
        check(task.target_object_id in self.object_bodies, "unknown target")
        body = self.object_bodies[task.target_object_id]
        return self._held_object() is None and self._supported(task.target_object_id, self._placement_region(task.destination.position)) and self.joint_state()[5] < .25 and np.linalg.norm(self.data.cvel[body]) < .08 and np.linalg.norm(
            self.data.xpos[body] - task.destination.position) < .01

    def stop(self):
        # 시뮬레이터만 제어하며 실물 모터와 통신하지 않는다.
        if not self.closed:
            self.data.ctrl[self.actuators] = self.data.qpos[self.qadr]
        self.stopped = True

    def close(self):
        self.stop()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.closed = True
