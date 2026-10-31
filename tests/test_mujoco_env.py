import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import mujoco

from action_generator import predict_action_chunk
from hssg_generator import generate_scene_graph
from planner import plan_next_subgoal
from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from pipeline.sample_data import Sample, load_sample, save_sample


@unittest.skipUnless((ROOT / "assets/so101/upstream/so101.xml").exists(), "run scripts/setup_assets.py first")
class MujocoEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = MujocoEnvironment("blocked", render=False)  # CPU physics tests need no OpenGL/GPU.
        self.addCleanup(self.env.close)

    def test_real_model_joint_mapping_contacts_and_graph(self):
        observation = self.env.reset(seed=42)
        self.assertEqual((self.env.model.nu, observation.robot_id, observation.source), (6, "so101", "mujoco_gt"))
        self.assertFalse(observation.rgb_valid)
        np.testing.assert_allclose(observation.robot_state[:5], self.env.data.qpos[self.env.qadr[:5]])
        self.assertEqual({o.object_id for o in observation.ground_truth}, {"A", "B", "C"})
        graph = generate_scene_graph(observation)
        self.assertEqual({(e.source, e.relation, e.target) for e in graph.edges},
                         {("A", "on_top_of", "B"), ("B", "supports", "A")})
        self.assertTrue(any({c.entity_a, c.entity_b} == {"A", "B"} and c.normal_force > 0 for c in observation.contacts))
        goal = plan_next_subgoal(graph, self.env.task, observation.robot_state)
        self.assertEqual((goal.primitive, goal.object_id), ("PICK", "A"))

    def test_real_actuation_advances_physics_without_teleporting_objects(self):
        observation = self.env.reset(42)
        command = observation.robot_state.copy()
        command[0] += .03
        after = self.env.step(command)
        self.assertGreater(after.robot_state[0] - observation.robot_state[0], .005)
        self.assertAlmostEqual(after.timestamp, self.env.config["control_dt"], places=8)
        for before_obj, after_obj in zip(observation.ground_truth, after.ground_truth):
            self.assertLess(np.linalg.norm(np.array(before_obj.pose.position) - after_obj.pose.position), .001)
        self.assertIsNone(after.held_object_id)
        self.assertFalse(self.env.task_complete(self.env.task))

    def test_seed_and_case_reproducibility(self):
        first, same, other = self.env.reset(7), self.env.reset(7), self.env.reset(8)
        np.testing.assert_allclose([o.pose.position for o in first.ground_truth], [o.pose.position for o in same.ground_truth])
        self.assertNotEqual(first.ground_truth[0].pose.position, other.ground_truth[0].pose.position)
        another = MujocoEnvironment("after_move", render=False)
        try:
            obs = another.reset()
            self.assertFalse(generate_scene_graph(obs).edges)
        finally:
            another.close()

    def test_invalid_control_is_rejected_before_physics(self):
        self.env.reset()
        for command in (np.full(6, 100., dtype=np.float32), np.zeros(6, dtype=np.float64)):
            with self.assertRaises(ValueError):
                self.env.step(command)
        self.assertEqual(self.env.observe().timestamp, 0.)

    def test_calibration_and_physical_sample_roundtrip(self):
        obs = self.env.reset()
        inverse = np.linalg.inv(obs.camera_to_robot_base)
        for obj in obs.ground_truth:
            p = inverse @ [*obj.pose.position, 1]
            self.assertGreater(p[2], 0)
            uv = obs.camera_intrinsics @ (p[:3] / p[2])
            self.assertTrue(0 < uv[0] < obs.rgb.shape[1] and 0 < uv[1] < obs.rgb.shape[0])
        graph = generate_scene_graph(obs)
        subgoal = plan_next_subgoal(graph, self.env.task, obs.robot_state)
        directory = ROOT / "outputs/test_samples" / uuid4().hex
        save_sample(directory, Sample(obs, graph, self.env.task, subgoal), config=self.env.config)
        loaded = load_sample(directory)
        self.assertEqual(loaded.graph, graph)
        self.assertEqual(loaded.subgoal, subgoal)
        np.testing.assert_array_equal(loaded.observation.robot_state, obs.robot_state)
        self.assertEqual(loaded.observation.contacts, obs.contacts)

    def test_c_starter_uses_real_joint_hold_not_mock_kinematics(self):
        obs = self.env.reset()
        graph = generate_scene_graph(obs)
        goal = plan_next_subgoal(graph, self.env.task, obs.robot_state)
        chunk = predict_action_chunk(obs, goal)
        self.assertEqual(chunk.robot_id, "so101")
        np.testing.assert_array_equal(chunk.commands[0], obs.robot_state)
        after = self.env.step(chunk.commands[0])
        self.assertFalse(self.env.subgoal_complete(goal))
        self.assertIsNone(after.held_object_id)

    def test_stacked_object_must_lift_above_its_own_start_height(self):
        obs = self.env.reset()
        graph = generate_scene_graph(obs)
        goal = plan_next_subgoal(graph, self.env.task, obs.robot_state)
        self.assertTrue(self.env._supported("A"))
        with patch.object(self.env, "_held_object", return_value="A"):
            self.assertFalse(self.env.subgoal_complete(goal))

    def test_three_shelves_have_stable_stacks_and_separate_hierarchies(self):
        env = MujocoEnvironment(render=False)
        try:
            for seed in (0, 7, 42):
                observation = env.reset(seed)
                graph = generate_scene_graph(observation)
                self.assertEqual(len(observation.ground_truth), 27)
                self.assertEqual(len(env.shelf_scene.shelves), 3)
                for shelf in env.shelf_scene.shelves:
                    objects = [obj for obj in observation.ground_truth if obj.parent_id == shelf["id"]]
                    self.assertEqual(len(objects), 9)
                    self.assertEqual(graph.groups[shelf["id"]], "workspace")
                    for obj in objects:
                        self.assertEqual(graph.groups[graph.object(obj.object_id).parent_id], shelf["id"])
                        self.assertGreater(obj.pose.position[2], 0)
                for edge in graph.edges:
                    self.assertEqual(graph.groups[graph.object(edge.source).parent_id],
                                     graph.groups[graph.object(edge.target).parent_id])
                self.assertGreaterEqual(len(graph.edges)//2, 9)
                self.assertTrue(any(e.relation == "on_top_of" and e.target == env.task.target_object_id for e in graph.edges))
                self.assertLess(max(np.linalg.norm(env.data.cvel[body]) for body in env.object_bodies.values()), .01)
        finally:
            env.close()

    def test_shelf_layout_repeats_by_seed_and_changes_without_reassigning_ids(self):
        env = MujocoEnvironment(render=False)
        try:
            first, repeated, another = env.reset(42), env.reset(42), env.reset(43)
            np.testing.assert_allclose([o.pose.position for o in first.ground_truth],
                                       [o.pose.position for o in repeated.ground_truth], atol=1e-10)
            self.assertEqual([o.object_id for o in first.ground_truth], [o.object_id for o in another.ground_truth])
            self.assertFalse(np.allclose([o.pose.position for o in first.ground_truth],
                                         [o.pose.position for o in another.ground_truth]))
            self.assertGreater(len({o.size for o in first.ground_truth}), 1)
        finally:
            env.close()

    def test_empty_placement_trays_support_objects_at_the_correct_goal_height(self):
        env = self.env
        for region, goal_name in (("buffer_tray", "buffer_pose"), ("output_tray", "destination")):
            env.reset(42)
            goal = getattr(env.task, goal_name)
            self.assertAlmostEqual(goal.position[2], env.config["placement_trays"]["floor_thickness"] + .015)
            address = env.object_qadr["B"]
            # 환경 초기 조건으로 박스를 선반 위에서 떨어뜨려 바닥 충돌/완료 판정을 검사한다.
            env.data.qpos[address:address+3] = np.array(goal.position) + [0., 0., .05]
            env.data.qpos[address+3:address+7] = [1., 0., 0., 0.]
            joint_id = env.model.body_jntadr[env.object_bodies["B"]]
            dof = env.model.jnt_dofadr[joint_id]
            env.data.qvel[dof:dof+6] = 0
            mujoco.mj_forward(env.model, env.data)
            for _ in range(30):
                observation = env.step(env.joint_state())
            obj = next(obj for obj in observation.ground_truth if obj.object_id == "B")
            self.assertAlmostEqual(obj.pose.position[2], goal.position[2], delta=.001)
            self.assertEqual(obj.parent_id, region)
            self.assertTrue(env._supported("B", region))
            self.assertEqual(env.task_complete(env.task), region == "output_tray")


if __name__ == "__main__":
    unittest.main()
