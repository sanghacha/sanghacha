from dataclasses import replace
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch

from action_generator.goal import GOAL_DIM, GOAL_VERSION, encode_goal
from action_generator.generator import _load_act, predict_action_chunk
from pipeline.types import Observation, Pose, Subgoal
from pipeline.mock_data import ROOT
from so101_act.data import (ChunkDataset, SegmentWriter, compute_stats, discover_segments,
                            read_segment, split_segments)
from so101_act.model import ACTConfig, GoalConditionedACT, act_loss
from so101_act.policy import ACTPolicy
from so101_act.train import CHECKPOINT_VERSION, DEFAULT_CONFIG, train_run

torch.set_num_threads(2)


def inputs():
    observation = Observation(np.full((32, 32, 3), 80, np.uint8), np.zeros(6, np.float32),
                              robot_id='so101', source='rgbd', episode_id='test-0')
    goal = Subgoal('pick-1', 'PICK', 'track_007', Pose((.2, 0., .03)), Pose((.2, 0., .1)))
    return observation, goal


def control():
    return {'lower': [-2.]*5+[0.], 'upper': [2.]*5+[1.], 'max_delta': [.06]*5+[.1]}


def model_config():
    return ACTConfig(image_size=(32, 32), chunk_size=4, hidden_dim=32, latent_dim=4,
                     encoder_layers=1, decoder_layers=1, posterior_layers=1, feedforward_dim=64, dropout=0.)


def identity_stats():
    return {'states': {'mean': [0.]*6, 'std': [1.]*6},
            'actions': {'mean': [0.]*6, 'std': [1.]*6}, 'goal_scale': [1.]*GOAL_DIM}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT/'outputs/test_act'/uuid4().hex
        self.root.mkdir(parents=True)

    def segment(self, group='seed-0', *, success=True, diagnostic=False):
        observation, goal = inputs()
        writer = SegmentWriter(self.root, observation, goal, group_id=group, image_size=(32, 32),
                               diagnostic=diagnostic, **control())
        for t in range(3):
            writer.append(replace(observation, timestamp=t/30), np.full(6, .01, np.float32))
        return writer.finish(success=success, reason='unit_test_fixture')

    def test_failed_and_diagnostic_data_are_excluded_by_default(self):
        self.segment(success=False)
        self.segment(diagnostic=True, success=False)
        with self.assertRaisesRegex(ValueError, 'no successful'):
            discover_segments(self.root)
        accepted = self.segment()
        self.assertEqual([s['path'] for s in discover_segments(self.root)], [accepted])
        self.assertEqual(len(discover_segments(self.root, allow_diagnostic=True)), 2)

    def test_windows_pad_at_subgoal_end_and_keep_pre_action_alignment(self):
        segment = read_segment(self.segment())
        dataset = ChunkDataset([segment], identity_stats(), model_config())
        first, last = dataset[0], dataset[2]
        np.testing.assert_allclose(first['state'], 0.)
        np.testing.assert_allclose(first['actions'][:3], .01)
        np.testing.assert_array_equal(first['padding'], [False, False, False, True])
        np.testing.assert_array_equal(last['padding'], [False, True, True, True])
        np.testing.assert_array_equal(last['actions'][1:], 0.)

    def test_velocity_condition_uses_only_previous_measured_state(self):
        segment = read_segment(self.segment())
        segment['states'][1] = .03
        segment['states'][2] = .05
        config = model_config()
        config.include_velocity = True
        data = ChunkDataset([segment], identity_stats(), config)
        np.testing.assert_allclose(data[0]['state'][6:], 0.)
        np.testing.assert_allclose(data[1]['state'][6:], np.array([.03]*6)/control()['max_delta'])
        segment['states'][2] = 1.
        np.testing.assert_allclose(data[1]['state'][6:], np.array([.03]*6)/control()['max_delta'])

    def test_multiple_roots_keep_calibration_checks_and_reject_overlap(self):
        first = self.segment('seed-0')
        second = self.segment('seed-1')
        self.assertEqual(len(discover_segments([first, second])), 2)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            discover_segments([self.root, first])
        path = second/'metadata.json'
        meta = json.loads(path.read_text())
        meta['camera_to_robot_base'][0][3] = .1
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'mixed camera'):
            discover_segments([first, second])

    def test_recovery_prefix_keeps_original_place_goal_and_records_only_teacher_actions(self):
        from types import SimpleNamespace
        from pipeline.types import ActionChunk, TaskRequest
        from so101_act.generate import generate_episode
        observation, pick = inputs()
        observation = replace(observation, ee_pose=Pose((.2, 0., .08)))
        goal = replace(pick, primitive='PLACE', place_pose=Pose((.08, .18, .03)))
        moved = replace(observation, timestamp=1/30, ee_pose=Pose((.21, 0., .10)))
        fake = SimpleNamespace(config={}, task=TaskRequest(goal.object_id, goal.place_pose, Pose((.08, -.18, .03))),
            lower=control()['lower'], upper=control()['upper'], close=lambda: None,
            subgoal_complete=lambda g: True, task_complete=lambda t: True)
        fake.config['max_delta'] = control()['max_delta']
        learner = SimpleNamespace(predict_action_chunk=lambda o, g: ActionChunk(g.subgoal_id,
            np.full((1, 6), .01, np.float32), robot_id='so101'))
        teacher = SimpleNamespace(predict_action_chunk=lambda o, g: ActionChunk(g.subgoal_id,
            np.full((1, 6), .02, np.float32), robot_id='so101'))
        def run(env, task, *, predict_action, on_step, **kwargs):
            prefix = predict_action(observation, goal)
            on_step(observation, prefix.commands[0], moved, goal)
            action = predict_action(moved, goal)
            on_step(moved, action.commands[0], replace(moved, timestamp=2/30), goal)
            return [goal]
        checkpoint = self.root/'prefix.pt'
        checkpoint.write_bytes(b'unit fixture checkpoint digest')
        with patch('so101_act.generate.MujocoEnvironment', return_value=fake), \
             patch('so101_act.generate.SO101Expert', return_value=teacher), \
             patch('so101_act.policy.ACTPolicy', return_value=learner), \
             patch('so101_act.generate.provenance', return_value={}), \
             patch('so101_act.generate.scene_group', return_value='scene'), \
             patch('so101_act.generate.run_pipeline', side_effect=run):
            result = generate_episode(self.root, 'accessible', 0, learner_checkpoint=checkpoint, learner_steps=1)
        self.assertTrue(result['success'])
        segment = read_segment(result['saved_segments'][0])
        np.testing.assert_allclose(segment['goal'], encode_goal(goal, observation.ee_pose))
        np.testing.assert_allclose(segment['actions'], .02)
        np.testing.assert_allclose(segment['timestamps'], [1/30])

    def test_split_groups_all_subgoals_from_same_scene_and_stats_use_train_only(self):
        for group in ('seed-0', 'seed-0', 'seed-1', 'seed-1'):
            self.segment(group)
        train, validation = split_segments(discover_segments(self.root), seed=42)
        self.assertFalse({s['meta']['group_id'] for s in train} & {s['meta']['group_id'] for s in validation})
        validation[0]['states'][:] = 100.
        self.assertLess(max(compute_stats(train)['states']['mean']), 1.)

    def test_rejects_different_timing_and_corrupted_goal(self):
        path = self.segment()
        with np.load(path/'trajectory.npz') as stored:
            arrays = {k: stored[k] for k in stored.files}
        arrays['timestamps'] = np.array([0., .1, .2])
        np.savez(path/'trajectory.npz', **arrays)
        with self.assertRaisesRegex(ValueError, 'timing'):
            read_segment(path)
        arrays['timestamps'] = np.arange(3)/30
        arrays['goal'][16] = 1.
        np.savez(path/'trajectory.npz', **arrays)
        with self.assertRaisesRegex(ValueError, 'primitive/validity'):
            read_segment(path)

    def test_training_saves_loadable_checkpoint_and_disjoint_split(self):
        self.segment('seed-0')
        self.segment('seed-1')
        config = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        config.update(epochs=1, batch_size=2, device='cpu', torch_threads=2)
        config['model'] = model_config().to_dict()
        checkpoint = train_run(self.root, self.root/'run', config)
        policy = ACTPolicy(checkpoint, device='cpu')
        observation, goal = inputs()
        command = policy.predict_action_chunk(observation, goal)
        self.assertEqual(command.commands.shape, (1, 6))
        payload = torch.load(checkpoint, weights_only=True)
        self.assertTrue(np.isfinite(payload['metrics']['val_prior_l1']))
        self.assertNotEqual(payload['split']['train'][0]['group_id'], payload['split']['validation'][0]['group_id'])

    def test_camera_transfer_is_explicit_and_recomputes_train_normalization(self):
        self.segment('seed-0')
        self.segment('seed-1')
        config = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        config.update(epochs=1, batch_size=2, device='cpu', torch_threads=2)
        config['model'] = model_config().to_dict()
        checkpoint = train_run(self.root, self.root/'original', config)
        payload = torch.load(checkpoint, weights_only=True)
        payload['camera']['camera_to_robot_base'][0][3] = .2
        payload['stats']['states']['mean'] = [3.]*6
        transferred = self.root/'different_camera.pt'
        torch.save(payload, transferred)
        with self.assertRaisesRegex(ValueError, 'camera calibration'):
            train_run(self.root, self.root/'rejected', config, init_checkpoint=transferred)
        result = train_run(self.root, self.root/'transferred', config, init_checkpoint=transferred, transfer_camera=True)
        new = torch.load(result, weights_only=True)
        self.assertTrue(new['camera_transfer'])
        np.testing.assert_allclose(new['stats']['states']['mean'], 0.)
        np.testing.assert_allclose(new['model']['state_to_action_scale'].numpy(),
            np.array(new['stats']['states']['std'])/new['stats']['actions']['std'])

    def test_feedback_transfer_expands_inputs_without_changing_action_contract(self):
        self.segment('seed-0')
        self.segment('seed-1')
        config = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        config.update(epochs=1, batch_size=2, device='cpu', torch_threads=2)
        config['model'] = model_config().to_dict()
        checkpoint = train_run(self.root, self.root/'base', config)
        config['model'].update(include_velocity=True, include_goal_error=True)
        with self.assertRaisesRegex(ValueError, 'velocity input transfer'):
            train_run(self.root, self.root/'rejected', config, init_checkpoint=checkpoint)
        config.update(allow_velocity_transfer=True, allow_goal_error_transfer=True)
        with patch('action_generator.kinematics.SO101Kinematics') as kinematics:
            kinematics.return_value.pose.return_value = Pose((.1, 0., .1))
            result = train_run(self.root, self.root/'feedback', config, init_checkpoint=checkpoint)
        payload = torch.load(result, weights_only=True)
        self.assertEqual(payload['model']['state_proj.weight'].shape[1], 15)
        obs, goal = inputs()
        chunk = ACTPolicy(result, device='cpu').predict_action_chunk(replace(obs, ee_pose=Pose((.1, 0., .1))), goal)
        self.assertEqual(chunk.commands.shape, (1, 6))

    def test_warm_start_expands_legacy_goal_without_validation_leakage(self):
        self.segment('seed-0')
        self.segment('seed-1')
        config = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        config.update(epochs=1, batch_size=2, device='cpu', torch_threads=2, balance_goals=True)
        config['model'] = model_config().to_dict()
        checkpoint = train_run(self.root, self.root/'first', config)
        payload = torch.load(checkpoint, weights_only=True)
        payload['checkpoint_version'], payload['goal_version'] = 'so101-act-v2', 'so101-goal-v1'
        payload['model']['goal_proj.weight'] = payload['model']['goal_proj.weight'][:, :17].clone()
        payload['stats']['goal_scale'] = payload['stats']['goal_scale'][:17]
        legacy = self.root/'legacy.pt'
        torch.save(payload, legacy)
        resumed = train_run(self.root, self.root/'resumed', config, init_checkpoint=legacy)
        self.assertEqual(torch.load(resumed, weights_only=True)['model']['goal_proj.weight'].shape[1], GOAL_DIM)
        payload['split']['train'] = payload['split']['validation']
        torch.save(payload, legacy)
        with self.assertRaisesRegex(ValueError, 'new validation scenes'):
            train_run(self.root, self.root/'leaked', config, init_checkpoint=legacy)


class ModelTests(unittest.TestCase):
    def test_absolute_action_mode_does_not_copy_measured_state_into_prediction(self):
        config = model_config()
        config.residual_actions = False
        model = GoalConditionedACT(config).eval()
        with torch.no_grad():
            model.action_head.weight.zero_()
            model.action_head.bias.fill_(.2)
        image = torch.zeros(1, 3, 32, 32)
        for value in (0., 2.):
            prediction, _, _ = model(image, torch.full((1, 6), value), torch.zeros(1, GOAL_DIM))
            torch.testing.assert_close(prediction, torch.full((1, 4, 6), .2))

    def test_zero_residual_is_measured_state_in_action_units(self):
        model = GoalConditionedACT(model_config()).eval()
        stats = identity_stats()
        stats['states'] = {'mean': [.1]*6, 'std': [.3]*6}
        stats['actions'] = {'mean': [-.2]*6, 'std': [.5]*6}
        model.set_normalization(stats)
        with torch.no_grad():
            model.action_head.weight.zero_()
            model.action_head.bias.zero_()
        state = torch.full((1, 6), .4)
        prediction, _, _ = model(torch.zeros(1, 3, 32, 32), state, torch.zeros(1, GOAL_DIM))
        actual = prediction*.5-.2
        torch.testing.assert_close(actual, (state*.3+.1).unsqueeze(1).expand(-1, 4, -1))

    def test_goal_changes_prediction_and_cvae_backpropagates(self):
        torch.manual_seed(0)
        model = GoalConditionedACT(model_config()).eval()
        image, state, goal = torch.rand(2, 3, 32, 32), torch.zeros(2, 6), torch.zeros(2, GOAL_DIM)
        goal[1, 2] = 1.
        prior, mu, _ = model(image[0:1].expand(2, -1, -1, -1), state, goal)
        self.assertIsNone(mu)
        self.assertFalse(torch.allclose(prior[0], prior[1]))
        actions, padding = torch.rand(2, 4, 6), torch.tensor([[False]*4, [False, False, True, True]])
        prediction, mu, logvar = model(image, state, goal, actions, padding)
        loss, _, _ = act_loss(prediction, actions, padding, mu, logvar)
        loss.backward()
        self.assertGreater(model.goal_proj.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.latent_params.weight.grad.abs().sum().item(), 0.)

    def test_padded_actions_do_not_change_posterior_or_loss(self):
        model = GoalConditionedACT(model_config()).eval()
        image, state, goal = torch.rand(1, 3, 32, 32), torch.zeros(1, 6), torch.zeros(1, GOAL_DIM)
        actions, padding = torch.zeros(1, 4, 6), torch.tensor([[False, True, True, True]])
        torch.manual_seed(7)
        prediction, mu, _ = model(image, state, goal, actions, padding)
        loss = act_loss(prediction, actions, padding)[0]
        actions[:, 1:] = 1000.
        torch.manual_seed(7)
        changed, changed_mu, _ = model(image, state, goal, actions, padding)
        torch.testing.assert_close(mu, changed_mu)
        torch.testing.assert_close(prediction, changed)
        torch.testing.assert_close(loss, act_loss(prediction, actions, padding)[0])


class PolicyTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT/'outputs/test_act'/uuid4().hex
        directory.mkdir(parents=True)
        self.checkpoint = directory/'unit_test.pt'
        observation, _ = inputs()
        self.payload = {'checkpoint_version': CHECKPOINT_VERSION, 'goal_version': GOAL_VERSION,
                        'robot_id': 'so101', 'diagnostic': False, 'primitives': ['PICK'],
                        'model_config': model_config().to_dict(), 'stats': identity_stats(),
                        'control': dict(control(), command_dt=1/30),
                        'camera': {key: getattr(observation, key).tolist()
                                   for key in ('camera_intrinsics', 'camera_to_robot_base')},
                        'model': GoalConditionedACT(model_config()).state_dict()}
        torch.save(self.payload, self.checkpoint)

    def test_control_bounds_and_cached_chunk_reset_on_goal_episode_or_time_change(self):
        policy = ACTPolicy(self.checkpoint, device='cpu')
        observation, goal = inputs()
        with patch.object(policy.model, 'forward', wraps=policy.model.forward) as forward:
            first = policy.predict_action_chunk(observation, goal)
            self.assertTrue(np.all(np.abs(first.commands[0]-observation.robot_state) <= np.array(control()['max_delta'])))
            self.assertGreaterEqual(first.commands[0, 5], 0.)
            observation = replace(observation, timestamp=1/30, robot_state=first.commands[0])
            policy.predict_action_chunk(observation, goal)
            self.assertEqual(forward.call_count, 1)
            goal = replace(goal, subgoal_id='new-goal')
            policy.predict_action_chunk(replace(observation, timestamp=2/30), goal)
            self.assertEqual(forward.call_count, 2)
            policy.predict_action_chunk(replace(observation, timestamp=0., episode_id='new-episode'), goal)
            self.assertEqual(forward.call_count, 3)
            policy.predict_action_chunk(replace(observation, timestamp=0., episode_id='new-episode'), goal)
            self.assertEqual(forward.call_count, 4)

    def test_temporal_ensemble_aligns_future_actions_and_clears_at_new_goal(self):
        policy = ACTPolicy(self.checkpoint, device='cpu', temporal_ensemble=True)
        observation, goal = inputs()
        first = torch.tensor([.01, .02, .03, .04]).view(1, 4, 1).expand(1, 4, 6)
        second = torch.full((1, 4, 6), .03)
        with patch.object(policy.model, 'forward', side_effect=[(first, None, None), (second, None, None), (second, None, None)]):
            policy.predict_action_chunk(observation, goal)
            command = policy.predict_action_chunk(replace(observation, timestamp=1/30), goal)
            expected = (.02*np.exp(-.2)+.03)/(np.exp(-.2)+1)
            np.testing.assert_allclose(command.commands[0], expected, atol=1e-7)
            command = policy.predict_action_chunk(replace(observation, timestamp=2/30), replace(goal, subgoal_id='next'))
            np.testing.assert_allclose(command.commands[0], .03, atol=1e-7)

    def test_velocity_state_is_reset_at_subgoal_boundary(self):
        config = model_config()
        config.include_velocity = True
        self.payload['model_config'] = config.to_dict()
        self.payload['model'] = GoalConditionedACT(config).state_dict()
        torch.save(self.payload, self.checkpoint)
        policy = ACTPolicy(self.checkpoint, device='cpu', execute_steps=1)
        observation, goal = inputs()
        states = []
        def forward(image, state, encoded_goal):
            states.append(state.cpu().numpy().copy())
            return torch.zeros(1, 4, 6), None, None
        with patch.object(policy.model, 'forward', side_effect=forward):
            policy.predict_action_chunk(observation, goal)
            changed = replace(observation, timestamp=1/30, robot_state=np.full(6, .03, np.float32))
            policy.predict_action_chunk(changed, goal)
            policy.predict_action_chunk(replace(changed, timestamp=2/30), replace(goal, subgoal_id='new'))
        np.testing.assert_allclose(states[0][0, 6:], 0.)
        np.testing.assert_allclose(states[1][0, 6:], np.array([.03]*6)/control()['max_delta'])
        np.testing.assert_allclose(states[2][0, 6:], 0.)

    def test_rejects_diagnostic_checkpoint_wrong_camera_and_untrained_primitive(self):
        self.payload['diagnostic'] = True
        torch.save(self.payload, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'diagnostic checkpoint'):
            ACTPolicy(self.checkpoint)
        policy = ACTPolicy(self.checkpoint, allow_diagnostic=True)
        observation, goal = inputs()
        with self.assertRaisesRegex(ValueError, 'no PLACE'):
            policy.predict_action_chunk(observation, replace(goal, primitive='PLACE', place_pose=Pose((.1, .1, .03))))
        transform = observation.camera_to_robot_base.copy()
        transform[0, 3] = 1.
        with self.assertRaisesRegex(ValueError, 'camera calibration'):
            policy.predict_action_chunk(replace(observation, camera_to_robot_base=transform), goal)

    def test_shared_public_function_loads_checkpoint_without_changing_its_interface(self):
        observation, goal = inputs()
        _load_act.cache_clear()
        self.addCleanup(_load_act.cache_clear)
        with patch.dict(os.environ, {'HSSG_ACT_CHECKPOINT': str(self.checkpoint), 'HSSG_ACT_DEVICE': 'cpu',
                                    'HSSG_C_BACKEND': 'act'}):
            chunk = predict_action_chunk(observation, goal)
        self.assertEqual(chunk.subgoal_id, goal.subgoal_id)
        self.assertEqual(chunk.robot_id, 'so101')
        self.assertEqual(chunk.commands.shape, (1, 6))

    def test_grasp_condition_is_frozen_for_a_place_subgoal(self):
        self.payload['primitives'].append('PLACE')
        torch.save(self.payload, self.checkpoint)
        policy = ACTPolicy(self.checkpoint, device='cpu')
        observation, pick = inputs()
        observation = replace(observation, ee_pose=Pose((.2, 0., .08)))
        goal = replace(pick, primitive='PLACE', place_pose=Pose((.1, .2, .03)))
        policy.predict_action_chunk(observation, goal)
        initial = policy._encoded_goal.copy()
        moved = replace(observation, timestamp=2/30, ee_pose=Pose((.1, .2, .08)))
        policy.predict_action_chunk(moved, goal)
        np.testing.assert_array_equal(policy._encoded_goal, initial)
        self.assertEqual(initial[-1], 1.)


if __name__ == '__main__':
    unittest.main()
