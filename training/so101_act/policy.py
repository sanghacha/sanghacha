"""Load a trained ACT and adapt predictions to the shared module C interface."""

from pathlib import Path

import numpy as np
import torch

from action_generator.goal import GOAL_VERSION, encode_goal
from pipeline.types import ActionChunk, check
from .data import preprocess_rgb
from .model import ACTConfig, GoalConditionedACT
from .train import CHECKPOINT_VERSION, select_device


class ACTPolicy:
    def __init__(self, checkpoint, *, device='auto', execute_steps=None, allow_diagnostic=False):
        self.device = select_device(device)
        payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        if self.device.type == 'cpu':
            torch.set_num_threads(min(torch.get_num_threads(), int(payload.get('training_config', {}).get('torch_threads', 4))))
        check(payload['checkpoint_version'] == CHECKPOINT_VERSION and payload['goal_version'] == GOAL_VERSION,
              'incompatible ACT checkpoint')
        check(payload['robot_id'] == 'so101', 'checkpoint is not for SO-101')
        check(allow_diagnostic or not payload.get('diagnostic', False),
              'diagnostic checkpoint is only for software checks, not a trained manipulation policy')
        self.config = ACTConfig(**payload['model_config'])
        if execute_steps is None:
            execute_steps = min(4, self.config.chunk_size)
        check(type(execute_steps) is int and 1 <= execute_steps <= self.config.chunk_size,
              'execute_steps must be within the trained chunk length')
        self.execute_steps = execute_steps
        self.stats, self.control, self.camera = payload['stats'], payload['control'], payload['camera']
        self.primitives = payload['primitives']
        self.model = GoalConditionedACT(self.config).to(self.device)
        self.model.load_state_dict(payload['model'], strict=True)
        self.model.eval()
        self._key, self._last_timestamp = None, None
        self._pending = []
        self._goal_identity, self._encoded_goal = None, None
        self._kinematics = None

    def predict_action_chunk(self, observation, subgoal):
        check(observation.robot_id == 'so101' and observation.rgb_valid, 'ACT requires valid SO-101 RGB')
        check(subgoal.primitive in self.primitives, f'checkpoint has no {subgoal.primitive} demonstrations')
        check(np.isclose(observation.control_dt, self.control['command_dt']), 'checkpoint control rate differs')
        for key in ('camera_intrinsics', 'camera_to_robot_base'):
            check(np.allclose(getattr(observation, key), self.camera[key], atol=1e-6),
                  f'camera calibration differs from training: {key}')
        base = encode_goal(subgoal)
        identity = (observation.episode_id, subgoal.subgoal_id, base[:17].tobytes())
        if identity != self._goal_identity or (self._last_timestamp is not None and observation.timestamp <= self._last_timestamp):
            ee_pose = observation.ee_pose
            if subgoal.primitive == 'PLACE' and ee_pose is None:
                from action_generator.kinematics import SO101Kinematics
                if self._kinematics is None:
                    self._kinematics = SO101Kinematics()
                ee_pose = self._kinematics.pose(observation.robot_state)
            self._encoded_goal = encode_goal(subgoal, ee_pose)
            self._goal_identity = identity
        goal = self._encoded_goal
        key = (observation.episode_id, subgoal.subgoal_id, goal.tobytes())
        continuous = self._last_timestamp is not None and np.isclose(
            observation.timestamp-self._last_timestamp, observation.control_dt, rtol=1e-5, atol=1e-8)
        if key != self._key or not continuous:
            self._pending = []
        self._key, self._last_timestamp = key, observation.timestamp
        if not self._pending:
            state = (observation.robot_state-self.stats['states']['mean'])/self.stats['states']['std']
            image = preprocess_rgb(observation.rgb, self.config.image_size)
            inputs = [torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
                      for x in (image, state, goal/self.stats['goal_scale'])]
            with torch.inference_mode():
                prediction, _, _ = self.model(*inputs)
            commands = prediction[0].cpu().numpy()*self.stats['actions']['std']+self.stats['actions']['mean']
            check(bool(np.isfinite(commands).all()), 'ACT predicted nonfinite actions')
            self._pending = list(commands[:self.execute_steps])
        target = self._pending.pop(0).copy()
        # Return one command per observed tick, filtering against ACTUAL measured q.
        # The internal chunk is invalidated on goal/episode/timing changes.
        state = observation.robot_state
        delta = np.asarray(self.control['max_delta'])*.8
        target = state+np.clip(target-state, -delta, delta)
        target = np.clip(target, np.array(self.control['lower'])+1e-5, np.array(self.control['upper'])-1e-5)
        check(np.all(np.abs(target-state) <= np.asarray(self.control['max_delta'])+1e-6),
              'measured state too far outside checkpoint control limits')
        return ActionChunk(subgoal.subgoal_id, target.astype(np.float32)[None, :],
                           observation.control_dt, observation.robot_id)
