"""Aligned pre-action observations and executed absolute targets, one subgoal per segment."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from action_generator.goal import GOAL_DIM, GOAL_FIELDS, GOAL_VERSION, encode_goal, control_goal_error
from pipeline.types import check

DATA_VERSION = 'so101-demo-v1'
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def resize_rgb(rgb, image_size):
    height, width = image_size
    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)).copy()


def preprocess_rgb(rgb, image_size):
    image = rgb if tuple(rgb.shape[:2]) == tuple(image_size) else resize_rgb(rgb, image_size)
    return image.transpose(2, 0, 1).astype(np.float32)/255.


class SegmentWriter:
    def __init__(self, output, observation, subgoal, *, group_id, lower, upper, max_delta,
                 image_size=(120, 160), provenance=None, diagnostic=False):
        check(observation.robot_id == 'so101' and observation.rgb_valid, 'record real SO-101 RGB observations')
        self.path = Path(output)/uuid4().hex
        (self.path/'rgb').mkdir(parents=True)
        self.image_size = tuple(image_size)
        check(subgoal.primitive != 'PLACE' or observation.ee_pose is not None, 'PLACE demos require initial measured EE pose')
        self.goal = encode_goal(subgoal, observation.ee_pose)
        self.states, self.actions, self.timestamps = [], [], []
        self.finished = False
        self.episode_id = observation.episode_id
        self.meta = {
            'data_version': DATA_VERSION, 'goal_version': GOAL_VERSION, 'goal_fields': GOAL_FIELDS,
            'group_id': group_id, 'episode_id': observation.episode_id, 'source': observation.source,
            'primitive': subgoal.primitive, 'object_id': subgoal.object_id, 'subgoal_id': subgoal.subgoal_id,
            'robot_id': observation.robot_id, 'joint_names': JOINT_NAMES, 'command_dt': observation.control_dt,
            'frame_id': observation.frame_id, 'image_size': list(image_size),
            'lower': np.asarray(lower).tolist(), 'upper': np.asarray(upper).tolist(),
            'max_delta': np.asarray(max_delta).tolist(), 'diagnostic': diagnostic,
            'camera_intrinsics': observation.camera_intrinsics.tolist(),
            'camera_to_robot_base': observation.camera_to_robot_base.tolist(),
            'provenance': provenance or {},
            'initial_ee_pose': ({'position': observation.ee_pose.position,
                                 'quaternion_xyzw': observation.ee_pose.quaternion_xyzw}
                                if observation.ee_pose is not None else None),
        }

    def append(self, observation, executed_command):
        check(not self.finished, 'segment already finalized')
        check(observation.rgb_valid and observation.robot_id == 'so101', 'invalid policy observation')
        check(observation.episode_id == self.episode_id, 'episode changed during segment')
        check(np.isclose(observation.control_dt, self.meta['command_dt']), 'command_dt changed')
        check(not self.timestamps or np.isclose(observation.timestamp-self.timestamps[-1],
                                               observation.control_dt, rtol=1e-5, atol=1e-8),
              'demonstration timestamps must advance by one control tick')
        action = np.asarray(executed_command)
        check(action.shape == (6,) and action.dtype == np.float32 and np.isfinite(action).all(), 'invalid action')
        check(np.all(action >= self.meta['lower']) and np.all(action <= self.meta['upper']), 'action outside limits')
        check(np.all(np.abs(action-observation.robot_state) <= np.array(self.meta['max_delta'])+1e-6),
              'action exceeds per-step delta')
        Image.fromarray(resize_rgb(observation.rgb, self.image_size)).save(self.path/'rgb'/f'{len(self.states):06d}.png')
        self.states.append(observation.robot_state.copy())
        self.actions.append(action.copy())
        self.timestamps.append(observation.timestamp)

    def finish(self, *, success, reason):
        check(not self.finished, 'segment already finalized')
        self.finished = True
        self.meta.update(success=bool(success), reason=str(reason), frames=len(self.states))
        if self.states:
            np.savez_compressed(self.path/'trajectory.npz',
                                states=np.asarray(self.states, dtype=np.float32),
                                actions=np.asarray(self.actions, dtype=np.float32), goal=self.goal,
                                timestamps=np.asarray(self.timestamps, dtype=np.float64))
        # metadata.json is the commit marker; interrupted writers remain undiscoverable.
        # Rename only the tiny metadata file, not a folder of PNGs Windows may still scan.
        temporary = self.path/'metadata.pending'
        temporary.write_text(json.dumps(self.meta, indent=2)+'\n', encoding='utf-8')
        temporary.replace(self.path/'metadata.json')
        return self.path


def read_segment(path):
    path = Path(path)
    meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    check(meta['data_version'] == DATA_VERSION and meta['goal_version'] == GOAL_VERSION, 'incompatible demo schema')
    check(meta['robot_id'] == 'so101' and meta['frame_id'] == 'robot_base' and
          meta['joint_names'] == JOINT_NAMES, 'incompatible robot/frame/joint order')
    check(meta['source'] in ('mujoco_gt', 'rgbd'), 'mock data is not a demonstration')
    with np.load(path/'trajectory.npz', allow_pickle=False) as stored:
        arrays = {k: stored[k].copy() for k in ('states', 'actions', 'goal', 'timestamps')}
    count = meta['frames']
    check(count > 0 and arrays['states'].shape == arrays['actions'].shape == (count, 6), 'invalid demo shape')
    check(arrays['goal'].shape == (GOAL_DIM,) and arrays['timestamps'].shape == (count,), 'invalid goal/time shape')
    check(all(np.isfinite(a).all() for a in arrays.values()), 'nonfinite demonstration')
    check(all(arrays[k].dtype == np.float32 for k in ('states', 'actions', 'goal')), 'demo values must be float32')
    check(meta['primitive'] in ('PICK', 'PLACE'), 'unknown demonstration primitive')
    place = meta['primitive'] == 'PLACE'
    check(np.array_equal(arrays['goal'][:2], [not place, place]) and arrays['goal'][16] == float(place),
          'goal primitive/validity differs from metadata')
    check(np.isclose(np.linalg.norm(arrays['goal'][5:9]), 1.), 'invalid goal object quaternion')
    check((place and np.isclose(np.linalg.norm(arrays['goal'][12:16]), 1.)) or
          (not place and not arrays['goal'][9:16].any()), 'invalid PLACE pose encoding')
    check(arrays['goal'][24] == float(place), 'invalid grasp validity')
    check((place and np.isclose(np.linalg.norm(arrays['goal'][20:24]), 1.)) or
          (not place and not arrays['goal'][17:24].any()), 'invalid grasp transform')
    dt = float(meta['command_dt'])
    check(np.isfinite(dt) and dt > 0 and np.allclose(np.diff(arrays['timestamps']), dt, rtol=1e-5, atol=1e-8),
          'invalid demonstration timing')
    check(np.all(arrays['actions'] >= meta['lower']) and np.all(arrays['actions'] <= meta['upper']), 'invalid target limits')
    check(np.all(np.abs(arrays['actions']-arrays['states']) <= np.array(meta['max_delta'])+1e-6), 'invalid target delta')
    check(all((path/'rgb'/f'{i:06d}.png').is_file() for i in range(count)), 'missing RGB frames')
    return {'path': path, 'meta': meta, **arrays}


def discover_segments(root, *, allow_diagnostic=False):
    segments = []
    roots = root if isinstance(root, (list, tuple)) else [root]
    paths = [path for directory in roots for path in Path(directory).rglob('metadata.json')]
    check(len({path.resolve() for path in paths}) == len(paths), 'overlapping demonstration roots')
    for path in sorted(paths):
        if path.parent.suffix == '.partial':
            continue
        meta = json.loads(path.read_text(encoding='utf-8'))
        if not meta.get('frames'):
            continue
        if meta.get('diagnostic', False):
            if not allow_diagnostic:
                continue
        elif not meta.get('success'):
            continue
        segments.append(read_segment(path.parent))
    check(bool(segments), 'no successful demonstration segments; collect demos first (diagnostic data excluded by default)')
    first = segments[0]['meta']
    for segment in segments:
        meta = segment['meta']
        check(meta['image_size'] == first['image_size'], 'mixed demo image sizes')
        check(np.isclose(meta['command_dt'], first['command_dt']), 'mixed demo control rates')
        for key in ('lower', 'upper', 'max_delta', 'camera_intrinsics', 'camera_to_robot_base'):
            check(np.allclose(meta[key], first[key]), f'mixed {key}; separate datasets for different calibrations/robots')
    return segments


def split_segments(segments, fraction=.2, seed=42):
    check(0 < fraction < 1, 'validation_fraction must be between 0 and 1')
    groups = sorted({s['meta']['group_id'] for s in segments})
    check(len(groups) >= 2, 'collect at least two different scene seeds for a held-out validation split')
    np.random.default_rng(seed).shuffle(groups)
    count = min(len(groups)-1, max(1, round(len(groups)*fraction)))
    validation = set(groups[:count])
    return ([s for s in segments if s['meta']['group_id'] not in validation],
            [s for s in segments if s['meta']['group_id'] in validation])


def compute_stats(segments):
    result = {}
    for key in ('states', 'actions'):
        values = np.concatenate([s[key] for s in segments])
        result[key] = {'mean': values.mean(0).tolist(), 'std': np.maximum(values.std(0), .01).tolist()}
    # Fixed units: positions /0.3m; one-hot, validity, and quaternion stay unchanged.
    scale = np.ones(GOAL_DIM, dtype=np.float32)
    scale[2:5] = scale[9:12] = .3
    scale[17:20] = .03
    result['goal_scale'] = scale.tolist()
    return result


class ChunkDataset:
    def __init__(self, segments, stats, config, *, cache_images=False):
        self.segments, self.stats, self.config = segments, stats, config
        self.index = [(s, t) for s, segment in enumerate(segments) for t in range(segment['meta']['frames'])]
        self._images = {} if cache_images else None
        families = [(s['meta']['primitive'], s['meta']['object_id']) for s in segments]
        self.family_names = sorted(set(families))
        self.family_ids = [self.family_names.index(family) for family in families]
        self._goal_errors = None
        if config.include_goal_error:
            from action_generator.kinematics import SO101Kinematics
            kinematics = SO101Kinematics()
            self._goal_errors = [np.array([control_goal_error(segment['goal'], kinematics.pose(state))
                                          for state in segment['states']]) for segment in segments]

    def sampling_weights(self):
        counts = {family: sum(len(s['actions']) for i, s in enumerate(self.segments) if self.family_ids[i] == family)
                  for family in range(len(self.family_names))}
        return [1./counts[self.family_ids[s]] for s, _ in self.index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        s, t = self.index[index]
        segment = self.segments[s]
        count = min(self.config.chunk_size, len(segment['actions'])-t)
        actions = np.zeros((self.config.chunk_size, 6), dtype=np.float32)
        stats = self.stats['actions']
        actions[:count] = (segment['actions'][t:t+count]-stats['mean'])/stats['std']
        padding = np.arange(self.config.chunk_size) >= count
        rgb = self._images.get(index) if self._images is not None else None
        if rgb is None:
            with Image.open(segment['path']/'rgb'/f'{t:06d}.png') as image:
                rgb = np.asarray(image.convert('RGB')).copy()
            if self._images is not None:
                self._images[index] = rgb
        state_stats = self.stats['states']
        state = ((segment['states'][t]-state_stats['mean'])/state_stats['std']).astype(np.float32)
        if self.config.include_velocity:
            previous = segment['states'][max(0, t-1)]
            velocity = (segment['states'][t]-previous)/np.asarray(segment['meta']['max_delta'])
            state = np.r_[state, velocity].astype(np.float32)
        if self.config.include_goal_error:
            state = np.r_[state, self._goal_errors[s][t]].astype(np.float32)
        return {
            'image': preprocess_rgb(rgb, self.config.image_size),
            'state': state,
            'goal': (segment['goal']/self.stats['goal_scale']).astype(np.float32),
            'actions': actions, 'padding': padding,
            'family': np.int64(self.family_ids[s]),
        }
