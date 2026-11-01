"""Copy v1 MuJoCo demonstrations and reconstruct the initial grasp transform from FK."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from action_generator.goal import GOAL_FIELDS, GOAL_VERSION, encode_goal
from action_generator.kinematics import SO101Kinematics
from pipeline.mock_data import ROOT
from pipeline.types import Pose, Subgoal, check


def upgrade_data(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    kinematics = SO101Kinematics()
    commit = json.loads((ROOT/'assets/so101/source.json').read_text(encoding='utf-8'))['commit']
    count = 0
    for path in sorted(source.rglob('metadata.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        check(meta['goal_version'] == 'so101-goal-v1', 'this migration accepts v1 goals only')
        check(meta['source'] == 'mujoco_gt' and meta['robot_id'] == 'so101', 'FK migration is for MuJoCo SO-101 only')
        check(meta['provenance']['model_source']['commit'] == commit, 'recorded model commit differs')
        if not meta.get('frames'):
            continue
        with np.load(path.parent/'trajectory.npz', allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        old = arrays['goal']
        check(old.shape == (17,), 'invalid v1 goal shape')
        pose = lambda values: Pose(tuple(values[:3]), tuple(values[3:7]))
        object_pose = pose(old[2:9])
        place_pose = pose(old[9:16]) if meta['primitive'] == 'PLACE' else None
        goal = Subgoal(meta['subgoal_id'], meta['primitive'], meta['object_id'], object_pose, object_pose, place_pose)
        ee = kinematics.pose(arrays['states'][0])
        arrays['goal'] = encode_goal(goal, ee)
        directory = output/path.parent.name
        directory.mkdir()
        shutil.copytree(path.parent/'rgb', directory/'rgb')
        np.savez_compressed(directory/'trajectory.npz', **arrays)
        meta.update(goal_version=GOAL_VERSION, goal_fields=GOAL_FIELDS,
                    initial_ee_pose={'position': ee.position, 'quaternion_xyzw': ee.quaternion_xyzw},
                    upgrade_source={'path': str(path.parent.resolve()),
                                    'trajectory_sha256': sha256((path.parent/'trajectory.npz').read_bytes()).hexdigest()})
        (directory/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print('upgraded segments:', upgrade_data(args.source, args.output))


if __name__ == '__main__':
    main()
