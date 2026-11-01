"""Capture a static scene at two times, with a real robot joint movement between."""
import argparse
import json
from pathlib import Path
import numpy as np
from uuid import uuid4
from pipeline.mujoco_env import MujocoEnvironment
from perception.acquisition import capture

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('outputs/multiview/robot_relative_seed_42'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--frames',type=int,default=2)
    args=p.parse_args()
    if args.seed<0 or args.frames<1:p.error('seed must be nonnegative and frames must be positive')
    env=MujocoEnvironment(render=False)
    try:
        env.reset(args.seed)
        env._episode += '-' + uuid4().hex[:10]
        positions=[]
        for i in range(args.frames):
            if i:
                command=env.joint_state()
                command[0]+=.025
                for _ in range(6):
                    env.step(command)
            meta=capture(env,args.output/f'frame_{i:03d}')
            positions.append([c['T_camera_to_base'] for c in meta['cameras'].values()])
            print('Captured',i,meta['timestamp'],list(meta['cameras']))
        assert all(np.allclose(pos,positions[0]) for pos in positions), 'Fixed cameras moved'
        (args.output/'acquisition_validation.json').write_text(json.dumps(dict(frames=args.frames,
                   fixed_cameras=True,robot_joint_perturbed=args.frames>1,source='mujoco RGB-D only'),indent=2))
    finally:
        env.close()

if __name__=='__main__':
    main()
