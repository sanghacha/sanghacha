"""Diagnostic software check, not a successful PICK/PLACE demonstration or evaluation."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from hssg_generator import generate_scene_graph
from planner import plan_next_subgoal
from pipeline.mock_data import ROOT
from pipeline.mujoco_env import MujocoEnvironment
from .collect import provenance, scene_group
from .data import SegmentWriter
from .policy import ACTPolicy
from .train import DEFAULT_CONFIG, train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'training/runs'/f'smoke-{uuid4().hex[:8]}')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = MujocoEnvironment('accessible')
    try:
        for seed in (700, 701):
            observation = env.reset(seed)
            goal = plan_next_subgoal(generate_scene_graph(observation), env.task, observation.robot_state)
            writer = SegmentWriter(args.output/'data', observation, goal, group_id=scene_group(env, seed),
                                   lower=env.lower, upper=env.upper, max_delta=env.config['max_delta'],
                                   image_size=(32, 32), diagnostic=True, provenance=provenance(env, seed))
            try:
                for _ in range(4):
                    before = observation
                    command = observation.robot_state.copy()
                    command[0] += .01
                    observation = env.step(command)
                    writer.append(before, command)
                writer.finish(success=False, reason='diagnostic_transport_only_NOT_manipulation_success')
            finally:
                if not writer.finished:
                    writer.finish(success=False, reason='diagnostic_interrupted')
        config = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        config.update(epochs=2, batch_size=4, device='cpu', torch_threads=2)
        config['model'].update(image_size=[32, 32], chunk_size=4, hidden_dim=32, latent_dim=4,
                               encoder_layers=1, decoder_layers=1, posterior_layers=1, feedforward_dim=64, dropout=0.)
        checkpoint = train_run(args.output/'data', args.output/'model', config, allow_diagnostic=True)
        policy = ACTPolicy(checkpoint, device='cpu', allow_diagnostic=True)
        observation = env.reset(702)
        goal = plan_next_subgoal(generate_scene_graph(observation), env.task, observation.robot_state)
        for _ in range(5):
            chunk = policy.predict_action_chunk(observation, goal)
            observation = env.step(chunk.commands[0])
        print('PASS: rendered RGB -> diagnostic dataset -> CVAE training -> checkpoint -> bounded MuJoCo actions.')
        print('This checkpoint is diagnostic-only and is rejected by normal inference/evaluation.')
        print('PICK/PLACE success was NOT demonstrated. Output:', args.output)
    finally:
        env.close()


if __name__ == '__main__':
    main()
