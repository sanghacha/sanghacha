"""모듈 C: 체크포인트 ACT 또는 즉시 실행 가능한 MuJoCo 전문가 제어기."""

from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from pipeline.types import ActionChunk, Observation, Subgoal, check


@lru_cache(maxsize=1)
def _load_act(checkpoint, device):
    try:
        from so101_act.policy import ACTPolicy
    except ImportError as exc:
        raise RuntimeError('ACT needs the training environment: uv run --project training --extra cpu ... '
                           '(or --extra cuda)') from exc
    return ACTPolicy(checkpoint, device=device)


@lru_cache(maxsize=1)
def _load_expert():
    from .expert import SO101Expert
    return SO101Expert()


def predict_action_chunk(observation: Observation, subgoal: Subgoal) -> ActionChunk:
    if observation.robot_id == "so101":
        checkpoint = os.environ.get('HSSG_ACT_CHECKPOINT')
        mode = os.environ.get('HSSG_C_BACKEND', 'auto')
        check(mode in ('auto', 'expert', 'act', 'hold'), 'unknown HSSG_C_BACKEND')
        if mode == 'act' or (mode == 'auto' and checkpoint):
            check(bool(checkpoint), 'ACT mode requires HSSG_ACT_CHECKPOINT')
            return _load_act(str(Path(checkpoint).resolve()), os.environ.get('HSSG_ACT_DEVICE', 'auto')).predict_action_chunk(
                observation, subgoal)
        if mode != 'hold':
            return _load_expert().predict_action_chunk(observation, subgoal)
        # Explicit diagnostic baseline, never used as an automatic fallback after failures.
        return ActionChunk(subgoal.subgoal_id, np.repeat(observation.robot_state[None, :], 4, axis=0),
                           observation.control_dt, observation.robot_id)
    check(observation.robot_id == "mock", "starter action generator only supports mock robot")
    current = observation.robot_state.copy()
    target = current.copy()
    target[:3] = np.array(subgoal.target_ee_pose.position) / .1
    commands = []
    for _ in range(16):
        difference = target - current
        if np.max(np.abs(difference[:5])) > 1e-5:
            current = current + np.clip(difference, -1, 1)
        else:
            current[5] = 1 if subgoal.primitive == "PICK" else 0
        commands.append(current.copy())
    return ActionChunk(subgoal.subgoal_id, np.array(commands, dtype=np.float32),
                       observation.control_dt, observation.robot_id)
