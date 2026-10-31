"""모듈 C 시작점. SO-101에서는 현재 관절을 유지한다. ACT/파지 정책은 팀 C가 구현한다."""

import numpy as np

from pipeline.types import ActionChunk, Observation, Subgoal, check


def predict_action_chunk(observation: Observation, subgoal: Subgoal) -> ActionChunk:
    if observation.robot_id == "so101":
        # 실제 관절 명령 형식을 독립 검사하는 hold baseline. Subgoal 수행 성공을 의미하지 않는다.
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
