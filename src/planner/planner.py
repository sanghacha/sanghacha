"""모듈 B 시작점: mock 적층 예시의 다음 PICK/PLACE를 선택한다."""

import numpy as np
from numpy.typing import NDArray

from pipeline.types import MOCK_GRASP_OFFSET, Pose, SceneGraph, Subgoal, TaskRequest, check


def _xy_extent(obj):
    x, y, z, w = obj.pose.quaternion_xyzw
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return np.array([c*obj.size[0]+s*obj.size[1], s*obj.size[0]+c*obj.size[1]])/2


def _free_xy(graph, obj, position):
    half = np.linalg.norm(obj.size[:2])/2  # conservative footprint under changed grasp yaw
    return all(other.object_id == obj.object_id or
               np.any(np.abs(np.array(other.pose.position[:2])-position[:2]) > _xy_extent(other)+half+.001)
               for other in graph.nodes)


def plan_next_subgoal(graph: SceneGraph, task: TaskRequest,
                      robot_state: NDArray[np.float32]) -> Subgoal:
    check(robot_state.shape == (6,) and robot_state.dtype == np.float32 and
          bool(np.isfinite(robot_state).all()), "invalid robot_state")
    obj = graph.object(task.target_object_id)
    target_height = obj.size[2]
    place_pose = None
    if graph.held_object_id is not None:
        obj = graph.object(graph.held_object_id)
        place_pose = task.destination if obj.object_id == task.target_object_id else task.buffer_pose
        if obj.object_id != task.target_object_id:
            # 임시 배치 후보의 표면 높이는 유지하고, 서로 다른 박스 높이만 보정한다.
            x, y, z = place_pose.position
            place_pose = Pose((x, y, z + (obj.size[2]-target_height)/2), place_pose.quaternion_xyzw)
            if graph.source != 'mock_gt':
                # The current 11cm tray fits two separate box footprints. Reserve
                # slots from the first placement instead of occupying its centre.
                candidates = [Pose((x+dx, y, place_pose.position[2]), place_pose.quaternion_xyzw)
                              for dx in (.025, -.025)]
                place_pose = next((p for p in candidates if _free_xy(graph, obj, p.position)), None)
                check(place_pose is not None, 'buffer tray has no free placement slot')
        if graph.source == 'mock_gt':
            for other in graph.nodes:
                if other.object_id != obj.object_id:
                    check(np.linalg.norm(np.array(place_pose.position) - other.pose.position) > .065,
                          "mock placement space occupied")
        else:
            check(_free_xy(graph, obj, place_pose.position), 'placement footprint occupied')
    else:
        visited = set()
        while True:
            check(obj.object_id not in visited, "support cycle")
            visited.add(obj.object_id)
            blockers = sorted(e.source for e in graph.edges if e.relation == "on_top_of" and e.target == obj.object_id)
            if not blockers:
                break
            obj = graph.object(blockers[0])
    primitive = "PLACE" if place_pose else "PICK"
    x, y, z = (place_pose or obj.pose).position
    return Subgoal(f"{graph.episode_id}:{graph.timestamp:.9f}:{primitive}:{obj.object_id}",
                   primitive, obj.object_id, obj.pose, Pose((x, y, z + MOCK_GRASP_OFFSET)), place_pose)
