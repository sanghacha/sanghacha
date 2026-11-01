"""모듈 B: 다음 PICK/PLACE를 선택한다.

- 목표가 그래프에 있으면 목표 위에 쌓인 물체를 위에서부터 치운다(GT 그래프 기존 동작).
- 부분 관측 그래프(source="rgbd")에 목표가 없으면 숨은 공간을 탐색한다. 더미마다 선반 바닥과 맨 위 물체 사이의
  빈 높이로 밑에 숨은 물체 수('?')를 추정하고, '?'가 가장 많은 선반의 그 더미 맨 위 물체를 치운다.
  목표 식별(드러난 물체에 목표 ID를 붙이는 일)은 A의 몫이고, B는 ID가 그래프에 나타나면 집는다.
- 그래프에 on_top_of 간선이 없으면(rgbd) 위치·크기로 쌓임 관계를 계산한다.
- 치운 물체는 임시 트레이에 놓고, 트레이가 차면 숨은 공간이 가장 적은 선반의 빈 바닥에, 선반도 차면 트레이 바깥 탁자 위에 놓는다.
"""

import json

import numpy as np
from numpy.typing import NDArray

from pipeline.mock_data import ROOT
from pipeline.types import MOCK_GRASP_OFFSET, Pose, SceneGraph, Subgoal, TaskRequest, check

_SHELF = json.loads((ROOT / "configs/envs/so101.json").read_text(encoding="utf-8"))["shelf_scene"]
_FLOOR = _SHELF["floor_thickness"]                                     # 선반 바닥 윗면 높이
_BOX_H = float(np.mean([size[2] for size in _SHELF["object_sizes"]]))  # 숨은 물체 하나의 높이 추정값
_STACK_GAP = .006                                                      # 위·아래 면 사이 허용 간격(rgbd 추정 오차 포함)
_TABLE = [(x, y) for x in (.08, .12, .04) for y in (-.27, .27)]        # 트레이 바깥쪽(|y|=0.27) 탁자 위 자리. 트레이 안쪽 탁자는 전문가 팔이 닿지 않음(IK 확인)
_REACH = np.array([.2, 0.])                                            # 동점일 때 우선할 팔 앞 중앙(선반 가장자리·모서리는 도달 실패가 잦음)


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


def _on_top_of(graph):
    """(위 물체, 아래 물체) 쌍. 그래프에 간선이 있으면 그대로, 없으면 xy 겹침과 면 높이로 계산"""
    pairs = {(e.source, e.target) for e in graph.edges if e.relation == "on_top_of"}
    if pairs or graph.source != "rgbd":
        return pairs
    for upper in graph.nodes:
        for lower in graph.nodes:
            if upper.object_id == lower.object_id or graph.held_object_id in (upper.object_id, lower.object_id):
                continue
            a, b = np.array(upper.pose.position), np.array(lower.pose.position)
            gap = (a[2] - upper.size[2]/2) - (b[2] + lower.size[2]/2)
            if a[2] > b[2] and abs(gap) < _STACK_GAP and np.all(np.abs(a[:2]-b[:2]) < _xy_extent(upper)+_xy_extent(lower)):
                pairs.add((upper.object_id, lower.object_id))
    return pairs


def _hidden_below(graph, obj, on):
    """obj 밑에 숨은 물체 수 추정: 바닥~obj 밑면 높이에서 그래프에 있는 아래 물체 높이를 빼고 물체 높이로 나눈다"""
    known, current, seen = 0., obj.object_id, set()
    while current not in seen:
        seen.add(current)
        below = sorted(lower for upper, lower in on if upper == current)
        if not below:
            break
        current = below[0]
        known += graph.object(current).size[2]
    region = _shelf_of(graph, obj, graph.regions)
    floor = graph.regions.get(region, {}).get('floor_top_m', _FLOOR)
    gap = obj.pose.position[2] - obj.size[2]/2 - floor - known
    return int(round(gap / _BOX_H)) if gap > _BOX_H/2 else 0


def _shelf_of(graph, obj, shelves):
    """물체가 속한 선반 ID(부모 그룹을 따라 올라가며 찾음, 예: stack 그룹 → 선반). 없으면 None"""
    current = obj.parent_id
    while current is not None and current not in shelves:
        current = graph.groups.get(current)
    return current


def _hidden_by_shelf(graph, on):
    """선반별 [(더미 맨 위 물체, 그 밑 숨은 수)]"""
    shelves = ({key: [] for key, region in graph.regions.items() if region.get('region_type') in ('shelf', 'table')}
               if graph.regions else {shelf["id"]: [] for shelf in _SHELF["shelves"]})
    covered = {lower for _, lower in on}
    for obj in graph.nodes:
        shelf = _shelf_of(graph, obj, shelves)
        if shelf is not None and obj.object_id not in covered and obj.object_id != graph.held_object_id:
            shelves[shelf].append((obj, _hidden_below(graph, obj, on)))
    return shelves


def _search_pick(graph, on, region=None):
    """목표가 안 보일 때: '?'가 가장 많은 선반의, '?'가 가장 많은 더미 맨 위 물체(같으면 팔 앞 중앙에 가까운 것)"""
    shelves = _hidden_by_shelf(graph, on)
    candidates = [(-sum(n for _, n in stacks), -n, float(np.linalg.norm(np.array(obj.pose.position[:2]) - _REACH)), obj.object_id)
                  for key, stacks in shelves.items() if region is None or key == region for obj, n in stacks if n > 0]
    check(bool(candidates), "target not visible and no hidden space left")
    return graph.object(min(candidates)[3])


def _shelf_slot(graph, obj, on):
    """임시 트레이가 찼을 때: 숨은 공간이 가장 적은 선반의 빈 바닥 칸(팔 앞 중앙에 가까운 순)"""
    shelves = _hidden_by_shelf(graph, on)
    regions = ([dict(value, id=key) for key, value in graph.regions.items() if value.get('region_type') == 'shelf']
               if graph.regions else _SHELF['shelves'])
    order = sorted(regions, key=lambda s: sum(n for _, n in shelves[s["id"]]))
    for shelf in order:
        # 벽 안쪽에서 물체가 들어가는 범위를 촘촘히 훑는다(빈 칸이 정해진 6칸 자리 사이에 있을 수 있음)
        hx, hy = np.array(shelf["size"])/2 - _SHELF["wall_thickness"] - np.linalg.norm(obj.size[:2])/2 - .003
        grid = [np.array(shelf["center"]) + (dx, dy) for dx in np.linspace(-hx, hx, 7) for dy in np.linspace(-hy, hy, 5)]
        cells = sorted(grid, key=lambda xy: np.linalg.norm(xy - _REACH))
        for xy in cells:
            if _free_xy(graph, obj, xy):
                return Pose((float(xy[0]), float(xy[1]), shelf.get('floor_top_m', _FLOOR) + obj.size[2]/2 + .0002))
    for xy in _TABLE:                                                  # 선반도 꽉 찼으면 트레이 바깥 탁자 위 빈 곳(선반 밖이라 숨은 공간 계산에 안 들어감)
        if _free_xy(graph, obj, np.array(xy)):
            return Pose((xy[0], xy[1], obj.size[2]/2 + .0002))
    return None


def plan_next_subgoal(graph: SceneGraph, task: TaskRequest,
                      robot_state: NDArray[np.float32]) -> Subgoal:
    check(robot_state.shape == (6,) and robot_state.dtype == np.float32 and
          bool(np.isfinite(robot_state).all()), "invalid robot_state")
    on = _on_top_of(graph)
    hidden_target = graph.source == "rgbd" and task.target_object_id not in {obj.object_id for obj in graph.nodes}
    obj = None if hidden_target else graph.object(task.target_object_id)
    target_height = obj.size[2] if obj else (task.target_spec.size_m[2] if task.target_spec and task.target_spec.size_m else _BOX_H)
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
                if place_pose is None:
                    place_pose = _shelf_slot(graph, obj, on)
                check(place_pose is not None, 'buffer tray and shelves have no free placement slot')
        if graph.source == 'mock_gt':
            for other in graph.nodes:
                if other.object_id != obj.object_id:
                    check(np.linalg.norm(np.array(place_pose.position) - other.pose.position) > .065,
                          "mock placement space occupied")
        else:
            check(_free_xy(graph, obj, place_pose.position), 'placement footprint occupied')
    else:
        if hidden_target:
            obj = _search_pick(graph, on, task.target_spec.region if task.target_spec else None)
        visited = set()
        while True:
            check(obj.object_id not in visited, "support cycle")
            visited.add(obj.object_id)
            blockers = sorted(upper for upper, lower in on if lower == obj.object_id)
            if not blockers:
                break
            obj = graph.object(blockers[0])
    primitive = "PLACE" if place_pose else "PICK"
    x, y, z = (place_pose or obj.pose).position
    return Subgoal(f"{graph.episode_id}:{graph.timestamp:.9f}:{primitive}:{obj.object_id}",
                   primitive, obj.object_id, obj.pose, Pose((x, y, z + MOCK_GRASP_OFFSET)), place_pose)
