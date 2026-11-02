"""Bounded planner comparisons; identical public observations and PLACE policy.

Greedy uses object geometry and local clearance, not a target dependency chain.
Flat SG retains dependency reasoning, ablating only region-level search scores.
HSG is the unchanged project planner. No baseline receives simulator object IDs.
"""
from dataclasses import replace

import numpy as np

from pipeline.types import MOCK_GRASP_OFFSET, Pose, Subgoal, check
from .planner import (_REACH, _hidden_by_shelf, _on_top_of, _shelf_of,
                      _xy_extent, plan_next_subgoal)

METHODS = ('greedy', 'flat', 'hsg')


def _clear_objects(objects, held):
    """Same geometric top-clearance gate as relational methods, no stored edges."""
    return [obj for obj in objects if obj.object_id != held and not any(
        other.object_id not in (obj.object_id, held) and
        other.pose.position[2] > obj.pose.position[2] and
        abs(other.pose.position[2]-other.size[2]/2-obj.pose.position[2]-obj.size[2]/2) < .006 and
        np.all(np.abs(np.array(other.pose.position[:2])-obj.pose.position[:2]) < _xy_extent(other)+_xy_extent(obj))
        for other in objects)]


def plan_baseline(method, graph, task, robot_state):
    check(method in METHODS, 'unknown benchmark planner')
    if method == 'hsg' or graph.held_object_id is not None:
        return plan_next_subgoal(graph, task, robot_state)
    check(robot_state.shape == (6,) and robot_state.dtype == np.float32 and np.isfinite(robot_state).all(),
          'invalid robot_state')
    target = next((obj for obj in graph.nodes if obj.object_id == task.target_object_id), None)
    if method == 'flat' and target is not None:
        # Identical target/blocker reasoning; only hidden-target search is ablated.
        return plan_next_subgoal(graph, task, robot_state)
    region = task.target_spec.region if task.target_spec else None
    if method == 'flat':
        candidates = [(obj, hidden) for key, stacks in _hidden_by_shelf(graph, _on_top_of(graph)).items()
                      if region is None or key == region for obj, hidden in stacks if hidden > 0]
        check(bool(candidates), 'target not visible and no hidden space left')
        obj = min(candidates, key=lambda pair: (-pair[1],
                  float(np.linalg.norm(np.array(pair[0].pose.position[:2])-_REACH)), pair[0].object_id))[0]
    else:
        # Flatten hierarchy into a per-object region label, preserving the same
        # task hint. Region geometry is public to all methods.
        objects = tuple(replace(obj, parent_id=_shelf_of(graph, obj, graph.regions)) for obj in graph.nodes)
        candidates = _clear_objects(objects, graph.held_object_id)
        if target is not None and any(obj.object_id == target.object_id for obj in candidates):
            obj = target
        else:
            candidates = [obj for obj in candidates if obj.parent_id in graph.regions and
                          graph.regions[obj.parent_id].get('region_type') == 'shelf' and
                          (region is None or obj.parent_id == region)]
            check(bool(candidates), 'no locally clear shelf object')
            anchor = np.array(target.pose.position[:2]) if target is not None else _REACH
            obj = min(candidates, key=lambda obj: (float(np.linalg.norm(np.array(obj.pose.position[:2])-anchor)),
                                                    obj.object_id))
    x, y, z = obj.pose.position
    return Subgoal(f'{graph.episode_id}:{graph.timestamp:.9f}:PICK:{obj.object_id}',
                   'PICK', obj.object_id, obj.pose, Pose((x, y, z+MOCK_GRASP_OFFSET)))
