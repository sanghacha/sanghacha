"""A/B/C comparison with fixed ACT and public Subgoal contracts.

A: a flat object list and nearest-to-robot-base greedy exploration.
B: HSG planner with hidden-space estimation disabled, not zero-candidate search.
C: same HSG planner with observation-derived hidden-space estimates enabled.

No method receives true hidden object IDs or positions. All methods share target
grounding, local geometry gates and a hidden-cue-free PLACE implementation.
"""
from dataclasses import replace

import numpy as np

from pipeline.types import MOCK_GRASP_OFFSET, Pose, Subgoal, check
from .baselines import _clear_objects
from .indexed import IndexedPlanner
from .planner import _SHELF, _TABLE, _xy_extent

CONDITIONS = ('a_greedy', 'b_no_hidden', 'c_full')
ROBOT_BASE = (0., 0., 0.)


def distance(obj):
    """Distance to fixed base origin, never TCP, target, or workspace centre."""
    return float(np.linalg.norm(np.asarray(obj.pose.position)-ROBOT_BASE))


def flat_regions(graph):
    """Public region labels derived from configured geometry, not graph edges."""
    result = {}
    for obj in graph.nodes:
        region = next((key for key,value in graph.regions.items() if value.get('region_type')=='shelf'
            and all(abs(obj.pose.position[i]-value['center'][i])<=value['size'][i]/2 for i in (0,1))), None)
        if region is None:
            region = next((key for key,value in graph.regions.items() if value.get('region_type')=='table'
                and all(abs(obj.pose.position[i]-value['center'][i])<=value['size'][i]/2 for i in (0,1))), None)
        # Trays are excluded from exploration even though tabletop contains them.
        in_tray = any(value.get('region_type')=='tray' and all(
            abs(obj.pose.position[i]-value['center'][i])<=value['size'][i]/2 for i in (0,1))
            for value in graph.regions.values())
        result[obj.object_id] = None if in_tray else region
    return result


def candidates(objects, regions, held, task):
    hint = task.target_spec.region if task.target_spec else None
    return [obj for obj in _clear_objects(objects,held) if regions.get(obj.object_id) is not None
            and (hint is None or regions[obj.object_id]==hint)]


def pick_goal(graph, obj):
    x,y,z = obj.pose.position
    return Subgoal(f'{graph.episode_id}:{graph.timestamp:.9f}:PICK:{obj.object_id}',
                   'PICK',obj.object_id,obj.pose,Pose((x,y,z+MOCK_GRASP_OFFSET)))


def place_goal(graph, task):
    """Identical deterministic placement for A/B/C, without hidden estimates."""
    obj = graph.object(graph.held_object_id)
    def free(p):
        half = np.linalg.norm(obj.size[:2])/2
        return all(other.object_id==obj.object_id or np.any(
            np.abs(np.array(other.pose.position[:2])-p[:2])>_xy_extent(other)+half+.001)
            for other in graph.nodes)
    if obj.object_id==task.target_object_id:
        place = task.destination
        check(free(place.position),'placement footprint occupied')
    else:
        # Infer the configured buffer surface from the public target size prior.
        height = task.target_spec.size_m[2] if task.target_spec and task.target_spec.size_m else .03
        target = next((o for o in graph.nodes if o.object_id==task.target_object_id),None)
        if target is not None:
            height = target.size[2]
        x,y,z = task.buffer_pose.position
        z += (obj.size[2]-height)/2
        options = [Pose((x+dx,y,z),task.buffer_pose.quaternion_xyzw) for dx in (.025,-.025)]
        place = next((p for p in options if free(p.position)),None)
        if place is None:
            for key,region in sorted(graph.regions.items()):
                if region.get('region_type')!='shelf':
                    continue
                hx,hy = np.array(region['size'])/2-_SHELF['wall_thickness']-np.linalg.norm(obj.size[:2])/2-.003
                cells = [np.array(region['center'])+(dx,dy) for dx in np.linspace(-hx,hx,7) for dy in np.linspace(-hy,hy,5)]
                for xy in sorted(cells,key=lambda xy:(float(np.linalg.norm(xy)),float(xy[0]),float(xy[1]))):
                    if free(xy):
                        place = Pose((float(xy[0]),float(xy[1]),region.get('floor_top_m',.006)+obj.size[2]/2+.0002))
                        break
                if place is not None:
                    break
        if place is None:
            place = next((Pose((x,y,obj.size[2]/2+.0002)) for x,y in _TABLE if free((x,y))),None)
        check(place is not None,'no shared placement slot available')
    x,y,z = place.position
    return Subgoal(f'{graph.episode_id}:{graph.timestamp:.9f}:PLACE:{obj.object_id}',
                   'PLACE',obj.object_id,obj.pose,Pose((x,y,z+MOCK_GRASP_OFFSET)),place)


class _GraphPlanner(IndexedPlanner):
    def __init__(self, use_hidden):
        super().__init__('hsg_incremental')
        self.use_hidden = use_hidden
        self.last_candidates = []
        self.eligible_ids = set()

    def _hidden(self,key):
        # This is a feature-use ablation: visible 3D poses remain unchanged.
        return super()._hidden(key) if self.use_hidden else 0

    def _search(self, region):
        pool = [self.objects[key] for key in self.eligible_ids]
        # Do not turn 'no estimated hidden space' into immediate failure.
        check(bool(pool),'no eligible observed exploration object')
        totals = {}
        for obj in pool:
            parent = self.region_of.get(obj.object_id)
            n = self.tops.get(parent,{}).get(obj.object_id,(0,))[0]
            totals[parent] = totals.get(parent,0)+n
        ranked = []
        for obj in pool:
            parent = self.region_of[obj.object_id]
            hidden = self.tops.get(parent,{}).get(obj.object_id,(0,))[0]
            total = totals.get(parent,0)
            ranked.append(((-total,-hidden,distance(obj),obj.object_id),obj))
        ranked.sort(key=lambda row:row[0])
        self.last_candidates = [dict(object_id=obj.object_id,region=self.region_of[obj.object_id],
            hidden_count=-score[1],region_hidden_count=-score[0],distance_to_robot_base_m=score[2])
            for score,obj in ranked]
        return ranked[0][1]


class ComparisonPlanner:
    def __init__(self, condition):
        check(condition in CONDITIONS,'unknown ACT comparison condition')
        self.condition = condition
        self.engine = None if condition=='a_greedy' else _GraphPlanner(condition=='c_full')
        self.last_decision = {}
        self._episode = None
        self._last_timestamp = -1.
        self._placed = set()

    def completed(self, goal):
        """The orchestrator/benchmark supplies actual completion events."""
        if goal.primitive=='PLACE':
            self._placed.add(goal.object_id)

    def plan_next_subgoal(self, graph, task, robot_state):
        check(graph.source=='rgbd','ACT comparison requires observed RGB-D graphs')
        check(robot_state.shape==(6,) and robot_state.dtype==np.float32 and np.isfinite(robot_state).all(),
              'invalid robot_state')
        if graph.episode_id!=self._episode or graph.timestamp<self._last_timestamp:
            self._episode = graph.episode_id
            self._placed.clear()
        self._last_timestamp = graph.timestamp
        if graph.held_object_id is not None:
            goal = place_goal(graph,task)
            self.last_decision = dict(mode='place',hidden_estimates_used=False,selected=goal.object_id)
            return goal
        # All methods receive the same feasible, not-already-cleared candidate set.
        regions = flat_regions(graph)
        target = next((obj for obj in graph.nodes if obj.object_id==task.target_object_id),None)
        # Region hints concern initial identification/search, not the location
        # of an already identified target that may have moved.
        request = task if target is None or task.target_spec is None else replace(
            task,target_spec=replace(task.target_spec,region=None))
        pool = candidates(graph.nodes,regions,None,request)
        clear = {obj.object_id for obj in _clear_objects(graph.nodes,None)}
        if target is not None and target.object_id in clear:
            goal = pick_goal(graph,target)
            self.last_decision = dict(mode='target_pick',hidden_estimates_used=False,selected=goal.object_id)
            return goal
        eligible = {obj.object_id for obj in pool if target is not None or obj.object_id not in self._placed}
        # Graph methods see only current tracks; exclusion of already-placed
        # tracks is shared history, not privileged information about hidden ones.
        if self.condition=='a_greedy':
            ranked = sorted((obj for obj in pool if obj.object_id in eligible),key=lambda o:(distance(o),o.object_id))
            check(bool(ranked),'no eligible observed exploration object')
            goal = pick_goal(graph,ranked[0])
            detail = [dict(object_id=o.object_id,distance_to_robot_base_m=distance(o)) for o in ranked]
        else:
            # Shared candidate gating is enforced for search. B/C retain all
            # nodes for support reasoning and collision/placement checks.
            self.engine.eligible_ids = eligible
            goal = self.engine.plan_next_subgoal(graph,task,robot_state)
            detail = self.engine.last_candidates if target is None else []
        self.last_decision = dict(mode='search' if target is None else 'unblock',condition=self.condition,
            hidden_estimates_used=self.condition=='c_full' and target is None,
            selected=goal.object_id,candidates=detail)
        return goal
