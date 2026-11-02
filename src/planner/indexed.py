"""Exact indexed version of the project policy, with a 2 x 2 storage ablation.

All variants use the same spatial hash and adjacency lookup. Flat incremental
caches relations and per-object candidates; HSG additionally caches regional
aggregates and best candidates. Snapshot diffing is included in the public call.
The delta API is also exposed for A implementations that can emit exact changes.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from math import floor
from time import perf_counter_ns

import numpy as np

from pipeline.types import MOCK_GRASP_OFFSET, Pose, Subgoal, check
from .planner import _BOX_H, _FLOOR, _REACH, _SHELF, _STACK_GAP, _TABLE, _xy_extent

VARIANTS = ('flat_full', 'flat_incremental', 'hsg_full', 'hsg_incremental')


@dataclass(frozen=True)
class GraphUpdate:
    episode_id: str
    source: str
    timestamp: float
    revision: int
    base_revision: int | None
    upserted: tuple
    removed: tuple[str, ...] = ()
    groups: dict | None = None
    regions: dict | None = None
    edges: tuple | None = None
    held_object_id: str | None = None


class DeltaEncoder:
    """Exact full-snapshot diff; no pose tolerance, including metadata changes."""
    def __init__(self):
        self.episode = self.source = None
        self.timestamp = -1.
        self.revision = -1
        self.objects, self.groups, self.regions, self.edges = {}, {}, {}, ()

    def encode(self, graph):
        reset = (graph.episode_id != self.episode or graph.source != self.source or
                 graph.timestamp < self.timestamp)
        nodes = {obj.object_id: obj for obj in graph.nodes}
        previous = {} if reset else self.objects
        revision = 0 if reset else self.revision+1
        update = GraphUpdate(graph.episode_id, graph.source, graph.timestamp, revision,
            None if reset else self.revision,
            tuple(obj for key,obj in nodes.items() if previous.get(key) != obj),
            tuple(sorted(previous.keys()-nodes.keys())),
            deepcopy(graph.groups) if reset or graph.groups != self.groups else None,
            deepcopy(graph.regions) if reset or graph.regions != self.regions else None,
            graph.edges if reset or graph.edges != self.edges else None, graph.held_object_id)
        self.episode, self.source, self.timestamp, self.revision = graph.episode_id, graph.source, graph.timestamp, revision
        self.objects, self.groups, self.regions, self.edges = nodes, deepcopy(graph.groups), deepcopy(graph.regions), graph.edges
        return update


@dataclass
class Work:
    snapshot_nodes_scanned: int = 0
    changed_objects: int = 0
    spatial_pair_tests: int = 0
    summary_objects_visited: int = 0
    query_objects_visited: int = 0
    query_regions_visited: int = 0
    collision_objects_tested: int = 0
    dirty_regions: list = field(default_factory=list)
    diff_ns: int = 0
    update_ns: int = 0
    query_ns: int = 0
    total_ns: int = 0


class IndexedPlanner:
    def __init__(self, variant='hsg_incremental', *, cell_size=.05):
        check(variant in VARIANTS, 'unknown indexed planner variant')
        check(np.isfinite(cell_size) and cell_size > 0, 'invalid spatial cell size')
        self.variant, self.cell_size = variant, cell_size
        self.incremental = variant.endswith('incremental')
        self.hierarchical = variant.startswith('hsg')
        self.encoder = DeltaEncoder()
        self._api = None
        self.stats = Work()
        self._clear()

    def _clear(self):
        self.episode, self.source, self.timestamp, self.revision = None, None, -1., -1
        self.objects, self.groups, self.regions = {}, {}, {}
        self.explicit_edges, self.held = (), None
        self.geometry, self.cells, self.region_of = {}, {}, {}
        self.grid = defaultdict(set)
        self.members = defaultdict(set)
        self.on = set()
        self.above, self.below = defaultdict(set), defaultdict(set)
        self.tops = defaultdict(dict)
        self.totals, self.best = {}, {}

    def _region(self, obj):
        valid = self.regions if self.regions else {s['id']:s for s in _SHELF['shelves']}
        parent = obj.parent_id
        while parent is not None and parent not in valid:
            parent = self.groups.get(parent)
        return parent

    def _keys(self, x, y, hx, hy):
        scale = self.cell_size
        return {(i,j) for i in range(floor((x-hx)/scale),floor((x+hx)/scale)+1)
                      for j in range(floor((y-hy)/scale),floor((y+hy)/scale)+1)}

    def _remove_geometry(self, key):
        for cell in self.cells.pop(key, ()):
            self.grid[cell].discard(key)
            if not self.grid[cell]:
                del self.grid[cell]
        parent = self.region_of.pop(key, None)
        self.members[parent].discard(key)
        self.geometry.pop(key, None)

    def _put_geometry(self, obj):
        key = obj.object_id
        x,y,z = obj.pose.position
        hx,hy = map(float,_xy_extent(obj))
        self.geometry[key] = (x,y,z,hx,hy,obj.size[2]/2)
        self.cells[key] = self._keys(x,y,hx,hy)
        for cell in self.cells[key]:
            self.grid[cell].add(key)
        parent = self._region(obj)
        self.region_of[key] = parent
        self.members[parent].add(key)

    def _neighbors(self, key):
        result = set()
        for cell in self.cells.get(key, ()):
            result.update(self.grid[cell])
        result.discard(key)
        return result

    def _support(self, a, b):
        self.stats.spatial_pair_tests += 1
        if self.held in (a,b):
            return None
        if self.geometry[a][2] < self.geometry[b][2]:
            a,b = b,a
        ax,ay,az,ahx,ahy,ahz = self.geometry[a]
        bx,by,bz,bhx,bhy,bhz = self.geometry[b]
        if az > bz and abs((az-ahz)-(bz+bhz)) < _STACK_GAP and abs(ax-bx) < ahx+bhx and abs(ay-by) < ahy+bhy:
            return a,b
        return None

    def _drop_edge(self, pair):
        a,b = pair
        self.on.discard(pair)
        self.above[b].discard(a)
        self.below[a].discard(b)

    def _add_edge(self, pair):
        a,b = pair
        self.on.add(pair)
        self.above[b].add(a)
        self.below[a].add(b)

    def _search_regions(self):
        if self.regions:
            return [key for key,r in self.regions.items() if r.get('region_type') in ('shelf','table')]
        return [s['id'] for s in _SHELF['shelves']]

    def _hidden(self, key):
        obj = self.objects[key]
        known, current, seen = 0., key, set()
        while current not in seen:
            seen.add(current)
            lower = sorted(self.below.get(current, ()))
            if not lower:
                break
            current = lower[0]
            known += self.objects[current].size[2]
        region = self.region_of[key]
        floor_top = self.regions.get(region,{}).get('floor_top_m',_FLOOR)
        gap = obj.pose.position[2]-obj.size[2]/2-floor_top-known
        return int(round(gap/_BOX_H)) if gap > _BOX_H/2 else 0

    def _refresh_summaries(self, dirty):
        search = set(self._search_regions())
        for region in dirty:
            self.tops.pop(region, None)
            self.totals.pop(region, None)
            self.best.pop(region, None)
            if region not in search:
                continue
            tops = {}
            for key in self.members.get(region, ()):
                self.stats.summary_objects_visited += 1
                if key != self.held and not self.above.get(key):
                    n = self._hidden(key)
                    x,y = self.objects[key].pose.position[:2]
                    tops[key] = (n,float(np.linalg.norm(np.array([x,y])-_REACH)),key)
            self.tops[region] = tops
            if self.hierarchical:
                self.totals[region] = sum(n for n,_,_ in tops.values())
                candidates = [(-n,d,key) for n,d,key in tops.values() if n>0]
                self.best[region] = min(candidates) if candidates else None

    def _apply(self, update):
        reset = update.base_revision is None
        check(type(update.revision) is int and update.revision>=0 and (not reset or update.revision==0),
              'invalid graph delta revision')
        if reset:
            self._clear()
        else:
            check(update.episode_id == self.episode and update.source == self.source and
                  update.base_revision == self.revision and update.revision == self.revision+1 and
                  update.timestamp >= self.timestamp, 'stale or missing graph delta; resynchronize with a snapshot')
        check(update.source in ('rgbd','mujoco_gt','mock_gt'), 'invalid graph update source')
        check(np.isfinite(update.timestamp) and update.timestamp >= 0, 'invalid graph update timestamp')
        upserts = {obj.object_id:obj for obj in update.upserted}
        check(len(upserts)==len(update.upserted) and not set(upserts)&set(update.removed), 'duplicate delta object')
        check(all(key in self.objects for key in update.removed), 'removed delta object is missing')
        groups = update.groups if update.groups is not None else self.groups
        regions = update.regions if update.regions is not None else self.regions
        if update.groups is not None:
            for key in groups:
                seen, current = set(), key
                while current is not None:
                    check(current in groups and current not in seen,'invalid delta hierarchy')
                    seen.add(current)
                    current = groups[current]
        check(set(regions)<=groups.keys(),'delta regions must reference groups')
        check(all(obj.parent_id in groups for obj in upserts.values()),'missing delta object parent')
        if update.groups is not None:
            check(all(obj.parent_id in groups for key,obj in self.objects.items()
                      if key not in upserts and key not in update.removed),'missing unchanged object parent')
        metadata_changed = update.groups is not None or update.regions is not None
        changed = set(upserts)|set(update.removed)
        dirty = {self.region_of.get(key) for key in changed}
        if self.held != update.held_object_id:
            changed.update(key for key in (self.held,update.held_object_id) if key is not None)
            dirty.update(self.region_of.get(key) for key in changed)
        old_mode = bool(self.explicit_edges) or self.source != 'rgbd'
        if update.groups is not None:
            self.groups = deepcopy(update.groups)
        if update.regions is not None:
            self.regions = deepcopy(update.regions)
        if metadata_changed:
            dirty.update(self.members)
        for key in set(upserts)|set(update.removed):
            self._remove_geometry(key)
            self.objects.pop(key,None)
        self.objects.update(upserts)
        for obj in upserts.values():
            self._put_geometry(obj)
        if metadata_changed:
            # Parent hierarchy/floor changes can affect otherwise unchanged objects.
            self.members.clear()
            for key,obj in self.objects.items():
                self.region_of[key] = self._region(obj)
                self.members[self.region_of[key]].add(key)
            dirty.update(self.members)
        self.held = update.held_object_id
        check(self.held is None or self.held in self.objects, 'unknown held delta object')
        if update.edges is not None:
            self.explicit_edges = tuple(e for e in update.edges if e.relation == 'on_top_of')
        explicit = bool(self.explicit_edges) or update.source != 'rgbd'
        if explicit:
            wanted = {(e.source,e.target) for e in self.explicit_edges}
            check(all(a in self.objects and b in self.objects for a,b in wanted), 'invalid support endpoint')
            removed, added = self.on-wanted, wanted-self.on
        else:
            # An explicit→inferred transition invalidates every inferred relation.
            if old_mode:
                changed.update(self.objects)
            removed = {pair for key in changed for pair in
                       [(key,b) for b in self.below.get(key,())]+[(a,key) for a in self.above.get(key,())]}
            candidates = {tuple(sorted((key,other))) for key in changed if key in self.objects
                          for other in self._neighbors(key)}
            added = set()
            for a,b in candidates:
                pair = self._support(a,b)
                if pair:
                    added.add(pair)
        touched = changed | {key for pair in removed|added for key in pair}
        dirty.update(self.region_of.get(key) for key in touched)
        for pair in removed:
            self._drop_edge(pair)
        for pair in added:
            self._add_edge(pair)
        for key in update.removed:
            self.above.pop(key,None)
            self.below.pop(key,None)
        # A support can cross region labels. Hidden-height traversal then affects
        # ancestors in other regions too; follow the actual dependency closure.
        pending = list(touched & self.objects.keys())
        visited = set(pending)
        while pending:
            key = pending.pop()
            dirty.add(self.region_of[key])
            for upper in self.above.get(key,()):
                if upper not in visited:
                    visited.add(upper)
                    pending.append(upper)
        dirty.discard(None)
        self.stats.changed_objects = len(changed)
        self.stats.dirty_regions = sorted(dirty)
        self._refresh_summaries(dirty)
        self.episode,self.source,self.timestamp,self.revision = update.episode_id,update.source,update.timestamp,update.revision

    def _totals(self):
        if self.hierarchical:
            self.stats.query_regions_visited += len(self.totals)
            return self.totals
        totals = {}
        for region,tops in self.tops.items():
            self.stats.query_regions_visited += 1
            self.stats.query_objects_visited += len(tops)
            totals[region] = sum(n for n,_,_ in tops.values())
        return totals

    def _search(self, region):
        regions = self._search_regions() if region is None else [region]
        candidates = []
        for key in regions:
            self.stats.query_regions_visited += 1
            if self.hierarchical:
                best = self.best.get(key)
                if best is not None:
                    candidates.append((-self.totals[key],*best))
            else:
                tops = self.tops.get(key,{})
                self.stats.query_objects_visited += len(tops)
                total = sum(n for n,_,_ in tops.values())
                candidates.extend((-total,-n,d,ident) for n,d,ident in tops.values() if n>0)
        check(bool(candidates), 'target not visible and no hidden space left')
        return self.objects[min(candidates)[-1]]

    def _free_xy(self, obj, position):
        half = np.linalg.norm(obj.size[:2])/2
        radius = half+.001
        candidates = set()
        for cell in self._keys(position[0],position[1],radius,radius):
            candidates.update(self.grid.get(cell,()))
        for key in sorted(candidates):
            if key == obj.object_id:
                continue
            self.stats.collision_objects_tested += 1
            x,y,_,hx,hy,_ = self.geometry[key]
            if abs(x-position[0]) <= hx+half+.001 and abs(y-position[1]) <= hy+half+.001:
                return False
        return True

    def _slot(self,obj):
        totals = self._totals()
        regions = ([dict(value,id=key) for key,value in self.regions.items() if value.get('region_type')=='shelf']
                   if self.regions else _SHELF['shelves'])
        for shelf in sorted(regions,key=lambda r:totals.get(r['id'],0)):
            hx,hy = np.array(shelf['size'])/2-_SHELF['wall_thickness']-np.linalg.norm(obj.size[:2])/2-.003
            grid = [np.array(shelf['center'])+(dx,dy) for dx in np.linspace(-hx,hx,7) for dy in np.linspace(-hy,hy,5)]
            for xy in sorted(grid,key=lambda xy:np.linalg.norm(xy-_REACH)):
                if self._free_xy(obj,xy):
                    return Pose((float(xy[0]),float(xy[1]),shelf.get('floor_top_m',_FLOOR)+obj.size[2]/2+.0002))
        for xy in _TABLE:
            if self._free_xy(obj,np.array(xy)):
                return Pose((xy[0],xy[1],obj.size[2]/2+.0002))
        return None

    def _query(self, task, robot_state):
        check(robot_state.shape==(6,) and robot_state.dtype==np.float32 and np.isfinite(robot_state).all(), 'invalid robot_state')
        obj = self.objects.get(task.target_object_id)
        hidden = self.source=='rgbd' and obj is None
        if not hidden:
            check(obj is not None, f'unknown object: {task.target_object_id}')
        height = obj.size[2] if obj else (task.target_spec.size_m[2] if task.target_spec and task.target_spec.size_m else _BOX_H)
        place = None
        if self.held is not None:
            obj = self.objects[self.held]
            place = task.destination if self.held==task.target_object_id else task.buffer_pose
            if self.held != task.target_object_id:
                x,y,z = place.position
                place = Pose((x,y,z+(obj.size[2]-height)/2),place.quaternion_xyzw)
                if self.source != 'mock_gt':
                    options = [Pose((x+dx,y,place.position[2]),place.quaternion_xyzw) for dx in (.025,-.025)]
                    place = next((p for p in options if self._free_xy(obj,p.position)),None)
                    if place is None:
                        place = self._slot(obj)
                    check(place is not None, 'buffer tray and shelves have no free placement slot')
            if self.source=='mock_gt':
                for other in self.objects.values():
                    self.stats.collision_objects_tested += 1
                    if other.object_id != obj.object_id:
                        check(np.linalg.norm(np.array(place.position)-other.pose.position)>.065,'mock placement space occupied')
            else:
                check(self._free_xy(obj,place.position),'placement footprint occupied')
        else:
            if hidden:
                obj = self._search(task.target_spec.region if task.target_spec else None)
            seen = set()
            while True:
                self.stats.query_objects_visited += 1
                check(obj.object_id not in seen,'support cycle')
                seen.add(obj.object_id)
                blockers = sorted(self.above.get(obj.object_id,()))
                if not blockers:
                    break
                obj = self.objects[blockers[0]]
        primitive = 'PLACE' if place else 'PICK'
        x,y,z = (place or obj.pose).position
        return Subgoal(f'{self.episode}:{self.timestamp:.9f}:{primitive}:{obj.object_id}',
                       primitive,obj.object_id,obj.pose,Pose((x,y,z+MOCK_GRASP_OFFSET)),place)

    def plan_next_subgoal(self, graph, task, robot_state):
        check(self._api in (None,'snapshot'),'do not mix delta and snapshot APIs on one planner')
        self._api = 'snapshot'
        start = perf_counter_ns()
        self.stats = Work(snapshot_nodes_scanned=len(graph.nodes))
        if self.incremental:
            update = self.encoder.encode(graph)
        else:
            update = GraphUpdate(graph.episode_id,graph.source,graph.timestamp,0,None,graph.nodes,
                                 groups=graph.groups,regions=graph.regions,edges=graph.edges,held_object_id=graph.held_object_id)
        self.stats.diff_ns = perf_counter_ns()-start
        return self._finish(update,task,robot_state,start)

    def plan_update(self, update, task, robot_state):
        """Exact ordered deltas; caller owns diff cost. Do not mix with snapshot API."""
        check(self.incremental,'delta API requires an incremental variant')
        check(self._api in (None,'delta'),'do not mix delta and snapshot APIs on one planner')
        self._api = 'delta'
        start = perf_counter_ns()
        self.stats = Work()
        return self._finish(update,task,robot_state,start)

    def _finish(self,update,task,robot_state,start):
        t = perf_counter_ns()
        try:
            self._apply(update)
        finally:
            self.stats.update_ns = perf_counter_ns()-t
        t = perf_counter_ns()
        try:
            return self._query(task,robot_state)
        finally:
            self.stats.query_ns = perf_counter_ns()-t
            self.stats.total_ns = perf_counter_ns()-start

    def cache_counts(self):
        return dict(objects=len(self.objects),spatial_cell_entries=sum(map(len,self.grid.values())),
                    support_edges=len(self.on),top_objects=sum(map(len,self.tops.values())),
                    region_summaries=len(self.totals),region_best_entries=len(self.best))
