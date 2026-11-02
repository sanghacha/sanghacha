from dataclasses import replace
import unittest

import numpy as np

from pipeline.mock_data import make_mock_data
from pipeline.types import Edge, Pose, SceneGraph, SceneObject, TargetSpec, TaskRequest
from planner.planner import _on_top_of, plan_next_subgoal
from planner.indexed import DeltaEncoder, IndexedPlanner, VARIANTS


STATE = np.zeros(6,np.float32)


def scene(nodes, *, timestamp=0., episode='indexed', held=None, edges=(), floor=.006):
    regions = {key:dict(region_type='shelf',center=[x,0.],size=[.17,.13],floor_top_m=floor)
               for key,x in [('shelf_1',.24),('shelf_2',.5)]}
    groups = {'workspace':None,'stack':'shelf_1','unassigned':'workspace',
              'shelf_1':'workspace','shelf_2':'workspace','buffer_tray':'workspace','output_tray':'workspace'}
    return SceneGraph(tuple(nodes),tuple(edges),groups,held,timestamp,episode,source='rgbd',regions=regions)


def box(key,x=.24,y=0.,z=.02,parent='shelf_1'):
    return SceneObject(key,Pose((x,y,z)),(.028,)*3,parent_id=parent)


def task(target='missing',region=None):
    return TaskRequest(target,Pose((.08,.18,.02)),Pose((.08,-.18,.02)),TargetSpec((.9,.2,.1),region=region))


def result(call):
    try:
        return call()
    except ValueError as exc:
        return str(exc)


class IndexedPlannerTests(unittest.TestCase):
    def assert_equivalent(self,planner,graph,request):
        expected=result(lambda:plan_next_subgoal(graph,request,STATE))
        actual=result(lambda:planner.plan_next_subgoal(graph,request,STATE))
        self.assertEqual(actual,expected)
        self.assertEqual(planner.on,_on_top_of(graph))

    def test_same_policy_including_place_and_failures_in_mock_fixtures(self):
        for variant in VARIANTS:
            planner=IndexedPlanner(variant)
            for case in ('blocked','accessible','after_move','missing_target'):
                data=make_mock_data(case)
                self.assert_equivalent(planner,data.graph,data.task)
                for held in ('A','B'):
                    if any(obj.object_id==held for obj in data.graph.nodes):
                        self.assert_equivalent(planner,replace(data.graph,held_object_id=held),data.task)

    def test_stream_insert_delete_move_reparent_held_floor_and_episode_reset(self):
        rng=np.random.default_rng(913)
        nodes=[box(f'o{i}',x=.24+(i%4)*.05,y=(i//4%3)*.05,z=.02+(i//12)*.028) for i in range(36)]
        planners=[IndexedPlanner(v) for v in VARIANTS]
        for tick in range(35):
            if tick%5==0:
                index=int(rng.integers(len(nodes)))
                obj=nodes[index]
                nodes[index]=replace(obj,pose=Pose((obj.pose.position[0]+.003,obj.pose.position[1],obj.pose.position[2])))
            if tick==5: nodes.pop(3)
            if tick==8: nodes.append(box('added',.5,z=.076,parent='shelf_2'))
            if tick==12: nodes[0]=replace(nodes[0],parent_id='shelf_2')
            held=nodes[0].object_id if tick in (15,16) else None
            graph=scene(nodes,timestamp=tick if tick<25 else tick-25,
                        episode='first' if tick<25 else 'second',held=held,floor=.007 if tick>18 else .006)
            for planner in planners:
                self.assert_equivalent(planner,graph,task())
                self.assert_equivalent(planner,graph,task(nodes[1].object_id))

    def test_cross_region_support_and_explicit_inferred_mode_switch(self):
        nodes=[box('lower',z=.02),box('middle',z=.048,parent='shelf_2'),box('upper',z=.076)]
        graphs=[scene(nodes),scene(nodes,timestamp=1,edges=[Edge('upper','lower','on_top_of')]),
                scene(nodes[:2],timestamp=2),scene([replace(nodes[0],size=(.028,.028,.03)),*nodes[1:]],timestamp=3)]
        for variant in VARIANTS:
            planner=IndexedPlanner(variant)
            for graph in graphs:
                self.assert_equivalent(planner,graph,task('lower'))
                self.assert_equivalent(planner,graph,task())

    def test_placement_collision_and_overflow_slots_match_legacy(self):
        nodes=[box('held',.15,z=.15,parent='unassigned'),box('left',.105,-.18),box('right',.055,-.18),
               box('target',.24,z=.02),box('upper',.24,z=.048)]
        for variant in VARIANTS:
            planner=IndexedPlanner(variant)
            for offset in (0.,-.000000001,.000000001):
                changed=[replace(obj,pose=Pose((obj.pose.position[0]+offset,*obj.pose.position[1:]))) for obj in nodes]
                self.assert_equivalent(planner,scene(changed,held='held'),task('target'))

    def test_delta_replay_detects_gaps_and_snapshot_api_cannot_silently_mix(self):
        encoder=DeltaEncoder()
        graphs=[scene([box('top',z=.076)],timestamp=i) for i in range(3)]
        updates=[encoder.encode(graph) for graph in graphs]
        planner=IndexedPlanner()
        self.assertEqual(planner.plan_update(updates[0],task(),STATE),plan_next_subgoal(graphs[0],task(),STATE))
        with self.assertRaisesRegex(ValueError,'stale or missing'):
            planner.plan_update(updates[2],task(),STATE)
        planner.plan_update(updates[1],task(),STATE)
        self.assertEqual(planner.stats.spatial_pair_tests,0)
        self.assertEqual(planner.stats.summary_objects_visited,0)
        with self.assertRaisesRegex(ValueError,'do not mix'):
            planner.plan_next_subgoal(graphs[2],task(),STATE)

    def test_exact_delta_has_no_pose_tolerance_and_does_not_alias_metadata(self):
        encoder=DeltaEncoder()
        original=scene([box('top',z=.076)])
        encoder.encode(original)
        changed=replace(original,nodes=(replace(original.nodes[0],pose=Pose((.24,0.,.076+1e-12))),))
        update=encoder.encode(changed)
        self.assertEqual(len(update.upserted),1)
        original.regions['shelf_1']['floor_top_m']=.01
        self.assertIsNotNone(encoder.encode(original).regions)

    def test_same_episode_time_reversal_resynchronizes_and_floor_invalidates_summary(self):
        planner=IndexedPlanner()
        for timestamp,floor in [(0.,.006),(2.,.006),(1.,.062)]:
            self.assert_equivalent(planner,scene([box('top',z=.076)],timestamp=timestamp,floor=floor),task())

    def test_delta_rejects_hierarchy_cycles_before_mutating_valid_state(self):
        encoder=DeltaEncoder();planner=IndexedPlanner()
        first=encoder.encode(scene([box('top',z=.076)]))
        planner.plan_update(first,task(),STATE)
        update=encoder.encode(scene([box('top',z=.076)],timestamp=1.))
        bad=replace(update,groups={'workspace':None,'shelf_1':'shelf_2','shelf_2':'shelf_1'})
        with self.assertRaisesRegex(ValueError,'invalid delta hierarchy'):
            planner.plan_update(bad,task(),STATE)
        planner.plan_update(update,task(),STATE)

    def test_spatial_hash_covers_boundary_contacts_and_large_boxes(self):
        nodes=[box('base',.0499,z=.02),box('upper',.05,z=.048),
               replace(box('large',.13,z=.02),size=(.2,.2,.028))]
        for cell_size in (.01,.05,.5):
            planner=IndexedPlanner(cell_size=cell_size)
            self.assert_equivalent(planner,scene(nodes),task('base'))

    def test_hierarchy_summary_reduces_hidden_query_visits_without_changing_goal(self):
        nodes=[box(str(i),.24+(i%5)*.06,y=(i//5)*.06,z=.076) for i in range(30)]
        flat,hsg=IndexedPlanner('flat_incremental'),IndexedPlanner('hsg_incremental')
        graph=scene(nodes)
        a=flat.plan_next_subgoal(graph,task(),STATE)
        b=hsg.plan_next_subgoal(graph,task(),STATE)
        self.assertEqual(a,b)
        self.assertGreater(flat.stats.query_objects_visited,hsg.stats.query_objects_visited)
        self.assertEqual(hsg.cache_counts()['top_objects'],30)


if __name__=='__main__':
    unittest.main()
