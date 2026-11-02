from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from action_generator.goal import encode_goal
from pipeline.types import Edge, Pose, SceneGraph, SceneObject, TargetSpec, TaskRequest
from planner.act_comparison import CONDITIONS, ComparisonPlanner
from planner.indexed import IndexedPlanner


STATE=np.zeros(6,np.float32)


def box(key,x,y,z=.020,parent='shelf_1'):
    return SceneObject(key,Pose((x,y,z)),(.028,)*3,parent_id=parent)


def scene(nodes,held=None,edges=(),timestamp=0.,episode='comparison'):
    regions={key:dict(region_type='shelf',center=[x,y],size=[.17,.13],floor_top_m=.006)
             for key,x,y in [('shelf_1',.12,-.18),('shelf_2',.24,0.),('shelf_3',.24,.18)]}
    regions.update(buffer_tray=dict(region_type='tray',center=[.08,-.18],size=[.11,.11],floor_top_m=.006),
                   output_tray=dict(region_type='tray',center=[.08,.18],size=[.11,.11],floor_top_m=.006),
                   tabletop=dict(region_type='table',center=[.15,0.],size=[.6,.6],floor_top_m=0.))
    return SceneGraph(tuple(nodes),tuple(edges),{'workspace':None,'unassigned':'workspace',
        **{key:'workspace' for key in regions}},held,timestamp,episode,source='rgbd',regions=regions)


def task(target='unresolved:target',region=None):
    return TaskRequest(target,Pose((.08,.18,.020)),Pose((.08,-.18,.020)),
                       TargetSpec((.9,.2,.1),region=region,size_m=(.028,)*3))


class ACTComparisonTests(unittest.TestCase):
    def test_greedy_measures_from_fixed_robot_base_not_target_or_workspace_center(self):
        graph=scene([box('base_near',.17,-.15),box('workspace_near',.24,0.,parent='shelf_2')])
        planner=ComparisonPlanner('a_greedy')
        self.assertEqual(planner.plan_next_subgoal(graph,task(),STATE).object_id,'base_near')
        changed=STATE.copy();changed[:5]=[1.,-.5,.5,.2,1.]
        self.assertEqual(planner.plan_next_subgoal(graph,task(),changed).object_id,'base_near')

    def test_ablation_never_calls_hidden_estimator_but_still_searches(self):
        graph=scene([box('near',.17,-.15),box('hidden_space',.24,0.,.104,'shelf_2')])
        with patch.object(IndexedPlanner,'_hidden',side_effect=AssertionError('hidden cue leaked')):
            goal=ComparisonPlanner('b_no_hidden').plan_next_subgoal(graph,task(),STATE)
        self.assertEqual(goal.object_id,'near')
        full=ComparisonPlanner('c_full').plan_next_subgoal(graph,task(),STATE)
        self.assertEqual(full.object_id,'hidden_space')

    def test_no_hidden_space_is_not_a_forced_failure_for_ablation_or_full(self):
        graph=scene([box('near',.17,-.15),box('far',.24,0.,parent='shelf_2')])
        goals=[ComparisonPlanner(c).plan_next_subgoal(graph,task(),STATE) for c in CONDITIONS]
        self.assertTrue(all(g.object_id=='near' for g in goals))

    def test_accessible_target_has_same_priority_for_all_conditions(self):
        graph=scene([box('near',.17,-.15),box('target',.24,0.,parent='shelf_2')])
        for condition in CONDITIONS:
            goal=ComparisonPlanner(condition).plan_next_subgoal(graph,task('target'),STATE)
            self.assertEqual(goal.object_id,'target')
            self.assertEqual(encode_goal(goal).shape,(25,))

    def test_graph_conditions_keep_same_visible_target_support_reasoning(self):
        graph=scene([box('target',.24,0.,parent='shelf_2'),box('blocker',.24,0.,.048,'shelf_2'),
                     box('near',.17,-.15)])
        self.assertEqual(ComparisonPlanner('a_greedy').plan_next_subgoal(graph,task('target'),STATE).object_id,'near')
        for condition in ('b_no_hidden','c_full'):
            self.assertEqual(ComparisonPlanner(condition).plan_next_subgoal(graph,task('target'),STATE).object_id,'blocker')

    def test_shared_place_and_overflow_never_use_hidden_estimates(self):
        graph=scene([box('held',.15,0.,.15,'unassigned'),box('left',.105,-.18),box('right',.055,-.18),
                     box('target',.24,0.,parent='shelf_2')],held='held')
        with patch.object(IndexedPlanner,'_hidden',side_effect=AssertionError('placement leaked hidden cue')):
            goals=[ComparisonPlanner(c).plan_next_subgoal(graph,task('target'),STATE) for c in CONDITIONS]
        self.assertTrue(all(g==goals[0] and g.primitive=='PLACE' for g in goals))

    def test_greedy_ignores_graph_edges_and_all_methods_obey_same_region_hint(self):
        graph=scene([box('near',.17,-.15),box('far',.24,0.,parent='shelf_2')])
        altered=replace(graph,edges=(Edge('far','near','on_top_of'),))
        planner=ComparisonPlanner('a_greedy')
        self.assertEqual(planner.plan_next_subgoal(graph,task(),STATE),planner.plan_next_subgoal(altered,task(),STATE))
        for condition in CONDITIONS:
            self.assertEqual(ComparisonPlanner(condition).plan_next_subgoal(graph,task(region='shelf_2'),STATE).object_id,'far')

    def test_completed_objects_not_researched_and_time_reversal_resets_history(self):
        graph=scene([box('near',.17,-.15),box('far',.24,0.,parent='shelf_2')])
        for condition in CONDITIONS:
            planner=ComparisonPlanner(condition)
            planner.plan_next_subgoal(graph,task(),STATE)
            planner.completed(SimpleNamespace(primitive='PLACE',object_id='near'))
            self.assertEqual(planner.plan_next_subgoal(replace(graph,timestamp=1.),task(),STATE).object_id,'far')
            self.assertEqual(planner.plan_next_subgoal(graph,task(),STATE).object_id,'near')


if __name__=='__main__': unittest.main()

