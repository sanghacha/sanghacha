from dataclasses import replace
import unittest

import numpy as np

from pipeline.mock_data import make_mock_data
from pipeline.types import Pose, SceneGraph, SceneObject, TaskRequest, TargetSpec
from planner import plan_next_subgoal
from planner.baselines import METHODS, plan_baseline


class PlannerComparisonTests(unittest.TestCase):
    def test_hsg_is_the_project_planner_and_all_methods_share_place(self):
        data = make_mock_data('blocked')
        state = data.observation.robot_state
        self.assertEqual(plan_baseline('hsg', data.graph, data.task, state),
                         plan_next_subgoal(data.graph, data.task, state))
        graph = replace(data.graph, held_object_id='A')
        goals = [plan_baseline(method, graph, data.task, state) for method in METHODS]
        self.assertTrue(all(goal == goals[0] and goal.primitive == 'PLACE' for goal in goals))

    def hidden_scene(self):
        # Shelf 1 has more total hidden space; shelf 2 has the deepest single stack.
        nodes = tuple(SceneObject(key, Pose((x, y, .006+.014+n*.028)), (.028,)*3, parent_id=region)
            for key, x, y, n, region in [('a',.24,-.18,2,'shelf_1'), ('b',.29,-.18,2,'shelf_1'),
                                       ('c',.24,.18,3,'shelf_2')])
        regions = {key: dict(region_type='shelf', floor_top_m=.006) for key in ('shelf_1','shelf_2')}
        graph = SceneGraph(nodes, (), {'workspace':None,'shelf_1':'workspace','shelf_2':'workspace'},
                           source='rgbd', regions=regions)
        task = TaskRequest('unresolved:target', Pose((.08,.18,.02)), Pose((.08,-.18,.02)), TargetSpec((.9,.2,.1)))
        return graph, task, np.zeros(6, np.float32)

    def test_flat_ablation_really_removes_region_aggregation(self):
        graph, task, state = self.hidden_scene()
        self.assertEqual(plan_baseline('hsg',graph,task,state).object_id, 'a')
        self.assertEqual(plan_baseline('flat',graph,task,state).object_id, 'c')

    def test_known_region_hint_is_identical_for_all_methods(self):
        graph, task, state = self.hidden_scene()
        task = replace(task, target_spec=replace(task.target_spec, region='shelf_2'))
        for method in METHODS:
            self.assertEqual(plan_baseline(method,graph,task,state).object_id, 'c')

    def test_greedy_has_local_clearance_but_no_target_dependency_traversal(self):
        nodes = tuple(SceneObject(key,Pose((x,y,z)),(.032,)*3,parent_id='shelf_2')
            for key,x,y,z in [('target',.24,0.,.022),('middle',.26,0.,.054),
                              ('top',.28,0.,.086),('other',.24,.035,.022)])
        graph = SceneGraph(nodes,(),{'workspace':None,'shelf_2':'workspace'},source='rgbd',
                           regions={'shelf_2':dict(region_type='shelf',floor_top_m=.006)})
        task = TaskRequest('target',Pose((.08,.18,.022)),Pose((.08,-.18,.022)))
        state=np.zeros(6,np.float32)
        self.assertEqual(plan_baseline('greedy',graph,task,state).object_id,'other')
        self.assertEqual(plan_baseline('flat',graph,task,state).object_id,'top')
        self.assertEqual(plan_baseline('hsg',graph,task,state).object_id,'top')


if __name__ == '__main__':
    unittest.main()
