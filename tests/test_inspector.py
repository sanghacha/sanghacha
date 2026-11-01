from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from pipeline.environment import MockEnvironment
from pipeline.inspector import PipelineInspector, short
from pipeline.mock_data import make_mock_data
from pipeline.runner import run_pipeline
from pipeline.types import Pose, SceneGraph, SceneObject, TaskRequest
from planner import plan_next_subgoal, explain_subgoal


class InspectorTests(unittest.TestCase):
    def test_event_stream_matches_the_actions_that_actually_complete(self):
        env = MockEnvironment('blocked')
        events = []
        inspector = PipelineInspector()
        def event(kind, **data):
            events.append(kind)
            if kind == 'planned':
                data['explanation'] = explain_subgoal(data['graph'], data['task'], data['subgoal'])
            inspector.on_event(kind, **data)
        try:
            goals = run_pipeline(env, env.task, on_event=event)
            self.assertEqual(tuple((g.primitive, g.object_id) for g in goals), make_mock_data().expected_order)
            self.assertEqual(events.count('planned'), len(goals))
            self.assertEqual(events.count('completed'), len(goals))
            self.assertEqual(events[-1], 'finished')
            self.assertEqual(len(inspector.history), len(goals))
            self.assertTrue(inspector.finished)
        finally:
            env.close()

    def test_planner_explanation_identifies_real_blocker_chain(self):
        data = make_mock_data('blocked')
        goal = plan_next_subgoal(data.graph, data.task, data.observation.robot_state)
        reason = explain_subgoal(data.graph, data.task, goal)
        self.assertEqual(reason['mode'], 'unblock')
        self.assertEqual(reason['chain'], ['B', 'A'])
        self.assertEqual(reason['support_source'], 'graph')
        self.assertEqual(reason['chain'][-1], goal.object_id)

    def test_hidden_search_explanation_ranking_matches_selected_object(self):
        nodes = (SceneObject('top1', Pose((.24, 0., .077)), (.028,)*3, parent_id='shelf_2'),
                 SceneObject('top2', Pose((.24, -.18, .048)), (.028,)*3, parent_id='shelf_1'))
        graph = SceneGraph(nodes, (), {'workspace': None, 'shelf_1': 'workspace', 'shelf_2': 'workspace'}, source='rgbd')
        task = TaskRequest('unseen', Pose((.08, .18, .021)), Pose((.08, -.18, .021)))
        goal = plan_next_subgoal(graph, task, np.zeros(6, np.float32))
        reason = explain_subgoal(graph, task, goal)
        self.assertEqual(reason['mode'], 'search')
        self.assertEqual(reason['candidates'][0]['object_id'], goal.object_id)
        self.assertEqual(reason['support_source'], 'geometry')

    def test_next_subgoal_pauses_at_next_plan_and_space_resumes(self):
        data = make_mock_data('blocked')
        goal = plan_next_subgoal(data.graph, data.task, data.observation.robot_state)
        inspector = PipelineInspector(paused=True)
        inspector.handle_key('next')
        self.assertFalse(inspector.paused)
        inspector.on_event('planned', observation=data.observation, graph=data.graph, task=data.task,
                           subgoal=goal, index=1, budget=100)
        self.assertTrue(inspector.paused)
        self.assertFalse(inspector.step_until_plan)
        inspector.handle_key('space')
        self.assertFalse(inspector.paused)

    def test_history_is_visible_but_never_inserted_into_planner_graph(self):
        data = make_mock_data('blocked')
        graph = replace(data.graph, source='rgbd')
        obs = replace(data.observation, source='rgbd', ground_truth=None)
        raw = SimpleNamespace(nodes=lambda **kwargs: [('old_track', dict(kind='object_candidate', status='history',
            obb=dict(center_m=[.2, 0., .03]), mean_rgb=[.2, .4, .8]))])
        inspector = PipelineInspector()
        inspector.observe_scene(observation=obs, graph=graph, raw_graph=raw)
        self.assertEqual(inspector.nodes['old_track']['status'], 'history')
        self.assertNotIn('old_track', [obj.object_id for obj in inspector.graph.nodes])
        inspector.on_event('scene', observation=obs, graph=graph, task=data.task, raw_graph=raw)
        self.assertEqual(inspector.snapshots, 1)
        pixels = inspector.render(600, 960)
        self.assertEqual(pixels.shape, (960, 600, 3))
        self.assertEqual(pixels.dtype, np.uint8)
        x1, y1, x2, y2 = inspector.hits['old_track']
        inspector.select_at((x1+x2)/2, (y1+y2)/2)
        self.assertEqual(inspector.selected, 'old_track')
        inspector.on_event('failed', error='test\nmultiline error')
        inspector.render(600, 960)
        self.assertEqual(short('instance_custom'), '…stom')

    def test_failure_event_preserves_stop_on_budget_exhaustion(self):
        env = MockEnvironment('blocked')
        env.config['max_steps_per_subgoal'] = 1
        events = []
        try:
            with self.assertRaisesRegex(ValueError, 'step budget'):
                run_pipeline(env, env.task, on_event=lambda kind, **data: events.append(kind))
            self.assertEqual(events[-1], 'failed')
            self.assertTrue(env.stopped)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
