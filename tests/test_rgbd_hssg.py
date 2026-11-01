"""Graph-only contracts, including invalid/stale inputs and no inference imports."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import importlib.util

if importlib.util.find_spec('networkx') is None:
    raise unittest.SkipTest('Install the sg or perception extra for graph tests')
import networkx as nx
import numpy as np

from rgbd_hssg import build_scene_graph, graph_delta, load_graph, load_snapshot, save_graph

ROOT = Path(__file__).resolve().parents[1]


def matrix(position):
    value = np.eye(4)
    value[:3, 3] = position
    return value.tolist()


def fixture(timestamp=0., position=(.2, 0., .04), status='seen'):
    track = dict(instance_id='instance_0001', visible_surface_centroid_m=list(position),
                 obb=dict(center_m=list(position), extent_m=[.03, .03, .03], rotation=np.eye(3).tolist()),
                 status=status, first_seen=0., last_seen=timestamp if status == 'seen' else 0.,
                 pose_kind='partial_surface_OBB_estimate_not_registered_6D_pose',
                 observations=[dict(camera='top', raw_mask=3, bbox=[10, 20, 30, 40], sam_score=.9)])
    meta = dict(timestamp=timestamp, episode_id='test', frame_id='robot_base', depth_units='m',
                regions=[dict(id='shelf', center=[.2, 0.], size=[.1, .1]),
                         dict(id='tray', center=[.05, -.18], size=[.1, .1])],
                cameras=dict(top=dict(T_camera_to_base=matrix([.2, 0., .7]), K=[[500, 0, 320], [0, 500, 240], [0, 0, 1]])),
                robot=dict(timestamp=timestamp, robot_id='so101', source='robot_joint_state_and_forward_kinematics',
                           T_world_base=np.eye(4).tolist(), joint_names=['joint'], joint_positions_rad=[.1],
                           gripper_closed_fraction=.2,
                           links=[dict(id='link', parent='robot_base', T_base_link=matrix([.1, 0., .1]))],
                           tcp=dict(id='gripper_tcp', parent='link', T_base_tcp=matrix([.1, 0., .2]))))
    return [track], meta


class RGBDHSSGTests(unittest.TestCase):
    def test_nodes_four_relations_and_geometry_provenance(self):
        objects, meta = fixture()
        graph = build_scene_graph(objects, meta)
        self.assertEqual(graph.graph['source'], 'rgbd')
        self.assertEqual({d['relation'] for _, _, d in graph.edges(data=True)},
                         {'inside', 'relative_pose_to', 'kinematic_child_of', 'fixed_in'})
        self.assertTrue(graph.has_edge('instance_0001', 'shelf'))
        self.assertEqual(graph.nodes['instance_0001']['observations'], objects[0]['observations'])
        self.assertEqual(graph.nodes['link']['kind'], 'robot_link')

    def test_region_membership_replaced_and_relative_pose_recomputed(self):
        objects, meta = fixture()
        before = build_scene_graph(objects, meta)
        objects, meta = fixture(1., (.05, -.18, .02))
        after = build_scene_graph(objects, meta)
        delta = graph_delta(before, after)
        self.assertFalse(after.has_edge('instance_0001', 'shelf'))
        self.assertTrue(after.has_edge('instance_0001', 'tray'))
        self.assertIn(dict(source='instance_0001', target='shelf', relation='inside'), delta['removed_relations'])
        self.assertIn(dict(source='instance_0001', target='tray', relation='inside'), delta['added_relations'])
        self.assertEqual(delta['object_changes'][0]['new_region'], 'tray')
        np.testing.assert_allclose(after.edges['instance_0001', 'gripper_tcp']['translation_m'], [-.05, -.18, -.18])
        delta['object_changes'][0]['new_center_m'][0] = 9.
        self.assertEqual(after.nodes['instance_0001']['obb']['center_m'][0], .05)

    def test_rotated_tcp_uses_local_axes_and_roundtrip(self):
        objects, meta = fixture(position=(.2, .1, .2))
        tcp = np.array(matrix([.1, 0., .2]))
        tcp[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        meta['robot']['tcp']['T_base_tcp'] = tcp
        graph = build_scene_graph(objects, meta)
        relative = np.array(graph.edges['instance_0001', 'gripper_tcp']['T_reference_object'])
        np.testing.assert_allclose(relative[:3, 3], [.1, -.1, 0], atol=1e-10)
        np.testing.assert_allclose((tcp @ relative)[:3, 3], objects[0]['obb']['center_m'])

    def test_robot_motion_is_recorded_without_object_translation(self):
        objects, meta = fixture()
        before = build_scene_graph(objects, meta)
        objects, meta = fixture(1.)
        meta['robot']['tcp']['T_base_tcp'] = matrix([.15, 0., .2])
        after = build_scene_graph(objects, meta)
        delta = graph_delta(before, after)
        self.assertEqual(delta['object_changes'], [])
        self.assertTrue(any(e['source'] == 'instance_0001' and e['target'] == 'gripper_tcp' for e in delta['updated_relations']))

    def test_history_has_no_current_membership_and_explicit_stale_pose(self):
        objects, meta = fixture(1., status='history')
        graph = build_scene_graph(objects, meta)
        self.assertFalse(graph.has_edge('instance_0001', 'shelf'))
        self.assertTrue(graph.edges['instance_0001', 'gripper_tcp']['object_pose_stale'])
        self.assertEqual(graph.edges['instance_0001', 'gripper_tcp']['object_last_seen'], 0.)

    def test_missing_robot_does_not_invent_pose_but_keeps_calibrated_camera(self):
        objects, meta = fixture()
        del meta['robot']
        graph = build_scene_graph(objects, meta)
        self.assertNotIn('robot_base', graph)
        self.assertFalse(graph.graph['robot_state_available'])
        self.assertTrue(graph.has_edge('camera:top', 'workspace'))
        self.assertFalse(any(e['relation'] == 'relative_pose_to' for _, _, e in graph.edges(data=True)))

    def test_private_arrays_are_excluded_and_inputs_are_not_mutated_or_aliased(self):
        objects, meta = fixture()
        objects[0]['_points'] = np.ones((100, 3))
        objects[0]['_mask'] = object()
        snapshot = deepcopy(meta)
        graph = build_scene_graph(objects, meta)
        self.assertNotIn('_points', graph.nodes['instance_0001'])
        graph.nodes['instance_0001']['obb']['center_m'][0] = 9.
        graph.nodes['robot_base']['joint_names'].append('changed')
        self.assertEqual(objects[0]['obb']['center_m'][0], .2)
        self.assertEqual(meta, snapshot)

    def test_outside_regions_and_empty_scene_are_supported(self):
        objects, meta = fixture(position=(.45, .3, .05))
        graph = build_scene_graph(objects, meta)
        self.assertFalse(any(e['relation'] == 'inside' for _, _, e in graph.out_edges('instance_0001', data=True)))
        empty = build_scene_graph([], meta)
        self.assertNotIn('instance_0001', empty)

    def test_stale_robot_and_invalid_se3_are_rejected(self):
        objects, meta = fixture(1.)
        meta['robot']['timestamp'] = 0.
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            build_scene_graph(objects, meta)
        objects, meta = fixture()
        meta['robot']['tcp']['T_base_tcp'][0][0] = 2.
        with self.assertRaisesRegex(ValueError, 'rotation'):
            build_scene_graph(objects, meta)

    def test_cycles_unknown_parents_and_duplicate_ids_are_rejected(self):
        for parent in ['link', 'unknown']:
            objects, meta = fixture()
            meta['robot']['links'][0]['parent'] = parent
            with self.assertRaisesRegex(ValueError, 'acyclic'):
                build_scene_graph(objects, meta)
        objects, meta = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_scene_graph(objects + deepcopy(objects), meta)
        objects[0]['instance_id'] = 'robot_base'
        with self.assertRaisesRegex(ValueError, 'reserved'):
            build_scene_graph(objects, meta)

    def test_nonfinite_units_extent_and_observation_times_are_rejected(self):
        def bad(field, value):
            objects, meta = fixture()
            if field == 'extent': objects[0]['obb']['extent_m'][0] = value
            elif field == 'time': objects[0]['last_seen'] = value
            else: meta[field] = value
            with self.assertRaises(ValueError): build_scene_graph(objects, meta)
        for field, value in [('timestamp', float('nan')), ('units', 'mm'), ('extent', 0.), ('time', 5.)]:
            bad(field, value)

    def test_different_episodes_frames_and_out_of_order_delta_are_rejected(self):
        objects, meta = fixture()
        before = build_scene_graph(objects, meta)
        with self.assertRaisesRegex(ValueError, 'later timestamp'):
            graph_delta(before, before)
        objects, meta = fixture(1.)
        meta['episode_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'episodes'):
            graph_delta(before, build_scene_graph(objects, meta))
        meta['episode_id'] = 'test'
        after = build_scene_graph(objects, meta)
        after.graph['frame_id'] = 'world'
        with self.assertRaisesRegex(ValueError, 'coordinate frames'):
            graph_delta(before, after)

    def test_json_roundtrip_and_invalid_graph_endpoints(self):
        objects, meta = fixture()
        graph = build_scene_graph(objects, meta)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scene_graph.json'
            save_graph(graph, path)
            self.assertTrue(nx.utils.graphs_equal(graph, load_graph(path)))
            data = json.loads(path.read_text())
            data['edges'][0]['target'] = 'unknown'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'endpoint'):
                load_graph(path)

    def test_command_hint_does_not_create_or_move_objects(self):
        objects, meta = fixture()
        meta['action_events'] = [dict(source_instance_id='instance_0001', expected_position_m=[.05, -.18, .02])]
        graph = build_scene_graph(objects, meta)
        self.assertTrue(graph.has_edge('instance_0001', 'shelf'))
        self.assertNotIn('instance_0001', build_scene_graph([], meta))

    def test_real_example_id_and_expected_delta_without_gt_input(self):
        objects, meta = load_snapshot(ROOT / 'examples/rgbd_hssg/before.json')
        before = build_scene_graph(objects, meta)
        objects, meta = load_snapshot(ROOT / 'examples/rgbd_hssg/after.json')
        after = build_scene_graph(objects, meta)
        self.assertTrue(before.has_edge('instance_0020', 'shelf_2'))
        self.assertTrue(after.has_edge('instance_0020', 'buffer_tray'))
        delta = graph_delta(before, after)
        self.assertAlmostEqual(delta['object_changes'][0]['displacement_m'], .194760043, places=7)
        self.assertEqual(after.nodes['instance_0020']['tracking_evidence']['method'], 'action_prior_and_rgbd_hypothesis')

    def test_cli_runs_with_inference_and_simulation_imports_blocked(self):
        script = '''
import importlib.abc, runpy, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco','torch','open3d','scipy','PIL','perception','pipeline'}:
            raise RuntimeError('Unexpected dependency: '+fullname)
sys.meta_path.insert(0, BlockHeavy())
runpy.run_module('rgbd_hssg',run_name='__main__')
'''
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'graph.json'
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
            result = subprocess.run([sys.executable, '-c', script,
                str(ROOT / 'examples/rgbd_hssg/before.json'), '--output', str(output)],
                env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())


if __name__ == '__main__':
    unittest.main()
