"""Check action-conditioned identity and edge replacement without object GT."""
import tempfile
import unittest
import importlib.util
if any(importlib.util.find_spec(name) is None for name in ('open3d','scipy','networkx')):
    raise unittest.SkipTest('Install the perception extra in a Python 3.11 environment')
import numpy as np
from perception.acquisition import configured_regions
from perception.association import ObjectTracker
from perception.graph_update import graph_delta
from perception.reconstruction import export_graph
from perception.visualization import instance_color
from test_perception import candidate


class GraphUpdateTests(unittest.TestCase):
    regions=[dict(id='shelf',center=[0.,0.],size=[.1,.1]),
             dict(id='tray',center=[.2,0.],size=[.1,.1])]

    def event(self,**changes):
        return dict(dict(event_id='move',episode_id='run',source_instance_id='instance_0001',
                    destination_region='tray',expected_position_m=[.2,0.,.04],started_at=0,completed_at=1),**changes)

    def test_unique_compatible_destination_reuses_id_and_replaces_membership(self):
        tracker=ObjectTracker()
        before=tracker.update([candidate()],'run',0)
        meta=dict(timestamp=0,episode_id='run',regions=self.regions)
        with tempfile.TemporaryDirectory() as directory:
            old=export_graph(before,meta,directory)
            after=tracker.update([candidate(.2)],'run',1,[self.event()],self.regions)
            self.assertEqual(len(after),1)
            self.assertEqual(after[0]['instance_id'],'instance_0001')
            self.assertEqual(after[0]['tracking_evidence']['method'],'action_prior_and_rgbd_hypothesis')
            meta['timestamp']=1
            new=export_graph(after,meta,directory)
            delta=graph_delta(old,new)
            self.assertIn(dict(source='instance_0001',target='shelf',relation='inside'),delta['removed_relations'])
            self.assertIn(dict(source='instance_0001',target='tray',relation='inside'),delta['added_relations'])
            self.assertFalse(new.has_edge('instance_0001','shelf'))
            self.assertAlmostEqual(delta['object_changes'][0]['displacement_m'],.2,places=5)

    def test_command_without_observation_does_not_teleport_node(self):
        tracker=ObjectTracker();before=tracker.update([candidate()],'run',0)[0]
        center=before['obb']['center_m'].copy()
        after=tracker.update([],'run',1,[self.event()],self.regions)
        self.assertEqual(after[0]['status'],'history')
        self.assertEqual(after[0]['last_seen'],0)
        np.testing.assert_allclose(after[0]['obb']['center_m'],center)

    def test_wrong_color_is_not_forced_to_match_commanded_identity(self):
        tracker=ObjectTracker();tracker.update([candidate()],'run',0)
        after=tracker.update([candidate(.2,(.1,.1,.8))],'run',1,[self.event()],self.regions)
        self.assertEqual(after[0]['status'],'history')
        self.assertNotEqual(next(t for t in after if t['status']=='seen')['instance_id'],'instance_0001')

    def test_two_similar_destination_objects_remain_ambiguous(self):
        tracker=ObjectTracker();tracker.update([candidate()],'run',0)
        after=tracker.update([candidate(.2),candidate(.201)],'run',1,[self.event()],self.regions)
        self.assertEqual(after[0]['status'],'history')
        self.assertEqual(sum(t['status']=='seen' for t in after),2)

    def test_large_motion_without_action_prior_does_not_reuse_id(self):
        tracker=ObjectTracker();tracker.update([candidate()],'run',0)
        after=tracker.update([candidate(.2)],'run',1)
        self.assertEqual(after[0]['status'],'history')
        self.assertEqual(after[1]['instance_id'],'instance_0002')

    def test_foreign_or_future_action_is_ignored(self):
        for event in [self.event(episode_id='other'),self.event(completed_at=2)]:
            tracker=ObjectTracker();tracker.update([candidate()],'run',0)
            after=tracker.update([candidate(.2)],'run',1,[event],self.regions)
            self.assertEqual(after[0]['status'],'history')
            self.assertEqual(after[1]['instance_id'],'instance_0002')

    def test_two_commands_cannot_claim_one_observation(self):
        tracker=ObjectTracker();tracker.update([candidate(),candidate(.05)],'run',0)
        after=tracker.update([candidate(.2)],'run',1,
            [self.event(),self.event(source_instance_id='instance_0002')],self.regions)
        self.assertTrue(all(t['status']=='history' for t in after[:2]))
        self.assertEqual(after[2]['status'],'seen')

    def test_instance_color_is_independent_of_order_and_frame(self):
        ids=['instance_0020','instance_0001','instance_0028']
        before={key:instance_color(key) for key in ids}
        after={key:instance_color(key) for key in reversed(ids)}
        for key in ids:np.testing.assert_array_equal(before[key],after[key])

    def test_static_tray_configuration_contains_no_object_gt(self):
        config=dict(shelf_scene=dict(shelves=[dict(id='shelf',center=[.2,0],size=[.1,.1])],floor_thickness=.006),
                    placement_trays=dict(size=[.11,.11],floor_thickness=.006),buffer_position=[.08,-.18,.02],destination=[.08,.18,.02])
        regions=configured_regions(config)
        self.assertEqual([r['region_type'] for r in regions],['shelf','tray','tray'])
        self.assertEqual(regions[1]['center'],[.08,-.18])
        self.assertTrue(all('object_id' not in r for r in regions))


if __name__=='__main__':unittest.main()
