"""Geometry/association tests use known synthetic geometry, never as demo inference."""
import tempfile
import unittest
import importlib.util
if any(importlib.util.find_spec(name) is None for name in ('open3d','scipy','networkx')):
    raise unittest.SkipTest('Install the perception extra in a Python 3.11 environment')
from pathlib import Path
import numpy as np
from perception.geometry import backproject,cloud,describe,snapshot_tsdf
from perception.association import match,ObjectTracker,fuse_views
from perception.reconstruction import export_graph
from perception.acquisition import load_frame
from perception.robot_frames import relative_transform,transform,add_robot_frames
import hashlib
import json
from PIL import Image

def candidate(x=0.,color=(.8,.1,.1),camera='a'):
    rng=np.random.default_rng(9)
    points=rng.uniform(-.012,.012,(250,3))+[x,0,.04]
    colors=np.tile(color,(len(points),1))
    p=cloud(points,colors)
    return dict(describe(p),_points=points,_colors=colors,observations=[dict(camera=camera)],label_hypotheses=[])

class PerceptionTests(unittest.TestCase):
    def test_backprojection_uses_meters_and_camera_to_base_direction(self):
        k=np.array([[100.,0,1],[0,100,1],[0,0,1]])
        t=np.eye(4);t[:3,3]=[.3,.1,.2]
        xyz=backproject(np.full((2,2),.5),k,t)
        np.testing.assert_allclose(xyz[0,0],[.2975,.0975,.7])
        recovered=(xyz-t[:3,3])@t[:3,:3]
        np.testing.assert_allclose(recovered[:,:,2],.5)

    def test_cross_view_is_one_to_one_and_keeps_unmatched(self):
        a=[candidate(),candidate(.08,(.1,.1,.8))]
        b=[candidate(.002,camera='b'),candidate(.081,(.1,.1,.8),camera='b'),candidate(.2,camera='b')]
        pairs,_,unmatched=match(a,b)
        self.assertEqual([(i,j) for i,j,c in pairs],[(0,0),(1,1)])
        self.assertEqual(unmatched,[2])
        fused=fuse_views(dict(a=a,b=b))
        self.assertEqual(len(fused),3)
        self.assertEqual(sum(len(o['observations'])==2 for o in fused),2)

    def test_ambiguous_identical_neighbors_are_not_forced_to_merge(self):
        pairs,_,_=match([candidate()], [candidate(.001),candidate(-.001)])
        self.assertEqual(pairs,[])

    def test_track_moves_retains_history_and_resets_on_episode(self):
        tracker=ObjectTracker()
        first=tracker.update([candidate()], 'run',0)[0]['instance_id']
        self.assertEqual(tracker.update([candidate(.03)],'run',1)[0]['instance_id'],first)
        self.assertEqual(tracker.update([],'run',2)[0]['status'],'history')
        self.assertEqual(tracker.update([candidate(.032)],'run',3)[0]['instance_id'],first)
        with self.assertRaises(AssertionError):tracker.update([],'run',3)
        after=tracker.update([candidate(.2)],'new',0)
        self.assertEqual(len(after),1)
        self.assertEqual(after[0]['first_seen'],0)

    def test_tracker_state_roundtrip_preserves_id(self):
        original=ObjectTracker()
        original.update([candidate()],'run',0)
        state=dict(episode='run',timestamp=0,next_id=original.next_id,
                   tracks=[{k:v for k,v in t.items() if not k.startswith('_')} for t in original.tracks.values()])
        resumed=ObjectTracker(json.loads(json.dumps(state)))
        self.assertEqual(resumed.update([candidate(.01)],'run',1)[0]['instance_id'],'instance_0001')

    def test_input_hashes_reject_modified_rgb_and_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rgb=root/'test_rgb.png';depth=root/'test_depth.npz'
            Image.fromarray(np.zeros((4,4,3),np.uint8)).save(rgb)
            np.savez_compressed(depth,depth=np.ones((4,4),np.float32))
            meta=dict(schema_version=2,frame_id='robot_base',depth_units='m',timestamp=0,
                cameras=dict(test=dict(rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
                    depth_sha256=hashlib.sha256(depth.read_bytes()).hexdigest(),timestamp=0,
                    K=np.eye(3).tolist(),T_camera_to_base=np.eye(4).tolist())))
            (root/'capture.json').write_text(json.dumps(meta))
            original=rgb.read_bytes()
            load_frame(root)
            rgb.write_bytes(original+b'changed')
            with self.assertRaisesRegex(AssertionError,'RGB hash'):load_frame(root)
            rgb.write_bytes(original)
            np.savez_compressed(depth,depth=np.zeros((4,4),np.float32))
            with self.assertRaisesRegex(AssertionError,'Depth hash'):load_frame(root)

    def test_obb_is_finite_and_explicitly_approximate(self):
        d=candidate()
        r=np.array(d['obb']['rotation'])
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(r),1.)
        self.assertIn('not_registered_6D',d['pose_kind'])

    def test_tsdf_uses_inverse_extrinsic_and_starts_new_volume(self):
        h=w=64
        k=np.array([[80.,0,w/2],[0,80,h/2],[0,0,1]])
        t=np.diag([1.,-1.,-1.,1.]);t[:3,3]=[.2,0,.5]
        frame=dict(rgb=np.full((h,w,3),100,np.uint8),depth=np.full((h,w),.45,np.float32),K=k,T=t)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            n=snapshot_tsdf(dict(a=frame),p/'first.ply')
            self.assertGreater(n,0)
            frame['depth'][:]=.40
            snapshot_tsdf(dict(a=frame),p/'second.ply')
            import open3d as o3d
            pts=np.asarray(o3d.io.read_point_cloud(str(p/'second.ply')).points)
            self.assertAlmostEqual(float(np.median(pts[:,2])),.10,places=2)
            self.assertLess(np.max(np.abs(pts[:,2]-.10)),.01)

    def test_graph_source_and_membership_without_gt(self):
        tracks=ObjectTracker().update([candidate(.2)],'run',0)
        meta=dict(timestamp=0,episode_id='run',regions=[dict(id='shelf',center=[.2,0],size=[.2,.2])])
        with tempfile.TemporaryDirectory() as folder:
            g=export_graph(tracks,meta,folder)
            self.assertEqual(g.graph['source'],'rgbd')
            self.assertEqual(g.edges[tracks[0]['instance_id'],'shelf']['relation'],'inside')
            self.assertFalse(any(e['relation']=='on_top_of' for a,b,e in g.edges(data=True)))

    def test_rotated_gripper_relative_pose_and_roundtrip(self):
        from scipy.spatial.transform import Rotation
        tcp=transform(Rotation.from_euler('z',90,degrees=True).as_matrix(),[.1,0,.2])
        obj=transform(np.eye(3),[.2,.1,.2])
        rel=relative_transform(tcp,obj)
        np.testing.assert_allclose(rel[:3,3],[.1,-.1,0],atol=1e-9)
        np.testing.assert_allclose(tcp@rel,obj,atol=1e-9)

    def test_robot_graph_and_stale_object_pose_are_explicit(self):
        import networkx as nx
        tracker=ObjectTracker();tracks=tracker.update([candidate(.2)],'run',0)
        tracks=tracker.update([],'run',1)
        tcp=transform(np.eye(3),[.1,0,.1])
        robot=dict(timestamp=1,robot_id='so101',source='joint_state',T_world_base=np.eye(4).tolist(),
            joint_names=[],joint_positions_rad=[],gripper_closed_fraction=0,links=[],
            tcp=dict(id='gripper_tcp',parent='robot_base',site_name='test',T_base_tcp=tcp.tolist()))
        meta=dict(timestamp=1,robot=robot,cameras={})
        g=nx.DiGraph();g.add_node('workspace');g.add_node(tracks[0]['instance_id'])
        add_robot_frames(g,meta,tracks)
        edge=g.edges[tracks[0]['instance_id'],'gripper_tcp']
        self.assertTrue(edge['object_pose_stale'])
        self.assertIn('robot_base',g)
        self.assertIn('gripper_tcp',g)
        np.testing.assert_allclose(tcp@np.array(edge['T_reference_object']),
            transform(tracks[0]['obb']['rotation'],tracks[0]['obb']['center_m']))
        meta['timestamp']=2
        with self.assertRaisesRegex(AssertionError,'Stale robot'):add_robot_frames(g,meta,tracks)

if __name__=='__main__':
    unittest.main()
