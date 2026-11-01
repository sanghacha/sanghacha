"""The newly wired exporter must agree with the shared graph-only package."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if any(importlib.util.find_spec(name) is None for name in ('open3d','scipy','networkx')):
    raise unittest.SkipTest('Install the perception extra in a Python 3.11 environment')
import networkx as nx
from perception.association import ObjectTracker
from perception.reconstruction import export_graph
from rgbd_hssg import build_scene_graph,load_graph
from test_perception import candidate

ROOT=Path(__file__).resolve().parents[1]

class PerceptionCLITests(unittest.TestCase):
    def test_exporter_uses_the_shared_graph_contract_and_writes_ply(self):
        tracks=ObjectTracker().update([candidate(.2)],'run',0)
        meta=dict(timestamp=0,episode_id='run',regions=[dict(id='shelf',center=[.2,0.],size=[.2,.2])])
        with tempfile.TemporaryDirectory() as folder:
            graph=export_graph(tracks,meta,folder)
            self.assertTrue(nx.utils.graphs_equal(graph,build_scene_graph(tracks,meta)))
            self.assertTrue(nx.utils.graphs_equal(graph,load_graph(Path(folder)/'scene_graph.json')))
            self.assertTrue((Path(folder)/'instance_0001.ply').exists())
            self.assertNotIn('_points',graph.nodes['instance_0001'])

    def test_missing_capture_is_an_error_instead_of_an_empty_success(self):
        with tempfile.TemporaryDirectory() as folder:
            process=subprocess.run([sys.executable,str(ROOT/'scripts/reconstruct_multiview.py'),
                '--capture',str(Path(folder)/'missing'),'--output',str(Path(folder)/'output')],
                env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)
            self.assertNotEqual(process.returncode,0)
            self.assertIn('No frame_* capture directories',process.stderr)

    def test_capture_cli_rejects_zero_frames(self):
        process=subprocess.run([sys.executable,str(ROOT/'scripts/capture_multiview.py'),'--frames','0'],
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)
        self.assertNotEqual(process.returncode,0)
        self.assertIn('frames must be positive',process.stderr)

if __name__=='__main__':unittest.main()
