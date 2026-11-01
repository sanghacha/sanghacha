"""Build the four implemented relations from already tracked RGB-D objects."""
import networkx as nx

from .frames import add_frames
from .validation import snapshot


def build_scene_graph(objects, metadata):
    """Return a new DiGraph. Inputs are not mutated and IDs are not associated.

    Object positions are supplied estimates in robot_base/metres. A seen object
    belongs to the first configured XY region containing its surface centroid.
    Historical objects retain old geometry but receive no current inside edge.
    """
    tracks, meta = snapshot(objects, metadata)
    graph = nx.DiGraph(frame_id='robot_base', source='rgbd', timestamp=meta['timestamp'],
                       episode_id=meta['episode_id'],
                       pose_note='Visible-surface OBB estimates, not model-registered 6D poses')
    graph.add_node('workspace', kind='workspace')
    for region in meta['regions']:
        attrs = {key: value for key, value in region.items() if key != 'id'}
        graph.add_node(region['id'], kind='region', **attrs)
        graph.add_edge(region['id'], 'workspace', relation='inside')
    for track in tracks:
        ident = track['instance_id']
        graph.add_node(ident, kind='object_candidate', **track)
        if track['status'] == 'seen':
            center = track['visible_surface_centroid_m']
            for region in meta['regions']:
                if all(abs(center[i] - region['center'][i]) <= region['size'][i] / 2 for i in (0, 1)):
                    graph.add_edge(ident, region['id'], relation='inside',
                                   evidence='RGB-D location + configured region boundary')
                    break
    add_frames(graph, meta, tracks)
    return graph
