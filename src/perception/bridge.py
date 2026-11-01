"""Convert RGB-D graphs into the shared planner contract; no simulator lookup."""
from dataclasses import replace

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from pipeline.types import Pose, SceneGraph, SceneObject, check


def fit_known_box(track, sizes):
    """Upright box prior fitted to observed top pixels, not a GT 6D pose.

    The original partial-surface OBB is preserved separately for auditability.
    This prior is intended only for the configured square box experiment.
    """
    points = np.asarray(track['_points'])
    top_z = float(np.percentile(points[:, 2], 98))
    top = points[points[:, 2] >= top_z - .0015, :2]
    check(len(top) >= 4, 'insufficient top surface for box fitting')
    try:
        hull = top[ConvexHull(top).vertices]
    except QhullError as exc:
        raise ValueError('degenerate top surface for box fitting') from exc
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    choices = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        local = top @ rotation
        lo, hi = local.min(axis=0), local.max(axis=0)
        choices.append((float(np.prod(hi-lo)), angle, (lo+hi)/2 @ rotation.T, hi-lo))
    _, yaw, center, extent = min(choices, key=lambda v: v[0])
    size = min(sizes, key=lambda v: np.linalg.norm(np.sort(v[:2])-np.sort(extent)))
    check(np.linalg.norm(np.sort(size[:2])-np.sort(extent)) < .018,
          'observed footprint does not match configured box sizes')
    yaw = (yaw + np.pi/4) % (np.pi/2) - np.pi/4
    return dict(position_m=[float(center[0]), float(center[1]), top_z-size[2]/2],
                quaternion_xyzw=Rotation.from_euler('z', yaw).as_quat().tolist(), size_m=list(size),
                observed_top_extent_m=extent.tolist(),
                evidence='observed top surface + configured upright square-box size prior')


def to_scene_graph(graph, *, held_object=None):
    check(graph.graph.get('source') == 'rgbd' and graph.graph.get('frame_id') == 'robot_base',
          'expected an RGB-D graph in robot_base')
    groups = {'workspace': None, 'unassigned': 'workspace'}
    for key, node in graph.nodes(data=True):
        if node.get('kind') == 'region':
            groups[key] = 'workspace'
    objects = []
    for key, node in graph.nodes(data=True):
        if node.get('kind') != 'object_candidate' or node['status'] != 'seen':
            continue  # Historical poses never become current collision/grasp targets.
        parent = next((b for _, b, attrs in graph.out_edges(key, data=True)
                       if attrs['relation'] == 'inside' and b in groups), 'unassigned')
        fit = node.get('box_model_fit')
        if fit:
            pose = Pose(tuple(fit['position_m']), tuple(fit['quaternion_xyzw']))
            size = tuple(fit['size_m'])
        else:
            box = node['obb']
            pose = Pose(tuple(box['center_m']), tuple(Rotation.from_matrix(box['rotation']).as_quat()))
            size = tuple(box['extent_m'])
        labels = node.get('label_hypotheses', [])
        label = max(labels, key=lambda h: h['score'])['label'].strip(' .') if labels else 'box'
        if label in ('cube', 'block'):
            label = 'box'
        objects.append(SceneObject(key, pose, size, label, parent, tuple(node['mean_rgb'])))
    if held_object is not None:
        objects = [obj for obj in objects if obj.object_id != held_object.object_id]
        objects.append(replace(held_object, parent_id='unassigned'))
    return SceneGraph(tuple(objects), (), groups,
                      held_object.object_id if held_object else None,
                      graph.graph['timestamp'], graph.graph['episode_id'], source='rgbd',
                      regions={key: dict(node) for key, node in graph.nodes(data=True) if node.get('kind') == 'region'})
