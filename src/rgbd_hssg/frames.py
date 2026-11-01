"""Robot/camera graph frames from supplied transforms; no simulator access."""
import numpy as np


def object_transform(track):
    result = np.eye(4)
    result[:3, :3] = track['obb']['rotation']
    result[:3, 3] = track['obb']['center_m']
    return result


def add_frames(graph, meta, tracks):
    robot = meta.get('robot')
    graph.graph['robot_state_available'] = robot is not None
    for name, camera in meta.get('cameras', {}).items():
        ident = 'camera:' + name
        graph.add_node(ident, kind='fixed_camera', T_base_frame=camera['T_camera_to_base'], K=camera['K'])
        graph.add_edge(ident, 'workspace', relation='fixed_in')
    if robot is None:
        return
    graph.graph['relative_pose_convention'] = 'T_reference_object maps object-local coordinates into reference frame; translation in metres'
    graph.add_node('robot_base', kind='robot', robot_id=robot['robot_id'],
                   T_base_frame=np.eye(4).tolist(), T_world_base=robot['T_world_base'],
                   timestamp=robot['timestamp'], source=robot['source'],
                   joint_names=robot['joint_names'], joint_positions_rad=robot['joint_positions_rad'],
                   gripper_closed_fraction=robot['gripper_closed_fraction'])
    graph.add_edge('robot_base', 'workspace', relation='inside')
    transforms = {'robot_base': np.eye(4)}
    for link in robot['links']:
        transforms[link['id']] = np.array(link['T_base_link'])
        graph.add_node(link['id'], kind='robot_link', T_base_frame=link['T_base_link'],
                       timestamp=robot['timestamp'], source=robot['source'])
    for link in robot['links']:
        relative = np.linalg.solve(transforms[link['parent']], transforms[link['id']])
        graph.add_edge(link['id'], link['parent'], relation='kinematic_child_of', T_parent_child=relative.tolist())
    tcp = robot['tcp']
    t_tcp = np.array(tcp['T_base_tcp'])
    graph.add_node('gripper_tcp', kind='end_effector', T_base_frame=tcp['T_base_tcp'],
                   timestamp=robot['timestamp'], source=robot['source'], site_name=tcp.get('site_name'))
    graph.add_edge('gripper_tcp', tcp['parent'], relation='kinematic_child_of',
                   T_parent_child=np.linalg.solve(transforms[tcp['parent']], t_tcp).tolist())
    for track in tracks:
        for reference, t_reference in [('robot_base', np.eye(4)), ('gripper_tcp', t_tcp)]:
            relative = np.linalg.solve(t_reference, object_transform(track))
            graph.add_edge(track['instance_id'], reference, relation='relative_pose_to',
                           T_reference_object=relative.tolist(), translation_m=relative[:3, 3].tolist(),
                           distance_between_frame_origins_m=float(np.linalg.norm(relative[:3, 3])),
                           reference_timestamp=robot['timestamp'], object_last_seen=track['last_seen'],
                           object_pose_stale=track['status'] != 'seen', pose_kind=track['pose_kind'],
                           note='Distance to estimated OBB center, not surface clearance, reachability or grasp pose')
