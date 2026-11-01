"""Robot kinematic frames and relative candidate poses, separate from object GT."""
import numpy as np


def transform(rotation, translation):
    t=np.eye(4)
    t[:3,:3]=np.asarray(rotation).reshape(3,3)
    t[:3,3]=translation
    return t


def validate_transform(t):
    t=np.asarray(t,dtype=float)
    assert t.shape==(4,4) and np.isfinite(t).all()
    assert np.allclose(t[3],[0,0,0,1])
    assert np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6)
    assert np.isclose(np.linalg.det(t[:3,:3]),1.)
    return t


def relative_transform(t_base_reference,t_base_object):
    """T_reference_object = inverse(T_base_reference) @ T_base_object."""
    return np.linalg.solve(validate_transform(t_base_reference),validate_transform(t_base_object))


def capture_robot(env):
    m,d=env.model,env.data
    bid=m.body('base').id
    t_world_base=transform(d.xmat[bid],d.xpos[bid])
    t_base_world=np.linalg.inv(t_world_base)
    # The current shelf ROI/floor configuration is expressed in this base-aligned scene.
    assert np.allclose(t_world_base,np.eye(4),atol=1e-7), 'Recalibrate shelf ROI after relocating the robot base'
    bodies=[int(m.jnt_bodyid[m.joint(name).id]) for name in env.joints]
    names={body:'robot_link:'+m.body(body).name for body in bodies}
    names[bid]='robot_base'
    links=[]
    for body in bodies:
        parent=int(m.body_parentid[body])
        assert parent in names, 'Unrepresented robot parent frame'
        links.append(dict(id=names[body],body_name=m.body(body).name,parent=names[parent],
                          T_base_link=(t_base_world@transform(d.xmat[body],d.xpos[body])).tolist()))
    return dict(robot_id='so101',timestamp=float(d.time-env._start_time),
                source='robot_joint_state_and_forward_kinematics',T_world_base=t_world_base.tolist(),
                joint_names=list(env.joints),joint_positions_rad=d.qpos[env.qadr].tolist(),
                gripper_closed_fraction=float(env.joint_state()[5]),links=links,
                tcp=dict(id='gripper_tcp',site_name='gripperframe',
                         parent=names[int(m.site_bodyid[env.ee_site])],
                         T_base_tcp=(t_base_world@transform(d.site_xmat[env.ee_site],d.site_xpos[env.ee_site])).tolist()))


def add_robot_frames(graph,meta,tracks):
    robot=meta.get('robot')
    if robot is None:
        graph.graph['robot_state_available']=False
        return
    assert np.isclose(robot['timestamp'],meta['timestamp'],rtol=0,atol=1e-9),'Stale robot state'
    graph.graph['robot_state_available']=True
    graph.graph['relative_pose_convention']='T_reference_object maps object-local coordinates into reference frame; translation in metres'
    graph.add_node('robot_base',kind='robot',robot_id=robot['robot_id'],T_base_frame=np.eye(4).tolist(),
                   T_world_base=robot['T_world_base'],timestamp=robot['timestamp'],source=robot['source'],
                   joint_names=robot['joint_names'],joint_positions_rad=robot['joint_positions_rad'],
                   gripper_closed_fraction=robot['gripper_closed_fraction'])
    graph.add_edge('robot_base','workspace',relation='inside')
    frame_transforms={'robot_base':np.eye(4)}
    for link in robot['links']:
        frame_transforms[link['id']]=validate_transform(link['T_base_link'])
        graph.add_node(link['id'],kind='robot_link',T_base_frame=link['T_base_link'],
                       timestamp=robot['timestamp'],source=robot['source'])
    for link in robot['links']:
        rel=relative_transform(frame_transforms[link['parent']],frame_transforms[link['id']])
        graph.add_edge(link['id'],link['parent'],relation='kinematic_child_of',T_parent_child=rel.tolist())
    tcp=robot['tcp'];t_base_tcp=validate_transform(tcp['T_base_tcp'])
    graph.add_node('gripper_tcp',kind='end_effector',T_base_frame=t_base_tcp.tolist(),
                   timestamp=robot['timestamp'],source=robot['source'],site_name=tcp['site_name'])
    graph.add_edge('gripper_tcp',tcp['parent'],relation='kinematic_child_of',
                   T_parent_child=relative_transform(frame_transforms[tcp['parent']],t_base_tcp).tolist())
    for name,camera in meta.get('cameras',{}).items():
        graph.add_node('camera:'+name,kind='fixed_camera',T_base_frame=camera['T_camera_to_base'],K=camera['K'])
        graph.add_edge('camera:'+name,'workspace',relation='fixed_in')
    for track in tracks:
        # OBB orientation/center are estimated from partial observed surfaces.
        t_base_obj=transform(track['obb']['rotation'],track['obb']['center_m'])
        for reference,t_base_ref in [('robot_base',np.eye(4)),('gripper_tcp',t_base_tcp)]:
            relative=relative_transform(t_base_ref,t_base_obj)
            graph.add_edge(track['instance_id'],reference,relation='relative_pose_to',
                T_reference_object=relative.tolist(),translation_m=relative[:3,3].tolist(),
                distance_between_frame_origins_m=float(np.linalg.norm(relative[:3,3])),
                reference_timestamp=robot['timestamp'],object_last_seen=track['last_seen'],
                object_pose_stale=track['status']!='seen',
                pose_kind=track['pose_kind'],
                note='Distance to estimated OBB center, not surface clearance, reachability or grasp pose')
