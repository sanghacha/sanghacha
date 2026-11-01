"""Validate the existing perception data contract at the package boundary."""
from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np


def json_value(value):
    """Return a detached JSON value, including NumPy arrays/scalars."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Data must contain only finite JSON values")


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def rotation(value, name):
    result = np.asarray(value, dtype=float)
    if (result.shape != (3, 3) or not np.isfinite(result).all()
            or not np.allclose(result.T @ result, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(result), 1., atol=1e-6, rtol=0)):
        raise ValueError(f"{name} must be a proper 3x3 rotation")
    return result


def transform_matrix(value, name):
    result = np.asarray(value, dtype=float)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError(f"{name} must be a finite homogeneous 4x4 transform")
    rotation(result[:3, :3], name)
    return result


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def identifier(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def snapshot(objects, metadata):
    """Normalize and check inputs; large private point/mask arrays are excluded."""
    tracks = [json_value({k: v for k, v in obj.items() if not k.startswith('_')}) for obj in objects]
    meta = json_value(deepcopy(metadata))
    timestamp = number(meta['timestamp'], 'timestamp')
    if timestamp < 0:
        raise ValueError("timestamp must be nonnegative")
    identifier(meta['episode_id'], 'episode_id')
    if meta.get('frame_id', 'robot_base') != 'robot_base':
        raise ValueError("Input coordinates must be expressed in robot_base")
    if meta.get('depth_units', 'm') != 'm' or meta.get('units', 'm') != 'm':
        raise ValueError("Input coordinates must use metres")
    if meta.get('source', 'rgbd') != 'rgbd':
        raise ValueError("This package expects tracked RGB-D observations")
    occupied = {'workspace', 'robot_base', 'gripper_tcp'}

    def reserve(value, name):
        identifier(value, name)
        if value in occupied:
            raise ValueError(f"Duplicate or reserved node ID: {value}")
        occupied.add(value)

    for region in meta['regions']:
        reserve(region['id'], 'region.id')
        if 'kind' in region:
            raise ValueError("region.kind is reserved")
        vector(region['center'], 2, 'region.center')
        if np.any(vector(region['size'], 2, 'region.size') <= 0):
            raise ValueError("Region sizes must be positive")
    for name, camera in meta.get('cameras', {}).items():
        identifier(name, 'camera name')
        reserve('camera:' + name, 'camera ID')
        transform_matrix(camera['T_camera_to_base'], 'camera.T_camera_to_base')
        k = np.asarray(camera['K'], dtype=float)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("Camera K must have positive focal lengths and a [0,0,1] last row")
    robot = meta.get('robot')
    if robot is not None:
        if not np.isclose(number(robot['timestamp'], 'robot.timestamp'), timestamp, atol=1e-9, rtol=0):
            raise ValueError("Robot state must have the snapshot timestamp")
        identifier(robot['robot_id'], 'robot.robot_id')
        identifier(robot['source'], 'robot.source')
        transform_matrix(robot['T_world_base'], 'robot.T_world_base')
        if len(robot['joint_names']) != len(robot['joint_positions_rad']):
            raise ValueError("Joint name and position counts differ")
        if len(set(robot['joint_names'])) != len(robot['joint_names']):
            raise ValueError("Joint names must be unique")
        for name in robot['joint_names']:
            identifier(name, 'joint name')
        for value in robot['joint_positions_rad']:
            number(value, 'joint position')
        if not 0 <= number(robot['gripper_closed_fraction'], 'gripper_closed_fraction') <= 1:
            raise ValueError("gripper_closed_fraction must be within [0,1]")
        parents = {}
        for link in robot['links']:
            reserve(link['id'], 'link.id')
            parents[link['id']] = link['parent']
            transform_matrix(link['T_base_link'], 'link.T_base_link')
        for link_id in parents:
            visited = set()
            current = link_id
            while current != 'robot_base':
                if current in visited or current not in parents:
                    raise ValueError("Robot links must form an acyclic tree rooted at robot_base")
                visited.add(current)
                current = parents[current]
        tcp = robot['tcp']
        if tcp['id'] != 'gripper_tcp' or tcp['parent'] not in {'robot_base', *parents}:
            raise ValueError("TCP must be gripper_tcp with a represented robot parent")
        transform_matrix(tcp['T_base_tcp'], 'tcp.T_base_tcp')
    for track in tracks:
        reserve(track['instance_id'], 'instance_id')
        if 'id' in track or 'kind' in track:
            raise ValueError("Object id/kind are output fields; use instance_id as input")
        vector(track['visible_surface_centroid_m'], 3, 'visible_surface_centroid_m')
        box = track['obb']
        vector(box['center_m'], 3, 'obb.center_m')
        rotation(box['rotation'], 'obb.rotation')
        if np.any(vector(box['extent_m'], 3, 'obb.extent_m') <= 0):
            raise ValueError("OBB extents must be positive")
        identifier(track['pose_kind'], 'pose_kind')
        if track['status'] not in ('seen', 'history'):
            raise ValueError("Object status must be seen or history")
        last = number(track['last_seen'], 'last_seen')
        first = number(track.get('first_seen', last), 'first_seen')
        if not 0 <= first <= last <= timestamp + 1e-9:
            raise ValueError("Object observation times must precede the snapshot")
        if track['status'] == 'seen' and not np.isclose(last, timestamp, atol=1e-9, rtol=0):
            raise ValueError("A seen object's last_seen must equal the snapshot timestamp")
    return tracks, meta
