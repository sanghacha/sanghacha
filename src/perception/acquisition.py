"""Synchronized two-view acquisition. No object states/IDs/contacts exported."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from .robot_frames import capture_robot

CAMERAS = ('fixed_top', 'fixed_oblique')

def configured_regions(config):
    """Static configured work regions, never simulator object poses or labels."""
    shelf=config['shelf_scene']
    regions=[dict(r,region_type='shelf',floor_top_m=shelf['floor_thickness']) for r in shelf['shelves']]
    tray=config['placement_trays']
    for ident,key in (('buffer_tray','buffer_position'),('output_tray','destination')):
        regions.append(dict(id=ident,center=config[key][:2],size=tray['size'],
                            region_type='tray',floor_top_m=tray['floor_thickness']))
    return regions


def capture(env, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    m, d = env.model, env.data
    timestamp = float(d.time-env._start_time)
    h,w = env.config['image_height'],env.config['image_width']
    result = dict(schema_version=3, episode_id=env._episode, timestamp=timestamp,
                  frame_id='robot_base', depth_units='m', camera_convention='OpenCV x-right y-down z-forward',
                  cameras={}, regions=configured_regions(env.config) if env.shelf_scene else [],
                  floor_top_m=env.config['shelf_scene']['floor_thickness'],
                  geometry_source='configured static shelves and trays; no simulator object state')
    result['robot']=capture_robot(env)
    # No physics step occurs between these renders.
    with mujoco.Renderer(m,height=h,width=w) as renderer:
        for name in CAMERAS:
            cid=m.camera(name).id
            assert m.cam_bodyid[cid]==0
            renderer.update_scene(d,camera=name)
            rgb=renderer.render().copy()
            renderer.enable_depth_rendering()
            depth=renderer.render().copy()
            renderer.disable_depth_rendering()
            fy=h/(2*np.tan(np.deg2rad(m.cam_fovy[cid])/2))
            k=np.array([[fy,0,w/2],[0,fy,h/2],[0,0,1.]])
            transform=np.eye(4)
            transform[:3,:3]=d.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.])
            transform[:3,3]=d.cam_xpos[cid]
            image_path=directory/f'{name}_rgb.png'
            Image.fromarray(rgb).save(image_path)
            depth_path=directory/f'{name}_depth.npz'
            np.savez_compressed(depth_path,depth=depth)
            result['cameras'][name]=dict(width=w,height=h,K=k.tolist(),T_camera_to_base=transform.tolist(),
                        timestamp=timestamp,rgb_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest())
    (directory/'capture.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def load_frame(directory):
    directory=Path(directory)
    meta=json.loads((directory/'capture.json').read_text(encoding='utf-8'))
    assert meta['frame_id']=='robot_base' and meta['depth_units']=='m'
    frames={}
    for name,c in meta['cameras'].items():
        path=directory/f'{name}_rgb.png'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==c['rgb_sha256'], 'RGB hash mismatch'
        depth_path=directory/f'{name}_depth.npz'
        if meta['schema_version']>=2:
            assert hashlib.sha256(depth_path.read_bytes()).hexdigest()==c['depth_sha256'],'Depth hash mismatch'
        rgb=np.array(Image.open(path).convert('RGB'))
        with np.load(directory/f'{name}_depth.npz',allow_pickle=False) as f:
            depth=f['depth'].copy()
        assert depth.shape==rgb.shape[:2] and np.isfinite(depth).all()
        assert c['timestamp']==meta['timestamp']
        frames[name]=dict(rgb=rgb,depth=depth,K=np.array(c['K']),T=np.array(c['T_camera_to_base']))
    return meta,frames
