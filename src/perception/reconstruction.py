"""CPU reconstruction from real model predictions; no ground truth API."""
import json
from pathlib import Path
import networkx as nx
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from .geometry import backproject,clean_cloud,describe
from .robot_frames import add_robot_frames

def box_iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    inter=float(np.prod(np.maximum(hi-lo,0)))
    return inter/max(1.,np.prod(np.array(a[2:])-a[:2])+np.prod(np.array(b[2:])-b[:2])-inter)

def proposals(name,frame,meta,prediction_dir):
    p=Path(prediction_dir)/name
    report=json.loads((p/'detections.json').read_text())
    assert report['rgb_sha256']==meta['cameras'][name]['rgb_sha256'],'Prediction/input mismatch'
    with np.load(p/'masks.npz',allow_pickle=False) as f:
        masks=f['masks'].copy();scores=f['scores'].copy()
    assert masks.ndim==3 and masks.dtype==bool and masks.shape[1:]==frame['depth'].shape
    assert len(masks)==len(scores)
    xyz=backproject(frame['depth'],frame['K'],frame['T'])
    gate=np.zeros(masks.shape[1:],dtype=bool)
    for region in meta['regions']:
        cx,cy=region['center'];sx,sy=region['size']
        gate|=(abs(xyz[:,:,0]-cx)<sx/2-.007)&(abs(xyz[:,:,1]-cy)<sy/2-.007)
    gate&=(xyz[:,:,2]>meta['floor_top_m']+.003)&(xyz[:,:,2]<.20)&(frame['depth']>0)&np.isfinite(frame['depth'])
    parts=[]
    for i,mask in enumerate(masks):
        valid=mask&gate
        if valid.sum()<25 or valid.sum()/max(1,mask.sum())<.55:continue
        components,n=label(valid)
        for j in range(1,n+1):
            part=components==j
            if part.sum()>=25:parts.append((part,i,float(scores[i])))
    parts.sort(key=lambda x:int(x[0].sum()),reverse=True)
    kept=[];result=[]
    for mask,raw_id,quality in parts:
        if any((mask&k).sum()/max(1,min(mask.sum(),k.sum()))>.80 for k in kept):continue
        pts=xyz[mask];colors=frame['rgb'][mask]/255.
        extent=np.ptp(pts,axis=0)
        if max(extent)>.13:continue # reject merged shelf/background surfaces
        pcd=clean_cloud(pts,colors)
        if len(pcd.points)<8:continue
        desc=describe(pcd)
        y,x=np.where(mask);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        detections=report['detections']
        best=max(detections,key=lambda d:box_iou(box,d['box'])) if detections else None
        hypotheses=[]
        if best and box_iou(box,best['box'])>=.3:
            hypotheses=[dict(label=best['label'],score=best['score'],box_iou=box_iou(box,best['box']))]
        result.append(dict(desc,_points=np.asarray(pcd.points).copy(),_colors=np.asarray(pcd.colors).copy(),
              _mask=mask,observations=[dict(camera=name,raw_mask=raw_id,bbox=box,sam_score=quality)],
              label_hypotheses=hypotheses,association_status='single_view'))
        kept.append(mask)
    return result

def public(track):
    return {k:v for k,v in track.items() if not k.startswith('_')}

def export_graph(tracks,meta,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    g=nx.DiGraph(frame_id='robot_base',source='rgbd',timestamp=meta['timestamp'],episode_id=meta['episode_id'],
                 pose_note='Visible-surface OBB estimates, not model-registered 6D poses')
    g.add_node('workspace',kind='workspace')
    for r in meta['regions']:
        g.add_node(r['id'],kind='region',**r)
        g.add_edge(r['id'],'workspace',relation='inside')
    for track in tracks:
        ident=track['instance_id']
        g.add_node(ident,kind='object_candidate',**public(track))
        if track['status']=='seen':
            c=track['visible_surface_centroid_m']
            for r in meta['regions']:
                if all(abs(c[i]-r['center'][i])<=r['size'][i]/2 for i in (0,1)):
                    g.add_edge(ident,r['id'],relation='inside',evidence='RGB-D location + configured region boundary')
                    break
            p=o3d.geometry.PointCloud()
            p.points=o3d.utility.Vector3dVector(track['_points'])
            p.colors=o3d.utility.Vector3dVector(track['_colors'])
            assert o3d.io.write_point_cloud(str(directory/f'{ident}.ply'),p)
    add_robot_frames(g,meta,tracks)
    # Partial visible surfaces do not prove contact/support; do not emit on/blocks.
    payload=nx.node_link_data(g,edges='edges')
    (directory/'scene_graph.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return g
