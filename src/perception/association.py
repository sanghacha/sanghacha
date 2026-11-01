"""Conservative one-to-one cross-view and temporal association."""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import clean_cloud,describe

def cost(a,b,temporal=False):
    distance=np.linalg.norm(np.array(a['visible_surface_centroid_m'])-b['visible_surface_centroid_m'])
    # Normalize brightness: a shaded side face and a lit top face can be one object.
    ca=np.array(a['mean_rgb']);cb=np.array(b['mean_rgb'])
    color=np.linalg.norm(ca/max(float(ca.sum()),1e-6)-cb/max(float(cb.sum()),1e-6))
    limit=.10 if temporal else .065
    if distance>limit or color>.24:return 1e6
    lo=np.maximum(a['aabb_min_m'],b['aabb_min_m']);hi=np.minimum(a['aabb_max_m'],b['aabb_max_m'])
    gap=np.linalg.norm(np.maximum(lo-hi,0))
    if not temporal and gap>.018:return 1e6
    return distance/limit+color/.24+.2*gap/.018

def match(left,right,temporal=False):
    if not left or not right:return [],list(range(len(left))),list(range(len(right)))
    matrix=np.array([[cost(a,b,temporal) for b in right] for a in left])
    rows,cols=linear_sum_assignment(matrix)
    accepted=[]
    for i,j in zip(rows,cols):
        value=matrix[i,j]
        if value>1.35:continue
        # Reject an ambiguous choice instead of forcing nearby identical objects together.
        alternatives=np.r_[np.delete(matrix[i],j),np.delete(matrix[:,j],i)]
        if alternatives.size and alternatives.min()-value<.12:continue
        accepted.append((int(i),int(j),float(value)))
    return accepted,[i for i in range(len(left)) if all(a!=i for a,b,c in accepted)], [j for j in range(len(right)) if all(b!=j for a,b,c in accepted)]

def fuse_views(by_camera):
    fused=[]
    for name,items in by_camera.items():
        pairs,_,unmatched=match(fused,items)
        for i,j,confidence in pairs:
            a,b=fused[i],items[j]
            p=clean_cloud(np.r_[a['_points'],b['_points']],np.r_[a['_colors'],b['_colors']])
            fused[i]=dict(describe(p),_points=np.asarray(p.points).copy(),_colors=np.asarray(p.colors).copy(),
                observations=a['observations']+b['observations'],label_hypotheses=a['label_hypotheses']+b['label_hypotheses'],
                association_cost=confidence,association_status='accepted_hypothesis')
        fused.extend(copy.deepcopy(items[j]) for j in unmatched)
    return fused

def action_associations(tracks,objects,events,regions,episode,previous_time,timestamp):
    """Command intent plus compatible unique RGB-D evidence, not proof of motion."""
    region_map={r['id']:r for r in regions}
    proposals=[];considered=set()
    for event in events:
        key=event['source_instance_id']
        if event['episode_id']!=episode or key not in tracks:continue
        if not previous_time<event['completed_at']<=timestamp or event['started_at']>event['completed_at']:continue
        considered.add(key)
        region=region_map.get(event['destination_region'])
        if region is None:continue
        expected=np.asarray(event['expected_position_m'],dtype=float)
        assert expected.shape==(3,) and np.isfinite(expected).all()
        source=tracks[key]
        ca=np.array(source['mean_rgb']);ca/=max(float(ca.sum()),1e-6)
        size_a=np.sort(np.array(source['aabb_max_m'])-source['aabb_min_m'])
        options=[]
        for j,obj in enumerate(objects):
            center=np.array(obj['visible_surface_centroid_m'])
            if any(abs(center[i]-region['center'][i])>region['size'][i]/2-.007 for i in (0,1)):continue
            distance=float(np.linalg.norm(center-expected))
            cb=np.array(obj['mean_rgb']);cb/=max(float(cb.sum()),1e-6)
            chroma=float(np.linalg.norm(ca-cb))
            size_b=np.sort(np.array(obj['aabb_max_m'])-obj['aabb_min_m'])
            ratio=size_b/np.maximum(size_a,.001)
            if distance>.04 or chroma>.12 or np.any(ratio<.5) or np.any(ratio>1.8):continue
            shape=float(np.mean(np.abs(np.log(ratio))))
            score=distance/.04+chroma/.12+.3*shape
            if score<=1.5:
                options.append((score,j,dict(method='action_prior_and_rgbd_hypothesis',event_id=event['event_id'],
                   destination_region=event['destination_region'],expected_position_m=expected.tolist(),
                   observation_distance_to_command_m=distance,chromaticity_difference=chroma,
                   shape_log_ratio_error=shape,score_kind='heuristic_not_calibrated_probability')))
        options.sort(key=lambda value:value[0])
        if not options:continue
        if len(options)>1 and options[1][0]-options[0][0]<.25:continue
        score,j,evidence=options[0]
        proposals.append((key,j,score,evidence))
    # Competing action hints never assign the same observation twice.
    accepted=[p for p in proposals if sum(q[1]==p[1] for q in proposals)==1]
    return accepted,considered


class ObjectTracker:
    def __init__(self,state=None):
        self.tracks={};self.next_id=1;self.episode=None;self.timestamp=-1
        if state is not None:
            self.episode=state['episode'];self.timestamp=state['timestamp'];self.next_id=state['next_id']
            self.tracks={t['instance_id']:t for t in state['tracks']}

    def update(self,objects,episode,timestamp,action_events=(),regions=()):
        if self.episode!=episode:
            self.tracks={};self.next_id=1;self.timestamp=-1;self.episode=episode
        assert timestamp>self.timestamp,'Duplicate/out-of-order snapshot'
        previous_time=self.timestamp
        self.timestamp=timestamp
        action_pairs,commanded=action_associations(self.tracks,objects,action_events,regions,episode,previous_time,timestamp)
        assigned_objects={j for key,j,c,evidence in action_pairs}
        # History is retained but only recent tracks are association candidates.
        keys=[k for k,v in self.tracks.items() if v['missed_snapshots']<=5 and k not in commanded]
        indices=[j for j in range(len(objects)) if j not in assigned_objects]
        matches,unseen,new=match([self.tracks[k] for k in keys],[objects[j] for j in indices],temporal=True)
        for k in self.tracks:
            self.tracks[k]['status']='history'
            self.tracks[k]['missed_snapshots']+=1
        for i,j,c in matches:
            key=keys[i]
            first=self.tracks[key]['first_seen']
            self.tracks[key]=dict(objects[indices[j]],instance_id=key,status='seen',missed_snapshots=0,
                first_seen=first,last_seen=timestamp,tracking_cost=c,tracking_evidence=dict(method='local_geometry_and_color'))
        for key,j,c,evidence in action_pairs:
            first=self.tracks[key]['first_seen']
            self.tracks[key]=dict(objects[j],instance_id=key,status='seen',missed_snapshots=0,
                first_seen=first,last_seen=timestamp,tracking_cost=c,tracking_evidence=evidence)
        for j in new:
            key=f'instance_{self.next_id:04d}';self.next_id+=1
            self.tracks[key]=dict(objects[indices[j]],instance_id=key,status='seen',missed_snapshots=0,
                first_seen=timestamp,last_seen=timestamp,tracking_cost=None)
        return list(self.tracks.values())
