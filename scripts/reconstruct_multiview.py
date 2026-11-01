"""Run after RGB-only GPU inference. CPU: clouds, association, OBB, TSDF, graph."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from perception.acquisition import load_frame
from perception.geometry import snapshot_tsdf
from perception.association import fuse_views,ObjectTracker
from perception.reconstruction import proposals,export_graph,public
from rgbd_hssg import graph_delta
from perception.visualization import instance_color

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,default=Path('outputs/multiview/robot_relative_seed_42'))
    p.add_argument('--predictions',type=Path,default=Path('outputs/multiview/robot_relative_seed_42/predictions'))
    p.add_argument('--output',type=Path,default=Path('outputs/multiview/robot_relative_seed_42/reconstruction'))
    p.add_argument('--resume-state',type=Path)
    p.add_argument('--disable-action-events',action='store_true',help='Ablation: ordinary local association only')
    args=p.parse_args()
    framepaths=[path for path in sorted(args.capture.glob('frame_*')) if path.is_dir()]
    if not framepaths:p.error('No frame_* capture directories found')
    args.output.mkdir(parents=True,exist_ok=True)
    tracker=ObjectTracker(json.loads(args.resume_state.read_text()) if args.resume_state else None);summary=[]
    previous_graph=None
    for framepath in framepaths:
        meta,frames=load_frame(framepath)
        if previous_graph is not None and previous_graph.graph['episode_id']!=meta['episode_id']:
            previous_graph=None
        if meta['episode_id']==tracker.episode and meta['timestamp']<=tracker.timestamp:
            continue
        by_camera={n:proposals(n,f,meta,args.predictions/framepath.name) for n,f in frames.items()}
        objects=fuse_views(by_camera)
        events=[] if args.disable_action_events else meta.get('action_events',[])
        tracks=tracker.update(objects,meta['episode_id'],meta['timestamp'],events,meta['regions'])
        out=args.output/framepath.name
        graph=export_graph(tracks,meta,out)
        delta=graph_delta(previous_graph,graph)
        (out/'graph_delta.json').write_text(json.dumps(delta,indent=2))
        previous_graph=graph
        tsdf_points=snapshot_tsdf(frames,out/'snapshot_tsdf.ply')
        for name,items in by_camera.items():
            rgb=frames[name]['rgb'].copy()
            def owner(obj):
                observation=obj['observations'][0]
                return next((t['instance_id'] for t in tracks if t['status']=='seen' and observation in t['observations']),None)
            for i,obj in enumerate(items):
                ident=owner(obj) or f'{name}_candidate_{i:02d}'
                mask=obj['_mask'];rgb[mask]=(.55*rgb[mask]+.45*instance_color(ident)).astype('uint8')
            img=Image.fromarray(rgb);draw=ImageDraw.Draw(img)
            font=ImageFont.load_default(size=13)
            for i,obj in enumerate(items):
                x,y,_,_=obj['observations'][0]['bbox']
                ident=owner(obj) or f'candidate_{i:02d}'
                draw.text((x,y),ident,font=font,fill='white',stroke_width=1,stroke_fill='black')
            img.save(out/f'{name}_candidates.png')
        report=dict(frame=framepath.name,timestamp=meta['timestamp'],camera_candidates={n:len(v) for n,v in by_camera.items()},
            fused_current_objects=len(objects),multi_view_hypotheses=sum(len(o['observations'])>1 for o in objects),
            current_ids=[t['instance_id'] for t in tracks if t['status']=='seen'],
            history_ids=[t['instance_id'] for t in tracks if t['status']=='history'],tsdf_points=tsdf_points,
            graph_nodes=graph.number_of_nodes(),graph_edges=graph.number_of_edges(),
            caveat='Candidates may be split/merged/missed. No GT used. OBB pose is approximate. TSDF reset each snapshot.')
        (out/'report.json').write_text(json.dumps(report,indent=2))
        summary.append(report);print(report)
    if summary:
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'tracker_state.json').write_text(json.dumps(dict(episode=tracker.episode,timestamp=tracker.timestamp,
           next_id=tracker.next_id,tracks=[public(t) for t in tracker.tracks.values()]),indent=2))

if __name__=='__main__':
    main()
