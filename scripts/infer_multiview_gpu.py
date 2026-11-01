"""Standalone RunPod worker. Input: RGB + hashes. Output: AMG masks + DINO boxes.

No depth, simulator metadata or object GT is needed on the GPU server.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import pipeline,Sam2Model,Sam2Processor,AutoProcessor,AutoModelForZeroShotObjectDetection

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt',default='box. cube. block.')
    args=p.parse_args()
    images=sorted(args.inputs.glob('frame_*/*_rgb.png'))
    if not images:p.error('No frame_*/*_rgb.png input images found')
    args.output.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    sam=Sam2Model.from_pretrained('facebook/sam2.1-hiera-large').to('cuda').eval()
    sp=Sam2Processor.from_pretrained('facebook/sam2.1-hiera-large')
    amg=pipeline('mask-generation',model=sam,image_processor=sp.image_processor,device=0)
    dp=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
    dino=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base').to('cuda').eval()
    summary=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,sam_revision=sam.config._commit_hash,
        dino_revision=dino.config._commit_hash,prompt=args.prompt,frames=[])
    for path in images:
        image=Image.open(path).convert('RGB')
        assert max(image.size)<=1280, 'Capture resolution is too large for this worker budget'
        start=time.perf_counter()
        with torch.inference_mode():
            output=amg(image,points_per_crop=32,points_per_batch=16,pred_iou_thresh=.80,stability_score_thresh=.90)
            inputs=dp(images=image,text=args.prompt,return_tensors='pt').to('cuda')
            detected=dino(**inputs)
        detections=dp.post_process_grounded_object_detection(detected,inputs.input_ids,threshold=.20,
                          text_threshold=.20,target_sizes=[image.size[::-1]])[0]
        torch.cuda.synchronize()
        name=path.stem.removesuffix('_rgb')
        out=args.output/path.parent.name/name
        out.mkdir(parents=True,exist_ok=True)
        masks=np.asarray(output['masks'],dtype=bool)
        if not len(masks):masks=np.zeros((0,image.height,image.width),bool)
        np.savez_compressed(out/'masks.npz',masks=masks,scores=np.asarray(output['scores'],np.float32))
        labels=detections.get('text_labels',detections.get('labels',[]))
        report=dict(camera_id=name,width=image.width,height=image.height,
                    rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    inference_seconds=time.perf_counter()-start,raw_masks=len(masks),
                    detections=[dict(box=b,label=str(l),score=float(s)) for b,l,s in
                        zip(detections['boxes'].cpu().tolist(),labels,detections['scores'].cpu())])
        (out/'detections.json').write_text(json.dumps(report,indent=2))
        summary['frames'].append(dict(frame=path.parent.name,**report))
        print(path.parent.name,name,'masks',len(masks),'boxes',len(report['detections']),flush=True)
    summary['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30
    summary['peak_reserved_GiB']=torch.cuda.max_memory_reserved()/2**30
    (args.output/'run.json').write_text(json.dumps(summary,indent=2))
    print('MULTIVIEW_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':
    main()
