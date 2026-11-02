"""Independent occlusion stress family: random order within existing columns.

Production scene generation is unchanged. Same counts, sizes, shelf positions,
cameras, physics settling, target selection and planners as the ordinary test.
Within-column object order is uniformly permuted BEFORE any camera observation.
This permits larger boxes above smaller ones rather than always largest first.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import benchmark_planner_search as benchmark
from pipeline.shelf_scene import ShelfScene

ORIGINAL_SAMPLE=ShelfScene.sample_layout
ORIGINAL_HASHES=benchmark.source_hashes


def permute_columns(layout,definitions,floor,rng):
    groups={}
    for key,(position,quaternion) in layout.items():
        groups.setdefault(tuple(position[:2]),[]).append(key)
    result={}
    for xy,keys in sorted(groups.items()):
        height=floor
        for key in rng.permutation(sorted(keys)):
            key=str(key);size=definitions[key]['size']
            result[key]=([*xy,height+size[2]/2+.0002],layout[key][1])
            height+=size[2]+.0002
    return result


def random_order(self,rng):
    return permute_columns(ORIGINAL_SAMPLE(self,rng),self.objects,self.config['floor_thickness'],rng)


def hashes():
    return dict(ORIGINAL_HASHES(),**{
        'scripts/benchmark_planner_occlusion.py':sha256(Path(__file__).read_bytes()).hexdigest(),
        'src/pipeline/shelf_scene.py':sha256((benchmark.ROOT/'src/pipeline/shelf_scene.py').read_bytes()).hexdigest()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('freeze','run'))
    parser.add_argument('--output',type=Path,default=benchmark.ROOT/'outputs/planner_search/occlusion')
    parser.add_argument('--count',type=int,default=40)
    parser.add_argument('--start-seed',type=int,default=9100)
    parser.add_argument('--max-scan',type=int,default=500)
    args=parser.parse_args();args.manifest=args.output/'manifest.json'
    ShelfScene.sample_layout=random_order
    benchmark.source_hashes=hashes
    descriptor=dict(family='random within-column object order',
        intervention='uniformly permute object order in each sampled stack before physics settling',
        unchanged=['27 boxes','size/color vocabulary','column xy','cameras','planner code','RGB-D perception'],
        purpose='separate stress test of occlusion; never pooled with ordinary family',source_sha256=hashes())
    path=args.output/'layout_protocol.json'
    if args.operation=='freeze':
        assert not path.exists()
        benchmark.dump(path,descriptor)
        benchmark.freeze(args)
    else:
        assert json.loads(path.read_text(encoding='utf-8'))==descriptor
        benchmark.run(args)


if __name__=='__main__': main()
