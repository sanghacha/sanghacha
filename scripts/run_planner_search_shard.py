"""Schedule disjoint frozen cases; no benchmark/planner logic changes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import benchmark_planner_search as benchmark


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--shards',type=int,required=True)
    args=parser.parse_args();args.output=args.input
    assert 0<=args.shard<args.shards
    manifest=json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
    if (args.input/'layout_protocol.json').exists():
        import benchmark_planner_occlusion as stress
        stress.ShelfScene.sample_layout=stress.random_order
        benchmark.source_hashes=stress.hashes
    assert benchmark.source_hashes()==manifest['protocol']['source_sha256']
    assert json.loads((args.input/'config.json').read_text())==manifest['config']
    cases=[c for c in manifest['cases'] if int(c['case_id'].split('_')[1])%args.shards==args.shard]
    metadata=dict(manifest_sha256=benchmark.digest(manifest),shard=args.shard,shards=args.shards,
        cases=[c['case_id'] for c in cases],scheduler_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        note='independent scenes scheduled concurrently; no runtime comparisons; same frozen run_case and planners')
    path=args.input/f'shard_{args.shard}_of_{args.shards}.json'
    if path.exists(): assert json.loads(path.read_text())==metadata
    else: benchmark.dump(path,metadata)
    for case in cases:
        offset=int(case['case_id'].split('_')[1])%3
        for method in benchmark.CONDITIONS[offset:]+benchmark.CONDITIONS[:offset]:
            row=benchmark.run_case(args,manifest,case,method)
            print(json.dumps({k:row[k] for k in ('case_id','condition','success','non_target_removals','error')}),flush=True)


if __name__=='__main__': main()
