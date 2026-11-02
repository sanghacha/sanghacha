"""Illustrate recorded RGB-D decisions, explicitly separate from robot motion."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from planner.act_comparison import CONDITIONS


def render(args):
    available={font.name for font in font_manager.fontManager.ttflist}
    korean_font=next((name for name in ('Malgun Gothic','NanumGothic','Noto Sans CJK KR') if name in available),None)
    if korean_font is None:
        raise RuntimeError('Install a Korean font: Malgun Gothic, NanumGothic or Noto Sans CJK KR')
    plt.rcParams.update({'font.family':korean_font,'axes.unicode_minus':False})
    manifest=json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
    selected=None
    for case in manifest['cases']:
        if args.case and case['case_id']!=args.case: continue
        paths=[args.input/'runs'/m/case['case_id']/'result.json' for m in CONDITIONS]
        if not all(p.exists() for p in paths): continue
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
        if args.case or (case['group']=='visible_blocked' and all(r['success'] for r in rows) and
                         rows[0]['non_target_removals']>rows[2]['non_target_removals']):
            selected=(case,rows);break
    assert selected,'no suitable completed example; no fabricated scene'
    case,rows=selected
    fig,axes=plt.subplots(3,3,figsize=(9,9),layout='constrained')
    titles=['A: 거리 우선 탐색','B: 숨은 공간 단서 제외','C: 제안 방법 (HSG)']
    sources=[]
    for i,(method,result) in enumerate(zip(CONDITIONS,rows)):
        folder=args.input/'runs'/method/case['case_id']
        inputs=json.loads((folder/'plan_inputs.json').read_text(encoding='utf-8'))
        frames=[0,min(1,len(inputs)-1),len(inputs)-1]
        for j,frame in enumerate(frames):
            capture=folder/'perception'/f'snapshot_{frame:03d}'/'inputs/frame_000'
            source=capture/'fixed_oblique_rgb.png';sources.append(str(source))
            rgb=np.array(Image.open(source))
            meta=json.loads((capture/'capture.json').read_text(encoding='utf-8'))['cameras']['fixed_oblique']
            point=np.r_[inputs[frame]['goal']['object_pose']['position'],1.]
            camera=np.linalg.solve(np.array(meta['T_camera_to_base']),point)[:3]
            pixel=np.array(meta['K'])@camera;pixel=pixel[:2]/pixel[2]
            ax=axes[i,j];ax.imshow(rgb)
            ax.scatter(*pixel,s=220,facecolors='none',edgecolors='#ffffff',linewidths=2)
            ax.scatter(*pixel,s=260,facecolors='none',edgecolors='#222222',linewidths=.7)
            # Shelves occupy this region in the fixed calibrated oblique camera.
            ax.set_xlim(195,440);ax.set_ylim(415,5);ax.set_xticks([]);ax.set_yticks([])
            label=f'비목표 {frame}개 제거\n'+('목표 집기 예정' if inputs[frame]['goal']['object_id']==inputs[frame]['task']['target_object_id'] else '탐색 / 방해물 제거')
            ax.set_title(label,fontsize=10)
            if j==0: ax.set_ylabel(titles[i],fontsize=11)
            for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle(f"MuJoCo RGB-D 계획 이력 (시드 {case['seed']})\n이상적 물체 제거 · 로봇 팔 고정 · ACT·물리 파지 제외",fontsize=13)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=220)
    fig.savefig(args.output.with_suffix('.svg'))
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(dict(case=case['case_id'],seed=case['seed'],
        selection='illustrative first visible-blocked case with all methods successful and C using fewer removals than A; post-evaluation illustration only',
        sources=sources,removals={r['condition']:r['non_target_removals'] for r in rows}),indent=2),encoding='utf-8')
    print(case['case_id'],[(r['condition'],r['non_target_removals']) for r in rows])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case')
    render(parser.parse_args())
