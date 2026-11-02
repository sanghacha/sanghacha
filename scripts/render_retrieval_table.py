"""Poster table figure generated from the recorded five-removal budget results."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]


def main():
    available={font.name for font in font_manager.fontManager.ttflist}
    font=next((name for name in ('Malgun Gothic','NanumGothic','Noto Sans CJK KR') if name in available),None)
    if font is None:
        raise RuntimeError('A Korean font is required: Malgun Gothic, NanumGothic or Noto Sans CJK KR')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'svg.fonttype':'path'})
    rows=[];sources={}
    for family,label in [('ordinary','기본 적재 환경'),('occlusion','적재 순서 무작위 가림 조건')]:
        path=ROOT/'experiments/planner_search'/family/'results.json'
        summary=json.loads(path.read_text(encoding='utf-8'))['summary']['hidden']
        a,c=summary['a_greedy'],summary['c_full']
        assert a['n']==c['n'] and a['n']>0
        rows.append(dict(environment=label,n=a['n'],baseline_successes=a['success_within_removal_budget']['5'],
                         proposed_successes=c['success_within_removal_budget']['5']))
        sources[str(path.relative_to(ROOT))]=sha256(path.read_bytes()).hexdigest()
    fig,ax=plt.subplots(figsize=(10.8,4.8))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ink='#17212b';muted='#586575';blue='#176b96';rule='#bcc6ce'
    ax.text(.06,.94,'목표 물체 회수율 비교',fontsize=21,weight='bold',color=ink,va='top')
    ax.text(.06,.83,'초기 목표 미식별 장면 · 비목표 물체 최대 5개 제거',fontsize=12,color=muted,va='top')
    edges=(.06,.48,.71,.94)
    top,header_bottom,middle,bottom=.73,.59,.415,.24
    ax.add_patch(Rectangle((edges[0],header_bottom),edges[-1]-edges[0],top-header_bottom,
                           facecolor='#f3f5f7',edgecolor='none'))
    ax.add_patch(Rectangle((edges[2],bottom),edges[3]-edges[2],top-bottom,
                           facecolor='#eaf4fa',edgecolor='none'))
    for y,color,width in [(top,ink,1.5),(header_bottom,rule,.9),(middle,rule,.65),(bottom,ink,1.2)]:
        ax.plot((edges[0],edges[-1]),(y,y),color=color,lw=width)
    header_y=(top+header_bottom)/2
    ax.text(.085,header_y,'실험 환경',fontsize=13,weight='bold',color=ink,va='center')
    ax.text((edges[1]+edges[2])/2,header_y,'가까운 물체부터 제거\n(A)',fontsize=12,
            color=ink,ha='center',va='center',linespacing=1.4)
    ax.text((edges[2]+edges[3])/2,header_y,'제안 방법\n(HSG, C)',fontsize=13,weight='bold',
            color=blue,ha='center',va='center',linespacing=1.4)
    for row,y in zip(rows,((header_bottom+middle)/2,(middle+bottom)/2)):
        ax.text(.085,y+.022,row['environment'],fontsize=13,color=ink,va='center')
        ax.text(.085,y-.032,f"{row['n']}개 장면",fontsize=10,color=muted,va='center')
        for left,right,key,color,bold in [(edges[1],edges[2],'baseline_successes',ink,False),
                                          (edges[2],edges[3],'proposed_successes',blue,True)]:
            rate=100*row[key]/row['n']
            ax.text((left+right)/2,y,f'{rate:.1f}%',fontsize=26,color=color,
                    weight='bold' if bold else 'normal',ha='center',va='center')
    ax.text(.06,.16,'※ 물체를 성공적으로 치운다고 가정한 MuJoCo 시뮬레이션 결과.',fontsize=10,color=muted,va='top')
    ax.text(.06,.105,'   실제 로봇의 파지·이동·배치 성공률은 평가하지 않음.',fontsize=10,color=muted,va='top')
    output=ROOT/'docs/figures/retrieval_comparison_ko'
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=300,facecolor='white')
    fig.savefig(output.with_suffix('.svg'),facecolor='white')
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(removal_budget=5,
        population='initially unlocalized targets',executor='ideal removal; no ACT or physical grasp',
        rows=rows,source_sha256=sources),indent=2,ensure_ascii=False),encoding='utf-8')
    print(output.with_suffix('.png'))


if __name__=='__main__': main()
