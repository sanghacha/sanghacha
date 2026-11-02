"""Publication-style diagram of the implemented RGB-D / indexed HSG / ACT path."""
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/figures'
ASSETS = OUT/'assets'
PDF = ROOT/'output/pdf/system_architecture.pdf'
INK = '#263645'
MUTED = '#5d6c78'
BLUE = '#396f9f'
AMBER = '#ae762e'
TEAL = '#247d72'
GRAY = '#64717b'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ASSETS.mkdir(parents=True,exist_ok=True)
    PDF.parent.mkdir(parents=True,exist_ok=True)
    source = ROOT/'outputs/hsg_indexed/gui42/snapshot_000/inputs/frame_000'
    for suffix in ('rgb.png','depth.npz'):
        asset=ASSETS/f'architecture_oblique_{suffix}'
        if not asset.exists():
            shutil.copy2(source/f'fixed_oblique_{suffix}',asset)
    rgb=np.array(Image.open(ASSETS/'architecture_oblique_rgb.png'))
    with np.load(ASSETS/'architecture_oblique_depth.npz') as archive:
        depth=archive['depth']

    plt.rcParams.update({'font.family':'Arial','font.size':11.5,'svg.fonttype':'none',
                         'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig=plt.figure(figsize=(12,7.05),facecolor='white')
    ax=fig.add_axes([.015,.025,.97,.95]);ax.set_xlim(0,16.5);ax.set_ylim(0,9.65);ax.axis('off')

    def text(x,y,value,size=11.5,color=INK,weight='normal',ha='center',va='center',**kw):
        return ax.text(x,y,value,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,**kw)
    def box(x,y,w,h,face='white',edge='#ccd4db',lw=1.,radius=.09,ls='solid'):
        item=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                            linewidth=lw,edgecolor=edge,facecolor=face,linestyle=ls,zorder=2)
        ax.add_patch(item);return item
    def arrow(points,color=INK,lw=1.1,dashed=False,head=11,z=4):
        p=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        item=FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=head,color=color,
                             linewidth=lw,linestyle='--' if dashed else '-',zorder=z)
        ax.add_patch(item);return item
    def panel(x,w,title,letter,color,tint):
        box(x,2.2,w,5.6,face=tint,edge=color,lw=1.2,radius=.12)
        ax.plot([x+.12,x+w-.12],[7.2,7.2],color=color,lw=.8,zorder=3)
        text(x+.22,7.51,letter,14,color,'bold',ha='left')
        text(x+.63,7.51,title,12.4,color,'bold',ha='left')
    def process(x,y,w,h,title,body,color):
        box(x,y,w,h,edge=color,lw=.85)
        if body:
            text(x+w/2,y+h*.67,title,11.5,INK,'bold')
            text(x+w/2,y+h*.28,body,10.6,MUTED)
        else:
            text(x+w/2,y+h/2,title,11.5,INK,'bold')

    # Perception inputs. Both thumbnails show the same real oblique acquisition.
    panel(.2,2.6,'Workspace','',GRAY,'#f4f6f8')
    text(1.5,6.98,'Two fixed RGB-D cameras',10.7)
    ax.imshow(rgb,extent=(.38,2.62,5.17,6.82),aspect='auto',zorder=3)
    text(1.5,5.00,'RGB (oblique view)',10.3,MUTED)
    # Exclude far-plane background from the display scale, not from A's input.
    valid=depth[np.isfinite(depth)&(depth>0)&(depth<2.)]
    lo,hi=np.percentile(valid,[2,98])
    shown=np.ma.masked_where(~np.isfinite(depth)|(depth<=0)|(depth>=2.),depth)
    cm=plt.get_cmap('cividis').copy();cm.set_bad('#e9eef2')
    ax.imshow(shown,extent=(.38,2.62,3.22,4.87),cmap=cm,vmin=lo,vmax=hi,aspect='auto',zorder=3)
    text(1.5,3.05,'Metric depth (m)',10.3,MUTED)
    text(1.5,2.79,'MuJoCo + SO-101',11.5,INK,'bold')
    text(1.5,2.48,'5 arm joints + gripper',10.0,MUTED)

    panel(3.25,4.,'Perception & HSG','A',BLUE,'#f1f6fa')
    process(3.49,6.35,3.52,.72,'RGB masks + back-projection','Color segmentation; calibrated depth',BLUE)
    process(3.49,5.27,3.52,.77,'3D fusion + box fitting','Two views; configured box-size prior',BLUE)
    process(3.49,4.20,3.52,.77,'Temporal association','Track IDs, geometry and history',BLUE)
    arrow([(5.25,6.35),(5.25,6.08)],BLUE)
    arrow([(5.25,5.27),(5.25,5.01)],BLUE)
    arrow([(5.25,4.20),(5.25,3.94)],BLUE)
    # Hierarchy is schematic, not a claim that B observes hidden objects.
    box(4.60,3.58,1.3,.34,edge=BLUE,lw=.8)
    text(5.25,3.75,'workspace',10.4,BLUE)
    for j,cx in enumerate((4.10,5.25,6.40)):
        ax.plot([5.25,cx],[3.58,3.31],color=BLUE,lw=.7,zorder=3)
        box(cx-.42,2.97,.84,.34,face='#e3edf5',edge=BLUE,lw=.7)
        text(cx,3.14,f'S{j+1}',10.1,BLUE)
        for shift in (-.22,.22):
            ox=cx+shift
            ax.plot([cx,ox],[2.97,2.73],color=BLUE,lw=.7,zorder=3)
            box(ox-.15,2.51,.30,.22,face=('#c4d9e8' if j!=1 else '#7aacc9'),edge=BLUE,lw=.6,radius=.025)
    text(5.25,2.34,'Region hierarchy + current / held tracks',10.3,MUTED)

    panel(7.75,4.,'HSG planning','B',AMBER,'#fbf6ed')
    process(7.99,6.35,3.52,.72,'Target grounding*','Attributes to an episode-local track',AMBER)
    process(7.99,5.27,3.52,.77,'Exact snapshot diff + indices','Spatial hash and support relations',AMBER)
    arrow([(9.75,6.35),(9.75,6.08)],AMBER)
    arrow([(9.75,5.27),(9.75,5.00)],AMBER)
    text(9.75,4.84,'Regional summaries',11.5,INK,'bold')
    for j,cx in enumerate((8.58,9.75,10.92)):
        changed=j==1
        box(cx-.43,4.16,.86,.48,face='#f1d9aa' if changed else 'white',edge=AMBER,lw=1 if changed else .7)
        text(cx,4.41,f'S{j+1}',10.8,AMBER,'bold')
        text(cx,4.05,'update' if changed else 'reuse',9.8,AMBER if changed else MUTED)
    arrow([(9.75,3.91),(9.75,3.72)],AMBER)
    box(7.99,2.49,3.52,1.20,edge=AMBER,lw=.85)
    text(9.75,3.43,'Subgoal selection',11.7,INK,'bold')
    text(9.75,3.09,'Unseen target: rank search regions',10.6,MUTED)
    text(9.75,2.84,'Visible target: resolve blockers',10.6,MUTED)
    text(9.75,2.60,'PICK / PLACE + placement pose',10.4,AMBER)

    panel(12.25,4.,'ACT control','C',TEAL,'#eff7f5')
    box(12.49,6.18,3.52,.89,edge=TEAL,lw=.85)
    text(14.25,6.85,'Observation / goal encoders',11.5,INK,'bold')
    text(14.25,6.55,'Front RGB (CNN) + joint state',10.6,MUTED)
    text(14.25,6.31,'Subgoal to 25D numerical features',10.5,MUTED)
    arrow([(14.25,6.18),(14.25,5.94)],TEAL)
    process(12.49,4.98,3.52,.92,'Goal-conditioned ACT','Transformer encoder-decoder',TEAL)
    arrow([(14.25,4.98),(14.25,4.73)],TEAL)
    # Time runs left to right; each column represents a 6-channel command.
    for col in range(16):
        for row in range(6):
            face='#3f9c8b' if col<4 else '#b8dcd3'
            ax.add_patch(Rectangle((12.58+col*.208,4.08+row*.085),.18,.067,facecolor=face,edgecolor='none',zorder=3))
    text(14.25,3.87,'Predicted chunk: K = 16, 6 channels',10.4,TEAL)
    arrow([(14.25,3.69),(14.25,3.48)],TEAL)
    box(12.49,2.49,3.52,.95,edge=TEAL,lw=.85)
    text(14.25,3.20,'Chunk buffer + command limits',11.1,INK,'bold')
    text(14.25,2.90,'4-step cache; one command per tick',10.2,MUTED)
    text(14.25,2.64,'Measured-state and joint-limit clipping',10.1,MUTED)

    # Typed interfaces. Connections are ports on complete modules.
    arrow([(2.8,6.65),(3.23,6.65)],BLUE,1.3)
    arrow([(7.25,6.65),(7.73,6.65)],INK,1.3)
    text(7.50,6.96,r'$G_t$',12)
    arrow([(11.75,6.65),(12.23,6.65)],INK,1.3)
    text(12.,6.96,r'$g_k$',12)

    box(7.75,8.22,4.,.70,face='white',edge=AMBER,lw=1.)
    text(9.75,8.70,'Task specification',11.8,AMBER,'bold')
    text(9.75,8.42,'Target attributes + destination / buffer',10.3,MUTED)
    arrow([(9.75,8.22),(9.75,7.82)],AMBER,1.1)

    # Per-tick observations for ACT use a distinct front RGB camera.
    arrow([(1.5,7.82),(1.5,9.23),(14.25,9.23),(14.25,7.82)],GRAY,1.0)
    text(5.0,9.42,'Front RGB + measured joint state '+r'$q_t$'+' (each control tick)',10.6,GRAY,ha='left')

    # Runner owns execution, completion tests, replanning and contract checks.
    box(3.25,.47,13.,.91,face='#f3f5f7',edge='#8795a0',lw=.95)
    text(3.52,1.11,'CLOSED-LOOP ORCHESTRATOR',11.7,INK,'bold',ha='left')
    text(9.75,.74,'Contract checks  |  execute  |  verify subgoal  |  re-observe and replan',10.9,MUTED)
    arrow([(14.25,2.18),(14.25,1.40)],TEAL,1.25)
    text(14.50,1.81,r'$u_t$'+'  (30 Hz)',10.6,TEAL,ha='left')
    arrow([(3.25,1.09),(2.43,1.09),(2.43,2.18)],INK,1.1)
    text(2.20,1.65,'execute',10.1,INK)
    arrow([(.70,2.18),(.70,.73),(3.23,.73)],GRAY,1.0,dashed=True)
    text(1.80,.47,'FK + tactile feedback',9.8,GRAY)
    arrow([(4.30,1.40),(4.30,2.18)],BLUE,1.0,dashed=True)
    text(4.55,1.80,'refresh',9.8,BLUE,ha='left')
    arrow([(9.75,1.40),(9.75,2.18)],AMBER,1.0,dashed=True)
    text(10.0,1.80,'replan',9.8,AMBER,ha='left')
    text(8.25,.15,'Shown: RGB-D + indexed HSG + ACT.  * Target grounding runs in the orchestrator.',9.7,MUTED)

    # One source for editable SVG, vector PDF, and high-resolution preview.
    fig.savefig(OUT/'system_architecture.svg',facecolor='white')
    fig.savefig(PDF,facecolor='white',metadata={'Title':'Closed-loop HSG manipulation architecture','Author':'HSSG manipulation project'})
    fig.savefig(OUT/'system_architecture.png',dpi=400,facecolor='white')
    plt.close(fig)
    print('SVG:',OUT/'system_architecture.svg')
    print('PDF:',PDF)
    print('PNG:',OUT/'system_architecture.png')


if __name__=='__main__': main()
