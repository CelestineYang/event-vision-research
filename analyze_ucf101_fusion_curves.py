"""Read-only experiment comparison; write standalone plots and snapshot report."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path):
    raw=path.read_text()
    # Ignore only an incomplete trailing write from the active trainer.
    lines=raw.splitlines()
    rows=[]
    for i,line in enumerate(lines):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not raw.endswith('\n'): break
            raise
    epochs=[r['epoch'] for r in rows]
    if epochs!=list(range(1,len(rows)+1)):
        raise ValueError(f'Unexpected epoch sequence in {path}')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion',type=Path,default=Path('experiments/ucf101_filename_v2_fusion_A_seed0_v1'))
    parser.add_argument('--images',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2'))
    parser.add_argument('--through-epoch',type=int)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    fusion=read_metrics(args.fusion/'metrics.jsonl')
    if args.through_epoch is not None: fusion=[r for r in fusion if r['epoch']<=args.through_epoch]
    images={v:read_metrics(args.images/'logs'/f'{v}_metrics.jsonl') for v in ('cstr3','xt_p','ty_p')}
    end=fusion[-1]['epoch']
    out=args.output or Path('reports')/f'ucf101_filename_v2_fusion_epoch{end:03d}'
    out.mkdir(parents=True,exist_ok=True)
    for r in fusion:
        if (r['test']['fusion']['videos'],r['test']['fusion']['windows'])!=(2555,27384):
            raise ValueError('Fusion evaluation population changed')
    for rs in images.values():
        if any((r['test']['videos'],r['test']['windows'])!=(2555,27384) for r in rs):
            raise ValueError('Image evaluation population differs')
    best_nll=min(fusion,key=lambda r:(r['test']['fusion']['video_nll'],-r['test']['fusion']['video_top1'],r['epoch']))
    best_acc=max(fusion,key=lambda r:(r['test']['fusion']['video_top1'],-r['test']['fusion']['video_nll'],-r['epoch']))
    indexed={r['epoch']:r for r in fusion}
    matched=[]
    comparisons={}
    for view,rs in images.items():
        pairs=[(indexed[r['epoch']],r) for r in rs if r['epoch'] in indexed]
        deltas=[]
        for f,r in pairs:
            ft,it=f['test']['fusion'],r['test']
            delta=100*(ft['video_top1']-it['video_top1'])
            deltas.append(delta)
            matched.append(dict(epoch=f['epoch'],view=view,fusion_top1_pct=100*ft['video_top1'],
                image_top1_pct=100*it['video_top1'],delta_top1_pp=delta,
                fusion_nll=ft['video_nll'],image_nll=it['video_nll']))
        selected=min(rs,key=lambda r:(r['test']['video_nll'],-r['test']['video_top1'],r['epoch']))
        comparisons[view]=dict(common_epochs=len(pairs),
            fusion_top1_higher=sum(f['test']['fusion']['video_top1']>r['test']['video_top1'] for f,r in pairs),
            fusion_nll_lower=sum(f['test']['fusion']['video_nll']<r['test']['video_nll'] for f,r in pairs),
            top1_delta_pp_min=min(deltas),top1_delta_pp_max=max(deltas),
            image_selected_epoch=selected['epoch'],image_selected_metrics=selected['test'])
    with (out/'matched_epochs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(matched[0]));writer.writeheader();writer.writerows(matched)
    recent=fusion[-10:]
    report=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),completed_epochs=end,
        fusion_source=str(args.fusion/'metrics.jsonl'),image_sources={v:str(args.images/'logs'/f'{v}_metrics.jsonl') for v in images},
        latest=fusion[-1],selected_by_nll=best_nll,highest_top1=best_acc,comparisons=comparisons,
        recent_mean=dict(video_top1=float(np.mean([r['test']['fusion']['video_top1'] for r in recent])),
                         video_nll=float(np.mean([r['test']['fusion']['video_nll'] for r in recent]))),
        limitations=['Test is used for selection, not independent held-out evaluation',
            'Fusion starts from trained/selected image and point weights; image pretraining starts from ImageNet',
            'Fusion global batch16 versus image batch64: same epoch means 4x as many optimizer steps',
            'Comparison is versus individual image branches, not an independently trained three-image fusion/ensemble',
            'Same video list may contain related groups across Train and Test',
            'No inference, weight updates or training changes performed by this analysis'])
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(13.8,9.2),constrained_layout=True)
    colors={'cstr3':'#2274A5','xt_p':'#2A9D8F','ty_p':'#E9A23B'}
    epochs=[r['epoch'] for r in fusion]
    a,b,c,d=axes.flat
    a.plot(epochs,[100*r['test']['fusion']['video_top1'] for r in fusion],color='#c03957',lw=2,label='Fusion A (V2 + 3 images)')
    b.plot(epochs,[r['test']['fusion']['video_nll'] for r in fusion],color='#c03957',lw=2,label='Fusion A')
    for v,rs in images.items():
        a.plot([r['epoch'] for r in rs],[100*r['test']['video_top1'] for r in rs],label=v,color=colors[v],alpha=.85,lw=1.2)
        b.plot([r['epoch'] for r in rs],[r['test']['video_nll'] for r in rs],label=v,color=colors[v],alpha=.85,lw=1.2)
        paired=[r for r in matched if r['view']==v]
        d.plot([r['epoch'] for r in paired],[r['delta_top1_pp'] for r in paired],color=colors[v],label='Fusion minus '+v)
    a.scatter([best_acc['epoch']],[100*best_acc['test']['fusion']['video_top1']],color='#c03957',s=45,zorder=5)
    a.annotate(f"Peak: {100*best_acc['test']['fusion']['video_top1']:.2f}% (e{best_acc['epoch']})",
        (best_acc['epoch'],100*best_acc['test']['fusion']['video_top1']),xytext=(8,-25),textcoords='offset points',fontsize=9)
    b.scatter([best_nll['epoch']],[best_nll['test']['fusion']['video_nll']],color='#c03957',s=45,zorder=5)
    b.annotate(f"Selected: {best_nll['test']['fusion']['video_nll']:.3f} (e{best_nll['epoch']})",
        (best_nll['epoch'],best_nll['test']['fusion']['video_nll']),xytext=(10,15),textcoords='offset points',fontsize=9)
    c.plot(epochs,[r['train']['fusion']['window_loss'] for r in fusion],color='#2274A5',label='Train window CE')
    c.plot(epochs,[r['test']['fusion']['window_loss'] for r in fusion],color='#c03957',label='Test window CE')
    c.set_yscale('log')
    d.axhline(0,color='#444444',lw=1)
    titles=['Video Test Top-1: fusion vs individual image branches',
            'Video Test NLL (lower is better)',
            'Fusion generalization gap (CE only; log scale)',
            'Matched-epoch Test Top-1 improvement']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontweight='bold',fontsize=11)
        ax.set_xlabel('Epoch');ax.legend(fontsize=8,loc='best')
    a.set_ylabel('Top-1 (%)');b.set_ylabel('Mean video NLL')
    c.set_ylabel('Mean window cross-entropy');d.set_ylabel('Percentage points')
    fig.suptitle(f'UCF101-DVS: completed through fusion epoch {end} | 2,555 Test videos / 27,384 windows\n'
                 'Soft voting; Test-selected models. Fusion is warm-started; equal epochs are not equal training budgets.',fontsize=12)
    fig.savefig(out/'curves.png',dpi=180)
    fig.savefig(out/'curves.pdf')
    plt.close(fig)
    table=[]
    for e in (1,5,6,10,20,30,40,50,end):
        if e not in indexed or e in [r['epoch'] for r in table]: continue
        r=indexed[e]['test']['fusion']
        row=dict(epoch=e,fusion_top1_pct=100*r['video_top1'],fusion_nll=r['video_nll'])
        for v,rs in images.items():
            old=next((x for x in rs if x['epoch']==e),None)
            row[v+'_top1_pct']=100*old['test']['video_top1'] if old else None
        table.append(row)
    (out/'key_epochs.json').write_text(json.dumps(table,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),completed_epochs=end,best_nll_epoch=best_nll['epoch'],
                         highest_top1_epoch=best_acc['epoch'],comparisons=comparisons),indent=2))


if __name__=='__main__': main()
