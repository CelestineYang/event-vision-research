"""Offline completed-stage-3 analysis; no fitting and no changes to V3 hashes."""
import csv
import json
from pathlib import Path

import numpy as np

from analyze_fusion_routing import compare, load_result
from evaluate_fusion_routing_v3 import paired_tables
from routing_data import digest, json_write, video_key


def contribution(pair):
    groups={k:[] for k in ('both_correct','both_wrong','baseline_only_correct','new_only_correct')}
    for row in pair['videos']:
        a=row['left_prediction']==row['label_id'];b=row['right_prediction']==row['label_id']
        group='both_correct' if a and b else 'baseline_only_correct' if a else 'new_only_correct' if b else 'both_wrong'
        groups[group].append(row['nll_right_minus_left'])
    return {k:dict(count=len(v),sum_delta=float(sum(v)),mean_contribution=float(sum(v)/len(pair['videos']))) for k,v in groups.items()}


def main():
    output=Path('reports/routing_stage3/full_analysis_v1')
    if output.exists():raise FileExistsError(output)
    paths=[f'reports/routing_stage3/{name}_eval' for name in ('G1','G2','G3','G4')]
    paired_tables(paths,'reports/routing_stage2_full_v1','reports/routing_stage2_static_frame_comparison',output)
    models={name:load_result(path) for name,path in zip(('G1','G2','G3','G4'),paths)}
    bases={name:load_result(Path('reports/routing_stage2_full_v1')/f'{name}__normal__prefix_1') for name in ('fusion','frame','fusion_frame','current')}
    bases['static_frame']=load_result('reports/routing_stage2_static_frame_comparison')
    pairs={}
    for a,b in [('G1','G2'),('G1','G3'),('G2','G4'),('G3','G4')]:
        p=compare(models[a],models[b]);p['nll_attribution']=contribution(p);pairs[a+'_vs_'+b]=p
    json_write(output/'factorial_pairs.json',pairs)
    histories={};curves={};sources={}
    for name in models:
        root=Path(f'checkpoints/routing_stage3_{name}_seed0')
        status=json.loads((root/'status.json').read_text());assert status['status']=='complete'
        rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
        assert len(rows)==80 and [r['epoch'] for r in rows]==list(range(1,81))
        best=min(rows,key=lambda r:r['validation']['video_nll']);last=rows[-1]
        np.testing.assert_allclose(best['validation']['video_nll'],models[name]['metadata']['metrics']['video_nll'],atol=2e-6)
        assert digest(root/'best_val_nll.pt')==models[name]['metadata']['checkpoint_sha256']
        histories[name]=dict(best=best,last=last,epoch0=json.loads((root/'epoch0.json').read_text()))
        curves[name]=rows
        sources[name]={f:dict(path=str(root/f),sha256=digest(root/f)) for f in ('config.json','metrics.jsonl','status.json','best_val_nll.pt','last.pt')}
    json_write(output/'training_summary.json',histories);json_write(output/'sources.json',sources)
    names={v:k for k,v in models['G2']['metadata']['class_to_idx'].items()}
    g2pairs={};all_changes=[]
    for name,baseline in bases.items():
        pair=compare(baseline,models['G2']);pair['nll_attribution']=contribution(pair)
        old={video_key(r):(r,lp) for r,lp in zip(baseline['index'],baseline['log_probs'])}
        new={video_key(r):(r,lp) for r,lp in zip(models['G2']['index'],models['G2']['log_probs'])}
        for row in pair['videos']:
            key=tuple(row['identity']);a,ap=old[key];b,bp=new[key];label=row['label_id']
            row.update(baseline=name,class_name=names[label],baseline_prediction=names[a['prediction']],
                g2_prediction=names[b['prediction']],baseline_nll=float(-ap[label]),g2_nll=float(-bp[label]),
                baseline_true_probability=float(np.exp(ap[label])),g2_true_probability=float(np.exp(bp[label])))
            if row['left_prediction']!=row['right_prediction']:all_changes.append(row)
        pair['largest_nll_improvements']=sorted(pair['videos'],key=lambda r:r['nll_right_minus_left'])[:10]
        pair['largest_nll_regressions']=sorted(pair['videos'],key=lambda r:r['nll_right_minus_left'],reverse=True)[:10]
        delta=np.array([r['nll_right_minus_left'] for r in pair['videos']])
        pair['nll_comparison_counts']=dict(g2_lower=int((delta < -1e-8).sum()),baseline_lower=int((delta>1e-8).sum()),tie=int((abs(delta)<=1e-8).sum()))
        g2pairs[name]=pair
    json_write(output/'g2_pairs.json',g2pairs)
    with (output/'g2_prediction_changes.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_changes[0]));writer.writeheader();writer.writerows(all_changes)
    # Trace diagnostics are descriptive only, never fed back as model inputs.
    traces={}
    with Path('reports/routing_stage3/G2_eval/window_traces.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);traces.setdefault(video_key(row),[]).append(row)
    within=[];video_stats={}
    for key,rows in traces.items():
        eta=np.array([r['eta'] for r in rows]);within.append(float(eta.std()))
        video_stats[str(key)]=dict(windows=len(rows),eta_mean=float(eta.mean()),eta_std=float(eta.std()),
            eta_min=float(eta.min()),eta_max=float(eta.max()))
    multi=[v['eta_std'] for v in video_stats.values() if v['windows']>1]
    trace_summary=dict(videos=len(traces),multi_window_videos=len(multi),
        mean_within_video_eta_std=float(np.mean(within)),mean_multi_window_eta_std=float(np.mean(multi)),
        fraction_multi_window_eta_std_above_001=float(np.mean(np.array(multi)>.01)))
    json_write(output/'g2_trace_summary.json',dict(summary=trace_summary,videos=video_stats))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for name,rows in curves.items():
        xs=[r['epoch'] for r in rows]
        for axis,values in zip(axes.flat,(
                [r['validation']['video_nll'] for r in rows],
                [r['validation']['video_top1']*100 for r in rows],
                [r['train']['objective_nll'] for r in rows],
                [r['validation_diagnostics']['eta']['mean'] for r in rows],
                [r['validation_diagnostics']['eta']['std'] for r in rows],
                [r['validation_diagnostics']['delta_eta']['std'] for r in rows])):
            axis.plot(xs,values,label=name)
    for ax,title in zip(axes.flat,('Selection video NLL','Selection Top1 (%)','Train objective (different definitions)',
                                   'Selection eta mean','Selection eta std','Selection delta_eta std')):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(output/'curves.png',dpi=150);plt.close(fig)
    lines=['# 第三阶段完整分析','',
        '四项均完成80轮；模型均按原Test视频NLL选中。单种子、非独立测试；以下为描述性比较，不作因果或显著性结论。','',
        '## 统一结果','', '| 方法 | 选中epoch | Top1 (%) | NLL | Brier | ECE |','|---|---:|---:|---:|---:|---:|']
    for name,result in {**bases,**models}.items():
        m=result['metadata']['metrics'];epoch=histories[name]['best']['epoch'] if name in histories else '—'
        lines.append(f"| {name} | {epoch} | {100*m['video_top1']:.4f} | {m['video_nll']:.6f} | {m['video_brier']:.6f} | {m['video_ece']:.6f} |")
    lines+=['','## G2配对：纠错与新增错误','', '| 基线 | 基线错/G2对 | 基线对/G2错 | 净增加正确 | ΔNLL (G2−基线) |','|---|---:|---:|---:|---:|']
    for name,p in g2pairs.items():
        c=p['counts'];lines.append(f"| {name} | {c['right_only_correct']} | {c['left_only_correct']} | {c['right_only_correct']-c['left_only_correct']} | {p['metrics_right_minus_left']['video_nll']:.6f} |")
    lines+=['','## 2×2直接比较','', '| 比较（左→右） | 右纠正 | 右损失 | ΔNLL |','|---|---:|---:|---:|']
    for name,p in pairs.items():
        c=p['counts'];lines.append(f"| {name} | {c['right_only_correct']} | {c['left_only_correct']} | {p['metrics_right_minus_left']['video_nll']:.6f} |")
    lines+=['','## 训练后期与Gate','', '| 模型 | 最优→末轮 NLL | 最优→末轮 eta均值 | 最优 eta标准差 |','|---|---:|---:|---:|']
    for name,h in histories.items():
        a,b=h['best'],h['last'];ad=a['validation_diagnostics'];bd=b['validation_diagnostics']
        lines.append(f"| {name} | {a['validation']['video_nll']:.6f} → {b['validation']['video_nll']:.6f} | {ad['eta']['mean']:.6f} → {bd['eta']['mean']:.6f} | {ad['eta']['std']:.6f} |")
    lines+=['','## G2 NLL贡献（按对错分组）','', '| 基线 | 组 | 视频数 | 对平均ΔNLL的贡献 |','|---|---|---:|---:|']
    for name,p in g2pairs.items():
        for group,g in p['nll_attribution'].items():lines.append(f"| {name} | {group} | {g['count']} | {g['mean_contribution']:.6f} |")
    lines+=['','## G2逐窗口变化','',json.dumps(trace_summary,ensure_ascii=False), '',
        'eta标准差只证明存在输入依赖，不自动证明选择最优；全窗口均值按窗口加权。显式路径权重不是因果模态贡献。','',
        '完整差异与极端NLL样本见g2_pairs.json，变更预测见g2_prediction_changes.csv，训练曲线见curves.png。',
        '本分析没有启动B1/B2、多种子或Memory实验。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(output,flush=True)


if __name__=='__main__':main()
