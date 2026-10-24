"""Validate completed stage-3 runs, analyze seeds and export selected small artifacts."""
import itertools
import json
from pathlib import Path
import shutil
import statistics

from analyze_fusion_routing import load_result, compare
from routing_data import digest, json_write, video_key
from train_fusion_routing_v3 import code_identity
from analyze_routing_stage3_results import contribution


def main():
    root=Path(__file__).resolve().parent
    output=root/'research_exports/ucf101_routing_stage3_2026-10-24'
    if output.exists():raise FileExistsError(output)
    results={};runs={};sources={}
    for name in ('G1','G2','G3','G4'):
        for seed in ((0,1,2) if name in ('G1','G2') else (0,)):
            key=f'{name}_seed{seed}';folder=root/f'checkpoints/routing_stage3_{key}'
            cfg=json.loads((folder/'config.json').read_text())
            if cfg['code']!=code_identity():raise ValueError('Source differs from executed model')
            status=json.loads((folder/'status.json').read_text())
            rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            if status['status']!='complete' or len(rows)!=80 or status['optimizer_steps']!=25760:
                raise ValueError('Incomplete experiment')
            best=min(rows,key=lambda r:r['validation']['video_nll'])
            result=load_result(folder/'best')
            if result['metadata']['checkpoint_sha256']!=digest(folder/'best_val_nll.pt'):
                raise ValueError('Checkpoint and prediction mismatch')
            eval_path=root/f'reports/routing_stage3/{name+"_seed"+str(seed) if seed else name}_eval'
            replay=load_result(eval_path)
            check=compare(result,replay)
            if check['counts']['prediction_changed'] or abs(check['metrics_right_minus_left']['video_nll'])>2e-6:
                raise ValueError('Checkpoint evaluation failed reproduction')
            results[key]=result
            runs[key]=dict(epoch=best['epoch'],metrics=best['validation'],last_metrics=rows[-1]['validation'],
                gate=best['validation_diagnostics'],last_gate=rows[-1]['validation_diagnostics'],
                gradient_norms=best['gradient_norms'],config=cfg,folder=str(folder))
    # Ensure corresponding seeds differ only in the intended eta architecture.
    for name in ('G1','G2'):
        base=runs[f'{name}_seed0']['config']
        for seed in (1,2):
            c=runs[f'{name}_seed{seed}']['config']
            if {k:v for k,v in c.items() if k!='seed'}!={k:v for k,v in base.items() if k!='seed'}:
                raise ValueError('Uncontrolled seed configuration difference')
    baselines={name:load_result(root/f'reports/routing_stage2_full_v1/{name}__normal__prefix_1')
               for name in ('fusion','frame','fusion_frame','current')}
    baselines['static_frame']=load_result(root/'reports/routing_stage2_static_frame_comparison')
    pairs={};aggregate={};stability={}
    def compact_pair(a,b):
        pair=compare(a,b);pair['nll_attribution']=contribution(pair)
        return {k:v for k,v in pair.items() if k not in ('videos','right_window_diagnostics')}
    for seed in (0,1,2):
        pairs[f'G1_vs_G2_seed{seed}']=compact_pair(results[f'G1_seed{seed}'],results[f'G2_seed{seed}'])
        for name,base in baselines.items():
            pairs[f'{name}_vs_G2_seed{seed}']=compact_pair(base,results[f'G2_seed{seed}'])
    for name in ('G1','G2'):
        keys=[f'{name}_seed{s}' for s in (0,1,2)]
        aggregate[name]={m:dict(mean=statistics.mean([runs[k]['metrics'][m] for k in keys]),
            sample_std=statistics.stdev([runs[k]['metrics'][m] for k in keys]))
            for m in ('video_top1','video_nll','video_top5','video_brier','video_ece')}
        errors=[];rescues=[]
        for key in keys:
            rows=results[key]['index']
            errors.append({video_key(r) for r in rows if r['prediction']!=r['label_id']})
            rescues.append({video_key(r) for r in rows if r['prediction']==r['label_id'] and r['fusion_prediction']!=r['label_id']})
        stability[name]=dict(error_intersection=len(set.intersection(*errors)),error_union=len(set.union(*errors)),
            rescue_intersection=len(set.intersection(*rescues)),rescue_union=len(set.union(*rescues)),pairwise={})
        for a,b in itertools.combinations(range(3),2):
            stability[name]['pairwise'][f'{a}_vs_{b}']=compact_pair(results[keys[a]],results[keys[b]])['counts']
    output.mkdir(parents=True)
    def copy(source,relative):
        source=Path(source);target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        sources[relative]=dict(source=str(source),sha256=digest(source),included=True)
    for key,run in runs.items():
        folder=Path(run['folder'])
        for name in ('config.json','metrics.jsonl','status.json','epoch0.json'):
            copy(folder/name,f'{key}/{name}')
        copy(folder/'best/metrics.json',f'{key}/best_metrics.json')
        # Full identity/prediction/NLL retained; verbose window arrays stay local.
        predictions=[{k:v for k,v in r.items() if k!='window_index'} for r in results[key]['index']]
        json_write(output/f'{key}/video_predictions_best.json',predictions)
        sources[f'{key}/video_predictions_best.json']=dict(source=str(folder/'best/video_index.json'),
            source_sha256=digest(folder/'best/video_index.json'),transformation='remove window_index only',included=True)
        for name in ('best_val_nll.pt','last.pt'):
            sources[f'{key}/{name}']=dict(source=str(folder/name),sha256=digest(folder/name),included=False)
    full=root/'reports/routing_stage3/full_analysis_v1'
    for name in ('report.md','curves.png','training_summary.json'):
        copy(full/name,f'seed0_analysis/{name}')
    copy(root/'reports/routing_stage3/train_baselines_v1/analysis.md','train_baselines.md')
    copy(root/'reports/routing_stage3/audit_v1/audit.json','input_audit.json')
    copy(root/'reports/routing_stage3/cpu_tests.log','cpu_tests.log')
    summary=dict(selection_protocol='original_test_as_val',independent_test=False,
        variance_scope='Fixed experts and unrestricted Static; Gate initialization and training order vary',
        aggregate=aggregate,runs={k:{n:v for n,v in r.items() if n not in ('config','folder')} for k,r in runs.items()},
        stability=stability)
    json_write(output/'summary.json',summary);json_write(output/'paired_summary.json',pairs)
    json_write(output/'sources.json',sources)
    lines=['# 第三阶段结果：外层Gate与视频监督','',
        'G1–G4 seed0及G1/G2 seed1/2共8次训练均完成80轮、25760步；选中权重均复评通过。',
        '固定专家和原Static；原Test视频NLL选模，非独立测试。标准差为3个种子的样本标准差。','',
        '| 方法 | Top1均值±标准差 (%) | NLL均值±标准差 |','|---|---:|---:|']
    for name,a in aggregate.items():
        t,n=a['video_top1'],a['video_nll'];lines.append(f"| {name} | {100*t['mean']:.4f} ± {100*t['sample_std']:.4f} | {n['mean']:.6f} ± {n['sample_std']:.6f} |")
    lines+=['','## 对应种子的直接比较','',
        '| seed | G1最优epoch | G2最优epoch | G2纠正G1 | G2损失G1 | 净增加正确 | ΔNLL (G2−G1) |','|---|---:|---:|---:|---:|---:|---:|']
    for seed in (0,1,2):
        p=pairs[f'G1_vs_G2_seed{seed}'];c=p['counts']
        lines.append(f"| {seed} | {runs[f'G1_seed{seed}']['epoch']} | {runs[f'G2_seed{seed}']['epoch']} | {c['right_only_correct']} | {c['left_only_correct']} | {c['right_only_correct']-c['left_only_correct']} | {p['metrics_right_minus_left']['video_nll']:.6f} |")
    lines+=['','## G2相对各基线的纠错/损失','', '| 基线 | seed0 | seed1 | seed2 |','|---|---:|---:|---:|']
    for name in baselines:
        cells=[]
        for seed in (0,1,2):
            c=pairs[f'{name}_vs_G2_seed{seed}']['counts'];cells.append(f"{c['right_only_correct']} / {c['left_only_correct']}")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','## 样本稳定性','']
    lines.append('上述current基线固定为第二阶段Current seed0，并非逐种子替换；G1/G2直接比较使用对应种子。')
    for name,s in stability.items():
        lines.append(f"- {name}：三种子错误交集/并集={s['error_intersection']}/{s['error_union']}；相对Fusion纠错交集/并集={s['rescue_intersection']}/{s['rescue_union']}。")
    lines+=['','## 结论与边界','',
        '- 三个对应种子中G2的Top1和NLL均优于G1；支持动态eta相对全局比例的优势，但不宣称统计显著。',
        '- G2准确率仍低于Frame的97.1037%和固定Fusion+Frame的97.0254%；NLL更好，是取舍而非全面超越。',
        '- G2最优epoch为9/6/7，后期饱和和选模集退化需关注，不能从早期选模推断更长训练必有收益。',
        '- G3/G4视频监督在seed0没有改善NLL；尚无多种子验证，不能把差异全归于聚合次序（视频权重也改变）。',
        '- B1/B2、V3 Memory和前缀实验未执行。本轮结束，不自动增加实验。','',
        '完整80轮指标、实际配置、epoch0与选中预测保留；权重只列路径/SHA，不上传。',
        'sources.json记录来源和变换；seed0_analysis含完整首轮分析与曲线。完整概率、窗口索引及轨迹留在原服务器。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(output,flush=True)


if __name__=='__main__':main()
