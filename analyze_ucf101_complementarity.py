"""CPU-only error overlap and fixed-rule diagnostics from saved predictions.

Oracle selection uses ground truth and is NOT a deployable accuracy estimate.
No inference, fitted gate, threshold search or training modification occurs.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path


VIEWS=('cstr3','xt_p','ty_p')
ERROR_FIELDS = ('video_id','label_id','fusion_prediction','fusion_true_label_in_top5',
                'experts_correct','point_prediction','cstr3_prediction','xt_p_prediction','ty_p_prediction')


def metric_change(before, after, *, higher_is_better):
    delta=after-before
    if math.isclose(delta,0.,abs_tol=1e-12): direction='unchanged'
    else: direction='improved' if (delta>0)==higher_is_better else 'worsened'
    return f'{direction}: {before:.6f} -> {after:.6f} (delta {delta:+.6f})'


def analysis_text(result):
    a=result['results']['nll_selected']
    rule=a['fixed_label_free_rules']['image_unanimous_cstr3_else_fusion']
    lines=['# Existing-output complementarity diagnostic','',
        'Offline analysis of saved video predictions; no fitting or additional inference.', '',
        '| Checkpoint role | Epoch from metadata | Video Top1 | Errors | Recoverable errors | Oracle Top1 |',
        '|---|---:|---:|---:|---:|---:|']
    for tag,r in result['results'].items():
        epoch=r.get('epoch') if r.get('epoch') is not None else 'unknown'
        lines.append(f"| {tag} | {epoch} | {100*r['models']['fusion']['top1']:.2f}% | {r['base_errors']} | {r['recoverable_errors']} | {100*r['oracles']['all_experts']['top1']:.2f}% |")
    unique=a['partition']['point_only']
    fraction=unique/a['base_errors'] if a['base_errors'] else 0.
    lines += ['', f"Point-exclusive rescue after all image heads: {unique}/{a['base_errors']} fusion errors ({100*fraction:.2f}%); {100*unique/a['videos']:.2f}% of all videos.",
        'When there are zero fusion errors, the error-fraction above is defined as zero.', '',
        'Fixed image-unanimity rule Top1 '+metric_change(a['models']['fusion']['top1'],rule['top1'],higher_is_better=True)+'.',
        'Fixed image-unanimity rule NLL '+metric_change(a['models']['fusion']['video_nll'],rule['video_nll'],higher_is_better=False)+'.', '',
        'Checkpoint metadata: '+json.dumps(result.get('checkpoint_metadata', {'status':'unknown'}),sort_keys=True)+'.', '',
        'Oracle selection uses ground truth and is not deployable Gate performance. Fixed rules are posthoc exploratory comparisons.',
        'Point is the co-trained head; video-level CSV cannot replace full per-window logits/statistics for Gate/Memory.',
        'Current protocol: original Train fits Gate and normalization; original Test evaluates and selects checkpoints by video NLL; independent_test=false.',
        'Expert fit/selection exposure remains recorded. This analysis does not introduce OOF exports or expert retraining.',
        'Labels, true-class probability and NLL must never enter Gate observations.', '']
    return '\n'.join(lines)


def predictions(path):
    data=path.read_bytes()
    rows=list(csv.DictReader(io.StringIO(data.decode())))
    mapped={r['video_id']:r for r in rows}
    if not rows or len(mapped)!=len(rows): raise ValueError('Empty/duplicate prediction videos')
    for row in rows:
        for key in ('label_id','prediction','windows'): row[key]=int(row[key])
        row['top5']=json.loads(row['top5'])
        row['nll']=float(row['nll'])
    return mapped,dict(source=str(path),sha256=hashlib.sha256(data).hexdigest())


def validate(records,expected):
    n=len(records)
    acc=sum(r['label_id']==r['prediction'] for r in records.values())/n
    nll=sum(r['nll'] for r in records.values())/n
    if n!=expected['videos'] or sum(r['windows'] for r in records.values())!=expected['windows']:
        raise ValueError('CSV/metric population mismatch; retry after epoch save completes')
    if not math.isclose(acc,expected['video_top1'],abs_tol=1e-12):
        raise ValueError('CSV/metric Top-1 mismatch; retry after epoch save completes')
    if not math.isclose(nll,expected['video_nll'],abs_tol=1e-10):
        raise ValueError('CSV/metric NLL mismatch; retry after epoch save completes')


def analyze(bank,base):
    ids=set(bank[base])
    if any(set(rs)!=ids for rs in bank.values()): raise ValueError('Different video populations')
    for vid in ids:
        if len({(rs[vid]['label_id'],rs[vid]['windows']) for rs in bank.values()})!=1:
            raise ValueError('Label/window identity mismatch')
    n=len(ids)
    correct={k:{v for v,r in rs.items() if r['label_id']==r['prediction']} for k,rs in bank.items()}
    top5={k:{v for v,r in rs.items() if r['label_id'] in r['top5']} for k,rs in bank.items()}
    errors=ids-correct[base]
    image_union=set().union(*(correct[v] for v in VIEWS))
    others=set().union(*(s for k,s in correct.items() if k!=base))
    point=correct['point']
    oracles={}
    for name,keys in dict(images=list(VIEWS),fusion_point=[base,'point'],
                          fusion_images=[base,*VIEWS],all_experts=list(bank)).items():
        union=set().union(*(correct[k] for k in keys))
        oracles[name]=dict(correct=len(union),top1=len(union)/n,all_wrong=n-len(union),
                           gain_over_base_pp=100*(len(union)-len(correct[base]))/n,
                           ground_truth_dependent=True)
    rules={}
    # Fixed, label-free selectors. No fitted weights or Test threshold tuning.
    # Return the chosen expert's full distribution, not just its top-1 class.
    for name in ('image_majority_first_matching_expert_cstr3_tie','image_unanimous_cstr3_else_fusion'):
        count=rescued=damaged=changed=0
        nll=0.
        for vid in ids:
            image_preds=[bank[v][vid]['prediction'] for v in VIEWS]
            if name.startswith('image_majority'):
                target,votes=Counter(image_preds).most_common(1)[0]
                chosen=next(v for v in VIEWS if bank[v][vid]['prediction']==target) if votes>=2 else 'cstr3'
            else:
                chosen='cstr3' if len(set(image_preds))==1 else base
            result=bank[chosen][vid]
            label=result['label_id']
            before=bank[base][vid]['prediction']
            after=result['prediction']
            count+=after==label
            rescued+=after==label and before!=label
            damaged+=after!=label and before==label
            changed+=after!=before
            nll+=result['nll']
        rules[name]=dict(correct=count,top1=count/n,video_nll=nll/n,rescued=rescued,
                        damaged=damaged,changed=changed,exploratory_posthoc=True)
    partition=dict(image_only=len((errors&image_union)-point),
                   point_only=len((errors&point)-image_union),
                   both=len(errors&image_union&point),neither=len(errors-image_union-point))
    result=dict(videos=n,models={k:dict(correct=len(s),errors=n-len(s),top1=len(s)/n,
                    top5=len(top5[k])/n,video_nll=sum(r['nll'] for r in bank[k].values())/n)
                    for k,s in correct.items()},
                base_errors=len(errors),recoverable_errors=len(errors&others),partition=partition,
                base_error_in_own_top5=len(errors&top5[base]),
                fusion_correct_all_image_heads_wrong=len(correct[base]-image_union),
                oracles=oracles,fixed_label_free_rules=rules,
                individual_rescue={k:dict(rescued=len(errors&s),
                    harmed_if_used_everywhere=len(correct[base]-s)) for k,s in correct.items() if k!=base})
    cases=[]
    for vid in sorted(errors):
        row=dict(video_id=vid,label_id=bank[base][vid]['label_id'],fusion_prediction=bank[base][vid]['prediction'],
                 fusion_true_label_in_top5=vid in top5[base],
                 experts_correct=';'.join(k for k,s in correct.items() if k!=base and vid in s))
        for k in ('point',*VIEWS): row[k+'_prediction']=bank[k][vid]['prediction']
        cases.append(row)
    return result,cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion',type=Path,default=Path('experiments/ucf101_filename_v2_fusion_A_seed0_v1'))
    parser.add_argument('--images',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    metrics=[]
    metric_path=args.fusion/'metrics.jsonl'
    for line in (metric_path.read_bytes() if metric_path.exists() else b'').splitlines(keepends=True):
        if line.endswith(b'\n') and line.strip(): metrics.append(json.loads(line))
    latest=metrics[-1] if metrics else None
    selected=min(metrics,key=lambda r:(r['test']['fusion']['video_nll'],-r['test']['fusion']['video_top1'],r['epoch'])) if metrics else None
    image_bank={};sources={}
    for v in VIEWS:
        image_bank[v],sources[v]=predictions(args.images/'reports'/f'{v}_test_video_predictions_best_loss.csv')
        path=args.images/'logs'/f'{v}_metrics.jsonl'
        rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
        best=min(rows,key=lambda r:(r['test']['video_nll'],-r['test']['video_top1'],r['epoch'])) if rows else None
        if best: validate(image_bank[v],best['test'])
        sources[v]['selected_epoch']=best['epoch'] if best else None
    results={}
    case_rows={}
    for tag,row,suffix in [('nll_selected',selected,'best_loss'),('latest',latest,'last')]:
        bank=dict(image_bank)
        for branch in ('fusion','point'):
            bank[branch],sources[tag+'_'+branch]=predictions(args.fusion/f'{branch}_predictions_{suffix}.csv')
            if row: validate(bank[branch],row['test'][branch])
        results[tag],case_rows[tag]=analyze(bank,'fusion')
        results[tag]['epoch']=row['epoch'] if row else None
    result=dict(source_files=sources,results=results,selection_protocol='original_test_as_val',independent_test=False,
        limitations=['Oracles require true labels and are not deployable gate performance',
            'CSV experts are video-level heads; window/feature-level complementarity is not measured',
            'Point head is co-trained with fusion, not an independently evaluated official checkpoint',
            'Only supplied prediction files are analyzed; unknown checkpoint epochs remain unknown',
            'Fixed rules are exploratory Test analyses, not newly validated selected models',
            'Video CSVs do not store full class probabilities or per-window routing observations',
            'true_class_probability and NLL require labels and must never be inputs to a deployable gate',
            'Same Test selection/filtering/group-overlap limitations as the underlying experiment'])
    result['checkpoint_metadata']=dict(nll_selected_epoch=selected['epoch'] if selected else None,
        latest_epoch=latest['epoch'] if latest else None,
        peak_top1_epoch=max(metrics,key=lambda r:r['test']['fusion']['video_top1'])['epoch'] if metrics else None,
        status='available' if metrics else 'unknown: metrics metadata missing')
    out=args.output or Path('reports')/('ucf101_fusion_complementarity_epoch'+(f'{latest["epoch"]:03d}' if latest else 'unknown'))
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for tag,cases in case_rows.items():
        with (out/f'{tag}_errors.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=ERROR_FIELDS);writer.writeheader();writer.writerows(cases)
    (out/'analysis.md').write_text(analysis_text(result))
    print(json.dumps(dict(output=str(out),results=results),indent=2))


if __name__=='__main__': main()
