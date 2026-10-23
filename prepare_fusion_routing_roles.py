"""Map audited UCF source videos to Gate roles; never generate a new split."""
import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import re

from routing_data import digest, json_write, protocol_config


def raw_lists(train, test, mapping):
    rows = []
    seen = set()
    for split, path in [('train', train), ('test', test)]:
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            name, label = ast.literal_eval(line)
            pieces = [p for p in name.replace('\\', '/').split('/') if p]
            category, filename = pieces[-2:]
            match = re.fullmatch(r'v_.+_g(\d+)_c(\d+)\.mat', filename)
            if not match or mapping.get(category) != label:
                raise ValueError('TXT category/filename mismatch')
            vid = f'{int(label):03d}_{Path(filename).stem}'
            if vid in seen:
                raise ValueError('Duplicate or overlapping TXT video')
            seen.add(vid)
            rows.append(dict(video_id=vid, split=split, label_id=int(label), group=int(match[1])))
    return rows


def build_roles(raw, splits, windows, *, dataset_id='UCF101-DVS'):
    source = {r['video_id']:r for r in raw}
    indexed = {r['video_id']:r for r in splits}
    if len(source)!=len(raw) or len(indexed)!=len(splits) or set(source)!=set(indexed):
        raise ValueError('Missing/duplicate videos between original TXT and audited video splits')
    for vid, r in indexed.items():
        if (r['split'],int(r['label_id'])) != (source[vid]['split'],int(source[vid]['label_id'])):
            raise ValueError('Audited video source/label differs from original TXT')
    valid, seen = {}, set()
    counts = Counter()
    for row in windows:
        vid = row['video_id']
        key = (vid,str(row['window_id']))
        if key in seen:
            raise ValueError('Duplicate valid window')
        seen.add(key)
        if vid not in source or (row['split'],int(row['label_id'])) != (source[vid]['split'],int(source[vid]['label_id'])):
            raise ValueError('Valid-window source/label differs from original TXT')
        valid[vid] = source[vid]
        counts[row['split']] += 1
    if not valid:
        raise ValueError('No audited valid windows')
    protocol = protocol_config('original_test_as_val')
    videos = [dict(source_split=r['split'],video_id=vid,
                   role='gate_train' if r['split']=='train' else 'gate_val') for vid,r in sorted(valid.items())]
    statistics = {}
    for split in ('train','test'):
        original = [r for r in raw if r['split']==split]
        retained = [r for r in valid.values() if r['split']==split]
        statistics[split] = dict(original_videos=len(original),valid_videos=len(retained),valid_windows=counts[split],
            original_class_coverage=len({int(r['label_id']) for r in original}),
            valid_class_coverage=len({int(r['label_id']) for r in retained}),
            valid_class_video_counts=dict(sorted(Counter(str(r['label_id']) for r in retained).items())),
            excluded_no_valid_windows=sorted(r['video_id'] for r in original if r['video_id'] not in valid))
    groups = {s:{(int(r['label_id']),int(r['group'])) for r in raw if r['split']==s} for s in ('train','test')}
    valid_groups = {s:{(int(r['label_id']),int(r['group'])) for r in valid.values() if r['split']==s} for s in ('train','test')}
    report = dict(**protocol, counts=statistics, missing_records=0, duplicate_records=0,
        exact_video_overlap=0, class_group_counts={s:len(g) for s,g in groups.items()},
        overlapping_class_groups=len(groups['train'] & groups['test']),
        valid_overlapping_class_groups=len(valid_groups['train'] & valid_groups['test']),
        group_overlap_policy='reported, allowed by existing user-selected split')
    return dict(schema=1,dataset_id=dataset_id,**protocol,videos=videos), report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2'))
    p.add_argument('--train-list',type=Path,default=Path('train_ucf.txt'))
    p.add_argument('--test-list',type=Path,default=Path('test_ucf(1).txt'))
    p.add_argument('--output',type=Path,default=Path('configs/fusion_routing_roles.original_test_as_val.json'))
    p.add_argument('--report',type=Path,default=Path('reports/routing_revision_v2/role_mapping.json'))
    p.add_argument('--check',action='store_true',help='Verify existing outputs match sources, without rewriting')
    args=p.parse_args()
    from ucf101_window_data import checked_audit
    audit=checked_audit(args.experiment)
    mapping=json.loads((args.experiment/'class_to_idx.json').read_text())
    for split,path in [('train',args.train_list),('test',args.test_list)]:
        if digest(path)!=audit['config']['source_manifests'][split]:
            raise ValueError('Original TXT identity differs from audited source')
    def read(name):
        with (args.experiment/name).open() as f:
            return list(csv.DictReader(f))
    roles,report=build_roles(raw_lists(args.train_list,args.test_list,mapping),read('video_splits.csv'),read('window_manifest.csv'))
    sources={name:digest(args.experiment/name) for name in
             ('window_manifest.csv','video_splits.csv','class_to_idx.json','dataset_audit.json')}
    sources.update(train_txt=digest(args.train_list),test_txt=digest(args.test_list))
    roles['source_sha256']=sources
    report['source_sha256']=sources
    report['audit_excluded_videos']=audit.get('excluded_videos', [])
    report['audit_coverage_matches'] = all(
        report['counts'][s][a]==audit['coverage'][s][b] for s in ('train','test')
        for a,b in [('original_videos','original_videos'),('valid_videos','retained_videos'),('valid_windows','windows')])
    if not report['audit_coverage_matches']:
        raise ValueError('Computed role coverage differs from current audit')
    if args.check:
        if json.loads(args.output.read_text())!=roles or json.loads(args.report.read_text())!=report:
            raise ValueError('Generated role/report identity differs from existing output')
    else:
        if args.output.exists() or args.report.exists():
            raise FileExistsError('Use --check or new output paths')
        for path,obj in [(args.output,roles),(args.report,report)]:
            path.parent.mkdir(parents=True,exist_ok=True)
            json_write(path,obj)
    print(json.dumps(dict(roles=str(args.output),report=str(args.report),counts={s:{k:v for k,v in r.items()
        if k not in ('excluded_no_valid_windows','valid_class_video_counts')} for s,r in report['counts'].items()},
        exact_video_overlap=report['exact_video_overlap'],overlapping_class_groups=report['overlapping_class_groups']),indent=2))


if __name__=='__main__':
    main()
