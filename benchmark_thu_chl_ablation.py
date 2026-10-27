"""Train-only finite batch probe; outputs never initialize formal training."""
import argparse
import copy
import torch
from thu_chl_ablation_runtime import configuration
from thu_chl_ablation_fusion import load_model
from thu_chl_sequence_dataset import THUVideoSequenceDataset
from gcstr_video_sequence_dataset import make_schedule
from feature_ablation_runtime import run_pass
from train_gcstr_feature_memory import seed_all, write_json

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config', default='configs/thu_chl_ablations/g.json')
    a=p.parse_args()
    c=configuration(a.config)
    c['video_batch_size_per_gpu']=a.batch
    seed_all(c['seed']);device=torch.device('cuda:0')
    m=load_model(c).to(device)
    d=THUVideoSequenceDataset(c,'train',1)
    # Fill every lane, including probes above batch 16; only Train windows.
    video_count=max(16,a.batch)
    window_count=7 if a.batch>16 else 8
    videos=sorted(v for v in d.groups if len(d.groups[v])>=window_count)[:video_count]
    assert len(videos)==video_count
    d.groups={v:d.groups[v][:window_count] for v in videos}
    d.position={i:(j,idx) for idx in d.groups.values() for j,i in enumerate(idx)}
    d.windows=sum(map(len,d.groups.values()))
    d.schedule=make_schedule(d.groups,a.batch,c['tbptt_steps'],c['seed'],1,True)
    groups,_=m.parameter_groups(c)
    opt=torch.optim.AdamW(groups,weight_decay=c['weight_decay'])
    torch.cuda.reset_peak_memory_stats()
    result,_,_=run_pass(m,d,device,c,opt)
    write_json(a.output,dict(batch=a.batch,train_windows=d.windows,metrics=result,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
