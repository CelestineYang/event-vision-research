"""Optional full-event gCSTR rendering on the exact point-window index slice."""
import argparse
import json
from pathlib import Path
import numpy as np
from gcstr_representations import GCSTRConfig, _normalize, _histogram2d, _resize_official


def render(events, start, end):
    x, y = events[:,0].astype(np.int64), events[:,1].astype(np.int64)
    t, p = events[:,2].astype(np.float64), events[:,3]
    if not len(t) or (t < start).any() or (t > end).any():
        raise ValueError('Invalid official window slice')
    cfg = GCSTRConfig(sensor_width=346, sensor_height=260)
    w,h,bins = cfg.sensor_width,cfg.sensor_height,cfg.time_bins
    pos,neg = p==1,p==0
    def mean(mask):
        sums,counts = np.zeros((w,h)),np.full((w,h),.01)
        np.add.at(sums,(x[mask],y[mask]),(t[mask]-start)/(end-start))
        np.add.at(counts,(x[mask],y[mask]),1.)
        return _normalize(sums/counts)
    density = _normalize(_histogram2d(x,y,(w,h),((0,w-1),(0,h-1))))
    result = {'cstr3':_resize_official(np.stack([mean(neg),density,mean(pos)]),224)}
    for view,a,b,shape,ranges in [('xt_p',x,t,(w,bins),((0,w-1),(start,end))),
                                ('ty_p',t,y,(bins,h),((start,end),(0,h-1)))]:
        counts = _normalize(np.stack([_histogram2d(a[m],b[m],shape,ranges) for m in (pos,neg)]))
        result[view] = _resize_official(np.concatenate([counts,np.zeros((1,*shape))]),224)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if json.loads((args.store/'status.json').read_text())['state']!='complete':
        raise ValueError('Event preprocessing is not complete')
    args.output.mkdir(parents=True,exist_ok=False)
    for split in ('train','test'):
        previous,array=None,None
        with (args.store/(split+'_windows.jsonl')).open() as f:
            for line in f:
                row=json.loads(line)
                if not row['valid']: continue
                if row['event_file']!=previous:
                    previous=row['event_file'];array=np.load(args.store/previous,mmap_mode='r')
                views=render(array[row['left']:row['right']],row['start_us'],row['end_us'])
                for view,image in views.items():
                    folder=args.output/split/view;folder.mkdir(parents=True,exist_ok=True)
                    np.save(folder/(str(row['point_row'])+'.npy'),image,allow_pickle=False)
    (args.output/'complete.json').write_text(json.dumps({'status':'complete','normalization':'raw projection; apply ImageNet normalization at training time'}))


if __name__=='__main__': main()
