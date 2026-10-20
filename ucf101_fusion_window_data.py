"""Aligned full-event frames and sampled points from one manifest window."""
import hashlib
import numpy as np
import torch

from gcstr_augmentation import augment_events
from gcstr_representations import GCSTRConfig
from ucf101_event_store import EventStore
from ucf101_window_data import WindowDataset, VIEWS, window_views


class FusionWindowDataset(WindowDataset):
    def __init__(self, experiment, split, store, *, point_count=2048, point_normalization='official', seed=0):
        super().__init__(experiment,split,VIEWS[0],seed=seed)
        if point_count < 1 or point_normalization not in ('official','window'):
            raise ValueError('Choose a positive point count and official/window normalization')
        self.point_count, self.point_normalization = point_count, point_normalization
        self.events = EventStore(store,experiment)
        self.split = split

    def __getitem__(self,index):
        row = self.rows[index]
        events = self.events.get(row)
        config = GCSTRConfig(sensor_width=int(row['sensor_width']),sensor_height=int(row['sensor_height']))
        if self.augmentation == 'spatial':
            events = augment_events(*events,mode='spatial',rng=self.augmentation_rng(row),config=config)
        t,x,y,p = events
        if len(t) == 0: raise ValueError('Empty augmented window')
        # Separate sampling RNG: changing point count cannot change frame augmentation.
        epoch = self.epoch if self.split == 'train' else 0
        key = f"points:{self.seed}:{epoch}:{row['video_id']}:{row['window_id']}"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
        indices = np.sort(rng.choice(len(t),self.point_count,replace=len(t)<self.point_count))
        times = np.asarray(t[indices],dtype=np.float64)
        if self.point_normalization == 'official':
            times -= times.min()
            times /= max(times.max(),1.)
            width,height = 320.,240.
        else:
            times = (times-int(row['start_us']))/(int(row['end_us'])-int(row['start_us']))
            width,height = float(config.sensor_width),float(config.sensor_height)
        points = np.stack((times,x[indices]/width,y[indices]/height)).astype(np.float32)
        # Test frames are unchanged by point sampling; reuse the verified frame
        # pretraining cache instead of rendering all full-event projections again.
        cached = {v:self.output/'cache'/v/(row['cache_key']+'.npy') for v in VIEWS}
        if self.split == 'test' and all(path.is_file() for path in cached.values()):
            images = {v:np.load(path,allow_pickle=False) for v,path in cached.items()}
        else:
            images = window_views(t,x,y,p,int(row['start_us']),int(row['end_us']),config,VIEWS)
        batch = {'events':torch.from_numpy(points), 'label':int(row['label_id']),
                 'video_id':row['video_id'],'window_id':str(row['window_id'])}
        for view,image in images.items():
            image = (image-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
            if not np.isfinite(image).all(): raise ValueError('Non-finite image')
            batch[view] = torch.from_numpy(image)
        return batch
