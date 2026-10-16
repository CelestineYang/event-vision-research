"""Full-event, per-video NPY memory maps with exact precomputed window indices.

This optional reader is deliberately separate from the active training entry.
It preserves original field dtypes, values, ordering and augmentation RNG.
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch

from gcstr_augmentation import augment_events
from gcstr_representations import GCSTRConfig
from ucf101_window_data import VIEWS, WindowDataset, sha, window_views

FIELDS = ('t','x','y','p')


class EventStore:
    def __init__(self, directory, experiment, *, max_open=8, _allow_pending=False):
        self.directory = Path(directory)
        self.summary = json.loads((self.directory/'summary.json').read_text())
        complete=self.summary.get('status')=='complete' and self.summary.get('rendering_validation')=='passed'
        if not complete and not (_allow_pending and self.summary.get('status')=='validating'):
            raise RuntimeError('Event preprocessing has not completed')
        if self.summary['window_manifest_sha256'] != sha(Path(experiment)/'window_manifest.csv'):
            raise RuntimeError('Event store belongs to a different window manifest')
        if sha(self.directory/'videos.json') != self.summary['videos_index_sha256']:
            raise RuntimeError('Event store video index changed')
        self.videos = json.loads((self.directory/'videos.json').read_text())
        self.max_open = max_open
        self._opened = OrderedDict()

    def __getstate__(self):
        # DataLoader spawn workers open their own lightweight memory maps.
        state = dict(self.__dict__)
        state['_opened'] = OrderedDict()
        return state

    def get(self, row):
        video = row['video_id']
        if video not in self._opened:
            record = self.videos[video]
            folder = self.directory/record['directory']
            if sha(folder/'metadata.json') != record['metadata_sha256']:
                raise RuntimeError('Event store metadata changed')
            metadata = json.loads((folder/'metadata.json').read_text())
            if metadata['source_sha256'] != row['source_sha256']:
                raise RuntimeError('Event store source identity mismatch')
            array = np.load(folder/'events.npy',mmap_mode='r',allow_pickle=False)
            if array.dtype.names != FIELDS or len(array) != metadata['events']:
                raise RuntimeError('Invalid event memory map')
            self._opened[video] = (array,metadata)
            while len(self._opened)>self.max_open:
                self._opened.popitem(last=False)
        array,metadata = self._opened[video]
        self._opened.move_to_end(video)
        window = metadata['windows'][str(row['window_id'])]
        if (window['start_us'],window['end_us'],window['event_count']) != (
                int(row['start_us']),int(row['end_us']),int(row['event_count'])):
            raise RuntimeError('Window identity mismatch')
        piece = array[window['left']:window['right']]
        if len(piece)!=int(row['event_count']): raise RuntimeError('Window count mismatch')
        return tuple(piece[k] for k in FIELDS)


class PreparedThreeViewDataset(WindowDataset):
    """Same full-event projections/normalization; direct indexed event reads."""
    def __init__(self,experiment,split,store,*,seed=0,_allow_pending=False):
        super().__init__(experiment,split,VIEWS[0],seed=seed)
        self.events = EventStore(store,experiment,_allow_pending=_allow_pending)

    def __getitem__(self,index):
        row = self.rows[index]
        events = self.events.get(row)
        config = GCSTRConfig(sensor_width=int(row['sensor_width']),sensor_height=int(row['sensor_height']))
        if self.augmentation=='spatial':
            events = augment_events(*events,mode='spatial',rng=self.augmentation_rng(row),config=config)
        images = window_views(*events,int(row['start_us']),int(row['end_us']),config,VIEWS)
        result = {}
        for view,image in images.items():
            if image.shape != (3,224,224) or not np.isfinite(image).all():
                raise ValueError('Invalid prepared projection')
            image = (image-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
            result[view] = torch.from_numpy(image)
        return result,int(row['label_id']),row['video_id']
