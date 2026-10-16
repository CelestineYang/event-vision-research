"""Exact event-store equivalence tests, CPU only, synthetic temporary files."""
import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from preprocess_ucf101_event_store import prepare_video,video_directory
from train_ucf101_window_joint import ThreeViewDataset
from train_ucf101_window_joint import check_resume_config
from ucf101_event_store import EventStore,PreparedThreeViewDataset
from ucf101_window_data import sha,write_csv,write_json,WindowDataset


class EventStoreTests(unittest.TestCase):
    def test_only_storage_backend_transition_is_allowed(self):
        previous=dict(epochs=50,batch_size=64,manifest_sha256='same',
                      code={'train_ucf101_window_joint.py':'old','gcstr_models.py':'unchanged'})
        current=dict(previous,code={'train_ucf101_window_joint.py':'new','gcstr_models.py':'unchanged',
                                   'ucf101_event_store.py':'reader'},data_backend={'train':'full_event_memory_map'})
        with self.assertRaises(ValueError):check_resume_config(previous,current)
        check_resume_config(previous,current,True)
        check_resume_config(current,current)
        with self.assertRaises(ValueError):check_resume_config(previous,dict(current,batch_size=128),True)
        bad=dict(current,code=dict(current['code'],**{'gcstr_models.py':'changed'}))
        with self.assertRaises(ValueError):check_resume_config(previous,bad,True)

    def test_exact_events_shared_overlap_render_and_pickle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);experiment=root/'experiment';experiment.mkdir();store=root/'store';store.mkdir()
            rng=np.random.default_rng(4);n=30002
            arrays=dict(t=np.arange(n,dtype=np.int64)*50,x=rng.integers(0,32,n,dtype=np.int16),
                        y=rng.integers(0,24,n,dtype=np.int16),p=rng.integers(0,2,n,dtype=np.uint8))
            source=root/'source.npz';np.savez_compressed(source,**arrays)
            rows=[]
            for split in ('train','test'):
                for window_id,start in enumerate((0,500000)):
                    rows.append(dict(video_id=split+'_video',split=split,window_id=window_id,start_us=start,end_us=start+1000000,
                                     event_count=20000,label_id=0,source_path=str(source),source_sha256=sha(source),
                                     sensor_width=32,sensor_height=24,cache_key=split+str(window_id)))
            write_csv(experiment/'window_manifest.csv',rows,list(rows[0]))
            index={}
            for split in ('train','test'):
                batch=[r for r in rows if r['split']==split]
                metadata=prepare_video(batch,store)
                self.assertEqual(metadata['events'],n)
                self.assertEqual(metadata['windows']['1']['left'],10000)
                self.assertEqual(prepare_video(batch,store),metadata)
                folder=store/video_directory(batch[0]['video_id'])
                index[batch[0]['video_id']]=dict(directory=video_directory(batch[0]['video_id']),metadata_sha256=sha(folder/'metadata.json'))
            write_json(store/'videos.json',index)
            summary=dict(status='validating',rendering_validation='pending',window_manifest_sha256=sha(experiment/'window_manifest.csv'),
                         videos_index_sha256=sha(store/'videos.json'))
            write_json(store/'summary.json',summary)
            with self.assertRaises(RuntimeError): EventStore(store,experiment)
            summary.update(status='complete',rendering_validation='passed');write_json(store/'summary.json',summary)
            events=EventStore(store,experiment,max_open=1)
            for row in rows:
                piece=events.get(row);left=int(row['start_us'])//50
                for k,a in zip(('t','x','y','p'),piece):
                    self.assertEqual(a.dtype,arrays[k].dtype)
                    np.testing.assert_array_equal(a,arrays[k][left:left+20000])
            self.assertEqual(len(events._opened),1)
            self.assertEqual(len(pickle.loads(pickle.dumps(events))._opened),0)
            for view in ('cstr3','xt_p','ty_p'):
                test=WindowDataset(experiment,'test',view,require_cache=False)
                for i in range(len(test)): test[i]
            for split in ('train','test'):
                old=ThreeViewDataset(experiment,split,seed=9)
                new=PreparedThreeViewDataset(experiment,split,store,seed=9)
                old.set_epoch(7);new.set_epoch(7)
                for i in range(len(old)):
                    a,la,ia=old[i];b,lb,ib=new[i];self.assertEqual((la,ia),(lb,ib))
                    for view in a: torch.testing.assert_close(a[view],b[view],rtol=0,atol=0)
            modified=dict(rows[0],event_count=20001)
            with self.assertRaises(RuntimeError): events.get(modified)
            # A different split/index cannot silently reuse the store.
            with (experiment/'window_manifest.csv').open('a') as f:f.write('\n')
            with self.assertRaises(RuntimeError): EventStore(store,experiment)


if __name__=='__main__': unittest.main()
