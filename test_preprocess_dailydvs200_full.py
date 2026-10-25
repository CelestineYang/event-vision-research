import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from prepare_dailydvs200_windows import bounds
from preprocess_dailydvs200_full import prepare_one


class FullPreprocessingTests(unittest.TestCase):
    def test_sparse_full_events_images_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            arrays=dict(t=np.array([0,500000,1100000],dtype=np.uint64),
                        x=np.array([10,20,30],dtype=np.uint16),y=np.array([11,21,31],dtype=np.uint16),
                        p=np.array([False,True,False]))
            source=root/'sample.npz'
            np.savez_compressed(source,**arrays)
            stat=source.stat()
            row=dict(sample_id='train_00000',split='train',label_id='0',class_name='action_001',
                     sensor_width='320',sensor_height='240',npz_relpath='sample.npz')
            record=dict(event_count=3,source_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,windows=bounds(arrays['t']))
            task=(str(root),str(root/'output'),row,record)
            first=prepare_one(task)
            self.assertFalse(first['resumed'])
            self.assertEqual(first['rows'][0]['event_count'],2)
            mmap=np.load(root/'output/event_store/videos/train_00000/events.npy',mmap_mode='r')
            for k,v in arrays.items():
                np.testing.assert_array_equal(mmap[k],v)
                self.assertEqual(mmap[k].dtype,v.dtype)
            repeated=prepare_one(task)
            self.assertTrue(repeated['resumed'])
            self.assertEqual(first['metadata_sha256'],repeated['metadata_sha256'])
            meta=json.loads((root/'output/event_store/videos/train_00000/metadata.json').read_text())
            self.assertEqual(len(meta['image_sha256']),3)
            image=root/'output/experiment'/next(iter(meta['image_sha256']))
            with image.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(RuntimeError,'image checksum'):
                prepare_one(task)


if __name__=='__main__':
    unittest.main()
