import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from prepare_daily import load_aedat, windows, sample_points, read_records, CLASSES


def reference_windows(t):
    start=int(t[0]); starts=[]; ends=[]; i=0
    while i<len(t):
        if t[i]>=start and start+1500000<t[-1]:
            starts.append(i); start+=500000
        else:i+=1
    i=len(t)-1; end=start-500000+1500000
    while i>=0:
        if t[i]<=end:ends.append(i);end-=500000
        else:i-=1
    return list(zip(starts,ends[:len(starts)][::-1]))


class DailyTests(unittest.TestCase):
    def test_stratified_split_coverage_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in CLASSES:
                (root/name).mkdir()
                for i in reversed(range(120)): (root/name/f'person{i:03d}.aedat').touch()
            records=read_records(root,0)
            self.assertEqual(records,read_records(root,0))
            self.assertNotEqual(records,read_records(root,1))
            train={r['filename'] for r in records if r['split']=='train'}
            test={r['filename'] for r in records if r['split']=='test'}
            self.assertFalse(train & test)
            self.assertEqual(len(train | test),1440)
            for label in range(12):
                self.assertEqual(sum(r['label']==label and r['split']=='train' for r in records),96)
                self.assertEqual(sum(r['label']==label and r['split']=='test' for r in records),24)

    def test_decode_preserves_last_event_and_rejects_damage(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.aedat'; header=b'#!AER-DAT2.0\r\n# comment\r\n'
            body=struct.pack('>IIII', (127<<8)|(126<<1)|1,1234,0,2234)
            p.write_bytes(header+body)
            np.testing.assert_array_equal(load_aedat(p),[[126,127,0,1],[0,0,1000,0]])
            p.write_bytes(header+body[:-1])
            with self.assertRaises(ValueError):load_aedat(p)
            p.write_bytes(header+struct.pack('>IIII',0,20,0,10))
            with self.assertRaises(ValueError):load_aedat(p)
            p.write_bytes(header+struct.pack('>II',0x8000,20))
            with self.assertRaises(ValueError):load_aedat(p)

    def test_explicit_timestamp_repairs_preserve_events(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.aedat'
            header=b'#!AER-DAT2.0\n'
            for times, expected in (([2**32-10,2**32-2,3,10],[0,8,13,20]),
                                    ([100,120,110,120,140],[0,10,20,20,40])):
                p.write_bytes(header+b''.join(struct.pack('>II',i*2,t) for i,t in enumerate(times)))
                e,a=load_aedat(p,repair=True,return_audit=True)
                np.testing.assert_array_equal(e[:,2],expected)
                self.assertEqual(sorted(e[:,0].tolist()),list(range(len(times))))
                if len(times)==5:self.assertEqual(e[:,0].tolist(),[0,2,1,3,4])
                self.assertTrue(a['backward_indices'])

    def test_sparse_duplicate_and_exact_boundaries(self):
        rng=np.random.default_rng(12)
        cases=[np.array([0,1500000]),np.array([0,1500000,1500000,2000001]),np.array([0,6000000])]
        cases += [np.sort(rng.integers(0,7000000,size=100)) for _ in range(100)]
        for t in cases:
            self.assertEqual([(l,r) for _,_,l,r in windows(t)],reference_windows(t))

    def test_official_base_point_normalization(self):
        n=12000
        e=np.column_stack((np.arange(n)%128,np.arange(n)%127,317+np.arange(n)*123,np.arange(n)%2))
        self.assertIsNone(sample_points(e[:10000],2048,np.random.RandomState(3)))
        actual=sample_points(e,2048,np.random.RandomState(3))
        idx=np.arange(n);rng=np.random.RandomState(3);rng.shuffle(idx);idx=np.sort(idx[:2048])
        expected=e[idx][:,[2,0,1]].copy()
        expected[:,0]//=1000
        expected[:,0]-=expected[:,0].min()
        expected=expected/expected.max(axis=0)
        expected[:,1:]=e[idx][:,:2]/128
        np.testing.assert_array_equal(actual,expected.astype('f4'))
        e[:,2]=0
        with self.assertRaises(ValueError):sample_points(e,2048,np.random.RandomState(3))


if __name__=='__main__':unittest.main()
