import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from prepare_dvsgesture import load_aedat, windows, sample_points, read_records, segment_bounds
from render_frames import render
from gcstr_representations import GCSTRConfig


def reference_windows(t):
    start=int(t[0]);starts=[];ends=[];i=0
    while i<len(t):
        if t[i]>=start and start+500000<t[-1]:starts.append(i);start+=250000
        else:i+=1
    i=len(t)-1;end=start-250000+500000
    while i>=0:
        if t[i]<=end:ends.append(i);end-=250000
        else:i-=1
    return list(zip(starts,ends[:len(starts)][::-1]))


class GestureTests(unittest.TestCase):
    def test_aedat_validity_overflow_and_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events.aedat'
            words=[((3<<17)|(4<<2)|3,7),((127<<17)|(126<<2)|1,9),(0,10)]
            payload=b''.join(struct.pack('<II',*w) for w in words)
            header=b'#!AER-DAT3.1\r\n#!END-HEADER\r\n'
            packet=struct.pack('<HHIIIIII',1,1,8,4,2,3,3,2)+payload
            p.write_bytes(header+packet)
            np.testing.assert_array_equal(load_aedat(p),[[3,4,(2<<31)+7,1],[127,126,(2<<31)+9,0]])
            p.write_bytes(header+packet[:-1])
            with self.assertRaises(ValueError):load_aedat(p)
            p.write_bytes(header+struct.pack('<HHIIIIII',1,1,8,4,0,3,3,3)+payload)
            with self.assertRaises(ValueError):load_aedat(p)

    def test_disorder_outside_annotations_preserves_segment(self):
        events=np.zeros((7,4),dtype='i8');events[:,2]=[0,3,2,10,11,12,20]
        self.assertEqual(segment_bounds(events,10,20,False),(3,6))
        with self.assertRaises(ValueError):segment_bounds(events,0,10,False)
        with self.assertRaises(ValueError):segment_bounds(events,0,3,False)

    def test_window_boundaries_match_upstream(self):
        rng=np.random.default_rng(2)
        cases=[np.array([0,500000]),np.array([0,250000,500000,500000,750001]),np.array([0,3000000,5000000])]
        cases += [np.sort(rng.integers(0,7000000,size=300)) for _ in range(100)]
        for t in cases:self.assertEqual([(l,r) for _,_,l,r in windows(t)],reference_windows(t))

    def test_sampling_threshold_order_and_normalization(self):
        for n in (100,101,2000):
            events=np.column_stack((np.arange(n)%128,np.arange(n)%127,80000000+np.arange(n)*100,np.arange(n)%2))
            actual=sample_points(events,1024,np.random.RandomState(3))
            if n==100:self.assertIsNone(actual);continue
            rng=np.random.RandomState(3);idx=np.arange(n)
            if n<1024:idx=rng.choice(idx,size=1024,replace=True)
            else:rng.shuffle(idx);idx=idx[:1024]
            idx.sort();expected=events[idx][:,[2,0,1]].astype('f4')
            expected[:,0]=(expected[:,0]-expected[:,0].min())/(expected[:,0].max()-expected[:,0].min()+1e-6)
            expected[:,1:]/=128
            np.testing.assert_array_equal(actual,expected)

    def test_dvs_render_dimensions(self):
        events=np.array([[0,0,0,0],[127,127,499999,1]])
        views=render(events,0,500000,GCSTRConfig(sensor_width=128,sensor_height=128))
        for im in views.values():
            self.assertEqual(im.shape,(3,224,224));self.assertTrue(np.isfinite(im).all())

    def test_subject_leakage_and_other_class(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for directory in ('splits','raw/events','annotations'):(root/directory).mkdir(parents=True)
            for split,user in (('train','user01'),('test','user24')):
                name=user+'_lab.aedat';(root/'splits'/f'trials_to_{split}.txt').write_text(name+'\n')
                (root/'raw/events'/name).touch()
                (root/'annotations'/(user+'_lab_labels.csv')).write_text('class,startTime_usec,endTime_usec\n1,0,1000000\n11,900000,2000000\n')
            records,audit=read_records(root,10)
            self.assertEqual(len(records),2);self.assertTrue(all(r['label']==0 for r in records))
            self.assertEqual(audit['train']['excluded_other_segments'],1)
            self.assertEqual(len(audit['train']['annotation_overlaps']),1)
            (root/'splits/trials_to_test.txt').write_text('user01_lab.aedat\n')
            with self.assertRaises(ValueError):read_records(root,10)


if __name__=='__main__':unittest.main()
