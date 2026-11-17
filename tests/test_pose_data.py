import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pose_data import load_tracks,window_ends,validation_sources

class PoseDataTests(unittest.TestCase):
    def test_windows_never_cross_a_gap_or_bad_target(self):
        frames=np.arange(10);valid=np.ones((10,17),dtype=bool)
        self.assertEqual(window_ends(frames,valid).tolist(),[8,9])
        frames[-1]=20
        self.assertEqual(window_ends(frames,valid).tolist(),[8])
        valid[8]=False
        self.assertEqual(window_ends(frames,valid).tolist(),[])

    def test_nan_joint_is_masked_and_short_track_produces_no_windows(self):
        raw=np.column_stack((np.arange(17),np.arange(17)*2,np.ones(17))).astype(float)
        raw[0,0]=float('nan');raw[1,2]=0.01
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'example.json';p.write_text(json.dumps({'1':{'0':{'keypoints':raw.ravel().tolist(),'scores':1}}}))
            _,frames,_,features,_,valid=next(load_tracks(p))
            self.assertTrue(np.isfinite(features).all())
            self.assertFalse(valid[0,0]);self.assertFalse(valid[0,1])
            self.assertEqual(window_ends(frames,valid).tolist(),[])

    def test_recording_split_is_deterministic_and_keeps_fit_sources(self):
        paths=[Path(f'{c:02d}_{i}_alphapose_tracked_person.json') for c in (1,2) for i in range(10)]
        held=validation_sources(paths)
        self.assertEqual(held,validation_sources(list(reversed(paths))))
        self.assertEqual(len(held),4)
        self.assertEqual(len(set(p.name for p in paths)-held),16)

if __name__=='__main__':unittest.main()
