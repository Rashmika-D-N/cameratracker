import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from video_features import interaction_proxies,contiguous_contexts

class VideoFeatureTests(unittest.TestCase):
    def fixture(self):
        k=np.column_stack((np.arange(17),np.arange(17)*2,np.ones(17))).astype(float)
        return {'keypoints':k.tolist()}

    def test_translation_does_not_change_distances(self):
        p=self.fixture();q={'keypoints':(np.asarray(p['keypoints'])+[20,30,0]).tolist()}
        self.assertEqual(interaction_proxies(p),interaction_proxies(q))

    def test_missing_joint_does_not_become_zero_distance(self):
        p=self.fixture();p['keypoints'][9][2]=.01
        self.assertIsNone(interaction_proxies(p))

    def test_context_cannot_cross_gaps_or_unknown_features(self):
        rows=[{'time':i/10,'features':{}} for i in range(12)]
        self.assertGreater(contiguous_contexts(rows,.8),0)
        rows[6]['features']=None
        self.assertEqual(contiguous_contexts(rows,.8),0)
        rows=[{'time':i/10+(1 if i>=6 else 0),'features':{}} for i in range(12)]
        self.assertEqual(contiguous_contexts(rows,.8),0)

if __name__=='__main__':unittest.main()
