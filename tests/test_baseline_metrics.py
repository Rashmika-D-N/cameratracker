import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline_metrics import ranking_metrics,aggregate_frames,frame_metrics
from pose_model import PosePredictor
import torch

class BaselineTests(unittest.TestCase):
    def test_ties_and_ranking(self):
        self.assertAlmostEqual(ranking_metrics([0,1],[.5,.5])['roc_auc'],.5)
        self.assertAlmostEqual(ranking_metrics([0,1],[.5,.5])['average_precision'],.5)
        self.assertEqual(ranking_metrics([0,1],[0,1])['roc_auc'],1.)
        self.assertEqual(ranking_metrics([0,1],[1,0])['roc_auc'],0.)
        self.assertIsNone(ranking_metrics([0,0],[0,1])['roc_auc'])

    def test_missing_coverage_is_not_a_normal_prediction(self):
        scores=aggregate_frames([('a',1,.6),('a',1,.8)],{'a':3})['a']
        self.assertTrue(np.isnan(scores[0]));self.assertEqual(scores[1],.8)
        m=frame_metrics([1,1,0],scores,.7)
        self.assertEqual(m['scored_frames'],1)
        self.assertEqual(m['recall_on_scored_positives'],1.)
        self.assertEqual(m['operational_recall_if_abstentions_emit_no_alert'],.5)

    def test_model_has_no_future_input_dependency(self):
        torch.manual_seed(17);model=PosePredictor();x=torch.randn(2,68,8);changed=x.clone();changed[:,:,5:]+=10
        torch.testing.assert_close(model.encode(x)[:,:,:5],model.encode(changed)[:,:,:5])

if __name__=='__main__':unittest.main()
