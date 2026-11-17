# Pose baseline v1: experimental, not deployable

Run completed 2026-11-17. **This model does not meet the project's detection goals.** Its anomaly ranking is only weakly discriminative, and the frozen normal-calibrated threshold misses most labeled anomalies. Do not describe these results as shoplifting accuracy.

## Experiment

- Own implementation of a 4,642-parameter causal next-pose predictor; no upstream STG-NF training code was copied.
- Publisher-described normal-only PoseLift fitting data; no positive test labels used for fitting, model selection, or threshold selection.
- Source groups: 84 fitting / 20 validation / 47 official test recordings. Actor/session independence is not established.
- 55,548 fitting windows and 9,955 validation windows; eight history frames, two CPU threads, batch 256, seed 17, learning rate 0.001.
- The common-joint score requires at least four joints visible at both previous and target frames. This removes one previously audit-eligible fitting window. No window crosses a frame gap or a source/person boundary.
- Maximum 15 epochs, patience 4, checkpoint chosen by normal validation prediction error. Best epoch: 15; fitting time: 25.99 seconds, excluding data load/imports.
- Threshold frozen before test labels were loaded: 0.06443268, the 99th percentile of covered normal validation **scene** scores. Multiple person scores at a frame are aggregated by maximum, without future smoothing.
- Compared against a zero-parameter previous-pose persistence predictor, with its own threshold from the same validation quantile. Both methods use the same target/previous joint mask and test coverage.

## Official test results

| Measure | Learned predictor | Persistence comparison |
|---|---:|---:|
| ROC-AUC on scored frames | 0.5495 | 0.4875 |
| Average precision on scored frames | 0.4500 | 0.4125 |
| Precision at frozen threshold | 36.54% | 31.43% |
| Recall among scored positive frames | 1.53% | 0.89% |
| Recall across all labeled positives, abstentions emit no alert | 1.24% | 0.72% |
| Scored normal-frame exceedance rate | 2.03% | 1.48% |

Scores cover 2,867/4,846 labeled frames (59.16%). Of 1,534 positive frames, 293 have no eligible score. Low coverage includes missing poses, quality filtering, discontinuities, and the required history. It is not silently counted as normal behavior. Average precision is conditional on scored frames; the scored-positive prevalence is 43.29%.

Learned-model covered-frame confusion counts: {'tp': 19, 'fp': 33, 'fn': 1222, 'tn': 1593}. This is a frame-level benchmark; event recall and false alerts per camera-hour have not been measured.

The learned model's validation prediction loss (0.007977) is worse than persistence (0.005433), despite its slightly better anomaly ranking on this one test. Neither method is suitable as the product's detection engine.

## Local compute measurement

Model-only CPU batch-one median/p95: 0.124/0.282 ms, after 50 warmups and 300 timed calls. These figures exclude decoding, YOLO/pose extraction, preprocessing, tracking, storage, and the application, and do not certify any shop-device camera count.

## Saved artifacts

`artifacts/baseline/pose_baseline_v1/` contains `best.pt`, `run_manifest.json`, `history.json`, `selection.json`, `report.json`, and `test_scene_scores.npz`. It is ignored by Git. Source hashes, model hash, split membership, settings, package versions, and per-test-source results are retained. The checkpoint is research-only, not a released model.

Reproduce as a new immutable run from PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\train_pose_baseline.py --run pose_baseline_v1_reproduction
```

The trainer refuses to overwrite an existing run. The reference result is one seed/run, not an estimate of cross-seed reliability.

## Interpretation and next work

The experiment tests whether short-window normalized-pose prediction error is a useful anomaly signal. It does not directly learn taking an item, concealment, payment, ownership, or intent. A low prediction loss is not proof of useful concealment detection.

Next investigate longer action context and explicit interaction/visibility features as hypotheses, and collect separately consented normal/ambiguous/staged-concealment validation footage. Public-shelf pickup, basket use, phone retrieval, clothing adjustment, and legitimate bag use must remain normal or uncertain cases. Select future detection methods using that independent validation evidence. This official test has now been inspected; it cannot remain an untouched final test for designs chosen in response to these results. Reserve new data for final claims.

Producer interpolation/smoothing causality, frame timing, joint semantics, per-person attribution, and the actual deployed YOLO extractor still need independent validation. Do not proceed to shop deployment or present this checkpoint as a trained theft classifier.
