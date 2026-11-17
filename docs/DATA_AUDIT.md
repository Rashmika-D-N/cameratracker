# Pose data audit and first model smoke test

Date: 2026-11-17. This is a research input pipeline, not a deployed shoplifting detector.

## Observed data quality

| Measure | Released training | Official test |
|---|---:|---:|
| Source files | 104 | 47 |
| Person tracks | 566 | 78 |
| Person/frame observations | 79,632 | 4,620 |
| Observations containing non-finite keypoints | 2,099 | 150 |
| Frames failing the chosen pose-quality rule | 4,837 | 270 |
| Gaps between observed frame indices within tracks | 197 | 7 |
| Eligible 8-frame-history/next-frame-target windows | 65,504 | 3,347 |

All observations have 51 coordinate/confidence values. All test observation indices are within their corresponding mask bounds. The 47 test masks contain 4,846 frame labels, including 1,534 positives. **1,125 labeled frames have no person pose at all**, so missing coverage must be reported rather than silently scored as normal.

No byte-identical files were found across the audit. This does not rule out duplicate content, shared actors, or recording-session overlap. Frame indices fit label bounds, but semantic event boundaries and person-level attribution have not been independently established from video.

## Baseline input policy

- Preserve immutable originals. Invalid/non-finite or confidence-below-0.2 joints become zero-valued numeric inputs with an explicit invalidity mask; their zero values are never treated as observed coordinates.
- Center on the valid-joint mean and normalize by RMS distance. A usable observation needs at least eight confident finite joints and nonzero scale. This generic normalization avoids assuming unverified joint names.
- Each input contains eight previous consecutive observed frames, with x/y, confidence, and validity per joint (68 values per frame). Predict the next observed normalized x/y pose; loss includes only valid target joints.
- Exclude windows that cross a frame gap, a track/source boundary, or insufficient-quality observations. Never interpolate over those gaps for this baseline.
- Use the publisher-described normal training data for normality prediction. Do not call an anomaly score a theft probability.
- Reserve whole files within each camera deterministically: 84 fitting files / 20 validation files, with 55,549 / 9,955 eligible windows. Keep all 47 official test sources out of fitting and threshold tuning.
- Test masks are scene/frame-level evidence, not automatically positive labels for every tracked person.

The numeric confidence/history/quality settings are initial experiment choices, not tuned production defaults. Frame rate, joint semantics, producer interpolation/smoothing causality, and performance under the actual YOLO pose extractor need further validation.

## Reproduce

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\audit_pose_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\smoke_pose_model.py
```

The smoke test uses 16 fitting-only examples and two optimizer updates on a small causal convolutional predictor. It checks finite outputs/gradients, weight updates, and independence of earlier encoder outputs from later inputs. It saves no model checkpoint and reports no detection accuracy.

Machine-readable outputs are under `artifacts/data_audit/` and `artifacts/smoke/`, both ignored by Git. The exact environment lock is `requirements-cpu.lock.txt`; install with the official CPU index in addition to PyPI. The isolated CPU environment is sufficient for this small plumbing check; GPU installation is a separate later experiment.

Next: run a controlled baseline training experiment with validation-only model selection, then report held-out scene-level scores and missing coverage separately. Add real video/YOLO extraction and ordinary-shopping hard negatives before any application or shop-performance claim.

## Verified execution

The isolated CPython 3.10.11 environment uses NumPy 2.2.6 and PyTorch 2.9.1+cpu. Dependency consistency checks and all three preprocessing tests passed there. The 4,642-parameter predictor completed both CPU optimizer updates with finite gradients and a passing causal-prefix check; smoke losses were 0.22380 and 0.21777. No checkpoint was saved and no official test data was fitted. These two losses are plumbing results, not validation or detection metrics.

The controlled first training experiment is now complete. See [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for measured results and remaining limitations. The preprocessing smoke check remains distinct from detector validation.
