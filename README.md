# CameraTracker

An open-source research project for local CCTV monitoring and human review of possible product concealment in small shops.

**Current status: experimental model and video-processing pipeline. Not ready for shop deployment.** Taking an item from a public shelf is normal shopping. This project does not establish theft, payment, ownership, or intent from video.

## What works today

- PoseLift download verification, numeric data audit, missing-joint masks, and gap-aware sequence preprocessing.
- A 4,642-parameter causal pose predictor trained for 15 epochs, with validation-only checkpoint and threshold selection.
- Official-test evaluation with explicit missing coverage and a previous-pose comparison.
- Pretrained YOLO11n-pose extraction and ByteTrack on eight public MNNIT video clips.
- Body-motion proxy features and 0.8-, 2.4-, and 4.8-second context availability checks.
- Nine automated tests for data gaps, validity masks, causal behavior, metric ties, and missing coverage.

YOLO has **not** been fine-tuned. The web application, live-camera service, and a reliable concealment classifier are not implemented. Phone-retrieval and clothing-adjustment negatives are not yet verified in the video pilot.

## First baseline results

| Official test measure | Result |
|---|---:|
| ROC-AUC on scored frames | 0.5495 |
| Precision at validation-frozen threshold | 36.54% |
| Recall over all positive frames, abstentions emit no alert | 1.24% |
| Labeled frames receiving a score | 59.16% |

These are frame-level anomaly results, not shoplifting accuracy. The model is insufficient for detection. See [the experiment report](docs/BASELINE_RESULTS.md) and [machine-readable results](reports/baseline_v1.json).

## Local setup

Use a working Python 3.10 interpreter. On Windows, select the actual installed interpreter if `python` resolves to a WindowsApps alias.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For the video extractor, install `requirements-video.lock.txt` in that environment. The experiments currently use CPU inference; a CUDA setup and low-spec-device certification remain future work.

Datasets and weights are downloaded separately from their publishers. The [download report](docs/DATASET_DOWNLOAD.md) describes the PoseLift layout under `data/poselift/original/`. Once that data is present:

```powershell
.\.venv\Scripts\python.exe scripts\audit_pose_data.py
.\.venv\Scripts\python.exe scripts\smoke_pose_model.py
.\.venv\Scripts\python.exe scripts\train_pose_baseline.py --run my_baseline_run
```

Use a new run name for each experiment. The [video pilot report](docs/RAW_VIDEO_PILOT.md) describes its separately downloaded clips, required model, commands, and label limitations. Its current scripts use the documented local pilot manifest and reference run layout.

## Project map

- `scripts/`: data audit, temporal model/training, evaluation, video extraction, and features.
- `tests/`: CPU unit tests; no dataset/model download is required to run them.
- `reports/`: small measured-result snapshots; no footage, poses, or model binaries.
- `docs/`: [architecture](docs/ARCHITECTURE.md), [implementation plan](docs/IMPLEMENTATION_PLAN.md), [data audit](docs/DATA_AUDIT.md), and experiment reports.

Next work is a reviewed, independently grouped set of ordinary shopping, phone retrieval, clothing adjustment, and staged concealment, followed by a controlled feature/context comparison. The existing official test has already been inspected and is not an untouched final test for subsequent designs.

## License and external assets

Project source is licensed under [GNU AGPL-3.0](LICENSE), matching the open-source path used for the Ultralytics integration. Dependencies, datasets, and pretrained weights retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). No dataset, pretrained weight, trained checkpoint, or customer footage is distributed in this Git repository.
