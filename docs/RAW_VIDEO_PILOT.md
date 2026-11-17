# Public raw-video pilot

Completed 2026-11-17. Actual YOLO extraction and longer-context feature availability were tested. **A reviewed four-action training/validation set is not yet complete, and no new classification accuracy is claimed.**

## Public sources checked

| Source | Available content | Decision |
|---|---|---|
| [MNNIT Shoplifting Dataset V1](https://data.mendeley.com/datasets/r3yjf35hzr/1) | 182 MP4 members: 90 Normal and 92 Shoplifting; archive 760,562,047 bytes; publisher states CC BY 4.0 | Selected as the first raw-video source. V2 currently has metadata only. |
| [Zenodo event dataset](https://zenodo.org/records/10149996) | Normal/shoplifting RAR archives totaling about 943 MB | Alternative to inspect; file contents/terms not re-audited for this pilot. |
| [Charades](https://prior.allenai.org/projects/charades) | Phone and dressing actions; 480p archive about 13 GB | Its [license](https://prior.allenai.org/projects/data/charades/license.txt) is non-commercial and restricts redistribution. Not incorporated in this shop-distribution pipeline. |
| [NVIDIA PhysicalAI Event Videos](https://huggingface.co/datasets/nvidia/PhysicalAI-Event-Videos) | Synthetic annotated events/hard negatives; chunk archive about 3.85 GB | Metadata examined only. Custom dataset terms and additional generator terms need separate review; no video included. |

## Downloaded pilot

Eight MNNIT videos (four per provider class), **55,649,268 bytes**. Selected source members were fetched with HTTP byte ranges, rather than the whole archive. ZIP CRC32 and local SHA-256 verified each selected member; the full archive's publisher hash was retained but cannot be verified without downloading the whole archive. Attribution is retained.

Files: `data/video_pilot/mnnit_v1/`. The source-label manifest and sparse-review notes are kept there, outside Git. Selection spread across source filenames is not proof of actor/session independence or representativeness.

Contact-sheet review found one empty-room negative and three clips showing product inspection/handling. The four provider-positive staged clips show pocket/torso/bag activity in sampled frames. Phone retrieval, clothing adjustment, exact event bounds, and person-level action labels remain **unverified**. Six preview frames per video are not a complete temporal review. Do not relabel all Normal clips as shopping or all visible people as concealing items.

## Executed extractor

- Official YOLO11n-pose weights: 6,255,593 bytes, with source URL and local SHA-256 in `artifacts/models/yolo11n-pose.provenance.json`.
- Ultralytics 8.4.210; ByteTrack; CPU, two Torch threads, one OpenCV thread, 416 input size, 10 analysis samples/second.
- Records keep decoded source-frame indices, frame/FPS timestamps, boxes, confidence, 17 keypoint triplets, and camera-local track IDs. Empty frames stay empty. Trackers reset at clip boundaries.
- 2,925 source frames decoded; 978 samples analyzed; 861 samples contain a pose. This includes an empty-room clip, so the all-sample pose fraction is not detector recall.

| Clip | Samples | Samples with pose | Mean inference + tracker ms |
|---|---:|---:|---:|
| normal_001 | 106 | 0 | 109.6 |
| normal_031 | 121 | 121 | 95.0 |
| normal_061 | 112 | 112 | 94.8 |
| normal_090 | 207 | 207 | 83.1 |
| shoplifting_001 | 104 | 104 | 92.2 |
| shoplifting_031 | 106 | 95 | 90.3 |
| shoplifting_063 | 108 | 108 | 90.0 |
| shoplifting_093 | 114 | 114 | 51.9 |

These are development-laptop measurements, not low-spec device certification. Pose extraction costs much more than the tiny temporal predictor. Timing excludes some decode/serialization work and includes initial warmup within clips; it is not a steady-state end-to-end FPS guarantee.

## Longer context and interaction proxies

The feature module computes confidence-masked wrist-to-hip/shoulder distances, wrist separation, and optional wrist displacement speed normalized by torso length. These are **body-motion proxies**, not verified wrist-to-product/bag interactions. Background people can contribute tracks; no primary actor was automatically given the source action label.

Track windows require adequate keypoints and no observation gap above 250 ms. Counts are person-track windows and can exceed scene-frame counts when multiple people are visible:

| Context length | Available windows |
|---|---:|
| 0.8 seconds | 925 |
| 2.4 seconds | 783 |
| 4.8 seconds | 581 |

This tests whether longer context and proxy features can be produced from actual YOLO outputs. It does **not** establish that they improve concealment detection. No classifier was trained or compared on these eight coarse-labeled clips, and no train/test split was invented.

## Reproduce locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\review_pilot_frames.py
.\.venv\Scripts\python.exe scripts\extract_video_pose.py --run mnnit_yolo11_reproduction
```

The original executed run is `artifacts/video_pose/mnnit_yolo11_v1/`; its JSONL observations, extraction report, and context report are retained. `analyze_video_context.py` currently analyzes that named reference run. `requirements-video.lock.txt` records exact installed versions. Original CPU dependency pins remain unchanged; the isolated environment adds video libraries.

## Remaining data work

The public source is sufficient to begin raw-video development without recording new footage. To carry out the requested four-action efficacy comparison, still curate/review phone-retrieval and clothing-adjustment hard negatives, annotate action intervals and relevant actors, and preserve whole actors/sessions/source groups in splits. A same-room laboratory pilot is not a final test for real shops. Restricted research datasets may support academic experiments, but must not silently become the source of distributed product weights.
