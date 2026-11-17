# PoseLift download report

Completed: 2026-11-17. All selected files downloaded; basic integrity checks passed. No model trained.

## Selected release

**PoseLift, official processed JSON pose release**, from the Google Drive folder linked by the [publisher repository](https://github.com/TeCSAR-UNCC/PoseLift).

Source folder: [PoseLift on Google Drive](https://drive.google.com/drive/folders/1aEkENZlVE4ZvF_BZXV1VJOwuiXQq6trn).

The publisher's Instructions.pdf recommends JSON for anomaly-model training. It organizes observations by person ID and frame number. We selected JSON train/test files and NumPy test masks, rather than duplicate pickle data or unrelated datasets. This is pose/behavior data, not raw CCTV video or product-image detection data.

## Exact downloaded payload

| Part | Files | Bytes | Decimal MB |
|---|---:|---:|---:|
| Training poses | 104 | 43,237,851 | 43.238 |
| Test poses | 47 | 2,485,342 | 2.485 |
| Test frame masks | 47 | 44,784 | 0.045 |
| **Data subtotal** | **198** | **45,767,977** | **45.768** |
| Instructions.pdf | 1 | 63,490 | 0.063 |
| Publisher README and LICENSE | 2 | 18,608 | 0.019 |
| **Total downloaded payload** | **201** | **45,850,075** | **45.850** |

Data subtotal: 43.648 MiB. Filesystem allocation, local manifests, and scratch listings add a small amount beyond payload size. No compressed archive needs extraction.

## Local layout

```text
data/poselift/
  original/
    pose/train/              104 JSON files
    pose/test/                47 JSON files
    gt/test_frame_mask/       47 NPY arrays
    documentation/           instructions, upstream README, upstream license
  download_manifest.json     public source references, sizes, SHA-256 hashes
  download_audit.json        basic validation results and sample structure
```

`data/` is ignored by Git. Original bytes are preserved. No third-party training code or pickle data was executed. The upstream README and Apache-2.0 license are retained for provenance; this does not by itself establish every externally hosted artifact's reuse terms for final distribution.

## Verification completed

- Confirmed all 104 training names against the complete public embedded folder listing; the connector's first 100 results were not treated as complete.
- Checked expected byte sizes and computed SHA-256 checksums for all 201 files. Independently reread and checked local manifest hashes. These are local hashes, not publisher-provided authenticity checksums.
- Parsed all 151 pose JSON files; all were dictionaries and none were empty.
- Loaded all 47 NumPy masks with `allow_pickle=False`; all were one-dimensional binary arrays.
- Matched every test JSON source name to a test mask. No source ID appears in both released train and test directories.
- Confirmed PDF header and retained repository license text. No unfinished `.part` files remain.

Sample structure: person ID -> frame number -> `keypoints` and `scores`. Inspected train/test samples contain 51 keypoint values per observation, consistent with 17 x/y/confidence triplets. An example mask is int64 with values 0 and 1. Comprehensive joint, frame-alignment, confidence, and person-label validation remains pending.

## Limits and next work

The actual processed release has 104 train and 47 test JSON files. Published repository/paper instance counts differ, so this report records observed files rather than claiming a byte-for-byte release of every source video.

The instructions describe an unsupervised training approach. Before fitting, confirm normal-only training suitability, label polarity, frame-index mapping, missing keypoints, and scene-versus-person annotation granularity. Disjoint filenames alone do not rule out content/session leakage. Keep official evaluation data out of fitting and threshold tuning.

Next: create a project-local environment, audit the data semantics, and run a small causal pose baseline. Later validate the actual YOLO/pose extractor on recorded video, including ordinary shopping and ambiguous actions. File validation does not establish shoplifting-detection accuracy.
