# Next step: data and environment feasibility

Scope confirmed: possible concealment for human review. Public-shelf pickup is ordinary shopping. Protected-area removal is separate optional work.

## Immediate task

Audit a **small PoseLift sample and its official training/evaluation split** before a training job or application implementation. Do not download every candidate dataset or train on held-out positives.

Record:

1. Actual release location, version, file inventory, access conditions, and license/notice files. The GitHub license alone may not describe every externally hosted artifact.
2. File structure, joint ordering, coordinate units, keypoint confidences, frame timing, person IDs, boxes, missing observations, and label structure.
3. Which files are training/validation/test; whether training contains normal-only examples or usable positive labels. Identify source/session overlap and preserve the official test.
4. Whether anomaly labels apply to the full frame, one person, or one action. Do not assign every person's track a positive label because a frame is anomalous.
5. One normal and one labeled anomalous evaluation sequence, with a skeleton-only preview if permitted. Hold evaluation examples out from fitting.
6. Missing/corrupt/empty tracks, camera coverage, class imbalance, duplicate windows, and alignment between pose frame indices and labels.
7. Choose a tiny normality/prediction model if training is normal-only; compare a supervised small causal TCN only if appropriate labeled training data actually exists. Pose-only anomalies are research scores, not established theft decisions.

Deserializing third-party pickle files can execute code. Inspect provenance, validate archives, and use an isolated conversion process before producing a restricted numeric training format. Do not accept arbitrary pickle uploads in the application.

## Development environment

Find the installed Python interpreter, project disk capacity, RAM, NVIDIA driver, and CUDA-enabled PyTorch availability. Use a project-local virtual environment with pinned tested versions after this inspection. The provided laptop has an i5 13th generation and RTX 2050 with 4 GB VRAM; actual usable memory and compatibility still need checking.

Read-only preflight on 2026-11-17 found CPython 3.10.11 x64 and PyTorch 2.9.1+cpu. The global PyTorch cannot use CUDA. The RTX 2050 has 4096 MiB VRAM and driver 581.57. Physical RAM is about 15.73 GiB, with about 2.09 GiB available at inspection; free memory is a changing snapshot. The project drive had about 14.20 GiB free. The PATH Python entries are WindowsApps aliases; the actual interpreter must be selected explicitly when creating the environment.

Start with a small audited pose sample and a project-local environment. A tiny feature-only baseline may run on CPU; install a compatible CUDA-enabled PyTorch build in that isolated environment before GPU training is needed. Do not modify the global installation. Account for package caches, environment size, extracted data, and checkpoints before large downloads. No environment has been created and no package installed during this preflight.

## First research deliverable

`held-out pose sequences -> causal temporal scores + quality masks -> evaluation report`

Measure errors and timing, and save the complete model/config/data provenance. Then integrate YOLO/pose extraction on recorded footage:

`video -> YOLO/pose + tracking -> the same temporal input contract -> review candidates + evidence`

Because public poses were extracted with a particular model, evaluate extractor mismatch on our actual video pipeline. Test ordinary pickup/carrying, basket placement, phone retrieval, clothing adjustment, legitimate bag use, occlusion, and multiple people. Public pose data does not supply missing product imagery.

## Done criteria

- Data sample contents and split/label meaning are verified, with unknowns documented.
- A runnable isolated environment and one-batch smoke test are available.
- A leakage-safe baseline objective/input contract is selected.
- The initial baseline can be reproduced; its metrics are clearly separated from full-camera/shop performance.

Update 2026-11-17: the complete selected PoseLift processed JSON train/test plus test masks have been downloaded (198 data files, 45,767,977 bytes), along with three documentation files. Basic sizes/hashes, JSON syntax, binary NPY masks, and matching test filenames passed. See [DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md). Annotation alignment/split semantics, environment creation, and model training remain pending.

Progress 2026-11-17: gap-aware preprocessing and deterministic source-file validation grouping are implemented. The numeric audit identifies missing test coverage and non-finite poses; see [DATA_AUDIT.md](DATA_AUDIT.md). Three preprocessing regression tests pass. Full model fitting and video-extractor evaluation remain separate next milestones.

The isolated `.venv` is now installed and dependency-checked, with exact versions in `requirements-cpu.lock.txt`. The two-step CPU model smoke test passed with finite gradients and causal-prefix checks. Next is controlled baseline fitting/validation, then the actual video/pose-extractor evaluation.

Baseline v1 is complete: a frozen 15-epoch CPU experiment, checkpoint, validation-only selection, and one official-test evaluation are saved. Results are weak; see [BASELINE_RESULTS.md](BASELINE_RESULTS.md). Next is independent labeled validation data and better temporal/interaction features, not deploying this checkpoint.

Actual raw-video extraction now works on eight publicly licensed MNNIT clips. Longer-context/body-proxy availability has been measured, not detection accuracy. Phone/clothing hard negatives and temporal actor-level review remain pending; see [RAW_VIDEO_PILOT.md](RAW_VIDEO_PILOT.md).
