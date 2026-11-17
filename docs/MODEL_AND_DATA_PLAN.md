# Model and data plan

Status: proposed experiments. Train/validate the pipeline before building the full application.

## 1. First target

Start with **possible concealment for human review**, using one camera and temporal body/interaction observations. Ordinary shelf pickup, carrying an item, and placing products in baskets are normal shopping. Item disappearance is not theft. Payment and ownership are outside the first model's observable ground truth.

Compare:

1. A small pose-only causal temporal baseline on released pose data, selecting its learning objective after inspecting official splits/labels.
2. The same baseline with deployed-extractor pose data and visibility/interaction context, only where those observations actually exist.

YOLO supplies live detections/keypoints, not a theft decision. Compare a small combined person/pose model against detector-plus-crop inference on target hardware. Optional restricted-area removal can be implemented later as a separately evaluated occupancy/state feature; it is not the primary shoplifting target.

## 2. Public data: use it for the right task

| Source | Useful for | Important limitation |
|---|---|---|
| [PoseLift](https://github.com/TeCSAR-UNCC/PoseLift) | Pose sequence behavior baselines; frame-level anomaly labels | Released poses/boxes/IDs rather than raw videos; cannot train a product-image detector from those files |
| [RetailS](https://github.com/TeCSAR-UNCC/RetailS) | Normal-behavior learning, real/staged tests, multiple cameras | Released pose representation; normal-only training split; not a labeled product-removal image dataset |
| [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) | Video anomaly experiments and external robustness checks | Original training labels are video-level; not every frame of a positive video is positive; not protected-slot ground truth |
| [Zenodo event videos](https://zenodo.org/records/10149996) | Small video-based behavior experiments after inspection | Published counts are inconsistent; audit files, origin, duplicates, labels, and terms |

Inspect actual downloaded files and licenses before selecting a training source. A repository's code license does not establish commercial permission for all footage/weights. Do not redistribute research footage as part of the shop installer.

Audit a small PoseLift sample and official splits first, then use compatible public pose data for the first concealment/anomaly baseline. Frame-level scene labels may not identify which tracked person is anomalous: check label granularity before assigning targets to every track. If training is normal-only, use a normality model and keep official test positives held out. Conversion between different pose estimators can create distribution shift even with the same joint layout; re-evaluate on the deployed extractor.

## 3. Collection and annotation protocol

Create a recording plan before training. Record multiple people, sessions, lighting conditions, item positions, partial occlusions, and at least two camera angles if available. Use consenting participants and staged events. Keep camera/session/actor identifiers for grouped splits.

Release A scenarios:

- Ordinary shelf pickup, inspecting/carrying products, returning products, and putting products into a basket.
- Retrieving a phone/wallet, adjusting clothing, bending, and carrying or opening bags normally.
- Legitimate personal-bag use and actions whose interpretation is ambiguous from video.
- Staged apparent concealment under clothing or into a bag, with visible preceding/following context and explicit staged-action labels.
- Long obstruction, glare/shadows, camera movement, stream disconnect, and actions with insufficient evidence.
- Multiple people near the zone, track switches, and ambiguous action-to-person association.

Optional later protected-slot footage adds occupied/empty/unobservable states, removal/return, and rearming after a reference change. Keep its task labels separate. Do not use appearance/demographic traits as suspiciousness features.

Annotations include source/session provenance, actor/track association where available, boxes/keypoints where needed, visibility, interaction start/end, review-event definition, confidence/ambiguity, and reasons. Label insufficient evidence explicitly rather than forcing every example into theft/non-theft. Use a second reviewer on ambiguous event boundaries and document disagreements.

Use an initial coverage inventory, not an arbitrary video-count promise. A small feasibility set can reveal camera/data failures but cannot justify commercial accuracy claims. Add long uninterrupted normal recordings to measure realistic false-alert rates.

## 4. Splits and leakage prevention

- Group by source recording/session before extracting frames or temporal windows.
- Keep actor overlap separate where feasible; reserve a camera/day or shop for a generalization test.
- Keep near-duplicates, crops, and augmentations in the same split as their original.
- Use training data for fitting, a training-only subset for INT8 calibration, validation for thresholds/hyperparameters, and untouched tests for final reporting.
- Preserve official public benchmark splits for comparisons. Avoid adapting on the held-out future stream without an explicitly separate online-learning experiment.
- Report each dataset separately; a mixed headline accuracy can conceal scene bias.

## 5. Laptop training strategy

Development hardware provided: 13th-gen i5 and RTX 2050 with 4 GB VRAM. Laptop RAM, storage, driver compatibility, and exact CPU remain unmeasured.

1. Run a device/environment check and one-batch smoke test before a long job.
2. Start from a legally usable small pretrained model instead of training a large detector from scratch.
3. Try batch size 1-4 and 320/416/640 inputs independently; reduce when VRAM is insufficient. These are experiment candidates, not guaranteed fits.
4. Use mixed precision where stable, optionally freeze early layers, and accumulate gradients when needed. Accumulation does not eliminate each microbatch's activation-memory requirement or reproduce large-batch batch-normalization statistics.
5. Keep data loaders bounded and avoid full-dataset RAM/VRAM caching. Save resumable checkpoints and logs; monitor thermal throttling.
6. Extract pose/box features once for temporal experiments, with extractor/preprocessing hashes. Train a small causal TCN first; compare a compact GRU only if justified. Avoid a large video transformer as the baseline.

[PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) document autocasting, scaling, and accumulation. Pin tested versions after the smoke test rather than selecting versions from memory.

## 6. Primary temporal model

Inputs: normalized keypoints, confidence/missing masks, wrist-to-zone/bag relations, optional item observations, track quality, and elapsed time. Use a sliding causal window available at alert time; no future frames during online evaluation.

Compare fixed cadence with the actual deployment's dropped frames and adaptive cadence. Train with realistic timing/gap augmentation; masks/interpolation must not fabricate a visible action. Reset or abstain across long gaps/track switches. Evaluate person-only, pose-only, and interaction-enhanced methods as separate ablations.

For normal-only public training, start with a small causal prediction/reconstruction model. For genuinely available person-level labeled training sequences, compare a supervised small causal TCN. Do not infer which regime is supported from a paper's dataset title. Keep official tests untouched and evaluate threshold selection without test leakage.

Do not require training an additional visual model until the pose baseline shows what fails. A rule/state-machine improvement may be better than additional ML.

## 7. Export and model bundle

Validate the research model, export FP32 ONNX, check output/preprocessing parity, then benchmark CPU end to end. Test OpenVINO and calibrated INT8 as independent candidates. Keep the FP32 fallback when it meets capacity; otherwise lower the certified profile, not the validation standard.

Each deployable bundle must include:

- Schema version, immutable model ID, artifact hashes, and compatible runtime versions.
- Input dimensions/color order/letterboxing/normalization; output decoding/NMS rules.
- Class/joint definitions, coordinate conventions, temporal sample policy and window.
- Model/config thresholds and required minimum observation quality.
- Code, checkpoint, dataset provenance, licenses, and redistribution decision.
- Event-quality metrics, quantization comparison, hardware profile, and known failure cases.

Keep trained binaries outside ordinary Git; use reviewed release artifacts later. Never load arbitrary customer-supplied pickle/PyTorch models in production.

## 8. Exit to application development

First, a held-out pose sequence must produce timestamped scores/quality masks and a reproducible evaluation report. This proves only the pose baseline. Before application development, an unseen recorded video must run through the selected YOLO/pose extractor and produce canonical observations, stable candidate IDs, reasons/uncertainty, and evidence references. Required checks cover temporal gaps, ordinary-shopping hard negatives, duplicate events, and export parity. The pipeline must run on CPU without CUDA; this laptop CPU result is not certification for slower shop hardware.

Only then build the dashboard around the already-defined event/evidence contract. Later live capture changes the input adapter and health behavior, not the meaning of incidents.

## 9. Follow-up dataset selection

The owner has confirmed open-source distribution. This does not remove dataset-specific reuse conditions.

No source audited so far establishes performance for the entire live camera pipeline, including ordinary shopping, ambiguous bag use, interruptions, long normal operation, and local camera conditions. Public pose datasets enable a first baseline; our intended deployment still needs consented staged footage and normal-operation recordings.

- **First concealment research:** inspect [PoseLift](https://github.com/TeCSAR-UNCC/PoseLift) files and terms first; its repository carries an Apache-2.0 license, and its release has poses/boxes/IDs and frame labels rather than raw product imagery. [RetailS](https://github.com/TeCSAR-UNCC/RetailS) is another pose source with normal-only training; do not relabel its test anomalies as training examples.
- **Optional later removal feature:** use a compatible person/item detector and our own protected-slot/state annotations. Keep public-shelf pickup separate from that restricted-area event definition.
- **Interaction supplement:** [RetailAction](https://huggingface.co/datasets/standard-cognition/RetailAction) publishes train/validation/test archives (about 31.9 GB total), with paired views and take/put/touch annotations. Clips have motion-selected frames, so live cadence requires separate validation. Its [custom license](https://huggingface.co/datasets/standard-cognition/RetailAction/blob/main/LICENSE) requires attribution and additional licensing when applicable product/service revenue exceeds USD 10,000. Do not treat it as unrestricted distribution data.
- **Other candidates:** [UR-PRD](https://github.com/chelsea9067/UR-PRD) releases skeletons and hand crops, not full scene video; its explicit reuse license remains unverified. [Mendeley shoplifting V2](https://data.mendeley.com/datasets/r3yjf35hzr/2) currently states metadata only, despite describing videos; do not count it as an available training archive.

Update 2026-11-17: the selected PoseLift processed JSON release is now local: 104 training poses, 47 test poses, and 47 binary test masks. Basic integrity/syntax checks passed; see [DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md). The next milestone is the annotation/frame-alignment/split audit in [NEXT_STEP.md](NEXT_STEP.md), followed by a collection protocol for ordinary-shopping hard negatives and staged concealment. The download does not establish that labels and temporal windows are already safe to use for training.
