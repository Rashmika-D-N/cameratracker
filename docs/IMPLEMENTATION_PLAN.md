# Implementation plan

Status: proposed work order, updated 2026-11-17. Primary scope: possible product concealment with human review. Ordinary public-shelf pickup is normal shopping. Protected-area removal is a separate optional later feature; payment/POS integration is outside phase one. No production readiness or completion dates are claimed.

## Milestone 0: audit a small PoseLift sample before training

- Inspect a small PoseLift sample from the author's release before a full download or training run. Record source/version, terms, hashes, and a schema/split inventory; inspect research pickle files only from a trusted source in an isolated development environment.
- Verify joint definitions, coordinates/confidences, person IDs, time/frame-rate assumptions, missing poses, frame labels, and official train/test membership. Do not assume reported file counts or supervision are sufficient without inspecting files.
- Establish whether training data contains labeled positive events or only normal sequences. Keep official tests held out; select the objective and validation construction accordingly.
- Define possible concealment boundaries, uncertainty, and ordinary actions that must not be treated as theft. Document where an anomaly score lacks evidence of concealment.
- Audit candidate detector/pose code and weights, dataset reuse/redistribution rights, and later FFmpeg builds. Open-source distribution is confirmed; the compatible repository license remains pending this audit.
- Inventory laptop RAM/storage/driver/runtime compatibility and a future one-camera CPU workload. Do not install multiple ML stacks before selecting a tested environment.

Done when: a small-data audit, usable sample schema, leakage-safe split plan, lawful candidate artifact route, and baseline objective exist. This planning update does not download data or install dependencies.

## Milestone 1: pose-only causal baseline

- Create a small Python package with audited pose-file replay, canonical timestamped pose/visibility contracts, grouped splits, event matching, and reproducible run configuration.
- Train a compact causal temporal baseline using only pose/confidence/timing features. Use labeled supervised training only where the audited training data supports it; otherwise start with a normal-only causal prediction/anomaly baseline. Do not fit or tune with official test positives.
- Do not use future frames at alert time. Preserve missing-observation masks and realistic gaps; reset or abstain on track switches and long gaps.
- Emit timestamped review candidates, scores/reasons, uncertainty, and evaluation output. A pose anomaly alone is an experimental review candidate, not proof of concealment or theft.
- Add meaningful checks for split leakage, causal windows, missing data, event matching, and duplicate alerts. Establish temporal-model CPU inference and memory measurements separately from full camera-pipeline capacity.

Done when: held-out pose sequences produce inspectable causal scores/candidates and reproducible event results, with a clearly labeled temporal-model CPU benchmark. No web application is necessary, and raw-camera detection remains unvalidated.

## Milestone 2: real-video concealment pipeline and hard negatives

- Collect consented staged recordings with ordinary shelf pickup, basket placement, phone retrieval, clothing adjustment, item returns, bags carried normally, obstructions, and staged concealment. Include long uninterrupted normal activity.
- Annotate action intervals, uncertainty/visibility, source/session provenance, and hard negatives; freeze recording/session-based splits before extracting windows.
- Add recorded-video decoding, licensed detector/pose extraction, per-camera tracking, camera-session timestamps, and coordinate transforms. Test the actual deployed pose extractor rather than assuming parity with pre-extracted public poses.
- Compare the pose-only baseline against interaction and visibility features under the same data split and compute budget. Fine-tune detection/pose only where observed failures justify it.
- Track runs, model/config hashes, and hard negative cases. Select thresholds on validation data; keep human review outcome separate from model claims.

Done when: an unseen video produces inspectable possible-concealment review events and held-out event-quality results, with known camera/data failures. No claim of theft certainty or payment verification is made.

## Milestone 3: CPU export and efficiency

- Export the selected model, establish parity, and benchmark ONNX Runtime CPU.
- Compare OpenVINO and INT8 on supported hardware; preserve the best accuracy/capacity option rather than selecting by model file size.
- Add fixed/bounded latest-frame handling, fair scheduling, timing masks, and measured thread budgets.
- Measure the complete decode/detection/tracking/pose/temporal pipeline, not just the small temporal model. Compare actual sampling gaps and cascade-trigger misses.
- Define a versioned model bundle and initial one-camera concealment profile. Resolution/cadence must preserve the relevant movements; low resolution is not automatically adequate.

Done when: a CUDA-free deployment pipeline meets provisional budgets on the representative low-spec device, with reproducible quality results.

## Milestone 4: headless live edge service

- Add RTSP ingestion, compressed evidence segments, pins, event spool, checksums, and recovery/reconciliation.
- Supervise decoder/inference children independently; report capture/analysis/recording progress.
- Add quotas and fault-injection coverage before continuous shop operation.

Done when: live events preserve evidence and failures are visible; the 24-hour and targeted recovery checks pass.

## Milestone 5: full stack incident review

- FastAPI control API, serialized SQLite writer/migrations, users/roles, cameras/zones, versioned config.
- Static React/TypeScript UI: commissioning, health overview, incident list, evidence timeline/player, review outcome/notes, basic reports.
- Durable metadata ingestion, evidence readiness updates, SSE reconnect/resync, protected clip playback.
- Installer role and owner/reviewer access; no mandatory cloud accounts.

Done when: setup -> live event -> clip -> review -> audit/report works end to end, including duplicate delivery and service restart. Do not treat manually seeded demo incidents as model integration.

## Milestone 6: installer and consenting shop pilot

- Native Linux service bundle with pinned dependencies, checksums/signature, license notices, model manifest, and offline installation route.
- Commissioning wizard checks camera access, body/hand visibility and interaction-region quality, timestamp alignment, inference capacity, disk budget, credentials, and health.
- Backup/restore, update compatibility, support bundle redaction, 72-hour stress and seven-day profile soak.
- Run one real-shop pilot; tune from validation/pilot feedback while preserving a separate final test.

Done when: installation and recovery are repeatable, measured operating limits are published, and the owner can review incidents without developer intervention. Expand camera count only through new qualification.

## Milestone 7: optional protected-area removal

- Add this separately only if a shop needs alerts for a configured protected stock area with explicit authorization rules. Public-shelf pickup remains ordinary shopping.
- Implement occupied/empty/unknown/unobservable slot evidence, validated persistence, return/rearm behavior, and scene/reference invalidation.
- Collect separate occupancy/removal ground truth and measure its false alerts independently from concealment.
- Benchmark combined operation before advertising both features on the same low-spec profile. An optional lightweight removal mode is not evidence that concealment fits that device.

Done when: the optional feature has its own event definition, held-out results, and hardware profile. It is not a prerequisite for the primary concealment release.

## Proposed repository layout when implementation starts

```text
apps/web/                    static React application
src/cameratracker/
  contracts/                 observations, config, event/evidence versions
  core/                      causal temporal features, event policy, optional slot states
  sources/                   audited pose replay, recorded-video and RTSP adapters
  inference/                 preprocessing, detector/runtime adapters
  tracking/                  timestamp-aware per-camera tracks
  edge/                      supervision, scheduling, evidence, spool
  api/                       routes, auth, repositories, migrations
  evaluation/                event matching, metrics, benchmarks
training/                    development-only training/export tooling
tests/                       core, integration, replay and fault tests
deploy/                      pinned profiles, service/install/update tooling
configs/examples/            non-secret examples only
docs/                        architecture and operational decisions
```

Later packages may separate these modules, but do not create empty services/frameworks before the pose baseline and real-video pipeline need them.

## Open decisions and defaults

| Question | Working default | Resolve by |
|---|---|---|
| Exact deployment CPU/RAM/OS | Dedicated supported Linux x86 machine; candidate four cores/8 GB/SSD | Before hardware certification |
| Camera count | One certified camera first | Measured profile admission |
| Camera brand/NVR and source access | Standard tested RTSP; ONVIF discovery later | First capture test |
| Project/license/business model | Open source confirmed; installation/support are potential services; exact compatible license still pending | Artifact/license audit |
| Public pose baseline | Small PoseLift sample/schema/split audit first; objective follows actual training supervision | Before training |
| Detector/pose checkpoint | Small model chosen by legality, action visibility, event quality, CPU performance | Before real-video integration |
| Concealment event and camera angle | Observable staged action, ordinary-action hard negatives, human review | Before annotation |
| Protected-area removal | Optional later feature with separate shop rules/data/profile | Only when requested |
| Retention and notification needs | Quota-bounded local storage; in-app alerts first | Pilot commissioning |
| Windows production installer | Later profile if needed | After Linux reference and demand validation |
| Multi-site cloud dashboard | Deferred; never required for local detection | After one-site reliability |

Source publication update 2026-11-17: the original project source is now licensed under GNU AGPL-3.0 for the open-source YOLO integration. External data/model artifact terms and deployment clearance remain separate; see the root LICENSE and THIRD_PARTY_NOTICES.md. Earlier undecided-license notes describe the planning stage.
