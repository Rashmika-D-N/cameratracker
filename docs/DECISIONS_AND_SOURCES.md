# Decisions and research sources

Scope updated 2026-11-17; technical source review recorded 2026-11-16. Sources support capabilities/constraints, not our unmeasured performance. Recheck release-specific requirements before packaging dependencies.

## Architecture decisions

| Decision | Rationale | Revisit when |
|---|---|---|
| Model-first development with stable contracts | Matches the owner's workflow and tests feasibility before UI investment | Contracts cannot express actual evidence |
| Possible concealment with human review first | Matches the shoplifting goal; ordinary public-shelf pickup is not suspicious by itself | Evidence shows a narrower event/camera setup is needed |
| PoseLift small-sample/schema/split audit before training | Establishes usable supervision, time/joint conventions, terms, and leakage-safe evaluation | Actual sample audit identifies a better lawful source |
| Small pose-only causal baseline before interaction features | Establishes a measurable reference without a large video model; anomaly alone is not proof of concealment | Held-out failures justify added evidence/features |
| Protected-area removal is a separate optional feature | Different event semantics and authorization rules; not a prerequisite for concealment | A shop requests and independently validates it |
| No payment/POS integration in phase one | Camera observations cannot establish payment status; limits integration scope | Explicit future requirements justify a separate design |
| Local-first dedicated device | Operates without internet and isolates shop POS workloads | Existing-PC support is explicitly required |
| Static React + FastAPI | One application stack and no Node runtime on shop devices | Actual UI/backend needs justify a change |
| SQLite on local SSD; one writer | Small local workload and fewer services | Measured contention or multi-site architecture requires PostgreSQL |
| Shared bounded inference worker | Limits model duplication, backlog, and CPU contention | Measured parallel hardware justifies added instances |
| Separate compressed evidence path | Preserve detail without full-frame recording transcode | Camera codec/stream limits require a qualified alternative |
| Model/runtime adapters | Hardware and artifact licensing remain uncertain | Interfaces create more complexity than measured alternatives need |
| Open-source distribution confirmed by owner | Practical student distribution and installation/support route | Choose exact compatible license after artifact audit |
| No automatic on-site learning | Avoid feedback contamination and unreviewed model drift | An independently validated update/review process exists |

## Licensing checkpoint

[Ultralytics](https://www.ultralytics.com/license) offers AGPL-3.0 and Enterprise paths and discusses proprietary deployments and custom-trained models. Export format does not substitute for checking upstream terms. Commercial activity is not inherently incompatible with open source; comply with the chosen terms. Do not assign the whole repository a license by inference from one dependency.

[YOLOX source](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) is Apache-2.0 and is a possible alternative to evaluate. That is not a blanket clearance of every checkpoint or training dataset. [ORT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) and [OpenVINO](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE) have separate runtime licenses; neither overrides model obligations.

[FFmpeg](https://ffmpeg.org/legal.html) licensing depends on build options/components. Record the binary's source, version, build configuration, and notices; do not bundle an unreviewed nonfree build. Final distribution needs an artifact-level checklist, not a generic statement that YOLO is free.

## Technical sources

| Source | Supports |
|---|---|
| [Frigate camera setup](https://docs.frigate.video/frigate/camera_setup/) | Separate analysis/recording settings; source FPS and resolution affect decoding cost |
| [Frigate planning](https://docs.frigate.video/frigate/planning_setup/) | Simultaneous activity and video decoding matter to capacity; not a benchmark for our behavior pipeline |
| [FFmpeg processing](https://ffmpeg.org/ffmpeg.html) | Decode/filter/encode flow and stream copy |
| [FFmpeg segment muxer](https://ffmpeg.org/ffmpeg-formats.html#segment_002c-stream_005fsegment_002c-ssegment) | Keyframe-aligned segment boundaries |
| [ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) | CPU threading/spinning/session tuning |
| [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) | Calibration and accuracy/precision tradeoffs |
| [OpenVINO releases](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html) | Runtime/ISA requirements, including AVX2 CPU support |
| [SQLite WAL](https://www.sqlite.org/wal.html) | Concurrency, local-host shared memory, durability considerations and fixed WAL-reset race |
| [SQLite backup API](https://sqlite.org/backup.html) | Safe online database snapshots |
| [systemd service](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html) | Restart and watchdog mechanisms; not automatic application correctness |
| [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) | Mixed precision and gradient accumulation |

## Related work and novelty limits

- [PoseLift, WACV Workshops 2025](https://openaccess.thecvf.com/content/WACV2025W/ASTAD/html/Rashvand_Exploring_Pose-Based_Anomaly_Detection_for_Retail_Security_A_Real-World_Shoplifting_WACVW_2025_paper.html): real-world pose-based dataset and benchmarks.
- [PoseLift author repository and data instructions](https://github.com/TeCSAR-UNCC/PoseLift): starting point for the small-sample schema/split/provenance audit. Released pose data does not establish product visibility or raw-camera pipeline capacity; inspect the actual files before selecting a training objective.
- [Shopformer, CVPR Workshops 2025](https://openaccess.thecvf.com/content/CVPR2025W/ABAW/html/Rashvand_Shopformer_Transformer-Based_Framework_for_Detecting_Shoplifting_via_Human_Pose_CVPRW_2025_paper.html): pose-sequence graph/transformer method. Pose plus transformer is existing work.
- [RetailS / periodic adaptation, 2026](https://arxiv.org/abs/2603.04723): streaming adaptation and deployment-focused evaluation; adaptive updating alone is not a new claim.
- [Paza, April 2026 preprint](https://arxiv.org/abs/2604.14846): detector/pose filters with conditional VLM analysis. A VLM addition alone is not novelty, and this preprint does not establish our desired low-spec reliability.
- [UCF-Crime original paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html): weak video-level training supervision.

Our proposed contribution requires evidence: measure false-alert reduction for possible concealment and observed coverage at a fixed CPU budget, comparing a pose-only causal baseline against interaction/visibility evidence and then adaptive cadence. Include ordinary-action hard negatives and misses introduced by the real detection/pose/candidate cascade. Report the temporal-model-only benchmark separately from complete camera-pipeline performance; low-spec feasibility remains unproven. A broader literature review is required before claiming academic originality.

Source publication update 2026-11-17: the original project source is now licensed under GNU AGPL-3.0 for the open-source YOLO integration. External data/model artifact terms and deployment clearance remain separate; see the root LICENSE and THIRD_PARTY_NOTICES.md. Earlier undecided-license notes describe the planning stage.
