# Validation and release gates

All thresholds below are proposed acceptance targets, not achieved results. Final supported profiles must publish measurements, dataset coverage, uncertainty, and exact tested versions.

The primary release detects possible concealment for review. Public-shelf pickup/carrying/basket use are ordinary shopping and must not become positive theft labels. Protected-area removal is an optional separate task.

## 1. Quality measures

Measure events, not just frame accuracy or detector mAP.

- Event precision/recall with one-to-one temporal matching. Publish the allowed matching window/overlap and event boundary definitions before final evaluation. Duplicate alerts count as extra alerts.
- False alerts per camera-hour on long normal footage. Include camera count, active hours, crowded periods, and uncertainty intervals. Zero observed false alerts is not proof of zero future false alerts.
- Detection delay from the first observable event condition; also report delay from completed action. Distinguish intentional persistence/post-roll time from processing delay.
- Evidence completeness: requested versus available context; missing/partial clip rate.
- Fraction of time coverage is healthy, degraded, or unavailable. Report conditional recall during healthy coverage and overall recall across the entire test, including outage/degraded intervals. Never remove difficult intervals silently.
- Breakdown by lighting, body/hand visibility, obstruction, camera, actor/session, ordinary shopping, phone retrieval, clothing adjustment, basket use, legitimate bag use, and ambiguous sequences.

Initial commissioning goals to validate with the first shop: event precision >=90%, recall >=85%, and <=1 false alert per 8 camera-hours on representative normal operation. These are provisional usefulness goals, not safety guarantees; the shop may require stricter alert rates. Report sample counts and confidence intervals. Change scope or hardware when necessary rather than tuning against the test set.

## 2. End-to-end performance

Record exact CPU/ISA/RAM/storage/OS, runtime, model/config hashes, power settings, stream codec/bitrate/resolution/FPS/GOP, camera count, active people, browser load, and hardware decoding availability.

Measure decode/preprocess/inference/postprocess/tracking time, achieved samples per camera, p50/p95 frame age and alert latency, sustained total system CPU, peak process-tree RSS, disk throughput/write volume, thermal behavior, spool/queue depth, dropped frames, and evidence failures. Model-only FPS is insufficient.

Illustrative budget: if detector calls cost 40 ms each, two cameras at 5 calls/s consume about 400 ms of serialized detector execution per second. This leaves the remaining budget for other work only if contention/decoding/pose costs fit. It does not imply 40% total CPU or certify that profile.

Proposed admission targets:

| Measure | Initial target to test |
|---|---|
| Latest-frame age before analysis | p95 <=500 ms in the certified active workload |
| End-to-end application overhead after event decision | p95 <=2 seconds to incident metadata availability; finalized clip waits for configured post-roll |
| Temporal cadence | At least the model's validated minimum; publish per-camera achieved cadence |
| System headroom | Sustained total CPU <=75%, no sustained swapping, >=20% physical RAM available |
| Memory budget | Explore a <=2 GB application process-tree RSS budget for the small concealment pipeline, excluding OS/browser; revise admission hardware if it cannot meet quality and memory requirements |
| Queues/storage | Within explicit quotas with no age/depth trend during soak |

Measure targets under simultaneous activity, evidence finalization, incident review, and supported viewers. If failing, disable optional features, reduce admitted cameras, or require stronger hardware. Do not certify a degraded cadence the model was never evaluated with.

The reference laptop's CPU-only run is an early regression benchmark. Borrow or obtain one representative slower shop device before publishing minimum hardware. A process thread limit or VM is useful stress testing but is not equivalent to an old CPU's decode/ISA/thermal behavior.

## 3. Test matrix

| Fault or stress | Required behavior and verification |
|---|---|
| Camera/network disconnect | Visible outage, bounded retry/backoff; new session and reset tracker after reconnect |
| Frozen stream / repeated frames | Distinguish plausible still scene from transport/PTS failure; alert on missing progress; do not rely on changing pixels alone |
| Bad timestamps / clock jump | Preserve monotonic timeouts; reset/mark ambiguous sequences and evidence alignment |
| Malformed stream / decoder crash | Restart only affected capture; other camera and API remain usable |
| Inference crash or hang | Supervisor restarts child; recording continues where possible; interrupt candidates, bound pins, reset temporal windows, and mark coverage gap; no cross-gap inference or duplicate incident |
| API unavailable | Worker uses cached config and bounded durable spool; acknowledgement retry avoids duplicate incidents |
| Worker/API killed during delivery | Published manifests replay; one incident per stable ID; pending media remains explicit |
| Sudden power loss | Check database, manifest recovery, segment validation, missing observations, and reboot startup |
| Full disk / read-only volume | Prune only eligible data; protect DB reserve; explicit evidence/storage failure; no false success |
| Crowded views / all cameras active | Fair scheduling, bounded candidate counts, explicit overload; no starved camera |
| Dashboard closed or multiple tabs | Detection independent of viewer; previews reused and viewers limited |
| Internet unavailable | Local detection/review continue; optional messages visibly queued/expired |
| Corrupt/unsupported model | Reject before activation; preserve known-good artifact; show fault |
| Configuration update | Validate/version atomically; invalid config preserves last good; zones reset affected state |
| Camera movement/zoom or reference change | Reset temporal windows, validate view/zone quality and recommission if needed; optional slots invalidate their occupancy reference |
| Evidence deletion interrupted | Sole media owner reconciles durable command/file/DB states; retry is idempotent and respects pins |
| Backup / restore | Consistent metadata/media manifest, authenticated restore, recoverable reviews/evidence states |
| Update / interrupted update | Verified artifact, recoverable service version, compatible database, tested refusal of unsafe downgrade |
| Credentials / forbidden camera URL | Denied access, redacted logs and support bundles; no arbitrary local/network fetch |

Crash/power tests cannot establish immunity to defective hardware. Document any in-flight-loss window. Test normal shutdown and actual forced interruption on disposable test data.

## 4. Model ablations

Primary release: small pose-only causal baseline; pose plus available interaction/visibility; fixed versus adaptive cadence; FP32 versus INT8. Preserve official splits and use only genuinely available training supervision. Report full-scene/person-association limitations, extractor mismatch, ordinary-shopping errors, and cascade-trigger misses, not just classifier performance on perfectly cropped positive clips.

Optional protected-area feature: pixel-change baseline; detector/occupancy plus naive disappearance; visibility-aware temporal state. Evaluate separately from concealment and never score public-shelf pickup as theft.

Success means an improvement in measured alert quality or coverage at the same device budget. A new framework name or extra network layer is not evidence of novelty.

## 5. Readiness ladder

1. **Data feasibility:** a small released pose sample/split/license audit and a leakage-safe learning objective; no training claim from metadata alone.
2. **Pose feasibility:** held-out sequence scores/quality masks exported; dataset limitations and known failure cases listed. This is not live video inference yet.
3. **Video model baseline:** actual deployed YOLO/pose extractor on recordings, grouped held-out results, reproducible environment/model, CPU export parity, and ordinary-shopping hard negatives.
4. **Edge prototype:** one camera plus evidence recovery and health visibility; >=24-hour stress run.
5. **Pilot candidate:** all relevant fault tests pass; >=72-hour simultaneous workload test.
6. **Supported shop release:** seven-day soak on each certified hardware/camera profile, security/license checklist, backup/restore/update rehearsal, and consenting real-shop pilot feedback.

Track wall-clock run length, actual analyzed coverage, faults, CPU/RAM/storage trends, and all exclusions. A seven-day powered-on machine with stalled analysis is not a successful soak.
