# RGB priors: production-readiness plan

Written after the score-comparability audit (see
`experiments/dino_proposer_pooled_ablation.md` for the measurements this plan
is built on). Scope: getting `dino_proposer`/`dino_verifier` from "trained
checkpoints with offline metrics" to "bounded evidence modules the ROS 2
detector can depend on", per `docs/dino-rgb-prior-ros-integration.md`.

Every phase below has a **falsifiable exit gate with a number**. Where a gate
cannot honestly be evaluated on the data we have, that is stated rather than
papered over.

## Where we actually are

| | measured |
| --- | --- |
| PCL geometry baseline | 46/62 centre (74.2%), **44/62 strict (71.0%)**, precision 78.0% |
| best composed config (gap-only pooling) | 48/62 centre, **45/62 strict (72.6%)**, precision 76.2% |
| oracle ceiling (crop to GT box, seeded fitter) | **55/62 (88.7%)** |
| `dino_proposer` block coverage | **62/62 (100%)**, incl. **10/10 `heavy_occlusion`** |
| `dino_verifier` within-block pose top-1 | 93.6% (held-out, 19-fold LOSO) |

The gap between 72.6% and the 88.7% oracle is almost entirely **instance
separation**: blocks alone in a predicted region are found 17/17 (100%);
blocks sharing a region 31/45, and `heavy_occlusion` in a shared region only
3/9. All 14 remaining misses sit in merged regions.

**The honest headline: the best measured gain over geometry alone is +1 block
out of 62, in-sample.** That is inside the noise of this dataset. Nothing here
is production-ready today, and the binding constraint is Phase 1 plus the
dataset problem in Phase 5 — not tuning.

## Phase 0 — correctness bugs that corrupt data or deployed behaviour

Do these first regardless of whether the RGB path ever ships. Two of them
touch ground truth or the production path.

| # | defect | location | fix |
| --- | --- | --- | --- |
| 0.1 | Production engine builds `SensorContext` from the **ROI-cropped** cloud, so ROI-restricted queries are scored inflated against a fixed `min_score=0.4`. Partially mitigated: `misses`/`miss_origins` are passed unmasked, so only through-box ray violations are lost. | `python/blockpose/engine.py:216-235` | Build the context from the full frame, pass only the masked points as the fitting input. The C++ side already takes cloud and context as independent arguments (`pipeline.hpp:196`), so the split is free. |
| 0.2 | Annotation GUI auto-promotes proposer candidates de-duplicated **only by id string**; proposer ids are namespaced (`dino_proposer:{i}:{h.id}`) so they can never collide with base ids. Measured: 43 pending targets for ~20 physical blocks, incl. three axis-swapped twins (93–95°) adopted beside the correct block, with no provenance shown to distinguish them. | `tools/annotate_snapshot.py:1091-1106`, `annotation_candidates.py:234,239` | Centre-distance de-dup against targets **already in the document**, not just the current batch; keep the whole-cloud candidate on a tie. Show provenance + source in the target label. |
| 0.3 | Supplying the verifier checkpoint makes `dino_verifier_reranked` the active base source, and **every** candidate returns `accepted=False` — auto-promotion silently drops to zero while the status label reports success. | `annotation_candidates.py:448`, `annotate_snapshot.py:1019` | Split `accepted` (= `fate == "final"`) from `evidence_available` and `rank`. Use `accepted = (fate == "final") and evidence_available` for `rgb_edge_prior` so its measured precision benefit is preserved as an intersection, then re-measure the table in `docs/annotation-tool.md`. |
| 0.4 | `tools/eval_sam_region_proposals.py` still has the uncorrected crop-context defect, so the published SAM2 pooled numbers are invalid. | `eval_sam_region_proposals.py:259,392` | Port `--score-context` from `eval_dino_proposer_pooled.py`, re-run, and replace the tables in `experiments/sam_region_proposals.md` (which already carries a staleness banner for an unrelated reason). |
| 0.5 | Proposer-region fits use default `DetectionParameters` while the whole-cloud pass uses the caller's; `ablate_rgb_proposals.py` shares the context bias; dead `new_fps` helper would index candidates by GT index. | `annotation_candidates.py:200-208`, `ablate_rgb_proposals.py:139-150`, `eval_dino_proposer_pooled.py:528` | Thread params through; same context fix; delete the dead helper. |

**Exit gate:** full suite green; `docs/annotation-tool.md`'s promotion table
re-measured; SAM2 report re-run and corrected. No accuracy claim attached.

## Phase 1 — instance separation (the actual bottleneck)

`experiments/rgb_region_prior/export.py` trains the head on a single **union**
foreground mask per snapshot, by explicit design ("is there a block anywhere in
this region", not "which specific block"), on the v1 assumption that geometry
would separate instances downstream. Geometry cannot separate touching blocks —
that is the problem the RGB path exists to solve. The design is circular and
this is where the capability is lost.

**1a — zero-retraining probe (do this first, it is cheap).** Split the existing
probability map without touching the model: raise the foreground threshold
(sweep 0.5→0.9), and/or distance-transform watershed the existing foreground.
This costs one afternoon and tells you how much of the merging is a weak neck
versus genuine fusion.

*Gate:* report instance-level region recall (one region per block) and
downstream block recall vs the current 48/62 centre / 45/62 strict. If a
threshold sweep alone recovers a meaningful share of the 14 merged-region
misses, 1b may be unnecessary.

**1b — seam-aware head (if 1a is insufficient).** Add a second output channel
predicting inter-block **seams**, then split foreground by seams (watershed).
Labels are free: `block_silhouette_mask` already renders per-block silhouettes,
so seams are the boundaries between adjacent instance masks. Same frozen
backbone, same 25 snapshots, one extra conv channel, no new annotation.

*Gate:* **held-out** (19-fold LOSO, not in-sample) instance region recall, and
downstream strict block recall. Target: close a material share of the 45/62 →
55/62 oracle gap. If held-out strict recall does not clear the 44/62 geometry
baseline by more than one block, stop and say so — that is the same
"+1 is noise" trap.

## Phase 2 — `dino_verifier` in its correct role

The verifier is trained as a **within-block pose disambiguator** — positives are
GT ±5 cm / ±8°, negatives are yaw 90°/270°, position 0.35/0.6 m, tilt 25°/45°
(`manifest.json::candidate_params`). It answers "does this projected outline
align with the block in RGB". It was never trained to compare two different
candidates, which is why the scalar-rerank usage produced a near-uniform,
non-discriminating shift.

**2a — refit calibration.** `calibration_temperature` is **12.497**;
`sigmoid(logit(raw)/T)` at that temperature compresses the entire raw range
0.01–0.99 into a contribution span of **0.073** against a ±0.20 cap. A
temperature that large on a sigmoid head is a red flag that its logits were
near-degenerate when fit. Re-fit on a held-out split and record NLL/ECE
alongside the value.

**2b — argmax disambiguator.** Per accepted candidate, generate its pose
hypothesis set (yaw 0/90/180/270 plus small translation/tilt offsets, mirroring
the training negatives), render each `render_conditioning` image, score all,
take the argmax as the corrected pose. Note this usage is **unaffected by 2a's
temperature bug** — the transform is strictly monotonic, so ranking survives
even uncalibrated; 2a matters for thresholding and abstention, not ranking.

*Gate:* strict recall on blocks that are found at centre level but fail the
rotation/IoU check (currently 3 under gap-only pooling; the pool grows as
Phase 1 lands). Must not reduce strict recall on already-correct blocks —
report gained/broken per block, not just the aggregate.

## Phase 3 — composed, held-out evaluation

- Adopt **gap-only pooling** as the policy: region candidates may only fill
  blocks geometry found nothing for. This is the only configuration measured
  net-positive (45/62, +1, zero regressions), and it makes the cross-source
  score-comparability defect structurally impossible.
- Re-run the composed ablation with **fold-held-out checkpoints** on both
  heads. Everything published so far is in-sample.

*Gate:* held-out composed strict recall vs the 44/62 geometry baseline, with a
per-block gained/broken breakdown. This is the promotion decision.

## Phase 4 — runtime readiness

Nothing here has been measured; all of it is required before a ROS node is
worth writing.

- **Fix the ROS contract first.** `docs/dino-rgb-prior-ros-integration.md`'s
  `ScoreRgbCandidates` returns `{stable_id, score, uncertainty, abstained}` — a
  scalar per candidate. That encodes the usage Phase 2 just falsified. It must
  carry a **pose hypothesis set** in the request and return per-hypothesis
  scores / a best index. Change the contract before implementing against it.
- **Latency.** The contract budgets `timeout_s: 0.20` per module. The composed
  path (dense map + region extraction + per-region projection + fit, then
  per-hypothesis verifier crops) has never been timed. Measure end-to-end p95
  against that budget; note the older `SCENE_DISCOVERY` benchmark was already
  failing its 500 ms budget.
- **nanobind Stage 1** (`docs/cpp-nanobind-consolidation.md`): custom
  `DetectionParameters` from Python is a hard prerequisite — region-scoped
  parameters cannot be passed today.
- **ONNX export + parity test.** The integration doc specifies a versioned ONNX
  artifact; we currently ship PyTorch checkpoints. Export and assert numerical
  parity against the PyTorch head before the node depends on it.
- **Graceful degradation.** Confirm the geometry-only fallback path on model
  unavailable / timeout / stale timestamp actually runs, per the contract's
  "RGB absence is not negative evidence".

## Phase 5 — the dataset problem (the real blocker)

`config/rgb_ablation_splits_seibersdorf.yaml` is marked
`selection_eligible: false` with `grouping_evidence:
scenario_label_directory_prefix_not_validated_run_ids`. There are 25 annotated
snapshots — **one per distinct scene** — and 62 GT blocks. Effects of ±1–2
blocks are inside the noise, and there is no promotion-gate set held back from
selection.

No amount of Phase 1–4 work fixes this. To make a production claim you need
**independent recording sessions** reserved as a promotion gate and never used
for selection, per the integration doc's "evaluate complete bags/capture
sessions, never random frames". Collecting them is on the critical path and
should start in parallel with Phase 1, not after Phase 4.

## Suggested ordering

```
Phase 0  (days)     -> correctness; unblocks trusting anything measured after it
Phase 1a (days)     -> cheap probe; may make 1b unnecessary
Phase 5  (parallel) -> start collecting held-out sessions NOW; long lead time
Phase 1b (weeks)    -> seam head, if 1a insufficient
Phase 2  (week)     -> verifier refit + argmax disambiguator
Phase 3  (days)     -> held-out composed ablation == promotion decision
Phase 4  (weeks)    -> contract change, latency, ONNX, nanobind Stage 1
```

**Kill criteria, stated up front:** if Phase 3's held-out composed strict recall
does not beat the 44/62 geometry baseline by more than the dataset's noise
floor, the honest outcome is that RGB priors do not earn a place on the runtime
path yet — and the correct action is to say so and keep the geometry-only
detector, not to tune until a number looks good.
