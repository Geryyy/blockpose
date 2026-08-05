<!-- provenance: commit b9c1b25 -->
# Region instance-split probe (Phase 1a, zero-retraining)

**Question:** how much of `dino_proposer`'s region merging (see `experiments/dino_proposer_pooled_ablation.md`, 'The real bottleneck is instance separation, not detection') is a weak neck in the predicted probability map -- splittable with no retraining -- versus genuine fusion that needs the Phase 1b seam-channel retrain?

**Method:** compute `DinoRegionProposerHead.predict_proposal_map` ONCE per snapshot over the 25 snapshots in `annotated_snapshots_seibersdorf.yaml` (checkpoint `/workspace/models/production/dino_proposer`), then evaluate several purely-CPU post-processing strategies against that same map: the shipped connected-component config, a foreground-threshold sweep (0.5, 0.6, 0.7, 0.8, 0.9), and a distance-transform marker watershed on the 0.5 foreground (`cv2.watershed`, scikit-image is not installed in this environment, markers = local maxima at least 20px apart and 3px from the nearest background pixel). Ground truth is `block_silhouette_mask` (`experiments/rgb_region_prior/export.py`) with `occluded_visibilities=()`, i.e. every approved block including `heavy_occlusion`/`unknown`.

**This does not measure downstream block recall.** No fitter, no pooling -- coverage + instance purity against the dense map alone is the Phase 1a gate (`docs/production-readiness-plan.md`), and it is far cheaper than the composed pipeline in `tools/eval_dino_proposer_pooled.py`. A strategy that improves purity without losing coverage is what would be re-measured downstream next; none of the numbers below are a substitute for that re-measurement.

## Results

`Mean cov.` is the mean fraction of a COVERED block's silhouette that its region actually contains, and `Empty regions` counts regions that are no block's best region. Both are here to keep `Alone` honest: a strategy that shatters one block into fragments scores it `Alone` (no OTHER block shares the fragment) while having destroyed the region rather than separated it. Read `Alone` only together with those two columns.

| Strategy | Regions (total, 25 snapshots) | Empty regions | Coverage >=30% | Mean cov. | Alone | Shared |
|---|---:|---:|---:|---:|---:|---:|
| baseline (shipped config) | 35 | 0 | 62/62 (100.0%) | 1.000 | 17/62 (27.4%) | 45/62 (72.6%) |
| threshold=0.5 | 35 | 0 | 62/62 (100.0%) | 1.000 | 17/62 (27.4%) | 45/62 (72.6%) |
| threshold=0.6 | 35 | 0 | 62/62 (100.0%) | 0.999 | 17/62 (27.4%) | 45/62 (72.6%) |
| threshold=0.7 | 35 | 0 | 62/62 (100.0%) | 0.995 | 17/62 (27.4%) | 45/62 (72.6%) |
| threshold=0.8 | 35 | 0 | 62/62 (100.0%) | 0.979 | 17/62 (27.4%) | 45/62 (72.6%) |
| threshold=0.9 | 35 | 0 | 62/62 (100.0%) | 0.937 | 17/62 (27.4%) | 45/62 (72.6%) |
| watershed | 86 | 42 | 61/62 (98.4%) | 0.789 | 28/61 (45.9%) | 33/61 (54.1%) |

## Per visibility class

### baseline (shipped config)

| Visibility | Total | Covered >=30% | Mean cov. | Alone | Shared |
|---|---:|---:|---:|---:|---:|
| full | 23 | 23/23 (100.0%) | 1.000 | 12/23 (52.2%) | 11/23 (47.8%) |
| partial | 24 | 24/24 (100.0%) | 1.000 | 4/24 (16.7%) | 20/24 (83.3%) |
| heavy_occlusion | 10 | 10/10 (100.0%) | 1.000 | 1/10 (10.0%) | 9/10 (90.0%) |
| unknown | 5 | 5/5 (100.0%) | 1.000 | 0/5 (0.0%) | 5/5 (100.0%) |

### threshold=0.5

| Visibility | Total | Covered >=30% | Mean cov. | Alone | Shared |
|---|---:|---:|---:|---:|---:|
| full | 23 | 23/23 (100.0%) | 1.000 | 12/23 (52.2%) | 11/23 (47.8%) |
| partial | 24 | 24/24 (100.0%) | 1.000 | 4/24 (16.7%) | 20/24 (83.3%) |
| heavy_occlusion | 10 | 10/10 (100.0%) | 1.000 | 1/10 (10.0%) | 9/10 (90.0%) |
| unknown | 5 | 5/5 (100.0%) | 1.000 | 0/5 (0.0%) | 5/5 (100.0%) |

### threshold=0.6

| Visibility | Total | Covered >=30% | Mean cov. | Alone | Shared |
|---|---:|---:|---:|---:|---:|
| full | 23 | 23/23 (100.0%) | 1.000 | 12/23 (52.2%) | 11/23 (47.8%) |
| partial | 24 | 24/24 (100.0%) | 0.999 | 4/24 (16.7%) | 20/24 (83.3%) |
| heavy_occlusion | 10 | 10/10 (100.0%) | 1.000 | 1/10 (10.0%) | 9/10 (90.0%) |
| unknown | 5 | 5/5 (100.0%) | 0.998 | 0/5 (0.0%) | 5/5 (100.0%) |

### threshold=0.7

| Visibility | Total | Covered >=30% | Mean cov. | Alone | Shared |
|---|---:|---:|---:|---:|---:|
| full | 23 | 23/23 (100.0%) | 0.997 | 12/23 (52.2%) | 11/23 (47.8%) |
| partial | 24 | 24/24 (100.0%) | 0.992 | 4/24 (16.7%) | 20/24 (83.3%) |
| heavy_occlusion | 10 | 10/10 (100.0%) | 0.998 | 1/10 (10.0%) | 9/10 (90.0%) |
| unknown | 5 | 5/5 (100.0%) | 0.991 | 0/5 (0.0%) | 5/5 (100.0%) |

### threshold=0.8

| Visibility | Total | Covered >=30% | Mean cov. | Alone | Shared |
|---|---:|---:|---:|---:|---:|
| full | 23 | 23/23 (100.0%) | 0.983 | 12/23 (52.2%) | 11/23 (47.8%) |
| partial | 24 | 24/24 (100.0%) | 0.972 | 4/24 (16.7%) | 20/24 (83.3%) |
| heavy_occlusion | 10 | 10/10 (100.0%) | 0.991 | 1/10 (10.0%) | 9/10 (90.0%) |
| unknown | 5 | 5/5 (100.0%) | 0.970 | 0/5 (0.0%) | 5/5 (100.0%) |

### threshold=0.9

| Visibility | Total | Covered >=30% | Mean cov. | Alone | Shared |
|---|---:|---:|---:|---:|---:|
| full | 23 | 23/23 (100.0%) | 0.944 | 12/23 (52.2%) | 11/23 (47.8%) |
| partial | 24 | 24/24 (100.0%) | 0.925 | 4/24 (16.7%) | 20/24 (83.3%) |
| heavy_occlusion | 10 | 10/10 (100.0%) | 0.958 | 1/10 (10.0%) | 9/10 (90.0%) |
| unknown | 5 | 5/5 (100.0%) | 0.917 | 0/5 (0.0%) | 5/5 (100.0%) |

### watershed

| Visibility | Total | Covered >=30% | Mean cov. | Alone | Shared |
|---|---:|---:|---:|---:|---:|
| full | 23 | 22/23 (95.7%) | 0.836 | 12/22 (54.5%) | 10/22 (45.5%) |
| partial | 24 | 24/24 (100.0%) | 0.858 | 9/24 (37.5%) | 15/24 (62.5%) |
| heavy_occlusion | 10 | 10/10 (100.0%) | 0.564 | 5/10 (50.0%) | 5/10 (50.0%) |
| unknown | 5 | 5/5 (100.0%) | 0.694 | 2/5 (40.0%) | 3/5 (60.0%) |

## Verdict

**Purity gains that do not survive inspection, reported rather than buried:** a strategy is disqualified if it loses blocks the baseline already finds, or if the blocks it does keep are no longer whole. The second rule matters more than it looks: `Alone` counts a block as separated when no OTHER block shares its region, which a strategy can achieve just as well by chopping the block into pieces as by finding the seam next to it. The mean-coverage and empty-region columns are what tell those two apart:

- `watershed`: alone 17 -> 28 (+11), but coverage 62 -> 61 of 62 (-1 block(s) lost), mean coverage of a covered block 1.000 -> 0.789, and 42 of its 86 regions cover no block at all (baseline: 0 of 35). The purity gain is predominantly fragmentation, not separation.

No zero-retraining strategy improves instance purity over the baseline without losing coverage or shredding the blocks it keeps. This is a negative result, not a null one: the merging in the probability map is not a splittable weak neck at this operating point -- aggressive thresholds leave the merged blob whole (no split found at all, at any threshold up to 0.9), and watershed splits the blob without regard for where the block boundaries actually are. The caveated strategies above are the strongest form of this result, not an asterisk on it. Watershed does move the `Alone` number, but it moves it by cutting the foreground up, not by finding seams: it emits far more regions than there are blocks, a large share of them covering no block at all, and the blocks it still covers are held only in part. In other words, the shape of the foreground probability alone does not tell a splitter where a seam is -- given only that shape, a splitter cannot distinguish the waist between two touching blocks from the middle of one block. That is precisely the information a supervised seam channel supplies, and precisely why no amount of post-processing tuning substitutes for it. **This justifies the Phase 1b seam-channel retrain** (`docs/production-readiness-plan.md`); the head must be given a supervised signal for the seam, since none survives in its current union-mask training target.
