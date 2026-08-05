# Multi-block hypotheses: fit 1-block and N-block candidates, let evidence choose

**Result: negative. Implemented, measured, net-negative, and the implementation
was NOT merged** -- only this record was. The C++ (`tile_row_candidates` plus a
`DetectionParameters::multi_block_hypotheses` flag) is preserved on the
`feat/multi-block-hypotheses` branch. It was correct and inert by default, but
per the precedent set by commit `74b3718`, unused machinery on the path the ROS
node links invites exactly the speculative complexity this repo keeps removing.
Recover it with `git show feat/multi-block-hypotheses -- cpp/` if the coverage-
aware scoring change below is ever attempted.
Strict recall 44/62 → 42/62, centre 46/62 → 45/62, detections 59 → 66.

## The idea

`pipeline.hpp::split_connected_row` already tiles a connected component into
block-sized regions, but all-or-nothing: it requires the component's extent to
land within 0.12 m of an exact multiple of the block width **and** every
internal boundary to show an observable density valley. Both fail on touching
blocks — a seam has no valley by definition, and a partially observed row has
an arbitrary extent. When it declines, the component is fit as one region and
any neighbour is simply never proposed.

So: emit the tiling as a *competing hypothesis* instead of a decision. Fit both
the whole component and the tiles, and let the existing evidence scoring and
overlap NMS choose — the same hypothesise-and-score structure the rest of the
pipeline uses. Block dimensions are known and fixed, so the count follows from
the extent; no seam needs to be observed at all. If it worked it would also
make the `dino_proposer` seam-channel retrain unnecessary, since a merged RGB
region could be tiled the same way.

Implemented as `pipeline.hpp::tile_row_candidates` behind
`DetectionParameters::multi_block_hypotheses` (default **false**), firing only
where the conservative splitter declined, with sections divided evenly rather
than stepped by block width.

## Measured, 25 annotated snapshots / 62 approved blocks

| | centre | strict | detections |
| --- | ---: | ---: | ---: |
| off (shipped) | 46/62 | **44/62** | 59 |
| on | 45/62 | **42/62** | 66 |

Per block: centre gained 1, broke 2; **strict gained 0, broke 2**. With the
flag off the numbers reproduce the shipped baseline exactly, so the mechanism
is genuinely inert when disabled.

## Why it failed — it targets a stage that is no longer the bottleneck

Tiling changed only **two** scenes, and both got worse:

| scene | strict off → on | is a known multi-block component? |
| --- | --- | --- |
| `partially_wet` | 2 → 1 | yes |
| `static` | 3 → 2 | no |

Every one of the other six components known to contain more than one ground-
truth block (`wall`, `wall2`, `wall3`, `misaligned_stack`,
`pickup_and_stack_rotated`, `pickup_and_stack_rotated2` — see
`detector_cluster_failures.md`) was **unchanged**. They did not need it: the
`cluster_max_extent_xy` 1.5 → 2.25 change already recovered them, and the wall
class now runs at 86% strict.

The blocks that *are* still missed never reach the splitter. Per the per-stage
diagnostics (commit `00ace61`), 8 of 10 `heavy_occlusion` blocks are rejected
at the **proposal gate**, dominated by `extent_z` — the nearest component to
such a block measures a median 0.19 m in Z against a 0.30 m minimum. Tiling
operates on components that already passed that gate, so it cannot see them.
The remaining failure is *thin components rejected early*, not *wide components
merged late*.

And where it did fire it hurt, which confirms a concern raised before the
experiment: the evidence score rates each box on **its own** support
(`support_points`, covered/expected faces, supported/incident rays), not on how
much of the component is left unexplained. A half-covered tile can therefore
score perfectly well, so "let evidence choose" does not arbitrate between a
1-block and a 2-block reading as cleanly as the design assumed. The +7 extra
detections at unchanged recall are the same effect seen from the other side.

## What this does not rule out

- Tiling *inside an RGB region* rather than a PCL component. Untested; the RGB
  region is a different, tighter crop, so the count estimate is better posed.
- A coverage-aware selection term (does this hypothesis SET explain the
  component?) rather than per-hypothesis support. That is a scoring change, not
  a proposal change, and would be the prerequisite for any version of this idea
  to work.
- Relaxing the minimum-extent proposal gate *only inside a region an RGB
  proposal asserts contains a block* — this is where the occluded blocks
  actually die, and a global relaxation was already falsified (recall unchanged
  at 46/62, precision down to 74.2%/75.4%).
