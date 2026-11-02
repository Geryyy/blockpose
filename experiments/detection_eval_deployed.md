# Deployed lidar detector on the reviewed snapshot set: 54/61 found, 98 % of observable blocks

**Result: the deployed `concrete_block_detector` finds 54/61 blocks (88.5 %) by centre and 52/61
(85.2 %) strict, at 96.4 % precision. Of the 55 blocks that are observable at all it finds 54.**
Source for Table 3 of the AiC paper (`wiki/paper_aic_draft.md` §5).

This replaces two older numbers, neither of which described the shipped detector:

- 74.2 / 71.0 / 86.8 % (25 scans / 62 blocks, `gripper_filter_in_evaluation.md`): blockpose's
  Python `_run_detection` with `DetectionParameters` defaults (min_score 0, proposal_max_components 8,
  cluster_max_extent_xy 2.25, no scene bounds, no FK prior). Never deployed. Its annotations are
  lost, so that set cannot be re-run.
- 55.7 % (`concrete_block_detector/tools/reviewed_snapshot_baseline.json`): revision 1dd50c2, both
  param hashes differ, and its `tf.yaml` files lack the rail/TCP frames, so the gripper filter and
  the FK prior were silently inert.

## Inputs

| input | value |
|---|---|
| GT | `concrete_block_detector/tools/reviewed_snapshot_manifest.json`, sha256 `1e8e45fe…646c` — 23 scans, 61 blocks, visibility labels |
| clouds, TF, runner output, stage dumps | `~/Documents/aic_det_eval/` (276 MB, outside the repo); `SHA256SUMS` here lists every file under `runs/` and `snaps/` |
| detector | `concrete_block_detector` 44f523b, `blockpose` 2ef567f |
| params | `concrete_block_detector/config/detector.yaml` sha256 `ae8cd24b…a6ec` |

The scan set's `tf.yaml` files lack the rail and TCP frames. `prep.py` writes two trees:
`snaps/orig/` as recorded, and `snaps/graft/` with `elastic/K9`, `K10_left_rail`, `K11`,
`K12_right_rail` and `K8_tool_center_point` interpolated from the two nearest
`seibersdorf_dataset` frames of the same bag (Δt ≤ 0.6 s).

## Reproduce

Scripts are in `detection_eval_deployed/`; paths default to the original host and are set by env
(`CBS_DETECTOR_TOOLS` → `concrete_block_detector/tools`, `REG_SNAPSHOTS`, `SEIBERSDORF_DATASET`).
`D=~/Documents/aic_det_eval`.

1. Runner, standalone (no colcon), from `src/concrete_block_stack`:
   ```bash
   g++ -std=c++17 -O2 -DNDEBUG -DCBP_DETECTOR_GIT_REVISION="\"44f523b0\"" \
     -Iconcrete_block_detector/include -Iblockpose/cpp/include -I/usr/include/eigen3 \
     concrete_block_detector/src/concrete_block_detector_snapshot_runner.cpp -lyaml-cpp -o runner
   ```
2. `python3 prep.py $D/snaps` — host only (reads the old blockpose clone); its output is in `$D/snaps`.
3. Runs (`C=concrete_block_detector/config`, `$ARGS_<tree>` = one `--snapshot snaps/<tree>/<snap>` per scan):
   ```bash
   ./runner --params $C/detector.yaml --params $C/grip_at_top_detector_scene_discovery.yaml $ARGS_orig > A_orig_det_overlay.json
   ./runner --params $C/detector.yaml $ARGS_orig                         > B_orig_det.json
   ./runner --params $C/detector.yaml $ARGS_graft                        > C_graft_det.json   # deployed
   ./runner --params $C/detector.yaml --no-fk-prior $ARGS_graft          > D_graft_det_nofk.json
   ./runner --params $C/detector.yaml --params nofilter.yaml $ARGS_graft > E_graft_det_nofilter.json
   ./runner --params $C/detector.yaml --dump-stages dumpC $ARGS_graft    > C2.json            # poses identical to C
   ```
4. Scoring, in `$D`:
   ```bash
   python3 analyze.py runs/C_graft_det.json dumpC C_rows.json    # likewise A, B, D, E; dumpC for all
   python3 summarize.py C_rows.json E_graft_det_nofilter_rows.json D_graft_det_nofk_rows.json B_orig_det_rows.json A_orig_det_overlay_rows.json
   python3 held.py snaps/graft C_rows.json D_graft_det_nofk_rows.json
   python3 render.py                                             # 5 miss PNGs; reads runs/C_graft_det.json, dumpC/
   ```
   Re-run on 2026-10-14 from the copied data: C gives 54/61 centre, 52/61 strict, 54/56 precision;
   held 19/19/20, without FK prior 12/10/20.

Matcher: Hungarian on centres ≤ 0.30 m; strict adds symmetric rotation error ≤ 30° (repo scorer's
`_rotation_error_deg`) and Monte-Carlo IoU ≥ 0.25. `analyze.py` also reports the repo's 0.5 m matcher.
Held = GT centre < 0.35 m from `T_world_K8_tool_center_point · (0, 0, 0.85)`.
Observable = ≥ 50 points on the top face, ≥ 1 side face, ≥ 40 % of the sensor-facing surface seen
(ray cast), centre below 1.5 m. Column meanings of `per_gt_deployed.csv` (run C, one row per GT)
are in the `analyze.py` output keys and `~/Documents/aic_det_eval/README.md`.

## Results

| run | config | centre | strict | precision |
|---|---|---|---|---|
| C | deployed (gripper filter + FK prior) | **54/61 (88.5 %)** | **52/61 (85.2 %)** | **54/56 (96.4 %)** |
| E | deployed, gripper filter off | 50 (82.0 %) | 48 (78.7 %) | 94.3 % |
| D | deployed, no FK prior | 47 (77.0 %) | 42 (68.9 %) | 95.9 % |
| B | `detector.yaml`, no rail TF | 46 (75.4 %) | 41 (67.2 %) | 93.9 % |
| A | + grip_at_top overlay, no rail TF | 46 (75.4 %) | 40 (65.6 %) | 93.9 % |

Run C by subgroup (centre / strict / n):

- annotator visibility: full 22/22/22 · partial 25/25/27 · heavy occlusion 6/4/10 · unknown 1/1/2
- observable 54/52/55 (98.2 % / 94.5 %) · not observable 0/0/5 · carried 0/0/1
- held in gripper 19/19/20 (no FK prior: 12/10/20) · free 35/33/41

## The 9 strict misses (run C)

| snapshot | id | cause |
|---|---|---|
| 1783428141_224458752_seq3 | manual_1 | not observable: 21 % seen, one side; placed 0.42 m off (= FP) |
| 1783428170_524189696_seq9 | manual_2 | not observable: 34 % seen, 33 % hidden by the gripper |
| 1783429344_720968448_seq6 | manual_0 | not observable: 30 % seen, 45 % hidden by the held block |
| 1783431798_015547904_seq9 | manual_1 | not observable: 24 % seen, 50 % hidden by the held block |
| 1783428995_118905856_seq1 | manual_0 | not observable: stack bottom, top covered by the held block, one side face |
| 1783428170_524189696_seq9 | manual_0 | heavy occlusion, centre found, yaw off by 90° |
| 1783428275_723442176_seq7 | manual_0 | heavy occlusion, centre found, yaw off by 88° |
| 1783428306_319403264_seq10 | rgb_candidate_1 | carried at z = 2.52 m > `cluster_max_center_z` 1.5; covered in operation by the grasped-block refinement |
| 1783429345_020967424_seq4 | manual_0 | real miss: 68 % seen, placed 0.31 m off next to the held block |

Both false positives are real blocks placed 0.31 / 0.42 m off; none come from the gripper or pallets.

## Runtime

The run-C command repeated 10 times over the same 23 scans, `OMP_NUM_THREADS=1`, waited for
`loadavg < 1.0` first. `runtime_ms` of the 230 detections (nearest-rank percentiles):

| p50 | p90 | max | min | mean | > 100 ms |
|---|---|---|---|---|---|
| 65.3 | 101.1 | 140.4 | 38.6 | 69.4 | 27/230 |

The 10 runs are in [`detector_timing/`](detector_timing/) — `det_{1..10}.json`, one object per
scan under `snapshots[]`, `runtime_ms` each; `detector_git_revision` 44f523b, `fk_prior_enabled`
true. This is the snapshot runner, so it excludes point-cloud transport, the TF lookups and the
service round trip the deployed node pays on top.

## Caveats

- Rail/TCP TF is grafted, Δt ≤ 0.6 s, TCP speed ≤ 0.53 m/s; 1783429000_seq7, 1783429007_seq4 and
  1783429344_seq6 may be a few cm off. World → lidar agrees to ≤ 18 mm.
- In-sample: `detector.yaml` (scene bounds, min_score 0.45, proposal_max_components 16) was tuned on
  captures overlapping these scenes.
- The runner has no RGB edge prior (disabled in deployment anyway) and no request priors.
- Revisions in the runner build were passed by hand (`git rev-parse` fails in the container).
