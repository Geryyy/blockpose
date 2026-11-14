#!/usr/bin/env python3
"""Refine-request failure bench: wrong fit and miss per request family (ticket 354).

One request = one synthetic scene, one requested block, a world-model prior for it.
  wrong fit  the detection the world model associates is > 0.10 m from truth
  miss       nothing associated while >= 1 face of the block is visible

Run: python tools/bench.py --seeds 25 --jobs 16 --out runs/bench
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose import _core  # noqa: E402
from blockpose.scenario_matrix import request_families, world_model_prior  # noqa: E402

DETECTOR_YAML = REPO_ROOT.parent / "concrete_block_detector/config/detector.yaml"
TOLERANCE_M, YAW_TOLERANCE_RAD, MIN_CONFIDENCE = 0.35, 0.70, 0.25  # world model defaults
WRONG_FIT_M = 0.10
FACE_MIN_RETURNS = 20  # = refine_min_points; fewer returns is not a visible face

_STATE: dict = {}


def load_detector(path: Path):
    """DetectionParameters and scene bounds as the node reads them from detector.yaml."""
    ros = yaml.safe_load(path.read_text())["concrete_block_detector"]["ros__parameters"]
    params, used, skipped = _core.DetectionParameters(), {}, []
    for key, value in ros["detector"].items():
        if key == "candidate_dims" and value:
            value = [tuple(value[i : i + 3]) for i in range(0, len(value), 3)]
        if hasattr(params, key) and value != []:
            setattr(params, key, value)
            used[key] = value
        else:
            skipped.append(key)  # node-side keys, e.g. elevated_prior_cluster_max_center_z
    bounds = ros.get("scene_bounds", {})
    bounds = (bounds["min_m"], bounds["max_m"]) if bounds.get("enabled") else None
    return params, used, skipped, bounds


def yaw_of(rotation) -> float:
    r = np.asarray(rotation)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def select_refine_match(poses, prior_xyz, prior_yaw) -> int:
    """world_model_utils.cpp selectRefineMatch: nearest within 0.35 m, yaw mod pi within 0.70."""
    best, best_d = -1, np.inf
    for i, pose in enumerate(poses):
        d = float(np.linalg.norm(np.asarray(pose["position"]) - prior_xyz))
        if d >= best_d or d > TOLERANCE_M or pose["confidence"] < MIN_CONFIDENCE:
            continue
        dyaw = abs(yaw_of(pose["rotation"]) - prior_yaw) % np.pi
        if min(dyaw, np.pi - dyaw) > YAW_TOLERANCE_RAD:
            continue
        best, best_d = i, d
    return best


def _init(detector_yaml: str, stale_share: float) -> None:
    cores = sorted(os.sched_getaffinity(0))  # one core per worker: no thread oversubscription
    worker = multiprocessing.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {cores[worker % len(cores)]})
    params, _, _, bounds = load_detector(Path(detector_yaml))
    _STATE.update(families=request_families(), params=params, bounds=bounds, stale=stale_share)


def run_request(task: tuple[str, int, int]) -> dict:
    family, index, seed = task
    case = _STATE["families"][family][index]
    rng = np.random.default_rng([seed, index, zlib.crc32(family.encode())])
    t0 = time.perf_counter()
    sample = case.scenario.generate(int(rng.integers(2**31)))
    t1 = time.perf_counter()
    truth = sample.ground_truth[case.target]
    prior_xyz, prior_yaw, stale = world_model_prior(truth, rng, _STATE["stale"])

    points = sample.points[sample.instance_ids != -2]  # gripper self-filter, assumed perfect
    if _STATE["bounds"] is not None:
        lo, hi = (np.asarray(b) for b in _STATE["bounds"])
        points = points[np.all((points >= lo) & (points <= hi), axis=1)]
    # node: rays from every raw return, no miss rays (a real lidar reports none)
    context = _core.SensorContext.from_arrays(
        np.ascontiguousarray(sample.sensor_origin, dtype=np.float64),
        np.ascontiguousarray(sample.ray_directions),
        np.ascontiguousarray(sample.ranges),
        np.zeros((0, 3)),
    )
    # PRIORS PLUG IN HERE (after 353): replace discover with the request-path binding, fed
    # the prior "endface+registered_block:<id>" at prior_xyz/prior_yaw. Until then the
    # prior only selects among plain scene-discovery detections.
    result = _core.discover(np.ascontiguousarray(points), _STATE["params"], context)
    t2 = time.perf_counter()

    poses = result["poses"]
    match = select_refine_match(poses, prior_xyz, prior_yaw)
    on_target = sample.instance_ids == case.target
    faces, counts = np.unique(sample.face_ids[on_target], return_counts=True)
    out = {
        "family": family,
        "scene": case.scenario.name,
        "seed": seed,
        "stale_prior": stale,
        "prior_error_m": float(np.linalg.norm(prior_xyz - truth.position)),
        "faces_visible": int(np.sum(counts >= FACE_MIN_RETURNS)),
        "target_returns": int(on_target.sum()),
        "detections": len(poses),
        "nearest_to_truth_m": min(
            (float(np.linalg.norm(np.asarray(p["position"]) - truth.position)) for p in poses),
            default=None,
        ),
        "raycast_s": t1 - t0,
        "detect_s": t2 - t1,
    }
    if match >= 0:
        evidence = result["diagnostics"][match]["evidence"]
        error = float(np.linalg.norm(np.asarray(poses[match]["position"]) - truth.position))
        out.update(
            error_m=error,
            fit_faces=max(evidence["covered_visible_faces"], evidence["observed_geometry_faces"]),
            wrong_fit=error > WRONG_FIT_M,
            miss=False,
        )
    else:
        out.update(error_m=None, fit_faces=None, wrong_fit=False, miss=out["faces_visible"] >= 1)
    return out


def table(outcomes: list[dict]) -> str:
    rows = [
        "| family | requests | >=1 face | wrong fit | of which 2-face | miss (>=1 face) "
        "| stale prior in failures | raycast / detect s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in dict.fromkeys(o["family"] for o in outcomes):
        fam = [o for o in outcomes if o["family"] == family]
        wrong = [o for o in fam if o["wrong_fit"]]
        miss = [o for o in fam if o["miss"]]
        failed = wrong + miss
        two = sum(o["fit_faces"] == 2 for o in wrong)
        rows.append(
            f"| {family} | {len(fam)} | {sum(o['faces_visible'] >= 1 for o in fam)} | "
            f"{len(wrong)} ({100 * len(wrong) / len(fam):.1f} %) | "
            f"{two}/{len(wrong)} | {len(miss)} ({100 * len(miss) / len(fam):.1f} %) | "
            f"{sum(o['stale_prior'] for o in failed)}/{len(failed)} | "
            f"{sum(o['raycast_s'] for o in fam):.0f} / {sum(o['detect_s'] for o in fam):.0f} |"
        )
    return "\n".join(rows)


def git_rev() -> str:
    try:
        rev = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "describe", "--always", "--dirty"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return rev
    except (OSError, subprocess.CalledProcessError):
        rev_file = REPO_ROOT / "REV"  # written by bench_remote.sh, which syncs without .git
        return rev_file.read_text().strip() if rev_file.exists() else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    families = request_families()
    parser.add_argument("--families", nargs="+", default=list(families), choices=list(families))
    parser.add_argument("--seeds", type=int, default=25, help="requests per scene")
    parser.add_argument("--scenes", type=int, default=None, help="first N scenes per family")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--stale-share", type=float, default=0.1, help="priors 0.15 m off")
    parser.add_argument("--detector-yaml", type=Path, default=DETECTOR_YAML)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs/bench")
    args = parser.parse_args()

    _, used, skipped, bounds = load_detector(args.detector_yaml)
    tasks = [
        (family, index, seed)
        for family in args.families
        for index in range(len(families[family][: args.scenes]))
        for seed in range(args.seeds)
    ]
    started = time.time()
    spawn = multiprocessing.get_context("spawn")  # fork deadlocks once Open3D is loaded
    with spawn.Pool(args.jobs, _init, (str(args.detector_yaml), args.stale_share)) as pool:
        outcomes = sorted(
            pool.imap_unordered(run_request, tasks, chunksize=4),
            key=lambda o: (args.families.index(o["family"]), o["scene"], o["seed"]),
        )
    wall_s = time.time() - started

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = table(outcomes)
    run = {
        "git_rev": git_rev(),
        "argv": sys.argv[1:],
        "wall_s": wall_s,
        "jobs": args.jobs,
        "detector_yaml": str(args.detector_yaml),
        "detector_params": used,
        "detector_keys_not_applied": skipped,
        "scene_bounds": bounds,
        "request_path": "discover+sensor_context, world-model association (no priors binding)",
        "table": report,
        "requests": outcomes,
    }
    (args.out / f"bench_{stamp}.json").write_text(json.dumps(run, indent=1))
    print(report)
    print(f"{len(outcomes)} requests in {wall_s:.0f} s on {args.jobs} jobs -> "
          f"{args.out / f'bench_{stamp}.json'}")


if __name__ == "__main__":
    main()
