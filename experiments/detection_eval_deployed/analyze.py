# Per-GT observability + detection table for a runner JSON with --dump-stages output.
import json, sys, math, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

TOOLS = os.environ.get('CBS_DETECTOR_TOOLS', '/home/gebmer/repos/ros2_baustelle_ws/src/concrete_block_stack/concrete_block_detector/tools')
sys.path.insert(0, TOOLS)
from score_snapshot_runner import _rotation_error_deg, _inside_roi, _maximum_matching  # noqa

SC = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1]
DUMP = sys.argv[2]
OUT = sys.argv[3]
MAN = json.load(open(os.path.join(TOOLS, 'reviewed_snapshot_manifest.json')))
BMIN, BMAX = np.array([-15.0, -4.5, -0.5]), np.array([-9.5, 3.5, 3.5])
rng = np.random.default_rng(0)


def box_local(P, c, R):
    return (P - c) @ R


def iou(c1, R1, d1, c2, R2, d2, n=40000):
    h1, h2 = np.array(d1) / 2, np.array(d2) / 2
    s1 = (rng.uniform(-1, 1, (n, 3)) * h1) @ R1.T + c1
    s2 = (rng.uniform(-1, 1, (n, 3)) * h2) @ R2.T + c2
    in12 = np.all(np.abs(box_local(s1, c2, R2)) <= h2, 1).mean()
    in21 = np.all(np.abs(box_local(s2, c1, R1)) <= h1, 1).mean()
    v1, v2 = np.prod(d1), np.prod(d2)
    inter = 0.5 * (in12 * v1 + in21 * v2)
    return inter / (v1 + v2 - inter)


def load(p, cols=3):
    a = np.loadtxt(p, ndmin=2)
    return a


rows = []
runner = {s['snapshot']: s for s in json.load(open(RUN))['snapshots']}
fp_rows = []
for snap in MAN['snapshots']:
    name = snap['snapshot']
    sr = runner[name]
    origin = np.array(sr['sensor_origin_world'])
    preds = [p for p in sr['poses'] if _inside_roi(p['position'], snap['scoring_roi'])]
    tg = snap['targets']
    raw = load(f'{DUMP}/{name}/raw.xyz')[:, :3]
    inp = load(f'{DUMP}/{name}/input.xyz')[:, :3]
    clu = load(f'{DUMP}/{name}/clusters.xyz')
    inb = np.all((raw >= BMIN) & (raw <= BMAX), 1)
    key = lambda A: {tuple(np.round(r, 4)) for r in A}
    inpk = key(inp)
    grip = np.array([inb[i] and tuple(np.round(raw[i], 4)) not in inpk for i in range(len(raw))])
    # reviewed matcher (0.5 m + dims + rot 30), per target
    rev = {t: p for t, p, _ in _maximum_matching(tg, preds)}
    # paper matcher: Hungarian centre <= 0.3 m, strict = rot <= 30 & IoU >= 0.25
    C = np.array([[np.linalg.norm(np.array(t['position_m']) - p['position']) for p in preds] for t in tg]) \
        if preds else np.zeros((len(tg), 0))
    pairs = {}
    if preds:
        Cm = np.where(C <= 0.30, C, 1e6)
        for i, j in zip(*linear_sum_assignment(Cm)):
            if Cm[i, j] < 1e6:
                pairs[i] = j
    Rs, cs, dims = [], [], []
    for t in tg:
        q = t['orientation_xyzw']
        Rs.append(Rot.from_quat(q).as_matrix()); cs.append(np.array(t['position_m'])); dims.append(np.array(t['dimensions_m']))
    # angular index of all raw returns
    dirs = raw - origin
    rngs = np.linalg.norm(dirs, axis=1)
    tree = cKDTree(dirs / rngs[:, None])
    def surf(P, k):
        L = box_local(P, cs[k], Rs[k]); hh = dims[k] / 2
        out = np.linalg.norm(np.maximum(np.abs(L) - hh, 0), axis=1)
        return np.where(out > 0, out, np.min(hh - np.abs(L), axis=1))
    SD = np.array([surf(inp, k) for k in range(len(tg))])
    owner = np.argmin(SD, 0)
    for i, t in enumerate(tg):
        c, R, d = cs[i], Rs[i], dims[i]
        h = d / 2
        L = box_local(inp, c, R)
        near = (SD[i] <= 0.05) & (owner == i)  # exclusive: point closest to this GT box
        npts = int(near.sum())
        upax = int(np.argmax(np.abs(R[2, :])))
        upsign = np.sign(R[2, upax])
        Ln = L[near] / h
        ax = np.argmax(np.abs(Ln), 1) if len(Ln) else np.array([], int)
        sg = np.sign(Ln[np.arange(len(Ln)), ax]) if len(Ln) else np.array([])
        top = int(((ax == upax) & (sg == upsign)).sum())
        side_counts = [int(((ax == a) & (sg == s)).sum()) for a in range(3) if a != upax for s in (-1, 1)]
        nsides = sum(1 for k in side_counts if k >= 20)
        # gripper points near the box (raw, before filter)
        Lg = box_local(raw[grip], c, R)
        gnear = int((np.linalg.norm(np.maximum(np.abs(Lg) - h, 0), axis=1) <= 0.05).sum())
        # ray analysis on sensor-facing faces (exclude bottom), 4 cm grid
        seen = occ_g = occ_b = occ_o = noret = 0
        for a in range(3):
            for s in (-1, 1):
                if a == upax and s == -upsign:
                    continue
                n_w = R[:, a] * s
                fc = c + n_w * h[a]
                if n_w @ (origin - fc) <= 0:
                    continue
                o = [b for b in range(3) if b != a]
                g1 = np.arange(-h[o[0]] + 0.02, h[o[0]], 0.04)
                g2 = np.arange(-h[o[1]] + 0.02, h[o[1]], 0.04)
                for u in g1:
                    for v in g2:
                        pl = np.zeros(3); pl[a] = s * h[a]; pl[o[0]] = u; pl[o[1]] = v
                        pw = c + R @ pl
                        dv = pw - origin; r = np.linalg.norm(dv)
                        idx = tree.query_ball_point(dv / r, 0.0035)  # ~0.2 deg
                        if not idx:
                            noret += 1; continue
                        rr = rngs[idx]
                        if np.any(np.abs(rr - r) <= 0.10):
                            seen += 1; continue
                        j = np.array(idx)[rr < r - 0.10]
                        if len(j) == 0:
                            noret += 1; continue  # only returns behind: hole / dark surface
                        jj = j[np.argmin(rngs[j])]
                        if grip[jj]:
                            occ_g += 1
                        elif any(np.all(np.abs(box_local(raw[jj][None], cs[k], Rs[k])[0]) <= dims[k] / 2 + 0.05)
                                 for k in range(len(tg)) if k != i):
                            occ_b += 1
                        else:
                            occ_o += 1
        tot = max(seen + occ_g + occ_b + occ_o + noret, 1)
        # cluster membership: components holding the block's points, and other GT blocks in them
        Lc = box_local(clu[:, :3], c, R)
        inc = np.linalg.norm(np.maximum(np.abs(Lc) - h, 0), axis=1) <= 0.05
        comps = clu[inc, 3].astype(int)
        comp = int(np.bincount(comps).argmax()) if len(comps) else -1
        shared = 0
        if comp >= 0:
            cp = clu[clu[:, 3] == comp, :3]
            for k in range(len(tg)):
                if k != i and (np.linalg.norm(np.maximum(np.abs(box_local(cp, cs[k], Rs[k])) - dims[k] / 2, 0), axis=1) <= 0.05).sum() >= 20:
                    shared += 1
        # nearest prediction regardless of matching
        nd = min((np.linalg.norm(c - p['position']) for p in sr['poses']), default=np.inf)
        j = pairs.get(i)
        strict = False; rot = iouv = None
        if j is not None:
            p = preds[j]
            rot = _rotation_error_deg(p, t)
            iouv = iou(np.array(p['position']), np.array(p['rotation']), p['dims'], c, R, d)
            strict = rot <= 30 and iouv >= 0.25
        tcp_d = None
        rows.append(dict(snap=name, id=t['id'], vis=t.get('visibility'), npts=npts, top=top, sides=nsides,
                         side_counts=side_counts, grip_pts_near=gnear,
                         seen=seen / tot, occ_grip=occ_g / tot, occ_block=occ_b / tot, occ_other=occ_o / tot,
                         noret=noret / tot, comp=comp, comp_shared_gt=shared, nearest_pred_m=float(nd),
                         centre=j is not None, strict=bool(strict), rot=rot, iou=iouv, reviewed=i in rev,
                         z=float(c[2])))
    used = set(pairs.values())
    for j, p in enumerate(preds):
        if j not in used:
            dmin = min((np.linalg.norm(np.array(t['position_m']) - p['position']) for t in tg), default=np.inf)
            fp_rows.append(dict(snap=name, pos=p['position'], src=p['source'], score=p['evidence']['score'],
                                prior=p.get('prior_match', {}).get('score'), nearest_gt=float(dmin),
                                faces=p['evidence']['observed_geometry_faces'], support=p['evidence']['support_points']))
    n_pred = len(preds)
    print(name, len(tg), 'preds', n_pred, 'centre', len(pairs), file=sys.stderr)
json.dump(dict(rows=rows, fps=fp_rows), open(OUT, 'w'), indent=1, default=float)
