import json, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

import os
MAN = {s['snapshot']: s for s in json.load(open(os.path.join(os.environ.get('CBS_DETECTOR_TOOLS', '/home/gebmer/repos/ros2_baustelle_ws/src/concrete_block_stack/concrete_block_detector/tools'), 'reviewed_snapshot_manifest.json')))['snapshots']}
RUN = {s['snapshot']: s for s in json.load(open('runs/C_graft_det.json'))['snapshots']}
BMIN, BMAX = np.array([-15.0, -4.5, -0.5]), np.array([-9.5, 3.5, 3.5])
E = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(c, R, d):
    h = np.array(d) / 2
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h
    return s @ np.array(R).T + c


def draw(ax, C, proj, col, lw):
    P = proj(C)
    for a, b in E:
        ax.plot(*zip(P[a], P[b]), color=col, lw=lw)


def render(name, tid, title, out):
    s = MAN[name]; r = RUN[name]; o = np.array(r['sensor_origin_world'])
    raw = np.loadtxt(f'dumpC/{name}/raw.xyz', ndmin=2)[:, :3]
    inp = np.loadtxt(f'dumpC/{name}/input.xyz', ndmin=2)[:, :3]
    ik = {tuple(np.round(p, 4)) for p in inp}
    inb = np.all((raw >= BMIN) & (raw <= BMAX), 1)
    grip = np.array([inb[i] and tuple(np.round(raw[i], 4)) not in ik for i in range(len(raw))])
    t = [x for x in s['targets'] if x['id'] == tid][0]; c = np.array(t['position_m'])
    win = np.all(np.abs(raw - c) < [1.6, 1.6, 2.5], 1)
    # view from sensor: camera frame with z along sensor->target
    f = c - o; f /= np.linalg.norm(f); up = np.array([0, 0, 1.0])
    x = np.cross(f, up); x /= np.linalg.norm(x); y = np.cross(x, f)
    views = [('top-down (x,y)', lambda P: P[:, :2]),
             ('sensor view', lambda P: np.c_[((P - o) @ x) / ((P - o) @ f), ((P - o) @ y) / ((P - o) @ f)])]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (lab, pr) in zip(axs, views):
        m = win & ~grip
        Q = pr(raw[m]); ax.scatter(Q[:, 0], Q[:, 1], s=1, c=raw[m, 2], cmap='viridis', label='lidar (after filter)')
        if (win & grip).any():
            Q = pr(raw[win & grip]); ax.scatter(Q[:, 0], Q[:, 1], s=1, c='orange', label='gripper-filtered')
        for u in s['targets']:
            R = Rot.from_quat(u['orientation_xyzw']).as_matrix()
            draw(ax, corners(np.array(u['position_m']), R, u['dimensions_m']), pr,
                 'red' if u['id'] == tid else 'pink', 2 if u['id'] == tid else 1)
        for p in r['poses']:
            if np.linalg.norm(np.array(p['position']) - c) < 2.0:
                draw(ax, corners(np.array(p['position']), p['rotation'], p['dims']), pr, 'blue', 1.2)
        ax.set_title(lab); ax.set_aspect('equal'); ax.legend(loc='lower right', markerscale=6, fontsize=7)
    fig.suptitle(f'{title}\n{name} {tid} [{t["visibility"]}]  red=GT (pink other GT), blue=detections', fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


for args in [
    ('1783431798_015547904_seq9', 'manual_1', 'MISS (a) occluded by neighbouring blocks: 242 pts, 24% of facing surface seen', 'miss_a_block_occluded.png'),
    ('1783428170_524189696_seq9', 'manual_2', 'MISS (a) gripper occlusion: 34% seen, 33% hidden by gripper', 'miss_a_gripper_occluded.png'),
    ('1783428995_118905856_seq1', 'manual_0', 'MISS (a/single face) stack bottom: top covered by upper block, one side face only', 'miss_a_single_face.png'),
    ('1783429345_020967424_seq4', 'manual_0', 'MISS (c) well-visible, pose 0.31 m off (merged wall row)', 'miss_c_offset.png'),
    ('1783428306_319403264_seq10', 'rgb_candidate_1', 'MISS (held) carried block z=2.52 m, above cluster_max_center_z=1.5', 'miss_held_carried.png'),
]:
    render(*args)
    print('wrote', args[3])
