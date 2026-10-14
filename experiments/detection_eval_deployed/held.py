# Held-in-gripper split: GT is held if its centre is < 0.35 m from TCP * (0, 0, 0.85).
# usage: held.py <snaps/graft> <rows.json>...
import json, os, sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

TOOLS = os.environ.get('CBS_DETECTOR_TOOLS', '/home/gebmer/repos/ros2_baustelle_ws/src/concrete_block_stack/concrete_block_detector/tools')
MAN = json.load(open(os.path.join(TOOLS, 'reviewed_snapshot_manifest.json')))
GT = {(s['snapshot'], t['id']): np.array(t['position_m']) for s in MAN['snapshots'] for t in s['targets']}
SNAPS = sys.argv[1]


def grip_point(snap):
    tf = yaml.safe_load(open(os.path.join(SNAPS, snap, 'tf.yaml')))
    tf = tf['transforms'] if isinstance(tf, dict) else tf
    t = next(e for e in tf if e['child'] == 'elastic/K8_tool_center_point')
    return np.array(t['xyz']) + Rot.from_quat(t['quaternion_xyzw']).apply([0, 0, 0.85])  # block centre below TCP


for f in sys.argv[2:]:
    g = {True: [], False: []}
    for r in json.load(open(f))['rows']:
        g[bool(np.linalg.norm(GT[(r['snap'], r['id'])] - grip_point(r['snap'])) < 0.35)].append(r)
    print(f, ' | '.join(f"{k} {sum(x['centre'] for x in g[v])}/{sum(x['strict'] for x in g[v])}/{len(g[v])}"
                        for k, v in (('held', True), ('free', False))))
