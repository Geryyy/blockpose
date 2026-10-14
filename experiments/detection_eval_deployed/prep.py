# Build two snapshot trees for the runner: orig (as recorded) and graft (rail/TCP TF from nearest
# seibersdorf frame of the same bag). Clouds are symlinked/copied, never modified.
import os, shutil, sys, yaml, numpy as np

SRC = os.environ.get('REG_SNAPSHOTS', '/home/gebmer/repos/blockpose/data/registration_snapshots')
DS = os.environ.get('SEIBERSDORF_DATASET', '/home/gebmer/Documents/seibersdorf_dataset')
OUT = sys.argv[1]
GRAFT = ['elastic/K9', 'elastic/K10_left_rail', 'elastic/K11', 'elastic/K12_right_rail',
         'elastic/K8_tool_center_point']
ds = sorted((int(n.split('__')[1].split('_')[0]) + int(n.split('__')[1].split('_')[1]) * 1e-9, n)
            for n in os.listdir(DS))
man = __import__('json').load(open(os.path.join(os.environ.get('CBS_DETECTOR_TOOLS', '/home/gebmer/repos/ros2_baustelle_ws/src/concrete_block_stack/concrete_block_detector/tools'), 'reviewed_snapshot_manifest.json')))
rows = []
for s in man['snapshots']:
    r = s['snapshot']
    a, b = r.split('_')[:2]
    t = int(a) + int(b) * 1e-9
    near = sorted(ds, key=lambda x: abs(x[0] - t))[:2]
    for variant in ('orig', 'graft'):
        d = os.path.join(OUT, variant, r)
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(SRC, r, 'cloud.pcd'), d)
        tf = yaml.safe_load(open(os.path.join(SRC, r, 'tf.yaml')))
        if variant == 'graft':
            ntf = yaml.safe_load(open(os.path.join(DS, near[0][1], 'tf.yaml')))
            ntf2 = yaml.safe_load(open(os.path.join(DS, near[1][1], 'tf.yaml')))
            have = {e['child'] for e in tf['transforms']}
            by = {e['child']: e for e in ntf['transforms']}
            by2 = {e['child']: e for e in ntf2['transforms']}
            from scipy.spatial.transform import Rotation as Rot, Slerp
            f = (t - near[0][0]) / (near[1][0] - near[0][0])  # interpolation fraction
            for c in GRAFT:
                if c not in have and c in by:
                    e = dict(by[c]); m1 = np.array(by[c]['matrix']); m2 = np.array(by2[c]['matrix'])
                    sl = Slerp([0, 1], Rot.from_matrix([m1[:3, :3], m2[:3, :3]]))
                    m = np.eye(4); m[:3, :3] = sl([min(max(f, 0), 1)]).as_matrix()[0]
                    m[:3, 3] = m1[:3, 3] + f * (m2[:3, 3] - m1[:3, 3])
                    e['matrix'] = m.tolist(); e['xyz'] = m[:3, 3].tolist()
                    e['quaternion_xyzw'] = Rot.from_matrix(m[:3, :3]).as_quat().tolist()
                    e['rpy'] = Rot.from_matrix(m[:3, :3]).as_euler('xyz').tolist()
                    e['name'] = e['name'] + '_interp'
                    tf['transforms'].append(e)
            p1 = np.array(by['elastic/K8_tool_center_point']['xyz'])
            p2 = np.array(by2['elastic/K8_tool_center_point']['xyz'])
            sw1 = np.array(by['seyond']['xyz']); sw0 = np.array(
                [e for e in tf['transforms'] if e['child'] == 'seyond'][0]['xyz'])
            rows.append((r, near[0][1], near[0][0] - t, np.linalg.norm(p1 - p2) / abs(near[1][0] - near[0][0]),
                         np.linalg.norm(sw1 - sw0)))
        yaml.safe_dump(tf, open(os.path.join(d, 'tf.yaml'), 'w'), sort_keys=False)
for r in rows:
    print('%s <- %s dt=%+.2fs tcp_speed=%.3fm/s seyond_diff=%.4fm' % r)
