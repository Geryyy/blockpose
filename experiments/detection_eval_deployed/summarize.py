import json, sys
from collections import defaultdict
for f in sys.argv[1:]:
    d = json.load(open(f)); R = d['rows']
    n = len(R); c = sum(r['centre'] for r in R); s = sum(r['strict'] for r in R); v = sum(r['reviewed'] for r in R)
    npred = c + len(d['fps'])
    print(f"== {f}: centre {c}/{n} {100*c/n:.1f}%  strict {s}/{n} {100*s/n:.1f}%  reviewed-matcher {v}/{n}  precision(centre) {c}/{npred} {100*c/npred:.1f}%  FP {len(d['fps'])}")
    def tab(key, label):
        g = defaultdict(list)
        for r in R: g[key(r)].append(r)
        print('  ' + label + ': ' + ' | '.join(f"{k} {sum(x['centre'] for x in g[k])}/{sum(x['strict'] for x in g[k])}/{len(g[k])}" for k in sorted(g)))
    tab(lambda r: r['vis'], 'vis C/S/N')
    tab(lambda r: '0 <0.4' if r['seen'] < 0.4 else ('1 0.4-0.6' if r['seen'] < 0.6 else '2 >=0.6'), 'seen-frac')
    tab(lambda r: '0 <400' if r['npts'] < 400 else ('1 400-800' if r['npts'] < 800 else '2 >=800'), 'pts<=5cm')
    tab(lambda r: 'carried z>1.5' if r['z'] > 1.5 else ('top+side' if r['top'] >= 50 and r['sides'] >= 1 and r['seen'] >= 0.4 else 'not obs'), 'observable')
