"""NEW (audit, 17 Nov 2026). Build sensitivity-study tables with RX_OVERRIDE applied (see param_override.py).
  exact  <src prelim pkl> <dst exact pkl>  : exact tables (N, PRR_min, P_t): geometry and candidate sites unchanged
  prelim <facade|ground> <city id> <dst>  : preliminary tables (altitude H changes the lattice LoS and no-fly mask)"""
import sys, os, pickle, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import param_override  # noqa: F401  (must precede the physics consumers)
import physics as PH
os.makedirs(os.path.dirname(os.path.abspath(sys.argv[-1])), exist_ok=True)   # output folder
if sys.argv[1] == 'exact':
    import build_exact as BE
    BE.build(sys.argv[2], sys.argv[3])
    import json; json.dump(dict(override=param_override.OVR, src=sys.argv[2]), open(sys.argv[3] + '.meta.json', 'w'))
else:
    plat, cid, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    P = dict(PH.REF); P['ris_mode'] = plat; P['D_g'] = 1.0; P['pitch'] = 25.0
    t = time.time(); C = PH.build_city(cid, P)
    for T in C['tabs']:
        for k in ('tau', 'deliv', 'txb', 'U'):
            T[k] = T[k].astype(np.float32)
        T['cov'] = T['cov'].astype(np.float16)
    with open(out + '.tmp', 'wb') as f:
        pickle.dump(C, f, protocol=4)
    os.replace(out + '.tmp', out); print(f'{out}: {len(C["tabs"])} cells in {time.time() - t:.0f} s', flush=True)
