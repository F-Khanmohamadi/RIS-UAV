import sys, os, pickle, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics import REF, build_city
platform, cid = sys.argv[1], int(sys.argv[2])
tag = ('g' if platform == 'ground' else '') + f'city{cid}'; from rx_paths import PRELIM_ROOT
out = os.path.join(PRELIM_ROOT, 'cities', f'{tag}.pkl'); os.makedirs(os.path.dirname(out), exist_ok=True)
P = dict(REF); P['ris_mode'] = platform; P['D_g'] = 1.0; P['pitch'] = 25.0
t = time.time(); C = build_city(cid, P)
for T in C['tabs']:
    for k in ('tau', 'deliv', 'txb', 'U'):
        T[k] = T[k].astype(np.float32)
    T['cov'] = T['cov'].astype(np.float16)
with open(out + '.tmp', 'wb') as f:
    pickle.dump(C, f, protocol=4)
os.replace(out + '.tmp', out); print(f'{tag}: {len(C["tabs"])} cells built in {time.time() - t:.0f} s', flush=True)
