import os, json
import physics as PH

ALLOWED = {'ris_n', 'prr_min', 'Ptx_dbm', 'cruise'}
OVR = json.loads(os.environ.get('RX_OVERRIDE', '{}') or '{}')
bad = set(OVR) - ALLOWED
if bad:
    raise ValueError(f'override keys not allowed: {bad}')
for k, v in OVR.items():
    PH.REF[k] = v
if 'cruise' in OVR:
    PH.REF['alts'] = (float(OVR['cruise']),)
