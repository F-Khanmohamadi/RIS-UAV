"""NEW (audit, 17 Nov 2026). Parameter overrides for the sensitivity study. Import this module FIRST (before physics
consumers such as build_exact, env, env_mr, coupling): it updates physics.REF in place from the JSON in the environment
variable RX_OVERRIDE, e.g. RX_OVERRIDE='{"ris_n": 24}'. Allowed keys and their manuscript symbols:
  ris_n (sqrt N), prr_min (PRR_min), Ptx_dbm (P_t), cruise (H; also sets alts = (H,))."""
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
