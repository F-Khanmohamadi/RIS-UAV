"""Export of the evaluation results (eval_rx.py jsonl) to long per-mission CSV files (one row per mission and method).
No simulation is run.  usage: export_rx.py <results root> <out dir>
Writes <out>/per_mission_<tag>.csv for every results/<tag>/ folder found, and <out>/plans_<tag>.jsonl (saved plans)."""
import os, sys, json, glob, numpy as np, pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
PL = {'facade': 'Facade RIS', 'ground': 'Ground-mobile RIS'}
LABEL = {'none': 'No RIS', 'random': 'Random RIS', 'coverage': 'Coverage-based RIS', 'proposed': 'Route-aware RIS (Proposed)'}
BINS = [(10, 13), (14, 17), (18, 21), (22, 25), (26, 30)]
binlab = lambda K: next(f'{lo}-{hi}' for lo, hi in BINS if lo <= K <= hi)


def row(p, s, r, k, x, tag):
    return dict(
        experiment=tag, platform=PL[p], seed=s, city=r['city'].lstrip('g'), mission_id=r['k'], n_congested_cells=r['K'],
        K_interval=binlab(r['K']), method=k, strategy=LABEL[k],
        E_UAV_kJ=x['E_kJ'], E_flight_kJ=x['E_fly_kJ'], E_hover_kJ=x['E_hov_kJ'], E_comm_kJ=x['E_comm_kJ'],
        E_comm_rx_kJ=x['E_rx_kJ'], E_comm_decode_backhaul_J=x['E_bit_J'],
        T_mission_s=x['T_s'], T_hover_s=x['tau_s'], T_flight_s=x['T_s'] - x['tau_s'], delay_per_cell_s=x['tau_s'] / r['K'],
        L_flight_km=x['L_km'], n_hovering_stops=x['stops'], n_shared_hovering_stops=x['shared_stops'],
        frac_shared_hovering_stops=x['frac_shared_stops'], R_ser_Mbps=x['R_ser_Mbps'], R_mis_Mbps=x['R_mis_Mbps'],
        E_GUE_J=x['E_gue_J'], n_GUE_served=x['n_gue'], delivered_Mbit=x['deliv_bits'] / 1e6, transmitted_Mbit=x['txb_bits'] / 1e6,
        PRR_mean=x['prr_mean'], PRR_min=x['prr_min'], delivery_prob_mean=x['delivery_mean'], delivery_prob_min=x['delivery_min'],
        expected_attempts_mean=x['attempts_mean'], PRR_target_met_all_GUEs=x['prr_ok'], all_cells_served_once=x['served_once'],
        battery_ok=x['battery_ok'], deadline_ok=x['deadline_ok'], feasible=x['feasible'],
        battery_slack_kJ=x['battery_slack_kJ'], deadline_slack_s=x['deadline_slack_s'],
        n_RIS_panels=x['n_panels'], n_RIS_assisted_cells=x['n_assisted_cells'], n_shared_RIS_panels=x['n_shared_panels'],
        mean_neighbourhood_points=x['nbh_mean'], min_neighbourhood_points=x['nbh_min'],
        SNR_mean_dB_multiRIS=x['snr_db_mean_multi'], SNR_mean_dB_serving_only=x['snr_db_mean_serving_only'],
        n_GUE_with_nonserving_RIS_power=x['gue_coupled'], E_RIS_circuit_J_not_in_E_UAV=x['E_ris_J'],
        t_inference_s=x['t_inference_s'], t_ILS_s=x['t_ils_s'], t_total_s=x['t_total_s'],
        ils_iterations=x.get('ils_iterations', np.nan), n_cells_repaired=x['repaired'], n_cells_unrepairable=x['unrepairable'],
        CQI_hist_1_to_15=' '.join(map(str, x['cqi_hist'])))


for d in sorted(glob.glob(f'{ROOT}/*/')):
    tag = os.path.basename(os.path.normpath(d)); rows = []; plans = []; serv = []
    for f in sorted(glob.glob(f'{d}/*_s*_*.jsonl')):
        p, s = os.path.basename(f).split('_')[:2]; s = int(s[1:])
        for line in open(f):
            r = json.loads(line)
            serv.append(dict(experiment=tag, platform=PL[p], seed=s, city=r['city'].lstrip('g'), mission_id=r['k'], n_congested_cells=r['K'],
                             servable_with_ris=bool(r.get('servable_ris')), servable_without_ris=bool(r.get('servable_noris', False))))
            if not r.get('servable_ris'):
                continue
            for k in LABEL:
                if r.get(k):
                    rows.append(row(p, s, r, k, r[k], tag))
                    plans.append(dict(experiment=tag, platform=p, seed=s, city=r['city'], k=r['k'], method=k, cells=r['cells'],
                                      depot=r['depot'], dest=r['dest'], plan=r[k]['plan'], panels=r[k]['panels'],
                                      E_kJ=r[k]['E_kJ'], L_km=r[k]['L_km'], stops=r[k]['stops'], tau_s=r[k]['tau_s']))
    if rows:
        pd.DataFrame(rows).to_csv(f'{OUT}/per_mission_{tag}.csv', index=False); pd.DataFrame(serv).to_csv(f'{OUT}/servability_{tag}.csv', index=False)
        with open(f'{OUT}/plans_{tag}.jsonl', 'w') as fh:
            for pl in plans:
                fh.write(json.dumps(pl) + '\n')
        print(tag, len(rows), 'rows')
