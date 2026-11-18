"""NEW (audit). Every main-result figure and table of the manuscript (and the additional metrics requested) from the
exported per-mission CSV and the training logs. No simulation is run.
usage: make_main.py <per_mission_main.csv> <runs dir> <out dir> [tag=proposed]
For every experiment: <name>.png/.svg/.eps (one file per platform where the manuscript has subfigures),
<name>.xlsx (data + native chart on one sheet per platform), <name>.csv (summary), <name>.tex (IEEE table)."""
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outlib import *

CSV, RUNS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; TAG = sys.argv[4] if len(sys.argv) > 4 else 'proposed'
os.makedirs(OUT, exist_ok=True)
D = pd.read_csv(CSV); D = D[D.method.isin(MAIN)]
PLAT_D = [p for p in PLAT if p in set(D.platform)]   # platforms with results (allows checks on partial runs)
PFILE = {'Facade RIS': 'Facade', 'Ground-mobile RIS': 'Ground_Mobile'}
NOTE = ('Mean over the test missions for each training seed, then mean and population std across the seeds '
        '(error bars = std across seeds). Seeds: {s}; missions per seed: {n}; platform: {p}.')
SEEDS = sorted(D.seed.unique())


# ------------------------------------------------------------------ 1 training convergence
def training():
    recs = []
    for f in sorted(glob.glob(f'{RUNS}/{TAG}_*_s*/log.json')):
        L = json.load(open(f)); plat = 'Facade RIS' if L['scen'] == 'facade' else 'Ground-mobile RIS'
        st = pd.DataFrame(L['steps']).groupby('epoch').mean(numeric_only=True)        # training metrics are logged per step
        for e in L['epochs_log']:
            s = st.loc[e['epoch']] if e['epoch'] in st.index else pd.Series(np.nan, index=st.columns)   # epoch 0 = initial validation
            recs.append(dict(platform=plat, seed=L['seed'], epoch=e['epoch'], val_E_kJ=e['val_E_kJ'],
                             train_cost_kJ=s['cost'] / 1e3, train_E_kJ=s['E_kJ'],
                             train_penalty_kJ=s['penalty'] / 1e3, entropy=s['entropy_per_decision'],
                             val_feasible=e['val_feasible'], baseline_replaced=e['baseline_replaced'], best_epoch=L.get('best_epoch')))
    if not recs:
        return
    T = pd.DataFrame(recs); T.to_csv(f'{OUT}/training_per_seed.csv', index=False)
    G = T.groupby(['platform', 'epoch']).agg(['mean', lambda x: x.std(ddof=0)])
    wb = new_wb(); summ = []
    for plat in PLAT:
        if plat not in G.index.get_level_values(0):
            continue
        g = G.loc[plat]; ep = g.index.values
        tc, tcs = g[('train_cost_kJ', 'mean')].values, g[('train_cost_kJ', '<lambda_0>')].values
        ve, ves = g[('val_E_kJ', 'mean')].values, g[('val_E_kJ', '<lambda_0>')].values
        f, a = plt.subplots(figsize=(3.5, 2.6))
        SER = ((tc, tcs, '#D98B2B', 'Training'), (ve, ves, '#2A78D6', 'Validation'))
        for y, s, col, lab in SER:
            m = np.isfinite(y); a.fill_between(ep[m], (y - s)[m], (y + s)[m], color=light(col), lw=0, zorder=1)
        for y, s, col, lab in SER:
            m = np.isfinite(y); a.plot(ep[m], y[m], color=col, lw=1.4, label=lab, zorder=3)
        a.set_xlabel('Epoch', fontsize=10, fontweight='bold'); a.set_ylabel('UAV Energy (kJ)', fontsize=10, fontweight='bold')
        a.tick_params(labelsize=9); a.set_ylim(120, 270)   # same range for both platforms
        for t in a.get_xticklabels() + a.get_yticklabels():
            t.set_fontweight('bold')
        a.grid(True, axis='y', lw=0.4, alpha=0.4); a.legend(frameon=False, prop={'size': 9, 'weight': 'bold'})
        f.tight_layout(); save_fig(f, f'{OUT}/Training_Energy_{PFILE[plat]}')
        en, ens = g[('entropy', 'mean')].values, g[('entropy', '<lambda_0>')].values
        f, a = plt.subplots(figsize=(3.5, 2.4)); m = np.isfinite(en)
        a.fill_between(ep[m], (en - ens)[m], (en + ens)[m], color=light('#6250D6'), lw=0); a.plot(ep[m], en[m], color='#6250D6', lw=1.1)
        a.set_xlabel('Epoch'); a.set_ylabel('Policy entropy (nats per decision)'); a.grid(True, axis='y', lw=0.4, alpha=0.4); a.set_title(plat, fontsize=8)
        f.tight_layout(); save_fig(f, f'{OUT}/Policy_Entropy_{PFILE[plat]}')
        raw = T[T.platform == plat].pivot_table(index='epoch', columns='seed', values=['train_cost_kJ', 'val_E_kJ']).reset_index()
        raw.columns = ['epoch'] + [f'{a_} seed {b_}' for a_, b_ in raw.columns[1:]]
        excel_sheet(wb, plat, f'Training ({plat}): training cost and validation UAV energy',
                    'Training cost = mean cost of the sampled plans of the epoch (UAV energy + penalty); validation = greedy policy on '
                    'the fixed validation missions (training layouts, no local search). Mean and population std over seeds '
                    f'{SEEDS}. Values from runs/*/log.json.', list(ep), [('Training cost (kJ)', 'D98B2B', tc, tcs), ('Validation energy (kJ)', '2A78D6', ve, ves)],
                    'line', 'Energy (kJ)', 'Epoch', raw=raw)
        last = T[(T.platform == plat) & (T.epoch == T.epoch.max())]
        best = T[T.platform == plat].groupby('seed').apply(lambda x: x.loc[x.val_E_kJ.idxmin()])
        summ.append(dict(Platform=plat, Seeds=len(last), **{'Initial val. (kJ)': T[(T.platform == plat) & (T.epoch == 0)].val_E_kJ.mean(),
                         'Final val. (kJ)': last.val_E_kJ.mean(), 'Final val. std': last.val_E_kJ.std(ddof=0),
                         'Best val. (kJ)': best.val_E_kJ.mean(), 'Best epochs': '/'.join(str(int(e)) for e in best.epoch),
                         'Baseline updates': int(T[(T.platform == plat) & T.baseline_replaced.astype(bool)].shape[0] / max(len(last), 1))}))
    wb.save(f'{OUT}/Training_Energy.xlsx')
    S = pd.DataFrame(summ); S.to_csv(f'{OUT}/training_summary.csv', index=False)
    latex_table(S, 'Training of the proposed planner (50 epochs, validation UAV energy without local search).', 'tab:training',
                f'{OUT}/training_summary.tex')


# ------------------------------------------------------------------ 2 metric vs number of congested cells (one figure per metric)
VS = [('E_UAV_kJ', 'UAV_Energy', r'UAV energy $E_{\rm UAV}$ (kJ)', 'UAV energy (kJ)'),
      ('R_ser_Mbps', 'Service_Throughput', r'Service throughput $\bar R_{\rm ser}$ (Mbit/s)', 'Service throughput (Mbit/s)'),
      ('delay_per_cell_s', 'Communication_Delay', 'Average delay per cell (s)', 'Delay per cell (s)'),
      ('T_mission_s', 'Mission_Time', 'Mission completion time $T$ (s)', 'Mission time (s)'),
      ('E_GUE_J', 'GUE_Energy', r'GUE energy $E_{\rm GUE}$ (J)', 'GUE energy (J)'),
      ('L_flight_km', 'Route_Length', 'Route length $L$ (km)', 'Route length (km)'),
      ('n_hovering_stops', 'Hovering_Stops', 'Number of hovering points', 'Hovering points'),
      ('R_mis_Mbps', 'Mission_Throughput', r'Mission throughput $\bar R_{\rm mis}$ (Mbit/s)', 'Mission throughput (Mbit/s)')]


def vs_cells():
    allrows = []
    for metric, name, ylab, short in VS:
        wb = new_wb()
        for plat in PLAT_D:
            X = D[D.platform == plat]
            st = seed_stats(X, ['K_interval', 'method'], metric)
            f, a = plt.subplots(figsize=(3.5, 2.6)); series = []
            for j, m in enumerate(MAIN):
                mu = [st.loc[(b, m), 'mean'] if (b, m) in st.index else np.nan for b in BINS]
                sd = [st.loc[(b, m), 'std_seeds'] if (b, m) in st.index else np.nan for b in BINS]
                a.errorbar(np.arange(5) + 0.05 * (j - 1.5), mu, yerr=sd, color='#' + COL[m], marker='o', ms=3, lw=1, capsize=2, label=SHORT[m])
                series.append((SHORT[m], COL[m], mu, sd))
                for b, u, s in zip(BINS, mu, sd):
                    allrows.append(dict(metric=metric, platform=plat, K_interval=b, method=m, mean=u, std_seeds=s,
                                        n_missions_per_seed=int(st.loc[(b, m), 'n_missions_per_seed']) if (b, m) in st.index else 0))
            a.set_xticks(range(5)); a.set_xticklabels(BINS); a.set_xlabel('Number of congested cells $I$'); a.set_ylabel(ylab)
            a.grid(True, lw=0.4, alpha=0.4); a.legend(fontsize=6.5, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0); f.tight_layout(); save_fig(f, f'{OUT}/{name}_{PFILE[plat]}')
            raw = X.groupby(['K_interval', 'method', 'seed'])[metric].mean().unstack(['method', 'seed']).reset_index()
            raw.columns = ['K_interval'] + [f'{SHORT[m_]} s{s_}' for m_, s_ in raw.columns[1:]]
            nmis = X[X.seed == SEEDS[0]].groupby('K_interval').size()
            excel_sheet(wb, plat, f'{short} vs number of congested cells ({plat})',
                        NOTE.format(s=SEEDS, n=dict(nmis), p=plat), BINS, series, 'line', short, 'Congested cells I', raw=raw)
        wb.save(f'{OUT}/{name}.xlsx')
    A = pd.DataFrame(allrows); A.to_csv(f'{OUT}/vs_cells_all_metrics.csv', index=False)


# ------------------------------------------------------------------ 3 energy breakdown
def breakdown():
    wb = new_wb(); rows = []
    for plat in PLAT_D:
        X = D[D.platform == plat]; comp = [('E_flight_kJ', 'Flight', '7FA7D6'), ('E_hover_kJ', 'Hovering', 'E8A33D'), ('E_comm_kJ', 'Communication', '6AA84F')]
        st = {c: seed_stats(X, ['method'], c) for c, _, _ in comp}
        f, a = plt.subplots(figsize=(3.5, 2.6)); bot = np.zeros(len(MAIN)); series = []
        for c, lab, col in comp:
            mu = np.array([st[c].loc[m, 'mean'] for m in MAIN]); sd = np.array([st[c].loc[m, 'std_seeds'] for m in MAIN])
            a.bar(range(len(MAIN)), mu, bottom=bot, color='#' + col, label=lab, width=0.6, edgecolor='white', lw=0.5); bot += mu
            series.append((lab + ' (kJ)', col, list(mu), list(sd)))
            for m, u, s in zip(MAIN, mu, sd):
                rows.append(dict(platform=plat, method=m, component=lab, mean_kJ=u, std_seeds_kJ=s))
        tot = seed_stats(X, ['method'], 'E_UAV_kJ')
        for i, m in enumerate(MAIN):
            a.text(i, bot[i] + 0.8, f'{tot.loc[m, "mean"]:.1f}', ha='center', fontsize=6.5)
        a.set_xticks(range(len(MAIN))); a.set_xticklabels([SHORT[m].replace(' (route-aware)', '').replace('Coverage-based', 'Coverage') for m in MAIN], fontsize=7)
        a.set_ylabel('UAV energy (kJ)'); a.legend(fontsize=6.5, frameon=False, loc='lower right'); a.grid(True, axis='y', lw=0.4, alpha=0.4)
        f.tight_layout(); save_fig(f, f'{OUT}/UAV_Energy_Breakdown_{PFILE[plat]}')
        excel_sheet(wb, plat, f'UAV energy breakdown ({plat})', NOTE.format(s=SEEDS, n=int(X[X.seed == SEEDS[0]].groupby("method").size().iloc[0]), p=plat),
                    [SHORT[m] for m in MAIN], series, 'bar', 'UAV energy (kJ)', 'Method', stacked=True)
    wb.save(f'{OUT}/UAV_Energy_Breakdown.xlsx'); pd.DataFrame(rows).to_csv(f'{OUT}/energy_breakdown.csv', index=False)


# ------------------------------------------------------------------ 4 overall table and paired statistics
MET = [('E_UAV_kJ', 'E_UAV (kJ)'), ('E_flight_kJ', 'Flight (kJ)'), ('E_hover_kJ', 'Hover (kJ)'), ('E_comm_kJ', 'Comm. (kJ)'),
       ('T_mission_s', 'T (s)'), ('T_hover_s', 'Service time (s)'), ('L_flight_km', 'L (km)'), ('n_hovering_stops', 'Stops'),
       ('n_shared_hovering_stops', 'Shared stops'), ('R_ser_Mbps', 'R_ser (Mbit/s)'), ('R_mis_Mbps', 'R_mis (Mbit/s)'),
       ('delay_per_cell_s', 'Delay/cell (s)'), ('E_GUE_J', 'E_GUE (J)'), ('n_RIS_panels', 'Panels'), ('PRR_mean', 'Mean PRR'),
       ('PRR_min', 'Min PRR'), ('t_total_s', 'Planning time (s)')]


def overall():
    rows = []
    meths = [m for m in MAIN if m in set(D.method)]
    for plat in PLAT_D:
        X = D[D.platform == plat]
        for m in meths:
            r = dict(platform=plat, method=m, label=SHORT[m], n_seeds=X[X.method == m].seed.nunique(),
                     n_missions_per_seed=int((X[(X.method == m) & (X.seed == SEEDS[0])]).shape[0]),
                     feasible_pct=100 * X[X.method == m].feasible.mean())
            for c, lab in MET:
                st = seed_stats(X[X.method == m], ['method'], c)
                r[c + '_mean'] = st['mean'].iloc[0]; r[c + '_std_seeds'] = st['std_seeds'].iloc[0]
            rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(f'{OUT}/summary_overall.csv', index=False)
    P = pd.concat([paired(D, 'proposed', b, 'E_UAV_kJ') for b in ['none', 'random', 'coverage'] if b in set(D.method)] +
                  [paired(D, 'proposed', b, 'R_ser_Mbps', lower_better=False) for b in ['none', 'random', 'coverage']] +
                  [paired(D, 'proposed', b, 'delay_per_cell_s') for b in ['none', 'random', 'coverage']], ignore_index=True)
    P['wilcoxon_p_holm'] = holm(P.wilcoxon_p.fillna(1).values); P.to_csv(f'{OUT}/paired_statistics.csv', index=False)
    # IEEE table: main methods, both platforms
    T = []
    for plat in PLAT_D:
        for m in MAIN:
            r = S[(S.platform == plat) & (S.method == m)].iloc[0]
            T.append({'Platform': plat.replace(' RIS', ''), 'Method': SHORT[m].replace(' (route-aware)', ''),
                      '$E_{\\rm UAV}$ (kJ)': f"{r.E_UAV_kJ_mean:.2f} $\\pm$ {r.E_UAV_kJ_std_seeds:.2f}",
                      'Flight (kJ)': f'{r.E_flight_kJ_mean:.2f}', 'Hover (kJ)': f'{r.E_hover_kJ_mean:.2f}',
                      '$T$ (s)': f'{r.T_mission_s_mean:.0f}', '$L$ (km)': f'{r.L_flight_km_mean:.2f}', 'Stops': f'{r.n_hovering_stops_mean:.1f}',
                      '$\\bar R_{\\rm ser}$': f'{r.R_ser_Mbps_mean:.1f}', 'Delay (s)': f'{r.delay_per_cell_s_mean:.2f}',
                      '$E_{\\rm GUE}$ (J)': f'{r.E_GUE_J_mean:.1f}'})
    latex_table(pd.DataFrame(T), 'Overall results on the test missions (mean $\\pm$ std across training seeds).', 'tab:overall',
                f'{OUT}/summary_overall.tex', col_fmt='ll' + 'r' * 9)
    wb = new_wb()
    for plat in PLAT_D:
        X = S[S.platform == plat]
        excel_sheet(wb, plat, f'Overall UAV energy by method ({plat})', NOTE.format(s=SEEDS, n=int(X.n_missions_per_seed.iloc[0]), p=plat),
                    list(X.label), [('E_UAV (kJ)', '2A78D6', list(X.E_UAV_kJ_mean), list(X.E_UAV_kJ_std_seeds))], 'bar', 'UAV energy (kJ)', 'Method',
                    raw=X[['label'] + [c for c in X.columns if c.endswith('_mean')]])
    wb.save(f'{OUT}/Summary_Overall.xlsx')
    Q = P[P.metric == 'E_UAV_kJ'][['platform', 'baseline', 'mean_gain', 'gain_pct_of_baseline', 'ci95_low', 'ci95_high',
                                   'win_rate_mission_seed', 'win_rate_seed_averaged', 'wilcoxon_p', 'wilcoxon_p_holm']].copy()
    Q['baseline'] = Q.baseline.map(SHORT); Q.columns = ['Platform', 'Baseline', 'Saving (kJ)', 'Saving (%)', 'CI low', 'CI high',
                                                         'Win (mission-seed)', 'Win (mission)', 'p', 'p (Holm)']
    latex_table(Q, 'Paired energy savings of the proposed method (bootstrap 95\\% CI over seed-averaged missions; Wilcoxon signed-rank).',
                'tab:paired', f'{OUT}/paired_statistics.tex', fmt={'p': '{:.1e}', 'p (Holm)': '{:.1e}', 'Win (mission-seed)': '{:.3f}', 'Win (mission)': '{:.3f}'},
                col_fmt='ll' + 'r' * 8)


training(); vs_cells(); breakdown(); overall(); print('main outputs written to', OUT)
