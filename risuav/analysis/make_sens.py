"""NEW (audit). Parameter sensitivity outputs from the exported CSVs (fixed seed-0 policy; 40 paired missions:
k = 0..19 of both test layouts; the reference point is taken from the main evaluation on the same missions).
usage: make_sens.py <exported csv dir> <out dir>
Per parameter: figure (UAV energy of each method + saving of the proposed method vs No RIS and vs Coverage, with 95%
bootstrap CIs), Excel workbook (one sheet per platform), CSV and LaTeX table. Retrained-policy points are reported next
to the fixed-policy points."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outlib import *

IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
PARAMS = {
    'N': ('RIS elements $N$', 'RIS elements N', [('N16', 256), ('N24', 576), ('REF', 1024), ('N40', 1600)], 'Number of RIS elements $N$'),
    'D': ('Backlog per GUE $D_k$ (Mbit)', 'D_k (Mbit)', [('D10', 10), ('D20', 20), ('REF', 40), ('D80', 80)], '$D_k$ (Mbit)')}
METH = ['none', 'random', 'coverage', 'proposed']
ref = pd.read_csv(f'{IN}/per_mission_main.csv'); ref = ref[(ref.seed == 0) & (ref.mission_id <= 19) & ref.method.isin(METH)]


def load(tag):
    if tag == 'REF':
        return ref
    f = os.path.join(IN, f'per_mission_sens_{tag}.csv')
    return pd.read_csv(f) if os.path.exists(f) else None


def load_rt(tag):
    f = os.path.join(IN, f'per_mission_sensrt_{tag}.csv')
    return pd.read_csv(f) if os.path.exists(f) else None


allrows = []
for key, (xlab, xshort, pts, xtex) in PARAMS.items():
    if not any(load(t) is not None or load_rt(t) is not None for t, _ in pts if t != 'REF'):
        print(f'{key}: no sensitivity results, skipped'); continue
    rows = []
    for tag, val in pts:
        X = load(tag)
        if X is None:
            continue
        for plat in PLAT:
            x = X[X.platform == plat]
            if x.empty:
                continue
            for m in METH:
                y = x[x.method == m]
                if y.empty:
                    continue
                rows.append(dict(parameter=key, value=val, tag=tag, platform=plat, method=m, policy='fixed (reference-trained)',
                                 n_missions=len(y), E_UAV_kJ=y.E_UAV_kJ.mean(), E_UAV_std=y.E_UAV_kJ.std(), E_flight_kJ=y.E_flight_kJ.mean(),
                                 E_hover_kJ=y.E_hover_kJ.mean(), feasible_pct=100 * y.feasible.mean(), R_ser=y.R_ser_Mbps.mean(),
                                 delay_s=y.delay_per_cell_s.mean(), L_km=y.L_flight_km.mean(), stops=y.n_hovering_stops.mean(),
                                 panels=y.n_RIS_panels.mean(), PRR_mean=y.PRR_mean.mean(), E_GUE_J=y.E_GUE_J.mean(),
                                 servable_noris_pct=100 * len(y) / max(len(x[x.method == 'proposed']), 1)))
            for b in ('none', 'coverage'):
                p = paired(x, 'proposed', b, 'E_UAV_kJ')
                if len(p):
                    p = p.iloc[0]
                    rows.append(dict(parameter=key, value=val, tag=tag, platform=plat, method=f'saving_vs_{b}', policy='fixed (reference-trained)',
                                     n_missions=p.n_missions, saving_pct=p.gain_pct_of_baseline, saving_kJ=p.mean_gain,
                                     ci95_low_kJ=p.ci95_low, ci95_high_kJ=p.ci95_high, win_rate=p.win_rate_mission_seed, wilcoxon_p=p.wilcoxon_p))
        R_ = load_rt(tag)
        if R_ is not None:
            for plat in PLAT:
                x = R_[R_.platform == plat]
                for m in METH:
                    y = x[x.method == m]
                    if not y.empty:
                        rows.append(dict(parameter=key, value=val, tag=tag, platform=plat, method=m, policy='retrained at this value',
                                         n_missions=len(y), E_UAV_kJ=y.E_UAV_kJ.mean(), E_UAV_std=y.E_UAV_kJ.std(),
                                         feasible_pct=100 * y.feasible.mean(), R_ser=y.R_ser_Mbps.mean(), delay_s=y.delay_per_cell_s.mean(),
                                         L_km=y.L_flight_km.mean(), stops=y.n_hovering_stops.mean(), panels=y.n_RIS_panels.mean()))
                for b in ('none', 'coverage'):
                    p = paired(x, 'proposed', b, 'E_UAV_kJ')
                    if len(p):
                        p = p.iloc[0]
                        rows.append(dict(parameter=key, value=val, tag=tag, platform=plat, method=f'saving_vs_{b}', policy='retrained at this value',
                                         n_missions=p.n_missions, saving_pct=p.gain_pct_of_baseline, saving_kJ=p.mean_gain,
                                         ci95_low_kJ=p.ci95_low, ci95_high_kJ=p.ci95_high, win_rate=p.win_rate_mission_seed, wilcoxon_p=p.wilcoxon_p))
    if not rows:
        continue
    T = pd.DataFrame(rows).sort_values(['platform', 'method', 'value']); T.to_csv(f'{OUT}/sensitivity_{key}.csv', index=False); allrows.append(T)
    F = T[T.policy.str.startswith('fixed')]; vals = sorted(F.value.unique())
    wb = new_wb()
    f, ax = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for c, plat in enumerate(PLAT):
        x = F[F.platform == plat]; a = ax[0, c]; series = []
        for m in METH:
            y = x[x.method == m].set_index('value').reindex(vals)
            a.errorbar(range(len(vals)), y.E_UAV_kJ, yerr=1.96 * y.E_UAV_std / np.sqrt(y.n_missions), color='#' + COL[m], marker='o', ms=3,
                       lw=1, capsize=2, label=SHORT[m])
            series.append((SHORT[m], COL[m], list(y.E_UAV_kJ), list(1.96 * y.E_UAV_std / np.sqrt(y.n_missions))))
        rt = T[(T.platform == plat) & T.policy.str.startswith('retrained') & (T.method == 'proposed')]
        for _, r in rt.iterrows():
            a.plot(vals.index(r.value), r.E_UAV_kJ, marker='*', ms=8, color='#2A78D6', mfc='white', ls='none', label='Proposed, retrained')
        a.set_xticks(range(len(vals))); a.set_xticklabels([f'{v:g}' for v in vals]); a.set_ylabel('UAV energy (kJ)'); a.set_title(plat, fontsize=8)
        a.grid(True, axis='y', lw=0.4, alpha=0.4)
        if c == 0:
            a.legend(fontsize=6, frameon=False)
        b2 = ax[1, c]
        for bname, col in (('none', '444444'), ('coverage', '6250D6')):
            y = x[x.method == f'saving_vs_{bname}'].set_index('value').reindex(vals)
            mean_b = x[x.method == bname].set_index('value').reindex(vals).E_UAV_kJ
            lo = 100 * y.ci95_low_kJ / mean_b; hi = 100 * y.ci95_high_kJ / mean_b
            b2.errorbar(range(len(vals)), y.saving_pct, yerr=[y.saving_pct - lo, hi - y.saving_pct], color='#' + col, marker='s', ms=3, lw=1,
                        capsize=2, label=f'vs {SHORT[bname]}')
            series.append((f'Saving vs {SHORT[bname]} (%)', col, list(y.saving_pct), None))
        b2.axhline(0, color='k', lw=0.5); b2.set_xticks(range(len(vals))); b2.set_xticklabels([f'{v:g}' for v in vals]); b2.set_xlabel(xlab)
        b2.set_ylabel('Energy saving of proposed (%)'); b2.grid(True, axis='y', lw=0.4, alpha=0.4)
        if c == 0:
            b2.legend(fontsize=6, frameon=False)
        excel_sheet(wb, plat, f'Sensitivity to {xshort} ({plat})',
                    'Fixed policy trained at the reference (seed 0); 40 paired test missions per value (k = 0-19 of cities 21 and 22). '
                    'Energy: mean over missions, error = 95% CI of the mean. Savings: paired, % of the baseline mean.',
                    [f'{v:g}' for v in vals], series, 'line', 'UAV energy (kJ) / saving (%)', xshort,
                    raw=x.drop(columns=['parameter', 'platform']))
    f.tight_layout(); save_fig(f, f'{OUT}/Sensitivity_{key}'); wb.save(f'{OUT}/Sensitivity_{key}.xlsx')
    L = []
    for plat in PLAT:
        x = F[F.platform == plat]
        for v in vals:
            g = lambda m, c: x[(x.method == m) & (x.value == v)][c].iloc[0] if not x[(x.method == m) & (x.value == v)].empty else np.nan
            L.append({'Platform': plat.replace(' RIS', ''), xtex: f'{v:g}', 'No RIS': g('none', 'E_UAV_kJ'), 'Random': g('random', 'E_UAV_kJ'),
                      'Coverage': g('coverage', 'E_UAV_kJ'), 'Proposed': g('proposed', 'E_UAV_kJ'),
                      'Save vs No RIS (\\%)': g('saving_vs_none', 'saving_pct'), 'Save vs Cov. (\\%)': g('saving_vs_coverage', 'saving_pct'),
                      '$\\bar R_{\\rm ser}$': g('proposed', 'R_ser'), 'Stops': g('proposed', 'stops'), 'Feas. (\\%)': g('proposed', 'feasible_pct')})
    latex_table(pd.DataFrame(L), f'Sensitivity of the UAV energy (kJ) to {xlab} (fixed reference-trained policy, 40 paired missions).',
                f'tab:sens_{key}', f'{OUT}/sensitivity_{key}.tex', fmt={'Stops': '{:.1f}', 'Feas. (\\%)': '{:.0f}'}, col_fmt='ll' + 'r' * 9)
if allrows:
    pd.concat(allrows, ignore_index=True).to_csv(f'{OUT}/sensitivity_all.csv', index=False)
print('sensitivity outputs written to', OUT)
