import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from openpyxl import Workbook
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.chart.error_bar import ErrorBars
from openpyxl.chart.data_source import NumDataSource, NumRef
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter as CL

FONT = 'Arial'
COL = {'none': '444444', 'random': 'E87BA4', 'coverage': '6250D6', 'proposed': '2A78D6'}
SHORT = {'none': 'No RIS', 'random': 'Random RIS', 'coverage': 'Coverage-based RIS', 'proposed': 'Proposed (route-aware)'}
MAIN = ['none', 'random', 'coverage', 'proposed']
PLAT = ['Facade RIS', 'Ground-mobile RIS']
BINS = ['10-13', '14-17', '18-21', '22-25', '26-30']
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',      # Times-like font shipped with matplotlib: same look on every OS
                     'font.size': 8, 'axes.linewidth': 0.6})


def seed_stats(D, keys, metric):
    """mean over missions per seed, then mean and population std across seeds; also pooled mission-level n."""
    ps = D.groupby(keys + ['seed'])[metric].mean()
    g = ps.groupby(level=list(range(len(keys))))
    out = pd.DataFrame({'mean': g.mean(), 'std_seeds': g.std(ddof=0), 'n_seeds': g.size()})
    n = D[D.seed == D.seed.min()].groupby(keys).size(); out['n_missions_per_seed'] = n
    return out


def paired(D, a, b, metric='E_UAV_kJ', keys=('platform',), lower_better=True, nboot=5000, seed=1):
    """paired comparison of method a vs b on identical (seed, city, mission) instances.
    returns: mean difference (b - a) and % of b, bootstrap 95% CI over seed-averaged missions, Wilcoxon p (seed-averaged
    missions), win rates per mission-seed and per seed-averaged mission."""
    rows = []; rng = np.random.default_rng(seed)
    for g, x in D[D.method.isin([a, b])].groupby(list(keys)):
        W = x.pivot_table(index=['seed', 'city', 'mission_id'], columns='method', values=metric).dropna()
        if a not in W or b not in W or len(W) == 0:
            continue
        d = W[b] - W[a] if lower_better else W[a] - W[b]
        M = W.groupby(level=['city', 'mission_id']).mean(); dm = (M[b] - M[a]) if lower_better else (M[a] - M[b])
        bs = [dm.values[rng.integers(0, len(dm), len(dm))].mean() for _ in range(nboot)]
        try:
            p = wilcoxon(M[a], M[b]).pvalue
        except ValueError:
            p = np.nan
        rec = dict(zip(keys, g if isinstance(g, tuple) else (g,)))
        rec.update(method=a, baseline=b, metric=metric, n_mission_seed_pairs=len(W), n_missions=len(M),
                   mean_a=W[a].mean(), mean_b=W[b].mean(), mean_gain=d.mean(), gain_pct_of_baseline=100 * d.mean() / W[b].mean(),
                   ci95_low=np.percentile(bs, 2.5), ci95_high=np.percentile(bs, 97.5),
                   win_rate_mission_seed=float((d > 0).mean()), win_rate_seed_averaged=float((dm > 0).mean()),
                   n_losses_mission_seed=int((d < 0).sum()), wilcoxon_p=p)
        rows.append(rec)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def light(col, f=0.25):
    """opaque tint of a colour (EPS has no transparency)."""
    c = np.array(matplotlib.colors.to_rgb(col)); return tuple(1 - f * (1 - c))


def save_fig(fig, base):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + '.png', dpi=300); fig.savefig(base + '.svg'); fig.savefig(base + '.eps'); plt.close(fig)


def latex_table(df, caption, label, path, fmt=None, col_fmt=None):
    fmt = fmt or {}
    cols = list(df.columns); spec = col_fmt or ('l' * 1 + 'r' * (len(cols) - 1))
    L = ['\\begin{table}[!t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\footnotesize',
         f'\\begin{{tabular}}{{@{{}}{spec}@{{}}}}', '\\toprule', ' & '.join(str(c) for c in cols) + ' \\\\', '\\midrule']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('--' if not np.isfinite(v) else (fmt.get(c, '{:.2f}')).format(v))
            else:
                cells.append(str(v))
        L.append(' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, 'w').write('\n'.join(L) + '\n')


_thin = Side(style='thin', color='999999'); _B = Border(top=_thin, bottom=_thin, left=_thin, right=_thin)
_H = PatternFill('solid', start_color='DDE6F2')


def excel_sheet(wb, name, title, note, cats, series, kind='line', ytitle='', xtitle='', ymin=None, ymax=None, raw=None,
                stacked=False):
    """one sheet: title, note, data table (category column + mean and std per series), native chart beside the table,
    optional raw/per-seed table below. series: list of (label, colour hex, means list, stds list or None)."""
    ws = wb.create_sheet(name[:31])
    ws['A1'] = title; ws['A1'].font = Font(name=FONT, bold=True, size=12)
    ws['A2'] = note; ws['A2'].font = Font(name=FONT, italic=True, size=9); ws.merge_cells('A2:N2')
    ws['A2'].alignment = Alignment(wrap_text=True, vertical='top'); ws.row_dimensions[2].height = 42
    r0 = 4; ws.cell(r0, 1, xtitle or 'Category')
    for j, (lab, col, mu, sd) in enumerate(series):
        ws.cell(r0, 2 + j, f'{lab} mean')
        if sd is not None:
            ws.cell(r0, 2 + len(series) + j, f'{lab} std')
    for i, c in enumerate(cats):
        ws.cell(r0 + 1 + i, 1, str(c))
        for j, (lab, col, mu, sd) in enumerate(series):
            ws.cell(r0 + 1 + i, 2 + j, None if mu[i] is None or not np.isfinite(mu[i]) else float(mu[i]))
            if sd is not None:
                ws.cell(r0 + 1 + i, 2 + len(series) + j, None if sd[i] is None or not np.isfinite(sd[i]) else float(sd[i]))
    last = r0 + len(cats); ncol = 1 + 2 * len(series)
    for row in ws.iter_rows(min_row=r0, max_row=last, max_col=ncol):
        for c in row:
            c.border = _B; c.font = Font(name=FONT, bold=(c.row == r0 or c.column == 1), size=9)
            if c.row == r0:
                c.fill = _H; c.alignment = Alignment(wrap_text=True, horizontal='center', vertical='center')
            elif c.column > 1:
                c.number_format = '0.000'
    ws.row_dimensions[r0].height = 48; ws.column_dimensions['A'].width = 18
    for j in range(2, ncol + 1):
        ws.column_dimensions[CL(j)].width = 12
    if kind == 'line':
        ch = LineChart()
    else:
        ch = BarChart(); ch.type = 'col'; ch.grouping = 'stacked' if stacked else 'clustered'
        if stacked:
            ch.overlap = 100
    ch.title = title.split(':')[0][:60]; ch.height = 9; ch.width = 16; ch.y_axis.title = ytitle; ch.x_axis.title = xtitle
    ch.x_axis.delete = False; ch.y_axis.delete = False; ch.legend.position = 'b'
    if ymin is not None:
        ch.y_axis.scaling.min = ymin
    if ymax is not None:
        ch.y_axis.scaling.max = ymax
    for j, (lab, col, mu, sd) in enumerate(series):
        ch.add_data(Reference(ws, min_col=2 + j, min_row=r0, max_row=last), titles_from_data=True); se = ch.series[-1]
        if kind == 'line':
            se.graphicalProperties.line.solidFill = col; se.graphicalProperties.line.width = 19050; se.smooth = False
            se.marker = Marker(symbol='circle', size=6); se.marker.graphicalProperties = GraphicalProperties(solidFill=col)
            se.marker.graphicalProperties.line.solidFill = col
        else:
            se.graphicalProperties = GraphicalProperties(solidFill=col)
        if sd is not None and not stacked:
            ref = f"'{ws.title}'!${CL(2 + len(series) + j)}${r0 + 1}:${CL(2 + len(series) + j)}${last}"
            eb = ErrorBars(errDir='y', errBarType='both', errValType='cust', noEndCap=False,
                           plus=NumDataSource(numRef=NumRef(f=ref)), minus=NumDataSource(numRef=NumRef(f=ref)))
            eb.spPr = GraphicalProperties(ln=LineProperties(solidFill=col)); se.errBars = eb
    ch.set_categories(Reference(ws, min_col=1, min_row=r0 + 1, max_row=last))
    ws.add_chart(ch, f'{CL(ncol + 2)}4')
    if raw is not None:
        rr = max(last + 3, 26); ws.cell(rr, 1, 'Source data (per seed / per configuration)').font = Font(name=FONT, bold=True)
        for j, c in enumerate(raw.columns):
            ws.cell(rr + 1, 1 + j, str(c)).font = Font(name=FONT, bold=True, size=9)
        for i, (_, r) in enumerate(raw.iterrows()):
            for j, c in enumerate(raw.columns):
                v = r[c]; ws.cell(rr + 2 + i, 1 + j, float(v) if isinstance(v, (float, np.floating, int, np.integer)) and not isinstance(v, bool) else str(v))
    return ws


def new_wb():
    wb = Workbook(); wb.remove(wb.active); return wb
