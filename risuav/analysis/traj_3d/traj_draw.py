import json, pickle, sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mission import load
import icons as I

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'path',
                     'ps.fonttype': 42, 'axes.titlesize': 8.5, 'legend.fontsize': 7.2})
OUT = os.environ['TRAJ_OUT']; os.makedirs(OUT, exist_ok=True)
H = 120.0
R = json.load(open(os.environ['TRAJ_PLANS'], encoding='utf-8'))
TCITY = os.environ.get('TRAJ_CITY', 'city21'); TTAB = os.environ['TRAJ_TABLES']
CB = pickle.load(open(os.path.join(TTAB, f'{TCITY}.pkl'), 'rb'))
S = pd.read_csv(os.environ['TRAJ_ROWS'])
GUE = {}
for plat, pre in (('facade', ''), ('ground', 'g')):
    X = pickle.load(open(os.path.join(TTAB, f'{pre}{TCITY}.pkl'), 'rb')); GUE[plat] = {h: np.asarray(T['gue']) for h, T in enumerate(X['tabs'])}
NAME = {'none': 'No RIS', 'random': 'Random RIS', 'coverage': 'Coverage-based RIS', 'proposed': 'Proposed'}
SN = {'none': 'No RIS', 'random': 'Random RIS', 'coverage': 'Coverage-based RIS', 'proposed': 'Route-aware RIS (Proposed)'}
PANEL = {'none': 'No RIS', 'random': 'Random RIS', 'coverage': 'Coverage-based RIS', 'proposed': 'Route-aware RIS (proposed)'}
PN = {'facade': 'Facade RIS', 'ground': 'Ground-mobile RIS'}
RIS_ICON = {'facade': 'ris_facade', 'ground': 'ris_ground'}
SHOW_BUILDINGS = False
COL = dict(route=I.C['uav'], ur='#178a3e', rc='#7b3fa0', direct='#7f7f7f', bld='#eef0f2')
CITY = {p: load(p) for p in ('facade', 'ground')}


# ---------------------------------------------------------------- data
def plan(plat, st):
    city, m = CITY[plat]; P = R[f'{plat}/{st}']; stops = np.array(P['stops']); path = np.array(P['path'])
    order = []                                  # visiting order of the stops along the path
    for q in path[1:-1]:
        d = np.hypot(*(stops - q).T)
        if d.min() < 1e-6 and int(d.argmin()) not in order:
            order.append(int(d.argmin()))
    rank = {s: i + 1 for i, s in enumerate(order)}
    links = []
    for a in P['assign']:
        T = city.H[a['h']]; ris = T['cands'][a['option'] - 1][:3] if a['option'] > 0 else None
        links.append(dict(cell=a['cell'], h=a['h'], g=np.asarray(T['g'], float), stop=a['stop'], ris=None if ris is None else np.asarray(ris, float),
                          tau=a['tau_s']))
    nmem = np.bincount([l['stop'] for l in links], minlength=len(stops))
    r = S[(S.platform == PN[plat]) & (S.strategy == SN[st])].iloc[0]
    return dict(city=city, m=m, stops=stops, path=path, rank=rank, links=links, nmem=nmem, rec=r)


def rep_gue(plat, l):
    U = GUE[plat][l['h']]; return U[int(np.argmin(np.hypot(*(U[:, :2] - l['g']).T)))]


def buildings(xl, yl):
    p, w, lo = CB['p'], CB['w'], CB['lo']; polys, cols = [], []
    for ix in range(CB['H'].shape[0]):
        for iy in range(CB['H'].shape[1]):
            x0 = lo + ix * p; y0 = lo + iy * p; h = float(CB['H'][ix, iy])
            if h <= 0 or x0 + w < xl[0] or x0 > xl[1] or y0 + w < yl[0] or y0 > yl[1]:
                continue
            x0, x1 = max(x0, xl[0]), min(x0 + w, xl[1]); y0, y1 = max(y0, yl[0]), min(y0 + w, yl[1])
            polys += [[(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)], [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)],
                      [(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)]]
            cols += ['#f4f5f7', '#eceef1', '#fafbfc']
    return polys, cols


def clip(p, q, xl, yl):
    p = np.asarray(p, float); q = np.asarray(q, float); d = q - p; t0, t1 = 0.0, 1.0
    for pk, qk in ((-d[0], p[0] - xl[0]), (d[0], xl[1] - p[0]), (-d[1], p[1] - yl[0]), (d[1], yl[1] - p[1])):
        if pk == 0:
            if qk < 0: return None
        else:
            r = qk / pk
            if pk < 0: t0 = max(t0, r)
            else: t1 = min(t1, r)
    return None if t0 > t1 else (p + t0 * d, p + t1 * d)


def inb(x, y, xl, yl, m=0.0):
    return xl[0] - m <= x <= xl[1] + m and yl[0] - m <= y <= yl[1] + m


# ---------------------------------------------------------------- 3D scene
def setup(ax, xl, yl, elev=24, azim=-62, zoom=1.0, zr=0.42):
    ax.computed_zorder = False
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(0, 130)
    span = max(xl[1] - xl[0], yl[1] - yl[0])
    ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zr * span), zoom=zoom); ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6.5, pad=-1.5); ax.set_zticks([0, 60, 120])
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    bf = FuncFormatter(lambda v, p: rf'$\mathbf{{{v:.0f}}}$')
    ax.xaxis.set_major_formatter(bf); ax.yaxis.set_major_formatter(bf)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor('#bbbbbb'); a._axinfo['grid'].update(linewidth=0.25, color='#dddddd')
    ax.set_xlabel(r'$\mathbf{x}$ (m)', fontsize=7.5, labelpad=-4, fontweight='bold'); ax.set_ylabel(r'$\mathbf{y}$ (m)', fontsize=7.5, labelpad=-4, fontweight='bold')
    ax.set_zticklabels([r'$\mathbf{0}$', r'$\mathbf{60}$', r'$\mathbf{120}$'])
    ax.text(xl[1], yl[1], 142, r'$\mathbf{z}$ (m)', fontsize=7, ha='left', va='bottom', zorder=2, fontweight='bold')


def route3d(D):
    p = D['path']; mid = np.c_[p, np.full(len(p), H)]
    return np.vstack([[p[0][0], p[0][1], 0.0], mid, [p[-1][0], p[-1][1], 0.0]])


def scene(ax, plat, st, xl, yl, gue=False, bld_alpha=0.38, lw=1.1, only_stop=None):
    """3D lines (drawn before projection is fixed). Returns the plan. only_stop: draw only that hovering point's associations."""
    D = plan(plat, st); m = D['m']; stops = D['stops']
    if only_stop is not None:
        D['links'] = [l for l in D['links'] if l['stop'] == only_stop]
    pol, col = buildings(xl, yl)
    if SHOW_BUILDINGS: ax.add_collection3d(Poly3DCollection(pol, facecolors=col, edgecolors='#dcdfe3', linewidths=0.1, zorder=0))
    P3 = route3d(D)
    for a, b in zip(P3[:-1], P3[1:]):
        cs = clip(a, b, xl, yl)
        if cs is not None:
            vert = abs(a[2] - b[2]) > 1
            ax.plot(*np.array(cs).T, color='k' if vert else COL['route'], lw=1.0 if vert else lw, ls=(0, (1, 1.3)) if vert else '-', zorder=6,
                    solid_capstyle='round')
    if inb(*m['dest'], xl, yl):
        ax.plot([m['dest'][0]] * 2, [m['dest'][1]] * 2, [0, H], color=I.C['dest'], lw=0.6, ls=':', zorder=4)
    ur, rc, dr, st_ = [], [], [], []
    for l in D['links']:
        g = l['g']; s = stops[l['stop']]
        if l['ris'] is not None:
            c = l['ris']
            if inb(c[0], c[1], xl, yl):
                st_.append([(c[0], c[1], 0), tuple(c)])
            cs = clip(c, (s[0], s[1], H), xl, yl)
            if cs is not None: ur.append([tuple(cs[0]), tuple(cs[1])])
            u = rep_gue(plat, l)                        # signal from one user of the cell to the RIS
            cs = clip(u, c, xl, yl)
            if cs is not None: rc.append([tuple(cs[0]), tuple(cs[1])])
        else:
            u = rep_gue(plat, l)                        # direct signal from one user to the UAV
            cs = clip(u, (s[0], s[1], H), xl, yl)
            if cs is not None: dr.append([tuple(cs[0]), tuple(cs[1])])
    if st_: ax.add_collection3d(Line3DCollection(st_, colors=I.C['ris'], linewidths=0.7, zorder=3))
    if rc: ax.add_collection3d(Line3DCollection(rc, colors=COL['rc'], linewidths=RC_LW, zorder=4))
    if dr: ax.add_collection3d(Line3DCollection(dr, colors=COL['direct'], linewidths=0.7, linestyles=(0, (1, 1.2)), zorder=4))
    if ur: ax.add_collection3d(Line3DCollection(ur, colors=COL['ur'], linewidths=0.75, linestyles=(0, (4, 2)), zorder=5))
    for a in ax.lines + ax.collections:          # the zoomed scene may extend beyond the 2D axes box: never clip it
        a.set_clip_on(False)
    return D


def P2(ax, x, y, z):
    a, b, _ = proj3d.proj_transform(x, y, z, ax.get_proj()); return a, b


def overlay(ax, plat, D, xl, yl, s_ic=10.0, numbers=True, arrows=True, ris_icons=True):
    """2D overlays at projected positions: icons, stop numbers, direction arrows. Call after the layout is final."""
    m = D['m']; stops = D['stops']
    if arrows:
        P3 = route3d(D)
        for a, b in zip(P3[1:-1], P3[2:]):
            if abs(a[2] - b[2]) > 1: continue
            if np.hypot(*(b - a)[:2]) < 230: continue
            cs = clip(a, b, xl, yl)
            if cs is None: continue
            p, q = cs; mid = (p + q) / 2; d = (q - p) / np.linalg.norm(q - p) * 25
            ax.annotate('', P2(ax, *(mid + d)), P2(ax, *(mid - d)), xycoords='data',
                        arrowprops=dict(arrowstyle='-|>,head_length=0.45,head_width=0.22', color=COL['route'], lw=0.0, shrinkA=0, shrinkB=0), zorder=8)
    s_gue = s_ic * GUE_SCALE
    for l in D['links']:
        U = GUE[plat][l['h']]
        for u in U[[inb(v[0], v[1], xl, yl) for v in U]]:
            I.place(ax, 'gue', P2(ax, *u), s_gue, zorder=13)
        u = rep_gue(plat, l); tgt = l['ris'] if l['ris'] is not None else (*stops[l['stop']], H)
        cs = clip(u, tgt, xl, yl)
        if cs is not None and inb(u[0], u[1], xl, yl):
            p3, q3 = np.asarray(cs[0]), np.asarray(cs[1]); col = COL['rc'] if l['ris'] is not None else COL['direct']
            fr = 0.75 if l['ris'] is not None else 0.55
            A = np.array(P2(ax, *(p3 + (q3 - p3) * (fr - 0.05)))); B = np.array(P2(ax, *(p3 + (q3 - p3) * (fr + 0.05))))
            if np.hypot(*(ax.transData.transform(B) - ax.transData.transform(A))) > 1.5:
                ax.annotate('', B, A, xycoords='data', zorder=9, annotation_clip=False,
                            arrowprops=dict(arrowstyle='-|>,head_length=0.4,head_width=0.2', color=col, lw=0, shrinkA=0, shrinkB=0))
    pt = lambda xy: ax.transData.transform(xy) * 72 / ax.figure.dpi
    cells_pt = [pt(P2(ax, l['g'][0], l['g'][1], 0)) for l in D['links'] if inb(l['g'][0], l['g'][1], xl, yl)]
    for l in D['links']:
        g = l['g']
        if inb(g[0], g[1], xl, yl): I.place(ax, 'cell', P2(ax, g[0], g[1], 0), s_ic * 0.95, zorder=9, offset=(-s_ic * 0.75, s_ic * 0.35))  # BTS beside its users
        if ris_icons and l['ris'] is not None and inb(l['ris'][0], l['ris'][1], xl, yl):
            xy = P2(ax, *l['ris']); p = pt(xy); c = pt(P2(ax, g[0], g[1], 0)); need = 1.05 * s_ic
            near = [q for q in cells_pt if np.hypot(*(p - q)) < need]
            if near:                                   # icon would cover a cell icon: move it aside, keep a leader line
                q = min(near, key=lambda q: np.hypot(*(p - q))); d = p - q
                u = d / np.hypot(*d) if np.hypot(*d) > 1 else np.array([1.0, 0.25]) / np.hypot(1, 0.25)
                off = q + u * need * 1.05 - p
                ax.scatter(*l['ris'], s=6, c=I.C['ris'], depthshade=False, zorder=12, linewidths=0, clip_on=False)
                I.place(ax, RIS_ICON[plat], xy, s_ic, zorder=12, offset=tuple(off), lead=I.C['ris']); l['icon_off'] = tuple(off)
            else:
                I.place(ax, RIS_ICON[plat], xy, s_ic, zorder=12); l['icon_off'] = (0.0, 0.0)
    if inb(*m['depot'], xl, yl): I.place(ax, 'depot', P2(ax, m['depot'][0], m['depot'][1], 0), s_ic * 1.35, zorder=15)
    if inb(*m['dest'], xl, yl): I.place(ax, 'destination', P2(ax, m['dest'][0], m['dest'][1], H), s_ic * 1.35, zorder=15)
    placed = []
    for k in sorted(range(len(stops)), key=lambda i: D['rank'].get(i, 0)):
        s = stops[k]
        if not inb(s[0], s[1], xl, yl): continue
        sh = D['nmem'][k] > 1; xy = P2(ax, s[0], s[1], H)
        I.place(ax, 'uav_shared' if sh else 'uav', xy, s_ic * (1.35 if sh else 1.0), zorder=16 if sh else 14)
        if numbers:
            px = ax.transData.transform(xy) * 72 / ax.figure.dpi; r = s_ic * (0.68 if sh else 0.52) + 1
            for off in [(0, r + 3), (r + 4, 0), (-r - 4, 0), (0, -r - 7), (r + 4, r), (-r - 4, r), (r + 4, -r), (-r - 4, -r)]:
                p = px + np.array(off)
                if all(np.abs(p - q).max() > 7.5 for q in placed): break
            placed.append(p)
            ax.annotate(str(D['rank'][k]), xy, xytext=off, textcoords='offset points', ha='center', va='center',
                        fontsize=6.0, fontweight='bold', color='#111', zorder=17, bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='#999', lw=0.3, alpha=0.9))


def legend_handles(plat, gue=False, direct=True):
    h = [(I.Icon('depot'), 'Depot (mission origin)'), (I.Icon('destination'), 'Destination'),
         (I.Icon('cell'), 'BTS (congested cell)'), (I.Icon('gue'), 'GUE (user)'), (I.Icon(RIS_ICON[plat]), f'Selected {PN[plat]}'),
         (I.Icon('uav'), 'UAV hovering point'), (I.Icon('uav_shared'), 'Shared UAV hovering point'),
         (Line2D([], [], color=COL['route'], lw=1.3, marker=r'$\rightarrow$', markersize=7, markevery=[1]), 'UAV trajectory ($H$ = 120 m)'),
         (Line2D([], [], color='k', lw=0.9, ls=':'), 'Climb / descent at depot'),
         (Line2D([], [], color=COL['ur'], lw=1.0, ls=(0, (4, 2))), 'UAV–RIS link'),
         (Line2D([], [], color=COL['rc'], lw=RC_LEG), 'GUE–RIS link')]
    if direct: h.append((Line2D([], [], color=COL['direct'], lw=1.0, ls=(0, (1, 1.2))), 'Direct GUE–UAV link'))
    if SHOW_BUILDINGS: h.append((Patch(fc=COL['bld'], ec='#c9cdd2', lw=0.4), 'Buildings'))
    return [a for a, _ in h], [b for _, b in h]


def fig_legend(f, plat, rect_y, ncol=6, gue=False, direct=True, fs=7.2):
    hs, ls = legend_handles(plat, gue, direct)
    lg = f.legend(hs, ls, handler_map=I.HMAP, loc='upper center', bbox_to_anchor=(0.5, rect_y), ncol=ncol, frameon=True, fancybox=False,
                  edgecolor='#444', framealpha=1, fontsize=fs, handlelength=2.2, handleheight=1.5, columnspacing=1.0, borderpad=0.5,
                  labelspacing=0.35)
    lg.get_frame().set_linewidth(0.6); return lg


def stats_text(D, extra=True):
    r = D['rec']; ris = sum(l['ris'] is not None for l in D['links']); sh = int((D['nmem'] > 1).sum())
    t = (f"$E_{{\\rm UAV}}$ = {r.E_UAV_kJ:.1f} kJ\n$L$ = {r.L_flight_km:.2f} km\n{len(D['stops'])} hovering points ({sh} shared)")
    if extra: t += f"\n{ris}/{len(D['links'])} cells via RIS"
    return t


def statbox(ax, D, xy=(0.02, 0.98), fs=6.8, extra=True):
    ax.text2D(xy[0], xy[1], stats_text(D, extra), transform=ax.transAxes, ha='left', va='top', fontsize=fs, zorder=30,
              bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#888', lw=0.5))


FULL = ((-80, 2900), (450, 3100))
GUE_SCALE = 0.42
RC_LW, RC_LEG = 0.9, 1.1
METHODS = ['none', 'random', 'coverage', 'proposed']


# ---------------------------------------------------------------- Fig. A/B: per-platform comparison (2 x 2)
def fig_platform(plat, f=None):
    own = f is None
    if own: f = plt.figure(figsize=(7.16, 7.2))
    axs = []
    pos = [(-0.01, 0.505, 0.49, 0.37), (0.475, 0.505, 0.49, 0.37), (-0.01, 0.065, 0.49, 0.37), (0.475, 0.065, 0.49, 0.37)]
    Ds = []
    for (x, y, w, h), st, lab in zip(pos, ['none', 'random', 'coverage', 'proposed'], 'abcd'):
        ax = f.add_axes([x, y, w, h], projection='3d'); setup(ax, *FULL, elev=30, zoom=1.14)
        Ds.append(scene(ax, plat, st, *FULL)); axs.append(ax)
        ax.patch.set_visible(False)
    fig_legend(f, plat, 0.995, ncol=3, fs=9.2)
    f.canvas.draw()
    for ax, D in zip(axs, Ds):
        overlay(ax, plat, D, *FULL, s_ic=8.5)
        r = D['rec']; ris = sum(l['ris'] is not None for l in D['links']); sh = int((D['nmem'] > 1).sum())
        (x, y, w, h), st, lab = pos[axs.index(ax)], ['none', 'random', 'coverage', 'proposed'][axs.index(ax)], 'abcd'[axs.index(ax)]
        name = PANEL[st].replace(' ', '\\ ').replace('-', '\\text{-}')
        txt = f"$\\mathbf{{({lab})\\ {name}}}$"
        cx = 0.25 if x < 0.4 else 0.75              # centred under each half of the page; one font size for all panels
        f.text(cx, y + 0.004, txt, fontsize=8.4, ha='center', va='top', zorder=50, linespacing=1.35)
    if not own: return axs, Ds
    f.savefig(f'{OUT}/fig_{plat}_methods_3d.svg'); f.savefig(f'{OUT}/fig_{plat}_methods_3d.eps'); f.savefig(f'{OUT}/fig_{plat}_methods_3d.png', dpi=600); plt.close(f)


# ---------------------------------------------------------------- Proposed: overview with zoom windows and callouts
ZOOM = {'facade': {'A': ((620, 1700), (1100, 1960)), 'B': ((1480, 2850), (640, 1520)), 'C': ((1450, 2350), (1960, 3060))},
        'ground': {'A': ((560, 1700), (1100, 1960)), 'B': ((1480, 2850), (600, 1520)), 'C': ((1450, 2350), (1960, 3060))}}


def zoom_rect(ax, xl, yl, lab):
    xs = [xl[0], xl[1], xl[1], xl[0], xl[0]]; ys = [yl[0], yl[0], yl[1], yl[1], yl[0]]
    ax.plot(xs, ys, [1] * 5, color='#222', lw=0.9, ls=(0, (5, 2)), zorder=9, clip_on=False)


def fig_overview(plat):
    f = plt.figure(figsize=(7.16, 5.9)); ax = f.add_axes([0.0, 0.0, 1.0, 0.80], projection='3d')
    setup(ax, *FULL, elev=34, azim=-62, zoom=1.22); D = scene(ax, plat, 'proposed', *FULL, lw=1.3)
    for k, (xl, yl) in ZOOM[plat].items():
        zoom_rect(ax, xl, yl, k)
    fig_legend(f, plat, 0.995, ncol=3, fs=9.2, direct=False)
    f.canvas.draw(); overlay(ax, plat, D, *FULL, s_ic=11.5)
    for k, (xl, yl) in ZOOM[plat].items():
        cx, cy, dx, dy, ha, va = (xl[1], yl[1], 3, 3, 'left', 'bottom') if k == 'C' else (xl[0], yl[0], -3, -3, 'right', 'top')
        ax.annotate(f'Zoom {k}', P2(ax, cx, cy, 0), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=7.5, fontweight='bold',
                    color='#222', zorder=30, bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='#222', lw=0.6))
    m = D['m']; r = D['rec']
    call = dict(fontsize=8, zorder=31, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#666', lw=0.5),
                arrowprops=dict(arrowstyle='-', color='#444', lw=0.6, shrinkA=0, shrinkB=6))
    ax.annotate('Depot: take-off and\nvertical climb to 120 m', P2(ax, m['depot'][0], m['depot'][1], 0), xytext=(0.03, 0.30), textcoords='axes fraction', **call)
    ax.annotate('Destination (delivery)', P2(ax, m['dest'][0], m['dest'][1], H), xytext=(0.36, 0.86), textcoords='axes fraction', **call)
    k = max(np.flatnonzero(D['nmem'] == D['nmem'].max()), key=lambda i: D['stops'][i][0] + D['stops'][i][1]); s = D['stops'][k]
    ax.annotate(f'Shared hovering point:\none UAV position serves\n{D["nmem"][k]} cells through RIS', P2(ax, s[0], s[1], H), xytext=(0.66, 0.86),
                textcoords='axes fraction', **call)
    t = (f"{PN[plat]}, Proposed\n$E_{{\\rm UAV}}$ = {r.E_UAV_kJ:.1f} kJ\n  flight {r.E_flight_kJ:.1f} | hover {r.E_hover_kJ:.1f} | comm. {r.E_comm_kJ:.2f}\n"
         f"$L$ = {r.L_flight_km:.2f} km, $T$ = {r.T_mission_s:.0f} s\n{len(D['stops'])} hovering points ({int((D['nmem'] > 1).sum())} shared)\n"
         f"Numbers: visiting order")
    ax.text2D(0.0, 1.0, t, transform=ax.transAxes, ha='left', va='top', fontsize=8, zorder=30, linespacing=1.2,
              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#888', lw=0.5))
    f.savefig(f'{OUT}/fig_{plat}_proposed_overview.svg'); f.savefig(f'{OUT}/fig_{plat}_proposed_overview.eps'); f.savefig(f'{OUT}/fig_{plat}_proposed_overview.png', dpi=600); plt.close(f)


# ---------------------------------------------------------------- Proposed: zoomed sections (A, B, C) with GUEs and RIS heights
def fig_zooms(plat):
    f = plt.figure(figsize=(7.16, 7.4)); axs = []; Ds = []
    pos = {'A': (0.0, 0.40, 0.5, 0.42), 'B': (0.5, 0.40, 0.5, 0.42), 'C': (0.25, 0.01, 0.5, 0.42)}
    for k, (x, y, w, h) in pos.items():
        xl, yl = ZOOM[plat][k]; ax = f.add_axes([x, y, w, h], projection='3d'); setup(ax, xl, yl, elev=28, azim=-60, zoom=1.0, zr=0.55)
        Ds.append(scene(ax, plat, 'proposed', xl, yl, gue=True, bld_alpha=0.3, lw=1.4)); axs.append(ax)
        ax.set_title(f'Zoom {k}', y=0.97, fontsize=10, fontweight='bold')
    fig_legend(f, plat, 0.995, ncol=3, fs=9.2, direct=False)
    f.canvas.draw()
    for (k, _), ax, D in zip(pos.items(), axs, Ds):
        xl, yl = ZOOM[plat][k]; overlay(ax, plat, D, xl, yl, s_ic=11.5)
        for l in D['links']:
            if l['ris'] is None or not inb(l['ris'][0], l['ris'][1], xl, yl): continue
            c = l['ris']
            if plat == 'ground': continue
            o = l.get('icon_off', (0, 0))
            ax.annotate(f"{c[2]:.0f} m", P2(ax, *c), xytext=(o[0], o[1] + 7.5), textcoords='offset points', fontsize=6.6, color=I.C['ris'], zorder=25,
                        fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.8))
        for l in D['links']:
            g = l['g']
            if not inb(g[0], g[1], xl, yl): continue
            ax.annotate(f"$\\tau$={l['tau']:.1f} s", P2(ax, g[0], g[1], 0), xytext=(0, -7.5), textcoords='offset points', ha='center', va='top',
                        fontsize=6.4, color='#333', zorder=25, bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.8))
    f.text(0.5, 0.004, ('Labels above RIS: mounting height. ' if plat == 'facade' else 'All ground-mobile RIS at 5 m. ') +
           '$\\tau$: service time of the cell at its hovering point. Numbers: visiting order.',
           ha='center', fontsize=8, color='#333')
    f.savefig(f'{OUT}/fig_{plat}_proposed_zooms.svg'); f.savefig(f'{OUT}/fig_{plat}_proposed_zooms.eps'); f.savefig(f'{OUT}/fig_{plat}_proposed_zooms.png', dpi=600); plt.close(f)


# ---------------------------------------------------------------- Proposed: association snapshots (one hovering point each)
def snapshot_stops(D):
    sh = [k for k in np.argsort(-D['nmem']) if D['nmem'][k] > 1][:2]
    single = [k for k in range(len(D['stops'])) if D['nmem'][k] == 1 and any(l['stop'] == k and l['ris'] is not None for l in D['links'])]
    return [int(k) for k in sh] + [int(single[0])] if single else [int(k) for k in sh]


def fig_snapshots(plat):
    D = plan(plat, 'proposed'); ks = snapshot_stops(D)[:3]
    f = plt.figure(figsize=(7.16, 5.3)); axs = []; win = []; Dks = []
    for j, k in enumerate(ks):
        s = D['stops'][k]; L = [l for l in D['links'] if l['stop'] == k]
        pts = np.array([s] + [l['g'] for l in L] + [l['ris'][:2] for l in L if l['ris'] is not None])
        c = (pts.min(0) + pts.max(0)) / 2; half = max(170.0, (pts.max(0) - pts.min(0)).max() / 2 + 70)
        xl, yl = (c[0] - half, c[0] + half), (c[1] - half, c[1] + half); win.append((xl, yl))
        ax = f.add_axes([j / 3 + 0.005, 0.31, 1 / 3 - 0.03, 0.47], projection='3d'); setup(ax, xl, yl, elev=22, azim=-58, zoom=0.92, zr=0.75)
        Dk = scene(ax, plat, 'proposed', xl, yl, gue=True, bld_alpha=0.25, lw=1.4, only_stop=k); axs.append(ax); Dks.append(Dk)
        ax.set_title(f'({"abc"[j]}) Hovering point {D["rank"][k]}' + (' (shared)' if D['nmem'][k] > 1 else ''), y=1.03, fontsize=9.5, fontweight='bold')
    fig_legend(f, plat, 0.995, ncol=3, fs=8.4, direct=False)
    f.canvas.draw()
    for j, (ax, k, (xl, yl)) in enumerate(zip(axs, ks, win)):
        overlay(ax, plat, Dks[j], xl, yl, s_ic=14, arrows=False)
        s = D['stops'][k]; rows = []
        for l in [l for l in D['links'] if l['stop'] == k]:
            g = l['g']
            if l['ris'] is not None:
                c = l['ris']; dh = np.hypot(*(s - c[:2])); el = np.degrees(np.arctan2(H - c[2], dh)); dgc = np.hypot(*(g - c[:2]))
                rows.append(f"Cell {l['cell']}: RIS at {c[2]:.0f} m, {dgc:.0f} m from cell centre\n  RIS–UAV {np.hypot(dh, H - c[2]):.0f} m, elevation {el:.0f}°; $\\tau$ = {l['tau']:.1f} s")
            else:
                rows.append(f"Cell {l['cell']}: direct link; $\\tau$ = {l['tau']:.1f} s")
        f.text(j / 3 + 0.012, 0.29, '\n'.join(rows), ha='left', va='top', fontsize=7, color='#222', linespacing=1.25,
               bbox=dict(boxstyle='round,pad=0.3', fc='#f7f7f7', ec='#999', lw=0.4))
    T = dict(bbox_inches='tight', pad_inches=0.03)
    f.savefig(f'{OUT}/fig_{plat}_proposed_snapshots.svg', **T); f.savefig(f'{OUT}/fig_{plat}_proposed_snapshots.eps', **T); f.savefig(f'{OUT}/fig_{plat}_proposed_snapshots.png', dpi=600, **T); plt.close(f)


# ---------------------------------------------------------------- Proposed: overview with Zoom A only, and Zoom A in the same figure
def fig_overview_zoomA(plat):
    from matplotlib.lines import Line2D as L2
    f = plt.figure(figsize=(7.16, 9.0))
    ax = f.add_axes([0.0, 0.47, 1.0, 0.40], projection='3d')
    setup(ax, *FULL, elev=34, azim=-62, zoom=1.22); D = scene(ax, plat, 'proposed', *FULL, lw=1.3)
    xa, ya = ZOOM[plat]['A']; zoom_rect(ax, xa, ya, 'A')
    az = f.add_axes([0.1, 0.035, 0.8, 0.40], projection='3d'); setup(az, xa, ya, elev=28, azim=-60, zoom=1.0, zr=0.55)
    DA = scene(az, plat, 'proposed', xa, ya, gue=True, bld_alpha=0.3, lw=1.4)
    fig_legend(f, plat, 0.995, ncol=3, fs=9.2, direct=False)
    f.canvas.draw()
    overlay(ax, plat, D, *FULL, s_ic=11.5); overlay(az, plat, DA, xa, ya, s_ic=12.5)
    # overview annotations
    ax.annotate('Zoom A', P2(ax, xa[0], ya[0], 0), xytext=(-3, -3), textcoords='offset points', ha='right', va='top', fontsize=8, fontweight='bold',
                color='#222', zorder=30, bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='#222', lw=0.6))
    m = D['m']; r = D['rec']
    call = dict(fontsize=8, zorder=31, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#666', lw=0.5),
                arrowprops=dict(arrowstyle='-', color='#444', lw=0.6, shrinkA=0, shrinkB=6))
    ax.annotate('Destination', P2(ax, m['dest'][0], m['dest'][1], H), xytext=(0.505, 0.93), textcoords='axes fraction', **call)
    k = max(np.flatnonzero(D['nmem'] == D['nmem'].max()), key=lambda i: D['stops'][i][0] + D['stops'][i][1]); s = D['stops'][k]
    ax.annotate(f'Shared hovering point:\none UAV position serves\n{D["nmem"][k]} cells through RIS', P2(ax, s[0], s[1], H), xytext=(0.76, 0.86),
                textcoords='axes fraction', **call)
    t = (f"{PN[plat]}, Proposed\n$E_{{\\rm UAV}}$ = {r.E_UAV_kJ:.1f} kJ\n  flight {r.E_flight_kJ:.1f} | hover {r.E_hover_kJ:.1f} | comm. {r.E_comm_kJ:.2f}\n"
         f"$L$ = {r.L_flight_km:.2f} km, $T$ = {r.T_mission_s:.0f} s\n{len(D['stops'])} hovering points ({int((D['nmem'] > 1).sum())} shared)\n"
         f"Numbers: visiting order")
    ax.text2D(0.0, 1.0, t, transform=ax.transAxes, ha='left', va='top', fontsize=8, zorder=30, linespacing=1.2,
              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#888', lw=0.5))
    # Zoom A labels (as in the zoom figure)
    for l in DA['links']:
        if plat == 'facade' and l['ris'] is not None and inb(l['ris'][0], l['ris'][1], xa, ya):
            o = l.get('icon_off', (0, 0))
            az.annotate(f"{l['ris'][2]:.0f} m", P2(az, *l['ris']), xytext=(o[0], o[1] + 8), textcoords='offset points', fontsize=7, color=I.C['ris'], zorder=25,
                        fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.8))
        g = l['g']
        if inb(g[0], g[1], xa, ya):
            az.annotate(f"$\\tau$={l['tau']:.1f} s", P2(az, g[0], g[1], 0), xytext=(0, -11), textcoords='offset points', ha='center', va='top',
                        fontsize=7, color='#333', zorder=25, bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.8))
    az.text2D(0.5, 0.97, 'Zoom A', transform=az.transAxes, ha='center', va='top', fontsize=10, fontweight='bold', zorder=30,
              bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='#222', lw=0.6))
    # connector lines: Zoom A box (overview) -> frame around the zoom panel
    from matplotlib.patches import Rectangle as R2
    tofig = lambda a, xy: f.transFigure.inverted().transform(a.transData.transform(xy))
    corners = [tofig(ax, P2(ax, x, y, 0)) for x in xa for y in ya]
    lo = sorted(corners, key=lambda p: p[1])[:2]; lo.sort(key=lambda p: p[0])            # two lowest box corners, left -> right
    fx0, fy0, fx1, fy1 = 0.06, 0.03, 0.94, 0.425
    f.add_artist(R2((fx0, fy0), fx1 - fx0, fy1 - fy0, transform=f.transFigure, fill=False, ec='#222', lw=0.8, ls=(0, (5, 2)), zorder=1))
    for p, q in ((lo[0], (fx0, fy1)), (lo[1], (fx1, fy1))):
        f.add_artist(L2([p[0], q[0]], [p[1], q[1]], transform=f.transFigure, color='#555', lw=0.7, ls=(0, (5, 2)), zorder=-1))
    ax.patch.set_visible(False); az.patch.set_visible(False)
    f.text(0.5, 0.003, ('Labels above RIS: mounting height. ' if plat == 'facade' else 'All ground-mobile RIS at 5 m. ') +
           '$\\tau$: service time of the cell at its hovering point.', ha='center', fontsize=8, color='#333')
    base = f'{OUT}/fig_{plat}_proposed_overview_zoomA'
    f.savefig(base + '.svg'); f.savefig(base + '.eps'); f.savefig(base + '.png', dpi=600); plt.close(f)


# ---------------------------------------------------------------- 2 x 3 compact: four methods + Zoom A of the proposed route, legend top-right
def fig_platform_zoomA(plat):
    """Legend on top; row 1: (a)-(c) at one-third width; row 2: (d) Proposed and (e) its Zoom A at half width."""
    f = plt.figure(figsize=(7.16, 6.3)); axs = []; Ds = []
    R1, R2 = (0.575, 0.36), (0.075, 0.43)                      # (bottom, height) of each row
    pos = [(-0.016, R1[0], 0.333, R1[1]), (0.314, R1[0], 0.333, R1[1]), (0.640, R1[0], 0.333, R1[1]), (-0.01, R2[0], 0.50, R2[1])]
    for (x, y, w, h), st in zip(pos, METHODS):
        ax = f.add_axes([x, y, w, h], projection='3d'); setup(ax, *FULL, elev=30, zoom=0.98 if st != 'proposed' else 1.05)
        Ds.append(scene(ax, plat, st, *FULL, lw=1.1 if st != 'proposed' else 1.3)); axs.append(ax); ax.patch.set_visible(False)
        ax.tick_params(labelsize=6.2, pad=-2); ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    xa, ya = ZOOM[plat]['A']
    zoom_rect(axs[3], xa, ya, 'A')
    az = f.add_axes([0.495, R2[0], 0.50, R2[1]], projection='3d'); setup(az, xa, ya, elev=28, azim=-60, zoom=1.05, zr=0.55)
    az.tick_params(labelsize=6.2, pad=-2); az.xaxis.label.set_size(7); az.yaxis.label.set_size(7); az.patch.set_visible(False)
    DA = scene(az, plat, 'proposed', xa, ya, gue=True, bld_alpha=0.3, lw=1.2)
    hs, ls = legend_handles(plat)
    lg = f.legend(hs, ls, handler_map=I.HMAP, loc='upper center', bbox_to_anchor=(0.5, 0.997), ncol=4, frameon=True, fancybox=False,
                  edgecolor='#444', framealpha=1, fontsize=8, handlelength=2.0, handleheight=1.4, borderpad=0.45, labelspacing=0.3,
                  columnspacing=0.9)
    lg.get_frame().set_linewidth(0.6)
    f.canvas.draw()
    for ax, D, (x, y, W, _), st, lab in zip(axs, Ds, pos, METHODS, 'abcd'):
        overlay(ax, plat, D, *FULL, s_ic=7.8 if st != 'proposed' else 9.5)
        r = D['rec']; sh = int((D['nmem'] > 1).sum())
        name = PANEL[st].replace(' ', '\\ ').replace('-', '\\text{-}')
        txt = (f"$\\mathbf{{({lab})\\ {name}}}$\n$E_{{\\rm UAV}}$ = {r.E_UAV_kJ:.1f} kJ | $L$ = {r.L_flight_km:.2f} km\n"
               f"{len(D['stops'])} hovering points ({sh} shared)")
        if st == 'proposed': txt = txt.replace('\n$E', ' $E', 0)
        f.text(x + W / 2 + 0.006, y + 0.008, txt, fontsize=8, ha='center', va='top', zorder=50, linespacing=1.3)
    axs[3].annotate('A', P2(axs[3], xa[0], ya[0], 0), xytext=(-2, -2), textcoords='offset points', ha='right', va='top', fontsize=8.5,
                    fontweight='bold', zorder=30, bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='#222', lw=0.5))
    overlay(az, plat, DA, xa, ya, s_ic=11)
    for l in DA['links']:
        if plat == 'facade' and l['ris'] is not None and inb(l['ris'][0], l['ris'][1], xa, ya):
            o = l.get('icon_off', (0, 0))
            az.annotate(f"{l['ris'][2]:.0f} m", P2(az, *l['ris']), xytext=(o[0], o[1] + 7.5), textcoords='offset points', fontsize=6.8, color=I.C['ris'],
                        zorder=25, fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.05', fc='white', ec='none', alpha=0.8))
    f.text(0.495 + 0.25, R2[0] + 0.008, "$\\mathbf{(e)\\ Zoom\\ A\\ of\\ (d)}$\n" +
           ("Labels: RIS mounting height" if plat == 'facade' else "All RIS at 5 m"), fontsize=8, ha='center', va='top', linespacing=1.3)
    base = f'{OUT}/fig_{plat}_methods_3d_zoomA'
    f.savefig(base + '.svg'); f.savefig(base + '.eps'); f.savefig(base + '.png', dpi=600); plt.close(f)


# ---------------------------------------------------------------- approved 2 x 2 figure, unchanged, with (e) Zoom A of (d) added underneath
def fig_platform_plus_zoomA(plat, eh=2.75, box=True):
    F0 = plt.figure(figsize=(7.16, 7.2 + eh))
    top, bot = F0.subfigures(2, 1, height_ratios=[7.2, eh], hspace=0, wspace=0)
    axs, Ds = fig_platform(plat, f=top)                      # identical to the approved figure (same code, same inch geometry)
    xa, ya = ZOOM[plat]['A']
    az = bot.add_axes([0.12, 0.10, 0.76, 0.98], projection='3d'); setup(az, xa, ya, elev=28, azim=-60, zoom=1.12, zr=0.5)
    az.patch.set_visible(False)
    DA = scene(az, plat, 'proposed', xa, ya, gue=True, bld_alpha=0.3, lw=1.3)
    if box: zoom_rect(axs[3], xa, ya, 'A')
    F0.canvas.draw()
    overlay(az, plat, DA, xa, ya, s_ic=11)
    if box:
        axs[3].annotate('A', P2(axs[3], xa[0], ya[0], 0), xytext=(-2, -2), textcoords='offset points', ha='right', va='top', fontsize=7.5,
                        fontweight='bold', zorder=30, bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='#222', lw=0.5))
    for l in DA['links']:
        if plat == 'facade' and l['ris'] is not None and inb(l['ris'][0], l['ris'][1], xa, ya):
            o = l.get('icon_off', (0, 0))
            az.annotate(f"{l['ris'][2]:.0f} m", P2(az, *l['ris']), xytext=(o[0], o[1] + 7.5), textcoords='offset points', fontsize=6.8,
                        color=I.C['ris'], zorder=25, fontweight='bold', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.06', fc='white', ec='none', alpha=0.8))
    bot.text(0.5, 0.115, "$\\mathbf{(e)\\ Zoom\\ A\\ of\\ (d)}$\n" +
             ("(labels: RIS mounting height)" if plat == 'facade' else "(all ground-mobile RIS at 5 m)"),
             fontsize=8.4, ha='center', va='top', zorder=50, linespacing=1.35)
    base = f'{OUT}/fig_{plat}_methods_3d_plusZoomA'
    F0.savefig(base + '.svg'); F0.savefig(base + '.eps'); F0.savefig(base + '.png', dpi=600); plt.close(F0)


if __name__ == '__main__':
    what = sys.argv[1:] or ['platform', 'overview', 'zooms', 'snapshots']
    for plat in ('facade', 'ground'):
        if 'platform' in what: fig_platform(plat)
        if 'overview' in what: fig_overview(plat)
        if 'zooms' in what: fig_zooms(plat)
        if 'snapshots' in what: fig_snapshots(plat)
        if 'methodsA' in what: fig_platform_zoomA(plat)
        if 'overviewA' in what and plat == 'facade': fig_overview_zoomA(plat)
        print(plat, 'done', flush=True)
