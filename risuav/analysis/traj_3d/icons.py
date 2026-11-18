"""Vector icon library for the manuscript figures (one definition, used in every figure; only the size changes).
Each icon is drawn in a unit box [-1, 1] x [-1, 1]. Icons are vector patches: they stay sharp in PDF.

Names: depot (house), destination (flag), cell (cellular hotspot / base station), ris_facade (wall-mounted RIS),
ris_ground (vehicle-mounted RIS), uav (hovering point), uav_shared (shared hovering point, highlighted UAV)."""
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Circle, Arc, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea, AnnotationBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.transforms import Affine2D

# manuscript palette (shared with lines in the figures)
C = dict(depot='#2f3e4e', roof='#b0413e', dest='#1f5fbf', cell='#4a4a4a', wave='#c2185b', ris='#0b3d91',
         wall='#b9b4ab', brick='#8f877a', van='#3c5a78', uav='#ef7d00', uav_rotor='#fff1de', shared='#c0392b',
         halo='#ffd400', ink='#222222')


def _grid(x0, y0, w, h, n, m, lw, col='white'):
    out = []
    for i in range(1, n):
        out.append(Line2D([x0 + w * i / n] * 2, [y0, y0 + h], color=col, lw=lw, solid_capstyle='butt'))
    for j in range(1, m):
        out.append(Line2D([x0, x0 + w], [y0 + h * j / m] * 2, color=col, lw=lw, solid_capstyle='butt'))
    return out


def parts(name, s):
    """Artists of icon `name` in unit coordinates; s = icon size in points (sets line widths)."""
    lw = max(0.25, 0.045 * s); ek = dict(ec=C['ink'], lw=lw * 0.8, joinstyle='round')
    if name == 'depot':
        return [Polygon([(-1, 0.05), (0, 0.95), (1, 0.05)], fc=C['roof'], **ek),
                Rectangle((-0.72, -0.95), 1.44, 1.0, fc=C['depot'], **ek),
                Rectangle((-0.18, -0.95), 0.36, 0.58, fc='white', ec='none'),
                Rectangle((0.32, -0.35), 0.24, 0.24, fc='#ffe9a8', ec='none')]
    if name == 'destination':
        return [Rectangle((-0.62, -0.98), 0.13, 1.96, fc=C['ink'], ec='none'),
                Polygon([(-0.5, 0.98), (0.95, 0.58), (-0.5, 0.12)], fc=C['dest'], **ek),
                Rectangle((-0.88, -1.0), 0.66, 0.16, fc=C['ink'], ec='none')]
    if name == 'cell':
        a = [Polygon([(-0.42, -0.98), (0, 0.45), (0.42, -0.98)], fill=False, ec=C['cell'], lw=lw * 1.3, joinstyle='round'),
             Line2D([-0.3, 0.3], [-0.55, -0.55], color=C['cell'], lw=lw), Line2D([-0.18, 0.18], [-0.1, -0.1], color=C['cell'], lw=lw),
             Line2D([-0.3, 0.18], [-0.55, -0.1], color=C['cell'], lw=lw * 0.8), Line2D([0.3, -0.18], [-0.55, -0.1], color=C['cell'], lw=lw * 0.8),
             Circle((0, 0.52), 0.15, fc=C['wave'], ec='none')]
        for r in (0.42, 0.72):
            a += [Arc((0, 0.52), 2 * r, 2 * r, theta1=-50, theta2=50, ec=C['wave'], lw=lw * 1.2),
                  Arc((0, 0.52), 2 * r, 2 * r, theta1=130, theta2=230, ec=C['wave'], lw=lw * 1.2)]
        return a
    if name == 'ris_facade':
        a = [Rectangle((-1.0, -1.0), 0.62, 2.0, fc=C['wall'], ec=C['brick'], lw=lw * 0.6)]
        for j, y in enumerate(np.linspace(-0.75, 0.75, 7)):
            a.append(Line2D([-1.0, -0.38], [y, y], color=C['brick'], lw=lw * 0.5))
        a += [Rectangle((-0.38, -0.08), 0.16, 0.16, fc=C['ink'], ec='none'),
              Rectangle((-0.22, -0.7), 1.12, 1.4, fc=C['ris'], **ek)]
        return a + _grid(-0.22, -0.7, 1.12, 1.4, 3, 4, lw * 0.6)
    if name == 'ris_ground':
        a = [Line2D([-0.2, -0.2], [0.05, 0.3], color=C['ink'], lw=lw * 1.2),
             Polygon([(-0.85, 0.28), (0.45, 0.28), (0.62, 0.98), (-0.68, 0.98)], fc=C['ris'], **ek)]
        for i in range(1, 4):
            t = i / 4; a.append(Line2D([-0.85 + 1.3 * t, -0.68 + 1.3 * t], [0.28, 0.98], color='white', lw=lw * 0.6))
        a.append(Line2D([-0.765, 0.535], [0.63, 0.63], color='white', lw=lw * 0.6))
        a += [Rectangle((-1.0, -0.62), 1.32, 0.7, fc=C['van'], **ek),
              Polygon([(0.32, -0.62), (0.32, 0.08), (0.66, 0.08), (0.98, -0.25), (0.98, -0.62)], fc=C['van'], **ek),
              Polygon([(0.42, -0.2), (0.42, 0.0), (0.62, 0.0), (0.82, -0.2)], fc='#cfe3f5', ec='none'),
              Circle((-0.58, -0.66), 0.24, fc=C['ink'], ec='white', lw=lw * 0.5), Circle((0.58, -0.66), 0.24, fc=C['ink'], ec='white', lw=lw * 0.5)]
        return a
    if name in ('uav', 'uav_shared'):
        sh = name == 'uav_shared'; body = C['shared'] if sh else C['uav']
        a = [Circle((0, 0), 1.0, fc=C['halo'], ec=C['shared'], lw=lw * 0.8, alpha=0.9)] if sh else []
        k = 0.78 if sh else 1.0
        for sx in (-1, 1):
            a.append(Line2D([-0.62 * k, 0.62 * k], [-0.62 * k * sx, 0.62 * k * sx], color=C['ink'], lw=lw * 1.6, solid_capstyle='round'))
        for sx in (-1, 1):
            for sy in (-1, 1):
                a.append(Circle((0.62 * k * sx, 0.62 * k * sy), 0.33 * k, fc=C['uav_rotor'], ec=body, lw=lw * 1.1))
        a.append(FancyBboxPatch((-0.26 * k, -0.26 * k), 0.52 * k, 0.52 * k, boxstyle=f'round,pad=0,rounding_size={0.12 * k}', fc=body, **ek))
        return a
    if name == 'gue':                              # user handset (same as Fig. 1)
        return [FancyBboxPatch((-0.42, -0.9), 0.84, 1.8, boxstyle='round,pad=0,rounding_size=0.22', fc='#84c0fc', ec='#243048', lw=lw * 1.0),
                Line2D([-0.15, 0.15], [-0.7, -0.7], color='#243048', lw=lw * 0.8)]
    raise KeyError(name)


def drawing(name, s):
    """DrawingArea of size s x s points holding the icon."""
    da = DrawingArea(s, s, 0, 0); tr = Affine2D().scale(s / 2.0).translate(s / 2.0, s / 2.0)
    for p in parts(name, s):
        p.set_transform(tr + da.get_transform()); da.add_artist(p)
    return da


def place(ax, name, xy, s, zorder=20, offset=None, lead=None, **kw):
    """Place an icon centred at xy (axes data coordinates; for 3D axes: projected coordinates).
    offset: (dx, dy) points to move the icon off its anchor (a leader line of colour `lead` then joins them)."""
    if offset is not None:
        kw.update(xybox=offset, boxcoords='offset points')
        if lead: kw['arrowprops'] = dict(arrowstyle='-', color=lead, lw=0.6, shrinkA=0, shrinkB=0)
    ab = AnnotationBbox(drawing(name, s), xy, xycoords=ax.transData, frameon=False, box_alignment=(0.5, 0.5),
                        pad=0, zorder=zorder, annotation_clip=False, **kw)
    ax.add_artist(ab); return ab


class Icon:
    def __init__(self, name):
        self.name = name


class HandlerIcon(HandlerBase):
    def create_artists(self, legend, h, xd, yd, w, ht, fs, trans):
        s = min(w, ht) * 1.35; tr = Affine2D().scale(s / 2.0).translate(w / 2.0 - xd, ht / 2.0 - yd) + trans
        out = []
        for p in parts(h.name, s):
            p.set_transform(tr); out.append(p)
        return out


HMAP = {Icon: HandlerIcon()}


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons_out'); os.makedirs(out, exist_ok=True)
    names = ['depot', 'destination', 'cell', 'ris_facade', 'ris_ground', 'uav', 'uav_shared']
    labels = ['Depot', 'Destination', 'Congested cell', 'Facade RIS', 'Ground-mobile RIS', 'Hovering point', 'Shared hovering point']
    for n in names:                       # one file per icon, for system-model and other figures
        f = plt.figure(figsize=(0.5, 0.5)); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.axis('off')
        for p in parts(n, 36):
            p.set_transform(ax.transData); ax.add_artist(p)
        f.savefig(f'{out}/icon_{n}.pdf', transparent=True); f.savefig(f'{out}/icon_{n}.svg', transparent=True)
        f.savefig(f'{out}/icon_{n}.png', dpi=600, transparent=True); plt.close(f)
    f, ax = plt.subplots(figsize=(7.16, 1.3)); ax.set_xlim(0, 7); ax.set_ylim(0, 1); ax.axis('off')
    for i, (n, l) in enumerate(zip(names, labels)):
        for j, s in enumerate((10, 16, 26)):
            place(ax, n, (i + 0.25 + 0.25 * j, 0.62), s)
        ax.text(i + 0.5, 0.12, l, ha='center', fontsize=6.5)
    f.savefig(f'{out}/icon_sheet.pdf'); f.savefig(f'{out}/icon_sheet.png', dpi=300); print('ok')
