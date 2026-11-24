
import os, re, sys, zipfile
import numpy as np, pandas as pd
from xml.sax.saxutils import escape

T = pd.read_csv(sys.argv[1]); TPL, OUT = sys.argv[2], sys.argv[3]
YMAX = 280
NOTE = ('Values from the training logs (runs/proposed_{facade,ground}_s{0,1,2}/log.json): mean and population standard deviation '
        'over the three training seeds at each epoch. Training cost = mean cost of the plans sampled in the epoch (UAV energy + '
        'constraint penalty); it is not defined at epoch 0 (cell shows blank, holds #N/A so that the chart skips it). Validation '
        'E_UAV = UAV energy of the greedy policy on the 200 validation missions, before local search. The grey chart-helper '
        'columns only draw the shaded bands (stacked areas built from mean +- std).')


def stats(plat):
    x = T[T.platform == plat]
    g = x.groupby('epoch')[['train_cost_kJ', 'val_E_kJ']].agg(['mean', lambda v: v.std(ddof=0)])
    assert list(g.index) == list(range(51))
    return {k: g[(c, f)].values for k, (c, f) in
            dict(tm=('train_cost_kJ', 'mean'), ts=('train_cost_kJ', '<lambda_0>'),
                 vm=('val_E_kJ', 'mean'), vs=('val_E_kJ', '<lambda_0>')).items()}


def pieces(tm, ts, vm, vs):
    """template helper order: base, val lower, train lower, overlap, gap, val upper, train upper (stacked)."""
    out = dict(base=0.0, vlo=0.0, tlo=0.0, ov=0.0, gap=0.0, vup=0.0, tup=0.0)
    V = (vm - vs, vm + vs) if np.isfinite(vm) and np.isfinite(vs) else None
    Tr = (tm - ts, tm + ts) if np.isfinite(tm) and np.isfinite(ts) else None
    if V is None and Tr is None:
        return out
    if Tr is None:
        out['base'] = V[0]; out['vlo'] = V[1] - V[0]; return out
    if V is None:
        out['base'] = Tr[0]; out['tlo'] = Tr[1] - Tr[0]; return out
    (lo_k, a), (hi_k, b) = sorted((('v', V), ('t', Tr)), key=lambda z: z[1][0])
    out['base'] = a[0]
    if b[0] >= a[1]:                                        # disjoint: lower band, gap, upper band
        out[lo_k + 'lo'] = a[1] - a[0]; out['gap'] = b[0] - a[1]; out[hi_k + 'up'] = b[1] - b[0]
    else:                                                   # overlapping
        out[lo_k + 'lo'] = b[0] - a[0]; top = min(a[1], b[1]); out['ov'] = top - b[0]
        out[('v' if V[1] >= Tr[1] else 't') + 'up'] = max(a[1], b[1]) - top
    return out


F, G = stats('Facade RIS'), stats('Ground-mobile RIS')
cells = {}                                                   # ref -> number, or 'NA' for =NA()
for i in range(51):
    r = 5 + i
    for col, d, k in (('B', F, 'tm'), ('C', F, 'ts'), ('D', F, 'vm'), ('E', F, 'vs'),
                      ('F', G, 'tm'), ('G', G, 'ts'), ('H', G, 'vm'), ('I', G, 'vs')):
        v = d[k][i]; cells[f'{col}{r}'] = float(v) if np.isfinite(v) else ('NA' if k == 'tm' else None)
    for cols, d in ((('Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF'), F), (('AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM'), G)):
        p = pieces(d['tm'][i], d['ts'][i], d['vm'][i], d['vs'][i])
        for c, key in zip(cols, ('base', 'vlo', 'tlo', 'ov', 'gap', 'vup', 'tup')):
            cells[f'{c}{r}'] = float(p[key])

VALS = dict(cells)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
zin = zipfile.ZipFile(TPL); files = {n: zin.read(n) for n in zin.namelist()}

# ---- sheet values
s = files['xl/worksheets/sheet1.xml'].decode('utf-8')


def put(m):
    ref, attrs = m.group(1), m.group(2)
    if ref not in cells:
        return m.group(0)
    st = re.search(r'\ss="\d+"', attrs); st = st.group(0) if st else ''
    v = cells.pop(ref)
    if v is None:
        return f'<c r="{ref}"{st}/>'
    if v == 'NA':
        return f'<c r="{ref}"{st} t="e"><f>NA()</f><v>#N/A</v></c>'
    return f'<c r="{ref}"{st}><v>{repr(v)}</v></c>'


s = re.sub(r'<c r="([A-Z]+\d+)"((?:\s[^>/]*?)?)(?:/>|>.*?</c>)', put, s)
assert not cells, f'cells missing in template: {sorted(cells)[:5]}'
s = re.sub(r'topLeftCell="[A-Z]+\d+"', 'topLeftCell="A1"', s)
s = re.sub(r'<selection [^>]*/>', '<selection activeCell="A1" sqref="A1"/>', s)
cf = ('<conditionalFormatting sqref="B5:I55"><cfRule type="containsErrors" dxfId="0" priority="1">'
      '<formula>ISERROR(B5)</formula></cfRule></conditionalFormatting>')
s = s.replace('</mergeCells>', '</mergeCells>' + cf, 1); assert cf in s
files['xl/worksheets/sheet1.xml'] = s.encode('utf-8')

# ---- error cells shown blank (white font)
st = files['xl/styles.xml'].decode('utf-8')
st = st.replace('<dxfs count="0"/>', '<dxfs count="1"><dxf><font><color rgb="FFFFFFFF"/></font></dxf></dxfs>'); assert 'dxfs count="1"' in st
files['xl/styles.xml'] = st.encode('utf-8')

# ---- note (shared string 1)
ss = files['xl/sharedStrings.xml'].decode('utf-8')
sis = re.findall(r'<si>.*?</si>', ss, flags=re.S)                     # [0] title, [1] note
ss = ss.replace(sis[1], f'<si><t>{escape(NOTE)}</t></si>', 1)
files['xl/sharedStrings.xml'] = ss.encode('utf-8')

# ---- charts: cached values rebuilt from the new cells (Excel 2007 draws series and legend keys from the cache),
#      y-axis title and maximum
import openpyxl
_wb = openpyxl.load_workbook(TPL); _ws = _wb.active
VAL = dict(VALS)                                              # all numeric cells written above (+ epochs in column A)
for i in range(51):
    VAL[f'A{5 + i}'] = i


def col_letters(ref):
    return re.match(r'\$?([A-Z]+)\$?(\d+)', ref).groups()


def numcache(f):
    a, b = f.split('!')[1].split(':')
    (c1, r1), (c2, r2) = col_letters(a), col_letters(b); assert c1 == c2
    rows = range(int(r1), int(r2) + 1); fmt = '0' if c1 == 'A' else '0.0'
    pts = ''.join(f'<c:pt idx="{k}"><c:v>{repr(VAL[f"{c1}{r}"])}</c:v></c:pt>' for k, r in enumerate(rows)
                  if isinstance(VAL.get(f'{c1}{r}'), (int, float)))
    return f'<c:numCache><c:formatCode>{fmt}</c:formatCode><c:ptCount val="{len(rows)}"/>{pts}</c:numCache>'


def strcache(f):
    c, r = col_letters(f.split('!')[1])
    return f'<c:strCache><c:ptCount val="1"/><c:pt idx="0"><c:v>{escape(str(_ws[f"{c}{r}"].value))}</c:v></c:pt></c:strCache>'


for n in ('xl/charts/chart1.xml', 'xl/charts/chart2.xml'):
    c = files[n].decode('utf-8')
    c = re.sub(r'<c:numRef><c:f>([^<]+)</c:f><c:numCache>.*?</c:numCache></c:numRef>',
               lambda m: f'<c:numRef><c:f>{m.group(1)}</c:f>{numcache(m.group(1))}</c:numRef>', c, flags=re.S)
    c = re.sub(r'<c:strRef><c:f>([^<]+)</c:f><c:strCache>.*?</c:strCache></c:strRef>',
               lambda m: f'<c:strRef><c:f>{m.group(1)}</c:f>{strcache(m.group(1))}</c:strRef>', c, flags=re.S)
    c, k = re.subn(r'<a:t>(Facade|Ground-mobile) RIS Energy \(kJ\)</a:t>', '<a:t>UAV Energy (kJ)</a:t>', c); assert k == 1
    c, k = re.subn(r'<c:max val="230"/>', f'<c:max val="{YMAX}"/>', c); assert k == 1
    files[n] = c.encode('utf-8')

# ---- Fig. 3(b): the template chart has no legend; use the legend of Fig. 3(a) (same box, position, font, line samples)
#      and the same series names (Training, Validation)
c1, c2 = files['xl/charts/chart1.xml'].decode('utf-8'), files['xl/charts/chart2.xml'].decode('utf-8')
LEG = re.search(r'<c:legend>.*?</c:legend>', c1, flags=re.S).group(0)
if '<c:legend>' not in c2:
    c2 = c2.replace('</c:plotArea>', '</c:plotArea>' + LEG, 1); assert LEG in c2
c2, k = re.subn(r'<c:tx><c:v>Training cost</c:v></c:tx>', '<c:tx><c:v>Training</c:v></c:tx>', c2); assert k == 1
c2, k = re.subn(r'<c:tx><c:v>Validation E_UAV</c:v></c:tx>', '<c:tx><c:v>Validation</c:v></c:tx>', c2); assert k == 1
files['xl/charts/chart2.xml'] = c2.encode('utf-8')

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in zin.namelist():                                  # same part order as the template
        z.writestr(n, files[n])
print('written', OUT)
