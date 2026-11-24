import os, re, sys, zipfile, numpy as np, pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xlchart import chart_xml, colL

CSV, METRIC, YT, STEP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
FONT = 'Times New Roman'; HF = PatternFill('solid', fgColor='DDEBF7'); S_ = Side(style='thin', color='999999'); B_ = Border(S_, S_, S_, S_)
METH = [('none', 'No RIS', '444444', 'circle'), ('random', 'Random RIS', 'E87BA4', 'triangle'),
        ('coverage', 'Coverage-based RIS', 'ED7D31', 'diamond'), ('proposed', 'Proposed (route-aware)', '2A78D6', 'circle')]
PLAT = ['Facade RIS', 'Ground-mobile RIS']
BINS = ['10-13', '14-17', '18-21', '22-25', '26-30']
SHEET = 'Versus congested cells'

V = pd.read_csv(CSV); V = V[V.metric == METRIC]; assert len(V) == 40, len(V)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
wb = Workbook(); ws = wb.active; ws.title = SHEET
ws['A1'] = f'{YT} versus the number of congested cells I'; ws['A1'].font = Font(name=FONT, bold=True, size=12)
nm = V[(V.platform == PLAT[0]) & (V.method == 'none')].set_index('K_interval').n_missions_per_seed.reindex(BINS)
ws['A2'] = ('Final evaluation: 120 test missions per platform (two unseen layouts) x 3 training seeds. Mean over the missions of '
            'each interval and the three seeds; std (seeds) = standard deviation across the seeds (population). Missions per '
            'interval: ' + ', '.join(f'{b}: {int(n)}' for b, n in zip(BINS, nm)) + '.')
ws['A2'].font = Font(name=FONT, italic=True, size=9); ws['A2'].alignment = Alignment(wrap_text=True, vertical='top')
ws.merge_cells('A2:Q2'); ws.row_dimensions[2].height = 36
r0, r1 = 5, 4 + len(BINS)
ws.cell(4, 1, 'Congested cells I'); cols = {}; c = 2
for plat in PLAT:
    ws.cell(3, c, plat)
    for m, lab, *_ in METH:
        cols[(plat, m)] = c; ws.cell(4, c, f'{lab}: mean'); ws.cell(4, c + 1, f'{lab}: std (seeds)'); c += 2
last = c - 1
data = {}
for i, b in enumerate(BINS):
    ws.cell(r0 + i, 1, b)
    for plat in PLAT:
        for m, *_ in METH:
            y = V[(V.platform == plat) & (V.method == m) & (V.K_interval == b)].iloc[0]
            ws.cell(r0 + i, cols[(plat, m)], float(y['mean'])); ws.cell(r0 + i, cols[(plat, m)] + 1, float(y.std_seeds))
            data.setdefault((plat, m), []).append((float(y['mean']), float(y.std_seeds)))
for row in ws.iter_rows(min_row=3, max_row=r1, max_col=last):
    for cl in row:
        if cl.row == 3 and cl.value is None:
            continue
        cl.font = Font(name=FONT, bold=cl.row in (3, 4) or cl.column == 1, size=10); cl.border = B_
        cl.alignment = Alignment(horizontal='center', vertical='center', wrap_text=cl.row == 4)
        if cl.row == 4:
            cl.fill = HF
        elif cl.row >= r0 and cl.column > 1:
            cl.number_format = '0.00'
for j in range(1, last + 1):
    ws.column_dimensions[colL(j)].width = 13
ws.row_dimensions[4].height = 48

allv = [a for v in data.values() for a, _ in v]
lo, hi = float(np.floor(min(allv) / STEP) * STEP), float(np.ceil(max(allv) / STEP) * STEP)
CH = {}
for k, (plat, (cc, rr), lab) in enumerate(zip(PLAT, (('A', 13), ('L', 13)), ('(a) Facade RIS', '(b) Ground-mobile RIS'))):
    ws[f'{cc}{rr - 1}'] = lab; ws[f'{cc}{rr - 1}'].font = Font(name=FONT, bold=True, size=12)
    ser = [dict(name=name, col=colL(cols[(plat, m)]), vals=[a for a, _ in data[(plat, m)]], color=colr, marker=mk)
           for m, name, colr, mk in METH]                    # no error bars (authors' choice); std stays in the table
    CH[k] = chart_xml(SHEET, BINS, r0, r1, ser, YT, 'Number of Congested Cells I', lo, hi, STEP)
    ph = LineChart(); ws.cell(300 + k, 60, f'CH{k}')
    ph.add_data(Reference(ws, min_col=60, min_row=300 + k, max_row=300 + k)); ph.width, ph.height = 25, 23
    ws.add_chart(ph, f'{cc}{rr}')
wb.save(OUT)

zin = zipfile.ZipFile(OUT); files = {n: zin.read(n) for n in zin.namelist()}; zin.close()
wbr = load_workbook(OUT); n_done = 0
for n in list(files):
    if re.match(r'xl/charts/chart\d+\.xml$', n):
        row = int(re.search(r"\$BH\$(\d+)", files[n].decode('utf-8')).group(1))
        files[n] = CH[int(wbr[SHEET].cell(row, 60).value[2:])].encode('utf-8'); n_done += 1
assert n_done == 2
for n in list(files):
    if re.match(r'xl/worksheets/sheet\d+\.xml$', n):
        s = files[n].decode('utf-8'); s = re.sub(r'<c r="BH\d+"[^>]*?(?:/>|>.*?</c>)', '', s)
        files[n] = re.sub(r'<row r="\d+"[^>]*?(?:/>|></row>)', '', s).encode('utf-8')
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for n, b in files.items():
        z.writestr(n, b)
print('written', OUT, 'y-axis', lo, hi)
