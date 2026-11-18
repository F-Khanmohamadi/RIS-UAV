"""NEW (18 Nov). Fig. energy_breakdown as Excel (style of the authors' Fig. 3 workbook): one sheet with the data table and two
native stacked-column charts, (a) Facade RIS and (b) Ground-mobile RIS: flight, hovering and communication energy of the
four methods (mean over the 120 test missions and three seeds), total on top of each bar.
usage: make_breakdown_excel.py <energy_breakdown.csv> <out.xlsx>"""
import os, re, sys, zipfile, numpy as np, pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xlchart import stacked_bar_xml, colL

CSV, OUT = sys.argv[1], sys.argv[2]
FONT = 'Times New Roman'; HF = PatternFill('solid', fgColor='DDEBF7'); S_ = Side(style='thin', color='999999'); B_ = Border(S_, S_, S_, S_)
METH = [('none', 'No RIS'), ('random', 'Random RIS'), ('coverage', 'Coverage-based RIS'), ('proposed', 'Proposed')]
COMP = [('Flight', '7FA7D6'), ('Hovering', 'E8A33D'), ('Communication', '6AA84F')]
PLAT = ['Facade RIS', 'Ground-mobile RIS']; SHEET = 'Energy breakdown'
E_ = pd.read_csv(CSV)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
wb = Workbook(); ws = wb.active; ws.title = SHEET
ws['A1'] = 'Breakdown of the UAV energy into flight, hovering, and communication energy (kJ)'; ws['A1'].font = Font(name=FONT, bold=True, size=12)
ws['A2'] = ('Final evaluation: mean over the 120 test missions per platform and the three training seeds; std = standard deviation '
            'across the seeds (population). Total = flight + hovering + communication (shown above each bar).')
ws['A2'].font = Font(name=FONT, italic=True, size=9); ws['A2'].alignment = Alignment(wrap_text=True, vertical='top')
ws.merge_cells('A2:Q2'); ws.row_dimensions[2].height = 30
r0, r1 = 5, 4 + len(METH)
ws.cell(4, 1, 'Method'); cols = {}; c = 2
for plat in PLAT:
    ws.cell(3, c, plat)
    for comp, _ in COMP:
        cols[(plat, comp)] = c; ws.cell(4, c, f'{comp} (kJ)'); ws.cell(4, c + 1, f'{comp} std'); c += 2
    cols[(plat, 'Total')] = c; ws.cell(4, c, 'Total (kJ)'); c += 1
last = c - 1; data = {}
for i, (m, lab) in enumerate(METH):
    r = r0 + i; ws.cell(r, 1, lab)
    for plat in PLAT:
        tot = 0.0
        for comp, _ in COMP:
            y = E_[(E_.platform == plat) & (E_.method == m) & (E_.component == comp)].iloc[0]
            ws.cell(r, cols[(plat, comp)], float(y.mean_kJ)); ws.cell(r, cols[(plat, comp)] + 1, float(y.std_seeds_kJ))
            data.setdefault((plat, comp), []).append(float(y.mean_kJ)); tot += float(y.mean_kJ)
        ws.cell(r, cols[(plat, 'Total')], tot); data.setdefault((plat, 'Total'), []).append(tot)
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
ws.column_dimensions['A'].width = 20
for j in range(2, last + 1):
    ws.column_dimensions[colL(j)].width = 12
ws.row_dimensions[4].height = 32
CH = {}
for k, (plat, (cc, rr), lab) in enumerate(zip(PLAT, (('A', 13), ('K', 13)), ('(a) Facade-mounted RIS', '(b) Ground-mobile RIS'))):
    ws[f'{cc}{rr - 1}'] = lab; ws[f'{cc}{rr - 1}'].font = Font(name=FONT, bold=True, size=12)
    comps = [dict(name=comp, col=colL(cols[(plat, comp)]), vals=data[(plat, comp)], color=col) for comp, col in COMP]
    CH[k] = stacked_bar_xml(SHEET, [l for _, l in METH], r0, r1, comps,
                            dict(col=colL(cols[(plat, 'Total')]), vals=data[(plat, 'Total')]), 'UAV Energy (kJ)', None, 0, 120, 20)
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
print('written', OUT)
