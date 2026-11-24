import os, re, sys, zipfile, numpy as np, pandas as pd
from xml.sax.saxutils import escape
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

IN, OUT = sys.argv[1], sys.argv[2]
FONT = 'Times New Roman'; HF = PatternFill('solid', fgColor='DDEBF7'); S_ = Side(style='thin', color='999999'); B_ = Border(S_, S_, S_, S_)
METH = [('none', 'No RIS', '444444', 'solid', 'circle'), ('random', 'Random RIS', 'E87BA4', 'solid', 'triangle'),
        ('coverage', 'Coverage-based RIS', '6250D6', 'solid', 'diamond'), ('proposed', 'Proposed (route-aware)', '2A78D6', 'solid', 'circle')]
SAV = [('none', 'vs No RIS', '444444', 'square'), ('coverage', 'vs Coverage-based RIS', '6250D6', 'square')]
PLAT = ['Facade RIS', 'Ground-mobile RIS']
PARS = [('N', 'RIS elements N', 'RIS elements N', 'Number of RIS elements N'),
        ('D', 'Data backlog D', 'Data backlog per GUE (Mbit)', 'Data backlog per GUE D (Mbit)')]


def colL(i):
    s = ''
    while i:
        i, r = divmod(i - 1, 26); s = chr(65 + r) + s
    return s


def rich(text, rot=None):
    bp = f'<a:bodyPr rot="{rot}" vert="horz"/>' if rot is not None else '<a:bodyPr/>'
    rp = '<a:solidFill><a:srgbClr val="000000"/></a:solidFill><a:latin typeface="Times New Roman"/>'
    return (f'<c:tx><c:rich>{bp}<a:lstStyle/><a:p><a:pPr><a:defRPr sz="2400" b="1">{rp}</a:defRPr></a:pPr>'
            f'<a:r><a:rPr lang="en-US" sz="2400" b="1">{rp}</a:rPr><a:t>{escape(text)}</a:t></a:r></a:p></c:rich></c:tx>')


TXPR = ('<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="2400" b="1"><a:solidFill><a:srgbClr val="000000"/></a:solidFill>'
        '<a:latin typeface="Times New Roman"/></a:defRPr></a:pPr><a:endParaRPr lang="en-US"/></a:p></c:txPr>')
TXPR_X = TXPR.replace('<a:bodyPr/>', '<a:bodyPr rot="0" vert="horz"/>')
AXLN = '<c:spPr><a:ln w="9525"><a:solidFill><a:srgbClr val="595959"/></a:solidFill><a:prstDash val="solid"/></a:ln></c:spPr>'


def numref(sheet, col, r0, r1, vals, fmt='0.0'):
    pts = ''.join(f'<c:pt idx="{k}"><c:v>{repr(float(v))}</c:v></c:pt>' for k, v in enumerate(vals) if np.isfinite(v))
    return (f"<c:numRef><c:f>'{sheet}'!${col}${r0}:${col}${r1}</c:f><c:numCache><c:formatCode>{fmt}</c:formatCode>"
            f'<c:ptCount val="{len(vals)}"/>{pts}</c:numCache></c:numRef>')


def chart_xml(sheet, cats, r0, r1, series, ytitle, xtitle, ymin, ymax, major, zero_line=False):
    cat = (f"<c:cat><c:strRef><c:f>'{sheet}'!$A${r0}:$A${r1}</c:f><c:strCache><c:ptCount val=\"{len(cats)}\"/>"
           + ''.join(f'<c:pt idx="{k}"><c:v>{escape(c)}</c:v></c:pt>' for k, c in enumerate(cats)) + '</c:strCache></c:strRef></c:cat>')
    sers = ''
    for i, s in enumerate(series):
        err = ''
        if s.get('plus_col'):
            err = ('<c:errBars><c:errDir val="y"/><c:errBarType val="both"/><c:errValType val="cust"/><c:noEndCap val="0"/>'
                   f'<c:plus>{numref(sheet, s["plus_col"], r0, r1, s["plus"])}</c:plus>'
                   f'<c:minus>{numref(sheet, s["minus_col"], r0, r1, s["minus"])}</c:minus>'
                   f'<c:spPr><a:ln w="19050"><a:solidFill><a:srgbClr val="{s["color"]}"/></a:solidFill></a:ln></c:spPr></c:errBars>')
        sers += (f'<c:ser><c:idx val="{i}"/><c:order val="{i}"/><c:tx><c:v>{escape(s["name"])}</c:v></c:tx>'
                 f'<c:spPr><a:ln w="38100"><a:solidFill><a:srgbClr val="{s["color"]}"/></a:solidFill><a:prstDash val="solid"/></a:ln></c:spPr>'
                 f'<c:marker><c:symbol val="{s["marker"]}"/><c:size val="11"/><c:spPr><a:solidFill><a:srgbClr val="{s["color"]}"/></a:solidFill>'
                 f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{s["color"]}"/></a:solidFill></a:ln></c:spPr></c:marker>'
                 f'{err}{cat}<c:val>{numref(sheet, s["col"], r0, r1, s["vals"])}</c:val><c:smooth val="0"/></c:ser>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<c:date1904 val="0"/><c:lang val="en-US"/><c:roundedCorners val="0"/><c:style val="2"/>'
            '<c:chart><c:autoTitleDeleted val="1"/><c:plotArea><c:layout/>'
            f'<c:lineChart><c:grouping val="standard"/><c:varyColors val="0"/>{sers}'
            '<c:marker val="1"/><c:axId val="10"/><c:axId val="100"/></c:lineChart>'
            '<c:catAx><c:axId val="10"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="b"/>'
            f'<c:title>{rich(xtitle)}<c:overlay val="0"/></c:title><c:numFmt formatCode="General" sourceLinked="0"/>'
            f'<c:majorTickMark val="out"/><c:minorTickMark val="none"/><c:tickLblPos val="low"/>{AXLN}'
            f'{TXPR_X}'
            '<c:crossAx val="100"/><c:crosses val="autoZero"/><c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/>'
            '<c:noMultiLvlLbl val="1"/></c:catAx>'
            f'<c:valAx><c:axId val="100"/><c:scaling><c:orientation val="minMax"/><c:max val="{ymax:g}"/><c:min val="{ymin:g}"/></c:scaling>'
            '<c:delete val="0"/><c:axPos val="l"/><c:majorGridlines><c:spPr><a:ln w="9525"><a:solidFill><a:srgbClr val="D9D9D9"/>'
            '</a:solidFill><a:prstDash val="solid"/></a:ln></c:spPr></c:majorGridlines>'
            f'<c:title>{rich(ytitle, rot=-5400000)}<c:overlay val="0"/></c:title><c:numFmt formatCode="0" sourceLinked="0"/>'
            f'<c:majorTickMark val="out"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>{AXLN}{TXPR}'
            f'<c:crossAx val="10"/><c:crosses val="{"autoZero" if zero_line else "min"}"/><c:crossBetween val="between"/>'
            f'<c:majorUnit val="{major:g}"/></c:valAx>'
            '<c:spPr><a:ln><a:solidFill><a:srgbClr val="595959"/></a:solidFill></a:ln></c:spPr></c:plotArea>'
            '<c:legend><c:legendPos val="t"/><c:overlay val="0"/><c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
            f'<a:ln w="9525"><a:solidFill><a:srgbClr val="595959"/></a:solidFill></a:ln></c:spPr>{TXPR}</c:legend>'
            '<c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/></c:chart>'
            '<c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:ln w="9525"><a:solidFill><a:srgbClr val="595959"/>'
            '</a:solidFill></a:ln></c:spPr></c:chartSpace>')


def axis(lo, hi, step):
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
wb = Workbook(); wb.remove(wb.active)
CHARTS = {}
for key, sheet, xshort, xtitle in PARS:
    T = pd.read_csv(os.path.join(IN, f'sensitivity_{key}.csv'))
    T = T[T.policy.str.startswith('fixed')]
    vals = sorted(T.value.unique()); cats = [f'{v:g}' for v in vals]; n = len(vals)
    ws = wb.create_sheet(sheet)
    ws['A1'] = f'Parameter sensitivity: {xtitle}'; ws['A1'].font = Font(name=FONT, bold=True, size=12)
    ws['A2'] = ('Policy of seed 0 trained at the reference values (no retraining); 40 paired test missions per value (first 20 '
                'missions of each test layout); all other parameters at their reference values. Energy: mean over the missions, '
                'CI = half-width of the 95% confidence interval of the mean. Saving of the proposed method: paired, in % of the '
                'baseline mean, with bootstrap 95% confidence interval (error bars: CI lower/upper distance from the mean). '
                'Win rate: share of missions in which the proposed method uses less energy. Reference value: '
                + ('N = 32 x 32 = 1024.' if key == 'N' else 'D = 40 Mbit.'))
    ws['A2'].font = Font(name=FONT, italic=True, size=9); ws['A2'].alignment = Alignment(wrap_text=True, vertical='top')
    ws.merge_cells('A2:AC2'); ws.row_dimensions[2].height = 48
    r0, r1 = 5, 4 + n
    ws.cell(4, 1, xshort)
    cols = {}; c = 2
    for plat in PLAT:
        ws.cell(3, c, plat)
        for m, lab, *_ in METH:
            cols[(plat, m)] = c; ws.cell(4, c, f'{lab}: E_UAV (kJ)'); ws.cell(4, c + 1, f'{lab}: 95% CI (kJ)'); c += 2
        for b, lab, *_ in SAV:
            cols[(plat, 's_' + b)] = c
            ws.cell(4, c, f'Saving {lab} (%)'); ws.cell(4, c + 1, 'error minus (%)'); ws.cell(4, c + 2, 'error plus (%)')
            ws.cell(4, c + 3, f'Win rate {lab} (%)'); c += 4
    last = c - 1
    data = {}
    for plat in PLAT:
        x = T[T.platform == plat]
        for i, v in enumerate(vals):
            r = r0 + i; ws.cell(r, 1, cats[i])
            for m, *_ in METH:
                y = x[(x.method == m) & (x.value == v)].iloc[0]
                ci = 1.96 * y.E_UAV_std / np.sqrt(y.n_missions); cc = cols[(plat, m)]
                ws.cell(r, cc, float(y.E_UAV_kJ)); ws.cell(r, cc + 1, float(ci))
                data.setdefault((plat, m), []).append((float(y.E_UAV_kJ), float(ci)))
            for b, *_ in SAV:
                y = x[(x.method == f'saving_vs_{b}') & (x.value == v)].iloc[0]
                mb = x[(x.method == b) & (x.value == v)].iloc[0].E_UAV_kJ
                lo, hi = 100 * y.ci95_low_kJ / mb, 100 * y.ci95_high_kJ / mb; cc = cols[(plat, 's_' + b)]
                ws.cell(r, cc, float(y.saving_pct)); ws.cell(r, cc + 1, float(y.saving_pct - lo)); ws.cell(r, cc + 2, float(hi - y.saving_pct))
                ws.cell(r, cc + 3, float(100 * y.win_rate))
                data.setdefault((plat, 's_' + b), []).append((float(y.saving_pct), float(y.saving_pct - lo), float(hi - y.saving_pct)))
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
        ws.column_dimensions[colL(j)].width = 13 if j > 1 else 11
    ws.row_dimensions[4].height = 62
    pos = {('E', PLAT[0]): ('A', 12), ('E', PLAT[1]): ('L', 12), ('S', PLAT[0]): ('A', 60), ('S', PLAT[1]): ('L', 60)}
    lab = {('E', PLAT[0]): '(a) Facade RIS', ('E', PLAT[1]): '(b) Ground-mobile RIS',
           ('S', PLAT[0]): '(c) Facade RIS', ('S', PLAT[1]): '(d) Ground-mobile RIS'}
    for (kind, plat), (cc, rr) in pos.items():
        ws[f'{cc}{rr - 1}'] = lab[(kind, plat)]; ws[f'{cc}{rr - 1}'].font = Font(name=FONT, bold=True, size=12)
        mark = f'{kind}{key}{PLAT.index(plat)}'
        if kind == 'E':
            ser = []; allv = []
            for m, name, colr, dash, mk in METH:
                cc_ = cols[(plat, m)]; v = [a for a, _ in data[(plat, m)]]; e = [b for _, b in data[(plat, m)]]
                ser.append(dict(name=name, col=colL(cc_), vals=v, plus_col=colL(cc_ + 1), plus=e, minus_col=colL(cc_ + 1), minus=e,
                                color=colr, marker=mk))
                allv += [a - b for a, b in zip(v, e)] + [a + b for a, b in zip(v, e)]
            step = 5 if key == 'N' else 10
            lo, hi = axis(min(allv), max(allv), step)
            CHARTS[mark] = chart_xml(sheet, cats, r0, r1, ser, 'UAV Energy (kJ)', xtitle, lo, hi, step)
        else:
            ser = []; allv = []
            for b, name, colr, mk in SAV:
                cc_ = cols[(plat, 's_' + b)]; d = data[(plat, 's_' + b)]
                ser.append(dict(name=name, col=colL(cc_), vals=[a for a, _, _ in d], minus_col=colL(cc_ + 1), minus=[q for _, q, _ in d],
                                plus_col=colL(cc_ + 2), plus=[q for _, _, q in d], color=colr, marker=mk))
                allv += [a + q for a, _, q in d]
            allb = [a + q for pl in PLAT for b, *_ in SAV for a, _, q in data[(pl, 's_' + b)]]
            hi = float(np.ceil(max(allb) / 2) * 2)
            CHARTS[mark] = chart_xml(sheet, cats, r0, r1, ser, 'Energy Saving (%)', xtitle, 0, hi, 2, zero_line=True)
        ph = LineChart(); ws.cell(200 + len(CHARTS), 60, mark)
        ph.add_data(Reference(ws, min_col=60, min_row=200 + len(CHARTS), max_row=200 + len(CHARTS)))
        ph.width, ph.height = 25, 23
        ws.add_chart(ph, f'{cc}{rr}')
wb.save(OUT)

zin = zipfile.ZipFile(OUT); files = {nm: zin.read(nm) for nm in zin.namelist()}; zin.close()
done = 0
for nm in list(files):
    if re.match(r'xl/charts/chart\d+\.xml$', nm):
        s = files[nm].decode('utf-8'); ref = re.search(r"<f>'?([^'<]+)'?!\$BH\$(\d+)", s)
        sheetname, row = ref.group(1), int(ref.group(2))
        files[nm + '.ref'] = f'{sheetname}|{row}'.encode()
from openpyxl import load_workbook
wbr = load_workbook(OUT)
for nm in [k for k in files if k.endswith('.ref')]:
    sheetname, row = files[nm].decode().split('|'); mark = wbr[sheetname].cell(int(row), 60).value
    files[nm[:-4]] = CHARTS[mark].encode('utf-8'); del files[nm]; done += 1
assert done == 8, done
for nm in list(files):
    if re.match(r'xl/worksheets/sheet\d+\.xml$', nm):
        s = files[nm].decode('utf-8'); s = re.sub(r'<c r="BH\d+"[^>]*?(?:/>|>.*?</c>)', '', s)
        s = re.sub(r'<row r="\d+"[^>]*?(?:/>|></row>)', '', s); files[nm] = s.encode('utf-8')
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for nm, b in files.items():
        z.writestr(nm, b)
print('written', OUT)
