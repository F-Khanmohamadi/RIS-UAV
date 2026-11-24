
import numpy as np
from xml.sax.saxutils import escape

def colL(i):                      # 1-based column index -> letters
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
    """series: list of dict(name, col, vals, plus_col, plus, minus_col, minus, color, marker)."""
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




def stacked_bar_xml(sheet, cats, r0, r1, comps, total, ytitle, xtitle, ymin, ymax, major):
    """Stacked column chart in the same style. comps: list of dict(name, col, vals, color) (bottom to top);
    total: dict(col, vals) drawn as an invisible line series whose data labels show the bar totals (its legend entry is
    deleted). Category labels from column A."""
    cat = (f"<c:cat><c:strRef><c:f>'{sheet}'!$A${r0}:$A${r1}</c:f><c:strCache><c:ptCount val=\"{len(cats)}\"/>"
           + ''.join(f'<c:pt idx="{k}"><c:v>{escape(c)}</c:v></c:pt>' for k, c in enumerate(cats)) + '</c:strCache></c:strRef></c:cat>')
    bars = ''
    for i, s in enumerate(comps):
        bars += (f'<c:ser><c:idx val="{i}"/><c:order val="{i}"/><c:tx><c:v>{escape(s["name"])}</c:v></c:tx>'
                 f'<c:spPr><a:solidFill><a:srgbClr val="{s["color"]}"/></a:solidFill><a:ln w="9525"><a:solidFill>'
                 f'<a:srgbClr val="FFFFFF"/></a:solidFill></a:ln></c:spPr><c:invertIfNegative val="0"/>'
                 f'{cat}<c:val>{numref(sheet, s["col"], r0, r1, s["vals"])}</c:val></c:ser>')
    k = len(comps)
    lbl = TXPR.replace('sz="2400"', 'sz="2000"')
    tot = (f'<c:ser><c:idx val="{k}"/><c:order val="{k}"/><c:tx><c:v>Total</c:v></c:tx>'
           '<c:spPr><a:ln><a:noFill/></a:ln></c:spPr><c:marker><c:symbol val="none"/></c:marker>'
           f'<c:dLbls><c:numFmt formatCode="0.0" sourceLinked="0"/><c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>{lbl}'
           '<c:dLblPos val="t"/><c:showLegendKey val="0"/><c:showVal val="1"/><c:showCatName val="0"/><c:showSerName val="0"/>'
           '<c:showPercent val="0"/><c:showBubbleSize val="0"/></c:dLbls>'
           f'{cat}<c:val>{numref(sheet, total["col"], r0, r1, total["vals"])}</c:val><c:smooth val="0"/></c:ser>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<c:date1904 val="0"/><c:lang val="en-US"/><c:roundedCorners val="0"/><c:style val="2"/>'
            '<c:chart><c:autoTitleDeleted val="1"/><c:plotArea><c:layout/>'
            f'<c:barChart><c:barDir val="col"/><c:grouping val="stacked"/><c:varyColors val="0"/>{bars}'
            '<c:gapWidth val="60"/><c:overlap val="100"/><c:axId val="10"/><c:axId val="100"/></c:barChart>'
            f'<c:lineChart><c:grouping val="standard"/><c:varyColors val="0"/>{tot}'
            '<c:marker val="1"/><c:axId val="10"/><c:axId val="100"/></c:lineChart>'
            '<c:catAx><c:axId val="10"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="b"/>'
            + (f'<c:title>{rich(xtitle)}<c:overlay val="0"/></c:title>' if xtitle else '') +
            f'<c:numFmt formatCode="General" sourceLinked="0"/><c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
            f'<c:tickLblPos val="low"/>{AXLN}{TXPR_X}'
            '<c:crossAx val="100"/><c:crosses val="autoZero"/><c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/>'
            '<c:noMultiLvlLbl val="1"/></c:catAx>'
            f'<c:valAx><c:axId val="100"/><c:scaling><c:orientation val="minMax"/><c:max val="{ymax:g}"/><c:min val="{ymin:g}"/></c:scaling>'
            '<c:delete val="0"/><c:axPos val="l"/><c:majorGridlines><c:spPr><a:ln w="9525"><a:solidFill><a:srgbClr val="D9D9D9"/>'
            '</a:solidFill><a:prstDash val="solid"/></a:ln></c:spPr></c:majorGridlines>'
            f'<c:title>{rich(ytitle, rot=-5400000)}<c:overlay val="0"/></c:title><c:numFmt formatCode="0" sourceLinked="0"/>'
            f'<c:majorTickMark val="out"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>{AXLN}{TXPR}'
            f'<c:crossAx val="10"/><c:crosses val="autoZero"/><c:crossBetween val="between"/><c:majorUnit val="{major:g}"/></c:valAx>'
            '<c:spPr><a:ln><a:solidFill><a:srgbClr val="595959"/></a:solidFill></a:ln></c:spPr></c:plotArea>'
            f'<c:legend><c:legendPos val="t"/><c:legendEntry><c:idx val="{k}"/><c:delete val="1"/></c:legendEntry>'
            '<c:overlay val="0"/><c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
            f'<a:ln w="9525"><a:solidFill><a:srgbClr val="595959"/></a:solidFill></a:ln></c:spPr>{TXPR}</c:legend>'
            '<c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/></c:chart>'
            '<c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:ln w="9525"><a:solidFill><a:srgbClr val="595959"/>'
            '</a:solidFill></a:ln></c:spPr></c:chartSpace>')
