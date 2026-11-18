"""Table tab:sensitivity of the manuscript (UAV energy of the four methods versus N and D_k, 40 missions, seed-0 policy)
as LaTeX rows, from the sensitivity CSVs written by make_sens.py.
usage: make_sens_table.py <sensitivity dir> <out.tex>"""
import os, sys
import pandas as pd

IN, OUT = sys.argv[1], sys.argv[2]
M = ['none', 'random', 'coverage', 'proposed']
PLAT = ['Facade RIS', 'Ground-mobile RIS']
rows = []
for key, lab in (('N', lambda v: f'$N={int(round(v ** 0.5))}^2$'), ('D', lambda v: f'$D_k={int(v)}$~Mbit')):
    T = pd.read_csv(os.path.join(IN, f'sensitivity_{key}.csv')); T = T[T.policy.str.startswith('fixed')]
    for v in sorted(T.value.unique()):
        cells = []
        for plat in PLAT:
            x = T[(T.platform == plat) & (T.value == v)].set_index('method')
            e = [x.loc[m, 'E_UAV_kJ'] for m in M]
            cells += [f'{a:.1f}' for a in e[:3]] + [f'\\textbf{{{e[3]:.1f}}}']
        rows.append(f'{lab(v)} & ' + ' & '.join(cells) + r' \\')
    rows.append(r'\midrule')
rows = rows[:-1]
head = (r'\begin{tabular}{@{}lcccccccc@{}}' '\n' r'\toprule' '\n'
        r' & \multicolumn{4}{c}{Facade RIS} & \multicolumn{4}{c}{Ground-mobile RIS}\\' '\n'
        r'\cmidrule(lr){2-5}\cmidrule(l){6-9}' '\n'
        r' & No & Rand. & Cov. & Prop. & No & Rand. & Cov. & Prop.\\' '\n' r'\midrule')
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(head + '\n' + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')
print('written', OUT)
