# Published artifacts

Produced by the code of this repository (Linux, CPU, Python 3.13, packages of `requirements.txt`).

## checkpoints/

`proposed_<platform>_s<seed>/best.pt`: policy weights at the best validation epoch (used for every test result).
`log.json`: complete training log: settings, every update (cost, energy, penalty, entropy, gradient norm) and every
epoch (validation energy, feasibility, baseline replacement, t-test p-value), best epoch, resumes.

| Run | Best epoch | Note |
|---|---|---|
| facade s0 / s1 / s2 | 49 / 16 / 37 | facade s2 jumps to 191 kJ validation energy in its last epoch; `best.pt` is from epoch 37 |
| ground s0 / s1 / s2 | 50 / 30 / 46 | |

## results/main/

`<platform>_s<seed>_best.jsonl`: one JSON line per test mission (cities 21 and 22, k = 0-59), with the mission
(cells, depot, destination) and, for each method (`proposed`, `none`, `random`, `coverage`), the full plan (stops,
members = (cell, RIS option, lattice point), deployed panels) and every metric (energy split, time, route, stops,
throughput, delay, PRR, CQI histogram, GUE energy, constraint checks, run time). `python -m risuav figures` turns
these files into every figure and table; `python -m risuav verify` re-runs missions and compares them with these files.

## results/sens_N16, sens_N24, sens_N40, sens_D10, sens_D20, sens_D80/

Parameter sensitivity of Section V (Table `tab:sensitivity`): `<platform>_s0_best.jsonl`, same format as above, for
missions k = 0-19 of both test cities (40 missions per platform), evaluated with the seed-0 checkpoint trained at the
reference values (no retraining). N16/N24/N40: RIS with 16x16, 24x24, 40x40 elements (service tables rebuilt);
D10/D20/D80: backlog of 10, 20, 80 Mbit per GUE. The reference point (N = 32x32, D = 40 Mbit) is taken from
`results/main/` (seed 0, same 40 missions).
