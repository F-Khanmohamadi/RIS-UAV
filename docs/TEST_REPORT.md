# Test report (18 Nov 2026)

## Environment

- Linux, 2 CPU cores, no GPU.
- Python 3.13.18 in a new virtual environment with the packages of `requirements.txt` from PyPI. PyTorch 2.15.0 (same
  version) was linked from an existing installation because of limited disk space on the build machine.
- The repository ZIP was extracted into an empty folder and every command was run from there.

## Commands run on the extracted repository

All exited with code 0.

| Command | Result |
|---|---|
| `python -m risuav check` | all pinned packages `ok` |
| `python -m risuav test --fast` | 7/7 project checks |
| `python -m unittest tests.test_project` | OK |
| `python -m risuav tables/train/evaluate/reproduce --dry-run`, `sensitivity --retrain --dry-run` | job graphs listed |
| `python -m risuav tables --cities city21 gcity21 city22 gcity22 --workers 2` | 4 tables built |
| `python -m risuav figures` | main, Excel, trajectory, sensitivity, `tab_sensitivity.tex`, `text_numbers.md` |
| `python -m risuav verify --platforms facade ground` | 48 plans (12 missions x 4 methods) identical to the published results, largest relative energy difference 0 |
| `python -m risuav quick` | 5/5 steps PASS (table, 20 unit tests, smoke training, re-evaluation = published, figures) |
| `python -m risuav sensitivity --params D --platforms facade --missions 0-0` | D = 10, 20, 80 Mbit, mission 0 of both test cities: plans and energies identical to `artifacts/results/sens_D*` |
| `python -m risuav test --platforms facade` | 7/7 project checks, 20/20 unit tests |

## Agreement with the results of the paper

- **Service tables** of city21, gcity21, city22 and gcity22 built in the clean environment are byte-identical to
  the tables of the reported runs.
- **Tables and statistics** (`summary_overall`, `paired_statistics`, `training_summary`, `vs_cells_all_metrics`,
  `energy_breakdown`, `training_per_seed`, `sensitivity_N`, `sensitivity_D`, CSV and LaTeX): byte-identical to the
  outputs from which Section V was written. `tab_sensitivity.tex` equals Table `tab:sensitivity`.
- **Numbers quoted in the text** (`text_numbers.md`): identical to the values used in Section V.
- **Excel figures**: chart contents identical to the workbooks prepared for the manuscript.
- **Trajectory mission values**: identical.
- **Figures**: same data; the font is STIXGeneral (bundled with matplotlib) on every system.

## Not repeated

- Full training (6 runs x 50 epochs) and the full 120-mission evaluation through this interface; they use the same
  scripts as the reported runs. The initial validation energy of a new training run equals that of the reported run,
  and an interrupted run resumes from its last epoch (checked earlier on the development copy).
- The rebuilt-table sensitivity path (`--params N`) on the extracted repository; it was run on the development copy,
  which produced the published `sens_N*` results.
- Windows and CUDA. Wheels exist for every pinned package (`pip download --platform win_amd64 --python-version 3.13`).
- Microsoft Excel: the Excel charts were checked in LibreOffice only.
