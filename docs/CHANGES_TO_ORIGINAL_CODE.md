# Changes to the original files

The original files and line-by-line diffs against them are kept in the authors' audit archive (not part of this repository).

| File | Change | Effect on numbers |
|---|---|---|
| `risuav/sim/coupling.py` | Fix C1: stored tables used with factor 1, recomputed entries scaled by D_g (replaces the undefined `build_exact.P_DG`) | Needed for any coupled evaluation (test T08) |
| `risuav/sim/coupling.py` | `details()` reconstructed (called by the original evaluation, absent from the released file) | Communication metrics only (test T17) |
| `risuav/sim/coupling.py` | Batched CQI search; `point()` and `feas()` queries | None, identical (tests T18-T20) |
| `risuav/sim/env_mr.py` | `svc_point()`: one lattice point of a coupled table | None, identical (test T19) |
| `risuav/sim/plan_mr.py` | Coupled ILS and repair use `svc_point()` for single points | None, identical (test T19) |
| all modules | Hard-coded paths of the original machine replaced by project-relative locations (`risuav/sim/rx_paths.py`) | None |
| `env.py`, `env_mr.py`, `build_exact.py` | City name taken with `os.path.basename` (Windows paths); see `IMPLEMENTATION_DECISIONS.md`, P2 | None on Linux |

The original evaluation and plotting scripts (`eval_mr.py`, `analysis_mr.py`, `figs_extra_mr.py`,
`fig9_convergence_mr.py`, `tests_mr.py`) are not part of this package. `risuav/sim/eval_rx.py` replaces `eval_mr.py`, using
its planner and metric functions unchanged.
