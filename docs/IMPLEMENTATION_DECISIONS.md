# Implementation decisions for the new runs

Status of every component, and every decision that changes behaviour relative to the original code or the
manuscript. Approved by the authors on 17 Nov 2026: D3 (coverage baseline; reduced to the manuscript definition on 17 Nov), E2 (four methods only), D4 (four training layouts),
D6 (implemented ILS; Algorithm 1 to be rewritten), D8 (full plan). The other items follow the recommendations of
`SCIENTIFIC_REVIEW_AND_PLAN.md`, Sec. 6.

## Component status

| Component | File | Status |
|---|---|---|
| City geometry, buildings, BTS, hotspots, GUEs, lattice, RIS candidates | `physics.py`, `build_prelim.py` | Verified original (bit-identical rebuild of city21/gcity21) |
| Exact PRR tables, CQI, ARQ, airtime | `build_exact.py` | Verified original |
| Coupled multi-RIS service model | `coupling.py` | Original + fix C1 + reconstructed `details()` + optimised (identical) computation |
| Energy, time, metrics, missions | `env.py`, `env_mr.py` | Verified original |
| Decoding, repair, coupled ILS, TSP | `plan.py`, `plan_mr.py` | Original |
| Policy (coupled masks) | `model.py`, `model_mr.py` | Original |
| Policy base class | `fastmodel.py` | **Reconstructed** (identical maths to `model.Proposed`; test T16) |
| Policy with manuscript cell feature | `model_rx.py` | **New** (adds K_i) |
| Training | `train_rx.py` | **Reconstructed** from `legacy/train_prelim50.py` and the original report |
| Evaluation (four methods) | `eval_rx.py` | **New**, built on `eval_mr.py` (planner and metrics unchanged) |
| Unit tests | `tests/unit_tests.py` | **New** (replace the lost `tests_mr.py`) |

## Decisions

| ID | Decision | Effect on results |
|---|---|---|
| C1 | `coupling.py`: stored tables used with factor 1, recomputed entries scaled by D_g = 40 Mbit (the released file referenced an undefined `P_DG`). Test T08: recomputation reproduces stored tables to 5e-8. | Necessary for any coupled evaluation. |
| C1b | `Coupler.details()` reconstructed (called by `eval_mr.py`, absent from the released `coupling.py`); same screening and CQI search as `table()`. Test T17. | Communication metrics (PRR, CQI, SNR) only. |
| O1 | Coupled CQI search vectorised (batched binary search, all affected points of a table at once). Same common random numbers and same per-sample survival function. Tests T17, T18. | None (identical numbers); 12x faster. |
| O2 | `Coupler.point()` / `City.svc_point()`: one lattice point of a coupled table, same arithmetic; used by the coupled ILS and repair for the other cells' current points. Test T19: identical values and identical ILS trajectories. | None (identical numbers). |
| O3 | `Coupler.feas()`: coupled feasibility mask only (PRR at the lowest CQI), for the coverage baseline. Test T20: identical to `table()` feasibility. | None (identical numbers). |
| M1 | Training memory: activation checkpointing of the point head re-enabled for training (the original `model_mr.py` disables it at import, for inference); arrays unused by serving-only training are dropped (`LEAN`). Peak 2.3 GB instead of 5.9 GB (the machine limit is 6.3 GB). | None (identical maths). |
| R1 | `fastmodel.FastProposed`: point head = `pt_ptr(zp, pt_emb(pf), pm)`, optional activation checkpointing. | None (identical maths). |
| R2 | Training script reconstructed: REINFORCE + greedy rollout baseline, paired one-sided t-test (α = 0.05), Adam 3e-4, StepLR(100, 0.99), clip 1.0, 50 epochs × 10 × 64, best-validation checkpoint, resumable. | As original, except R3, D4, D5, F1. |
| R3 | **Training on serving-only exact tables; all tests use the coupled model.** Evidence: the original report gives 2.5-3 h CPU per run, which only uncoupled training achieves (smoke test: about 0.1 s per rollout uncoupled vs 0.5-1.7 s coupled); the original sensitivity test found that switching unintended reflections off changes plan energy by at most 0.009%. | Training cost is a near-exact proxy; validation curves are serving-only. |
| D4 | Training layouts: cities 11-14 (four), validation missions k = 20000-20049 on each. | Text "four layouts" now correct. |
| D5 | Training penalty: 50 kJ per cell that cannot be served under the full deployment + battery excess (J) + P_h × deadline excess (s). | Manuscript penalty now implemented; rarely active after early training. |
| F1 | Cell feature f_i = [g_i, K_i, Σ D_k] as in Eq. (cell_feature). | Encoder input 4 instead of 3. |
| D3 | Coverage baseline: manuscript definition, coupled Σ\|N_i(r)\|/\|Q_i\| by coordinate ascent (≤ 2 passes), started from the per-cell largest-area site. The per-cell rule alone is no longer reported (authors' decision, 17 Nov 2026). | Coverage baseline follows the manuscript definition. |
| D6 | ILS: original random-cell ILS (relocate + merge, best move for the drawn cell, N_ILS = 100, TSP after). Algorithm 1 to be rewritten to match. | None on results. |
| D7 | TSP: LKH (elkai, 0.1 m resolution, 4 runs). Text to cite LKH. | None. |
| E1 | Random streams: one generator per (seed, mission, method) instead of one shared stream. | Results independent of method order; evaluation resumable. |
| E2 | **Methods: Proposed (route-aware), Coverage-based RIS, Random RIS, No RIS only (authors' decision, 17 Nov 2026).** The other variants were removed from the code: per-cell coverage, policy + TSP, policy only, the ablations (random/heuristic initialisation, frozen sites, sequential, no grouping, serving-only planning) and the long-ILS reference. The four methods keep their random-stream index, so the reduced `eval_rx.py` reproduces the saved plans and energies exactly (checked on city21 k = 6, 7, all four methods). Raw records of the removed variants (from runs made before this decision) are not part of this repository. | None on the four methods. |

## Project packaging (17 Nov 2026)

| ID | Change | Effect on numbers |
|---|---|---|
| P1 | All locations are project-relative (`risuav/sim/rx_paths.py`; overridable by environment variables). The command-line interface (`python -m risuav`) replaces the shell scripts and the job list of the original machine. | None |
| P2 | Windows compatibility: the city name is taken with `os.path.basename` (`env.py`, `env_mr.py`) instead of splitting the path on '/'. The city name seeds the mission generator, so with Windows paths the old code would have generated different missions. Results on Linux unchanged (checked: `verify`, identical plans and energies). | None on Linux; required on Windows |
| P3 | Training device option `TRAIN_DEVICE` (`--device auto|cpu|cuda`): on a GPU the policy network runs on the GPU (`torch.set_default_device`), decoding, repair and evaluation stay on the CPU. CPU path unchanged (checked: identical initial validation energy, 233.42 kJ for facade seed 0). Not tested on a GPU here (no GPU on the build machine). | None on CPU |
| P4 | Trajectory drawing code takes its inputs from arguments instead of fixed paths (`analysis/traj_3d/traj_draw.py`); the approved changes (legend, panel titles, line widths) are written into the code. Output pixel-identical to the previous figures. | None |
| P5 | Published raw results (`artifacts/results/main`) contain only the four methods (entries of removed variants deleted). Exported CSVs and all tables identical to the previous outputs. | None |
| P6 | Figures use the STIX font shipped with matplotlib (Times-like) instead of "Times New Roman, else DejaVu Serif", and `requirements.txt` pins the dependencies of the packages as well. With the pinned wheels, two runs give pixel-identical figures; the font no longer depends on the fonts installed on the computer. | Figure appearance only |
