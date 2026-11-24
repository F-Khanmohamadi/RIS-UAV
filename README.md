# Joint RIS Placement and UAV Mission Planning for Energy-Efficient Logistics-Assisted Cellular Offloading


## Repository structure

```
RIS-UAV/
├── README.md
├── requirements.txt              exact package versions
├── pyproject.toml
├── configs/
│   ├── experiment.json           platforms, seeds, training/test cities, training and evaluation settings
│   ├── sensitivity.json          parameter sensitivity (N and D_k)
│   └── quick.json                quick test
├── risuav/                       Python package (command line: python -m risuav ...)
│   ├── cli.py, runner.py, paths.py, tools.py
│   ├── sim/
│   │   ├── physics.py            city, buildings, LoS, channels, RIS model, propulsion energy; all parameters (REF)
│   │   ├── build_prelim.py       users, RIS candidate sites, hovering lattice of a city
│   │   ├── build_exact.py        PRR / CQI / ARQ / airtime tables per cell, hovering point and RIS option
│   │   ├── coupling.py           reflections of non-serving panels (multi-RIS model)
│   │   ├── env.py, env_mr.py     missions, energy and time model, constraint checks, metrics
│   │   ├── batch.py, model*.py, fastmodel.py   attention policy (encoder, node / site / point heads)
│   │   ├── plan.py, plan_mr.py   decoding, repair, iterative local search, TSP ordering (LKH)
│   │   ├── train_rx.py           REINFORCE with greedy rollout baseline
│   │   ├── eval_rx.py            evaluation of the four methods
│   │   ├── build_variant.py, param_override.py, rx_common.py   parameter variants (sensitivity)
│   │   └── rx_paths.py           data locations
│   └── analysis/                 export, statistics, figures, tables, Excel charts, 3D trajectories,
│                                 numbers quoted in the text (text_numbers.py)
├── tests/                        project checks (no data needed) and 20 numerical unit tests
├── artifacts/
│   ├── checkpoints/              6 trained policies (best.pt) with complete training logs (log.json)
│   └── results/                  raw evaluation records (main study and sensitivity), one JSON line per mission
└── .vscode/                      optional VS Code run configurations
```

Generated data go to `work/` (service tables, new checkpoints, new results, logs) and `outputs/` (figures and
tables). Both are created on first use and are excluded by `.gitignore`.

## Installation

Requirements: Python **3.13** (64-bit), 8 GB RAM, about 1 GB of disk for the quick test and 3 GB for the service
tables of the full study. Windows, Linux and macOS are supported. A GPU is optional (training only).

```bash
python3.13 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m risuav check                       # every package must show "ok"
```

GPU training: install the CUDA build of `torch==2.15.0` from [pytorch.org](https://pytorch.org/get-started/locally/)
before `requirements.txt`. Only the policy network runs on the GPU; GPU runs are reproducible on the same GPU but
not bit-identical to CPU runs. The published checkpoints were trained on the CPU.

VS Code: open the folder, select the `.venv` interpreter, and use the entries of the *Run and Debug* panel, which
call the same commands.


