import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
WORK = os.environ.get('RISUAV_WORK', os.path.join(ROOT, 'work'))
MR_ROOT = os.environ.get('MR_ROOT', os.path.join(WORK, 'tables'))
PRELIM_ROOT = os.environ.get('PRELIM_ROOT', os.path.join(WORK, 'prelim'))
RUNS_DIR = os.environ.get('RUNS_DIR', os.path.join(WORK, 'runs'))
RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(WORK, 'results'))
