"""Project-relative locations (no machine-specific paths). Every location can be overridden by an environment variable;
the command-line interface (python -m risuav ...) sets them explicitly.
  RISUAV_WORK  generated data root            default <project>/work
  MR_ROOT      exact service tables root      default <work>/tables   (cities in <MR_ROOT>/<CITY_DIR>/)
  PRELIM_ROOT  preliminary tables root        default <work>/prelim   (cities in <PRELIM_ROOT>/cities/)
  RUNS_DIR     training runs / checkpoints    default <work>/runs
  RESULTS_DIR  raw evaluation results         default <work>/results"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
WORK = os.environ.get('RISUAV_WORK', os.path.join(ROOT, 'work'))
MR_ROOT = os.environ.get('MR_ROOT', os.path.join(WORK, 'tables'))
PRELIM_ROOT = os.environ.get('PRELIM_ROOT', os.path.join(WORK, 'prelim'))
RUNS_DIR = os.environ.get('RUNS_DIR', os.path.join(WORK, 'runs'))
RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(WORK, 'results'))
