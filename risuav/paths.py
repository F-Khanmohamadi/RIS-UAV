"""Project-relative locations used by the command-line interface (see also risuav/sim/rx_paths.py)."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
WORK = os.path.abspath(os.environ.get('RISUAV_WORK', os.path.join(ROOT, 'work')))   # generated data (git-ignored)
OUTPUTS = os.path.join(ROOT, 'outputs')                                              # figures and tables (git-ignored)
CONFIGS = os.path.join(ROOT, 'configs')
ARTIFACTS = os.path.join(ROOT, 'artifacts')                                          # published checkpoints and raw results

TABLES = os.path.join(WORK, 'tables')        # exact service tables:  tables/cities/<city>.pkl, tables/cities_<variant>/
PRELIM = os.path.join(WORK, 'prelim')        # preliminary tables:    prelim/cities/<city>.pkl, prelim/cities_<variant>/
RUNS = os.path.join(WORK, 'runs')            # checkpoints of runs trained on this computer
RESULTS = os.path.join(WORK, 'results')      # raw results (jsonl) of evaluations run on this computer
SHIPPED_RUNS = os.path.join(ARTIFACTS, 'checkpoints')
SHIPPED_RESULTS = os.path.join(ARTIFACTS, 'results')


def city_names(ids, platforms):
    return [('g' if p == 'ground' else '') + f'city{i}' for p in platforms for i in ids]


def exact_table(city, variant=''):
    return os.path.join(TABLES, 'cities' + (f'_{variant}' if variant else ''), f'{city}.pkl')


def prelim_table(city, variant=''):
    return os.path.join(PRELIM, 'cities' + (f'_{variant}' if variant else ''), f'{city}.pkl')


def rel(path):
    """path relative to the project folder for messages (absolute if it is on another drive)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path
