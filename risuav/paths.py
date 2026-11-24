import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
WORK = os.path.abspath(os.environ.get('RISUAV_WORK', os.path.join(ROOT, 'work')))
OUTPUTS = os.path.join(ROOT, 'outputs')
CONFIGS = os.path.join(ROOT, 'configs')
ARTIFACTS = os.path.join(ROOT, 'artifacts')

TABLES = os.path.join(WORK, 'tables')
PRELIM = os.path.join(WORK, 'prelim')
RUNS = os.path.join(WORK, 'runs')
RESULTS = os.path.join(WORK, 'results')
SHIPPED_RUNS = os.path.join(ARTIFACTS, 'checkpoints')
SHIPPED_RESULTS = os.path.join(ARTIFACTS, 'results')


def city_names(ids, platforms):
    return [('g' if p == 'ground' else '') + f'city{i}' for p in platforms for i in ids]


def exact_table(city, variant=''):
    return os.path.join(TABLES, 'cities' + (f'_{variant}' if variant else ''), f'{city}.pkl')


def prelim_table(city, variant=''):
    return os.path.join(PRELIM, 'cities' + (f'_{variant}' if variant else ''), f'{city}.pkl')


def rel(path):
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path
