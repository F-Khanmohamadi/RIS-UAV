import os, sys, time, subprocess
from dataclasses import dataclass, field

from .paths import ROOT, WORK, rel


@dataclass
class Job:
    name: str
    script: str
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    deps: list = field(default_factory=list)
    creates: str = None
    label: str = ''


def base_env(extra=None):
    env = dict(os.environ)
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MALLOC_ARENA_MAX='2', RISUAV_WORK=WORK)
    env.setdefault('MR_ROOT', os.path.join(WORK, 'tables'))
    env.setdefault('PRELIM_ROOT', os.path.join(WORK, 'prelim'))
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def command(job):
    return [sys.executable, '-X', 'utf8', '-W', 'ignore', os.path.join(ROOT, job.script)] + [str(a) for a in job.args]


DRY_RUN = False


def run_jobs(jobs, workers=1, stream_single=False):
    logs = os.path.join(WORK, 'logs'); os.makedirs(logs, exist_ok=True)
    jobs = {j.name: j for j in jobs}
    for j in jobs.values():
        j.deps = [d for d in j.deps if d in jobs]
    if DRY_RUN:
        order, seen = [], set()
        def visit(n):
            if n not in seen:
                seen.add(n)
                for d in jobs[n].deps:
                    visit(d)
                order.append(n)
        for n in jobs:
            visit(n)
        for i, n in enumerate(order, 1):
            j = jobs[n]; st = 'exists, skipped' if j.creates and os.path.exists(j.creates) else 'to run'
            print(f'{i:4d}. {n:<34} [{st}] {j.label}' + (f'\n        after: {", ".join(j.deps)}' if j.deps else ''))
        print(f'{len(order)} job(s); nothing was run (--dry-run)')
        return True
    done, failed, running = set(), set(), {}
    for j in jobs.values():
        if j.creates and os.path.exists(j.creates):
            done.add(j.name); print(f'[skip] {j.name}: {rel(j.creates)} exists', flush=True)
    t_start = time.time()
    try:
        while len(done) + len(failed) < len(jobs):
            for name, (p, t0, fh) in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                del running[name]
                if fh:
                    fh.close()
                if rc == 0:
                    done.add(name); print(f'[done] {name} ({(time.time() - t0) / 60:.1f} min)', flush=True)
                    _notes(os.path.join(logs, name + '.log'))
                else:
                    failed.add(name); print(f'[FAIL] {name} (exit code {rc}); log: {rel(os.path.join(logs, name + ".log"))}', flush=True)
                    _tail(os.path.join(logs, name + '.log'))
            blocked = {n for n, j in jobs.items() if n not in done | failed and any(d in failed for d in j.deps)}
            for n in blocked:
                failed.add(n); print(f'[skip] {n}: a job it needs failed', flush=True)
            ready = [j for n, j in jobs.items() if n not in done | failed and n not in running and all(d in done for d in j.deps)]
            for j in ready:
                if len(running) >= workers:
                    break
                live = stream_single and workers == 1 and len(jobs) - len(done) - len(failed) == 1
                fh = None if live else open(os.path.join(logs, j.name + '.log'), 'a', encoding='utf-8')
                print(f'[start] {j.name}' + (f': {j.label}' if j.label else '') + ('' if live else f'  (log: work/logs/{j.name}.log)'), flush=True)
                p = subprocess.Popen(command(j), cwd=ROOT, env=base_env(j.env), stdout=fh, stderr=subprocess.STDOUT if fh else None)
                running[j.name] = (p, time.time(), fh)
            if not ready and not running and len(done) + len(failed) < len(jobs):
                raise RuntimeError('job graph has a cycle')
            time.sleep(0.5 if running else 0)
    except KeyboardInterrupt:
        print('\ninterrupted: stopping running jobs (finished work is kept; re-run the same command to resume)', flush=True)
        for p, _, fh in running.values():
            p.terminate()
        for p, _, fh in running.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            if fh:
                fh.close()
        raise SystemExit(130)
    print(f'{len(done)} job(s) done, {len(failed)} failed, {(time.time() - t_start) / 60:.1f} min', flush=True)
    return not failed


def _notes(path):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if 'skipped' in line:
                    print('       ' + line.rstrip(), flush=True)
    except OSError:
        pass


def _tail(path, n=25):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()[-n:]
        print('  | ' + '  | '.join(lines), end='' if lines and lines[-1].endswith('\n') else '\n', flush=True)
    except OSError:
        pass
