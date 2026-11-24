import argparse, json, os, shutil, subprocess, sys, time

from . import __version__
from .paths import (ROOT, WORK, OUTPUTS, CONFIGS, TABLES, PRELIM, RUNS, RESULTS, SHIPPED_RUNS, SHIPPED_RESULTS,
                    city_names, exact_table, prelim_table, rel)
from . import runner
from .runner import Job, run_jobs, base_env, command

SIM = 'risuav/sim'
ANA = 'risuav/analysis'


def load_config(name):
    with open(os.path.join(CONFIGS, name), encoding='utf-8') as f:
        return json.load(f)


def default_workers():
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def table_jobs(cities, variant='', override=None, rebuild_prelim=False):
    jobs = []
    ovr = {'RX_OVERRIDE': json.dumps(override)} if override else {}
    for c in cities:
        plat = 'ground' if c.startswith('g') else 'facade'; cid = int(c.lstrip('g')[4:])
        tag = f'_{variant}' if variant else ''
        pre_name = f'prelim{tag}_{c}'
        if not variant or rebuild_prelim:
            if variant:
                jobs.append(Job(pre_name, f'{SIM}/build_variant.py', ['prelim', plat, cid, prelim_table(c, variant)], ovr,
                                creates=prelim_table(c, variant), label=f'preliminary table {c} ({variant})'))
            else:
                jobs.append(Job(pre_name, f'{SIM}/build_prelim.py', [plat, cid], creates=prelim_table(c),
                                label=f'preliminary table {c} (geometry, users, RIS sites, lattice)'))
            src = prelim_table(c, variant if variant else '')
            deps = [pre_name]
        else:
            src = prelim_table(c); deps = [f'prelim_{c}']
            if not os.path.exists(src):
                jobs.append(Job(f'prelim_{c}', f'{SIM}/build_prelim.py', [plat, cid], creates=src, label=f'preliminary table {c}'))
        if variant:
            jobs.append(Job(f'exact{tag}_{c}', f'{SIM}/build_variant.py', ['exact', src, exact_table(c, variant)], ovr, deps,
                            creates=exact_table(c, variant), label=f'exact table {c} ({variant})'))
        else:
            jobs.append(Job(f'exact_{c}', f'{SIM}/build_exact.py', [src, exact_table(c)], {}, deps,
                            creates=exact_table(c), label=f'exact table {c} (PRR, CQI, ARQ, airtime)'))
    return jobs


def train_job(plat, seed, tcfg, train_ids, device, runs_dir, tag='proposed', env=None, deps=(), city_dir='cities'):
    e = dict(RUNS_DIR=runs_dir, RUN_TAG=tag, TRAIN_IDS=','.join(map(str, train_ids)), TRAIN_DEVICE=device,
             TRAIN_MODEL=tcfg.get('train_model', 'serving'), CITY_DIR=city_dir)
    e.update(env or {})
    run = os.path.join(runs_dir, f'{tag}_{plat}_s{seed}'); final = os.path.join(run, 'final.pt')
    lg = os.path.join(run, 'log.json')
    if os.path.exists(lg):
        with open(lg, encoding='utf-8') as f:
            L = json.load(f)
        have = (L.get('epochs'), L.get('steps_per_epoch'), L.get('batch')); want = (tcfg['epochs'], tcfg['steps_per_epoch'], tcfg['batch'])
        if have != want:
            msg = (f'{rel(run)} holds a run with epochs/steps/batch = {have}, not {want}.\n'
                   f'Delete that folder (or rename it) to train with the new settings.')
            if not runner.DRY_RUN:
                sys.exit(msg)
            print('WARNING: ' + msg)
    return Job(f'train_{tag}_{plat}_s{seed}', f'{SIM}/train_rx.py', [plat, seed, tcfg['epochs'], tcfg['steps_per_epoch'], tcfg['batch']],
               e, list(deps), creates=final, label=f'{plat}, seed {seed}, {tcfg["epochs"]} epochs x {tcfg["steps_per_epoch"]} x {tcfg["batch"]}, device {device}')


def eval_job(plat, seed, ecfg, test_ids, runs_dir, results_dir, out_tag='main', tag='proposed', env=None, deps=(),
             city_dir='cities', missions=None, methods=None, name=None):
    e = dict(RUNS_DIR=runs_dir, RESULTS_DIR=results_dir, RUN_TAG=tag, OUT_TAG=out_tag, CITY_DIR=city_dir,
             TEST_CITIES=','.join(f'city{i}' for i in test_ids), K_RANGE=missions or ecfg['missions'], N_ILS=ecfg['n_ils'],
             METHODS=','.join(methods or ecfg['methods']))
    e.update(env or {})
    return Job(name or f'eval_{out_tag}_{plat}_s{seed}', f'{SIM}/eval_rx.py', [plat, seed, ecfg['checkpoint']], e, list(deps),
               label=f'{plat}, seed {seed}, missions {e["K_RANGE"]} of {e["TEST_CITIES"]}, methods {e["METHODS"]}')


def need_tables(cities):
    missing = [c for c in cities if not os.path.exists(exact_table(c))]
    return table_jobs(missing), {c: f'exact_{c}' for c in missing}


def checkpoints_dir(choice):
    return {'artifacts': SHIPPED_RUNS, 'work': RUNS}.get(choice, os.path.abspath(choice))


def results_dir(choice):
    return {'artifacts': SHIPPED_RESULTS, 'work': RESULTS}.get(choice, os.path.abspath(choice))


def check_checkpoints(runs_dir, platforms, seeds, tag='proposed', ckpt='best'):
    miss = [f'{tag}_{p}_s{s}/{ckpt}.pt' for p in platforms for s in seeds if not os.path.exists(os.path.join(runs_dir, f'{tag}_{p}_s{s}', f'{ckpt}.pt'))]
    if miss:
        sys.exit(f'missing checkpoints in {rel(runs_dir)}: {", ".join(miss)}\n'
                 f'(train them with `python -m risuav train`, or use the published ones with `--checkpoints artifacts`)')


def cmd_check(a):
    import importlib.metadata
    print(f'RIS-UAV {__version__}\nproject folder : {ROOT}\nwork folder    : {WORK}\npython         : {sys.version.split()[0]} ({sys.executable})')
    print(f'UTF-8 mode     : {bool(sys.flags.utf8_mode)}')
    req = {}
    with open(os.path.join(ROOT, 'requirements.txt'), encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if '==' in line:
                k, v = line.split('=='); req[k.strip()] = v.strip()
    ok = True
    for pkg, want in req.items():
        try:
            have = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            have = 'NOT INSTALLED'
        flag = 'ok' if str(have).split('+')[0] == want else ('MISSING' if have == 'NOT INSTALLED' else 'differs')
        ok = ok and flag == 'ok'
        print(f'  {pkg:<17} required {want:<9} installed {have:<14} {flag}')
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f'GPU (CUDA)     : {"available: " + torch.cuda.get_device_name(0) if cuda else "not available (training runs on the CPU)"}')
    except Exception:
        pass
    du = shutil.disk_usage(ROOT)
    print(f'free disk      : {du.free / 1e9:.1f} GB (full study: about 3 GB for the tables)')
    base = [c for c in city_names([11, 12, 13, 14, 21, 22], ['facade', 'ground'])]
    have = [c for c in base if os.path.exists(exact_table(c))]
    print(f'service tables : {len(have)}/{len(base)} built' + (f' ({", ".join(have)})' if have else ' (build: python -m risuav tables)'))
    ck = [d for d in sorted(os.listdir(SHIPPED_RUNS))] if os.path.isdir(SHIPPED_RUNS) else []
    print(f'published checkpoints : {len(ck)} ({", ".join(ck)})')
    print(f'published results     : {", ".join(sorted(os.listdir(SHIPPED_RESULTS))) if os.path.isdir(SHIPPED_RESULTS) else "none"}')
    return 0 if ok else 1


def cmd_tables(a):
    cfg = load_config('experiment.json')
    if a.cities:
        cities = a.cities
    else:
        ids = {'test': cfg['test_cities'], 'train': cfg['train_cities'], 'all': cfg['train_cities'] + cfg['test_cities']}[a.set]
        cities = city_names(ids, a.platforms or cfg['platforms'])
    bad = [c for c in cities if not (c.lstrip('g').startswith('city') and c.lstrip('g')[4:].isdigit())]
    if bad:
        sys.exit(f'unknown city name(s): {bad} (use e.g. city21 for Facade RIS and gcity21 for Ground-mobile RIS)')
    print(f'building service tables of {len(cities)} cities: {" ".join(cities)} (7-12 min and about 240 MB per city, one core each)')
    return 0 if run_jobs(table_jobs(cities), a.workers) else 1


def cmd_test(a):
    print('--- project checks (no tables needed) ---', flush=True)
    rc = subprocess.call([sys.executable, '-X', 'utf8', '-m', 'unittest', 'tests.test_project', '-v'], cwd=ROOT, env=base_env())
    if a.fast:
        return rc
    for plat in a.platforms:
        c = ('g' if plat == 'ground' else '') + 'city21'
        if not os.path.exists(exact_table(c)):
            print(f'table {c} missing: building it first')
            if not run_jobs(table_jobs([c]), 1):
                return 1
        print(f'--- unit tests ({plat}) ---', flush=True)
        rc |= subprocess.call([sys.executable, '-X', 'utf8', '-W', 'ignore', os.path.join(ROOT, 'tests', 'unit_tests.py'), plat],
                              cwd=ROOT, env=base_env())
    return rc


def cmd_train(a):
    cfg = load_config(a.config)
    t = dict(cfg['train'])
    for k, v in (('epochs', a.epochs), ('steps_per_epoch', a.steps), ('batch', a.batch)):
        if v:
            t[k] = v
    plats, seeds = a.platforms or cfg['platforms'], a.seeds if a.seeds is not None else cfg['seeds']
    tj, dep = need_tables(city_names(cfg['train_cities'], plats))
    jobs = tj + [train_job(p, s, t, cfg['train_cities'], a.device, RUNS, deps=[dep[c] for c in city_names(cfg['train_cities'], [p]) if c in dep])
                 for p in plats for s in seeds]
    print(f'training {len(plats) * len(seeds)} run(s) -> {rel(RUNS)} (resumable: re-run the same command to continue)')
    return 0 if run_jobs(jobs, a.workers, stream_single=True) else 1


def cmd_evaluate(a):
    cfg = load_config(a.config); e = dict(cfg['evaluate'])
    plats, seeds = a.platforms or cfg['platforms'], a.seeds if a.seeds is not None else cfg['seeds']
    rd = checkpoints_dir(a.checkpoints); check_checkpoints(rd, plats, seeds, ckpt=e['checkpoint'])
    tj, dep = need_tables(city_names(cfg['test_cities'], plats))
    tag = 'main' if not a.methods or set(a.methods) == set(e['methods']) else 'main_' + '_'.join(a.methods)
    jobs = tj + [eval_job(p, s, e, cfg['test_cities'], rd, RESULTS, out_tag=tag, missions=a.missions, methods=a.methods,
                          deps=[dep[c] for c in city_names(cfg['test_cities'], [p]) if c in dep]) for p in plats for s in seeds]
    print(f'evaluating checkpoints of {rel(rd)} -> {rel(os.path.join(RESULTS, tag))} (resumable: finished missions are kept)')
    return 0 if run_jobs(jobs, a.workers, stream_single=True) else 1


def compare_results(new_dir, ref_dir, tag='main'):
    import glob
    n = worst = 0; plan_diff = []
    for f in sorted(glob.glob(os.path.join(new_dir, tag, '*.jsonl'))):
        ref_f = os.path.join(ref_dir, tag, os.path.basename(f))
        if not os.path.exists(ref_f):
            print(f'  no published file for {os.path.basename(f)}'); continue
        with open(ref_f, encoding='utf-8') as fh:
            ref = {(r['city'], r['k']): r for r in map(json.loads, fh)}
        with open(f, encoding='utf-8') as fh:
            rows = [json.loads(l) for l in fh]
        for r in rows:
            o = ref.get((r['city'], r['k']))
            if o is None:
                continue
            for m in ('proposed', 'none', 'random', 'coverage'):
                if not r.get(m) or not o.get(m):
                    continue
                n += 1; d = abs(r[m]['E_kJ'] - o[m]['E_kJ']) / max(abs(o[m]['E_kJ']), 1e-12); worst = max(worst, d)
                same = r[m]['plan'] == o[m]['plan']
                if not same:
                    plan_diff.append((r['city'], r['k'], m))
                print(f'  {os.path.basename(f)[:-6]:<16} {r["city"]:<8} k={r["k"]:<3} {m:<9} E_UAV {r[m]["E_kJ"]:9.4f} kJ  published {o[m]["E_kJ"]:9.4f} kJ  '
                      f'{"same plan" if same else "DIFFERENT PLAN"}')
    if n == 0:
        print('  nothing to compare'); return False
    print(f'compared {n} plans: largest relative energy difference {worst:.2e}; {len(plan_diff)} plan(s) differ')
    if worst <= 1e-9 and not plan_diff:
        print('VERIFIED: identical to the published results'); return True
    if worst <= 1e-6:
        print('VERIFIED within floating-point tolerance (1e-6); small differences can come from the platform\'s numerical libraries')
        return True
    print('NOT VERIFIED: results differ from the published ones'); return False


def cmd_verify(a):
    cfg = load_config('experiment.json'); e = dict(cfg['evaluate'])
    out = os.path.join(WORK, 'verify', 'results')
    if os.path.isdir(out) and not a.keep:
        shutil.rmtree(out)
    plats = a.platforms
    check_checkpoints(SHIPPED_RUNS, plats, [a.seed])
    cities = [f'city{a.city}'] if a.city else [f'city{i}' for i in cfg['test_cities']]
    ids = [int(c[4:]) for c in cities]
    tj, dep = need_tables(city_names(ids, plats))
    jobs = tj + [eval_job(p, a.seed, e, ids, SHIPPED_RUNS, out, missions=a.missions, name=f'verify_{p}_s{a.seed}',
                          deps=[dep[c] for c in city_names(ids, [p]) if c in dep]) for p in plats]
    if not run_jobs(jobs, a.workers):
        return 1
    print('--- comparison with artifacts/results (published) ---')
    return 0 if compare_results(out, SHIPPED_RESULTS) else 1


def sensitivity_jobs(scfg, ecfg, test_ids, plats, rd, res, params, retrain, keep, device, tcfg, train_ids):
    jobs = []
    tj, dep = need_tables(city_names(test_ids, plats)); jobs += tj
    base = lambda p: [dep[c] for c in city_names(test_ids, [p]) if c in dep]
    exact_deps = lambda tag, p: [f'exact_{tag}_{c}' for c in city_names(test_ids, [p])]
    rt_tags = [t for t in scfg['retrain'] if not t.startswith('_')] if retrain else []
    prev, users = [], {}
    for key in params:
        for tag, pt in scfg['parameters'][key]['points'].items():
            env = dict(pt.get('env', {}))
            if 'override' in pt:
                env['RX_OVERRIDE'] = json.dumps(pt['override'])
                for j in table_jobs(city_names(test_ids, plats), tag, pt['override'], pt.get('rebuild_prelim', False)):
                    j.deps = j.deps + prev; jobs.append(j)
                ev = [eval_job(p, scfg['seed'], ecfg, test_ids, rd, res, out_tag=f'sens_{tag}', env=env, city_dir=f'cities_{tag}',
                               missions=scfg['missions'], methods=scfg['methods'], name=f'sens_{tag}_{p}', deps=exact_deps(tag, p)) for p in plats]
                jobs += ev; prev = [j.name for j in ev]; users[tag] = list(prev)
            else:
                jobs += [eval_job(p, scfg['seed'], ecfg, test_ids, rd, res, out_tag=f'sens_{tag}', env=env, missions=scfg['missions'],
                                  methods=scfg['methods'], name=f'sens_{tag}_{p}', deps=base(p)) for p in plats]
    if rt_tags:
        tj2, dep2 = need_tables(city_names(train_ids, plats)); jobs += tj2
        for tag in rt_tags:
            rt = scfg['retrain'][tag]; env = dict(rt.get('env', {})); tdir = 'cities'
            tdeps = {p: [dep2[c] for c in city_names(train_ids, [p]) if c in dep2] for p in plats}
            if 'override' in rt:
                env['RX_OVERRIDE'] = json.dumps(rt['override']); tdir = f'cities_{tag}_train'
                for c in city_names(train_ids, plats):
                    dst = os.path.join(TABLES, tdir, f'{c}.pkl'); p = 'ground' if c.startswith('g') else 'facade'
                    jobs.append(Job(f'exact_{tag}_train_{c}', f'{SIM}/build_variant.py', ['exact', prelim_table(c), dst],
                                    {'RX_OVERRIDE': env['RX_OVERRIDE']}, [f'prelim_{c}'], creates=dst, label=f'exact table {c} ({tag}, training)'))
                    tdeps[p].append(f'exact_{tag}_train_{c}')
                if tag not in users:
                    jobs += table_jobs(city_names(test_ids, plats), tag, rt['override'])
                    users[tag] = []
            for p in plats:
                tr = train_job(p, scfg['seed'], tcfg, train_ids, device, RUNS, tag=f'retrain{tag}', env=env, deps=tdeps[p], city_dir=tdir)
                ev = eval_job(p, scfg['seed'], ecfg, test_ids, RUNS, res, out_tag=f'sensrt_{tag}', tag=f'retrain{tag}', env=env,
                              city_dir=f'cities_{tag}' if 'override' in rt else 'cities', missions=scfg['missions'], methods=scfg['methods'],
                              name=f'sensrt_{tag}_{p}', deps=[tr.name] + (exact_deps(tag, p) if 'override' in rt else base(p)))
                jobs += [tr, ev]
                if 'override' in rt:
                    users[tag].append(ev.name)
    if not keep:
        for tag, u in users.items():
            dirs = [os.path.join(TABLES, f'cities_{tag}'), os.path.join(PRELIM, f'cities_{tag}'), os.path.join(TABLES, f'cities_{tag}_train')]
            jobs.append(Job(f'clean_{tag}', 'risuav/tools.py', ['rmtree'] + dirs, deps=u, label=f'delete the {tag} variant tables'))
    return jobs


def cmd_sensitivity(a):
    cfg = load_config('experiment.json'); scfg = load_config('sensitivity.json')
    plats = a.platforms or cfg['platforms']; params = a.params or list(scfg['parameters'])
    unknown = [p for p in params if p not in scfg['parameters']]
    if unknown:
        sys.exit(f'unknown parameter(s) {unknown}; choose from {list(scfg["parameters"])}')
    rd = checkpoints_dir(a.checkpoints); check_checkpoints(rd, plats, [scfg['seed']])
    if a.missions:
        scfg['missions'] = a.missions
    jobs = sensitivity_jobs(scfg, cfg['evaluate'], cfg['test_cities'], plats, rd, RESULTS, params, a.retrain, a.keep_tables,
                            a.device, cfg['train'], cfg['train_cities'])
    print(f'sensitivity: parameters {" ".join(params)}{" + retraining at N16 and D80" if a.retrain else ""}; '
          f'fixed policy from {rel(rd)} (seed {scfg["seed"]}); results -> {rel(RESULTS)}/sens_*')
    return 0 if run_jobs(jobs, a.workers) else 1


def figure_jobs(res, runs, out, traj=True, sens=True):
    exp = os.path.join(out, 'data')
    jobs = [Job('export', f'{ANA}/export_rx.py', [res, exp], label='raw results -> per-mission CSV'),
            Job('figures_main', f'{ANA}/make_main.py', [os.path.join(exp, 'per_mission_main.csv'), runs, os.path.join(out, 'main'), 'proposed'],
                deps=['export'], label='main figures and tables')]
    if traj:
        cfg = load_config('experiment.json')['trajectory']; c = cfg['city']
        need = [exact_table(c), exact_table('g' + c), prelim_table(c), prelim_table('g' + c)]
        if all(os.path.exists(p) for p in need):
            jobs.append(Job('figures_trajectory', f'{ANA}/traj_figs.py', [os.path.join(exp, 'per_mission_main.csv'), os.path.join(exp, 'plans_main.jsonl'),
                                                                     cfg['seed'], c, cfg['mission'], os.path.join(out, 'trajectory')],
                            deps=['export'], label=f'3D trajectory figures (mission {c} k = {cfg["mission"]}, seed {cfg["seed"]})'))
        else:
            print(f'trajectory figures skipped: they need the tables of {c} and g{c} (python -m risuav tables --cities {c} g{c})')
    xl, mn = os.path.join(out, 'excel'), os.path.join(out, 'main')
    jobs += [Job('excel_fig3', f'{ANA}/make_fig3_excel.py', [os.path.join(mn, 'training_per_seed.csv'),
                 os.path.join(ROOT, 'risuav', 'analysis', 'templates', 'fig3_training_template.xlsx'),
                 os.path.join(xl, 'Fig3_Training_Validation_Energy.xlsx')], deps=['figures_main'], label='Excel: Fig. 3 (training)'),
             Job('excel_breakdown', f'{ANA}/make_breakdown_excel.py', [os.path.join(mn, 'energy_breakdown.csv'),
                 os.path.join(xl, 'UAV_Energy_Breakdown.xlsx')], deps=['figures_main'], label='Excel: energy breakdown')]
    for metric, ytitle, step, fn in (('E_UAV_kJ', 'UAV Energy (kJ)', 10, 'UAV_Energy_vs_Cells'),
                                     ('R_ser_Mbps', 'Service Throughput (Mbit/s)', 10, 'Service_Throughput_vs_Cells'),
                                     ('delay_per_cell_s', 'Delay per Cell (s)', 1, 'Communication_Delay_vs_Cells')):
        jobs.append(Job(f'excel_{fn}', f'{ANA}/make_vs_cells_excel.py', [os.path.join(mn, 'vs_cells_all_metrics.csv'), metric, ytitle,
                        str(step), os.path.join(xl, f'{fn}.xlsx')], deps=['figures_main'], label=f'Excel: {fn}'))
    if sens and os.path.isdir(res) and any(d.startswith('sens_') for d in os.listdir(res)):
        jobs.append(Job('figures_sensitivity', f'{ANA}/make_sens.py', [exp, os.path.join(out, 'sensitivity')], deps=['export'],
                        label='sensitivity figures and tables'))
        jobs.append(Job('excel_sensitivity', f'{ANA}/make_sens_excel.py', [os.path.join(out, 'sensitivity'),
                        os.path.join(xl, 'Sensitivity_N_D.xlsx')], deps=['figures_sensitivity'], label='Excel: sensitivity (N, D)'))
        jobs.append(Job('table_sensitivity', f'{ANA}/make_sens_table.py', [os.path.join(out, 'sensitivity'),
                        os.path.join(out, 'sensitivity', 'tab_sensitivity.tex')], deps=['figures_sensitivity'],
                        label='manuscript table tab:sensitivity'))
    elif sens:
        print('sensitivity figures skipped: no sens_* results in ' + rel(res))
    jobs.append(Job('text_numbers', f'{ANA}/text_numbers.py', [out, res, os.path.join(out, 'text_numbers.md')],
                    deps=[j.name for j in jobs if j.name in ('figures_main', 'figures_sensitivity')], label='numbers quoted in Section V'))
    return jobs


def cmd_figures(a):
    res = results_dir(a.results)
    runs = checkpoints_dir(a.runs or ('artifacts' if a.results == 'artifacts' else 'work'))
    if not os.path.isdir(os.path.join(res, 'main')):
        sys.exit(f'no main results in {rel(res)} (run `python -m risuav evaluate`, or use --results artifacts)')
    out = os.path.abspath(a.out)
    print(f'figures and tables from {rel(res)} (training logs: {rel(runs)}) -> {rel(out)}')
    ok = run_jobs(figure_jobs(res, runs, out, not a.no_trajectory), a.workers)
    if ok:
        print(f'outputs: {rel(out)}' + os.sep + ' (main/, excel/, trajectory/, sensitivity/, data/)')
    return 0 if ok else 1


def cmd_quick(a):
    q = load_config('quick.json'); t0 = time.time()
    steps = []
    print('quick test: 1) table of city21  2) unit tests  3) 1-epoch smoke training  4) re-evaluation of 2 missions with the '
          'published checkpoint, compared with the published results  5) main figures from the published results\n', flush=True)
    if not run_jobs(table_jobs(['city21']), 1):
        return 1
    steps.append(('table city21', True))
    rc = subprocess.call([sys.executable, '-X', 'utf8', '-W', 'ignore', os.path.join(ROOT, 'tests', 'unit_tests.py'), 'facade'], cwd=ROOT, env=base_env())
    steps.append(('unit tests (20)', rc == 0))
    qruns = os.path.join(WORK, 'quick', 'runs')
    if os.path.isdir(qruns):
        shutil.rmtree(qruns)
    ok = run_jobs([train_job('facade', 0, q['train'], q['train_cities'], a.device, qruns, tag='smoke')], 1, stream_single=True)
    steps.append(('smoke training (1 epoch, 2 x 4 missions)', ok and os.path.exists(os.path.join(qruns, 'smoke_facade_s0', 'final.pt'))))
    out = os.path.join(WORK, 'quick', 'results')
    if os.path.isdir(out):
        shutil.rmtree(out)
    ok = run_jobs([eval_job('facade', 0, q['evaluate'], q['test_cities'], SHIPPED_RUNS, out, name='quick_eval_facade_s0')], 1, stream_single=True)
    print('--- comparison with artifacts/results (published) ---')
    steps.append(('re-evaluation = published results', ok and compare_results(out, SHIPPED_RESULTS)))
    qout = os.path.join(WORK, 'quick', 'outputs')
    ok = run_jobs(figure_jobs(SHIPPED_RESULTS, SHIPPED_RUNS, qout, traj=False, sens=False), 1)
    steps.append(('main figures and tables', ok and os.path.exists(os.path.join(qout, 'main', 'UAV_Energy_Facade.png'))))
    print('\n=== quick test summary ===')
    for name, good in steps:
        print(f'  {"PASS" if good else "FAIL"}  {name}')
    print(f'time {(time.time() - t0) / 60:.1f} min; figures: {rel(qout)}')
    return 0 if all(g for _, g in steps) else 1


def cmd_reproduce(a):
    cfg = load_config('experiment.json'); scfg = load_config('sensitivity.json')
    plats, seeds = cfg['platforms'], cfg['seeds']
    jobs = table_jobs([c for c in city_names(cfg['train_cities'] + cfg['test_cities'], plats) if not os.path.exists(exact_table(c))])
    names = {j.name for j in jobs}
    tdep = lambda ids, p: [f'exact_{c}' for c in city_names(ids, [p]) if f'exact_{c}' in names]
    rd = SHIPPED_RUNS if a.skip_training else RUNS
    if not a.skip_training:
        jobs += [train_job(p, s, cfg['train'], cfg['train_cities'], a.device, RUNS, deps=tdep(cfg['train_cities'], p)) for p in plats for s in seeds]
    for p in plats:
        for s in seeds:
            d = tdep(cfg['test_cities'], p) + ([] if a.skip_training else [f'train_proposed_{p}_s{s}'])
            jobs.append(eval_job(p, s, cfg['evaluate'], cfg['test_cities'], rd, RESULTS, deps=d))
    if not a.skip_sensitivity:
        sj = sensitivity_jobs(scfg, cfg['evaluate'], cfg['test_cities'], plats, rd, RESULTS, list(scfg['parameters']), not a.skip_training,
                              False, a.device, cfg['train'], cfg['train_cities'])
        have = {j.name for j in jobs}
        for j in sj:
            if j.name in have:
                continue
            if j.name.startswith(('sens_', 'sensrt_')):
                j.deps = j.deps + [f'eval_main_{p}_s{scfg["seed"]}' for p in plats]
            jobs.append(j)
    out = os.path.join(OUTPUTS, 'reproduced')
    print(f'full reproduction: {len(jobs)} jobs before the figures; workers {a.workers}; device {a.device}. '
          'Expect several days on one CPU core; every step resumes if interrupted.')
    if not run_jobs(jobs, a.workers):
        return 1
    return 0 if run_jobs(figure_jobs(RESULTS, rd, out), a.workers) else 1


def build_parser():
    P = argparse.ArgumentParser(prog='python -m risuav', description='RIS-assisted logistics UAV: tables, training, evaluation, '
                                'sensitivity analysis and figures. Run `python -m risuav <command> -h` for the options of a command.')
    P.add_argument('--version', action='version', version=f'RIS-UAV {__version__}')
    S = P.add_subparsers(dest='cmd', required=True, metavar='command')
    W = dict(type=int, default=default_workers(), help=f'parallel processes (default {default_workers()}; each needs 1 core and up to 2.5 GB RAM)')
    PL = dict(nargs='+', choices=['facade', 'ground'], help='platforms (default: both)')

    s = S.add_parser('check', help='check the Python packages, GPU, disk space and what has been built'); s.set_defaults(f=cmd_check)
    s = S.add_parser('quick', help='quick test of the whole pipeline (about 15 min, half of it building one table)')
    s.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto']); s.set_defaults(f=cmd_quick)
    s = S.add_parser('tables', help='build the service tables (needed by every other step)')
    s.add_argument('--set', default='all', choices=['all', 'train', 'test'], help='training cities 11-14, test cities 21-22, or all (default)')
    s.add_argument('--cities', nargs='+', help='explicit city names, e.g. city21 gcity21 (g = Ground-mobile RIS)')
    s.add_argument('--platforms', **PL); s.add_argument('--workers', **W); s.add_argument('--dry-run', action='store_true', help='list the jobs without running them'); s.set_defaults(f=cmd_tables)
    s = S.add_parser('test', help='project checks + unit tests (20 numerical checks of the model)')
    s.add_argument('--platforms', nargs='+', default=['facade'], choices=['facade', 'ground'])
    s.add_argument('--fast', action='store_true', help='project checks only (no tables needed, seconds)'); s.set_defaults(f=cmd_test)
    s = S.add_parser('train', help='train the policy (50 epochs x 10 x 64 per run by default) -> work/runs')
    s.add_argument('--platforms', **PL); s.add_argument('--seeds', nargs='+', type=int, help='seeds (default: 0 1 2)')
    s.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='auto = GPU if available (default)')
    s.add_argument('--epochs', type=int); s.add_argument('--steps', type=int, help='updates per epoch'); s.add_argument('--batch', type=int)
    s.add_argument('--config', default='experiment.json'); s.add_argument('--workers', **W); s.add_argument('--dry-run', action='store_true', help='list the jobs without running them'); s.set_defaults(f=cmd_train)
    s = S.add_parser('evaluate', help='evaluate the four methods on the 120 test missions -> work/results/main')
    s.add_argument('--checkpoints', default='artifacts', help="'artifacts' (published, default), 'work' (trained here) or a folder")
    s.add_argument('--platforms', **PL); s.add_argument('--seeds', nargs='+', type=int)
    s.add_argument('--missions', help='mission range per test city, e.g. 0-59 (default) or 0-4')
    s.add_argument('--methods', nargs='+', choices=['proposed', 'none', 'random', 'coverage'])
    s.add_argument('--config', default='experiment.json'); s.add_argument('--workers', **W); s.add_argument('--dry-run', action='store_true', help='list the jobs without running them'); s.set_defaults(f=cmd_evaluate)
    s = S.add_parser('verify', help='re-evaluate published missions and check that the results are identical')
    s.add_argument('--platforms', nargs='+', default=['facade'], choices=['facade', 'ground']); s.add_argument('--seed', type=int, default=0)
    s.add_argument('--city', type=int, choices=[21, 22], help='one test city only (default: both)')
    s.add_argument('--missions', default='0-2'); s.add_argument('--keep', action='store_true'); s.add_argument('--workers', **W)
    s.set_defaults(f=cmd_verify)
    s = S.add_parser('sensitivity', help='parameter sensitivity study -> work/results/sens_*')
    s.add_argument('--params', nargs='+', help='subset of N D (default: both)')
    s.add_argument('--checkpoints', default='artifacts'); s.add_argument('--platforms', **PL)
    s.add_argument('--retrain', action='store_true', help='also retrain the policy at N = 16x16 and D_k = 80 Mbit (2 x 2 training runs)')
    s.add_argument('--keep-tables', action='store_true', help='keep the variant tables (about 0.6-1 GB per variant)')
    s.add_argument('--missions', help='mission range per test city (default 0-19, from configs/sensitivity.json)')
    s.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda']); s.add_argument('--workers', **W)
    s.add_argument('--dry-run', action='store_true', help='list the jobs without running them'); s.set_defaults(f=cmd_sensitivity)
    s = S.add_parser('figures', help='every figure and table (PNG, SVG, EPS, Excel, CSV, LaTeX) -> outputs/')
    s.add_argument('--results', default='artifacts', help="'artifacts' (published results, default), 'work' or a folder")
    s.add_argument('--runs', help="training logs for the convergence figures: 'artifacts' or 'work' (default: matches --results)")
    s.add_argument('--out', default=os.path.join(OUTPUTS, 'figures')); s.add_argument('--no-trajectory', action='store_true')
    s.add_argument('--workers', **W); s.add_argument('--dry-run', action='store_true', help='list the jobs without running them'); s.set_defaults(f=cmd_figures)
    s = S.add_parser('reproduce', help='the whole study from scratch: tables, training, evaluation, sensitivity, figures')
    s.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda']); s.add_argument('--workers', **W)
    s.add_argument('--skip-training', action='store_true', help='use the published checkpoints instead of training')
    s.add_argument('--skip-sensitivity', action='store_true'); s.add_argument('--dry-run', action='store_true', help='list the jobs without running them'); s.set_defaults(f=cmd_reproduce)
    return P


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not sys.flags.utf8_mode:
        sys.exit(subprocess.call([sys.executable, '-X', 'utf8', '-m', 'risuav'] + argv, cwd=ROOT))
    args = build_parser().parse_args(argv)
    runner.DRY_RUN = getattr(args, 'dry_run', False)
    os.makedirs(WORK, exist_ok=True)
    sys.exit(args.f(args) or 0)
