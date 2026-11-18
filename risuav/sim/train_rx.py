"""NEW (audit implementation, 17 Nov 2026) -- RECONSTRUCTS the lost train_mr.py.

Sources of the reconstruction (no other information was used):
  * train_prelim50.py (preliminary training script, kept in the authors' archive of the same authors' pipeline): REINFORCE with greedy rollout
    baseline, paired one-sided t-test (alpha = 0.05) on a fixed validation set, Adam 3e-4, StepLR(100, 0.99), gradient
    clipping 1.0, resumable state saved atomically after every epoch;
  * the original experiment report (REPORT_multiris.md, authors' archive): 50 epochs x 10 updates x 64 missions, seeds 0-2, coupled masks during decoding,
    repair under the full deployment, validation = greedy policy without local search, best-validation checkpoint;
  * the manuscript (Sec. IV-C): cost = UAV energy + constraint-violation penalty.
Approved changes against the original run: four training layouts (11-14) instead of two; cell feature with K_i
(model_rx.py); penalty term (below). Everything else as in the sources.

Training model (reconstruction decision, documented in docs/IMPLEMENTATION_DECISIONS.md, R3): the policy is trained on the
exact serving-only tables (non-serving reflections off: CoupledProposed.couple = False, repair/evaluation with the
serving panel only). Reasons: (i) the original report gives 2.5-3 h CPU per run, which is only compatible with
uncoupled training (coupled rollouts cost ~0.5-1.7 s per mission here, i.e. 20-60 h per run); (ii) the original
sensitivity analysis found that switching the unintended reflections off changes the energy of the final plans by at
most 0.009%. All test evaluations use the full coupled model. Set TRAIN_MODEL=coupled to train with coupling.

Training cost C(O) [J] = E_UAV + PEN,
  PEN = 5e4 * n_unrepairable_cells + max(0, E_UAV + E_vert - (E_bat - E_res)) + P_h * max(0, T - T_max).
  (50 kJ per cell that cannot be served under the full deployment; battery and deadline excess converted to joules.)

usage: train_rx.py <facade|ground> <seed> [epochs=50] [steps=10] [batch=64]
env:   MR_ROOT, RUNS_DIR (rx_paths.py), TRAIN_IDS (default 11,12,13,14), RUN_TAG (default proposed), MIRROR (optional dir),
       TRAIN_DEVICE (cpu | cuda | auto; default cpu). On a GPU the policy network runs on the GPU; decoding to plans, repair
       and energy evaluation stay on the CPU. GPU runs use the GPU random stream and are not bit-identical to CPU runs."""
import sys, os, json, time, copy, random, shutil, numpy as np, torch
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import param_override  # noqa: F401  RX_OVERRIDE (retraining for the sensitivity study)
import env_mr as EM
from env_mr import City, make_mission, servable, evaluate
from batch import CityArrays
from model_rx import RxProposed, make_batch_rx
from plan_mr import decode_stops, to_eval, repair
import fastmodel
fastmodel.enable_point_checkpoint()   # model_mr disables it at import (inference); training needs it (memory: B x L x D per step)

torch.set_num_threads(1)
DEV = os.environ.get('TRAIN_DEVICE', 'cpu')
if DEV == 'auto':
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEV != 'cpu':
    if DEV.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit(f'TRAIN_DEVICE={DEV} but torch.cuda.is_available() is False (install a CUDA build of PyTorch)')
    torch.set_default_device(DEV)                       # policy parameters and decoding tensors on the GPU
scen, seed = sys.argv[1], int(sys.argv[2])
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 64
from rx_paths import MR_ROOT as R, RUNS_DIR
pre = 'g' if scen == 'ground' else ''
TRAIN = [f'{pre}city{i}' for i in os.environ.get('TRAIN_IDS', '11,12,13,14').split(',')]
TAG = os.environ.get('RUN_TAG', 'proposed')
OUT = os.path.join(RUNS_DIR, f'{TAG}_{scen}_s{seed}'); os.makedirs(OUT, exist_ok=True)
MIRROR = os.environ.get('MIRROR')
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); rng = np.random.default_rng(seed)
TRAIN_MODEL = os.environ.get('TRAIN_MODEL', 'serving')
if TRAIN_MODEL == 'serving':
    os.environ['LEAN'] = '1'                                      # memory: arrays unused by serving-only training
from rx_common import load_city, DS
CDIR = os.environ.get('CITY_DIR', 'cities')
arrs = [CityArrays(load_city(os.path.join(R, CDIR, f'{c}.pkl'))) for c in TRAIN]
if TRAIN_MODEL == 'serving':
    for a_ in arrs:                                                     # serving-panel-only service tables (no coupling)
        a_.city.svc = (lambda c: (lambda h, o, dep, _orig=c.svc: _orig(h, o, frozenset())))(a_.city)
PEN_CELL = 5e4


def cost_of(r, nbad):
    E = r['E_kJ'] * 1e3
    pen = PEN_CELL * nbad + max(0.0, E + EM.E_VERT - (EM.E_BAT - EM.E_RES)) + EM.P_H * max(0.0, r['T_s'] - EM.T_MAX)
    return E + pen, pen


def sample_missions(n):
    out = []
    while len(out) < n:
        a = arrs[int(rng.integers(len(arrs)))]
        m = make_mission(a.city, int(rng.integers(100000, 10 ** 7)))
        if servable(a.city, m, allow_ris=True):
            out.append((a, m))
    return out


VAL = []
for a in arrs:
    for k in range(20000, 20050):
        m = make_mission(a.city, k)
        if servable(a.city, m, allow_ris=True):
            VAL.append((a, m))


def rollout(net, missions, mode):
    bt = make_batch_rx(missions, D_bits=40e6 * DS)
    if DEV != 'cpu':
        bt = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in bt.items()}
    seq, lp, ent = net(bt, mode=mode)
    S = seq.cpu().numpy(); out = []
    for b, (a, m) in enumerate(missions):
        st = decode_stops(a.city, m, S[b], bt['Kp'])
        st, nrep, nbad = repair(a.city, m, st)
        r = evaluate(a.city, m, to_eval(st)); c, pen = cost_of(r, nbad)
        out.append(dict(cost=c, pen=pen, E_kJ=r['E_kJ'], nrep=nrep, nbad=nbad,
                        feasible=bool(r['prr_ok'] and r['served_once'] and r['battery_ok'] and r['deadline_ok'] and nbad == 0),
                        L_km=r['L_km'], stops=r['stops'], tau_s=r['tau_s']))
    return out, lp, ent


def val_costs(net):
    net.eval(); res = []
    with torch.no_grad():
        for i in range(0, len(VAL), 50):
            rr, _, _ = rollout(net, VAL[i:i + 50], 'greedy'); res += rr
    net.train()
    return (np.array([r['cost'] for r in res]), np.array([r['E_kJ'] for r in res]),
            np.array([r['feasible'] for r in res]), res)


def atomic_save(obj, path):
    tmp = path + '.tmp'; torch.save(obj, tmp); os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def mirror():
    if MIRROR:
        dst = os.path.join(MIRROR, os.path.basename(OUT)); os.makedirs(dst, exist_ok=True)
        for f in os.listdir(OUT):
            if f.endswith('.tmp'):
                continue
            s, d = os.path.join(OUT, f), os.path.join(dst, f)
            if not os.path.exists(d) or os.path.getmtime(s) > os.path.getmtime(d):
                shutil.copy2(s, d + '.tmp'); os.replace(d + '.tmp', d)


net = RxProposed(); net.couple = (TRAIN_MODEL == 'coupled')
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.99)
bl = copy.deepcopy(net); bl.eval()
STATE = f'{OUT}/state.pt'

if os.path.exists(STATE):
    S = torch.load(STATE, map_location='cpu', weights_only=False)
    net.load_state_dict(S['net']); bl.load_state_dict(S['bl']); bl.eval()
    opt.load_state_dict(S['opt']); sched.load_state_dict(S['sched'])
    blC = S['blC']; log = S['log']; step = S['step']; ep0 = S['epoch']; sec0 = S['sec']; bestE = S['bestE']
    rng.bit_generator.state = S['np_rng']; np.random.set_state(S['np_global']); random.setstate(S['py_rng'])
    torch.set_rng_state(S['torch_rng'])
    if DEV != 'cpu' and S.get('cuda_rng') is not None:
        torch.cuda.set_rng_state_all(S['cuda_rng'])
    log.setdefault('resumes', []).append(dict(at_epoch=ep0, time=time.time()))
    print(f'{scen} s{seed} resumed after epoch {ep0} (step {step})', flush=True)
else:
    blC, _, _, _ = val_costs(bl)
    log = dict(script='train_rx.py (reconstruction of train_mr.py)', scen=scen, seed=seed, epochs=EPOCHS,
               steps_per_epoch=STEPS, batch=BATCH, n_val=len(VAL), train_cities=TRAIN, lr=3e-4,
               lr_schedule='StepLR(100, 0.99)', grad_clip=1.0, train_model=TRAIN_MODEL, override=param_override.OVR, D_scale=DS, city_dir=CDIR, t_test_alpha=0.05, penalty_per_cell_J=PEN_CELL,
               torch=torch.__version__, device=DEV, steps=[], epochs_log=[])
    c0, e0, ok0, _ = val_costs(net)
    log['epochs_log'].append(dict(epoch=0, val_cost=float(c0.mean()), val_E_kJ=float(e0.mean()), val_E_kJ_std=float(e0.std()),
                                  val_feasible=float(ok0.mean()), baseline_replaced=False, sec=0.0))
    bestE = float(e0.mean()); torch.save(net.state_dict(), f'{OUT}/best.pt')
    print(f'{scen} s{seed} epoch 0 val E {e0.mean():.2f} kJ feas {ok0.mean():.3f} nval {len(VAL)}', flush=True)
    step = 0; ep0 = 0; sec0 = 0.0
t0 = time.time() - sec0
for ep in range(ep0 + 1, EPOCHS + 1):
    for _ in range(STEPS):
        ms = sample_missions(BATCH)
        Rr, lp, ent = rollout(net, ms, 'sample')
        with torch.no_grad():
            Rb, _, _ = rollout(bl, ms, 'greedy')
        c = torch.tensor([r['cost'] for r in Rr]); cb = torch.tensor([r['cost'] for r in Rb])
        adv = (c - cb) / 1e3
        loss = (adv * lp).mean()
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step(); sched.step(); step += 1
        log['steps'].append(dict(step=step, epoch=ep, cost=float(c.mean()), cost_std=float(c.std()), bl_cost=float(cb.mean()),
                                 E_kJ=float(np.mean([r['E_kJ'] for r in Rr])), penalty=float(np.mean([r['pen'] for r in Rr])),
                                 infeasible=int(sum(not r['feasible'] for r in Rr)), repaired=int(sum(r['nrep'] for r in Rr)),
                                 loss=float(loss), grad_norm=gn, entropy=float(ent.mean()),
                                 entropy_per_decision=float((ent / torch.tensor([3.0 * len(m['cells']) + 1 for _, m in ms])).mean()),
                                 nan=bool(torch.isnan(loss)), sec=time.time() - t0, lr=opt.param_groups[0]['lr']))
    cv, ev, okv, rv = val_costs(net)
    p = stats.ttest_rel(cv, blC, alternative='less').pvalue if (cv != blC).any() else 1.0
    rep = bool(cv.mean() < blC.mean() and p < 0.05)
    if rep:
        bl = copy.deepcopy(net); bl.eval(); blC = cv
    st_ep = [s for s in log['steps'] if s['epoch'] == ep]
    log['epochs_log'].append(dict(epoch=ep, val_cost=float(cv.mean()), val_E_kJ=float(ev.mean()), val_E_kJ_std=float(ev.std()),
                                  val_feasible=float(okv.mean()), val_L_km=float(np.mean([r['L_km'] for r in rv])),
                                  val_stops=float(np.mean([r['stops'] for r in rv])), t_test_p=float(p), baseline_replaced=rep,
                                  train_cost=float(np.mean([s['cost'] for s in st_ep])),
                                  train_E_kJ=float(np.mean([s['E_kJ'] for s in st_ep])),
                                  train_penalty=float(np.mean([s['penalty'] for s in st_ep])),
                                  entropy=float(np.mean([s['entropy'] for s in st_ep])),
                                  entropy_per_decision=float(np.mean([s['entropy_per_decision'] for s in st_ep])),
                                  grad_norm=float(np.mean([s['grad_norm'] for s in st_ep])), sec=time.time() - t0))
    if float(ev.mean()) < bestE:
        bestE = float(ev.mean()); atomic_save(net.state_dict(), f'{OUT}/best.pt'); log['best_epoch'] = ep
    if ep % 10 == 0:
        torch.save(net.state_dict(), f'{OUT}/ep{ep}.pt')
    atomic_save(dict(net=net.state_dict(), bl=bl.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), blC=blC,
                     log=log, step=step, epoch=ep, sec=time.time() - t0, bestE=bestE, np_rng=rng.bit_generator.state,
                     np_global=np.random.get_state(), py_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if DEV != 'cpu' else None), STATE)
    atomic_json(log, f'{OUT}/log.json'); mirror()
    print(f'{scen} s{seed} epoch {ep} train {log["epochs_log"][-1]["train_cost"] / 1e3:.1f} kJ val E {ev.mean():.2f} kJ '
          f'feas {okv.mean():.3f} p {p:.3f} repl {rep} {time.time() - t0:.0f}s', flush=True)
log['updates'] = step; log['train_missions'] = step * BATCH; log.setdefault('best_epoch', 0)
atomic_json(log, f'{OUT}/log.json')
atomic_save(net.state_dict(), f'{OUT}/final.pt'); mirror()
print('done', OUT, flush=True)
