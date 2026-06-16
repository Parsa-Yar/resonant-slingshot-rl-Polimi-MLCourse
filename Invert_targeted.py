import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import forward_surrogate as F 

MODEL_PKL = "forward_surrogate.pkl"
BO_PKL    = "bo_sweep_dataset.pkl"
OUT_DIR   = "forward_surrogate_out"
PKEYS     = ['f0', 'f1', 't1', 'amp', 'offset', 'KR', 'DR']


def load_bo_with_achieved(path):
    """Return (params (N,7), achieved (N,)) from your dataset format."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    if isinstance(d, dict) and 'params_dict' in d and 'targets' in d:
        tgts = list(d['targets'])
        pdict = d['params_dict']
        params = np.array([[float(pdict[t][k]) for k in PKEYS] for t in tgts])
        if 'achieved_dict' in d:
            achieved = np.array([float(d['achieved_dict'][t]) for t in tgts])
        else:
            t_ver = np.linspace(0, F.SIMULATION_TIME, 600)
            achieved = np.array([F.measure_peak(p, t_ver) for p in params])
        return params, achieved
    raise ValueError("Unexpected pkl structure")


def invert_targeted(target, predict_peak, rng, t_ver, bo_lib, bo_achieved,
                    pool=20000, elite=200, global_iters=5,
                    local_iters=4, n_local=20, n_verify_global=8):
    cand = F.sample_params(pool, rng)
    for _ in range(global_iters):
        pred = predict_peak(cand)
        order = np.argsort(np.abs(pred - target))
        elites = cand[order[:elite]]
        mu, sig = elites.mean(0), elites.std(0) + 1e-6
        cand = np.clip(mu + rng.normal(0, 1, (pool, 7)) * sig, F.LOWS, F.HIGHS)
    global_props = cand[np.argsort(np.abs(predict_peak(cand) - target))[:n_verify_global]]

    nearest_idx = int(np.argmin(np.abs(bo_achieved - target)))
    nearest_anchor = bo_lib[nearest_idx]

    mu = nearest_anchor.copy()
    sig = F.SPAN * 0.03
    for _ in range(local_iters):
        loc = np.clip(mu + rng.normal(0, 1, (n_local * 10, 7)) * sig, F.LOWS, F.HIGHS)
        pred = predict_peak(loc)
        order = np.argsort(np.abs(pred - target))
        elites = loc[order[:max(5, n_local // 2)]]
        mu, sig = elites.mean(0), elites.std(0) + 1e-6
    local_refines = np.clip(mu + rng.normal(0, 1, (n_local, 7)) * (F.SPAN * 0.01),
                            F.LOWS, F.HIGHS)

    proposals = np.vstack([global_props, nearest_anchor[None, :], local_refines])
    labels = (["SURR"] * len(global_props)
              + ["BO_NEAREST"]
              + ["BO_LOCAL"] * len(local_refines))

    best = dict(err=np.inf)
    for th, lab in zip(proposals, labels):
        pk = F.measure_peak(th, t_ver)
        e = abs(pk - target)
        if e < best['err']:
            best = dict(err=e, theta=th.copy(), peak=pk, src=lab)
    best['n_sims'] = len(proposals)
    return best


def main():
    rng = np.random.default_rng(7)
    t_ver = np.linspace(0, F.SIMULATION_TIME, 600)

    with open(MODEL_PKL, "rb") as fh:
        M = pickle.load(fh)
    predict_peak = F.make_predictor(M['nn'], M['scaler_X'], M['scaler_y'])

    bo_lib, bo_achieved = load_bo_with_achieved(BO_PKL)
    print(f"Loaded BO library: {bo_lib.shape[0]} entries, "
          f"achieved range [{bo_achieved.min():.4f}, {bo_achieved.max():.4f}] m\n")

    targets = [0.73, 0.745, 0.76, 0.775, 0.79, 0.795, 0.805, 0.815]
    rows = []
    print(f"{'target':>8} | {'achieved':>9} | {'err(mm)':>7} | {'source':>11} | sims")
    print("-" * 56)
    for tgt in targets:
        r = invert_targeted(tgt, predict_peak, rng, t_ver, bo_lib, bo_achieved)
        rows.append((tgt, r['peak'], r['err'] * 1000, r['src'], r['n_sims']))
        print(f"{tgt:8.3f} | {r['peak']:9.4f} | {r['err']*1000:7.1f} | "
              f"{r['src']:>11} | {r['n_sims']}")
    worst = max(r[2] for r in rows)
    avg_sims = np.mean([r[4] for r in rows])
    print("-" * 56)
    print(f"WORST-CASE targeted-hybrid error: {worst:.1f} mm   "
          f"(avg {avg_sims:.0f} sims/target vs ~72 in the old version)")

    plt.figure(figsize=(8, 6))
    plt.plot([0.72, 0.82], [0.72, 0.82], 'k--', alpha=0.5, label='Perfect')
    cols = {'SURR': 'darkgreen', 'BO_NEAREST': 'darkorange', 'BO_LOCAL': 'chocolate'}
    seen = set()
    for t, a, e, s, _ in rows:
        lab = {'SURR': 'Forward surrogate (learned)',
               'BO_NEAREST': 'Nearest BO anchor',
               'BO_LOCAL': 'Local refinement of nearest anchor'}[s]
        plt.scatter(t, a, c=cols[s], s=80, zorder=5,
                    label=lab if s not in seen else None)
        seen.add(s)
    plt.xlabel('Target (m)'); plt.ylabel('Achieved peak (m)')
    plt.title(f'Targeted Hybrid: nearest-anchor + local refinement\n'
              f'worst-case {worst:.1f} mm', weight='bold')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, "targeted_hybrid_results.png"), dpi=300)
    plt.close()
    print(f"Saved plot to {OUT_DIR}/targeted_hybrid_results.png")


if __name__ == "__main__":
    main()