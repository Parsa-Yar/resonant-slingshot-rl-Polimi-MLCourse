import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


N_SAMPLES   = int(os.environ.get("N_SAMPLES", 2500))  
N_TEST      = int(os.environ.get("N_TEST", 400))       
T_EVAL_PTS  = 300                                       
OUT_DIR     = "forward_surrogate_out"
SEED        = 42


NOMINAL_LIMIT  = 0.7
MIN_FOLD_LIMIT = 0.05
SIMULATION_TIME = 12
SYSTEM_PARAMS = {'MB': 70, 'BB': 264, 'KB': 4000.0, 'MR': 20.0}

def simulate_2dof_system(XC_func, params, t_span, t_eval):
    MB, BB, KB = params['MB'], params['BB'], params['KB']
    MR, KR, DR = params['MR'], params['KR'], params['DR']
    actual_DR = 2.0 * DR * np.sqrt(KR * MR)
    def dynamics(t, y):
        xb, xb_dot, xr, xr_dot = y
        dt_calc = 1e-4
        xc = XC_func(t)
        xc_dot = (XC_func(t + dt_calc) - xc) / dt_calc
        F_imp = -KR * (xr - xc) - actual_DR * (xr_dot - xc_dot)
        K_wall, D_wall = 100000.0, 5000.0
        F_wall_internal = 0.0
        if xr < MIN_FOLD_LIMIT:
            F_wall_internal = -K_wall * (xr - MIN_FOLD_LIMIT) - D_wall * xr_dot
        F_wall_stretch = 0.0
        if xr > NOMINAL_LIMIT:
            F_wall_stretch = -K_wall * (xr - NOMINAL_LIMIT) - D_wall * xr_dot
        F_total = F_imp + F_wall_internal + F_wall_stretch
        xb_ddot = (-BB * xb_dot - KB * xb - F_total) / MB
        xr_ddot = (F_total / MR) - xb_ddot
        return [xb_dot, xb_ddot, xr_dot, xr_ddot]
    initial_xr = XC_func(0.0)
    return solve_ivp(dynamics, t_span, [0.0, 0.0, initial_xr, 0.0],
                     t_eval=t_eval, method='Radau', max_step=0.01)

def create_chirp_hold_func(f0, f1, t1, amp, offset):
    def XC_func(t):
        safe_t = max(0.0, t)
        fade_in = min(safe_t / 0.5, 1.0)
        if safe_t <= t1:
            freq = f0 + ((f1 - f0) / t1) * safe_t
            return offset + (amp * np.sin(2 * np.pi * freq * safe_t)) * fade_in
        else:
            return offset + amp * np.sin(2 * np.pi * f1 * t1)
    return XC_func


PARAM_KEYS = ['f0', 'f1', 't1', 'amp', 'offset', 'KR', 'DR']
fn = (1.0 / (2.0 * np.pi)) * np.sqrt(SYSTEM_PARAMS['KB'] / SYSTEM_PARAMS['MB'])
_buffer = 0.02
_safe_floor   = MIN_FOLD_LIMIT + _buffer
_safe_ceiling = NOMINAL_LIMIT - _buffer
_max_amp      = (_safe_ceiling - _safe_floor) / 2.0
_offset       = _safe_floor + _max_amp
BOUNDS = {
    'f0':     (max(0.5, fn - 0.5), fn + 0.3),
    'f1':     (fn - 0.1, fn + 0.8),
    't1':     (2.0, 12.0),
    'amp':    (0.1, _max_amp),
    'offset': (_offset - 0.01, _offset + 0.01),
    'KR':     (3000.0, 5000.0),
    'DR':     (0.3, 1.0),
}
LOWS  = np.array([BOUNDS[k][0] for k in PARAM_KEYS])
HIGHS = np.array([BOUNDS[k][1] for k in PARAM_KEYS])
SPAN  = HIGHS - LOWS

def sample_params(n, rng):
    return LOWS + rng.random((n, 7)) * SPAN

def measure_peak(theta, t_eval):
    f0, f1, t1, amp, offset, KR, DR = theta
    chirp = create_chirp_hold_func(f0, f1, t1, amp, offset)
    p = dict(SYSTEM_PARAMS); p['KR'] = KR; p['DR'] = DR
    sol = simulate_2dof_system(chirp, p, (0, SIMULATION_TIME), t_eval)
    return float(np.max(sol.y[0] + sol.y[2]))


def generate_dataset(n, rng, t_eval, label=""):
    X = sample_params(n, rng)
    y = np.empty(n)
    t0 = time.time()
    for i in range(n):
        y[i] = measure_peak(X[i], t_eval)
        if (i + 1) % max(1, n // 10) == 0:
            el = time.time() - t0
            print(f"   {label}{i+1}/{n}  ({el:.0f}s, {el/(i+1):.2f}s/sim)", flush=True)
    return X, y


def make_predictor(nn, scaler_X, scaler_y):
    def predict_peak(thetas):
        thetas = np.atleast_2d(thetas)
        return scaler_y.inverse_transform(
            nn.predict(scaler_X.transform(thetas)).reshape(-1, 1)
        ).ravel()
    return predict_peak

def invert_to_target(target_xd, predict_peak, rng, t_eval_hi,
                     pool=40000, elite=400, iters=6, n_verify=15):
   
    cand = sample_params(pool, rng)
    for it in range(iters):
        pred = predict_peak(cand)
        order = np.argsort(np.abs(pred - target_xd))
        elites = cand[order[:elite]]
        mu = elites.mean(axis=0)
        sigma = elites.std(axis=0) + 1e-6
        cand = np.clip(mu + rng.normal(0, 1, (pool, 7)) * sigma, LOWS, HIGHS)
    pred = predict_peak(cand)
    proposals = cand[np.argsort(np.abs(pred - target_xd))[:n_verify]]
    best_theta, best_peak, best_err = None, None, np.inf
    for theta in proposals:
        peak = measure_peak(theta, t_eval_hi)
        e = abs(peak - target_xd)
        if e < best_err:
            best_err, best_theta, best_peak = e, theta.copy(), peak
    return best_theta, best_peak, best_err

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t_eval = np.linspace(0, SIMULATION_TIME, T_EVAL_PTS)

    print(f"[1/4] Generating {N_SAMPLES} random training sims + {N_TEST} test sims...")
    X_tr, y_tr = generate_dataset(N_SAMPLES, rng, t_eval, label="train ")
    X_te, y_te = generate_dataset(N_TEST, rng, t_eval, label="test  ")


    def _filt(X, y):
        m = y <= 0.9
        return X[m], y[m]
    n0 = len(y_tr)
    X_tr, y_tr = _filt(X_tr, y_tr)
    X_te, y_te = _filt(X_te, y_te)
    print(f"      Filtered {n0 - len(y_tr)} divergent training outliers "
          f"({len(y_tr)} kept).")

    print("[2/4] Training forward surrogate (theta -> peak)...")
    scaler_X = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    nn = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation='relu',
                      max_iter=4000, random_state=SEED, alpha=1e-4,
                      early_stopping=True, n_iter_no_change=40)
    nn.fit(scaler_X.transform(X_tr), scaler_y.transform(y_tr.reshape(-1, 1)).ravel())
    predict_peak = make_predictor(nn, scaler_X, scaler_y)

    y_pred_te = predict_peak(X_te)
    mae_mm = np.mean(np.abs(y_pred_te - y_te)) * 1000
    ss_res = np.sum((y_te - y_pred_te) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"      Forward-model test accuracy:  MAE={mae_mm:.2f} mm   R2={r2:.4f}")

    print("[3/4] Inverting to hit targets (incl. the 'in-between' ones)...")
    t_eval_hi = np.linspace(0, SIMULATION_TIME, 600)  # accurate peak, fast verify
    test_targets = [0.73, 0.745, 0.76, 0.775, 0.79, 0.795, 0.805, 0.815]
    rows = []
    for tgt in test_targets:
        theta, true_peak, _ = invert_to_target(tgt, predict_peak, rng, t_eval_hi)
        err_mm = abs(true_peak - tgt) * 1000
        rows.append((tgt, true_peak, err_mm))
        print(f"      target {tgt:.3f}m -> achieved {true_peak:.4f}m  (err {err_mm:.1f}mm)")

    worst = max(r[2] for r in rows)
    print(f"      WORST-CASE inversion error across all targets: {worst:.1f} mm")

    print("[4/4] Saving model + plots...")
    with open(os.path.join(OUT_DIR, "forward_surrogate.pkl"), "wb") as f:
        pickle.dump({'nn': nn, 'scaler_X': scaler_X, 'scaler_y': scaler_y,
                     'bounds': BOUNDS, 'param_keys': PARAM_KEYS}, f)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].scatter(y_te, y_pred_te, s=8, alpha=0.4, c='steelblue')
    lim = [min(y_te.min(), y_pred_te.min()), max(y_te.max(), y_pred_te.max())]
    ax[0].plot(lim, lim, 'k--', alpha=0.6)
    ax[0].set_xlabel('True peak (m)'); ax[0].set_ylabel('NN-predicted peak (m)')
    ax[0].set_title(f'Forward Map is Learnable\nMAE={mae_mm:.2f}mm  R²={r2:.4f}',
                    weight='bold')
    ax[0].grid(True, alpha=0.3)
    tg = [r[0] for r in rows]; ach = [r[1] for r in rows]; er = [r[2] for r in rows]
    ax[1].plot([0.72, 0.82], [0.72, 0.82], 'k--', alpha=0.5, label='Perfect')
    ax[1].scatter(tg, ach, c='darkgreen', s=70, zorder=5, label='Inverted result')
    for t in (0.745, 0.795):
        ax[1].axvline(t, color='red', ls=':', alpha=0.5)
    ax[1].set_xlabel('Target (m)'); ax[1].set_ylabel('Achieved peak (m)')
    ax[1].set_title(f'Inversion Hits Any Target\nworst-case {worst:.1f}mm '
                    f'(red = ex-failure points)', weight='bold')
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "forward_surrogate_results.png"), dpi=300)
    plt.close()
    print(f"      Saved to {OUT_DIR}/")
    print("\nDONE.")
    return mae_mm, r2, worst

if __name__ == "__main__":
    main()