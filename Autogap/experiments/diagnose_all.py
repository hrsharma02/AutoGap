"""
Diagnostics for three open issues in the AutoGap manuscript.

Usage (from project root D:\\harish\\AiRSiM\\Autogap):
    python experiments/diagnose_all.py

Produces:
  Step 1 - d=10 Sobol parameter names and ST values (Fig 4 verification)
  Step 2 - Sum S_T over critical vs non-critical params (mechanism claim)
  Step 3 - Independent re-run of w/o BO at d=20 (Table III collision check)
"""
import sys
import os
import pickle

# ------------------------------------------------------------------
# Path setup so the script works from any working directory.
#   _HERE = ...\Autogap\experiments
#   ROOT  = ...\Autogap
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import numpy as np
from run_experiments2 import run_ablation   # noqa: E402

RES_DIR = os.path.join(ROOT, 'results')

# Step 3 reruns the w/o BO ablation. Each seed at d=20 takes roughly
# 20 minutes on a single CPU core. Reduce this value for a faster,
# less statistically reliable check.
N_SEEDS_STEP3 = 5


# ==================================================================
# Helper
# ==================================================================
def _load_checkpoint(param_set):
    path = os.path.join(RES_DIR, f'main_checkpoint_{param_set}_v2.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==================================================================
# Step 1 - Fig 4 verification
# ==================================================================
def step1_d10_sobol():
    print('=' * 68)
    print('STEP 1 - d=10 Sobol parameters (Fig 4 verification)')
    print('=' * 68)

    results = _load_checkpoint('d10')
    sb = results[0]['sobol']
    names = list(sb['names'])
    S1 = np.asarray(sb['S1'])
    ST = np.asarray(sb['ST'])

    print(f'Number of parameters: {len(names)}   (expected 10, no dummy_*)')
    print()
    print(f'{"parameter":>16s}   {"S1":>9s}   {"ST":>9s}')
    print('-' * 40)
    for n, s1, st in zip(names, S1, ST):
        print(f'{n:>16s}   {s1:9.4f}   {st:9.4f}')

    has_dummy = any(n.startswith('dummy_') for n in names)
    print()
    if has_dummy:
        print('[FAIL] Checkpoint contains dummy_* entries - this is d=20 data.')
        print('       Fig 4 must be regenerated from a genuine d=10 run.')
    else:
        print('[OK]   d=10 Sobol array contains only physical parameters.')
        print('       Regenerate Fig 4 from these values.')
    print()
    return names, ST


# ==================================================================
# Step 2 - Variance support
# ==================================================================
def step2_variance_support():
    print('=' * 68)
    print('STEP 2 - Variance support over critical vs non-critical params')
    print('=' * 68)
    print('Tests the mechanism claim in Section VII-A:')
    print('"the non-critical params collectively carry more variance at')
    print(' d=10 than at d=20."')
    print()

    summary = {}
    for pset in ['d10', 'd20']:
        results = _load_checkpoint(pset)
        sb = results[0]['sobol']
        names = list(sb['names'])
        ST = np.asarray(sb['ST'])

        crit = [n for n, s in zip(names, ST) if s > 0.05]
        nc = [n for n, s in zip(names, ST) if s <= 0.05]
        nc_real = [n for n in nc if not n.startswith('dummy_')]

        sum_crit = float(sum(ST[names.index(n)] for n in crit))
        sum_nc_real = float(sum(ST[names.index(n)] for n in nc_real))
        sum_nc_all = float(sum(ST[names.index(n)] for n in nc))

        summary[pset] = {
            'crit': crit,
            'nc_real': nc_real,
            'sum_crit': sum_crit,
            'sum_nc_real': sum_nc_real,
            'sum_nc_all': sum_nc_all,
        }

        print(f'--- {pset} ---')
        print(f'  |Theta_crit| = {len(crit)}  -> {crit}')
        print(f'  Non-critical real params: {nc_real}')
        print(f'  Sum ST over critical:            {sum_crit:.4f}')
        print(f'  Sum ST over non-critical (real): {sum_nc_real:.4f}')
        print(f'  Sum ST over non-critical (all):  {sum_nc_all:.4f}')
        print()

    d10_nc = summary['d10']['sum_nc_real']
    d20_nc = summary['d20']['sum_nc_real']
    print('--- Comparison ---')
    print(f'  Sum ST non-critical (real) at d=10: {d10_nc:.4f}')
    print(f'  Sum ST non-critical (real) at d=20: {d20_nc:.4f}')
    ratio = d10_nc / max(d20_nc, 1e-9)
    print(f'  Ratio d10 / d20: {ratio:.2f}x')
    print()
    if d10_nc > 1.5 * d20_nc:
        print('[SUPPORTED] Non-critical real params carry meaningfully more')
        print('            variance at d=10 than at d=20. Mechanism holds.')
    else:
        print('[NOT SUPPORTED] Non-critical variance is comparable between')
        print('                dimensions. Section VII-A must be rewritten.')
    print()
    return summary


# ==================================================================
# Step 3 - Rerun w/o BO at d=20
# ==================================================================
def step3_wobbo_d20_rerun(n_seeds=N_SEEDS_STEP3):
    print('=' * 68)
    print(f'STEP 3 - Re-run w/o BO at d=20 ({n_seeds} seeds)')
    print('=' * 68)
    print('Checks whether Table III d=20 w/o BO = 0.8352 is genuine, or')
    print('an artifact that happens to coincide with Table I d=10 RS30.')
    print(f'Runtime estimate: ~20 min/seed  ->  ~{n_seeds * 20} min total')
    print()

    # Build the d=20 parameter set exactly as run_experiments2.main() does.
    base_names = [
        "friction", "mass", "damping", "actuator_gain", "sensor_noise",
        "latency", "inertia", "gravity", "compliance", "torque_limit",
    ]
    base_bounds = [
        (0.1, 1.5), (0.5, 2.5), (0.0, 0.5), (0.5, 1.5), (0.0, 0.15),
        (0.5, 1.5), (0.5, 2.0), (9.0, 10.0), (0.0, 0.1), (0.5, 1.5),
    ]
    base_target = {
        "friction": 1.3, "mass": 2.3, "damping": 0.42, "actuator_gain": 0.55,
        "sensor_noise": 0.14, "latency": 1.4, "inertia": 1.8,
        "gravity": 9.81, "compliance": 0.09, "torque_limit": 0.6,
    }
    dummy_names = [f"dummy_{i}" for i in range(1, 11)]
    dummy_bounds = [(0.0, 1.0)] * 10
    dummy_target = {n: 0.5 for n in dummy_names}

    param_names = base_names + dummy_names
    bounds = base_bounds + dummy_bounds
    target_params = {**base_target, **dummy_target}

    print(f'Parameter set: d=20 (d = {len(param_names)})')
    print()

    reds = []
    for seed in range(n_seeds):
        r = run_ablation(seed, param_names, bounds, target_params, 'no_bo')
        reds.append(r)
        print(f'  seed {seed}: {r:.6f}')

    reds = np.array(reds)
    mean = float(reds.mean())
    std = float(reds.std())
    print()
    print(f'  mean = {mean:.6f}')
    print(f'  std  = {std:.6f}')
    print(f'  Table III currently reports 0.8352')
    print()

    if abs(mean - 0.8352) < 5e-4:
        print('[MATCH] Rerun reproduces 0.8352.')
        print('        Collision with Table I d=10 RS30 is a numeric')
        print('        coincidence, not a data-entry error.')
    else:
        print(f'[MISMATCH] Rerun gives {mean:.4f}, not 0.8352.')
        print(f'           Update Table III d=20 w/o BO to {mean:.4f} +/- {std:.4f}')
    print()
    return mean, std


# ==================================================================
if __name__ == '__main__':
    try:
        step1_d10_sobol()
    except Exception as e:
        print(f'[Step 1 failed] {type(e).__name__}: {e}\n')

    try:
        step2_variance_support()
    except Exception as e:
        print(f'[Step 2 failed] {type(e).__name__}: {e}\n')

    try:
        step3_wobbo_d20_rerun()
    except Exception as e:
        print(f'[Step 3 failed] {type(e).__name__}: {e}\n')