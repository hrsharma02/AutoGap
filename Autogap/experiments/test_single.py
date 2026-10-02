"""
Single-seed test for AutoGap pipeline.
Runs Sobol + BO on seed 0 and reports key metrics.
Runtime: ~3-4 minutes.
"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.run_experiments import run_single_seed

param_names = [
    "friction", "mass", "damping", "actuator_gain", "sensor_noise",
    "latency", "inertia", "gravity", "compliance", "torque_limit",
]
bounds = [
    (0.1, 1.5), (0.5, 2.5), (0.0, 0.5), (0.5, 1.5), (0.0, 0.15),
    (0.5, 1.5), (0.5, 2.0), (9.0, 10.0), (0.0, 0.1), (0.5, 1.5),
]
target_params = {
    "friction": 1.3, "mass": 2.3, "damping": 0.42, "actuator_gain": 0.55,
    "sensor_noise": 0.14, "latency": 1.4, "inertia": 1.8, "gravity": 9.81,
    "compliance": 0.09, "torque_limit": 0.6,
}

print("Running single seed (seed=0)...")
print("This takes 3-4 minutes. Please wait.\n")

r = run_single_seed(seed=0, param_names=param_names,
                    bounds=bounds, target_params=target_params)

print("=" * 55)
print("SINGLE SEED TEST RESULTS")
print("=" * 55)
print(f"gap_init          : {r['gap_init']:.4f}   (before tuning)")
print(f"gap_autogap       : {r['gap_autogap']:.4f}   (after AutoGap)")
print(f"gap_rs            : {r['gap_rs']:.4f}   (random search)")
print(f"gap_dr            : {r['gap_dr']:.4f}   (domain randomization)")
print(f"gap_direct        : {r['gap_direct']:.4f}   (no tuning)")
print()
print(f"critical params   : {r['critical']}")
print()
print(f"alignment (autogap): {r['al_autogap']:.4f}")
print(f"alignment (rs)     : {r['al_rs']:.4f}")
print(f"alignment (dr)     : {r['al_dr']:.4f}")
print(f"alignment (direct) : {r['al_direct']:.4f}")
print()
print("=" * 55)
print("CHECKPOINTS")
print("=" * 55)

gap_red = (r["gap_init"] - r["gap_autogap"]) / r["gap_init"]
print(f"AutoGap gap reduction : {gap_red:.2%}   "
      f"{'PASS' if gap_red > 0.20 else 'FAIL (need >20%)'}")
print(f"AutoGap < gap_init    : {'PASS' if r['gap_autogap'] < r['gap_init'] else 'FAIL'}")
print(f"AutoGap < DR          : {'PASS' if r['gap_autogap'] < r['gap_dr'] else 'FAIL'}")
print(f"AutoGap < RS          : {'PASS' if r['gap_autogap'] < r['gap_rs'] else 'FAIL'}")
print(f"Critical params found : {'PASS' if len(r['critical']) >= 2 else 'FAIL'}")
print()
print("=" * 55)
print("ALIGNMENT DIRECTION CHECK")
print("=" * 55)
print(f"Alignment: AutoGap > Direct : {'PASS' if r['al_autogap'] > r['al_direct'] else 'FAIL'}")
print(f"Alignment: AutoGap > DR     : {'PASS' if r['al_autogap'] > r['al_dr'] else 'FAIL'}")
print(f"Alignment: AutoGap > RS     : {'PASS' if r['al_autogap'] > r['al_rs'] else 'FAIL'}")
print()