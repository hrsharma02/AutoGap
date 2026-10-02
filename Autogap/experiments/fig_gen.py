"""
Regenerate Fig 4 (correlation) and Fig 7 (stability) from saved checkpoints.
Run from project root: python experiments/fix_pub.py
Outputs:
  experiments1/fig4_correlation.png
  experiments1/fig7_stability.png
"""
import sys, os
# Insert project root at the front of sys.path so `import src` works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.synthetic_sim import SyntheticSimulator, RandomPolicy
from src.gap_metrics import composite_gap
from src import statistical_validation as sv
from src import visualize as viz

# Paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.path.join(PROJECT_ROOT, "experiments1")
os.makedirs(FIG_DIR, exist_ok=True)


def regenerate_correlation():
    """Regenerate Fig 4: pooled method-by-seed correlation at d=10."""
    import pickle
    ckpt = os.path.join(PROJECT_ROOT, "results", "main_checkpoint_d10_v2.pkl")
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        return

    with open(ckpt, "rb") as f:
        all_results = pickle.load(f)

    all_gaps, all_alignments = [], []
    for r in all_results:
        for gk, ak in [("gap_autogap", "al_autogap"),
                       ("gap_autogap_simple", "al_autogap_simple"),
                       ("gap_rs10", "al_rs10"),
                       ("gap_rs30", "al_rs30"),
                       ("gap_dr", "al_dr"),
                       ("gap_direct", "al_direct")]:
            if gk in r and ak in r:
                all_gaps.append(r[gk])
                all_alignments.append(r[ak])

    all_gaps = np.array(all_gaps)
    all_alignments = np.array(all_alignments)

    print(f"[Fig 4] Extracted N = {len(all_gaps)} points")

    corr = sv.correlation_with_ci(all_gaps, all_alignments, method="pearson")
    print(f"[Fig 4] Pearson r = {corr['r']:.4f}, "
          f"p = {corr['p']:.4e}, "
          f"CI = [{corr['ci_lo']:.4f}, {corr['ci_hi']:.4f}]")

    out = os.path.join(FIG_DIR, "fig4_correlation.png")
    viz.fig_correlation_scatter(all_gaps, all_alignments, corr, out)
    print(f"[Fig 4] Saved {out}")


def regenerate_stability():
    """Regenerate Fig 7: gap estimate vs rollout count at default parameters."""
    target_params = {
        "friction": 1.3, "mass": 2.3, "damping": 0.42, "actuator_gain": 0.55,
        "sensor_noise": 0.14, "latency": 1.4, "inertia": 1.8,
        "gravity": 9.81, "compliance": 0.09, "torque_limit": 0.6,
    }
    policy = RandomPolicy(dim=4, seed=0)
    src_stab = SyntheticSimulator(dim=4, horizon=20, seed=0)
    tgt_stab = SyntheticSimulator(dim=4, horizon=20,
                                  params=target_params, seed=1000)

    rollout_counts = [10, 25, 50, 100, 200, 500]
    gaps_mean, gaps_std = [], []
    for n in rollout_counts:
        gs = [composite_gap(src_stab.rollout(policy, n),
                            tgt_stab.rollout(policy, n))["composite"]
              for _ in range(5)]
        gaps_mean.append(np.mean(gs))
        gaps_std.append(np.std(gs))
        print(f"[Fig 7] N={n:4d}  gap={gaps_mean[-1]:.4f}  "
              f"std={gaps_std[-1]:.4f}")

    out = os.path.join(FIG_DIR, "fig7_stability.png")
    viz.fig_rollout_stability(rollout_counts, gaps_mean, gaps_std, out)
    print(f"[Fig 7] Saved {out}")


if __name__ == "__main__":
    print("=" * 55)
    print("Regenerating Fig 4 (correlation)")
    print("=" * 55)
    regenerate_correlation()

    print()
    print("=" * 55)
    print("Regenerating Fig 7 (stability)")
    print("=" * 55)
    regenerate_stability()

    print()
    print("Done. Both figures regenerated at 300 DPI.")