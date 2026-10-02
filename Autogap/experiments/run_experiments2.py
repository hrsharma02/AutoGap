"""
AutoGap experimental driver — canonical version.

Run: python experiments/run_experiments2.py
Outputs: figures/*.png (300 DPI), results/*_v2.json

PARAM_SET = "d10" or "d20"

============================================================
KNOWN-GOOD FIXES (do not remove any of these without diffing)
============================================================
FIX-FORCE-RERUN    FORCE_RERUN flag gates both checkpoint loads
FIX-SOBOL-NSAMPLES SOBOL_N_SAMPLES = 64 at ALL dimensions
FIX-SOBOL-FALLBACK Top-3 critical params by ST rank, not list order
FIX-SHAPIRO-DIFF   Shapiro on paired differences, not raw groups
FIX-OMNIBUS-ACTIVE Omnibus tests exclude Direct (constant anchor)
FIX-RS-TWO-BUDGETS RS10 and RS30 both computed per seed
FIX-HOLM           Holm-Bonferroni on all paired-test families
FIX-GD-HELPER      Gap reduction via gd() everywhere
FIX-DR-ISOLATION   Standalone DR characterization at startup
FIX-OPTION-C       Dual-curve Fig 1; AutoGapSimple for Fig 5
FIX-V2-FILENAMES   All outputs suffixed _v2, never overwrite old
============================================================
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
import numpy as np
from tqdm import tqdm

from src.synthetic_sim import SyntheticSimulator, RandomPolicy
from src.gap_metrics import composite_gap, wasserstein_gap
from src.sobol_sensitivity import compute_sobol_indices, identify_gap_critical
from src.bayesian_tuner import BayesianGapTuner
from src import statistical_validation as sv
from src import visualize as viz


# ============================================================
# Config
# ============================================================
PARAM_SET = "d20"           # "d10" or "d20"
FORCE_RERUN = False         # True = ignore all checkpoints

SEEDS = list(range(10))
ABLATION_SEEDS = list(range(5))
N_ROLLOUTS = 50
T_MAX = 30
DIM = 4
HORIZON = 20
SOBOL_N_SAMPLES = 64        # FIX-SOBOL-NSAMPLES: fixed at all dimensions
OUT_DIR = "figures"
RES_DIR = "results"


# ============================================================
# Helpers
# ============================================================
def gd(x, y):
    """Gap reduction: (init - final) / init, guarded."""
    return (x - y) / (x + 1e-12)


def baseline_random_search(bounds, param_names, gap_fn, n_iter, seed):
    rng = np.random.default_rng(seed)
    best_theta, best_gap = None, np.inf
    for _ in range(n_iter):
        theta = {n: rng.uniform(b[0], b[1]) for n, b in zip(param_names, bounds)}
        g = gap_fn(theta)
        if g < best_gap:
            best_gap, best_theta = g, theta
    return best_theta


def baseline_domain_randomization(bounds, param_names, seed):
    rng = np.random.default_rng(seed)
    return {n: rng.uniform(b[0], b[1]) for n, b in zip(param_names, bounds)}


# ============================================================
# FIX-DR-ISOLATION: characterize DR as a single-draw method
# ============================================================
def dr_isolation_test(bounds, param_names, target_params, n_draws=100):
    policy = RandomPolicy(dim=DIM, seed=9999)
    source = SyntheticSimulator(dim=DIM, horizon=HORIZON, seed=9999)
    target = SyntheticSimulator(dim=DIM, horizon=HORIZON,
                                params=target_params, seed=9999 + 1000)

    def gap_fn(theta):
        source.set_params(theta)
        return composite_gap(source.rollout(policy, N_ROLLOUTS),
                             target.rollout(policy, N_ROLLOUTS))["composite"]

    gap_init = gap_fn(source.params)
    rng = np.random.default_rng(42)
    reductions = []
    for _ in range(n_draws):
        theta = {n: rng.uniform(b[0], b[1])
                 for n, b in zip(param_names, bounds)}
        gap = gap_fn(theta)
        reductions.append(gd(gap_init, gap))

    reductions = np.array(reductions)
    return {
        "n_draws": int(n_draws),
        "mean": float(reductions.mean()),
        "std": float(reductions.std()),
        "median": float(np.median(reductions)),
        "q25": float(np.percentile(reductions, 25)),
        "q75": float(np.percentile(reductions, 75)),
        "min": float(reductions.min()),
        "max": float(reductions.max()),
        "frac_negative": float((reductions < 0).mean()),
    }


# ============================================================
# Single seed
# ============================================================
def run_single_seed(seed, param_names, bounds, target_params,
                    ablate_sobol=False, ablate_bo=False,
                    ablate_composite=False):
    policy = RandomPolicy(dim=DIM, seed=seed)
    source = SyntheticSimulator(dim=DIM, horizon=HORIZON, seed=seed)
    target = SyntheticSimulator(dim=DIM, horizon=HORIZON,
                                params=target_params, seed=seed + 1000)

    def rollouts(sim):
        return sim.rollout(policy, n_episodes=N_ROLLOUTS)

    def composite_fn(theta):
        source.set_params(theta)
        return composite_gap(rollouts(source), rollouts(target))["composite"]

    if ablate_composite:
        def opt_fn(theta):
            source.set_params(theta)
            tA = rollouts(source)
            tB = rollouts(target)
            sA = np.concatenate([ep["states"] for ep in tA], axis=0)
            sB = np.concatenate([ep["states"] for ep in tB], axis=0)
            return wasserstein_gap(sA, sB)
    else:
        opt_fn = composite_fn

    gap_init = composite_fn(source.params)
    default_params = source.params.copy()

    # FIX-SOBOL-NSAMPLES: single value, no dimension branch
    sobol_res = compute_sobol_indices(
        composite_fn, param_names, bounds,
        n_samples=SOBOL_N_SAMPLES, seed=seed
    )

    if ablate_sobol:
        critical = list(param_names)
    else:
        critical = identify_gap_critical(sobol_res, threshold=0.05)
        if not critical:
            # FIX-SOBOL-FALLBACK: top-3 by ST rank, not list position
            names = list(sobol_res["names"])
            ST = np.asarray(sobol_res["ST"])
            top3_idx = np.argsort(ST)[::-1][:3]
            critical = [names[i] for i in top3_idx]

    crit_idx = [param_names.index(c) for c in critical]
    crit_bounds = [bounds[i] for i in crit_idx]

    history = {"iter": [0], "gap": [gap_init]}
    if ablate_bo:
        rng = np.random.default_rng(seed)
        best_opt_value = gap_init
        best_crit_values = np.array([default_params[c] for c in critical])
        for t in range(T_MAX):
            theta_crit = np.array([rng.uniform(b[0], b[1])
                                   for b in crit_bounds])
            theta_full = default_params.copy()
            for name, val in zip(critical, theta_crit):
                theta_full[name] = val
            g = opt_fn(theta_full)
            history["iter"].append(t + 1)
            history["gap"].append(g)
            if g < best_opt_value:
                best_opt_value = g
                best_crit_values = theta_crit
    else:
        tuner = BayesianGapTuner(crit_bounds, critical,
                                 length_scale=0.5, seed=seed)
        for t in range(T_MAX):
            theta_crit = tuner.suggest()
            theta_full = default_params.copy()
            for name, val in zip(critical, theta_crit):
                theta_full[name] = val
            g = opt_fn(theta_full)
            tuner.observe(theta_crit, g)
            history["iter"].append(t + 1)
            history["gap"].append(g)
        best_crit_values = np.array(tuner.best()[0])

    best_theta_full = {**default_params,
                       **dict(zip(critical, best_crit_values))}
    gap_autogap = composite_fn(best_theta_full)

    # FIX-RS-TWO-BUDGETS: RS10 (cost-matched) and RS30 (query-matched)
    theta_rs10 = baseline_random_search(
        bounds, param_names, composite_fn, T_MAX // 3, seed)
    gap_rs10 = composite_fn(theta_rs10)

    theta_rs30 = baseline_random_search(
        bounds, param_names, composite_fn, T_MAX, seed + 5000)
    gap_rs30 = composite_fn(theta_rs30)

    theta_dr = baseline_domain_randomization(bounds, param_names, seed)
    gap_dr = composite_fn(theta_dr)

    gap_direct = gap_init

    def transfer_alignment(theta_source):
        from scipy.spatial import cKDTree
        source.set_params(theta_source)
        s_terms = np.array([ep["states"][-1] for ep in rollouts(source)])
        t_terms = np.array([ep["states"][-1] for ep in rollouts(target)])
        tree_t = cKDTree(t_terms)
        self_d, _ = tree_t.query(t_terms, k=2)
        radius = float(np.median(self_d[:, 1])) + 1e-9
        d_st, _ = tree_t.query(s_terms)
        return float(np.exp(-float(np.mean(d_st)) / radius))

    al_autogap = transfer_alignment(best_theta_full)
    al_direct = transfer_alignment(default_params)
    al_rs10 = transfer_alignment(theta_rs10)
    al_rs30 = transfer_alignment(theta_rs30)
    al_dr = transfer_alignment(theta_dr)

    return {
        "seed": seed,
        "gap_init": gap_init,
        "gap_autogap": gap_autogap,
        "gap_rs10": gap_rs10,
        "gap_rs30": gap_rs30,
        "gap_dr": gap_dr,
        "gap_direct": gap_direct,
        "al_autogap": al_autogap,
        "al_rs10": al_rs10,
        "al_rs30": al_rs30,
        "al_dr": al_dr,
        "al_direct": al_direct,
        "history": history,
        "sobol": sobol_res,
        "critical": critical,
        "best_crit_values": list(best_crit_values),
    }


def run_ablation(seed, param_names, bounds, target_params, config):
    kwargs = {
        "no_sobol":      {"ablate_sobol": True},
        "no_bo":         {"ablate_bo": True},
        "single_metric": {"ablate_composite": True},
    }[config]
    r = run_single_seed(seed, param_names, bounds, target_params, **kwargs)
    return gd(r["gap_init"], r["gap_autogap"])


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

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

    if PARAM_SET == "d20":
        dummy_names = [f"dummy_{i}" for i in range(1, 11)]
        dummy_bounds = [(0.0, 1.0)] * 10
        dummy_target = {name: 0.5 for name in dummy_names}
        param_names = base_names + dummy_names
        bounds = base_bounds + dummy_bounds
        target_params = {**base_target, **dummy_target}
    else:
        param_names = base_names
        bounds = base_bounds
        target_params = base_target

    print(f"Parameter set: {PARAM_SET} (d = {len(param_names)})")
    print(f"FORCE_RERUN:   {FORCE_RERUN}")
    print(f"Sobol n_samples: {SOBOL_N_SAMPLES}")

    # ---------- FIX-DR-ISOLATION ----------
    print("\n=== DR-isolation test (100 draws) ===")
    dr_stats = dr_isolation_test(bounds, param_names, target_params,
                                 n_draws=100)
    print(f"  mean:            {dr_stats['mean']:.3f}")
    print(f"  std:             {dr_stats['std']:.3f}")
    print(f"  median [IQR]:    {dr_stats['median']:.3f} "
          f"[{dr_stats['q25']:.3f}, {dr_stats['q75']:.3f}]")
    print(f"  min:             {dr_stats['min']:.3f}")
    print(f"  max:             {dr_stats['max']:.3f}")
    print(f"  frac negative:   {dr_stats['frac_negative']:.3f}")

    # ---------- Main experiment ----------
    print("\nRunning 10 seeds (main experiment)...")
    main_ckpt = f"{RES_DIR}/main_checkpoint_{PARAM_SET}_v2.pkl"
    if os.path.exists(main_ckpt) and not FORCE_RERUN:
        with open(main_ckpt, "rb") as f:
            all_results = pickle.load(f)
        print(f"  Loaded {len(all_results)} seeds from checkpoint")
    else:
        all_results = []
        for seed in tqdm(SEEDS, desc="main"):
            r = run_single_seed(seed, param_names, bounds, target_params)
            r_simple = run_single_seed(seed, param_names, bounds,
                                       target_params, ablate_sobol=True)
            r["gap_autogap_simple"] = r_simple["gap_autogap"]
            r["al_autogap_simple"] = r_simple["al_autogap"]
            # FIX-OPTION-C: store AutoGapSimple fields for dual-curve Fig 1
            r["history_simple"] = r_simple["history"]
            r["critical_simple"] = r_simple["critical"]
            r["best_crit_values_simple"] = r_simple["best_crit_values"]
            all_results.append(r)
            with open(main_ckpt, "wb") as f:
                pickle.dump(all_results, f)
        print(f"  Main experiment complete: {len(all_results)} seeds")

    gap_red_autogap = np.array([gd(r["gap_init"], r["gap_autogap"])
                                for r in all_results])
    gap_red_autogap_simple = np.array(
        [gd(r["gap_init"], r["gap_autogap_simple"]) for r in all_results])
    gap_red_rs10 = np.array([gd(r["gap_init"], r["gap_rs10"])
                             for r in all_results])
    gap_red_rs30 = np.array([gd(r["gap_init"], r["gap_rs30"])
                             for r in all_results])
    gap_red_dr = np.array([gd(r["gap_init"], r["gap_dr"])
                           for r in all_results])
    gap_red_direct = np.array([gd(r["gap_init"], r["gap_direct"])
                               for r in all_results])

    al_autogap = np.array([r["al_autogap"] for r in all_results])
    al_autogap_simple = np.array([r["al_autogap_simple"]
                                  for r in all_results])
    al_rs10 = np.array([r["al_rs10"] for r in all_results])
    al_rs30 = np.array([r["al_rs30"] for r in all_results])
    al_dr = np.array([r["al_dr"] for r in all_results])
    al_direct = np.array([r["al_direct"] for r in all_results])

    # ---------- Statistical tests ----------
    stats_out = {}

    # FIX-SHAPIRO-DIFF: paired differences
    for name, arr in [("rs10", gap_red_rs10), ("rs30", gap_red_rs30),
                      ("dr", gap_red_dr), ("autogap", gap_red_autogap)]:
        stats_out[f"shapiro_diff_simple_vs_{name}"] = sv.shapiro_normality(
            gap_red_autogap_simple - arr)

    simple_tests = {}
    for name, arr in [("rs10", gap_red_rs10), ("rs30", gap_red_rs30),
                      ("dr", gap_red_dr), ("autogap", gap_red_autogap),
                      ("direct", gap_red_direct)]:
        simple_tests[f"vs_{name}"] = {
            "paired_t": sv.paired_ttest(gap_red_autogap_simple, arr),
            "wilcoxon": sv.wilcoxon_test(gap_red_autogap_simple, arr),
            "cliffs_delta": sv.cliffs_delta(gap_red_autogap_simple, arr),
        }
    stats_out["simple_tests"] = simple_tests

    autogap_tests = {}
    for name, arr in [("rs10", gap_red_rs10), ("rs30", gap_red_rs30),
                      ("dr", gap_red_dr)]:
        autogap_tests[f"vs_{name}"] = {
            "paired_t": sv.paired_ttest(gap_red_autogap, arr),
            "wilcoxon": sv.wilcoxon_test(gap_red_autogap, arr),
        }
    stats_out["autogap_tests"] = autogap_tests

    # FIX-HOLM: Holm-Bonferroni on family of Simple-vs-X comparisons
    family_p = [
        simple_tests["vs_rs10"]["paired_t"]["p"],
        simple_tests["vs_rs30"]["paired_t"]["p"],
        simple_tests["vs_dr"]["paired_t"]["p"],
        simple_tests["vs_autogap"]["paired_t"]["p"],
    ]
    family_labels = ["Simple_vs_RS10", "Simple_vs_RS30",
                     "Simple_vs_DR", "Simple_vs_AutoGap"]
    holm_adj = sv.holm_bonferroni(family_p)
    stats_out["holm_family"] = {
        label: {"p_raw": float(pr), "p_holm": float(pa)}
        for label, pr, pa in zip(family_labels, family_p, holm_adj)
    }

    # FIX-OMNIBUS-ACTIVE: exclude Direct from omnibus
    active_methods = [gap_red_autogap, gap_red_autogap_simple,
                      gap_red_rs10, gap_red_rs30, gap_red_dr]
    stats_out["kruskal_active"] = sv.kruskal_wallis(active_methods)
    stats_out["levene_active"] = sv.levene_homogeneity(active_methods)
    stats_out["anova_active"] = sv.one_way_anova(active_methods)
    stats_out["tukey_posthoc_active"] = sv.tukey_hsd_manual(
        active_methods,
        ["AutoGap", "AutoGapSimple", "RS10", "RS30", "DR"])
    stats_out["direct_summary"] = {
        "mean": float(gap_red_direct.mean()),
        "std": float(gap_red_direct.std()),
        "note": "Direct (no tuning) is a constant-zero anchor by "
                "construction; excluded from omnibus tests.",
    }

    for name, arr in [("autogap", gap_red_autogap),
                      ("autogap_simple", gap_red_autogap_simple),
                      ("rs10", gap_red_rs10),
                      ("rs30", gap_red_rs30),
                      ("dr", gap_red_dr),
                      ("direct", gap_red_direct)]:
        stats_out[f"ci_{name}"] = sv.bootstrap_ci(arr)

    all_gaps, all_alignments = [], []
    for r in all_results:
        for gk, ak in [("gap_autogap", "al_autogap"),
                       ("gap_autogap_simple", "al_autogap_simple"),
                       ("gap_rs10", "al_rs10"),
                       ("gap_rs30", "al_rs30"),
                       ("gap_dr", "al_dr"),
                       ("gap_direct", "al_direct")]:
            all_gaps.append(r[gk])
            all_alignments.append(r[ak])
    all_gaps = np.array(all_gaps)
    all_alignments = np.array(all_alignments)

    stats_out["corr_gap_vs_al"] = sv.correlation_with_ci(
        all_gaps, all_alignments, method="pearson")
    stats_out["spearman_gap_vs_al"] = sv.correlation_with_ci(
        all_gaps, all_alignments, method="spearman")

    # ---------- Ablations ----------
    print("\nRunning ablations (5 seeds each, 3 configs)...")
    ablations = {}
    ckpt_path = f"{RES_DIR}/ablations_checkpoint_{PARAM_SET}_v2.json"
    if os.path.exists(ckpt_path) and not FORCE_RERUN:
        with open(ckpt_path) as f:
            ablations = json.load(f)
        print(f"  Loaded prior ablation checkpoint: {ablations}")
    for config in ["no_sobol", "no_bo", "single_metric"]:
        if config in ablations:
            print(f"  Skipping {config} (already done: "
                  f"{ablations[config]:.4f})")
            continue
        reds = []
        for seed in tqdm(ABLATION_SEEDS, desc=config):
            reds.append(run_ablation(seed, param_names, bounds,
                                     target_params, config))
        ablations[config] = float(np.mean(reds))
        with open(ckpt_path, "w") as f:
            json.dump(ablations, f, indent=2)
        print(f"  {config}: {ablations[config]:.4f} (saved)")

    # ---------- Summary ----------
    summary = {
        "param_set": PARAM_SET,
        "num_params": len(param_names),
        "force_rerun": FORCE_RERUN,
        "sobol_n_samples": SOBOL_N_SAMPLES,
        "dr_isolation": dr_stats,
        "methods": {
            "AutoGap": {
                "gap_reduction_mean": float(gap_red_autogap.mean()),
                "gap_reduction_std": float(gap_red_autogap.std()),
                "alignment_mean": float(al_autogap.mean()),
                "alignment_std": float(al_autogap.std()),
            },
            "AutoGapSimple": {
                "gap_reduction_mean": float(gap_red_autogap_simple.mean()),
                "gap_reduction_std": float(gap_red_autogap_simple.std()),
                "alignment_mean": float(al_autogap_simple.mean()),
                "alignment_std": float(al_autogap_simple.std()),
            },
            "RandomSearch10": {
                "gap_reduction_mean": float(gap_red_rs10.mean()),
                "gap_reduction_std": float(gap_red_rs10.std()),
                "alignment_mean": float(al_rs10.mean()),
                "alignment_std": float(al_rs10.std()),
            },
            "RandomSearch30": {
                "gap_reduction_mean": float(gap_red_rs30.mean()),
                "gap_reduction_std": float(gap_red_rs30.std()),
                "alignment_mean": float(al_rs30.mean()),
                "alignment_std": float(al_rs30.std()),
            },
            "DomainRandom": {
                "gap_reduction_mean": float(gap_red_dr.mean()),
                "gap_reduction_std": float(gap_red_dr.std()),
                "alignment_mean": float(al_dr.mean()),
                "alignment_std": float(al_dr.std()),
            },
            "Direct": {
                "gap_reduction_mean": float(gap_red_direct.mean()),
                "gap_reduction_std": float(gap_red_direct.std()),
                "alignment_mean": float(al_direct.mean()),
                "alignment_std": float(al_direct.std()),
            },
        },
        "ablations": {
            "full": float(gap_red_autogap.mean()),
            "autogap_simple": float(gap_red_autogap_simple.mean()),
            "no_sobol": ablations["no_sobol"],
            "no_bo": ablations["no_bo"],
            "single_metric": ablations["single_metric"],
        },
        "statistics": stats_out,
    }

    summary_path = f"{RES_DIR}/summary_{PARAM_SET}_v2.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    with open(f"{RES_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- Figures ----------
    print("\nGenerating figures...")

    # FIX-OPTION-C: dual-curve convergence figure
    best_i_a = int(np.argmax(gap_red_autogap))
    best_i_s = int(np.argmax(gap_red_autogap_simple))
    print(f"  Best seed (AutoGap):       {best_i_a}, "
          f"reduction = {gap_red_autogap[best_i_a]:.4f}")
    print(f"  Best seed (AutoGapSimple): {best_i_s}, "
          f"reduction = {gap_red_autogap_simple[best_i_s]:.4f}")

    viz.fig_dual_convergence(
        all_results[best_i_a]["history"],
        all_results[best_i_s]["history_simple"],
        f"{OUT_DIR}/fig1_convergence.png",
    )
    viz.fig_method_comparison(summary["methods"],
                              f"{OUT_DIR}/fig2_methods.png")

    sb = all_results[0]["sobol"]
    viz.fig_sobol_indices(sb["names"], sb["S1"], sb["ST"],
                          f"{OUT_DIR}/fig3_sobol.png")
    viz.fig_correlation_scatter(all_gaps, all_alignments,
                                stats_out["corr_gap_vs_al"],
                                f"{OUT_DIR}/fig4_correlation.png")

    # Fig 5 uses AutoGapSimple best seed (winning method)
    policy = RandomPolicy(dim=DIM, seed=0)
    src = SyntheticSimulator(dim=DIM, horizon=HORIZON, seed=0)
    tgt = SyntheticSimulator(dim=DIM, horizon=HORIZON,
                             params=target_params, seed=1000)
    traj_A = src.rollout(policy, n_episodes=1)[0]["states"]
    traj_B = tgt.rollout(policy, n_episodes=1)[0]["states"]
    best_theta = {**src.params}
    for name, val in zip(all_results[best_i_s]["critical_simple"],
                         all_results[best_i_s]["best_crit_values_simple"]):
        best_theta[name] = val
    src.set_params(best_theta)
    traj_tuned = src.rollout(policy, n_episodes=1)[0]["states"]
    viz.fig_trajectory_comparison(traj_A, traj_B, traj_tuned,
                                  f"{OUT_DIR}/fig5_trajectory.png")

    viz.fig_boxplots(
        [gap_red_autogap, gap_red_autogap_simple, gap_red_rs10,
         gap_red_rs30, gap_red_dr, gap_red_direct],
        ["AutoGap", "AutoGapSimple", "RS10", "RS30", "DR", "Direct"],
        f"{OUT_DIR}/fig6_boxplots.png")

    rollout_counts = [10, 25, 50, 100, 200, 500]
    gaps_mean, gaps_std = [], []
    for n in rollout_counts:
        gs = [composite_gap(src.rollout(policy, n),
                            tgt.rollout(policy, n))["composite"]
              for _ in range(5)]
        gaps_mean.append(np.mean(gs))
        gaps_std.append(np.std(gs))
    viz.fig_rollout_stability(rollout_counts, gaps_mean, gaps_std,
                              f"{OUT_DIR}/fig7_stability.png")

    ablation_fig = {
        "Full AutoGap": float(gap_red_autogap.mean()),
        "AutoGapSimple": float(gap_red_autogap_simple.mean()),
        "w/o Sobol": ablations["no_sobol"],
        "w/o BO": ablations["no_bo"],
        "Single metric (W1)": ablations["single_metric"],
    }
    viz.fig_ablation(ablation_fig, f"{OUT_DIR}/fig8_ablation.png")

    # ---------- Final summary print ----------
    print(f"\nDone. Figures saved to {OUT_DIR}/")
    print(f"Statistics saved to {summary_path}")
    print()
    print("=== Method comparison ===")
    print(f"  AutoGap:        {gap_red_autogap.mean():.3f} "
          f"± {gap_red_autogap.std():.3f}")
    print(f"  AutoGapSimple:  {gap_red_autogap_simple.mean():.3f} "
          f"± {gap_red_autogap_simple.std():.3f}")
    print(f"  RS10:           {gap_red_rs10.mean():.3f} "
          f"± {gap_red_rs10.std():.3f}")
    print(f"  RS30:           {gap_red_rs30.mean():.3f} "
          f"± {gap_red_rs30.std():.3f}")
    print(f"  DR:             {gap_red_dr.mean():.3f} "
          f"± {gap_red_dr.std():.3f}")
    print(f"  Direct:         {gap_red_direct.mean():.3f} "
          f"± {gap_red_direct.std():.3f}")

    print()
    print("=== Statistical tests (Simple vs ..., raw p) ===")
    for name in ["rs10", "rs30", "dr", "autogap"]:
        t = simple_tests[f"vs_{name}"]["paired_t"]
        print(f"  Simple vs {name:8s} : p = {t['p']:.4e}, "
              f"d = {t['cohens_d']:.3f}")
    print(f"  Levene (active):  p = "
          f"{stats_out['levene_active']['p']:.4e}")
    print(f"  Kruskal (active): H = "
          f"{stats_out['kruskal_active']['H']:.3f}, "
          f"p = {stats_out['kruskal_active']['p']:.4e}")
    print(f"  Correlation:      r = "
          f"{stats_out['corr_gap_vs_al']['r']:.4f}")

    print()
    print("=== Holm-Bonferroni adjusted p-values ===")
    for label in family_labels:
        h = stats_out["holm_family"][label]
        print(f"  {label:22s}: raw = {h['p_raw']:.4e}, "
              f"adj = {h['p_holm']:.4e}")

    print()
    print("=== Shapiro on paired differences ===")
    for k in ["shapiro_diff_simple_vs_rs10",
              "shapiro_diff_simple_vs_rs30",
              "shapiro_diff_simple_vs_dr",
              "shapiro_diff_simple_vs_autogap"]:
        s = stats_out[k]
        print(f"  {k:40s}: W = {s['W']:.3f}, p = {s['p']:.4f}")

    print()
    print("=== Ablations ===")
    for k, v in ablation_fig.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()