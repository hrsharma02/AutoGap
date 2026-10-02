"""300 DPI publication-quality figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Publication style
rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

DPI = 300
PALETTE = sns.color_palette("colorblind", 6)


def fig_gap_convergence(history, out_path):
    """Gap vs BO iteration."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    gaps = np.asarray(history["gap"])
    best_so_far = np.minimum.accumulate(gaps)
    ax.plot(history["iter"], gaps, marker="o", color=PALETTE[0],
            alpha=0.5, label="AutoGap (raw)")
    ax.plot(history["iter"], best_so_far, color=PALETTE[0],
            linewidth=2.2, label="Best so far")
    ax.axhline(gaps[0], linestyle="--", color="gray", label="Initial")
    ax.axhline(best_so_far[-1], linestyle=":", color=PALETTE[2], label="Best tuned")
    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Composite Gap")
    ax.set_title("Gap Minimization Convergence")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_method_comparison(results, out_path):
    """Bar plot with error bars for gap reduction across methods."""
    methods = list(results.keys())
    means = [results[m]["gap_reduction_mean"] for m in methods]
    stds = [results[m]["gap_reduction_std"] for m in methods]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=4,
           color=[PALETTE[i % len(PALETTE)] for i in range(len(methods))])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Gap Reduction $\\Delta\\mathcal{G}$")
    ax.set_title("Method Comparison (mean $\\pm$ std over 10 seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_sobol_indices(names, S1, ST, out_path):
    """Sobol first-order and total-order indices."""
    order = np.argsort(ST)[::-1]
    names_sorted = [names[i] for i in order]
    S1_sorted = np.array(S1)[order]
    ST_sorted = np.array(ST)[order]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(names_sorted))
    ax.bar(x - 0.2, S1_sorted, 0.4, label="$S_i$ (first-order)",
           color=PALETTE[0])
    ax.bar(x + 0.2, ST_sorted, 0.4, label="$S_i^T$ (total-order)",
           color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=30, ha="right")
    ax.set_ylabel("Sobol Index")
    ax.set_title("Gap-Critical Parameter Identification")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_correlation_scatter(gaps, alignments, r_info, out_path):
    """Gap vs distributional alignment correlation."""
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(gaps, alignments, s=25, alpha=0.7, color=PALETTE[0])
    z = np.polyfit(gaps, alignments, 1)
    xs = np.linspace(min(gaps), max(gaps), 100)
    ax.plot(xs, np.polyval(z, xs), color=PALETTE[2],
            label=f"$r={r_info['r']:.3f}$, $p={r_info['p']:.2e}$")
    ax.set_xlabel("Composite Gap $\\mathcal{G}$ (lower is better)")
    ax.set_ylabel("Distributional Alignment (higher is better)")
    ax.set_title("Gap Predicts Distributional Alignment")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_trajectory_comparison(traj_A, traj_B, traj_tuned, out_path):
    """2D projection of trajectories before/after tuning."""
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(traj_A[:, 0], traj_A[:, 1], "o-", label="Source (untuned)",
            color=PALETTE[0], markersize=3)
    ax.plot(traj_B[:, 0], traj_B[:, 1], "s--", label="Target",
            color=PALETTE[1], markersize=3)
    ax.plot(traj_tuned[:, 0], traj_tuned[:, 1], "^:", label="Source (tuned)",
            color=PALETTE[2], markersize=3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Trajectory Distribution Alignment")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_boxplots(groups, names, out_path):
    """Boxplot comparison of gap reduction distributions."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bp = ax.boxplot(groups, labels=names, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Gap Reduction $\\Delta\\mathcal{G}$")
    ax.set_title("Distribution Comparison Across Methods")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_rollout_stability(rollout_counts, gaps_mean, gaps_std, out_path):
    """Gap metric stability vs number of rollouts."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.errorbar(rollout_counts, gaps_mean, yerr=gaps_std, marker="o",
                capsize=4, color=PALETTE[0])
    ax.set_xscale("log")
    ax.set_xlabel("Number of Rollouts $N$")
    ax.set_ylabel("Gap Estimate $\\hat{\\mathcal{G}}$")
    ax.set_title("Metric Stability vs Sample Size")
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_ablation(ablation_results, out_path):
    """Bar chart of ablations."""
    labels = list(ablation_results.keys())
    values = [ablation_results[k] for k in labels]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.barh(labels, values, color=PALETTE[3])
    ax.set_xlabel("Gap Reduction $\\Delta\\mathcal{G}$")
    ax.set_title("Ablation Study")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

def fig_dual_convergence(history_A, history_B, out_path,
                         label_A="AutoGap (Sobol $\\rightarrow$ BO)",
                         label_B="AutoGapSimple (BO full)"):
    """
    Best-so-far convergence curves for two methods on the same axes.
    Shows the central finding: full-space BO reaches a lower minimum.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for hist, color, marker, label in [
        (history_A, PALETTE[3], "o", label_A),
        (history_B, PALETTE[2], "s", label_B),
    ]:
        gaps = np.asarray(hist["gap"])
        best_so_far = np.minimum.accumulate(gaps)
        ax.plot(hist["iter"], best_so_far, marker=marker,
                color=color, markersize=4, linewidth=1.6, label=label)

    ax.axhline(history_A["gap"][0], linestyle="--", color="gray",
               linewidth=1.0, label="Initial gap")
    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Composite Gap (best so far)")
    ax.set_title("Convergence: Pruned vs Full-Space BO")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)