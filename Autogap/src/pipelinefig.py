"""
Pipeline diagram for AutoGap. Corrected for v2 results.
Shows Sobol pruning hurts at d=10, neutral at d=20.
Output: experiments1/fig0_pipelineupdated.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

DPI = 300
BLUE = "#0173B2"
ORANGE = "#DE8F05"
GREEN = "#029E73"
GREY = "#949494"
PURPLE = "#CC78BC"
RED = "#D55E00"


def box(ax, xy, w, h, text, color, fontsize=9, bold=False,
        alpha_fill=0.18, lw=1.4):
    x, y = xy
    fill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=color, alpha=alpha_fill)
    ax.add_patch(fill)
    border = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=color, facecolor="none")
    ax.add_patch(border)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color="#111111")


def arrow(ax, p1, p2, color=GREY, style="-|>", lw=1.3, rad=0.0,
          linestyle="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=linestyle, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # ---------- Inputs ----------
    box(ax, (0.2, 5.7), 1.6, 0.8,
        "Source $S_A$", BLUE, fontsize=9, bold=True)
    box(ax, (0.2, 4.7), 1.6, 0.8,
        "Target $S_B$", ORANGE, fontsize=9, bold=True)
    box(ax, (0.2, 3.7), 1.6, 0.8,
        "Policy $\\pi$", PURPLE, fontsize=9)

    # ---------- Rollout ----------
    box(ax, (2.3, 4.4), 1.7, 1.7,
        "Rollouts\n\n"
        "$\\mathcal{T}_{S_A}, \\mathcal{T}_{S_B}$\n"
        "$N=50$ episodes",
        GREY, fontsize=8)

    # ---------- Quantify ----------
    box(ax, (4.4, 4.4), 1.9, 1.7,
        "Composite gap\n\n"
        "$\\mathcal{G} = \\frac{1}{3}W_1$\n"
        "$+\\, \\frac{1}{3}\\mathrm{MMD}^2$\n"
        "$+\\, \\frac{1}{3}\\mathrm{DTW}$",
        BLUE, fontsize=8)

    # ---------- Path A ----------
    box(ax, (7.0, 5.6), 2.7, 0.75,
        "Path A: AutoGap\nSobol prune $\\to$ BO",
        RED, fontsize=9)
    box(ax, (7.0, 4.7), 2.7, 0.75,
        "3-param tuning\n(at $d=10$ and $d=20$)",
        RED, fontsize=8, alpha_fill=0.10)
    # FIX-APPLIED: corrected d=20 number (was -6.8 pp)
    ax.text(8.35, 4.55,
            "$-4.0$ pp @ $d{=}10$\n$-0.04$ pp @ $d{=}20$",
            ha="center", va="top", fontsize=7.5,
            color=RED, style="italic")
    ax.text(9.55, 6.20, "$\\times$", ha="center", va="center",
            fontsize=15, color=RED, fontweight="bold")

    # ---------- Path B ----------
    box(ax, (7.0, 3.2), 2.7, 0.75,
        "Path B: AutoGapSimple\nBO on full space",
        GREEN, fontsize=9, bold=True)
    box(ax, (7.0, 2.3), 2.7, 0.75,
        "Full-param tuning\n$(d = 10$ or $20)$",
        GREEN, fontsize=8, bold=True)
    ax.text(8.35, 2.15,
            "$\\mathbf{85.3\\%}$ gap reduction\nat both dimensions",
            ha="center", va="top", fontsize=7.5,
            color=GREEN, style="italic", fontweight="bold")

    # ---------- Validate ----------
    box(ax, (2.5, 0.5), 3.2, 0.9,
        "Distributional alignment\n"
        "$\\mathrm{Align} = \\exp(-\\bar{d}_{\\mathrm{NN}}/r_{\\mathrm{int}})$",
        PURPLE, fontsize=8)
    box(ax, (6.0, 0.5), 3.6, 0.9,
        "Statistical tests\n"
        "paired $t$, Wilcoxon, Kruskal, Levene",
        GREY, fontsize=8)

    # ---------- Arrows ----------
    arrow(ax, (1.8, 6.1), (2.3, 5.6), BLUE)
    arrow(ax, (1.8, 5.1), (2.3, 5.2), ORANGE)
    arrow(ax, (1.8, 4.1), (2.3, 4.9), PURPLE, rad=-0.15)
    arrow(ax, (4.0, 5.25), (4.4, 5.25))

    arrow(ax, (6.3, 5.6), (7.0, 5.95), BLUE, rad=-0.15)
    arrow(ax, (6.3, 5.0), (7.0, 3.55), BLUE, rad=0.10)

    arrow(ax, (8.35, 5.6), (8.35, 5.45), RED)
    arrow(ax, (8.35, 3.2), (8.35, 3.05), GREEN)

    # Path A dashed red arrow to Statistical tests
    arrow(ax, (9.7, 4.9), (9.6, 1.5),
          RED, rad=-0.20, lw=1.0, linestyle=(0, (4, 3)))

    # Path B solid green arrows to both validation boxes
    arrow(ax, (7.5, 2.3), (5.7, 1.4), GREEN, rad=0.15, lw=1.5)
    arrow(ax, (7.5, 2.3), (7.8, 1.4), GREEN, rad=-0.15, lw=1.5)

    # ---------- Stage labels ----------
    ax.text(0.2, 6.75, "Inputs", fontsize=9,
            fontweight="bold", color=GREY)
    ax.text(2.3, 6.75, "Rollout", fontsize=9,
            fontweight="bold", color=GREY)
    ax.text(4.4, 6.75, "Quantify", fontsize=9,
            fontweight="bold", color=GREY)
    ax.text(7.0, 6.75, "Optimize (two designs)", fontsize=9,
            fontweight="bold", color=GREY)
    ax.text(4.5, 0.15, "Validate", fontsize=9,
            fontweight="bold", color=GREY)

    for x in [2.15, 4.25, 6.75]:
        ax.axvline(x, ymin=0.03, ymax=0.97,
                   linestyle="--", color="#CCCCCC",
                   linewidth=0.6, zorder=0)

    out_path = "experiments/fig0_pipelineupdated.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path} (300 DPI)")


if __name__ == "__main__":
    main()