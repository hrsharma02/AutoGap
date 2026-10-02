"""Statistical tests for comparing AutoGap against baselines."""
import numpy as np
from scipy import stats


def paired_ttest(scores_A, scores_B):
    """Paired t-test with Cohen's d effect size."""
    a, b = np.asarray(scores_A), np.asarray(scores_B)
    t, p = stats.ttest_rel(a, b)
    diff = a - b
    d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
    return {"t": float(t), "p": float(p), "cohens_d": d, "mean_diff": float(np.mean(diff))}


def wilcoxon_test(scores_A, scores_B):
    """Non-parametric paired test."""
    a, b = np.asarray(scores_A), np.asarray(scores_B)
    if np.allclose(a, b):
        return {"stat": 0.0, "p": 1.0}
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    return {"stat": float(stat), "p": float(p)}


def one_way_anova(groups):
    """ANOVA across multiple groups (list of arrays)."""
    F, p = stats.f_oneway(*groups)
    return {"F": float(F), "p": float(p)}


def kruskal_wallis(groups):
    """Non-parametric alternative to ANOVA."""
    H, p = stats.kruskal(*groups)
    return {"H": float(H), "p": float(p)}


def tukey_hsd_manual(groups, group_names):
    """
    Pairwise Welch t-tests with Bonferroni correction as a simple post-hoc.
    For strict Tukey HSD, use statsmodels.
    """
    n = len(groups)
    comparisons = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.asarray(groups[i]), np.asarray(groups[j])
            t, p = stats.ttest_ind(a, b, equal_var=False)
            comparisons.append({
                "pair": (group_names[i], group_names[j]),
                "t": float(t),
                "p_raw": float(p),
            })
    # Bonferroni correction
    m = len(comparisons)
    for c in comparisons:
        c["p_adj"] = min(c["p_raw"] * m, 1.0)
        c["significant"] = c["p_adj"] < 0.05
    return comparisons


def bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap 95% confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    n = len(scores)
    means = [scores[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return {"mean": float(scores.mean()), "lo": float(lo), "hi": float(hi)}


def shapiro_normality(scores):
    """Shapiro-Wilk test for normality."""
    if len(scores) < 3:
        return {"W": np.nan, "p": np.nan}
    W, p = stats.shapiro(scores)
    return {"W": float(W), "p": float(p)}


def levene_homogeneity(groups):
    """Levene test for equal variances."""
    stat, p = stats.levene(*groups)
    return {"stat": float(stat), "p": float(p)}


def correlation_with_ci(x, y, method="pearson", seed=42, n_boot=2000):
    """Correlation with bootstrap CI."""
    x, y = np.asarray(x), np.asarray(y)
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)

    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if method == "pearson":
            rr, _ = stats.pearsonr(x[idx], y[idx])
        else:
            rr, _ = stats.spearmanr(x[idx], y[idx])
        rs.append(rr)
    rs = np.array(rs)
    return {
        "r": float(r),
        "p": float(p),
        "ci_lo": float(np.percentile(rs, 2.5)),
        "ci_hi": float(np.percentile(rs, 97.5)),
    }


def cliffs_delta(a, b):
    """Non-parametric effect size (Cliff's delta)."""
    a, b = np.asarray(a), np.asarray(b)
    m, n = len(a), len(b)
    greater = sum((ai > bj) for ai in a for bj in b)
    lesser = sum((ai < bj) for ai in a for bj in b)
    return float((greater - lesser) / (m * n))

def holm_bonferroni(p_values):
    """
    Holm-Bonferroni step-down correction.
    Input:  list/array of raw p-values
    Output: adjusted p-values in the SAME order as input
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    adjusted_sorted = np.zeros(n)
    running_max = 0.0
    for k in range(n):
        adj = sorted_p[k] * (n - k)
        running_max = max(running_max, adj)
        adjusted_sorted[k] = min(running_max, 1.0)

    adjusted = np.zeros(n)
    adjusted[sorted_idx] = adjusted_sorted
    return adjusted

