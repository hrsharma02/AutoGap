"""Gap quantification metrics: Wasserstein, MMD, DTW."""
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel


def wasserstein_gap(states_A, states_B):
    """
    Compute average 1-Wasserstein distance across state dimensions.
    Handles multivariate via per-dimension averaging.
    """
    states_A = np.asarray(states_A)
    states_B = np.asarray(states_B)
    if states_A.ndim == 1:
        return wasserstein_distance(states_A, states_B)
    dims = states_A.shape[1]
    return float(np.mean([
        wasserstein_distance(states_A[:, d], states_B[:, d])
        for d in range(dims)
    ]))


def mmd_gap(states_A, states_B, sigma=.1):
    """
    Unbiased MMD^2 estimate with RBF kernel.
    Uses the standard U-statistic formulation.
    """
    states_A = np.asarray(states_A)
    states_B = np.asarray(states_B)
    if states_A.ndim == 1:
        states_A = states_A.reshape(-1, 1)
        states_B = states_B.reshape(-1, 1)

    n = len(states_A)
    m = len(states_B)
    if n < 2 or m < 2:
        return 0.0

    gamma = 1.0 / (2.0 * sigma ** 2)
    K_AA = rbf_kernel(states_A, states_A, gamma=gamma)
    K_BB = rbf_kernel(states_B, states_B, gamma=gamma)
    K_AB = rbf_kernel(states_A, states_B, gamma=gamma)

    # Unbiased estimators (exclude diagonal)
    term_A = (K_AA.sum() - np.trace(K_AA)) / (n * (n - 1))
    term_B = (K_BB.sum() - np.trace(K_BB)) / (m * (m - 1))
    term_AB = K_AB.mean()

    mmd2 = term_A + term_B - 2 * term_AB
    return float(max(mmd2, 0.0))


def dtw_distance(traj_A, traj_B):
    """
    Dynamic Time Warping between two trajectories.
    Returns normalized distance (per-step alignment cost).
    """
    traj_A = np.asarray(traj_A)
    traj_B = np.asarray(traj_B)
    if traj_A.ndim == 1:
        traj_A = traj_A.reshape(-1, 1)
        traj_B = traj_B.reshape(-1, 1)

    n, m = len(traj_A), len(traj_B)
    cost = cdist(traj_A, traj_B, metric="euclidean")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / max(n, m))


def composite_gap(traj_A, traj_B, weights=(1/3, 1/3, 1/3), sigma=.1):
    """
    Composite gap: weighted sum of W1, MMD^2, DTW.
    Assumes traj_A, traj_B are dicts with 'states' and 'trajectories'.
    """
    states_A = np.concatenate([t["states"] for t in traj_A], axis=0)
    states_B = np.concatenate([t["states"] for t in traj_B], axis=0)

    w1 = wasserstein_gap(states_A, states_B)
    mmd = mmd_gap(states_A, states_B, sigma=sigma)

    # DTW averaged over paired episodes
    dtws = []
    for ta, tb in zip(traj_A, traj_B):
        dtws.append(dtw_distance(ta["states"], tb["states"]))
    dtw = float(np.mean(dtws))

    w1_w, mmd_w, dtw_w = weights
    return {
        "wasserstein": w1,
        "mmd": mmd,
        "dtw": dtw,
        "composite": w1_w * w1 + mmd_w * mmd + dtw_w * dtw,
    }