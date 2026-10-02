"""Sobol global sensitivity analysis for gap-critical parameter identification."""
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol


def compute_sobol_indices(gap_function, param_names, bounds, n_samples=256, seed=42):
    """
    Sobol first-order and total-order sensitivity indices.

    Args:
        gap_function: callable(theta_dict) -> float (gap value)
        param_names: list of parameter names
        bounds: list of (low, high) tuples
        n_samples: base sample count (total = n_samples * (2d + 2))

    Returns:
        dict with S1, ST, and names
    """
    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": bounds,
    }
    X = saltelli.sample(problem, n_samples, calc_second_order=False)
    Y = np.array([gap_function(dict(zip(param_names, x))) for x in X])

    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    return {
        "names": param_names,
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    }


def identify_gap_critical(sobol_results, threshold=0.05):
    """Return parameter names whose total-order index exceeds threshold."""
    names = sobol_results["names"]
    ST = sobol_results["ST"]
    return [n for n, s in zip(names, ST) if s > threshold]