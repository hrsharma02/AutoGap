# AutoGap: An Empirical Study of Bayesian Optimization for Simulator Discrepancy Reduction
-
## Overview

Simulators are the primary training ground for learning-based robot control, but
no simulator reproduces reality exactly. The resulting discrepancy — the
*reality gap* — is a persistent problem. Existing solutions are fragmented:
domain randomization blindly robustifies policies, system identification
requires real-world data, and automated tuning is simulator-specific.

This study asks a narrower and more tractable question:

> Given a distributional gap metric, what is the best way to use it for
> simulator parameter tuning?

We evaluate three design choices on a controlled synthetic simulator pair at
parameter dimensions $d=10$ and $d=20$, each over ten seeds:

1. **Which gap metric to optimize** — a composite of Wasserstein distance,
   Maximum Mean Discrepancy, and Dynamic Time Warping, or 1-Wasserstein only.
2. **Whether to prune the parameter space** using Sobol global sensitivity
   analysis before tuning.
3. **Whether to use Bayesian optimization or random search** at matched query
   budget.

### Key Findings

- Bayesian optimization over the **full** parameter space (**AutoGapSimple**)
  achieves **85.3 %** gap reduction at both dimensions with standard deviation
  below 0.004.
- **Sobol pruning does not systematically help.** It hurts by 4.0 percentage
  points at $d=10$ and is neutral at $d=20$.
- The **composite gap metric provides no consistent advantage** over
  Wasserstein-only optimization.
- The composite gap is strongly inversely correlated with an independent
  distributional alignment metric ($r \approx -0.91$ at both dimensions),
  validating it as a **diagnostic** even though it is not the best
  **optimization objective**.
- Bayesian optimization outperforms query-matched random search at both
  dimensions after Holm–Bonferroni correction
  ($p_{\text{adj}} = 0.045$ at $d=10$; $p_{\text{adj}} = 0.029$ at $d=20$).

---

## Repository Structure
