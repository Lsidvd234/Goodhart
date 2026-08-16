"""
Simulation companion to model.py.

Best-of-n selection: draw n iid samples of (U, R) from the jointly Gaussian
model, pick the one with the highest R (this is "optimization pressure" --
larger n = more pressure), and look at the U of the selected sample. This is
the standard best-of-n / rejection-sampling picture used in RLHF
overoptimization studies (Gao et al. and others use n as a proxy for
optimization strength, analogous to KL budget for policy-gradient RL).

Under the *pure* regressional model (this file), E[U | selected] increases
monotonically with n and asymptotes -- it does not turn over. That's the
numerical confirmation of the negative result proven in model.py. The
turnover (true overoptimization) requires the nonstationary mechanism added
in extremal.py.
"""

from __future__ import annotations
import numpy as np


def sample_joint_gaussian(rho: float, n_samples: int, rng: np.random.Generator,
                           mu_u: float = 0.0, mu_r: float = 0.0,
                           sigma_u: float = 1.0, sigma_r: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Draw n_samples iid pairs (U, R) from the bivariate Gaussian model."""
    mean = [mu_u, mu_r]
    cov = [[sigma_u**2, rho * sigma_u * sigma_r],
           [rho * sigma_u * sigma_r, sigma_r**2]]
    draws = rng.multivariate_normal(mean, cov, size=n_samples)
    return draws[:, 0], draws[:, 1]


def best_of_n_expected_u(rho: float, n_values: list[int], trials: int = 20_000,
                          seed: int = 0, **gaussian_kwargs) -> dict[str, np.ndarray]:
    """
    For each n in n_values: repeat `trials` times -> draw n samples of (U,R),
    keep the one with max R, record its U. Return the mean (and stderr) of U
    over trials, as a function of n.

    Returns
    -------
    dict with keys "n", "mean_u", "stderr_u", "mean_r_selected"
    """
    rng = np.random.default_rng(seed)
    means_u = np.empty(len(n_values))
    stderr_u = np.empty(len(n_values))
    means_r = np.empty(len(n_values))

    for i, n in enumerate(n_values):
        selected_u = np.empty(trials)
        selected_r = np.empty(trials)
        for t in range(trials):
            u, r = sample_joint_gaussian(rho, n, rng, **gaussian_kwargs)
            best = np.argmax(r)
            selected_u[t] = u[best]
            selected_r[t] = r[best]
        means_u[i] = selected_u.mean()
        stderr_u[i] = selected_u.std(ddof=1) / np.sqrt(trials)
        means_r[i] = selected_r.mean()

    return {
        "n": np.array(n_values),
        "mean_u": means_u,
        "stderr_u": stderr_u,
        "mean_r_selected": means_r,
    }


def gradient_style_optimization(rho: float, pressure_values: list[float],
                                 sigma_u: float = 1.0, sigma_r: float = 1.0) -> dict[str, np.ndarray]:
    """
    Alternative notion of "optimization pressure": rather than best-of-n
    sampling, directly evaluate E[U | R = r] along increasing r (as if a
    gradient-based optimizer pushes the *expected* proxy value up to r).
    This is the closed-form curve from model.conditional_mean, exposed here
    so it can be plotted on the same axes as the sampling-based curve for
    comparison (they should agree in the large-n / large-r limit up to the
    reparametrization between "n" and "r").
    """
    from . import model
    r = np.array(pressure_values, dtype=float)
    eu = np.array([model.conditional_mean(rho, ri, sigma_u=sigma_u, sigma_r=sigma_r) for ri in r])
    return {"r": r, "mean_u": eu}
