"""
Part 1, question 3: extend the model to a second Goodhart mechanism.

Manheim & Garrabrant (2018) distinguish four mechanisms:
  - Regressional: R is a noisy measure of U; optimizing R regresses to the
    mean of U given R. Modeled in model.py -- no turnover on its own.
  - Extremal: the (R, U) relationship that holds in the training
    distribution breaks down in the tails, because the tails are exactly
    where the process generating that relationship was least tested.
  - Causal:  R and U share a common cause but R does not cause U; optimizing
    R directly (rather than the shared cause) drives a wedge between them.
  - Adversarial: an agent actively manipulates R once it learns R is the
    optimization target.

This file models EXTREMAL Goodhart: the correlation between U and R is not
constant -- it degrades as |R| moves away from the region where the proxy
was "trained" / is trustworthy. Concretely, we let the *effective*
correlation rho(r) decay smoothly outside a trusted window [-tau, tau]:

    rho(r) = rho0                                   for |r| <= tau
    rho(r) = rho0 * exp(-kappa * (|r| - tau))        for |r| >  tau

and define E[U|R=r] by locally applying the regressional formula with
rho(r) in place of a constant rho, i.e.

    E[U | R=r] = rho(r) * r     (standardized case, sigma_u = sigma_r = 1)

This is a reduced-form model, not a claim about the "true" functional form of
the extremal breakdown -- but it's the simplest one that (a) matches the
regressional model exactly inside the trusted region, (b) is continuous, and
(c) produces a genuine interior maximum (i.e. a finite overoptimization
threshold), which is the qualitative signature this project needs to
reproduce.
"""

from __future__ import annotations
import numpy as np
import sympy as sp


def rho_of_r(r: np.ndarray, rho0: float, tau: float, kappa: float) -> np.ndarray:
    """Effective, r-dependent correlation capturing extremal Goodhart decay."""
    r = np.asarray(r, dtype=float)
    decay = np.exp(-kappa * (np.abs(r) - tau))
    return np.where(np.abs(r) <= tau, rho0, rho0 * decay)


def expected_u_extremal(r: np.ndarray, rho0: float, tau: float, kappa: float) -> np.ndarray:
    """E[U|R=r] under the extremal-Goodhart model (standardized U, R)."""
    r = np.asarray(r, dtype=float)
    return rho_of_r(r, rho0, tau, kappa) * r


def overoptimization_threshold(rho0: float, tau: float, kappa: float) -> float:
    """
    Solve d/dr E[U|R=r] = 0 for r > tau (the region where the proxy is
    degrading), giving the finite optimization pressure beyond which pushing
    R further makes true utility U decrease in expectation.

    For r > tau: E[U|r] = rho0 * exp(-kappa*(r - tau)) * r
    d/dr E[U|r] = rho0 * exp(-kappa*(r-tau)) * (1 - kappa*r)
    Setting to zero (the exponential is never zero): r* = 1 / kappa.

    This is solved symbolically below (rather than just stated) so the
    result is checked, and returned only if it actually lies in the r > tau
    regime this derivation assumed; otherwise the maximum is at the boundary
    r = tau and there's no interior extremal turnover for these parameters.
    """
    r, k = sp.symbols("r k", positive=True)
    rho0_s, tau_s = sp.symbols("rho0 tau", real=True)
    expr = rho0_s * sp.exp(-k * (r - tau_s)) * r
    deriv = sp.diff(expr, r)
    crit_points = sp.solve(sp.Eq(deriv, 0), r)
    # Expect a single critical point r = 1/k, independent of rho0, tau.
    assert len(crit_points) == 1
    r_star_symbolic = crit_points[0]  # sp expression, should simplify to 1/k
    r_star = float(r_star_symbolic.subs(k, kappa))

    if r_star <= tau:
        raise ValueError(
            f"Critical point r*={r_star:.4f} falls inside the trusted region "
            f"(tau={tau}); with these parameters the curve is still rising "
            f"at r=tau and monotonic decay dominates immediately after -- "
            f"check kappa/tau, or treat tau itself as the (degenerate) threshold."
        )
    return r_star


def overoptimization_curve(rho0: float, tau: float, kappa: float,
                            r_max: float = 8.0, n_points: int = 400) -> dict[str, np.ndarray]:
    """Convenience: (r, E[U|r]) arrays over [0, r_max] for plotting."""
    r = np.linspace(0.0, r_max, n_points)
    eu = expected_u_extremal(r, rho0, tau, kappa)
    return {"r": r, "mean_u": eu}


if __name__ == "__main__":
    rho0, tau, kappa = 0.7, 1.5, 0.6
    r_star = overoptimization_threshold(rho0, tau, kappa)
    print(f"rho0={rho0}, tau={tau}, kappa={kappa} -> overoptimization threshold r* = {r_star:.4f}")
    print(f"  (analytically, r* = 1/kappa = {1/kappa:.4f}, independent of rho0 and tau, "
          f"provided r* > tau)")
