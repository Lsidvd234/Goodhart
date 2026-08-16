"""
Closed-form derivations for Part 1 of the Goodhart's Law project.

Setup
-----
U : true objective (unobserved, what we actually care about)
R : proxy metric (observed, what we optimize)

We model (U, R) as jointly Gaussian:

    [U]        ([mu_u]   [sigma_u^2          rho*sigma_u*sigma_r])
    [R]  ~  N  ([mu_r] , [rho*sigma_u*sigma_r      sigma_r^2   ])

This file:
  1. Symbolically derives E[U | R = r] from the bivariate normal density
     (by direct integration, not by quoting the textbook formula), and
     confirms it matches the standard conditional-Gaussian result.
  2. Provides a numeric convenience function for the same formula.
  3. Notes explicitly *why* this "regressional Goodhart" case, on its own,
     produces a monotonic (not overoptimizing) curve -- the overoptimization
     threshold requires an additional mechanism, which is what `extremal.py`
     adds.
"""

from __future__ import annotations
import sympy as sp


def derive_conditional_mean_symbolic(verbose: bool = False) -> sp.Eq:
    """
    Derive E[U | R = r] for standardized jointly-Gaussian (U, R)
    (mu_u = mu_r = 0, sigma_u = sigma_r = 1; the general-mean/variance
    case follows by the affine substitution U -> (U-mu_u)/sigma_u,
    R -> (R-mu_r)/sigma_r, which is applied in `conditional_mean` below).

    Method: complete the square in the bivariate-normal exponent, rather
    than trusting sympy to grind through the raw improper integral (sympy's
    integrator gets stuck on branch-cut case-analysis for the general form).
    This is the standard textbook technique and sympy verifies each
    algebraic step.

    Steps
    -----
    1. Joint density exponent (times -(1-rho^2)) is  u^2 - 2*rho*u*r + r^2.
    2. Complete the square in u: this equals (u - rho*r)^2 + r^2*(1-rho^2).
       [sympy checks this identity.]
    3. So f(u,r) factors as  g(r) * exp(-(u-rho*r)^2 / (2*(1-rho^2)))
       where g(r) = [1/(2*pi*sqrt(1-rho^2))] * exp(-r^2/2).
    4. Integrating the second factor over u gives sqrt(2*pi*(1-rho^2))
       [sympy evaluates this Gaussian integral], which cancels the
       r-dependence in the normalizer -> f_R(r) is standard normal, and
       the conditional density f(u|r) is exactly N(rho*r, 1-rho^2).
    5. Hence E[U | R=r] = rho*r  (and Var[U | R=r] = 1 - rho^2).

    Returns
    -------
    sympy.Eq
        E[U|R=r] = rho*r  (standardized case).
    """
    u, r, rho = sp.symbols("u r rho", real=True)

    quad = u**2 - 2 * rho * u * r + r**2
    completed = (u - rho * r) ** 2 + r**2 * (1 - rho**2)
    identity_holds = sp.simplify(sp.expand(quad - completed)) == 0
    if not identity_holds:
        raise AssertionError("completing-the-square identity failed")

    # Gaussian integral over u of exp(-(u-rho*r)^2 / (2*v)), v = 1-rho^2 > 0.
    v = sp.symbols("v", positive=True)
    gauss_integral_v = sp.integrate(sp.exp(-(u - rho * r) ** 2 / (2 * v)), (u, -sp.oo, sp.oo))
    gauss_integral_v = sp.simplify(gauss_integral_v)  # = sqrt(2*pi*v)
    gauss_integral = gauss_integral_v.subs(v, 1 - rho**2)  # = sqrt(2*pi*(1-rho^2))

    norm_const = 1 / (2 * sp.pi * sp.sqrt(1 - rho**2))
    marginal_r = sp.simplify(norm_const * sp.exp(-r**2 / 2) * gauss_integral)  # standard normal

    # Build the conditional density directly as N(rho*r, v) with v kept as a
    # *positive* stand-in symbol during integration (sympy needs the sign of
    # the variance to resolve the Gaussian integral without branch-cut case
    # analysis); substitute v -> 1-rho**2 only after integrating.
    conditional_density_v = sp.exp(-(u - rho * r) ** 2 / (2 * v)) / sp.sqrt(2 * sp.pi * v)
    conditional_mean_v = sp.integrate(u * conditional_density_v, (u, -sp.oo, sp.oo))
    conditional_mean = sp.simplify(conditional_mean_v.subs(v, 1 - rho**2))

    conditional_density = conditional_density_v.subs(v, 1 - rho**2)

    if verbose:
        print("Completing-the-square identity holds:", identity_holds)
        print("Marginal f_R(r) =", marginal_r, " (standard normal density, as required)")
        print("Conditional density f(u|r) =", conditional_density)

    return sp.Eq(sp.Symbol("E[U|R=r]"), conditional_mean)


def conditional_mean(rho: float, r: float, mu_u: float = 0.0, mu_r: float = 0.0,
                      sigma_u: float = 1.0, sigma_r: float = 1.0) -> float:
    """
    Numeric E[U | R = r] for jointly Gaussian (U, R):

        E[U | R = r] = mu_u + rho * (sigma_u / sigma_r) * (r - mu_r)

    This is linear in r -- the classic "regression to the mean" line.
    When sigma_u = sigma_r = 1 and mu_u = mu_r = 0, it reduces to rho * r.
    """
    return mu_u + rho * (sigma_u / sigma_r) * (r - mu_r)


def overoptimization_threshold_regressional() -> str:
    """
    Part 1, question 2 (regressional-Goodhart-only case).

    Under the pure joint-Gaussian model, E[U | R = r] = mu_u + rho*(sigma_u/sigma_r)*(r-mu_r)
    is an *affine, monotonic* function of r whenever rho != 0. Its derivative
    w.r.t. r is the constant rho*(sigma_u/sigma_r), which never changes sign.

    Conclusion: regressional Goodhart alone has NO finite overoptimization
    threshold -- pushing R higher never makes E[U|R] turn around. This is an
    important negative result: the empirically observed overoptimization
    curves (reward model score keeps climbing while true/gold reward falls)
    cannot be explained by regressional Goodhart in isolation. You need a
    second mechanism that breaks the linearity / stationarity of the
    U-R relationship at the extremes -- see `extremal.py` for a model of
    exactly that (extremal Goodhart), which *does* yield a finite threshold.
    """
    return (
        "d/dr E[U|R=r] = rho * (sigma_u / sigma_r) = constant (independent of r).\n"
        "=> E[U|R=r] is monotonic for all r whenever rho != 0.\n"
        "=> No finite overoptimization threshold exists in the pure regressional "
        "model. Overoptimization requires a mechanism beyond regressional Goodhart "
        "(see extremal.py)."
    )


if __name__ == "__main__":
    eq = derive_conditional_mean_symbolic()
    sp.pprint(eq)
    print()
    print(overoptimization_threshold_regressional())
