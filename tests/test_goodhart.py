import numpy as np
import pytest

from goodhart import model, simulate, extremal


def test_symbolic_derivation_matches_closed_form():
    eq = model.derive_conditional_mean_symbolic()
    # E[U|R=r] = rho * r  (standardized case)
    import sympy as sp
    r, rho = sp.symbols("r rho", real=True)
    assert sp.simplify(eq.rhs - rho * r) == 0


@pytest.mark.parametrize("rho,r", [(0.0, 1.0), (0.5, 2.0), (-0.3, -1.5), (0.99, 0.5)])
def test_conditional_mean_standardized(rho, r):
    # standardized case should reduce to rho * r
    assert model.conditional_mean(rho, r) == pytest.approx(rho * r)


def test_conditional_mean_general_means_variances():
    val = model.conditional_mean(rho=0.5, r=3.0, mu_u=1.0, mu_r=0.0, sigma_u=2.0, sigma_r=1.0)
    expected = 1.0 + 0.5 * (2.0 / 1.0) * (3.0 - 0.0)
    assert val == pytest.approx(expected)


def test_regressional_curve_is_monotonic():
    rho = 0.6
    rs = np.linspace(-5, 5, 50)
    us = np.array([model.conditional_mean(rho, r) for r in rs])
    diffs = np.diff(us)
    assert np.all(diffs > 0)  # strictly increasing for rho > 0 -> no turnover


def test_best_of_n_expected_u_increases_with_n_on_average():
    result = simulate.best_of_n_expected_u(rho=0.7, n_values=[1, 10, 50], trials=1500, seed=1)
    # more optimization pressure (larger n) should not decrease E[U|selected]
    # under the pure regressional model (monotonic, no overoptimization).
    assert result["mean_u"][0] < result["mean_u"][1] < result["mean_u"][2]


def test_extremal_threshold_matches_analytic_1_over_kappa():
    rho0, tau, kappa = 0.7, 1.0, 0.5
    r_star = extremal.overoptimization_threshold(rho0, tau, kappa)
    assert r_star == pytest.approx(1.0 / kappa)


def test_extremal_threshold_raises_if_inside_trusted_region():
    # tau larger than 1/kappa -> critical point falls inside the trusted
    # region, so there's no genuine interior extremal turnover.
    with pytest.raises(ValueError):
        extremal.overoptimization_threshold(rho0=0.7, tau=10.0, kappa=0.5)


def test_extremal_curve_rises_then_falls_past_threshold():
    rho0, tau, kappa = 0.7, 1.5, 0.6
    r_star = extremal.overoptimization_threshold(rho0, tau, kappa)
    curve = extremal.overoptimization_curve(rho0, tau, kappa, r_max=8.0, n_points=2000)
    r, eu = curve["r"], curve["mean_u"]

    below = eu[r < r_star - 0.05]
    above = eu[r > r_star + 0.05]
    # value near the threshold should exceed values well past it (turnover)
    assert eu[np.argmin(np.abs(r - r_star))] > above[-1]
    assert np.all(np.diff(below[-10:]) >= -1e-9) or below.size < 10  # still rising just before threshold
