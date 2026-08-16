"""
Plotting helpers. Kept separate from model/simulate/extremal so those stay
matplotlib-free and easy to unit test.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


def plot_regressional_curve(rho: float, r_range=(-4, 4), ax=None, **kwargs):
    """Plot the closed-form E[U|R=r] = rho*r line (standardized case)."""
    from . import model
    ax = ax or plt.gca()
    r = np.linspace(*r_range, 200)
    eu = np.array([model.conditional_mean(rho, ri) for ri in r])
    ax.plot(r, eu, label=f"regressional, rho={rho}", **kwargs)
    ax.set_xlabel("R (proxy, optimization pressure)")
    ax.set_ylabel("E[U | R = r]  (true objective)")
    return ax


def plot_best_of_n(result: dict, ax=None, **kwargs):
    """Plot output of simulate.best_of_n_expected_u."""
    ax = ax or plt.gca()
    ax.errorbar(result["n"], result["mean_u"], yerr=result["stderr_u"], marker="o", **kwargs)
    ax.set_xscale("log")
    ax.set_xlabel("n (best-of-n optimization pressure)")
    ax.set_ylabel("E[U | selected]")
    return ax


def plot_extremal_curve(rho0: float, tau: float, kappa: float, ax=None, mark_threshold=True, **kwargs):
    """Plot the extremal-Goodhart E[U|R=r] curve and mark the threshold r*."""
    from . import extremal
    ax = ax or plt.gca()
    curve = extremal.overoptimization_curve(rho0, tau, kappa)
    ax.plot(curve["r"], curve["mean_u"], label=f"extremal, tau={tau}, kappa={kappa}", **kwargs)
    ax.axvline(tau, color="gray", linestyle=":", label="trusted-region edge (tau)")
    if mark_threshold:
        try:
            r_star = extremal.overoptimization_threshold(rho0, tau, kappa)
            ax.axvline(r_star, color="red", linestyle="--", label=f"overoptimization threshold r*={r_star:.2f}")
        except ValueError:
            pass
    ax.set_xlabel("R (proxy, optimization pressure)")
    ax.set_ylabel("E[U | R = r]  (true objective)")
    ax.legend()
    return ax


def make_overview_figure(save_path: str | None = None):
    """
    Reproduce the qualitative comparison called for in Part 1, question 4:
    plot regressional-only (monotonic), best-of-n simulation (also
    monotonic/asymptoting), and extremal (rises then falls) side by side, so
    the *shape* difference is visible directly -- this is the shape that
    should be compared, qualitatively, against empirical RLHF
    overoptimization curves (proxy reward keeps climbing while gold/true
    reward eventually declines).
    """
    from . import simulate

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    plot_regressional_curve(0.7, ax=axes[0])
    axes[0].set_title("(a) Regressional only\n(monotonic -- no threshold)")
    axes[0].legend()

    result = simulate.best_of_n_expected_u(rho=0.7, n_values=[1, 2, 5, 10, 20, 50, 100, 200], trials=4000)
    plot_best_of_n(result, ax=axes[1])
    axes[1].set_title("(b) Best-of-n sampling\n(monotonic, asymptotes)")

    plot_extremal_curve(rho0=0.7, tau=1.5, kappa=0.6, ax=axes[2])
    axes[2].set_title("(c) + Extremal Goodhart\n(rises then falls -- finite threshold)")

    fig.suptitle("Regressional Goodhart alone never turns over; adding extremal Goodhart does", y=1.03)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
