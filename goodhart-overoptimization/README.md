# Goodhart's Law: a mathematical model of reward overoptimization

A derivation + simulation scaffold, not a black-box demo: `sympy` checks the
algebra step by step, and the simulation confirms the closed-form result
numerically.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## What's here

| File | Contents |
|---|---|
| `goodhart/model.py` | Symbolic derivation of `E[U\|R=r]` under the joint-Gaussian model (regressional Goodhart), by completing the square in the bivariate-normal exponent — sympy verifies each algebraic step. Also proves this case alone is monotonic (no finite overoptimization threshold). |
| `goodhart/simulate.py` | Monte Carlo best-of-*n* selection: draw *n* samples of `(U, R)`, keep the one with highest `R`, track `E[U]` as *n* (optimization pressure) grows. Confirms the analytic monotonicity result numerically. |
| `goodhart/extremal.py` | Extends the model with a second Goodhart mechanism (extremal): the effective correlation between `U` and `R` decays outside a "trusted region" `[-τ, τ]`. This *does* produce a finite overoptimization threshold `r* = 1/κ`, solved symbolically. |
| `goodhart/plots.py` | Reproduces the three curves (regressional / best-of-*n* / extremal) side by side, so the qualitative shape difference is visible. |
| `figures/overoptimization_overview.png` | Pre-generated output of `plots.make_overview_figure()`. |
| `tests/test_goodhart.py` | 9 unit tests. |

## How to run it

```bash
# Print the symbolic derivation + the monotonicity proof
python -m goodhart.model

# Print the extremal-Goodhart threshold r* = 1/kappa
python -m goodhart.extremal

# Regenerate the comparison figure
python -c "from goodhart import plots; plots.make_overview_figure(save_path='figures/overoptimization_overview.png')"

# Run the tests
pytest tests/ -v
```

## Key results

- `E[U | R=r] = ρr` (standardized case) under the pure joint-Gaussian
  model — linear, hence monotonic, hence **no overoptimization threshold
  exists from regressional Goodhart alone**. This is a deliberately
  negative result: it's *why* overoptimization needs a second mechanism.
- Adding extremal Goodhart (proxy-truth correlation decays outside a
  trusted region) yields a genuine interior maximum at `r* = 1/κ`, matching
  the qualitative "rises then falls" shape reported empirically for RLHF
  reward-model overoptimization.

## Still open (do this part yourself)

The project's question 4 asks you to compare your derived curve's *shape*
against the empirical curves in Gao, Schulman & Hilton (2022). I don't have
their raw numbers here (no network access when this was built), so
`plots.py` only produces the shape from the model — it doesn't overlay
their data. Pull the numbers or digitize their Figure 1, then eyeball or
curve-fit against `extremal.overoptimization_curve()`'s output, and add a
paragraph on where it matches/diverges (e.g. their curve keeps rising
longer before turning, or falls off faster/slower than the exponential
decay assumed here).

## References

- Manheim & Garrabrant, *Categorizing Variants of Goodhart's Law* (2018).
- Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* (2022).
- Skalse et al., *Defining and Characterizing Reward Hacking* (2022).

## License

MIT — see `LICENSE`.
