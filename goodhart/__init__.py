"""
goodhart: A mathematical model of Goodhart's Law / reward overoptimization.

Modules
-------
model      : closed-form derivations (regressional Goodhart, Gaussian case)
simulate   : Monte Carlo experiments (best-of-n selection, optimization pressure)
extremal   : extension to extremal Goodhart (nonstationary proxy-truth relation)
plots      : plotting helpers for the overoptimization curves

Submodules are imported lazily by whoever needs them (e.g. `from goodhart
import model`) rather than eagerly here, so that `python -m goodhart.model`
etc. doesn't trigger the "module found in sys.modules before execution"
RuntimeWarning that comes from a package eagerly re-importing the very
submodule being run as `__main__`.
"""
