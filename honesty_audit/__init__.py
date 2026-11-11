"""honesty_audit: does a compressed forecasting model still keep its confidence promises?

The package measures how the prediction intervals ("80% ranges") of a
time-series foundation model change as the model is compressed, and how much a
conformal recalibration step restores them.

Modules
-------
data       load series and cut them into calibration / test windows
models     wrappers around forecasting models (Chronos-Bolt, TimesFM-2.5, stand-in)
compress   simulated weight quantization and magnitude pruning
conformal  split CQR, weighted CQR and adaptive conformal inference (ACI)
metrics    coverage, width, interval score, bootstrap confidence intervals
audit      runs the full experiment and applies the pre-registered pass/fail rule
report     turns results into an interactive "honesty report" HTML page
"""

__version__ = "0.1.0"
