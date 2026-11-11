"""Compressing a model: simulated quantization and magnitude pruning.

SIMULATED ("fake") quantization
-------------------------------
Each weight is rounded to one of 2**bits - 1 evenly spaced values and then turned
back into an ordinary float. The model then behaves exactly as an n-bit model
would (same rounding errors), so ACCURACY and CALIBRATION are measured
faithfully - but it does NOT run faster or use less memory. Real speed needs a
real low-bit runtime; see `real_int8_speed` in audit.py.

We use symmetric, per-output-channel scaling: every row of a weight matrix gets
its own scale = max|w| / (2**(bits-1) - 1). This is a common default in the
quantization literature (Gholami et al., 2021).

Only weights of linear (fully connected) layers are changed. Biases, layer norms
and embeddings stay in full precision; they are a small share of the parameters.

PRUNING
-------
Unstructured magnitude pruning sets the smallest |w| in each linear layer to 0.
Without special sparse kernels this gives NO speed or memory saving, so pruning
results are reported as calibration effects only.

The maths is written once, in NumPy. PyTorch models (Chronos, TimesFM) are handled
by copying each weight out, compressing it, and copying it back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class CompressionReport:
    method: str  # "none", "quant", "prune"
    level: float  # bits for quant, sparsity for prune
    layers_changed: int
    layers_kept: int
    params_changed: int
    params_total: int
    est_weight_megabytes: float  # memory the linear weights WOULD need at this precision

    def to_dict(self):
        return self.__dict__.copy()


def fake_quantize(w: np.ndarray, bits: int) -> np.ndarray:
    """Round a weight matrix to `bits` bits (per output row) and convert back to float."""
    w = np.asarray(w, dtype=np.float32)
    if bits >= 16:
        return w.astype(np.float16).astype(np.float32)  # FP16 round trip
    qmax = 2 ** (bits - 1) - 1  # e.g. 127 for 8-bit, 7 for 4-bit, 1 for 2-bit
    w2 = w.reshape(w.shape[0], -1)
    scale = np.maximum(np.abs(w2).max(axis=1, keepdims=True), 1e-12) / qmax
    q = np.clip(np.round(w2 / scale), -qmax, qmax)
    return (q * scale).reshape(w.shape).astype(np.float32)


def magnitude_prune(w: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero the `sparsity` fraction of entries with the smallest absolute value."""
    w = np.asarray(w, dtype=np.float32)
    k = int(sparsity * w.size)
    if k <= 0:
        return w
    threshold = np.partition(np.abs(w).ravel(), k - 1)[k - 1]
    return np.where(np.abs(w) <= threshold, 0.0, w).astype(np.float32)


def _linear_weights(model):
    """Yield (name, get, set) for every linear layer, for NumPy or PyTorch models."""
    if hasattr(model, "linear_weights"):  # our NumPy stand-in
        for name, arr in model.linear_weights().items():
            yield name, (lambda a=arr: a), (lambda new, a=arr: a.__setitem__(Ellipsis, new))
        return
    import torch  # only needed for real foundation models
    from torch import nn

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            def get(m=m):
                return m.weight.detach().float().cpu().numpy()

            def put(new, m=m):
                with torch.no_grad():
                    m.weight.copy_(torch.from_numpy(new).to(m.weight.dtype).to(m.weight.device))
            yield name, get, put


def compress_(model, method: str, level: float,
              keep_patterns: list[str] | None = None) -> CompressionReport:
    """Compress `model` IN PLACE. Layers whose name matches any regex in `keep_patterns`
    are left in full precision (e.g. the input and output layers)."""
    keep = [re.compile(p) for p in (keep_patterns or [])]
    changed = kept = p_changed = p_total = 0
    bits_total = 0.0
    for name, get, put in _linear_weights(model):
        w = get()
        n = w.size
        p_total += n
        if method == "none" or any(k.search(name) for k in keep):
            kept += 1
            bits_total += n * 32
            continue
        if method == "quant":
            put(fake_quantize(w, int(level)))
            bits_total += n * min(int(level), 32)
        elif method == "prune":
            put(magnitude_prune(w, float(level)))
            bits_total += n * 32  # unstructured zeros save nothing without sparse storage
        else:
            raise ValueError(f"unknown method {method!r}")
        changed += 1
        p_changed += n
    return CompressionReport(method, level, changed, kept, p_changed, p_total,
                             round(bits_total / 8 / 1e6, 3))
