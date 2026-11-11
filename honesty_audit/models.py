"""Model wrappers: every forecaster answers `predict(contexts, horizon) -> [n, H, Q] quantiles`.

Three wrappers:
  ChronosBoltForecaster  Amazon's Chronos-Bolt (fixed quantiles 0.1 ... 0.9)   needs PyTorch
  TimesFM25Forecaster    Google's TimesFM-2.5 (mean + quantiles 0.1 ... 0.9)   needs PyTorch
  TinyQuantileNet        a small NumPy network trained in seconds, used ONLY to test the
                         pipeline offline on any laptop. Its numbers are not research results.

Each wrapper exposes `.module` (what compress.py changes) and `.clone()` (a full,
independent copy, so the original model is never touched).
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

DEFAULT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Forecaster:
    name: str = "forecaster"
    quantile_levels: np.ndarray = np.array(DEFAULT_LEVELS)
    batch_size: int = 64

    @property
    def module(self):
        raise NotImplementedError

    def set_module(self, m) -> None:
        """Swap in a different module (used for the real int8 speed check)."""
        raise NotImplementedError

    def _predict_batch(self, contexts: Sequence[np.ndarray], horizon: int) -> np.ndarray:
        raise NotImplementedError

    def predict(self, contexts: Sequence[np.ndarray], horizon: int) -> np.ndarray:
        out = []
        for i in range(0, len(contexts), self.batch_size):
            out.append(self._predict_batch(contexts[i:i + self.batch_size], horizon))
        return np.concatenate(out, axis=0).astype(float)

    def clone(self) -> "Forecaster":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------- Chronos-Bolt


def _hf_revision(model_id: str):
    """The commit hash of the model files in the local Hugging Face cache (None if unknown)."""
    try:
        import os
        from huggingface_hub import snapshot_download
        return os.path.basename(snapshot_download(model_id, local_files_only=True))
    except Exception:
        return None


class ChronosBoltForecaster(Forecaster):
    """Chronos-Bolt via the `chronos-forecasting` package (pip install chronos-forecasting).

    model_id examples: amazon/chronos-bolt-tiny, -mini, -small, -base.
    float32 is used so that compression is the ONLY source of extra rounding.
    """

    def __init__(self, model_id: str = "amazon/chronos-bolt-small", device: str = "cpu"):
        from chronos import BaseChronosPipeline

        self.name = model_id.split("/")[-1]
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
        # force float32 (works on every transformers version, unlike the dtype argument)
        self.pipeline.model = self.pipeline.model.float().eval()
        self.quantile_levels = np.array(DEFAULT_LEVELS)
        self.revision = _hf_revision(model_id)  # exact version of the downloaded weights

    @property
    def module(self):
        return self.pipeline.model

    def set_module(self, m):
        self.pipeline.model = m

    def _predict_batch(self, contexts, horizon):
        import torch

        ctx = [torch.tensor(np.asarray(c), dtype=torch.float32) for c in contexts]
        with torch.no_grad():
            quantiles, _mean = self.pipeline.predict_quantiles(
                ctx, prediction_length=horizon,
                quantile_levels=[float(q) for q in self.quantile_levels])
        return quantiles.float().cpu().numpy()  # [n, H, Q]


# --------------------------------------------------------------------------- TimesFM-2.5


def _find_torch_module(obj):
    from torch import nn

    if isinstance(obj, nn.Module):
        return obj
    for attr in ("model", "_model", "module", "net"):
        sub = getattr(obj, attr, None)
        if isinstance(sub, nn.Module):
            return sub
    raise AttributeError("could not find the torch nn.Module inside this model object")


class TimesFM25Forecaster(Forecaster):
    """TimesFM-2.5 (200M) via the `timesfm` package.

    Follows the usage in the TimesFM GitHub README: forecast() returns
    (point, quantiles) with quantiles shaped [n, H, 10] = mean, then the 0.1 ... 0.9
    quantiles. CHECK this against the README of the version you install.

    fix_quantile_crossing is OFF by default: the audit wants to SEE crossings
    caused by compression (we count them, then sort).
    """

    def __init__(self, model_id: str = "google/timesfm-2.5-200m-pytorch",
                 max_context: int = 1024, max_horizon: int = 256,
                 fix_quantile_crossing: bool = False):
        import timesfm

        self.name = "timesfm-2.5-200m"
        self.tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        self.tfm.compile(timesfm.ForecastConfig(
            max_context=max_context, max_horizon=max_horizon, normalize_inputs=True,
            use_continuous_quantile_head=True, fix_quantile_crossing=fix_quantile_crossing))
        self.quantile_levels = np.array(DEFAULT_LEVELS)

    @property
    def module(self):
        return _find_torch_module(self.tfm)

    def _predict_batch(self, contexts, horizon):
        _point, q = self.tfm.forecast(horizon=horizon, inputs=[np.asarray(c) for c in contexts])
        return np.asarray(q)[:, :horizon, 1:]  # drop the mean channel -> 9 quantiles


# --------------------------------------------------------------------------- stand-in


class TinyQuantileNet(Forecaster):
    """A small NumPy quantile network (3 linear layers, ReLU). FOR TESTING THE PIPELINE ONLY.

    It standardises each context (subtract its mean, divide by its spread), so it can
    forecast series of any size. Trained with the pinball (quantile) loss and Adam.
    Layer names mirror the real models so the "keep input/output layers" option works.
    """

    def __init__(self, context_len: int, horizon: int, hidden: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.name = "tiny-quantile-net (stand-in)"
        self.context_len, self.horizon = context_len, horizon
        self.quantile_levels = np.array(DEFAULT_LEVELS)
        nq = len(self.quantile_levels)
        dims = [("input_layer", context_len, hidden), ("hidden_layer", hidden, hidden),
                ("output_layer", hidden, horizon * nq)]
        # weights stored as [out, in], like PyTorch, so per-row quantization means per-output-channel
        self.W = {n: (rng.normal(0, np.sqrt(2 / i), (o, i))).astype(np.float32) for n, i, o in dims}
        self.b = {n: np.zeros(o, np.float32) for n, _, o in dims}
        self.order = [n for n, _, _ in dims]

    # compress.py looks for this method on NumPy models
    def linear_weights(self):
        return self.W

    @property
    def module(self):
        return self

    def _forward(self, x, keep=False):
        acts = [x]
        h = x
        for k, n in enumerate(self.order):
            z = h @ self.W[n].T + self.b[n]
            h = np.maximum(z, 0) if k < len(self.order) - 1 else z
            acts.append(h)
        return (h, acts) if keep else h

    def fit(self, series: list[np.ndarray], epochs: int = 800, lr: float = 2e-3,
            batch: int = 256, seed: int = 0, log=print) -> "TinyQuantileNet":
        rng = np.random.default_rng(seed)
        L, H, q = self.context_len, self.horizon, self.quantile_levels.astype(np.float32)
        params = [(self.W, n) for n in self.order] + [(self.b, n) for n in self.order]
        m = [np.zeros_like(d[n]) for d, n in params]
        v = [np.zeros_like(d[n]) for d, n in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(1, epochs + 1):
            xs, ys = [], []
            for _ in range(batch):
                s = series[rng.integers(len(series))]
                i = rng.integers(L, len(s) - H)
                ctx, tgt = s[i - L:i], s[i:i + H]
                mu, sd = ctx.mean(), ctx.std() + 1e-8
                xs.append((ctx - mu) / sd)
                ys.append((tgt - mu) / sd)
            x, y = np.array(xs, np.float32), np.array(ys, np.float32)
            out, acts = self._forward(x, keep=True)
            pred = out.reshape(batch, H, len(q))
            d = y[..., None] - pred
            # gradient of mean pinball loss with respect to the prediction
            g = np.where(d > 0, -q, 1 - q).astype(np.float32) / d.size
            g = g.reshape(batch, -1)
            grads_W, grads_b = {}, {}
            for k in reversed(range(len(self.order))):
                n = self.order[k]
                grads_W[n] = g.T @ acts[k]
                grads_b[n] = g.sum(axis=0)
                if k > 0:
                    g = (g @ self.W[n]) * (acts[k] > 0)
            for j, (dct, n) in enumerate(params):
                gr = grads_W[n] if dct is self.W else grads_b[n]
                m[j] = b1 * m[j] + (1 - b1) * gr
                v[j] = b2 * v[j] + (1 - b2) * gr ** 2
                mh, vh = m[j] / (1 - b1 ** step), v[j] / (1 - b2 ** step)
                dct[n] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
            if step % 100 == 0 and log:
                loss = np.maximum(q * d, (q - 1) * d).mean()
                log(f"[stand-in] step {step}: pinball loss {loss:.4f}")
        return self

    def _predict_batch(self, contexts, horizon):
        assert horizon == self.horizon, "stand-in model was trained for one horizon only"
        ctx = np.array([c[-self.context_len:] for c in contexts], np.float32)
        mu = ctx.mean(axis=1, keepdims=True)
        sd = ctx.std(axis=1, keepdims=True) + 1e-8
        out = self._forward((ctx - mu) / sd).reshape(len(ctx), horizon, -1)
        return out * sd[:, :, None] + mu[:, :, None]


def load_model(name: str, context_len: int = 512, horizon: int = 24, **kw) -> Forecaster:
    """Shortcut used by the command line: 'chronos-bolt-small', 'timesfm-2.5', 'stand-in'."""
    if name.startswith("chronos-bolt"):
        return ChronosBoltForecaster(f"amazon/{name}", **kw)
    if name.startswith("timesfm"):
        return TimesFM25Forecaster(**kw)
    if name == "stand-in":
        return TinyQuantileNet(context_len, horizon)
    raise ValueError(f"unknown model {name!r}")
