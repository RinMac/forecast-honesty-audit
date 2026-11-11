"""Tests for compression and the stand-in model (NumPy only, no downloads)."""

import numpy as np
import pytest

from honesty_audit.compress import compress_, fake_quantize, magnitude_prune
from honesty_audit.data import make_synthetic_load
from honesty_audit.models import TinyQuantileNet


def test_fake_quantize_number_of_levels_and_error():
    w = np.random.default_rng(0).normal(size=(64, 128)).astype(np.float32)
    for bits in (8, 4, 3, 2):
        q = fake_quantize(w, bits)
        most = max(len(np.unique(row)) for row in q)
        assert most <= 2 ** bits - 1  # symmetric rounding: 2**bits - 1 possible values per row
    err = {b: np.abs(fake_quantize(w, b) - w).mean() for b in (16, 8, 4, 2)}
    assert err[16] < err[8] < err[4] < err[2]


def test_two_bit_means_three_values_per_row():
    w = np.array([[-1.0, -0.4, 0.1, 0.6, 1.0]], np.float32)
    assert sorted(set(fake_quantize(w, 2)[0].tolist())) == [-1.0, 0.0, 1.0]


def test_prune_sparsity():
    w = np.random.default_rng(1).normal(size=(100, 100))
    z = (magnitude_prune(w, 0.5) == 0).mean()
    assert 0.49 <= z <= 0.51


def test_compress_keeps_io_layers_and_never_touches_the_original():
    m = TinyQuantileNet(48, 12)
    before = {k: v.copy() for k, v in m.W.items()}
    c = m.clone()
    rep = compress_(c.module, "quant", 4, [r"input_layer", r"output_layer"])
    assert rep.layers_kept == 2 and rep.layers_changed == 1
    assert np.array_equal(c.W["input_layer"], before["input_layer"])
    assert not np.array_equal(c.W["hidden_layer"], before["hidden_layer"])
    for k in before:  # the original model is untouched
        assert np.array_equal(m.W[k], before[k])


def test_stand_in_learns_sensible_quantiles():
    series = [s.to_numpy() for s in make_synthetic_load(n_series=3, n_days=60).values()]
    m = TinyQuantileNet(96, 24, seed=0).fit(series, epochs=800, log=None)
    ctx = [s[-24 - 96:-24] for s in series]
    truth = np.array([s[-24:] for s in series])
    q = m.predict(ctx, 24)
    assert q.shape == (3, 24, 9)
    inside = ((truth >= q[..., 0]) & (truth <= q[..., 8])).mean()
    assert inside > 0.4  # trained net brackets the truth reasonably often
    assert (np.diff(q, axis=-1) >= -1e-6).mean() > 0.85  # quantiles mostly in order


def test_torch_linear_layers_are_quantized_in_place():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    before = net[0].weight.detach().clone()
    rep = compress_(net, "quant", 2)
    assert rep.layers_changed == 2
    assert len(torch.unique(net[0].weight[0])) <= 3
    assert not torch.equal(net[0].weight, before)
