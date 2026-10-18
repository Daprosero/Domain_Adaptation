"""Tests for the DALN comparator (Chen et al., CVPR 2022, arXiv:2204.03838).

The adapter wraps the vendored xiaoachen98/DALN `nwd.py` + `grl.py`. These tests
hold the adapter contract and the official semantics:

* the value is the official transfer term `-NWD` (README: `discrepancy_loss =
  -discrepancy(f)`), checked against a hand-written NWD built from singular
  values: (||softmax(h(f_s))||_* - ||softmax(h(f_t))||_*) / n_t;
* the critic is the caller's shared head, re-run on GRL(features); the adapter
  owns no trainable parameter;
* the head receives the plain gradient (it maximises NWD) and the features the
  reversed one, scaled by the warm-start GRL coefficient (0 at the first call).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from CREDA.comparators.daln import DALN

N = 12
D = 8
C = 5


def _setup(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    f_s = torch.randn(N, D, generator=gen)
    f_t = torch.randn(N, D, generator=gen) + 0.5
    y_s = torch.arange(N) % C
    torch.manual_seed(seed)
    head = nn.Linear(D, C)
    with torch.no_grad():
        g_s, g_t = head(f_s), head(f_t)
    return f_s, f_t, g_s, g_t, y_s, head


def _hand_nwd(head, f_s, f_t):
    """Paper Eq. (7)-(8) as the official code realises it: batch nuclear norms of
    the softmax prediction matrices, divided by the target batch size."""
    p_s = torch.softmax(head(f_s), dim=1)
    p_t = torch.softmax(head(f_t), dim=1)
    nuc_s = torch.linalg.svdvals(p_s).sum()
    nuc_t = torch.linalg.svdvals(p_t).sum()
    return (nuc_s - nuc_t) / f_t.shape[0]


def test_returns_finite_zero_dim_scalar():
    f_s, f_t, g_s, g_t, y_s, head = _setup()
    method = DALN(D, C)
    loss = method(f_s, f_t, g_s, g_t, y_s, head=head)
    assert loss.dim() == 0
    assert math.isfinite(loss.item())


def test_value_is_negative_hand_written_nwd():
    f_s, f_t, g_s, g_t, y_s, head = _setup(1)
    method = DALN(D, C)
    loss = method(f_s, f_t, g_s, g_t, y_s, head=head)
    with torch.no_grad():
        expected = -_hand_nwd(head, f_s, f_t)
    assert torch.allclose(loss, expected, atol=1e-6)
    assert expected.abs().item() > 1e-4  # not a vacuous zero-vs-zero match


def test_owns_no_trainable_parameters_and_does_not_capture_the_head():
    f_s, f_t, g_s, g_t, y_s, head = _setup()
    method = DALN(D, C)
    method(f_s, f_t, g_s, g_t, y_s, head=head)
    assert list(method.parameters()) == []
    assert all(not isinstance(m, nn.Linear) for m in method.modules())


def test_head_gets_plain_gradient_and_features_get_reversed_scaled_gradient():
    f_s, f_t, g_s, g_t, y_s, head = _setup(2)
    method = DALN(D, C)
    method.nwd.grl.iter_num = 1000  # past warm start: coeff = 2/(1+e^-1) - 1
    coeff = 2.0 / (1.0 + math.exp(-1.0)) - 1.0

    fs, ft = f_s.clone().requires_grad_(True), f_t.clone().requires_grad_(True)
    head.zero_grad()
    method(fs, ft, g_s, g_t, y_s, head=head).backward()
    head_grad = head.weight.grad.clone()

    # Reference without the GRL: d(-NWD)/d(f) and d(-NWD)/d(W).
    rs, rt = f_s.clone().requires_grad_(True), f_t.clone().requires_grad_(True)
    head.zero_grad()
    (-_hand_nwd(head, rs, rt)).backward()

    for got, ref in ((fs.grad, rs.grad), (ft.grad, rt.grad)):
        assert got is not None and torch.isfinite(got).all()
        assert got.abs().sum().item() > 0
        assert torch.allclose(got, -coeff * ref, atol=1e-6)
    assert torch.allclose(head_grad, head.weight.grad, atol=1e-6)
    assert head_grad.abs().sum().item() > 0


def test_warm_start_first_call_blocks_feature_gradient_but_trains_the_head():
    f_s, f_t, g_s, g_t, y_s, head = _setup(3)
    method = DALN(D, C)
    ft = f_t.clone().requires_grad_(True)
    head.zero_grad()
    method(f_s, ft, g_s, g_t, y_s, head=head).backward()
    assert ft.grad.abs().sum().item() == 0.0  # upstream lo=0 at iter 0
    assert head.weight.grad.abs().sum().item() > 0
    assert method.nwd.grl.iter_num == 1  # auto_step=True: one step per call


def test_requires_the_head_and_equal_batches():
    f_s, f_t, g_s, g_t, y_s, head = _setup()
    method = DALN(D, C)
    with pytest.raises(TypeError):
        method(f_s, f_t, g_s, g_t, y_s)
    with pytest.raises(ValueError):
        method(f_s, f_t[:-1], g_s, g_t[:-1], y_s, head=head)
