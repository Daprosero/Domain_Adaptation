"""Smoke tests for the vendored official comparator losses.

Each comparator in `CREDA.comparators` wraps code copied verbatim from the
paper's own repository. These tests do not re-derive any paper's maths; they
hold the adapter contract:

* every registered method returns a finite scalar on seeded random tensors;
* the gradient reaches the target side (f_t, or g_t for SHOT, whose IM loss
  sees predictions only) and is finite and non-zero;
* where a method's discrepancy is zero BY CONSTRUCTION on identical source and
  target batches, it is (CS-DA, LMMD, PFAN's semantic term). JUMBOT, CDAN+E
  and SHOT have no such property and are not asserted on it;
* the PFAN centroid update reads the TARGET moving centroid for the target
  prototype (the deviation from upstream model.py L143, which reads the source
  one).
"""

from __future__ import annotations

import math

import pytest
import torch

from CREDA import comparators
from CREDA.comparators import REGISTRY, build

N = 16  # n_s == n_t: CDAN+E and LMMD require equal batch sizes upstream.
D = 8
C = 3


def _batch(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    f_s = torch.randn(N, D, generator=gen)
    f_t = torch.randn(N, D, generator=gen)
    g_s = torch.randn(N, C, generator=gen)
    g_t = torch.randn(N, C, generator=gen)
    y_s = torch.arange(N) % C
    return f_s, f_t, g_s, g_t, y_s


def test_registry_lists_the_seven_methods():
    assert set(REGISTRY) == {
        "shot", "cdan_e", "dsan_lmmd", "cs_da", "pfan", "jumbot", "daln",
    }


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_loss_is_finite_scalar_and_gradient_reaches_target(name):
    torch.manual_seed(0)
    f_s, f_t, g_s, g_t, y_s = _batch()
    f_t.requires_grad_(True)
    g_t.requires_grad_(True)
    f_s.requires_grad_(True)
    g_s.requires_grad_(True)
    method = build(name, feature_dim=D, num_classes=C)
    extra = {}
    if name == "daln":
        # DALN re-runs the shared head behind its warm-start GRL, whose
        # coefficient is 0 on the first call: advance it so features get gradient.
        extra["head"] = torch.nn.Linear(D, C)
        method.nwd.grl.iter_num = 500
    loss = method(f_s, f_t, g_s, g_t, y_s, **extra)
    assert loss.dim() == 0
    assert math.isfinite(loss.item())
    loss.backward()
    grad = g_t.grad if name == "shot" else f_t.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0


def test_cs_da_is_zero_on_identical_batches():
    f, _, g, _, y = _batch(1)
    method = build("cs_da", feature_dim=D, num_classes=C)
    assert method(f, f.clone(), g, g.clone(), y).abs().item() < 1e-5


def test_cs_da_marginal_flag_matches_paper_default():
    # Paper Eq. (15): lambda * D_CS(marginal) + beta * D_CCS, lambda = beta = 1.
    f_s, f_t, g_s, g_t, y_s = _batch(2)
    with_marg = build("cs_da", feature_dim=D, num_classes=C)
    cond_only = build("cs_da", feature_dim=D, num_classes=C, include_marginal=False)
    assert with_marg.include_marginal is True
    assert not torch.isclose(with_marg(f_s, f_t, g_s, g_t, y_s), cond_only(f_s, f_t, g_s, g_t, y_s))


def test_lmmd_is_zero_when_target_matches_source_and_predicts_its_labels():
    f, _, _, _, y = _batch(3)
    g_t = 100.0 * torch.nn.functional.one_hot(y, C).float()
    method = build("dsan_lmmd", feature_dim=D, num_classes=C)
    assert method(f, f.clone(), g_t, g_t, y).abs().item() < 1e-4


def test_lmmd_refuses_unequal_batches():
    f_s, f_t, g_s, g_t, y_s = _batch(4)
    method = build("dsan_lmmd", feature_dim=D, num_classes=C)
    with pytest.raises(ValueError):
        method(f_s, f_t[:-1], g_s, g_t[:-1], y_s)


def test_pfan_semantic_term_is_zero_on_identical_batches():
    f, _, _, _, y = _batch(5)
    g = 100.0 * torch.nn.functional.one_hot(y, C).float()
    method = build("pfan", feature_dim=D, num_classes=C)
    method.adloss(f, f.clone(), g, y)
    assert method.semantic_loss.abs().item() < 1e-10


def test_pfan_target_centroid_uses_target_moving_centroid():
    f_s, f_t, g_s, g_t, y_s = _batch(6)
    method = build("pfan", feature_dim=D, num_classes=C)
    method.source_moving_centroid.zero_()
    method.target_moving_centroid.fill_(1.0)
    _, _, sc, tc = method.adloss(f_s, f_t, g_t, y_s)
    pred = g_t.argmax(1)
    current_t = torch.zeros(C, D)
    for k in range(C):
        if (pred == k).any():
            current_t[k] = f_t[pred == k].mean(0)
    expected = 0.3 * current_t + 0.7 * torch.ones(C, D)
    assert torch.allclose(tc, expected, atol=1e-6)
    # The upstream L143 form would have produced 0.3 * current_t here.
    assert not torch.allclose(tc, 0.3 * current_t)
    method.update_centroids(sc, tc)
    assert torch.allclose(method.target_moving_centroid, expected, atol=1e-6)


def test_shot_obtain_label_runs_as_a_per_epoch_procedure():
    gen = torch.Generator().manual_seed(8)
    x = torch.randn(30, D, generator=gen)
    labels = torch.randint(0, C, (30,), generator=gen)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, labels), batch_size=10)
    net_f = torch.nn.Identity()
    net_b = torch.nn.Linear(D, D)
    net_c = torch.nn.Linear(D, C)
    method = build("shot", feature_dim=D, num_classes=C)
    pseudo = method.obtain_label(loader, net_f, net_b, net_c)
    assert pseudo.shape == (30,)
    assert pseudo.min() >= 0 and pseudo.max() < C


def test_vendored_files_carry_citation_headers():
    from pathlib import Path

    root = Path(comparators.__file__).parent
    vendored = sorted(root.glob("*/upstream_*.py")) + sorted(root.glob("*/upstream/*.txt"))
    assert len(vendored) >= 7
    for path in vendored:
        head = path.read_text(encoding="utf-8")[:2500]
        assert "Repository:" in head and "Commit:" in head and "Upstream path:" in head, path
