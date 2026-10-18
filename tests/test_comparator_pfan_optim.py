"""PFAN's discriminator has its own optimiser (spec S5).

PFAN's adversarial term is G = -D (model.py L151-152), not a gradient-reversal
layer, and upstream trains D with its own optimiser on D_loss + Dregloss
(`adoptimize`, model.py L74-87: momentum 0.9, biases at 2x lr, 0.0005 * mean
l2_loss of the weights). These tests pin the caller API that keeps the two
updates apart: the generator backward never reaches D, the discriminator step
never reaches the features, and the forward value is the one adloss computes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from CREDA.comparators import build

D, C = 8, 3


def _batch(seed, n=6):
    gen = torch.Generator().manual_seed(seed)
    f_s = torch.randn(n, D, generator=gen)
    f_t = torch.randn(n, D, generator=gen) + 0.5
    g_s = torch.randn(n, C, generator=gen)
    g_t = torch.randn(n, C, generator=gen)
    y_s = torch.randint(0, C, (n,), generator=gen)
    return f_s, f_t, g_s, g_t, y_s


def _own_bce(method, f_s, f_t):
    # model.py L149-151, x0.1 at L157: D's own objective (target=1, source=0).
    s_logits, _ = method.D(f_s)
    t_logits, _ = method.D(f_t)
    bce = F.binary_cross_entropy_with_logits(t_logits, torch.ones_like(t_logits)) + \
        F.binary_cross_entropy_with_logits(s_logits, torch.zeros_like(s_logits))
    return 0.1 * bce


def _dregloss(method):
    # model.py L83: 0.0005 * reduce_mean([l2_loss(w) for D weights]), l2_loss = sum(w^2)/2.
    ws = [method.D.weights, method.D.weights2, method.D.weights3]
    return 0.0005 * torch.stack([(w ** 2).sum() / 2 for w in ws]).mean()


def test_generator_and_discriminator_parameter_sets_are_disjoint_and_cover_all():
    method = build("pfan", feature_dim=D, num_classes=C)
    gen_ids = {id(p) for p in method.generator_parameters()}
    disc_ids = {id(p) for p in method.discriminator_parameters()}
    all_ids = {id(p) for p in method.parameters()}
    assert gen_ids.isdisjoint(disc_ids)
    assert gen_ids | disc_ids == all_ids
    assert disc_ids == {id(p) for p in method.D.parameters()}


def test_forward_value_is_unchanged():
    f_s, f_t, g_s, g_t, y_s = _batch(1)
    method = build("pfan", feature_dim=D, num_classes=C)
    method.source_moving_centroid.normal_()
    method.target_moving_centroid.normal_()
    torch.manual_seed(123)
    G_ref, D_ref, sc_ref, tc_ref = method.adloss(f_s, f_t, g_t, y_s)
    expected = (G_ref + method.semantic_loss).item()
    torch.manual_seed(123)
    out = method(f_s, f_t, g_s, g_t, y_s)
    assert out.dim() == 0
    assert out.item() == expected
    assert method.last_d_loss.item() == D_ref.item()
    assert torch.equal(method.last_centroids[0], sc_ref)
    assert torch.equal(method.last_centroids[1], tc_ref)


def test_generator_backward_reaches_features_and_never_d():
    f_s, f_t, g_s, g_t, y_s = _batch(2)
    f_s.requires_grad_(True)
    f_t.requires_grad_(True)
    method = build("pfan", feature_dim=D, num_classes=C)
    out = method(f_s, f_t, g_s, g_t, y_s)
    out.backward()
    assert f_s.grad is not None and f_s.grad.abs().sum().item() > 0.0
    assert f_t.grad is not None and f_t.grad.abs().sum().item() > 0.0
    for p in method.discriminator_parameters():
        assert p.grad is None


def test_generator_gradient_is_the_negated_adversarial_gradient_on_features():
    # G = -D on phi: the feature gradient of the returned loss's adversarial part
    # is exactly minus the feature gradient of D's own loss (same dropout masks).
    f_s, f_t, g_s, g_t, y_s = _batch(3)
    method = build("pfan", feature_dim=D, num_classes=C)
    method.source_moving_centroid.zero_()
    method.target_moving_centroid.zero_()
    fs1 = f_s.clone().requires_grad_(True)
    torch.manual_seed(7)
    method(fs1, f_t, g_s, g_t, y_s)
    G_part = method.last_g_loss  # adversarial part of the returned value
    (g_gen,) = torch.autograd.grad(G_part, fs1)
    fs2 = f_s.clone().requires_grad_(True)
    torch.manual_seed(7)
    (g_own,) = torch.autograd.grad(_own_bce(method, fs2, f_t), fs2)
    assert torch.allclose(g_gen, -g_own, atol=1e-7)
    assert g_gen.abs().sum().item() > 0.0


def test_discriminator_loss_gradient_is_the_descent_direction_of_d_own_bce():
    f_s, f_t, g_s, g_t, y_s = _batch(4)
    method = build("pfan", feature_dim=D, num_classes=C)
    method(f_s, f_t, g_s, g_t, y_s)
    params = list(method.discriminator_parameters())
    torch.manual_seed(11)
    got = torch.autograd.grad(method.discriminator_loss(), params)
    torch.manual_seed(11)
    want = torch.autograd.grad(_own_bce(method, f_s, f_t) + _dregloss(method), params)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-7)
    # And not the ascent direction the main optimiser would have taken from the
    # returned (G = -D) loss before the fix.
    assert sum((a * b).sum() for a, b in zip(got, want)).item() > 0.0


def test_discriminator_step_trains_d_only_and_leaves_phi_grads_untouched():
    f_s, f_t, g_s, g_t, y_s = _batch(5)
    f_s.requires_grad_(True)
    f_t.requires_grad_(True)
    method = build("pfan", feature_dim=D, num_classes=C)
    out = method(f_s, f_t, g_s, g_t, y_s)
    out.backward()
    phi_grads = (f_s.grad.clone(), f_t.grad.clone())
    opt = torch.optim.SGD(method.discriminator_param_groups(lr=0.01, momentum=0.9))
    before = [p.detach().clone() for p in method.discriminator_parameters()]
    method.discriminator_step(opt)  # last forward's features
    method.discriminator_step(opt, f_s, f_t)  # explicit features that require grad
    assert torch.equal(f_s.grad, phi_grads[0])
    assert torch.equal(f_t.grad, phi_grads[1])
    after = list(method.discriminator_parameters())
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, after))


def test_discriminator_steps_decrease_d_own_loss():
    f_s, f_t, g_s, g_t, y_s = _batch(6, n=32)
    method = build("pfan", feature_dim=D, num_classes=C)
    method(f_s, f_t, g_s, g_t, y_s)
    opt = torch.optim.SGD(method.discriminator_param_groups(lr=0.05, momentum=0.9))
    torch.manual_seed(0)
    start = sum(_own_bce(method, f_s, f_t).item() for _ in range(50)) / 50
    torch.manual_seed(1)
    for _ in range(40):
        method.discriminator_step(opt)
    torch.manual_seed(0)
    end = sum(_own_bce(method, f_s, f_t).item() for _ in range(50)) / 50
    assert end < start


def test_discriminator_param_groups_follow_adoptimize():
    method = build("pfan", feature_dim=D, num_classes=C)
    groups = method.discriminator_param_groups(lr=0.01, momentum=0.9)
    by_lr = {g["lr"]: {id(p) for p in g["params"]} for g in groups}
    assert by_lr[0.01] == {id(method.D.weights), id(method.D.weights2), id(method.D.weights3)}
    assert by_lr[0.02] == {id(method.D.biases), id(method.D.biases2), id(method.D.biases3)}
    assert all(g["momentum"] == 0.9 for g in groups)
    # Regularisation lives in the loss (Dregloss), not in optimiser weight decay.
    assert all(g.get("weight_decay", 0.0) == 0.0 for g in groups)
