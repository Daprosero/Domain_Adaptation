"""JUMBOT per-dataset hyperparameter presets (Fatras et al., ICML 2021).

Holds the adapter to the paper's own settings (appendix C.1, PDF p.21):
digits eta1=0.1 eta2=0.1 tau=1 eps=0.1; Office-Home eta1=0.01 eta2=0.5
tau=0.5 eps=0.01; VisDA eta1=0.005 eta2=1 tau=0.3 eps=0.01. Office-Home is
the default preset (the paper has no Office-31 setting). Explicit kwargs
override the preset, unknown presets refuse, and the package contract
`method(f_s, f_t, g_s, g_t, y_s, **extra)` is honoured.
"""

from __future__ import annotations

import contextlib
import io

import pytest
import torch

from CREDA.comparators.jumbot.adapter import JUMBOT
from CREDA.comparators.jumbot.upstream_jumbot import Jumbot

N, D, C = 16, 8, 3

PAPER = {
    "office_home": (0.01, 0.5, 0.5, 0.01),
    "visda": (0.005, 1.0, 0.3, 0.01),
    "digits": (0.1, 0.1, 1.0, 0.1),
}


def _values(m: JUMBOT):
    j = m.jumbot
    return (j.eta1, j.eta2, j.tau, j.epsilon)


def _batch(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    f_s = torch.randn(N, D, generator=gen)
    f_t = torch.randn(N, D, generator=gen)
    g_s = torch.randn(N, C, generator=gen)
    g_t = torch.randn(N, C, generator=gen)
    y_s = torch.arange(N) % C
    return f_s, f_t, g_s, g_t, y_s


def test_default_build_uses_office_home_values():
    assert _values(JUMBOT(feature_dim=D, num_classes=C)) == PAPER["office_home"]


@pytest.mark.parametrize("preset", sorted(PAPER))
def test_each_preset_sets_the_four_values(preset):
    assert _values(JUMBOT(feature_dim=D, num_classes=C, preset=preset)) == PAPER[preset]


@pytest.mark.parametrize("key,idx", [("eta1", 0), ("eta2", 1), ("tau", 2), ("epsilon", 3)])
def test_explicit_kwarg_overrides_preset(key, idx):
    m = JUMBOT(feature_dim=D, num_classes=C, preset="visda", **{key: 0.123})
    expected = list(PAPER["visda"])
    expected[idx] = 0.123
    assert _values(m) == tuple(expected)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="office31"):
        JUMBOT(feature_dim=D, num_classes=C, preset="office31")


def test_extra_kwargs_are_accepted_and_ignored():
    m = JUMBOT(feature_dim=D, num_classes=C)
    batch = _batch(1)
    plain = m(*batch)
    with_extra = m(*batch, indices=torch.arange(N), step=7)
    assert torch.equal(plain, with_extra)


def test_loss_with_explicit_values_matches_upstream_direct_call():
    # Previous behaviour: the adapter forwarded the four values to
    # Jumbot.__init__ and returned da_loss(g_t, f_s, f_t, y_s) unchanged.
    hp = dict(eta1=0.001, eta2=0.0001, tau=1.0, epsilon=0.1)
    f_s, f_t, g_s, g_t, y_s = _batch(2)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = Jumbot(None, None, C, **hp).da_loss(g_t, f_s, f_t, y_s)
    got = JUMBOT(feature_dim=D, num_classes=C, **hp)(f_s, f_t, g_s, g_t, y_s)
    assert torch.equal(got, reference)
