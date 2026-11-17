"""`latent.py`'s own sigma consumer, spied by name.

`represent` threads `config.KERNEL_SIGMA` into the attention machinery
through `Arm.bags_of`. A value-based check recomputes its OWN expectation from
the same `config.KERNEL_SIGMA`, so a mutation that moved every call site the
same wrong way -- a shared, scaled bandwidth -- would move both sides of that
comparison identically and pass unnoticed. This spies on the primitive
instead, by name, so a mutation to the call site inside `latent.py` is caught
there rather than by an accident of how a fixture happened to be built.

The digit report's other four readers (`geometry`, `correspondence`,
`attention_spread`, `bag_pairs`) were retired with the image benchmark;
`represent` is the one the toyset still calls.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import MIL_CREDA.attention as attention_module
from MIL_CREDA_Benchmark import bags, config, latent, wiring


class _Encoder(nn.Module):
    """A stand-in for the pretrained resnet18: something with an `output_dim`."""

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


def _bagset(domain: str, seed: int, classes: int = 3, bags_per_class: int = 4,
           m: int = 6) -> bags.BagSet:
    generator = torch.Generator().manual_seed(seed)
    n_bags = classes * bags_per_class
    images = torch.randn(n_bags * m, 3, 8, 8, generator=generator)
    members = torch.arange(images.shape[0]).reshape(n_bags, m)
    labels = torch.tensor([c for c in range(classes) for _ in range(bags_per_class)])
    idx = torch.arange(n_bags)
    return bags.BagSet(domain=domain, images=images, members=members, labels=labels,
                       train_idx=idx, valid_idx=idx, eval_idx=idx, manifest={})


def _model(monkeypatch, arm_id: str = "G"):
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder)
    source = _bagset("M", 1)
    target = _bagset("U", 2)
    pool_source = wiring.Pool(source.images, source.members[source.train_idx],
                              source.labels[source.train_idx])
    pool_target = wiring.Pool(target.images, target.members[target.train_idx],
                              target.labels[target.train_idx])
    torch.manual_seed(3)
    model = wiring.build(arm_id, 3, pool_source, pool_target)
    return model, source, target


def test_represent_uses_the_declared_sigma(monkeypatch) -> None:
    """The sigma-consuming primitive `latent.represent` reaches is spied by
    name and must receive exactly `config.KERNEL_SIGMA`.

    Reachable red: change `represent`'s own `sigma = config.KERNEL_SIGMA` line
    to read `config.KERNEL_SIGMA * 3` (or any other scaled/stale value) -- the
    spy call below then records a sigma different from the declared constant
    and the final assertion fails.
    """
    seen: list[tuple[str, object]] = []

    real_relevance_logits = attention_module.relevance_logits

    def spy_relevance_logits(H, V_R, b_R, v_R, gamma, sigma):
        seen.append(("relevance_logits", sigma))
        return real_relevance_logits(H, V_R, b_R, v_R, gamma, sigma)

    # `Arm.select`/`Arm.weights_for` (both reached through `bags_of`, which is
    # how `represent` threads sigma into the attention machinery) call
    # `relevance_logits` via `wiring`'s own top-level import, so the spy is
    # placed there -- patching the source module would not affect a name
    # already bound at `wiring` import time.
    monkeypatch.setattr(wiring, "relevance_logits", spy_relevance_logits)

    model, source, target = _model(monkeypatch, "G")
    device = torch.device("cpu")

    latent.represent(model, source, source.eval_idx, device, "source")
    latent.represent(model, target, target.eval_idx, device, "target")

    assert seen, "no sigma consumer was ever called from latent.py"
    consumers = {name for name, _ in seen}
    assert consumers == {"relevance_logits"}, consumers
    for name, sigma in seen:
        assert sigma == config.KERNEL_SIGMA, (
            f"{name} was called from latent.py with sigma={sigma!r}, not the "
            f"declared constant {config.KERNEL_SIGMA!r}"
        )


def test_represent_refuses_a_call_with_no_domain(monkeypatch) -> None:
    """Spec `per-domain-encoders` -- "No embedding without a declared domain":
    a caller that cannot state which domain it holds MUST be refused rather
    than defaulted to either encoder. `domain` carries no default anywhere in
    the signature, so omitting it is a `TypeError` from Python itself,
    exactly the shape `config.KERNEL_SIGMA`'s own declared rule already uses
    for the same kind of quantity ("every caller passes this explicitly, with
    no default anywhere in the call chain").

    Reachable red: give the parameter a default of `"source"` -- the call
    below would then silently succeed.
    """
    model, source, target = _model(monkeypatch, "G")
    device = torch.device("cpu")

    with pytest.raises(TypeError):
        latent.represent(model, source, source.eval_idx, device)


def test_represent_refuses_an_unknown_domain(monkeypatch) -> None:
    """The other half of the same requirement: a `domain` that is neither
    `"source"` nor `"target"` is refused by name, in `mechanism_weights`'s own
    idiom (`wiring.py`), naming `wiring.SIDES` so a caller sees exactly what
    was expected.

    Reachable red: compare `domain` against a hardcoded pair of strings
    instead of `wiring.SIDES`, or swallow the mismatch instead of raising.
    """
    model, source, target = _model(monkeypatch, "G")
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="unknown domain"):
        latent.represent(model, source, source.eval_idx, device, "both")
