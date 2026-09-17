"""`latent.py`'s own sigma consumers, spied by name.

Five functions compare two trained arms' spaces against each other and each
threads `config.KERNEL_SIGMA` to a kernel primitive or to `Arm.bags_of`:
`geometry`, `correspondence`, `represent`, `attention_spread`, `bag_pairs`.
Every value-based check elsewhere in this suite (`test_latent_geometry.py`,
for instance) recomputes its OWN expectation from the same `config.KERNEL_SIGMA`,
so a mutation that moved every call site the same wrong way -- a shared,
scaled bandwidth -- would move both sides of that comparison identically and
pass unnoticed. This spies on the primitives instead, by name, so a mutation
to any one call site inside `latent.py` is caught there rather than by an
accident of how a fixture happened to be built.

`geometry` and `correspondence`/`bag_pairs` import `MIL_CREDA.bag_kernel`'s
functions locally, inside their own bodies, rather than at module level --
so the spy patches the SOURCE module's attributes directly: the `from ...
import ...` those functions execute on every call re-resolves against
whatever the source module currently holds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import MIL_CREDA.attention as attention_module
import MIL_CREDA.bag_kernel as bag_kernel_module
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


def test_geometry_correspondence_represent_attention_spread_and_bag_pairs_use_the_declared_sigma(
        monkeypatch) -> None:
    """Every sigma-consuming primitive `latent.py`'s five functions call is
    spied by name and must receive exactly `config.KERNEL_SIGMA`.

    Reachable red: change any one of `geometry`'s, `correspondence`'s,
    `represent`'s, `attention_spread`'s or `bag_pairs`' own `sigma =
    config.KERNEL_SIGMA` line to read `config.KERNEL_SIGMA * 3` (or any other
    scaled/stale value) -- the corresponding spy call below then records a
    sigma different from the declared constant and the final assertion fails.
    """
    seen: list[tuple[str, object]] = []

    real_bag_kernel = bag_kernel_module.bag_kernel
    real_bag_kernel_matrix = bag_kernel_module.bag_kernel_matrix
    real_relevance_logits = attention_module.relevance_logits

    def spy_bag_kernel(H_u, w_u, H_v, w_v, sigma):
        seen.append(("bag_kernel", sigma))
        return real_bag_kernel(H_u, w_u, H_v, w_v, sigma)

    def spy_bag_kernel_matrix(rows, cols, sigma):
        seen.append(("bag_kernel_matrix", sigma))
        return real_bag_kernel_matrix(rows, cols, sigma)

    def spy_relevance_logits(H, V_R, b_R, v_R, gamma, sigma):
        seen.append(("relevance_logits", sigma))
        return real_relevance_logits(H, V_R, b_R, v_R, gamma, sigma)

    monkeypatch.setattr(bag_kernel_module, "bag_kernel", spy_bag_kernel)
    monkeypatch.setattr(bag_kernel_module, "bag_kernel_matrix", spy_bag_kernel_matrix)
    # `Arm.select`/`Arm.weights_for` (both reached through `bags_of`, which is
    # how `represent`, `attention_spread`, `correspondence` and `bag_pairs`
    # all thread sigma into the attention machinery) call `relevance_logits`
    # via `wiring`'s own top-level import, so the spy is placed there --
    # patching the source module would not affect a name already bound at
    # `wiring` import time.
    monkeypatch.setattr(wiring, "relevance_logits", spy_relevance_logits)

    model, source, target = _model(monkeypatch, "G")
    device = torch.device("cpu")

    source_rows, source_labels = latent.represent(model, source, source.eval_idx, device)
    target_rows, target_labels = latent.represent(model, target, target.eval_idx, device)
    latent.geometry(source_rows, source_labels, target_rows, target_labels)
    latent.attention_spread(model, target, target.eval_idx, device)
    latent.correspondence(model, source, target, device)
    latent.bag_pairs(model, source, target, device)

    assert seen, "no sigma consumer was ever called from latent.py"
    consumers = {name for name, _ in seen}
    assert consumers >= {"bag_kernel", "bag_kernel_matrix", "relevance_logits"}, (
        f"only {consumers} were exercised -- geometry/correspondence/represent/"
        f"attention_spread/bag_pairs together should reach all three"
    )
    for name, sigma in seen:
        assert sigma == config.KERNEL_SIGMA, (
            f"{name} was called from latent.py with sigma={sigma!r}, not the "
            f"declared constant {config.KERNEL_SIGMA!r}"
        )
