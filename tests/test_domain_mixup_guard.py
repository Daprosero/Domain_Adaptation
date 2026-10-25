"""The two-layer guard against a domain silently reaching the wrong encoder.

Design Decision 7 (`two-encoders-because-each-domain-has-its-own-space`)
names a residual copy-paste hazard even once `represent`/`attention_spread`
require a `domain` argument: the string itself can still be passed wrong at
a call site that names `instance_embeddings`/`target_embeddings` directly
rather than through that argument (the digit report's `correspondence` and
`bag_pairs` were the four such sites; they were retired with the image
benchmark). Two guards answer different halves of the same risk, and
neither may be inferred from the other -- this repository has already paid
for the inferred version:

- **Behavioural, one assertion per consumer.** Perturbing one encoder's
  parameters must move every reading that consumer is supposed to take
  through it, and must not move any reading that consumer is supposed to
  take through the OTHER encoder.
- **Structural, derived rather than listed.** No module may call
  `instance_embeddings` except the two files that are allowed to name it at
  all: `wiring.py` (where it is defined, and legitimately called for the
  source batch in `Arm.forward`/`Arm.training_step`) and `latent.py` (where
  Decision 7's own choice table names it directly, by design, for the source
  side of `correspondence`/`bag_pairs`, and `represent`/`attention_spread`
  route through it for `domain="source"`). A NEW call site anywhere else is
  exactly the shape the behavioural guard above cannot see: a wrong-shaped
  tensor is not what this bug produces, so no value-based assertion catches
  it, only naming the call site does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from MIL_CREDA_Benchmark import bags, config, latent, wiring


class _Encoder(nn.Module):
    """A stand-in for the pretrained resnet18: something with an `output_dim`.

    No manual seed is reset inside `__init__`, so two constructions draw two
    different weight tensors from whatever generator state precedes them --
    exactly the two independently-parameterized encoders design Decision 3
    already relies on (`self.encoder` first, `self.target_encoder` LAST).
    """

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


def _perturb(encoder: nn.Module) -> None:
    """Move every parameter of one encoder, in place, under no grad."""
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.add_(5.0)


def _snapshot(model, source, target, device) -> dict:
    device_arg = device
    source_rows, _ = latent.represent(model, source, source.eval_idx, device_arg, "source")
    target_rows, _ = latent.represent(model, target, target.eval_idx, device_arg, "target")
    return {
        "source_rows": source_rows.clone(),
        "target_rows": target_rows.clone(),
    }


def test_perturbing_the_target_encoder_moves_only_target_side_readings(monkeypatch) -> None:
    """Design Decision 7's behavioural half, target side: perturb `theta_t`
    and the target-side reading `represent` takes must move, while the
    source-side reading must not -- proof that the target side actually reads
    `theta_t` and the source side never does.

    `represent` is the one latent reader the toyset still calls; the digit
    report's `attention_spread`/`bag_pairs`/`correspondence` were retired with
    the image benchmark.

    Reachable red: route `represent(domain="target")` through
    `instance_embeddings` instead of `target_embeddings`.
    """
    model, source, target = _model(monkeypatch, "G")
    device = torch.device("cpu")

    before = _snapshot(model, source, target, device)
    _perturb(model.target_encoder)
    after = _snapshot(model, source, target, device)

    # target-side readings moved
    assert not torch.equal(before["target_rows"], after["target_rows"]), (
        "represent(domain='target') did not move when theta_t moved")

    # source-side readings did not move
    assert torch.equal(before["source_rows"], after["source_rows"]), (
        "represent(domain='source') moved when only theta_t moved")


def test_perturbing_the_source_encoder_moves_only_source_side_readings(monkeypatch) -> None:
    """The mirror of the test above: perturb `theta_s` instead, and the
    source-side reading of `represent` moves while the target-side reading
    does not.

    Reachable red: route `represent(domain="source")` through
    `target_embeddings` instead of `instance_embeddings`.
    """
    model, source, target = _model(monkeypatch, "G")
    device = torch.device("cpu")

    before = _snapshot(model, source, target, device)
    _perturb(model.encoder)
    after = _snapshot(model, source, target, device)

    # source-side readings moved
    assert not torch.equal(before["source_rows"], after["source_rows"]), (
        "represent(domain='source') did not move when theta_s moved")

    # target-side readings did not move
    assert torch.equal(before["target_rows"], after["target_rows"]), (
        "represent(domain='target') moved when only theta_s moved")


def test_no_module_besides_wiring_and_latent_calls_instance_embeddings() -> None:
    """Structural, derived rather than listed (design Decision 7): the ONLY
    two files under `src/MIL_CREDA_Benchmark/` allowed to name
    `instance_embeddings` at all are `wiring.py` -- where it is defined, and
    legitimately called for the source batch in `Arm.forward` and
    `Arm.training_step` -- and `latent.py` -- where Decision 7's own choice
    table names it directly, by design, for the source side of
    `correspondence` and `bag_pairs`, and where `represent`/`attention_spread`
    route through it for `domain="source"`.

    (Decision 7's own prose names only `wiring.py` as exempt; read literally
    that would also forbid `latent.py`'s own, deliberately-chosen source-side
    calls its choice table requires -- so this test exempts both files
    actually named as legitimate call sites in the design, rather than the
    one file whose prose omitted the second.)

    Derived (any call site, anywhere) rather than a roster of today's known
    ones, so it reddens on a NEW site the behavioural guard above structurally
    cannot see: a wrong-shaped tensor is not what a domain-mixup bug produces,
    so nothing value-based catches a target-side reading quietly added
    elsewhere and routed through `instance_embeddings`.

    Reachable red: add a call to `model.instance_embeddings` inside any
    module other than `wiring.py`/`latent.py`.
    """
    exempt = {"wiring.py", "latent.py"}
    root = Path(config.__file__).parent
    offending = []
    for path in sorted(root.glob("*.py")):
        if path.name in exempt:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "instance_embeddings"):
                offending.append(f"{path.name}:{node.lineno}")

    assert not offending, (
        f"instance_embeddings called outside wiring.py/latent.py: {offending}"
    )
