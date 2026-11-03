"""Experiments plan v40, Sec. 7.3: EEGNet-8,2 split into trunk and common head."""

from __future__ import annotations

import pytest
import torch

from MIL_CREDA_EEG import eegnet


def _x(n: int, channels: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, channels, eegnet.SAMPLES, generator=g)


@pytest.mark.parametrize("channels", [29, 14])
def test_the_trunk_maps_a_window_to_one_embedding_for_both_montages(channels):
    torch.manual_seed(0)
    trunk = eegnet.EEGNetTrunk(channels)
    out = trunk(_x(5, channels))
    assert out.shape == (5, eegnet.EMBEDDING_DIM)
    assert eegnet.EMBEDDING_DIM == eegnet.F2 * (eegnet.SAMPLES // 32) == 256
    head = eegnet.EEGNetHead(eegnet.EMBEDDING_DIM, 2)
    p = head(out.double())
    assert p.shape == (5, 2)
    assert torch.allclose(p.sum(dim=1), torch.ones(5, dtype=p.dtype))


def test_the_configuration_is_eegnet_8_2():
    trunk = eegnet.EEGNetTrunk(29)
    assert trunk.conv_temporal.weight.shape == (8, 1, 1, 64)
    assert trunk.depthwise.weight.shape == (16, 1, 29, 1)
    assert trunk.depthwise.groups == 8
    assert trunk.separable_depth.weight.shape == (16, 1, 1, 16)
    assert trunk.separable_point.weight.shape == (16, 16, 1, 1)
    assert trunk.pool1.kernel_size == (1, 4) and trunk.pool2.kernel_size == (1, 8)
    assert trunk.drop1.p == trunk.drop2.p == 0.25
    convs = [m for m in trunk.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs and all(m.bias is None for m in convs)
    assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in trunk.modules()) == 3
    assert sum(isinstance(m, torch.nn.ELU) for m in trunk.modules()) == 2


def test_the_max_norm_constraints_hold_after_a_step():
    torch.manual_seed(1)
    trunk = eegnet.EEGNetTrunk(14)
    head = eegnet.EEGNetHead(eegnet.EMBEDDING_DIM, 2)
    with torch.no_grad():
        trunk.depthwise.weight.mul_(50.0)
        head.linear.weight.mul_(50.0)
    optimizer = torch.optim.SGD([*trunk.parameters(), *head.parameters()], lr=10.0)
    loss = -head(trunk(_x(4, 14)).double())[:, 0].log().mean()
    loss.backward()
    optimizer.step()
    eegnet.apply_max_norm(trunk, head)
    depth = trunk.depthwise.weight.detach().reshape(trunk.depthwise.weight.shape[0], -1).norm(dim=1)
    rows = head.linear.weight.detach().norm(dim=1)
    assert float(depth.max()) <= eegnet.DEPTHWISE_MAX_NORM + 1e-6
    assert float(rows.max()) <= eegnet.HEAD_MAX_NORM + 1e-6


def test_a_fresh_network_satisfies_its_max_norm_constraints():
    trunk, head = eegnet.EEGNetTrunk(29), eegnet.EEGNetHead(eegnet.EMBEDDING_DIM, 2)
    depth = trunk.depthwise.weight.detach().reshape(16, -1).norm(dim=1)
    assert float(depth.max()) <= eegnet.DEPTHWISE_MAX_NORM + 1e-6
    assert float(head.linear.weight.detach().norm(dim=1).max()) <= eegnet.HEAD_MAX_NORM + 1e-6


def _trained_source(channels: int) -> eegnet.EEGNetTrunk:
    torch.manual_seed(7)
    source = eegnet.EEGNetTrunk(channels)
    source.train()
    with torch.no_grad():
        source(_x(16, channels, seed=3))           # moves the BatchNorm statistics
        for p in source.parameters():
            p.add_(0.01)
    return source


def test_the_layer_copy_copies_every_layer_and_buffer_when_channels_match():
    source = _trained_source(29)
    torch.manual_seed(99)
    target = eegnet.EEGNetTrunk(29)
    record = eegnet.copy_compatible_layers(source, target)
    assert record["kept_initialization"] == []
    assert set(record["copied"]) == set(eegnet.LAYERS)
    for (name, a), (_, b) in zip(source.state_dict().items(), target.state_dict().items()):
        assert torch.equal(a, b), name


def test_the_layer_copy_keeps_the_depthwise_standard_init_when_channels_differ():
    source = _trained_source(29)
    torch.manual_seed(99)
    target = eegnet.EEGNetTrunk(14)
    fresh = target.depthwise.weight.detach().clone()
    torch.manual_seed(99)
    reference = eegnet.EEGNetTrunk(14)              # the same fresh standard init
    record = eegnet.copy_compatible_layers(source, target)
    assert record["kept_initialization"] == ["depthwise"]
    assert set(record["copied"]) == set(eegnet.LAYERS) - {"depthwise"}
    assert torch.equal(target.depthwise.weight, fresh)
    assert torch.equal(target.depthwise.weight, reference.depthwise.weight)
    assert float(target.depthwise.weight.detach().abs().sum()) > 0.0          # not zeros
    assert record["channels"] == {"source": 29, "target": 14}
    src, tgt = source.state_dict(), target.state_dict()
    for name in tgt:
        if name.startswith("depthwise."):
            continue
        assert torch.equal(src[name], tgt[name]), name


def test_the_standard_init_is_glorot_uniform_with_keras_fans():
    torch.manual_seed(0)
    trunk = eegnet.EEGNetTrunk(29)
    w = trunk.depthwise.weight
    # Keras DepthwiseConv2D kernel (C, 1, F1, D): fan_in = F1*C, fan_out = D*C.
    limit = (6.0 / (8 * 29 + 2 * 29)) ** 0.5
    # 464 uniform draws: the largest lands above 0.9 * limit unless the limit is
    # another one (PyTorch's own xavier_uniform_ gives 0.110 here, not 0.144).
    assert float(w.detach().abs().max()) <= limit + 1e-7
    assert float(w.detach().abs().max()) > 0.9 * limit
    assert torch.equal(trunk.bn1.weight, torch.ones(8))
    assert torch.equal(trunk.bn1.running_var, torch.ones(8))


def test_the_official_code_commit_is_pinned():
    """Sec. 7.3: the implementation records the commit of the official code it follows."""
    import re
    from MIL_CREDA_EEG import eegnet as module
    assert re.fullmatch(r"[0-9a-f]{40}", module.OFFICIAL_CODE["commit"] or "")
    assert module.OFFICIAL_CODE["commit"] in module.OFFICIAL_CODE["url"]
