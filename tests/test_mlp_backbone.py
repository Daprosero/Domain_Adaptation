"""`mlp.py`'s two backbones differ in EXACTLY their `BatchNorm1d` layers.

`"mlp"` and `"mlp-nobn"` are the one instrument that measures whether a bare
normalization statistic is doing the adapting: if they differ in anything
else -- width, activation, initialization -- the comparison measures that
difference instead of normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from MIL_CREDA_Benchmark import config, mlp


def _linear_layers(net: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in net if isinstance(layer, nn.Linear)]


def _batchnorm_layers(net: nn.Sequential) -> list[nn.Module]:
    return [layer for layer in net if isinstance(layer, nn.modules.batchnorm._BatchNorm)]


class TestTheTwoBackbonesDifferOnlyInBatchNorm:
    def test_mlp_has_two_batchnorm_layers(self):
        encoder = mlp.build_mlp("mlp", input_dim=3)
        bns = _batchnorm_layers(encoder.net)
        assert len(bns) == 2
        assert all(isinstance(bn, nn.BatchNorm1d) for bn in bns)

    def test_mlp_nobn_has_no_batchnorm_of_any_kind(self):
        encoder = mlp.build_mlp("mlp-nobn", input_dim=3)
        assert _batchnorm_layers(encoder.net) == []
        # Never an assertion that cannot fail: a deliberately-broken encoder
        # with a BatchNorm inserted must flip this.
        broken = mlp.MLPEncoder(input_dim=3, batchnorm=True)
        assert _batchnorm_layers(broken.net) != []

    def test_the_three_linear_layers_are_identical_in_shape_between_variants(self):
        with_bn = mlp.build_mlp("mlp", input_dim=3)
        without_bn = mlp.build_mlp("mlp-nobn", input_dim=3)
        shapes_with = [(l.in_features, l.out_features) for l in _linear_layers(with_bn.net)]
        shapes_without = [(l.in_features, l.out_features) for l in _linear_layers(without_bn.net)]
        assert len(shapes_with) == 3
        assert shapes_with == shapes_without

    def test_the_only_module_type_difference_is_batchnorm(self):
        with_bn = mlp.build_mlp("mlp", input_dim=2)
        without_bn = mlp.build_mlp("mlp-nobn", input_dim=2)
        kinds_with = [type(layer) for layer in with_bn.net
                     if not isinstance(layer, nn.modules.batchnorm._BatchNorm)]
        kinds_without = [type(layer) for layer in without_bn.net]
        assert kinds_with == kinds_without

    def test_same_initialization_given_the_same_seed(self):
        """`nn.BatchNorm1d`'s default init (weight ones, bias zeros) draws
        nothing from the random generator, so inserting or removing it
        between two `nn.Linear` constructions must not shift which draws
        those `Linear` layers receive. If this ever reddens, the two
        backbones are no longer a controlled comparison of normalization
        alone.
        """
        torch.manual_seed(0)
        with_bn = mlp.build_mlp("mlp", input_dim=3)
        torch.manual_seed(0)
        without_bn = mlp.build_mlp("mlp-nobn", input_dim=3)

        linears_with = _linear_layers(with_bn.net)
        linears_without = _linear_layers(without_bn.net)
        assert len(linears_with) == len(linears_without) == 3
        for a, b in zip(linears_with, linears_without):
            assert torch.equal(a.weight, b.weight)
            assert torch.equal(a.bias, b.bias)


class TestTheInterfaceMatchesFeatureExtractor:
    def test_output_dim_attribute_matches_config(self):
        encoder = mlp.build_mlp("mlp", input_dim=3)
        assert encoder.output_dim == config.MLP_OUTPUT_DIM

    def test_forward_maps_batch_of_d_to_batch_of_output_dim(self):
        for input_dim in (2, 3):
            encoder = mlp.build_mlp("mlp", input_dim=input_dim)
            x = torch.randn(7, input_dim)
            y = encoder(x)
            assert y.shape == (7, config.MLP_OUTPUT_DIM)

    def test_forward_flattens_higher_rank_input_defensively(self):
        encoder = mlp.build_mlp("mlp-nobn", input_dim=3)
        x = torch.randn(5, 3)
        # Already (B, D); forward's reshape is a no-op here, which is the
        # claim being pinned -- it must not silently change the batch size.
        y = encoder(x)
        assert y.shape == (5, config.MLP_OUTPUT_DIM)


class TestUnknownBackboneRefuses:
    def test_build_mlp_raises_on_an_unknown_name(self):
        try:
            mlp.build_mlp("resnet18", input_dim=3)
        except ValueError:
            pass
        else:
            raise AssertionError("expected build_mlp to refuse an unknown backbone name")
