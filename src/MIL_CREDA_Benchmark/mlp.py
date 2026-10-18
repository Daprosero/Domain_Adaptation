"""Three tiny backbones differing in EXACTLY their normalization layer.

Every measurement of the adaptation term so far ran under `resnet18` -- 11M
ImageNet-pretrained parameters on 32x32 digit images. The objection this family
of backbones exists to answer: with a backbone that strong, a normalization
statistic (mean and variance) is already sufficient, so the method's kernels
are superfluous *by construction*. `toy.py`'s low-dimensional manifolds remove
the overbuilt backbone; this module removes it on the model side too.

Every variant is Linear, [normalization], activation, Linear, [normalization],
activation, Linear -- three `nn.Linear` layers and two activations, with the
same widths (`config.MLP_HIDDEN_WIDTH`, `config.MLP_OUTPUT_DIM`), the same
activation (ReLU) and the same initialization given the same seed. The ONLY
axis any two of them differ on is which layer sits in the two bracketed
positions, and `NORM_LAYERS` is the whole of that axis:

* `"mlp-nobn"` puts nothing there.
* `"mlp"` puts `nn.BatchNorm1d`.
* `"mlp-layernorm"` puts `nn.LayerNorm`.

**Why the third one exists, and it is not symmetry with CREDA's figure.** It is
the control that separates *normalizing helps* from *re-estimating helps*.
BatchNorm does two things at once: it normalizes, and it carries
`running_mean`/`running_var` -- state the target domain can re-estimate without
an optimizer and without one target label, which is exactly what AdaBN is.
LayerNorm normalizes and holds **no running statistics at all**, so AdaBN
cannot act on it: `adabn_reading` finds no re-estimable buffer on this backbone
and reports that rather than a number. Without this row, a BatchNorm row that
reads above the un-normalized one cannot be attributed: it may have won by
normalizing, or by having had its statistics re-estimated, and nothing else in
this benchmark tells those apart. The regime where the distinction decides is
the one where AdaBN reads at or above the method's own rung -- which the pilot
record already contains, per cell, in `adabnTargetAccuracy` beside
`targetAccuracy`. No figure from that comparison is quoted here: a measurement
typed into a docstring is one nobody recomputes.

**The draw order is part of "nothing else different", and it holds for all
three.** `nn.BatchNorm1d`'s and `nn.LayerNorm`'s default initializations
(weight ones, bias zeros) draw nothing from the random generator, so inserting,
removing or swapping one between two `nn.Linear` constructions never shifts
which draws those `Linear` layers receive -- the three weight matrices come out
bit-identical across all three variants under the same seed, which is what
makes this a controlled comparison of normalization alone rather than of
normalization-plus-something-else-that-happened-to-move. It is measured, in
`tests/test_mlp.py`, and not assumed: a future layer type whose init did draw
would break the control silently.

**Parameter counts are equal where the layer type allows and never forced.**
`nn.BatchNorm1d(n)` and `nn.LayerNorm(n)` both carry `2n` learnable
parameters, so those two variants hold the same total and `mlp-nobn` holds
`4 * config.MLP_HIDDEN_WIDTH` fewer. What differs between the first two is the
BUFFERS: BatchNorm has three per layer and LayerNorm has none, and that
difference is the row's whole purpose rather than a blemish in the control.

Every variant matches `CREDA.models.FeatureExtractor`'s interface: an
`output_dim` attribute, and a forward mapping `(B, D)` to `(B, output_dim)`. A
toy domain's instances are shape `(D,)` with `D` of 2 or 3, never
`(3, 32, 32)` -- `wiring.Arm.instance_embeddings` flattens a bag's leading
dimensions down to one batch axis before calling the encoder
(`bags.reshape(B * m, *bags.shape[2:])`), so what reaches `forward` here is
already `(B * m, D)`. `forward` reshapes defensively anyway
(`x.reshape(x.shape[0], -1)`), which is a no-op on that shape and keeps this
module from silently assuming its caller never hands it anything higher-rank.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from MIL_CREDA_Benchmark import config

#: The normalization axis, whole: the row name `CREDA.toy_experiment.POS_FILAS`
#: draws, mapped to the layer that row inserts in the two bracketed positions.
#: A factory taking one positional width, so there is ONE construction site and
#: a variant cannot acquire a second difference by being built differently.
#: `None` is a kind and not a gap -- "no normalization" is the row CREDA calls
#: `ninguna`, and spelling it here keeps the three cases on one footing.
NORM_LAYERS: dict[str, type[nn.Module] | None] = {
    "ninguna": None,
    "batchnorm": nn.BatchNorm1d,
    "layernorm": nn.LayerNorm,
}

#: Backbone name -> normalization kind. The backbone name is what
#: `config.BACKBONE` holds and `wiring._resolve_encoder` reads; the kind is what
#: the toyset's rows are called. Two names for one thing, joined here once.
BACKBONE_NORM = {
    "mlp": "batchnorm",
    "mlp-nobn": "ninguna",
    "mlp-layernorm": "layernorm",
}

#: The backbone names `config.BACKBONE` may select to build one of these in
#: place of `CREDA.models.FeatureExtractor` -- read by
#: `wiring._resolve_encoder`. DERIVED from `BACKBONE_NORM` and never written
#: beside it: two lists of the same thing drift, and the one that drifts is the
#: one nobody re-reads.
MLP_BACKBONES = tuple(BACKBONE_NORM)


class MLPEncoder(nn.Module):
    """Three `nn.Linear` layers and two activations, with one normalization
    kind between them -- see the module docstring for why that is the ONLY axis
    this class varies on.
    """

    def __init__(self, input_dim: int, batchnorm: bool | None = None, *,
                 norm: str | None = None):
        """One of `NORM_LAYERS`' kinds, named by `norm`.

        `batchnorm` is the two-state door this class had while there were two
        kinds, kept because callers outside this module construct through it.
        It is an alias and not a second axis: `True` is `norm="batchnorm"` and
        `False` is `norm="ninguna"`. Exactly one of the two has to be given --
        both would be two answers to one question and neither would make the
        normalization a default, which is the one thing a controlled comparison
        may not have.
        """
        super().__init__()
        if (batchnorm is None) == (norm is None):
            raise ValueError(
                "pass exactly one of `norm` (the knob: one of "
                f"{tuple(NORM_LAYERS)}) or `batchnorm` (its two-state alias); "
                f"got norm={norm!r}, batchnorm={batchnorm!r}")
        kind = norm if norm is not None else (
            "batchnorm" if batchnorm else "ninguna")
        if kind not in NORM_LAYERS:
            raise ValueError(
                f"unknown normalization {kind!r}; known: {tuple(NORM_LAYERS)}")
        self.normalization = kind
        self.output_dim = config.MLP_OUTPUT_DIM
        hidden = config.MLP_HIDDEN_WIDTH
        capa = NORM_LAYERS[kind]

        def block(in_features: int, out_features: int) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
            if capa is not None:
                layers.append(capa(out_features))
            layers.append(nn.ReLU())
            return layers

        self.net = nn.Sequential(
            *block(input_dim, hidden),
            *block(hidden, hidden),
            nn.Linear(hidden, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        return self.net(x)


def build_mlp(backbone: str, input_dim: int) -> MLPEncoder:
    """One of the declared MLP backbones, by the name `config.BACKBONE` holds.

    `input_dim` is the caller's to supply -- a toy domain's own dimensionality
    (2 or 3), read off the actual material rather than assumed here, since
    this module has no business knowing which toy family is in play.
    """
    if backbone not in MLP_BACKBONES:
        raise ValueError(f"unknown MLP backbone {backbone!r}; known: {MLP_BACKBONES}")
    return MLPEncoder(input_dim, norm=BACKBONE_NORM[backbone])
