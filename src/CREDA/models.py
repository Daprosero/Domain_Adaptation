"""The architectures the campaign trains, and CREDA's adaptation term itself.

**This module is two things, and the provenance names the half that carries
mathematics.** `CREDALoss` is the term: it implements the managed revision's
class-conditional global score at sample level, which is why the declaration
below is a term's -- a revision, its sections, its equations. The rest of the
file is architecture: `FeatureExtractor` and `Classifier` over torchvision and
timm backbones, the gradient-reversal layer, and the three prior-work wrappers
`DANN_ResNet`, `ADDA_ResNet` and `CDAN_ResNet`. Those implement no equation of
this revision at all; they are the comparison's other arms, and they are here
because this is the module the campaign imports them from.

The two halves could not be declared separately without splitting the module,
which would move every import in the repository and every notebook's stamp. So
the declaration is the term's, and `equations` is read as what this module
implements -- never as all of what it contains. A reader after the prior-work
architectures should read their own papers, not this revision.

**Which equations, and where they are.** `forward` builds Eq. 22's hard
pseudo-label with `argmax`, Eq. 23's per-class index sets as masks, Eq. 24's
confidence weights, Eq. 25's `D_c^t` as an outer product, Eq. 26's weighted
target block (and, under `weight_cross_block`, its cross block), Eq. 27's
concatenated mixture, and Eq. 33's score from three Eq. 10 entropies.
`_mix_kernel_concat` implements Eq. 32's balanced mixture under
`balance_mix_blocks`. Eqs. 35 and 36 are APPLIED here, per class and on the
marginal block, through `CREDA.bounds`, which implements them and declares them
too; `normalize_by_bounds` is off by default, so the published numbers were
taken without them. `_eq37_aggregate` implements Eq. 37 at sample level.

**Eq. 37 joined the list on 2026-10-18, and that is a change of kind.** Until
then this paragraph said the equation was NOT implemented, because `forward`
reduced with a plain mean while r31's Eq. 37 was a sub-convex combination
weighted by the pseudo-label confidences -- the mean was a simplification of it
and declaring it would have overclaimed. r32 replaced those weights with the
observed source proportions `n_c^s / N_s` and charged an inactive class
`l~_c = 1`, and a plain weighted mean over counts IS implementable here: at
sample level `n_c^s` is a sample count instead of a bag count, which is the
same substitution Eq. 21 already makes. So the equation is now implemented
rather than approximated. The two agree exactly when every class is active and
the source batch is balanced, which is how the toyset runs.

**One thing the declaration still does not claim.** Eq. 32's leading one-half
is deliberately not written; `_mix_kernel_concat`'s docstring says why it is a
no-op here.

**Why `revision` says `research-concept-r34.md`.** It said r28 until
2026-10-17, r31 and then r33 until 2026-10-18. The r31 -> r33 rebind DID change
arithmetic here, unlike the other two: r32 rewrote Eq. 37, which
`_eq37_aggregate` now implements, and r33 scoped Eq. 26's confidence weighting
to the class-conditional blocks, which took the weighting out of
`_marginal_score`. The r33 -> r34 rebind changed nothing here, and the reason is
narrow: r34 moved Eq. 15's consensus term to a leave-one-out average, and Eq. 15
is MIL-CREDA's attention over the instances of a bag. This module is the
SAMPLE-level method -- it has no bags and no attention -- so it implements no
equation r34 touched, and every equation it does implement is byte-identical in
r33 and r34. The paragraph below is the first rebind's reasoning, kept because
it is the one that explains why a declaration can move without a number
moving.

**Why the r28 -> r31 rebind changed nothing here.** The rebind was not
housekeeping and it was not an upgrade either: measured equation by equation against the two revisions, this module
implements nine of them -- 10, 22, 23, 24, 25, 26, 27, 32, 33 -- and EIGHT are
byte-identical in `research-concept-r28.md` and `research-concept-r31.md`. The
ninth is Eq. 32, and it moved by renaming the congruence matrix `B_c` to
`Q_c`: the same mathematics under a different letter, and a letter this file
never writes, so there is nothing here to rename. So `r28` had stopped naming
a binding and was naming the date of the last edit, which is the drift a
declaration exists to make visible rather than to accumulate.

Those are the nine the module's arithmetic answers for. What the rebind is NOT
is a claim that r31 left everything else alone: r31 narrowed Eq. 35 to its
upper end, rewrote Eq. 36's denominator, and split section 5 into five
subsections, so a comment here pointing at "r28 section 5" points into a
structure that moved. Those mentions are kept as the history they are and are
not silently renumbered.

**What the r31 declaration claims about its own default, and what the default
used to make it claim.** Eqs. 35 and 36 are in `equations` because they are
APPLIED here, and they are applied through ONE map: `entropy_cosh_bounds`/
`class_global_loss_cosh`, reached when `normalize_by_bounds` is on.
`BOUNDS_FORMS` holds a single admissible value and `bounds_form` survives only
so that a caller naming the retired map is refused by name rather than meeting
`TypeError: unexpected keyword argument`.

**Which path a caller gets WITHOUT asking is the part this paragraph used to
state wrongly, four times over.** It read that `bounds_form` "selects r25's
`conservative_bounds`/`class_global_loss` or r31's"; both r25 symbols were
removed on 2026-10-18 and neither exists under `src/`. It read that
"**`entropy-cosh` is the default**, so the path a caller gets without asking
computes r31's Eq. 36"; `normalize_by_bounds` defaults to `False`, so no default
path applies Eq. 36 at all -- `bounds_form` names which map would be used if the
normalization were on, which is a different claim. It read that `train_creda`
"asks for `bounds_form='conservative', balance_mass='cardinality'` by name";
`training_pipeline.py` passes `balance_mass='cardinality'` and nothing else. And
it read that `CREDA.bounds` "carries the same r25/r31 split in its own
`__provenance__` -- which stays at r25"; that module declares
`research-concept-r34.md` and holds one pair of ends.

**What the superseded paragraph was protecting is still real.** Every number
published out of this repository was measured under some specific pair of flags,
and a number protected by a default is protected by nothing once the default
moves. That is why the two reachable sets are named rather than defaulted:
`Plots_Generator`/`Results_Generator` reach `CREDA.training_pipeline`, whose
`train_creda` names `balance_mass='cardinality'`, and the toyset reaches
`CREDA.toy_experiment`, which names `NORMALIZAR_POR_COTAS`/`FORMA_DE_COTAS`/
`BALANCE_DE_MASA`. Those two namings are what the campaign's numbers rest on,
and `normalize_by_bounds`'s own default is a separate open question recorded
beside Eq. 37's aggregation.

Nothing in this file implements either equation: neither the subtraction
Eq. 36's numerator is, nor either denominator any revision gave it, is written
anywhere in it -- `test_models_says_which_of_its_equations_it_only_applies`
searches for them by their literal text, which is also why this paragraph
describes them instead of quoting them. So the statement above cannot quietly
stop being true.

Every id in `invariants` below is a test that already existed when this
declaration was written; none was added to back it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.serialization
from torchvision import models, datasets, transforms
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import random
import math
import os
import zipfile
from itertools import permutations
from tqdm import tqdm
import timm

# Eqs. (35)-(36), copied onto the CREDA side rather than imported from
# `MIL_CREDA` -- `CREDA.bounds`' own module docstring says why, and
# `tests/test_creda_sample.py` is what holds each copy equal to its original.
# Read only by `CREDALoss.normalize_by_bounds`, which is off by default.
#
# Two revisions arrive on that line. `conservative_bounds`/`class_global_loss`
# are r25's ends; `entropy_cosh_bounds`/`class_global_loss_cosh` are r31's
# (introduced at r28, and r31 rewrote Eq. (36)'s denominator under them), and
# `bounds_form` is which of the two a run uses. Both are reachable on purpose,
# and r31's is the default: every number already published out of this
# repository was measured under r25's, and `training_pipeline.train_creda` --
# the only construction the old notebooks reach -- asks for r25's by name
# rather than leaning on a default that has since moved. That name is what
# keeps `conservative_bounds` and `class_global_loss` alive in this file: they
# are not retained out of history, they are what one declared consumer
# computes with.
from CREDA.bounds import class_global_loss_cosh


#: What `CREDALoss` implements, and the revision it is bound to. See the module
#: docstring for why a file holding prior work's architectures declares a term's
#: provenance, for the two equations this list deliberately does not name, and
#: for why `revision` moved from r28 to r31 without a line of arithmetic
#: changing in this file.
__provenance__ = {
    "revision": "research-concept-r34.md",
    "sections": ["2", "3", "5"],
    "equations": ["10", "22", "23", "24", "25", "26", "27", "32", "33",
                  "35", "36", "37"],
    "invariants": [
        # The default composition, pinned numerically: Eqs. 22-27 and 33 as
        # `forward` assembles them, against means read off a run from before
        # the three forks existed.
        "the_pinned_pre_fork_means_still_come_out_of_the_default_path",
        "the_new_keywords_default_to_todays_behaviour",
        "normalize_kernel_embedding_bounds_what_the_kernels_see",
        # Eq. 26's second half, the one the class used to leave raw.
        "weight_cross_block_multiplies_the_cross_block_by_d_from_the_right",
        # Eq. 27's plain concatenation, and what Eq. 32's balance repairs in it.
        "balance_mix_blocks_off_is_the_plain_concatenation",
        "balance_mix_blocks_closes_the_cardinality_exploit_the_default_leaves_open",
        # Eq. 37 at sample level: the weights are the observed source
        # proportions, an abandoned class costs the maximum, and the charge
        # carries no gradient.
        "eq37_weights_are_the_observed_source_proportions",
        "eq37_charges_one_for_a_class_the_routing_abandoned",
        "eq37_inactive_charge_carries_no_gradient",
        # r33 scopes Eq. 26's `D` to the class-conditional blocks only.
        "the_marginal_block_is_not_confidence_weighted",
        # Eq. 32: trace against cardinality, where they agree and where they
        # part company.
        "the_two_balances_agree_exactly_when_every_weight_is_one",
        "the_two_balances_differ_by_a_real_margin_once_weights_bite",
        # Eqs. 35-36 as applied here, and the hypothesis that pairing rests on.
        "normalize_by_bounds_maps_each_class_onto_its_own_unit_interval",
        "the_trace_balance_never_leaves_the_unit_interval",
        "entropy_cosh_refuses_an_unbalanced_mixture",
        "entropy_cosh_refuses_the_cardinality_balance",
        # Section 5's active-class rule, which runs under every setting.
        "the_collapsed_class_does_not_reach_the_per_class_reduction",
        "the_collapsed_class_produces_no_nan_and_no_inf",
        "the_floor_is_a_declared_constant_tied_to_the_estimator",
        # The marginal block, which belongs to no class.
        "the_marginal_block_equals_the_loop_when_one_class_holds_everything",
    ],
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Gradient Reversal Layer (usada en DANN y opcionalmente en CDAN)
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class FeatureExtractor(nn.Module):
    """
    backbone : str
        Opciones:
        - 'resnet18'
        - 'resnet50'
        - 'vit_tiny_patch16_224'
    pretrained : bool
        Si carga pesos preentrenados.
    """
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()

        self.backbone = backbone

        if backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.flatten = nn.Flatten()
            self.output_dim = 512

        elif backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.flatten = nn.Flatten()
            self.output_dim = 2048

        elif backbone == 'vit_tiny_patch16_224':
            self.features = timm.create_model(
                backbone,
                pretrained=False,
                img_size=32
            )
            self.output_dim = self.features.head.in_features
            self.features.head = nn.Identity()
            self.flatten = nn.Identity()

            if pretrained:
                pretrained_model = timm.create_model(backbone, pretrained=True)
                state_dict = pretrained_model.state_dict()
                state_dict = self._interpolate_pos_embed(state_dict, self.features)
                self.features.load_state_dict(state_dict, strict=False)

        else:
            raise ValueError(
                f"Backbone no soportado: {backbone}. "
                f"Usa 'resnet18', 'resnet50' o 'vit_tiny_patch16_224'."
            )

    def _interpolate_pos_embed(self, state_dict, model):
        if 'pos_embed' not in state_dict:
            return state_dict

        posemb = state_dict['pos_embed']
        posemb_new = model.pos_embed

        num_extra_tokens = posemb.shape[1] - int(math.sqrt(posemb.shape[1] - 1)) ** 2

        cls_token = posemb[:, :num_extra_tokens]
        posemb_grid = posemb[:, num_extra_tokens:]

        gs_old = int(math.sqrt(posemb_grid.shape[1]))
        gs_new = int(math.sqrt(posemb_new.shape[1] - num_extra_tokens))

        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(
            posemb_grid,
            size=(gs_new, gs_new),
            mode='bicubic',
            align_corners=False
        )
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

        posemb = torch.cat([cls_token, posemb_grid], dim=1)
        state_dict['pos_embed'] = posemb
        return state_dict

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

class Classifier(nn.Module):
    def __init__(self, feature_dim=256, num_classes=10):
        super().__init__()
        set_seed(42)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.BatchNorm1d(feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim//4),
            nn.BatchNorm1d(feature_dim//4),
            nn.ReLU(),
            nn.Linear(feature_dim//4, num_classes)
        )

    def forward(self, x):
        return self.net(x)
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        set_seed(42)
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.BatchNorm1d(input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, 1)
        )
    def forward(self, x):
        return self.net(x)
class DANN_ResNet(nn.Module):
    def __init__(self, backbone, num_classes=10, lambda_grl=1.0, pretrained=True):
        super().__init__()
        set_seed(42)
        self.feature = FeatureExtractor(backbone=backbone, pretrained=pretrained)
        self.classifier = Classifier(feature_dim=self.feature.output_dim, num_classes=num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim=self.feature.output_dim)
        self.grl = GradientReversalLayer(lambda_grl)

    def forward(self, x, mode='class'):
        f = self.feature(x)
        if mode == 'class':
            return self.classifier(f)
        elif mode == 'domain':
            return self.domain_discriminator(self.grl(f))
class ADDA_ResNet(nn.Module):
    def __init__(self, backbone, num_classes=10, pretrained=True):
        super().__init__()
        set_seed(42)
        self.Fs = FeatureExtractor(backbone=backbone, pretrained=pretrained)
        self.Ft = FeatureExtractor(backbone=backbone, pretrained=pretrained)
        self.classifier = Classifier(feature_dim=self.Fs.output_dim, num_classes=num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim=self.Fs.output_dim)

    def forward(self, x, domain='source', mode='class'):
        if domain == 'source':
            f = self.Fs(x)
        elif domain == 'target':
            f = self.Ft(x)
        else:
            raise ValueError("domain must be 'source' or 'target'")
        if mode == 'class':
            return self.classifier(f)
        elif mode == 'domain':
            return self.domain_discriminator(f)
        else:
            raise ValueError("mode must be 'class' or 'domain'")
class CDAN_ResNet(nn.Module):
    def __init__(self, backbone, num_classes=10, pretrained=True):
        super().__init__()
        set_seed(42)
        self.feature = FeatureExtractor(backbone=backbone, pretrained=pretrained)
        self.classifier = nn.Linear(self.feature.output_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim=self.feature.output_dim * num_classes)

    def forward(self, x, mode='class'):
        f = self.feature(x)
        y = self.classifier(f)
        if mode == 'class':
            return y
        elif mode == 'domain':
            y_soft = F.softmax(y, dim=1)
            outer = torch.bmm(y_soft.unsqueeze(2), f.unsqueeze(1)).view(x.size(0), -1)
            return self.domain_discriminator(outer)

class CREDA_ResNet(nn.Module):
    def __init__(self, backbone, num_classes=10, pretrained=True):
        super().__init__()
        set_seed(42)
        self.feature = FeatureExtractor(backbone=backbone, pretrained=pretrained)
        self.classifier = nn.Linear(self.feature.output_dim, num_classes)

    def forward(self, x, mode='class'):
        f = self.feature(x)
        y = self.classifier(f)
        if mode == 'class':
            return y  # softmax se aplica externamente si es necesario
        elif mode == 'feature':
            return f
#: The scopes `CREDALoss.sigma_from` accepts.
#:
#: `class-pooled` is today's behaviour and the default. **It is not what the
#: formulation declares.** r25 line 732: "Un unico ancho de banda sigma gobierna
#: los tres bloques, ya que los tres derivan del mismo kernel de instancia", and
#: line 826 reuses that same kernel for the local term precisely so as not to
#: introduce "un segundo kernel ni otro ancho de banda sigma". One instance
#: kernel means one sigma; a per-class median is C of them.
#:
#: And the per-class median is measurably defective, not merely undeclared.
#: Displacing one class's target features and reading that class's `creda_c`:
#: the per-class sigma rises to 0.505 at a displacement of 10 and then FALLS to
#: 0.437 and 0.397 at 30 and 100, so a worse-aligned class reports a smaller
#: discrepancy. Under one shared sigma the same sweep rises monotonically and
#: saturates at ln 2 = 0.69313, the correct maximum for two separated clouds of
#: equal size. The cause is that the per-class median is taken over that class's
#: source AND target, so displacing the target grows sigma with it and the
#: kernel's own unit of measurement moves with the thing it measures.
#:
#: `batch-pooled` is the declared reading: one sigma for every class, over the
#: whole batch's source and target. `batch-source` is the scope every toy
#: measurement of the free two-phase regime actually used. Neither is the
#: default, because the default's job is to keep published numbers reproducible.
SIGMA_SCOPES = ('class-pooled', 'class-source', 'batch-pooled', 'batch-source')

#: The maps `CREDALoss.bounds_form` accepts, read only when
#: `normalize_by_bounds` is on.
#:
#: `conservative` is r25's Eq. (35) -- `L_c = -ln(n_s + n_t)` and
#: `U_c = (ln n_s + ln n_t)/2`, both functions of cardinality alone. It was the
#: default until 2026-10-17, because every number already published out of this
#: repository was measured under it; it is now asked for BY NAME at the one
#: place that reaches those numbers, `training_pipeline.train_creda`, which is
#: a stronger guarantee than a default and not a weaker one.
#:
#: `entropy-cosh` is the entropy-dependent map r28 introduced and r31 still
#: states, and it is the default, so a caller that does not ask computes what
#: the repository currently states: `U_c = 2 ln cosh((H_2(K~_t) - H_2(K_s))/4)` from Eq. (35), with
#: `L_c = -ln 2` from Eq. (34), both evaluated on the same trace-normalized
#: matrices that build the score. The difference is not a tightening. r25's
#: ends are CONSTANTS once the counts are fixed, and an affine map with
#: constant ends preserves the order of what it maps -- so it inherits the one
#: place the raw score misbehaves: with the source maximally spread and the
#: target collapsed far away, `J_c` is +0.52 at `n_s = 43`, above the 0 of
#: perfect alignment, and the map reports that collapse as BETTER than a
#: perfect match. r28 §5 states this as the reason for the change.
#:
#: Measured here on that configuration: conservative reads 0.3949 for the
#: collapse against 0.4578 for the alignment, entropy-cosh reads 0.3764
#: against 0.0000.
#:
#: **2026-10-17: that entropy-cosh pair read 0.1773 against 0.0000 until
#: today, and only the first number moved.** It was measured under r29's
#: denominator, the per-class span `U_c - L_c`; r31 divides by `ln 2` instead,
#: so every value this map returns is scaled by `(U_c + ln 2) / ln 2` -- here
#: `U_c = 0.7782`, a factor of 2.1227, and `0.1773` becomes `0.3764`. The
#: 0.0000 is unchanged because at perfect alignment the two marginals are the
#: same matrix, so `U_c = 0` and the two denominators coincide. What the
#: paragraph exists to say survives the rescaling intact: the collapse still
#: reads WORSE than the alignment, which is the ordering r25's map inverts.
#:
#: Neither figure is quoted from memory. `tests/test_r31_ec36.py` rebuilds this
#: configuration, measures it, reproduces the superseded 0.1773 to prove the
#: fixture is the one this comment means, and holds the number written above to
#: what the module returns -- so the next revision that touches Eq. (36)
#: reddens this paragraph instead of aging it a third time.
#: **2026-10-18: `'conservative'` left this tuple, and the parameter stayed.**
#: r33's Eq. (35) states one pair of ends, the `cosh` pair, and r25's
#: cardinality pair is no longer anywhere in the formulation -- so
#: `conservative_bounds` and `class_global_loss` were removed from
#: `CREDA.bounds` rather than kept behind a name. The keyword survives with a
#: single legal value on purpose: a run that still asks for `'conservative'`
#: is told that this map is gone, which a `TypeError: unexpected keyword
#: argument` would not say. Reaching r25's numbers now means reading them off
#: the artefacts that recorded them, not recomputing them here.
BOUNDS_FORMS = ('entropy-cosh',)

#: The balances `CREDALoss.balance_mass` accepts, read only when
#: `balance_mix_blocks` is on.
#:
#: `cardinality` is what `_mix_kernel_concat` did before r28 and what every
#: published number was measured under: `K_s/n_s`, `K_t/n_t`,
#: `K_st/sqrt(n_s n_t)`. It was the default until 2026-10-17 and is now asked
#: for by name at `training_pipeline.train_creda`, the one place that reaches
#: those numbers -- see `BOUNDS_FORMS` above for why a name beats a default.
#:
#: `trace` is the default, and it is r28's Eq. (32): the same congruence, with
#: each block's own trace
#: in place of its count. On the source side the two agree exactly, because the
#: Gaussian kernel has 1 on its diagonal and so `tr(K_s) = n_s`. On the target
#: side they do not, because `K~_t` arrives already weighted -- `K_t * outer(w,
#: w)` is `D K_t D`, so its diagonal is `w_j^2` and `tr(K~_t) = sum_j w_j^2`,
#: which equals `n_t` only when every weight is exactly 1. Dividing by `n_t`
#: therefore hands the mixture a target block whose mass was never rescaled to
#: the confidence it earned, which is the defect r28 repairs. Measured on a
#: 129-sample batch with softmax pseudo-labels rather than one-hot ones,
#: `tr(K~_t)/n_t` came out 0.470, 0.774 and 0.482 across the three classes,
#: while `tr(K_s)` was 43.0000 against `n_s = 43` in every one of them. So the
#: source side is exact and the target side is a 20-55% misstatement of the
#: mass -- not a rounding difference.
BALANCE_MASSES = ('cardinality', 'trace')

#: The floor below which a class's trace counts as collapsed -- r28 §5's active
#: class requires the three matrices that must be normalized to have POSITIVE
#: trace, and this is where that word gets a number.
#:
#: **Su razon escrita murio el 2026-10-18 y el valor se queda, declarado como
#: lo que ahora es: una decision sin medir.** Decia "es el epsilon propio de
#: `_renyi_entropy_order_2`: esa funcion divide por `trace + 1e-6`, asi que en
#: o por debajo de 1e-6 el estimador no distingue la matriz del cero". Ese
#: epsilon se removio --- dividia por una constante sumada a una traza que
#: puede ser mucho menor que ella --- y el estimador es ahora exacto a 7
#: digitos en float32 desde `tr(K) = 1` hasta `1.175e-38`, el minimo normal de
#: float32. Medido:
#:
#:     tr(K)      forma nueva       forma vieja
#:     1e+00      1.14525199        1.14525080
#:     1e-06      1.14525199        2.53153396
#:     1e-09      1.14525199       13.53976822
#:     1e-38      1.14525199       13.81551075
#:
#: Asi que el piso que el ESTIMADOR necesita es `torch.finfo(torch.float32).
#: tiny`, treinta y dos ordenes de magnitud por debajo de este.
#:
#: **Y aun asi el valor NO baja, porque la Ec. (37) se contradice consigo misma
#: en ese caso.** Su condicion de membresia pide traza POSITIVA (r34 linea
#: 1109: "aquellas para las cuales existen sujetos en ambos dominios y las tres
#: matrices que deben normalizarse poseen traza positiva"), de modo que por la
#: letra una clase de traza `2.9e-15` entra al conjunto activo. Pero su carga
#: de uno existe, dice la misma ecuacion en la linea 1132, para "que `L_glob`
#: no descienda cuando el enrutamiento abandona clases, de modo que ese
#: abandono se lea como un aumento de la perdida y no como una CONVERGENCIA
#: APARENTE".
#:
#: Medido sobre `_one_dead_class` de `tests/test_creda_r28_bounds.py`, donde el
#: clasificador queda maximamente incierto sobre nueve bolsas y la Ec. (24) les
#: da peso `1.8e-08`:
#:
#:     piso 0.0   -> 3 clases activas; la colapsada lee 0.00059861, L_glob = 0.00053086
#:     piso 1e-6  -> 2 clases activas; la colapsada cobra 1,        L_glob = 0.33366465
#:
#: Seguir la letra hace que una clase sin evidencia objetivo lea alineacion casi
#: perfecta, que es exactamente la lectura que la carga de uno existe para
#: impedir. Las dos mitades de la Ec. (37) no pueden cumplirse a la vez aqui, y
#: elegir cual gana no es una decision de implementacion: queda para una
#: deliberacion y el valor se queda donde se midio todo lo publicado.
#:
#: Lo que cuesta mantenerlo, medido: 1 de 1158 bloques de clase de un barrido
#: aleatorio cae en la franja `(1.2e-38, 1e-6]`, un 0.09%.
MIN_CLASS_TRACE = 1e-6


class CREDALoss(nn.Module):
    """CREDA's class-conditional adaptation term, with three additive forks.

    `normalize_kernel_embedding`, `weight_cross_block` and `normalize_by_bounds`
    are ports of three things MIL-CREDA already does, brought down to sample
    level so each can be tested without the bag machinery. **Every one defaults
    to the value that leaves this class bit-identical to what it was before they
    existed**, which is the whole contract: `True` selects ours, and every
    number already in this repository stays reproducible with no flag set.

    **Run-level, never per-arm.** Each of the three is passed once when the loss
    is constructed for a run, exactly as `config.NORMALIZE_KERNEL_EMBEDDING`
    already establishes on the MIL side and for its stated reason: a per-arm
    flag would let two adjacent rungs of an ablation ladder differ in the
    component under test AND in the normalization, so no difference between them
    could be attributed to either one.

    `normalize_kernel_embedding`
        L2-normalize the embedding the kernels evaluate on. The form is
        `wiring.Arm._kernel_embedding`'s: `F.normalize(H, p=2, dim=-1)`, and
        with the flag off `_kernel_embedding` returns its argument itself --
        not even a clone. `KERNEL_SIGMA`-style constants aside, the kernel's
        regime is set by the ratio of the embedding's own scale to the
        bandwidth, and a trained encoder is free to move that scale;
        normalizing bounds ||h - h'||^2 to [0, 4] by construction and pins it.
        The pseudo-label distribution `g_t` is the CALLER's and arrives already
        computed from raw features, so this flag never reaches it.

    `weight_cross_block`
        Eq. (26)'s missing half. This class already weights the target block,
        `K_t_c * outer(w, w)` = `D K_c^t D`, and then feeds the cross block into
        the mixture RAW. Eq. (26) also states `K~_c^{st} = K_c^{st} D`: rows of
        `K_st` are source and columns are target, so the weight enters from the
        right and ONLY on the target side -- never an outer product, which
        would weight the source too, and the source labels are observed rather
        than estimated. Does nothing unless `use_entropy_weighting` is also on,
        because without it there is no `D`.

    `normalize_by_bounds`
        Eqs. (35)-(36): map each class's score affinely onto [0, 1] by its own
        conservative bounds, so classes of different cardinality land on a
        common scale. `creda_c = h_mix - 0.5 (h_s + h_t)` here and MIL's
        `J_c = 0.5 (H_s + H_t) - H_mix` are the same quantity with opposite
        sign, so what is handed to `class_global_loss` is `-creda_c`. The
        bounds are functions of cardinality alone, which is what lets MIL's
        bag counts become sample counts here with no change to the equation.

    `bounds_form`
        WHICH Eq. (35) `normalize_by_bounds` applies -- r25's constants or
        r31's entropy-dependent ends. `entropy-cosh`, r31's, is the default;
        `conservative` is r25's and reproduces every published number, and the
        one path that reaches those numbers asks for it by name. See
        `BOUNDS_FORMS` for what separates them and for the configuration where
        the constant ends invert the order of the thing they map. Read only
        when `normalize_by_bounds` is on, exactly as `balance_mass` is read
        only under `balance_mix_blocks`.

    `balance_mass`
        WHICH quantity `balance_mix_blocks` equalizes between the two domains
        -- each block's cardinality or, as r28 Eq. (32) writes it, each
        block's trace. `trace` is the default; `cardinality` is the pre-r28
        arithmetic and is asked for by name at the same place. The two agree
        exactly when every confidence weight is 1 and part company as soon as
        they are not, because the target block reaches this class already
        weighted. See `BALANCE_MASSES`.

    **Both of the above were opt-in values until 2026-10-17 and are now the
    defaults**, which is where they part company with the three forks above:
    those still hold the contract that nothing moves unless a caller names it,
    and these two now move the other way -- a caller that does not name them
    gets r31's. Nothing published moved with them, because both are read only
    under a flag that is still off by default and no construction in `src/`
    turns either flag on; what moved is which map the NEXT caller inherits.
    `tests/test_r31_ec36.py` holds that measurement, the pin that protects the
    published path, and the defaults themselves.

    The ONE exception is r28 §5's active-class
    rule, which `forward` now applies under every setting -- a class whose
    target trace has collapsed is skipped rather than divided by, because the
    score is not defined there. It used to produce a finite number anyway, out
    of `_renyi_entropy_order_2`'s epsilon, which is a number that was never
    about the data. No ordinary class comes near the floor; see
    `MIN_CLASS_TRACE`.

    A trainable bandwidth is deliberately absent and must stay absent: measured
    with features frozen, a learned sigma grows about seventeenfold and drives
    this loss down two hundredfold by saturating the kernel, which is an escape
    from the objective rather than a solution to it.
    """

    def __init__(self, sigma='auto', lambda_creda=1.0, use_entropy_weighting=True,
                 normalize_kernel_embedding=False, weight_cross_block=False,
                 normalize_by_bounds=False, sigma_from='class-pooled',
                 detach_target_entropy=False,
                 class_mixing=1.0, balance_mix_blocks=False,
                 bounds_form='entropy-cosh', balance_mass='trace'):
        super(CREDALoss, self).__init__()
        if sigma_from not in SIGMA_SCOPES:
            raise ValueError(
                f"sigma_from must be one of {SIGMA_SCOPES}, not {sigma_from!r}")
        # Refused here rather than where they are read, the way `sigma_from`
        # already is: a run misspelling either one would otherwise construct
        # fine, fall through to the default branch, and report numbers from a
        # map nobody asked for.
        if bounds_form not in BOUNDS_FORMS:
            raise ValueError(
                f"bounds_form must be one of {BOUNDS_FORMS}, not {bounds_form!r}")
        if balance_mass not in BALANCE_MASSES:
            raise ValueError(
                f"balance_mass must be one of {BALANCE_MASSES}, not {balance_mass!r}")
        if not 0.0 <= float(class_mixing) <= 1.0:
            raise ValueError(
                f"class_mixing is a convex coefficient in [0, 1], not {class_mixing!r}")
        # Eq. 35's ends are a THEOREM about Eq. 32's balanced mixture, and this
        # is where that stops being two independent axes.
        #
        # r28 line 1010 states the hypothesis outright: "Las cotas de la
        # puntuacion se derivan de la estructura por bloques de la conjunta
        # BALANCEADA." The derivation needs all of Eq. 32: the identity
        # `rho_mix = rho_s/4 + rho_t/4 + ||K_st||_F^2 / 2` comes from the
        # leading 1/2 together with the trace-normalized diagonal blocks, and
        # Cauchy-Schwarz on those same blocks is what bounds the cross block by
        # `sqrt(rho_s rho_t)`. `L_c = -ln 2` and `U_c` follow from the two.
        #
        # Measured over 300 random configurations -- dimension, both
        # cardinalities, both scales, the displacement, the confidences and the
        # bandwidth all drawn: under the trace balance NONE of them leaves
        # [0, 1]; under the cardinality balance about 190 do and under no
        # balance about 160, spanning roughly [-0.83, +1.85]. A NEGATIVE loss
        # is not a cosmetic range violation -- it tells the optimizer to keep
        # going in the direction that produced it.
        #
        # The mechanism, for whoever hits this: as the confidences fall, the
        # cardinality balance hands the source almost the whole mixture, so
        # `h_mix -> h_s` and `J_c -> (h_t - h_s)/2`, LINEAR in the entropy gap,
        # while `U_c = 2 ln cosh((h_t - h_s)/4)` is QUADRATIC in it. Linear
        # overtakes quadratic and `J_c` passes `U_c`.
        #
        # A refusal and not a warning: the whole point is that the pairing
        # become unreachable. `conservative` is deliberately NOT narrowed --
        # r25's ends are constants, so they assume nothing about the mixture,
        # and every number already published here was taken with them.
        # Las cotas de las Ecs. 35-36 se prueban sobre la mezcla balanceada
        # con los TRES bloques compartiendo el normalizador por traza. El
        # estadistico U le da a cada bloque el suyo --- `S^2 - sum u_j^2`, que
        # depende de los pesos de ESE bloque --- asi que esa hipotesis se
        # rompe y los extremos dejan de ser `L = -ln 2` y
        # `U_c = 2 ln cosh((h_t - h_s)/4)`.
        #
        if normalize_by_bounds and bounds_form == 'entropy-cosh':
            if not balance_mix_blocks:
                raise ValueError(
                    "bounds_form='entropy-cosh' is Eq. 35, proved on Eq. 32's "
                    "balanced mixture, so it requires balance_mix_blocks=True; "
                    "with the raw Eq. 27 concatenation about 160 of 300 random "
                    "configurations give a loss outside [0, 1], some negative")
            if balance_mass != 'trace':
                raise ValueError(
                    "bounds_form='entropy-cosh' is Eq. 35, proved on Eq. 32's "
                    f"balanced mixture, so it requires balance_mass='trace', "
                    f"not {balance_mass!r}; under the cardinality balance about "
                    "190 of 300 random configurations give a loss outside "
                    "[0, 1], some negative")
            # **La tercera condicion de la misma guarda, medida 2026-10-18.** La
            # Ec. 26 sopesa LOS DOS bloques que tocan bolsas objetivo:
            # `K~_c^t = D K_c^t D` y `K~_c^st = K_c^st D`. Con
            # `weight_cross_block=False` solo se sopesa el primero, de modo que la
            # mezcla de la Ec. 27 no es la que el teorema de la Ec. 35 demuestra:
            # el bloque cruzado queda grande frente a un bloque objetivo
            # encogido, `J_c` pasa `U_c` y la Ec. 36 da negativo.
            #
            # MEDIDO sobre 600 configuraciones aleatorias (1733 lecturas por
            # clase): con el cruzado sin sopesar, 1378/1733 caen fuera de [0, 1],
            # hasta -20.26. Con el cruzado sopesado, 7/1733, hasta -0.41. No es un
            # borde, es el 80%.
            #
            # Refutado por la misma razon que las otras dos: un puntaje fuera de
            # [0, 1] bajo un mapa que declara acotarlo ahi se lee como un valor
            # medido y ninguna cota lo delata.
            if use_entropy_weighting and not weight_cross_block:
                raise ValueError(
                    "bounds_form='entropy-cosh' with use_entropy_weighting=True "
                    "requires weight_cross_block=True: Eq. 26 weights BOTH blocks "
                    "that touch target bags, and Eq. 35's bound is proved on that "
                    "mixture. Measured, 1378 of 1733 per-class readings fall "
                    "outside [0, 1] with the cross block left raw, down to -20.26; "
                    "with it weighted, 7 of 1733.")
        self.lambda_creda = lambda_creda
        self.sigma = sigma
        self.use_entropy_weighting = use_entropy_weighting
        self.normalize_kernel_embedding = normalize_kernel_embedding
        self.weight_cross_block = weight_cross_block
        self.normalize_by_bounds = normalize_by_bounds
        self.sigma_from = sigma_from
        # Si `h_t` entra en `creda_c` SIN gradiente. Opt-in; `False` deja
        # la linea base bit a bit.
        #
        # Es un instrumento y no una variante a proponer: aisla la fuerza
        # DISPERSIVA que el bloque objetivo ejerce sobre si mismo. Esa
        # fuerza entra pesada por `w_j^2 w_k^2`, de modo que con el peso
        # declarado se concentra en los pares de muestras SEGURAS y separa
        # el nucleo confiado de cada clase; con pesos barajados cae sobre
        # pares al azar y no ataca al nucleo. Es el unico candidato que
        # cumple las tres condiciones que la medicion le exige a un
        # mecanismo de la caida de confianza: sobrevive sin el cruzado
        # sopesado, sobrevive con `w` sin gradiente, y depende de la
        # ASIGNACION.
        #
        # `detach` NO cambia lo que `forward` devuelve, solo por donde
        # fluye hacia atras, asi que un test sobre la perdida pasaria con
        # la bandera prendida y con la bandera rota. El test lee el
        # GRADIENTE.
        self.detach_target_entropy = detach_target_entropy
        # Si la pureza se calcula como estadistico U: sin los pares de
        # cada muestra CONSIGO MISMA. Opt-in; `False` deja la linea base
        # bit a bit.
        #
        # `tr(K^2)` cuenta los pares `j = k`, que para un kernel de
        # diagonal unitaria valen 1 y para el bloque del blanco SOPESADO
        # valen `w_j^4`. Ese bloque aporta `1/n_eff` con `n_eff` el tamaño
        # efectivo de muestra, de modo que desparejar los pesos baja
        # `n_eff` e infla la pureza SIN QUE LA DISTRIBUCION HAYA CAMBIADO.
        # Es el mismo sesgo que el MMD insesgado quita con estadisticos U.
        #
        # Medido con la geometria congelada: el castigo por pesos
        # desiguales es 97% ese artefacto, y la correlacion con el
        # coeficiente de variacion de `w` cae de r=+0.970 a r=+0.057.
        #
        # El bloque CRUZADO no tiene pares consigo mismo --- fuente y
        # blanco son muestras distintas --- asi que el sesgo lo cargan solo
        # las purezas internas; en la mezcla concatenada la diagonal es la
        # union de esos dos conjuntos de pares propios, y quitarla deja los
        # terminos cruzados intactos.
        self.class_mixing = float(class_mixing)
        self.balance_mix_blocks = balance_mix_blocks
        self.bounds_form = bounds_form
        self.balance_mass = balance_mass

    def _kernel_embedding(self, H):
        """The embedding the kernels evaluate on -- the one fork
        `normalize_kernel_embedding` governs.

        With the flag off this returns `H` itself, unmodified and not even a
        clone, so every path is bit-identical to before this fork existed.

        Applied to the whole batch before the per-class filter rather than once
        per class inside it, which is a choice of cost and not of value:
        `F.normalize(..., dim=-1)` acts on each row independently, so selecting
        rows and normalizing rows commute exactly. The bandwidth is derived from
        the same view the kernels then see -- `_compute_sigma`'s median
        heuristic reads distances in the embedding, so leaving it on the raw
        view while the kernels read the normalized one would hand the
        exponential a bandwidth from a different space and produce exactly the
        dead or saturated kernel this flag exists to prevent.
        """
        if not self.normalize_kernel_embedding:
            return H
        return F.normalize(H, p=2, dim=-1)

    def _squared_euclidean_dist(self, x, y):
        # x: (N, D), y: (M, D)
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)  # (N, 1)
        y_norm = (y ** 2).sum(dim=1).view(1, -1)  # (1, M)
        dist_sq = x_norm + y_norm - 2.0 * (x @ y.T)
        return torch.clamp(dist_sq, min=0.0)

    def _compute_sigma(self, x, y):
        combined = torch.cat([x, y], dim=0)
        dist_sq = self._squared_euclidean_dist(combined, combined)
        # Elimina diagonal (distancia 0) para evitar sesgo
        non_diag = dist_sq[~torch.eye(dist_sq.shape[0], dtype=bool, device=dist_sq.device)]
        return torch.sqrt(torch.median(non_diag) + 1e-6)

    def _median_pairwise(self, X):
        """`_compute_sigma`'s heuristic over ONE set instead of a pooled pair.

        `_compute_sigma(X, X)` would not do: it concatenates, so every point
        would appear twice and the duplicate pairs contribute a distance of
        zero to the median. A test pins this against `_compute_sigma` on a
        concatenation, which is the only way to know the heuristic is the same.
        """
        dist_sq = self._squared_euclidean_dist(X, X)
        non_diag = dist_sq[~torch.eye(dist_sq.shape[0], dtype=bool, device=dist_sq.device)]
        return torch.sqrt(torch.median(non_diag) + 1e-6)

    def _sigma_for(self, f_s_c, f_t_c, f_s_batch, f_t_batch):
        """The bandwidth, under whichever scope `sigma_from` names.

        **Why the scope is a fork at all.** `class-pooled` -- today's behaviour
        and the default -- reads the TARGET's features, which are exactly what
        the optimizer is moving, so the bandwidth is a function of the thing
        being aligned. Measured on the toy transfers over 800 steps of the
        free regime, the pooled median drifts by a factor of 2.65 to 3.36 while
        the source-only median drifts 1.03 to 1.06: the source encoder is
        frozen, so its median varies only with batch sampling. Whether the
        feedback loop helps or hurts is NOT settled -- it moves the trajectory's
        shape rather than its attainable level, and which scope reads higher
        depends on the step -- so this is a scope a caller chooses, not a
        default this changes.
        """
        if self.sigma != 'auto':
            return self.sigma
        if self.sigma_from == 'class-pooled':
            return self._compute_sigma(f_s_c, f_t_c)
        if self.sigma_from == 'class-source':
            return self._median_pairwise(f_s_c)
        if self.sigma_from == 'batch-pooled':
            return self._compute_sigma(f_s_batch, f_t_batch)
        return self._median_pairwise(f_s_batch)

    def _gaussian_kernel(self, x, y, sigma_val):
        dist_sq = self._squared_euclidean_dist(x, y)
        return torch.exp(-dist_sq / (2 * sigma_val ** 2))


    def _renyi_entropy_order_2(self, K):
        """Ec. (10): `H_2(A) = -ln[tr(A^T A)] = -ln||A||_F^2`, sobre la matriz
        normalizada por traza de la Ec. (9).

        **Una sola forma, y el 2026-10-18 dejo de haber dos.** Habia una rama
        optativa, `unbiased_purity`, que estimaba la pureza como estadistico U:
        saca los pares `j = k` del numerador Y del denominador. Se retiro por
        tres razones acumuladas.

        **No esta en la formulacion.** r34 declara UNA entropia, la Ec. (10),
        escrita con el mismo `tr(A^T A)` que la linea de abajo. Un estadistico U
        de la pureza no aparece en el documento, y la declaracion del toyset lo
        rotulaba `# Instrumento`.

        **Nunca corrio.** `PUREZA_U` valia `False` en la declaracion y `False`
        en cada corrida que el cuaderno ejecutado registra, una por una. Nada lo
        buscaba y nada lo ponia en un espacio de busqueda.

        **Y estaba refutado donde se podia pedir.** Medido sobre 600
        configuraciones: con esta forma, 0 de 600 fuera de `[0, 1]`; con el
        estadistico U, 109 de 600, en un rango de `-0.7631` a `1.0517`. Una
        guarda lo rechazaba junto con `bounds_form='entropy-cosh'`, que es el
        unico valor admisible, de modo que lo unico alcanzable era con la
        normalizacion por cotas apagada. Rederivar los extremos de las
        Ecs. (35)-(36) para ese estimador es deliberacion y no una bandera.

        Su cuenta llevaba un cuarto epsilon, `+1e-6` dentro del logaritmo, y era
        el UNICO de los cuatro que tapaba un caso alcanzable: el estadistico U
        puede salir no positivo de verdad --- medido, 139 de 400 veces con dos
        puntos y 54 de 400 con cuarenta --- y ahi el logaritmo no esta definido.
        Se va con la rama que lo necesitaba.
        """
        # La traza VERDADERA, sin epsilon, y una negativa en vez de un
        # amortiguador. Es la forma de `MIL_CREDA.renyi.trace_normalize`, y la
        # razon es medida y no de estilo: `trace(K) + 1e-6` suma una constante
        # a una cantidad que puede ser mucho menor que ella, asi que en cuanto
        # `tr(K)` se acerca a `1e-6` el epsilon ES el normalizador y la
        # entropia reportada es la del epsilon. Sobre un bloque de rango
        # completo en float32, entropia exacta 1.14525202:
        #
        #     tr(K) = 1e-06  ->   2.53153396
        #     tr(K) = 1e-09  ->  13.53976822
        #     tr(K) = 1e-20  ->  13.81551075   saturada en -log(1e-6)
        #
        # Y la ultima no es solo erronea: una matriz de densidad 5x5 no puede
        # llevar mas de `ln 5 = 1.609` nats, de modo que el estimador se salia
        # del rango de su propia cantidad. Esta forma es exacta a 7 digitos en
        # float32 desde `tr(K) = 1` hasta `1.175e-38`, el minimo normal.
        tr_K = torch.trace(K)
        if float(tr_K) <= 0.0:
            # La Ec. (9) solo define la entropia para traza positiva. El portal
            # de actividad (`MIN_CLASS_TRACE`) excluye esas clases antes de
            # llegar aca, asi que esto es el respaldo y no la via normal ---
            # pero refuta, porque un epsilon aqui devolveria un numero que es
            # del epsilon y la clase se reportaria con una entropia medida.
            # Misma disciplina que `MIL_CREDA.renyi.trace_normalize`: la
            # Ec. (9) no define la entropia ahi, y decirlo es mejor que
            # devolver un numero.
            raise ValueError(
                f'una matriz de Gram con traza no positiva no tiene '
                f'normalizacion: la Ec. (9) la exige positiva y esta vale '
                f'{float(tr_K):.3e}')
        K_norm = K / tr_K
        # Sin epsilon dentro del log tampoco, y esta vez no hace falta ninguno:
        # para `A` PSD de traza uno con autovalores `l_i >= 0` que suman uno,
        # Cauchy-Schwarz da `tr(A^2) = sum l_i^2 >= 1/n > 0`. El epsilon no
        # protegia de un cero alcanzable y si sesgaba el resultado --- es el
        # 1.75e-06 de error relativo que quedaba incluso a traza uno.
        return -torch.log(torch.trace(K_norm.T @ K_norm))

    def _mix_kernel_concat(self, K_s, K_t, K_st):
        """The mixed kernel, with `balance_mix_blocks` governing whose it is.

        With the flag off this is the plain concatenation, byte for byte as
        before the fork existed.

        **What the flag repairs.** `_renyi_entropy_order_2` divides by the trace,
        which makes it invariant to the kernel's SCALE. But a Gaussian kernel has
        1 on its diagonal, so `tr(K_mix) = n_s + n_t`, and dividing the whole
        matrix by that total is a single scalar -- a scalar cannot change what
        FRACTION of the trace each block owns. At `n_s = 43` against `n_t = 400`
        the target block owns 90.3% of it, its leading eigenvalue over the trace
        reaches 0.903, and `h_mix` becomes the entropy of that block alone.

        That is the damage, because the overlap between the two domains lives
        ONLY in `K_st`, whose contribution to the numerator goes to zero -- 98.9%
        of the measured value is explained by `K_t`. With one block dominating,
        the score reduces to `(H(K_t) - H(K_s))/2` with `H(K_s)` frozen: the term
        stops asking "align" and asks "collapse the target".

        Measured on a clump held at a fixed distance from the source, never
        overlapping it: the score ran from +1.0569 at `n_t = 10` to -0.7420 at
        `n_t = 400`, crossing the value of PERFECT alignment near `n_t = 2 n_s`,
        so a large enough misaligned clump reads BETTER than a perfect match.
        Balanced it stays at 0.3010 across a 40x range of cardinality and the
        clump never reads better than alignment.

        **Why the cross block takes `sqrt(n_s n_t)`.** It is not a choice; the
        congruence fixes it. The repair is `D K_mix D` with
        `D = diag(I/sqrt(n_s), I/sqrt(n_t))`, which scales the diagonal blocks by
        `1/n_s` and `1/n_t` and the cross block by `1/sqrt(n_s n_t)`, all three at
        once. A congruence preserves positive semidefiniteness exactly. Dividing
        only the diagonal blocks and leaving `K_st` raw is NOT a congruence:
        measured, the smallest eigenvalue falls to -14.3 and two COINCIDENT
        clouds read -6.83 instead of 0, so its "entropy" is not an entropy.

        **Only the mix needs it.** `h_s` and `h_t` are each taken over their own
        matrix, where the trace division already absorbs any scalar: measured,
        `H2(K/n) - H2(K)` is 1e-6, the epsilon inside that function. So this fork
        is one method and not three.

        **`balance_mass` chooses WHICH mass is equalized**, and r28 Eq. (32)
        says trace rather than cardinality. The congruence is the same shape
        either way -- `D K_mix D` with `D = diag(a^-1/2 I, b^-1/2 I)` -- so
        positive semidefiniteness survives both; what changes is `a` and `b`.
        With `cardinality` they are `n_s` and `n_t`. With `trace` they are
        `tr(K_s)` and `tr(K~_t)`.

        On the source side that is the same number: the Gaussian kernel has 1
        on its diagonal, so `tr(K_s) = n_s` exactly. On the target side it is
        not, and this is the whole of the change. `K_t` arrives here ALREADY
        weighted -- the caller has applied `K_t * outer(w, w)`, which is
        `D_t K_t D_t` -- so its diagonal is `w_j^2` and
        `tr(K~_t) = sum_j w_j^2`. Dividing by `n_t` therefore gives the target
        domain the mass of its COUNT while r28 gives it the mass it earned:
        Eq. (32) equalizes `tr(K_s)` against `sum_j w_j^2 ||Psi(B_j)||^2`, and
        a sample with `w_j -> 0` is equivalent to withdrawing it. Under
        `cardinality` that withdrawal never reaches the mixture's proportions.
        Measured on a 129-sample batch with softmax pseudo-labels,
        `tr(K~_t)/n_t` ran 0.470 / 0.774 / 0.482 across the three classes
        while `tr(K_s)` was exactly `n_s` in all three, so this is a 20-55%
        misstatement of the target's mass and not a rounding difference.

        **Eq. (32)'s leading 1/2 is deliberately NOT written.** r28 multiplies
        the balanced block matrix by 1/2 so that its trace is exactly 1 and it
        is a valid density for Eq. (9). Here that factor is a no-op:
        `_renyi_entropy_order_2` divides by the trace of whatever it is handed,
        so it reaches r28's matrix either way, and the only observable effect
        of writing it would be to halve the matrix this method RETURNS. Leaving
        it out also keeps `trace` and `cardinality` bit-identical when every
        weight is 1, which is what makes "the new path is the same
        construction" an assertion with `torch.equal` in it rather than a
        tolerance.
        """
        if self.balance_mix_blocks:
            if self.balance_mass == 'trace':
                mass_s = torch.trace(K_s)
                mass_t = torch.trace(K_t)
                K_s = K_s / mass_s
                K_t = K_t / mass_t
                K_st = K_st / torch.sqrt(mass_s * mass_t)
            else:
                n_s, n_t = K_s.shape[0], K_t.shape[0]
                K_s = K_s / n_s
                K_t = K_t / n_t
                K_st = K_st / ((n_s * n_t) ** 0.5)
        return torch.cat([
            torch.cat([K_s, K_st], dim=1),
            torch.cat([K_st.T, K_t], dim=1)
        ], dim=0)

    def _marginal_score(self, f_s, f_t):
        """The same construction with the class index removed: ONE block.

        Deliberately a copy of the per-class loop's body rather than a
        refactor of it, so the default path's bytes do not move; a test pins
        the two equal on a batch whose every sample routes to one class, which
        is the configuration in which they must agree exactly.

        **`bounds_form` reaches here too**, and it has to: a run that asked for
        r31's entropy ends and got r25's cardinality ends on the marginal half
        of a mixture would be adding two numbers normalized by different maps,
        and the sum would still land in [0, 1] and look fine.

        **r28 §5's active-class rule does NOT reach here, and that is not an
        omission.** The rule is about CLASSES -- it says which members of
        `C_act` the global term aggregates over. This block belongs to no
        class; it is the whole batch with the class index removed. A collapsed
        trace here would mean every pseudo-label in the batch was uniform, at
        which point there is no per-class decomposition to exclude anything
        from either.
        """
        sigma_val = self._sigma_for(f_s, f_t, f_s, f_t)
        K_s = self._gaussian_kernel(f_s, f_s, sigma_val)
        K_t = self._gaussian_kernel(f_t, f_t, sigma_val)
        K_st = self._gaussian_kernel(f_s, f_t, sigma_val)
        # **No confidence weighting here, and that is r33's own sentence.**
        # Until 2026-10-18 this block applied Eq. (26)'s `D` -- `D K_t D` and
        # `K_st D` -- under `use_entropy_weighting`, like the per-class blocks
        # above. r33 scopes that weighting: "La ponderacion de confianza se
        # aplica a todo bloque condicional por clase que involucre bolsas
        # objetivo (Ec. 26 y Ec. 27)". This block is the whole batch with the
        # class index REMOVED, so it is not a conditional block and Eq. (26)
        # does not reach it. `target_weights` therefore left this method's
        # signature rather than being ignored inside it, so no caller can hand
        # it something this block would silently drop.
        K_mix = self._mix_kernel_concat(K_s, K_t, K_st)
        # Hoisted into names rather than written inline as before: Eq. (35)'s
        # upper end is a function of these same two entropies, so the bounds
        # need them separately. Same operations in the same order, so the score
        # is the same bits it was.
        h_s = self._renyi_entropy_order_2(K_s)
        h_t = self._renyi_entropy_order_2(K_t)
        score = self._renyi_entropy_order_2(K_mix) - 0.5 * (h_s + h_t)
        if self.normalize_by_bounds:
            return class_global_loss_cosh(-score, h_s, h_t)
        return score

    def forward(self, f_s, f_t, y_s, g_t, reduction='mean',
                target_weights=None):
        # Fork 1: the view the kernels (and the bandwidth derived for them) read.
        # Identity when the flag is off -- see `_kernel_embedding`.
        f_s = self._kernel_embedding(f_s)
        f_t = self._kernel_embedding(f_t)

        y_t_pseudo = torch.argmax(g_t, dim=1)
        unique_classes = torch.unique(y_s, sorted=True)
        losses_per_class = []
        # Fork 3's per-class normalizers, in the SAME order as
        # `losses_per_class`: one entry appended on exactly the iterations that
        # append a loss, which is what the length assertion after the loop
        # checks. A class ordering that desynchronized from the losses would
        # apply one class's bounds to another class's score and the number
        # would still look perfectly plausible.
        #
        # WHAT each entry holds depends on `bounds_form`, and the two are not
        # interchangeable: under `conservative` it is r25's `(lower, upper)`,
        # two floats derived from the class's two cardinalities, and under
        # `entropy-cosh` it is `(h_s, h_t)`, the class's two marginal
        # entropies as tensors that still carry a gradient. Which it is, is
        # read once after the loop and never guessed from the contents.
        bounds_per_class = []
        valid_class_count = 0
        # r33 Eq. (37)'s weights and its charge for an abandoned class.
        # `unique_classes` already runs over exactly the classes with
        # `n_c^s > 0`, so a class absent from the source batch weighs
        # `0 / N_s` and is correctly never seen here. `N_s` is every source
        # sample, because each one belongs to one of those classes.
        active_source_counts = []
        inactive_source_mass = 0

        # A weight vector the caller supplies, opt-in and default-identical.
        #
        # **Why it has to exist at all.** For a BINARY weight, selecting the
        # rows and multiplying are the same operation -- a zero contributes
        # neither to `tr(M)` nor to `tr(M^2)` -- so a ceiling arm can run by
        # subsetting `f_t` before calling this, without touching this module.
        # For a CONTINUOUS weight it cannot: a SHUFFLED weight needs the same
        # VALUES on different points, and no choice of subset expresses that.
        #
        # And that shuffled control is what turns a comparison into a test of
        # MECHANISM: if the declared weight and the same weights randomly
        # reassigned reach the same place, the weight is selecting nothing and
        # only its distribution matters. Without this parameter that question
        # cannot be asked, so every null measured so far is compatible with it.
        #
        # `None` keeps the published behaviour bit for bit.
        if target_weights is not None:
            # Refused rather than ignored: a caller handing weights to an arm
            # that would discard them believes it is running something it is
            # not, and the number it reports is the unweighted one under a
            # weighted label.
            if not self.use_entropy_weighting:
                raise ValueError(
                    'target_weights was supplied but use_entropy_weighting is '
                    'False, so this arm would discard them and report an '
                    'unweighted result under a weighted name')
            if target_weights.shape[0] != g_t.shape[0]:
                # Never broadcast: a length that is not the batch's weights the
                # wrong points and the run looks perfectly healthy.
                raise ValueError(
                    f'target_weights has {target_weights.shape[0]} entries for '
                    f'{g_t.shape[0]} target samples')
            target_weights = target_weights.to(g_t.device)
        elif self.use_entropy_weighting:
            # Ec. (24): `w_j^t = 1 - H_2(g_j^t) / ln C`, con
            # `H_2(g) = -ln(sum_c g_c^2)`.
            #
            # **Sin epsilon, y por la misma razon que `_renyi_entropy_order_2`
            # perdio los suyos: no hay cero del que guardar.** La Ec. (20)
            # garantiza que `g_j^t` es una distribucion --- su ultima capa es
            # una normalizacion exponencial, "estrictamente positivas que
            # suman uno" --- de modo que `sum_c g_c^2` vive en `[1/C, 1]`.
            #
            # Lo que el `+1e-6` hacia era sesgar el peso JUSTO en el extremo.
            # `w` vale cero en la uniforme, que es la bolsa de la que el
            # clasificador no sabe nada y por eso no debe pesar; ahi
            # `sum g^2` toca su piso `1/C` y el epsilon pesa mas que en
            # ningun otro punto. Medido: daba `w = 2.89e-06` con C=2,
            # `2.73e-06` con C=3 y `4.34e-06` con C=10 --- un sesgo de
            # `~1e-6 * C / ln C`, que crece con el numero de clases. Un peso
            # que deberia anular una bolsa le dejaba peso.
            #
            # Y alimentaba el porton de `C_act` (Ec. 37), que mira
            # `tr(K~_c^t) = sum_j w_j^2`: un peso espurio ahi mueve la traza
            # que decide si la clase entra en la suma.
            squared_sum = torch.sum(g_t ** 2, dim=1)
            entropy = -torch.log(squared_sum)
            target_weights = 1.0 - entropy / torch.log(torch.tensor(float(g_t.shape[1]), device=g_t.device))
        else:
            target_weights = None

        for c in unique_classes:
            f_s_c = f_s[y_s == c]
            f_t_c = f_t[y_t_pseudo == c]

            if f_s_c.shape[0] == 0 or f_t_c.shape[0] == 0:
                # Not in C_act: Eq. (37) charges `l~_c = 1`, weighted by this
                # class's own `n_c^s`. r31 dropped the class from the sum
                # instead, so a routing that abandoned classes reported a
                # SMALLER term for it.
                inactive_source_mass += int(f_s_c.shape[0])
                continue

            sigma_val = self._sigma_for(f_s_c, f_t_c, f_s, f_t)

            K_s_c = self._gaussian_kernel(f_s_c, f_s_c, sigma_val)
            K_t_c = self._gaussian_kernel(f_t_c, f_t_c, sigma_val)
            K_st_c = self._gaussian_kernel(f_s_c, f_t_c, sigma_val)

            if self.use_entropy_weighting:
                weights_c = target_weights[y_t_pseudo == c]
                W_c = torch.outer(weights_c, weights_c)
                K_t_c = K_t_c * W_c
                if self.weight_cross_block:
                    # Fork 2: Eq. (26)'s `K~_c^{st} = K_c^{st} D`. `weights_c`
                    # has one entry per COLUMN of `K_st_c`, and unsqueezing on
                    # dim 0 broadcasts it across rows -- the right-hand
                    # multiplication by D, with the source side untouched.
                    K_st_c = K_st_c * weights_c.unsqueeze(0)

            # r28 section 5's SECOND condition on an active class. The counts
            # above are the first: "existen sujetos en ambos dominios". This is
            # the other half -- "las tres matrices que deben normalizarse
            # poseen traza positiva" -- and `forward` used to check only the
            # counts.
            #
            # It binds on the target. `tr(K_s_c)` is the class's source count,
            # because the Gaussian kernel has a unit diagonal; `tr(K_t_c)` is
            # `sum_j w_j^2` once the weighting above has run, so a class whose
            # every pseudo-label came out uniform has a trace of about zero
            # while still having samples. The third matrix is Eq. (27)'s
            # unbalanced mixture, whose trace is the sum of the other two --
            # written out rather than inferred, because that implication rests
            # on Eq. (27)'s block structure and not on anything checked here.
            #
            # Skipped and not clamped: r28 says the score is not defined for
            # such a class, and the epsilon inside `_renyi_entropy_order_2`
            # would otherwise turn the division into a finite number that is
            # the epsilon's and not the data's.
            trace_s = torch.trace(K_s_c)
            trace_t = torch.trace(K_t_c)
            if (trace_s <= MIN_CLASS_TRACE or trace_t <= MIN_CLASS_TRACE
                    or trace_s + trace_t <= MIN_CLASS_TRACE):
                # The other half of Eq. (37)'s membership, and so the other
                # place the charge of `1` is owed. Same reasoning as the count
                # check above.
                inactive_source_mass += int(f_s_c.shape[0])
                continue

            K_mix_c = self._mix_kernel_concat(K_s_c, K_t_c, K_st_c)

            h_s = self._renyi_entropy_order_2(K_s_c)
            h_t = self._renyi_entropy_order_2(K_t_c)
            h_mix = self._renyi_entropy_order_2(K_mix_c)
            # `h_t` sin gradiente apaga la fuerza dispersiva del bloque
            # objetivo sobre si mismo, dejando intacto todo lo demas --- `h_mix`
            # sigue llevando al objetivo adentro, asi que la alineacion no se
            # toca. Las COTAS leen `h_t` aparte y no por aca.
            h_t_score = h_t.detach() if self.detach_target_entropy else h_t
            creda_c = h_mix - 0.5 * (h_s + h_t_score)
            losses_per_class.append(creda_c)
            active_source_counts.append(int(f_s_c.shape[0]))
            if self.normalize_by_bounds:
                # Eq. (34)/(35)'s ends are functions of the two entropies this
                # class already computed, not of its counts -- which is exactly
                # the property that makes them reachable. r28 introduced them
                # and r33 still states `U_c` this way.
                bounds_per_class.append((h_s, h_t))
            valid_class_count += 1

        if valid_class_count == 0:
            # **Eq. (37) answers 1 here, not 0, and r31 answered 0.** No class
            # is active, so every `l~_c` is Eq. (37)'s charge and the weights
            # still sum to one. The per-class half is therefore DEFINED in this
            # case and equals exactly 1; r31 returned zero, which told a
            # collapsed encoder it had converged.
            #
            # `+ f_t[:0].sum()` y NO `torch.tensor(1.0)`, por la misma razon
            # por la que el retorno anterior era `f_t[:0].sum()` y no
            # `torch.tensor(0.0)`:
            #
            # Un escalar suelto no lleva `grad_fn`, y en el regimen declarado
            # (`mover_fuente=False`) la cross-entropy de la fuente llega
            # detachada: este termino es el UNICO camino de gradiente del paso.
            # Un valor sin grafo mata `backward()` con «element 0 of tensors
            # does not require grad», y «no hubo clase activa» no es «no hubo
            # paso»: Adam tiene que poder seguir moviendose por su momento.
            #
            # La suma de una tajada VACIA, y no `0 * f_t.sum()`: vale 0.0
            # exacto, su gradiente es cero exacto, y no depende del CONTENIDO
            # de `f_t` --- con un inf o un NaN adentro, el producto daria NaN y
            # el paso envenenaria los pesos en vez de no moverlos. Sumarle el 1
            # de la Ec. 37 no cambia nada de eso: el 1 es literal y no aporta
            # gradiente, que es exactamente lo que la ecuacion pide.
            #
            # El camino estaba vivo y nunca se recorria: el unico piso que
            # llegaba aca era `MIN_CLASS_TRACE = 1e-6`, que casi no dispara. Lo
            # destapo una medicion que puso un piso de CONTEO por bloque de
            # clase. Era alcanzable antes de ese piso igual --- basta con que
            # ninguna clase tenga muestras de los DOS dominios --- y hay un
            # test que lo demuestra sin tocar ningun piso.
            per_class = 1.0 + f_t[:0].sum()
            if reduction == 'none':
                # No row to report: `reduction='none'` decomposes by ACTIVE
                # class and there is none. Eq. (37)'s charge is not a per-class
                # reading, it is what the aggregation pays for their absence.
                return self.lambda_creda * f_t[:0].sum().reshape(0)
            if self.class_mixing < 1.0:
                # With mixing on, the marginal half is defined too, and
                # returning only one of them would silently drop a term the
                # caller asked for.
                alpha = self.class_mixing
                return self.lambda_creda * (
                    alpha * per_class
                    + (1.0 - alpha) * self._marginal_score(f_s, f_t))
            return self.lambda_creda * per_class

        if self.normalize_by_bounds:
            # Fork 3: Eqs. (35)-(36). `-creda_c` is J_c, the same quantity with
            # the opposite sign -- see the class docstring.
            assert len(bounds_per_class) == len(losses_per_class), (
                f"{len(bounds_per_class)} classes normalized against "
                f"{len(losses_per_class)} per-class losses: the bounds have "
                f"desynchronized from the scores they belong to")
            # r31's Eqs. (35)-(36). Same sign convention: `-creda_c` is
            # `J_c`.
            #
            # **2026-10-17: until today this comment asserted that the
            # denominator was excluded from the gradient inside
            # `class_global_loss_cosh`, and called that the bound
            # revision's own rule.** It was true of r29 and is now false of
            # the code it sits beside. r29's Eq. (36) divided by the
            # per-class span `U_c` minus `L_c` and excluded it from the
            # gradient by design; r30
            # proved the span of the corrected score is bounded by `ln 2`
            # regardless, so r31 divides by that literal constant and the
            # exception went with the denominator -- a `math.log(2.0)`
            # carries no gradient to remove. The history is kept rather
            # than deleted because the detachment WAS a deliberate rule and
            # a reader who remembers it needs to be told it ended, not left
            # to notice its absence.
            #
            # `U_c` keeps its gradient, in the numerator, where it is half
            # of the corrected score. What r31 removed is the path that
            # rewarded widening the entropy gap through the denominator.
            losses_per_class = [
                class_global_loss_cosh(-loss_c, h_s_c, h_t_c)
                for loss_c, (h_s_c, h_t_c) in zip(losses_per_class,
                                                  bounds_per_class)
            ]

        losses = torch.stack(losses_per_class)

        if reduction == 'none':
            if self.class_mixing < 1.0:
                raise ValueError(
                    "class_mixing below one has no per-class decomposition: "
                    "the marginal block belongs to no class, so "
                    "reduction='none' cannot report it")
            # The ACTIVE classes' `l_glob,c`, which is what a caller asking
            # for the decomposition wants to look at. It is not a summand list
            # for the reduced term: that one also carries Eq. (37)'s charge for
            # the classes this vector has no row for, so its mean is not what
            # `reduction='mean'` returns.
            return self.lambda_creda * losses

        per_class = self._eq37_aggregate(
            losses, active_source_counts, inactive_source_mass, f_s.shape[0])
        if self.class_mixing < 1.0:
            # The convex family. `class_mixing=1` is the declared per-class
            # term and the default; `0` is the marginal block alone. Both ends
            # live in [0, 1] when `normalize_by_bounds` is on, so the mixture
            # inherits that bound with nothing to prove.
            alpha = self.class_mixing
            return self.lambda_creda * (
                alpha * per_class
                + (1.0 - alpha) * self._marginal_score(f_s, f_t))
        return self.lambda_creda * per_class

    @staticmethod
    def _eq37_aggregate(losses, active_counts, inactive_mass, n_source):
        """Implement Eq. (37) at sample level: `sum_c (n_c^s / N_s) l~_c`.

        `losses` holds `l_glob,c` for the active classes and `active_counts`
        their `n_c^s` in the same order; `inactive_mass` is the total `n_c^s`
        of the classes the target routing abandoned, each of which contributes
        `l~_c = 1`; `n_source` is `N_s`.

        **This was `losses.mean()` until 2026-10-18**, which is r31's Eq. (37)
        read at sample level: a uniform average over the ACTIVE classes alone.
        The two agree exactly when every class is active and the source batch
        is balanced -- then `n_c^s / N_s` is `1 / |C|`, which is what the mean
        computes -- and part company in both of the cases r32 wrote the weights
        for. An unbalanced source batch now weighs each class by the evidence
        it actually brought, from OBSERVED labels; and a class the routing
        abandoned raises the term instead of leaving it.

        The charge carries no gradient, and nothing detaches it: the routing
        comes from an `argmax` (Eq. 22), so there is no path to differentiate,
        and the charge is a literal. Its job is the VALUE -- collapse visible
        in the loss rather than rewarded by it.
        """
        if n_source <= 0:
            raise ValueError("N_s is zero: Eq. (37)'s weights are undefined")
        # `(sum_c n_c^s l_c + inactive_mass) / N_s`, and NOT
        # `sum_c (n_c^s / N_s) * l_c`. The two are the same equation and not the
        # same arithmetic: forming each weight first multiplies by an inexact
        # reciprocal and rounds once per class, which moved the pinned default
        # mean by 6e-08 in float32 -- enough to redden
        # `test_the_default_path_still_reads_the_pinned_values` at its ten
        # places while nothing mathematical had changed. Integer counts are
        # exact in float32 and there is one division, so where every class is
        # active and the batch balanced this reproduces `losses.mean()` bit for
        # bit, which is what that pin is there to detect.
        counts = torch.tensor(
            active_counts, dtype=losses.dtype, device=losses.device)
        return (counts @ losses + inactive_mass) / n_source
