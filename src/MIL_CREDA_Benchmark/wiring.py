"""The arms declared in `config.ARMS`: what each one computes, and nothing
about how it is trained.

Every arm shares the same encoder, the same linear head, the same data, the same
schedules and the same optimizer. What separates them is written in `config.ARMS`
and read here: which statistical unit the arm trains on, which adaptation term it
carries, whether the target blocks are weighted by confidence, whether the local
correspondence is on, how a bag becomes a representation, and which of a bag's
instances the arm is allowed to look at.

`FeatureExtractor` and `CREDALoss` are used exactly as they are; the only thing
this file decides is which tensors go in.

Prior work carries one change, and it is a reparameterization rather than a
change of method: CREDA's adaptation coefficient moved out of
`CREDALoss.lambda_creda` and into the ceiling of `CREDA.schedules.creda_ramp`, so both
families of this comparison drive one implementation instead of two copies of the
same formula. `get_lambda` is untouched and still serves DANN, ADDA and CDAN+E at
full strength — scaling their coefficient would switch CDAN+E's gradient-reversal
hook off rather than weaken it. `tests/test_creda_schedule.py` holds the new
composition against a golden frozen before the move.

The head is built here rather than taken from `CREDA_ResNet` because that class
calls `set_seed(42)` in its constructor, which would reset the global generator
in the middle of a seeded run and make every repetition identical in the parts
that matter least. It is the same `nn.Linear(512, C)` either way.

The mathematics stays where it is: every equation is called from `MIL_CREDA`,
which is bound to the revision. This file owns the learnable parameters and the
bookkeeping, and not one formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from CREDA.models import CREDALoss, FeatureExtractor
from MIL_CREDA.attention import bag_embedding, bag_weights, relevance_logits
from MIL_CREDA.bag_kernel import bag_kernel, bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import confidences, pseudolabel
from MIL_CREDA.global_term import (
    class_global_loss,
    conservative_bounds,
    dependency_score,
    global_loss,
    is_active,
)
from MIL_CREDA.local_term import local_distance, local_loss, total_correspondence
from MIL_CREDA.objective import source_loss, total_objective
from MIL_CREDA_Benchmark import config


@dataclass
class Pool:
    """One domain's training bags, already on the device the run uses."""

    images: torch.Tensor      # (n_images, 3, 32, 32)
    members: torch.Tensor     # (n_bags, 30) indices into `images`
    labels: torch.Tensor      # (n_bags,) observed classes; target labels are never read

    def take(self, positions: torch.Tensor) -> torch.Tensor:
        """The instances of the chosen bags, as (B, 30, 3, 32, 32)."""
        return self.images[self.members[positions]]


class Arm(nn.Module):
    """One row of the ladder, trainable end to end."""

    def __init__(self, spec: dict, classes: int, source: Pool, target: Pool,
                hyper: dict | None = None):
        super().__init__()
        self.spec = spec
        self.classes = classes
        self.source = source
        self.target = target

        # Decision 1's bandwidth and Eq. (15)/(16)/(28)'s three hyperparameters,
        # resolved once at construction rather than read off `config` at every
        # call site. `hyper` is the ceiling search's own override -- every key
        # absent from it (or `hyper is None`, the ordinary campaign run before a
        # search wires its winner in) falls back to the declared constant, so a
        # caller that never searches trains exactly as before.
        hyper = hyper or {}
        self.sigma = hyper.get("kernelSigma", config.KERNEL_SIGMA)
        self.attention_gamma = hyper.get("attentionGamma", config.ATTENTION_GAMMA)
        self.attention_temperature = hyper.get(
            "attentionTemperature", config.ATTENTION_TEMPERATURE)
        self.tau_local = hyper.get("tauLocal", config.TAU_LOCAL)

        self.encoder = FeatureExtractor(backbone=config.BACKBONE, pretrained=config.PRETRAINED)
        self.head = nn.Linear(self.encoder.output_dim, classes)

        width = self.encoder.output_dim
        if spec["attention"] == "learned":
            # The parameters of Eq. (15)'s R_phi. The equation itself, and its
            # ell_1-ball reparametrization of v_R, is applied by
            # `MIL_CREDA.attention`; only its raw, unconstrained weights live
            # here.
            self.V_R = nn.Parameter(torch.empty(config.ATTENTION_WIDTH, width))
            self.b_R = nn.Parameter(torch.zeros(config.ATTENTION_WIDTH))
            self.v_R = nn.Parameter(torch.empty(config.ATTENTION_WIDTH))
            nn.init.xavier_uniform_(self.V_R)
            nn.init.normal_(self.v_R, std=0.1)
        # Selection arms (SU, SA, SK) are removed. Every declared arm's
        # `spec["selection"]` is `None`, so `select` always returns every
        # instance of a bag unchanged; this branch stays only as the
        # construction that would feed a future arm declaring one again.

        if spec["adaptation"] == "creda":
            # lambda_creda is one because the coefficient is applied outside, from
            # the shared schedule; leaving it here as well would apply it twice.
            # No declared arm (B, E, F, G, SU, SA, SK) carries "creda" here as
            # `spec["adaptation"]` -- this is this benchmark's own harness code,
            # not `src/CREDA/`'s prior work, and no declared arm reaches this
            # branch today. It is kept rather than deleted because a future arm
            # could still declare `"adaptation": "creda"` and land here.
            self.creda = CREDALoss(sigma="auto", lambda_creda=1.0,
                                   use_entropy_weighting=spec["weighting"])

    # ------------------------------------------------------------ representation

    def instance_embeddings(self, bags: torch.Tensor) -> torch.Tensor:
        """Encode every instance of every bag: (B, m, ...) -> (B, m, d).

        The encoder never sees a bag. It is applied to each instance separately,
        as Eq. (13) states, and identically in both units.
        """
        B, m = bags.shape[0], bags.shape[1]
        flat = self.encoder(bags.reshape(B * m, *bags.shape[2:]))
        return flat.reshape(B, m, -1)

    def weights_for(self, H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """The in-bag weights beta of Eq. (16), over whatever instances survive.

        `sigma` is Decision 1's one constant bandwidth -- `self.sigma` at every
        call site, resolved once at construction from the search's override or
        `config.KERNEL_SIGMA` -- passed explicitly and with no default, so a
        caller supplying a different value is a visible, deliberate choice
        rather than something this method could quietly default to.
        """
        if self.spec["attention"] == "learned":
            logits = relevance_logits(
                H, self.V_R, self.b_R, self.v_R,
                self.attention_gamma, sigma,
            )
            return bag_weights(logits, self.attention_temperature)
        # uniform: Eq. (19) with beta = 1/m
        return torch.full((H.shape[0],), 1.0 / H.shape[0],
                          dtype=H.dtype, device=H.device)

    def select(self, H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """The instances of ONE bag this arm is allowed to look at: (m, d) -> (k, d).

        Every declared arm keeps every instance of a bag: the selection arms
        (`SU`, `SA`, `SK`) that once budgeted this down to `SELECT_K` are
        removed, and no arm this package declares sets `spec["selection"]` to
        anything but `None`. The method stays -- selection happens here and
        nowhere else, so the kernels, the attention, the bag representation and
        the decision at evaluation would all see the same instances if a future
        arm declared a rule again -- but the budgeted rules themselves
        (`"regular"`, `"arbitrary"`, `"topk"`) are gone along with `SELECT_K`
        and `SELECTION_SEED`, which no longer exist in `config`.
        """
        rule = self.spec["selection"]
        if rule is not None:
            raise ValueError(
                f"unknown selection rule {rule!r}: every declared arm keeps "
                "every instance of a bag; the budgeted rules were removed "
                "with the selection arms"
            )
        return H

    def bags_of(self, embeddings: torch.Tensor, sigma: float | torch.Tensor) -> list[tuple]:
        """Every bag as the (instances, weights) pair the rest of the file consumes.

        `sigma` is threaded through rather than read from `config` here, so
        every caller of this method states which bandwidth it is using —
        Decision 1's whole point, applied at the one place that fans it out
        to both `select` and `weights_for`.
        """
        kept = [self.select(H, sigma) for H in embeddings]
        return [(H, self.weights_for(H, sigma)) for H in kept]

    def bag_representations(self, embeddings: torch.Tensor, sigma: float | torch.Tensor):
        """Eq. (19) for every bag, plus the weights that produced it."""
        pairs = self.bags_of(embeddings, sigma)
        Z = torch.stack([bag_embedding(H, w) for H, w in pairs])
        return Z, [w for _, w in pairs]

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """Bag-level scores, whatever unit the arm trains on.

        The comparison happens at the bag because it is the only unit where both
        families decide without inventing a rule. An instance-unit arm reaches it
        by averaging the class distributions of its 30 instances; the logarithm of
        that average is returned so cross-entropy over it is exactly the negative
        log-likelihood of the averaged decision.
        """
        embeddings = self.instance_embeddings(bags)
        if self.spec["unit"] == "bag":
            Z, _ = self.bag_representations(embeddings, self.sigma)
            return self.head(Z)
        probabilities = F.softmax(self.head(embeddings), dim=-1).mean(dim=1)
        return torch.log(probabilities + config.EPSILON)

    # ------------------------------------------------------------- adaptation

    def _draw_target(self, generator: torch.Generator) -> torch.Tensor:
        """The step's unlabelled target bags. Plain: nothing is stratified by a
        label the method is not allowed to see."""
        order = torch.randperm(self.target.members.shape[0], generator=generator)
        # The generator stays on the host so a run is reproducible whatever the
        # device is; the indices move to wherever the bags actually live.
        return order[: config.BAGS_PER_STEP].to(self.target.members.device)

    def _target_embeddings(self, target_bags: torch.Tensor) -> torch.Tensor:
        """Encode a target batch during training.

        Every adapted arm passes the target batch through the encoder exactly
        as it passes the source batch: normalization layers stay in training
        mode and learn from this forward like the rest of the model does. That
        is the whole of it, and this method exists as a named place rather than
        as a branch -- it is where a freeze would have to be reintroduced, and
        `tests/test_arm_objectives.py` points its guard at this exact name.

        `GN` is retired, and with it the one `spec["normalization"] ==
        "sourceBatch"` branch that used to stand here, plus the three helpers
        that served only it (`_source_embeddings`, `_encode_with_frozen_stats`,
        `_batchnorm_modules`). No declared arm normalizes its target forward
        with anything but the encoder's own training-mode statistics.
        """
        return self.instance_embeddings(target_bags)

    def _milcreda_term(self, H_s, source_labels, target_bags):
        """Eqs. (14), (16)-(20), (22)-(38): the global score, and the local
        correspondence if on.

        The source side is the step's own supervised batch, already encoded —
        which is what `train_creda` does with `feats_src`, so both families draw
        their source evidence the same way and the encoder runs once.

        The two terms are returned apart rather than summed, because Eq. (39)
        carries a coefficient for each and it is `total_objective` that combines
        them. Summing here would fold both into one number and leave this file
        deciding a weight the formulation states.
        """
        H_t = self._target_embeddings(target_bags)

        # Decision 1: one constant bandwidth for the whole method -- the
        # consensus term, the top-k ranking, every kernel block below and
        # `local_distance` all read `self.sigma` (`config.KERNEL_SIGMA`, or the
        # ceiling search's override) and nothing else. r21 l.715: "Un unico
        # ancho de banda sigma gobierna los tres bloques, ya que los tres
        # derivan del mismo kernel de instancia."
        sigma = self.sigma
        bags_s = self.bags_of(H_s, sigma)
        bags_t = self.bags_of(H_t, sigma)

        K_ss = bag_kernel_matrix(bags_s, bags_s, sigma)
        K_st = bag_kernel_matrix(bags_s, bags_t, sigma)
        K_tt = bag_kernel_matrix(bags_t, bags_t, sigma)

        Z_t = torch.stack([bag_embedding(H, w) for H, w in bags_t])
        G_t = F.softmax(self.head(Z_t), dim=1)                    # Eq. (20)
        w_t = confidences(G_t)                                    # Eq. (24)
        pseudo = torch.tensor([pseudolabel(g) for g in G_t],      # Eq. (22)
                              device=K_ss.device)

        losses = []
        for class_id in range(self.classes):
            s_idx = torch.nonzero(source_labels == class_id, as_tuple=False).reshape(-1)
            t_idx = torch.nonzero(pseudo == class_id, as_tuple=False).reshape(-1)
            if s_idx.numel() == 0 or t_idx.numel() == 0:
                continue                                          # excluded by Eq. (37)
            K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, s_idx, t_idx)
            if self.spec["weighting"]:
                K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w_t[t_idx])
            else:
                K_st_w, K_t_w = K_st_c, K_t_c
            K_mix = mixed_matrix(K_s_c, K_st_w, K_t_w)
            if not is_active(K_s_c, K_t_w, K_mix):
                continue
            score = dependency_score(K_s_c, K_t_w, K_mix)
            lower, upper = conservative_bounds(int(s_idx.numel()), int(t_idx.numel()))
            losses.append(class_global_loss(score, lower, upper))

        global_term = global_loss(losses)
        local_term = torch.zeros_like(global_term)

        if self.spec["local"]:
            distances = []
            for column in range(len(bags_t)):
                cross = K_st[:, column]
                pi = total_correspondence(cross, source_labels, G_t[column],
                                          self.tau_local)
                H_j, w_j = bags_t[column]
                distances.append(
                    local_distance(bag_kernel(H_j, w_j, H_j, w_j, sigma), cross, K_ss, pi)
                )
            local_term = local_loss(torch.stack(distances), w_t, config.EPSILON_LOCAL)

        return global_term, local_term

    def _creda_term(self, H_s, source_labels, target_bags):
        """CREDA's own loss, over instances, exactly as it was written."""
        instances = H_s.reshape(-1, self.encoder.output_dim)
        H_t = self._target_embeddings(target_bags).reshape(-1, self.encoder.output_dim)
        y_s = source_labels.repeat_interleave(H_s.shape[1])
        g_t = F.softmax(self.head(H_t), dim=1)
        return self.creda(instances, H_t, y_s, g_t)

    # ------------------------------------------------------------------- step

    def training_step(self, bags: torch.Tensor, labels: torch.Tensor,
                      ramp: float, generator: torch.Generator) -> dict:
        """The arm's own objective, and nothing the arm does not have.

        The supervised term is the arm's. A bag-unit arm calls `source_loss`,
        which is Eq. (21) as the revision states it, normalized by its own
        supremum B_src; an instance-unit arm keeps CREDA's per-instance
        cross-entropy, because prior work is used exactly as it was written. The
        adaptation term is added with the shared coefficient — the same ramp and
        the same constant for every arm that has one — so nothing separates the
        arms except the term itself.

        The two sides' supervised terms are therefore on different numeric
        scales: MIL-CREDA's lands in [0, 1) and CREDA's does not. That asymmetry
        is the formulation's, not the harness's, and it is reported rather than
        removed — un-normalizing this side to make the two look alike would
        delete the very thing the comparison exists to show.
        """
        embeddings = self.instance_embeddings(bags)
        if self.spec["unit"] == "bag":
            Z, _ = self.bag_representations(embeddings, self.sigma)
            logits = self.head(Z)
            supervised = source_loss(                                 # Eq. (21)
                F.softmax(logits, dim=1),
                F.one_hot(labels, self.classes).to(logits.dtype),
                config.EPSILON,
            )
        else:
            instance_logits = self.head(embeddings)
            per_instance = labels.repeat_interleave(bags.shape[1])
            supervised = F.cross_entropy(
                instance_logits.reshape(-1, self.classes), per_instance
            )
            logits = torch.log(F.softmax(instance_logits, dim=-1).mean(dim=1)
                               + config.EPSILON)

        # Decision 2: a floor never lets a target image reach the encoder,
        # ever, in training. `_draw_target` is still called for every arm,
        # floor included, so the generator is consumed identically across
        # arms (SKILL.md: arms must not differ in how much of the generator
        # they consume) -- what a floor never does with the indices it draws
        # is `take` or encode the images they name.
        target_indices = self._draw_target(generator)
        coefficient = ramp
        adaptation = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.spec["adaptation"] == "milcreda":
            target_bags = self.target.take(target_indices)
            global_term, local_term = self._milcreda_term(
                embeddings, labels, target_bags
            )
            adaptation = global_term + local_term
            total = total_objective(                                  # Eq. (39)
                supervised, global_term, local_term, coefficient, coefficient
            )
        elif self.spec["adaptation"] == "creda":
            # CREDA's objective, not Eq. (39): one term, one coefficient, as its
            # own code writes it.
            target_bags = self.target.take(target_indices)
            adaptation = self._creda_term(embeddings, labels, target_bags)
            total = supervised + coefficient * adaptation
        else:
            # The floor: no target image passes through the encoder, ever, in
            # training (Decision 2). `target_indices` is drawn above and
            # deliberately unused past this point.
            total = supervised

        return {
            "logits": logits,
            "loss": total,
            # Reported as its own column: if the coefficient scaled the term to
            # irrelevance, the run is talking about the scale and not the methods,
            # and that has to be visible rather than inferred.
            "supervised": float(supervised.detach()),
            "adaptation": float(adaptation.detach()),
            "contribution": float((coefficient * adaptation).detach()),
        }


# ------------------------------------------------------- attention mechanisms
#
# Section 4's comparison ("which attention mechanism") and nothing a declared
# arm reads: `config.ARMS`'s own `G` keeps Eq. (15)/(16) exactly as `Arm`
# above computes them. Everything below is comparison-only plumbing -- it
# carries no `__provenance__` and implements no numbered equation of the
# revision, the same reason `harness`/`tables` do not either.
#
# Five mechanisms, on the full method (`G`) and nothing else: ours (Eq. (15)'s
# hybrid, exactly `Arm.weights_for`'s own "learned" branch), ABMIL as
# published (Ilse, Tomczak & Welling 2018 -- `v_R` unconstrained, no
# consensus term), ABMIL with a gating unit (the same paper's gated variant),
# max pooling read at the INSTANCE level (the bag is the instance holding its
# largest activation, entire -- see `_max_weights` for why the coordinatewise
# reading cannot be one of these five), and mean pooling (Eq. (16) with uniform
# weights, already `Arm.weights_for`'s own "uniform" branch under a different
# name).
MECHANISMS = ("ours", "abmil-published", "abmil-gated", "max", "mean")


def _abmil_published_logits(H: torch.Tensor, V_R: torch.Tensor, b_R: torch.Tensor,
                            v_R: torch.Tensor) -> torch.Tensor:
    """ABMIL as published: a_k = w^T tanh(V h_k^T), read here as
    `R(h) = v_R^T tanh(V_R h + b_R)` per instance -- `Arm`'s own
    `relevance_component` (Eq. (15)'s `R_phi`) with two differences, both
    deliberate: `v_R` is used RAW, never passed through the l1-ball
    reparametrization `_l1_ball_reparametrization` applies, and there is no
    bias-free variant either way -- the published gate has none, so `b_R`
    reproduces exactly what the paper's affine layer already has.
    """
    hidden = torch.tanh(H @ V_R.transpose(0, 1) + b_R)
    return hidden @ v_R.reshape(-1)


def _abmil_gated_logits(H: torch.Tensor, V: torch.Tensor, U: torch.Tensor,
                        w: torch.Tensor) -> torch.Tensor:
    """ABMIL's gated attention: a_k = w^T (tanh(V h_k^T) (dot) sigm(U h_k^T)),
    per instance -- the same paper's gated variant, own `V`/`U`/`w`, never
    `Arm`'s `V_R`/`b_R`/`v_R`: the gate is a product of two projections, not
    one affine layer, so it cannot share parameters with either the hybrid or
    the published mechanism above without secretly becoming a fourth thing.
    """
    gate = torch.tanh(H @ V.transpose(0, 1)) * torch.sigmoid(H @ U.transpose(0, 1))
    return gate @ w.reshape(-1)


def _max_weights(H: torch.Tensor) -> torch.Tensor:
    """Max pooling, instance-level: `beta` is one on the winning instance and
    zero on every other, where the winner is the instance holding the largest
    activation in the whole bag -- `argmax_a (max_j H[a, j])`.

    This is the MIL "max operator" reading and NOT the coordinatewise maximum
    `H.max(dim=0).values`, and the difference is what makes this mechanism
    expressible at all. The coordinatewise maximum takes each of the `d`
    coordinates from whichever instance happens to win it, so the vector it
    builds is generally no instance of the bag and lies OUTSIDE their convex
    hull -- it dominates every `h_a` in every coordinate. Eq. (19)'s `z = sum_a
    beta_a h_a` with `beta >= 0` summing to one is a convex combination, so it
    is always INSIDE that hull: no weight vector whatsoever reproduces the
    coordinatewise maximum, and Eq. (18)'s bag kernel, which weights instance
    kernels pairwise by that same `beta`, has nothing to be handed.

    Under this reading both consumers see the same instance: `z = h_{a*}` for
    the head, `beta = one-hot(a*)` for the kernel, neither approximated. The
    ranking is by the largest single activation rather than by the norm because
    that is what max pooling asserts -- one strong activation carries the bag --
    and ranking by norm would slide the mechanism toward `mean`, which is the
    other arm of this comparison.

    The gradient is the gradient max pooling already had: `beta` is a constant
    here (a hard `argmax` is not differentiable in its selection), so Eq. (19)
    passes gradient to the winning row alone.
    """
    scores = H.max(dim=1).values
    weights = torch.zeros(H.shape[0], dtype=H.dtype, device=H.device)
    weights[int(torch.argmax(scores))] = 1.0
    return weights


def _mean_weights(H: torch.Tensor) -> torch.Tensor:
    """Mean pooling: beta_a = 1/m for every instance -- Eq. (16) with uniform
    weights, exactly `Arm.weights_for`'s own "uniform" branch, named for this
    comparison rather than for a declared arm's `spec["attention"]`.
    """
    m = H.shape[0]
    return torch.full((m,), 1.0 / m, dtype=H.dtype, device=H.device)


def mechanism_weights(mechanism: str, H: torch.Tensor, params: dict) -> torch.Tensor:
    """The per-instance weights `mechanism` assigns to one bag's embeddings `H`.

    Every mechanism has them, `"max"` included -- see `_max_weights` for why
    that is a statement about WHICH max, not a convenience. So every one of
    them reduces to Eq. (19)'s own `bag_embedding(H, beta)`, and the bag kernel
    of Eq. (18) is handed a real `beta` in all five cases rather than a fallback
    in one of them.

    `params` carries exactly the parameters the named mechanism reads --
    `MechanismArm._params()` builds it -- so a caller mismatching a
    mechanism with the wrong parameter set fails on a missing key rather
    than silently mixing two mechanisms' weights.
    """
    if mechanism == "max":
        return _max_weights(H)
    if mechanism == "mean":
        return _mean_weights(H)
    if mechanism == "ours":
        logits = relevance_logits(H, params["V_R"], params["b_R"], params["v_R"],
                                  params["gamma"], params["sigma"])
    elif mechanism == "abmil-published":
        logits = _abmil_published_logits(H, params["V_R"], params["b_R"], params["v_R"])
    elif mechanism == "abmil-gated":
        logits = _abmil_gated_logits(H, params["V"], params["U"], params["w"])
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}; known: {MECHANISMS}")
    return bag_weights(logits, params["tau_att"])


class MechanismArm(Arm):
    """`G`'s full method -- encoder, head, the weighted global term, the local
    correspondence -- with Eq. (15)/(16)'s attention swapped for one of
    `MECHANISMS`. Every axis but pooling is exactly `G`'s, unchanged, by
    inheriting `Arm` rather than reimplementing beside it: a difference
    between two mechanisms is a difference of pooling and nothing else.

    Built by `build_mechanism` below, never by `wiring.build`: no entry in
    `config.ARMS` ever names one of `MECHANISMS`, so nothing here is
    reachable from a declared arm's own construction path.

    All five mechanisms reduce to Eq. (19) over their own weights, `"max"`
    included, so this class overrides `weights_for` and nothing else: the bag
    representation the head reads and the `beta` the bag kernel of Eq. (18)
    weights instance kernels by are the same object for every mechanism, and
    no consumer is handed a fallback. `_max_weights` carries why that is true
    of max pooling only under its instance-level reading, and what the
    coordinatewise reading would have cost.
    """

    def __init__(self, mechanism: str, classes: int, source: Pool, target: Pool,
                hyper: dict | None = None):
        if mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {mechanism!r}; known: {MECHANISMS}")
        super().__init__(config.ARMS_BY_ID["G"], classes, source, target, hyper=hyper)
        self.mechanism = mechanism
        width = self.encoder.output_dim
        if mechanism == "abmil-published":
            # Own parameters, never `self.V_R`/`b_R`/`v_R`: those are trained
            # WITH the l1-ball reparametrization and a fair comparison trains
            # each mechanism under its own definition, not one mechanism's
            # weights read through another's rule.
            self.mech_V = nn.Parameter(torch.empty(config.ATTENTION_WIDTH, width))
            self.mech_b = nn.Parameter(torch.zeros(config.ATTENTION_WIDTH))
            self.mech_v = nn.Parameter(torch.empty(config.ATTENTION_WIDTH))
            nn.init.xavier_uniform_(self.mech_V)
            nn.init.normal_(self.mech_v, std=0.1)
        elif mechanism == "abmil-gated":
            self.mech_V = nn.Parameter(torch.empty(config.ATTENTION_WIDTH, width))
            self.mech_U = nn.Parameter(torch.empty(config.ATTENTION_WIDTH, width))
            self.mech_w = nn.Parameter(torch.empty(config.ATTENTION_WIDTH))
            nn.init.xavier_uniform_(self.mech_V)
            nn.init.xavier_uniform_(self.mech_U)
            nn.init.normal_(self.mech_w, std=0.1)
        # "ours" reuses `self.V_R`/`b_R`/`v_R`, already allocated by
        # `Arm.__init__` (`G`'s own `spec["attention"] == "learned"`).
        # "mean"/"max" need no parameters of their own.

    def _params(self) -> dict:
        if self.mechanism == "ours":
            return {"V_R": self.V_R, "b_R": self.b_R, "v_R": self.v_R,
                    "gamma": self.attention_gamma, "sigma": self.sigma,
                    "tau_att": self.attention_temperature}
        if self.mechanism == "abmil-published":
            return {"V_R": self.mech_V, "b_R": self.mech_b, "v_R": self.mech_v,
                    "tau_att": self.attention_temperature}
        if self.mechanism == "abmil-gated":
            return {"V": self.mech_V, "U": self.mech_U, "w": self.mech_w,
                    "tau_att": self.attention_temperature}
        return {}

    def weights_for(self, H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """Overrides `Arm.weights_for`, and it is the ONLY override: every
        caller that pools through it (`bags_of`, `bag_representations`, and so
        every kernel `training_step` builds from them) reads this mechanism's
        own weights instead of `G`'s Eq. (15)/(16), and `Arm`'s own Eq. (19)
        then builds the bag representation from them unchanged.
        """
        return mechanism_weights(self.mechanism, H, self._params())


def build_mechanism(mechanism: str, classes: int, source: Pool,
                    target: Pool, hyper: dict | None = None) -> MechanismArm:
    """One attention mechanism, by the name `MECHANISMS` declares -- the
    comparison's own entry point, the sibling of `build` that never returns
    a declared arm.
    """
    return MechanismArm(mechanism, classes, source, target, hyper=hyper)


def build(arm_id: str, classes: int, source: Pool, target: Pool,
         hyper: dict | None = None) -> Arm:
    """One arm, by the identifier the ladder names it with.

    `hyper` is the ceiling search's own override for Decision 1's bandwidth and
    Eq. (15)/(16)/(28)'s three hyperparameters -- see `Arm.__init__`. Omitted,
    every arm trains at the declared `config` constants, exactly as before this
    override existed.
    """
    if arm_id not in config.ARMS_BY_ID:
        raise ValueError(f"unknown arm {arm_id!r}; known: {list(config.ARMS_BY_ID)}")
    return Arm(config.ARMS_BY_ID[arm_id], classes, source, target, hyper=hyper)
