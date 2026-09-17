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
        #: `GN`'s own captured source-batch BatchNorm statistics for the
        #: current step, set by `_source_embeddings` and consumed by
        #: `_target_embeddings`; `None` for every other arm, always.
        self._source_bn_stats: dict | None = None

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

        Every adapted arm but `GN` passes the target batch through the encoder
        exactly as it passes the source batch: normalization layers stay in
        training mode and learn from this forward like the rest of the model
        does. `GN` (`spec["normalization"] == "sourceBatch"`) is the one
        exception -- its target forward is normalized with the CURRENT SOURCE
        BATCH statistics of this same step (`self._source_bn_stats`, captured
        by `_source_embeddings` a few lines above it in `training_step`) and
        never updates the running statistics the source forward already set.
        `self._source_bn_stats` is `None` outside of training (`forward`,
        used for evaluation, never calls this method at all) and for every
        arm that never captures it, so the fallback below is exact and not a
        guess.
        """
        if self.spec.get("normalization") == "sourceBatch" and self._source_bn_stats is not None:
            return self._encode_with_frozen_stats(target_bags, self._source_bn_stats)
        return self.instance_embeddings(target_bags)

    # ------------------------------------------------------- GN's normalization

    def _batchnorm_modules(self) -> list[nn.Module]:
        """Every BatchNorm layer of the encoder, in the order `.modules()` walks them."""
        return [m for m in self.encoder.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def _source_embeddings(self, bags: torch.Tensor) -> torch.Tensor:
        """Encode the step's source batch, capturing `GN`'s own statistics along the way.

        For every arm but `GN` this is `instance_embeddings` and nothing else.
        For `GN` it also records, per BatchNorm layer, the batch mean and
        variance that layer's own train-mode forward just normalized WITH --
        not the EMA-blended running statistics the same forward updates as a
        side effect (Eq.-free bookkeeping `torch.nn.BatchNorm2d` does
        internally) -- into `self._source_bn_stats`, so `_target_embeddings`
        can reuse the identical numbers a few lines later in the same step.
        Every other arm leaves `self._source_bn_stats` at `None`, so
        `_target_embeddings`'s guard above never takes the frozen-stats path
        for them.
        """
        if self.spec.get("normalization") != "sourceBatch":
            self._source_bn_stats = None
            return self.instance_embeddings(bags)

        captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def _capture(module, inputs):
            x = inputs[0]
            reduce_dims = tuple(d for d in range(x.dim()) if d != 1)
            captured[id(module)] = (
                x.mean(dim=reduce_dims).detach(),
                x.var(dim=reduce_dims, unbiased=False).detach(),
            )

        hooks = [module.register_forward_pre_hook(_capture)
                for module in self._batchnorm_modules()]
        try:
            embeddings = self.instance_embeddings(bags)
        finally:
            for hook in hooks:
                hook.remove()
        self._source_bn_stats = captured
        return embeddings

    def _encode_with_frozen_stats(
            self, bags: torch.Tensor,
            captured: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Encode `bags` with every BatchNorm layer normalizing from `captured`.

        Every layer's `running_mean`/`running_var` is temporarily overwritten
        with `captured`'s (mean, var) -- the source batch's own, from this
        same step -- and the layer switched to evaluation mode, which
        normalizes from those two buffers and, unlike training mode, never
        updates them. Both are restored and training mode resumed once the
        forward returns, whether it raised or not, so this leaves no trace on
        the module beyond the encoded output: the running statistics the
        SOURCE forward already set are exactly what a caller reads afterward.
        """
        saved: list[tuple[nn.Module, torch.Tensor, torch.Tensor, bool]] = []
        try:
            for module in self._batchnorm_modules():
                stats = captured.get(id(module))
                if stats is None:
                    continue
                mean, var = stats
                saved.append((module, module.running_mean.clone(),
                             module.running_var.clone(), module.training))
                module.running_mean.copy_(mean.to(module.running_mean.dtype))
                module.running_var.copy_(var.to(module.running_var.dtype))
                module.eval()
            return self.instance_embeddings(bags)
        finally:
            for module, mean, var, was_training in saved:
                module.running_mean.copy_(mean)
                module.running_var.copy_(var)
                module.train(was_training)

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
        # `_source_embeddings`, not `instance_embeddings` directly: for every
        # arm but `GN` the two are identical, and for `GN` this is also where
        # `self._source_bn_stats` gets captured for `_target_embeddings`, a
        # few lines below inside `_milcreda_term`, to reuse.
        embeddings = self._source_embeddings(bags)
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
