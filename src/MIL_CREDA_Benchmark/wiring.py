"""The nine arms: what each one computes, and nothing about how it is trained.

Every arm shares the same encoder, the same linear head, the same data, the same
schedules and the same optimizer. What separates them is written in `config.ARMS`
and read here: which statistical unit the arm trains on, which adaptation term it
carries, whether the target blocks are weighted by confidence, whether the local
correspondence is on, and how a bag becomes a representation.

Nothing in `src/CREDA/` is modified. `FeatureExtractor` and `CREDALoss` are used
exactly as they are; the only thing this file decides is which tensors go in.

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


def _median_sigma(H: torch.Tensor) -> torch.Tensor:
    """The bandwidth rule CREDA already uses, applied where each kernel lives.

    `CREDALoss._compute_sigma` takes the median of the pairwise squared distances
    with the diagonal removed. The same rule serves MIL-CREDA, over the instance
    embeddings its kernel operates on rather than over bag representations. Same
    rule, different domain of application — which is not the same as the same
    number, and saying so is the point.
    """
    squared = torch.cdist(H, H) ** 2
    off_diagonal = squared[~torch.eye(H.shape[0], dtype=torch.bool, device=H.device)]
    return torch.sqrt(torch.median(off_diagonal) + 1e-6)


class Arm(nn.Module):
    """One row of the ladder, trainable end to end."""

    def __init__(self, spec: dict, classes: int, source: Pool, target: Pool):
        super().__init__()
        self.spec = spec
        self.classes = classes
        self.source = source
        self.target = target

        self.encoder = FeatureExtractor(backbone=config.BACKBONE, pretrained=config.PRETRAINED)
        self.head = nn.Linear(self.encoder.output_dim, classes)

        width = self.encoder.output_dim
        if spec["attention"] == "learned":
            # The parameters of Eq. (14). The equation itself is applied by
            # `MIL_CREDA.attention`; only its weights live here.
            self.V_R = nn.Parameter(torch.empty(config.ATTENTION_WIDTH, width))
            self.b_R = nn.Parameter(torch.zeros(config.ATTENTION_WIDTH))
            self.v_R = nn.Parameter(torch.empty(config.ATTENTION_WIDTH))
            nn.init.xavier_uniform_(self.V_R)
            nn.init.normal_(self.v_R, std=0.1)
        elif spec["attention"] == "random":
            # A fixed arbitrary relevance function: not learned, not uniform, and
            # deterministic given the embedding. Re-drawing it every step would
            # test noise rather than test attention.
            self.register_buffer("r_R", torch.randn(width) / width**0.5)

        if spec["adaptation"] == "creda":
            # lambda_creda is one because the coefficient is applied outside, from
            # the shared schedule; leaving it here as well would apply it twice.
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

    def weights_for(self, H: torch.Tensor) -> torch.Tensor:
        """The in-bag weights beta of Eq. (15), by whichever rule this arm uses."""
        if self.spec["attention"] == "learned":
            logits = relevance_logits(H, self.V_R, self.b_R, self.v_R)
        elif self.spec["attention"] == "random":
            logits = H @ self.r_R
        else:                                   # uniform: Eq. (16) with beta = 1/m
            return torch.full((H.shape[0],), 1.0 / H.shape[0],
                              dtype=H.dtype, device=H.device)
        return bag_weights(logits)

    def bag_representations(self, embeddings: torch.Tensor):
        """Eq. (16) for every bag, plus the weights that produced it."""
        weights = [self.weights_for(H) for H in embeddings]
        Z = torch.stack([bag_embedding(H, w) for H, w in zip(embeddings, weights)])
        return Z, weights

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
            Z, _ = self.bag_representations(embeddings)
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

    def _milcreda_term(self, H_s, source_labels, target_bags):
        """Eqs. (19)-(38): the global score, and the local correspondence if on.

        The source side is the step's own supervised batch, already encoded —
        which is what `train_creda` does with `feats_src`, so both families draw
        their source evidence the same way and the encoder runs once.
        """
        H_t = self.instance_embeddings(target_bags)
        beta_s = [self.weights_for(H) for H in H_s]
        beta_t = [self.weights_for(H) for H in H_t]

        sigma = _median_sigma(torch.cat([H_s.reshape(-1, H_s.shape[-1]),
                                         H_t.reshape(-1, H_t.shape[-1])]))

        bags_s = list(zip(H_s, beta_s))
        bags_t = list(zip(H_t, beta_t))
        K_ss = bag_kernel_matrix(bags_s, bags_s, sigma)
        K_st = bag_kernel_matrix(bags_s, bags_t, sigma)
        K_tt = bag_kernel_matrix(bags_t, bags_t, sigma)

        Z_t = torch.stack([bag_embedding(H, w) for H, w in bags_t])
        G_t = F.softmax(self.head(Z_t), dim=1)                    # Eq. (17)
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

        term = global_loss(losses)

        if self.spec["local"]:
            distances = []
            for column in range(len(bags_t)):
                cross = K_st[:, column]
                pi = total_correspondence(cross, source_labels, G_t[column],
                                          config.TAU_LOCAL)
                H_j, w_j = bags_t[column]
                distances.append(
                    local_distance(bag_kernel(H_j, w_j, H_j, w_j, sigma), cross, K_ss, pi)
                )
            term = term + local_loss(torch.stack(distances), w_t)

        return term

    def _creda_term(self, H_s, source_labels, target_bags):
        """CREDA's own loss, over instances, exactly as it was written."""
        instances = H_s.reshape(-1, self.encoder.output_dim)
        H_t = self.instance_embeddings(target_bags).reshape(-1, self.encoder.output_dim)
        y_s = source_labels.repeat_interleave(H_s.shape[1])
        g_t = F.softmax(self.head(H_t), dim=1)
        return self.creda(instances, H_t, y_s, g_t)

    # ------------------------------------------------------------------- step

    def training_step(self, bags: torch.Tensor, labels: torch.Tensor,
                      ramp: float, generator: torch.Generator) -> dict:
        """The arm's own objective, and nothing the arm does not have.

        The supervised term is the arm's: per bag for a bag-unit arm, which is
        Eq. (18) with the stabilizer at zero, and per instance for an instance-unit
        arm, which is what CREDA optimizes. The adaptation term is added with the
        shared coefficient — the same ramp and the same constant for every arm
        that has one — so nothing separates the arms except the term itself.
        """
        embeddings = self.instance_embeddings(bags)
        if self.spec["unit"] == "bag":
            Z, _ = self.bag_representations(embeddings)
            logits = self.head(Z)
            supervised = F.cross_entropy(logits, labels)
        else:
            instance_logits = self.head(embeddings)
            per_instance = labels.repeat_interleave(bags.shape[1])
            supervised = F.cross_entropy(
                instance_logits.reshape(-1, self.classes), per_instance
            )
            logits = torch.log(F.softmax(instance_logits, dim=-1).mean(dim=1)
                               + config.EPSILON)

        # Every arm encodes a target batch, including the floors that have no use
        # for it. This is not waste: resnet18 carries forty running means and
        # variances, and encoding target images in training mode updates all of
        # them with target statistics. An arm that computes an adaptation term
        # gets that for free, and a floor that never looks at the target does not
        # — so a comparison between them would measure the normalization rather
        # than the loss. Measured on M->U: with the coefficient forced to zero the
        # arm still scored 0.778 against the floor's 0.722, which is the whole gap
        # and none of it the loss. Encoding the same batch everywhere leaves the
        # loss as the only difference, and makes the wall-time row comparable too.
        target_bags = self.target.take(self._draw_target(generator))
        adaptation = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.spec["adaptation"] == "milcreda":
            adaptation = self._milcreda_term(embeddings, labels, target_bags)
        elif self.spec["adaptation"] == "creda":
            adaptation = self._creda_term(embeddings, labels, target_bags)
        else:
            self.instance_embeddings(target_bags)

        coefficient = ramp * config.LAMBDA_CONST
        total = supervised + coefficient * adaptation
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


def build(arm_id: str, classes: int, source: Pool, target: Pool) -> Arm:
    """One arm, by the identifier the ladder names it with."""
    if arm_id not in config.ARMS_BY_ID:
        raise ValueError(f"unknown arm {arm_id!r}; known: {list(config.ARMS_BY_ID)}")
    return Arm(config.ARMS_BY_ID[arm_id], classes, source, target)
