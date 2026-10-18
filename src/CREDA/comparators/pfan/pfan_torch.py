# PyTorch TRANSLATION of the official PFAN TensorFlow 1 code.
# Paper: Chen, Xie, Huang, Rong, Ding, Huang, Xu, Huang. Progressive Feature Alignment for
#        Unsupervised Domain Adaptation. CVPR 2019. arXiv:1811.08585
# Repository: https://github.com/Xiewp/PFAN
# Commit: 328c4c5b4e24a6d04b06136d9e83d8acb391a289
# Upstream path: model.py (protoloss L4-5, adloss L111-158, D L239-254, optimize L194-195),
#                train.py (adaptation_factor L26-31, cos_distance L33-44), pseudo.py (L22-33)
# License as found: none found in the repository
# The verbatim TF sources are kept, never imported, under ./upstream/*.txt.
#
# Translation rules: one TF op -> one torch op, upstream constants kept (decay 0.3 at
# model.py L139, loss scale 0.1 at L156-157, D widths 1024/1024/1, init stddev
# 0.01/0.01/0.3 and dropout keep_prob 0.5 at L243-251, top-3 per class at pseudo.py L28-33).
#
# DEVIATION FROM UPSTREAM (the only one, deliberate):
#   model.py L143 builds the TARGET centroid from `self.source_moving_centroid`:
#       target_centroid=(decay)*current_target_centroid+(1.-decay)*self.source_moving_centroid
#   with the target form commented out on L142. The paper (Sec. 3.3, Eq. 8 and Algorithm 1
#   line 12) updates the target global prototype c^T_{k(I)} from its own previous value
#   c^T_{k(I-1)}, and model.py L195 maintains `target_moving_centroid` for exactly that
#   purpose -- but nothing ever reads it. This translation uses the L142 form
#   (target moving centroid), which is the paper's intended update.
#
# NOT translated (recorded, not resolved): the paper's Eq. 7-8 update -- accumulated mean
# of the local prototypes and a cosine-adaptive weight rho^2 -- differs from the code's
# fixed exponential decay 0.3 on the current mini-batch centroid; the code's mechanism is
# kept. The paper's Eq. 9 sums squared distances over classes; the code (L5) takes the mean
# over C x D entries; the code is kept. The paper's EHTS uses a threshold tau (Eq. 3-4);
# the code (pseudo.py) keeps the top-3 most similar targets per class; the code is kept.
import math

import torch
import torch.nn.functional as F
from torch import nn


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        'third-party baseline: PyTorch translation of the official TF1 PFAN '
        'code (Xiewp/PFAN@328c4c5, model.py), Eq. 8 and 9; it implements no '
        'equation of the CREDA research concept (see CITATIONS.md)'
    ),
    "invariants": [
        "pfan_semantic_term_is_zero_on_identical_batches",
        "pfan_target_centroid_uses_target_moving_centroid",
    ],
}


def protoloss(sc, tc):
    # model.py L4-5: tf.reduce_mean(tf.square(sc-tc))
    return torch.mean(torch.square(sc - tc))


def adaptation_factor(x):
    # train.py L26-31
    if x >= 1.0:
        return 1.0
    den = 1.0 + math.exp(-10 * x)
    lamb = 2.0 / den - 1.0
    return lamb


def _trunc_normal(shape, stddev):
    # tf.truncated_normal: normal truncated at two standard deviations.
    w = torch.empty(*shape)
    nn.init.trunc_normal_(w, mean=0.0, std=stddev, a=-2 * stddev, b=2 * stddev)
    return w


class Discriminator(nn.Module):
    """model.py L239-254 (`D`). Returns (logits, sigmoid(logits)).

    TF applies `tf.nn.dropout(..., 0.5)` unconditionally (no training flag), so the
    translation keeps dropout on regardless of `self.training`.
    """

    def __init__(self, num_units_in):
        super().__init__()
        num_units_out = 1
        self.weights = nn.Parameter(_trunc_normal((num_units_in, 1024), 0.01))
        self.biases = nn.Parameter(torch.zeros(1024))
        self.weights2 = nn.Parameter(_trunc_normal((1024, 1024), 0.01))
        self.biases2 = nn.Parameter(torch.zeros(1024))
        self.weights3 = nn.Parameter(_trunc_normal((1024, num_units_out), 0.3))
        self.biases3 = nn.Parameter(torch.zeros(num_units_out))

    def forward(self, x):
        hx = x @ self.weights + self.biases
        ax = F.dropout(F.relu(hx), 0.5, training=True)
        hx2 = ax @ self.weights2 + self.biases2
        ax2 = F.dropout(F.relu(hx2), 0.5, training=True)
        hx3 = ax2 @ self.weights3 + self.biases3
        return hx3, torch.sigmoid(hx3)


def _unsorted_segment_sum(data, segment_ids, num_segments):
    out = torch.zeros(num_segments, data.size(1), dtype=data.dtype, device=data.device)
    return out.index_add(0, segment_ids, data)


class PFAN(nn.Module):
    """Holds the state model.py keeps as TF variables: the two moving centroids and D."""

    def __init__(self, num_classes, featurelen, temperature=1.8, decay=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.featurelen = featurelen
        self.temperature = temperature  # model.py L60: softmax(score/1.8)
        self.decay = decay  # model.py L139
        # model.py L13-14: zeros, trainable=False
        self.register_buffer("source_moving_centroid", torch.zeros(num_classes, featurelen))
        self.register_buffer("target_moving_centroid", torch.zeros(num_classes, featurelen))
        self.D = Discriminator(featurelen)

    def adloss(self, source_feature, target_feature, target_score, source_label, detach_discriminator=False):
        """model.py L111-158. `target_score` is the classifier logit on the target
        (`self.score`); `source_label` holds class indices (TF takes one-hot `y` and
        argmaxes it at L127). Returns (G_loss, D_loss, source_centroid, target_centroid),
        and sets `self.semantic_loss` as TF sets `self.Semanticloss`.

        `detach_discriminator=True` runs D with its parameters detached: same values
        and same dropout draws, but no gradient reaches D. Upstream gets this from
        `var_list` (the F op at L187 never updates D's variables); in PyTorch the
        caller's backward would otherwise accumulate G = -D gradients into D."""
        target_pred = F.softmax(target_score / self.temperature, dim=1)  # L60/L119
        disc = self._detached_D if detach_discriminator else self.D
        source_logits, _ = disc(source_feature)  # L121
        target_logits, _ = disc(target_feature)  # L123
        source_result = source_label  # L127 (argmax of one-hot)
        target_result = torch.argmax(target_pred, 1)  # L128
        ones = torch.ones_like(source_feature)  # L129
        current_source_count = _unsorted_segment_sum(ones, source_result, self.num_classes)  # L130
        current_target_count = _unsorted_segment_sum(torch.ones_like(target_feature), target_result, self.num_classes)  # L131

        current_positive_source_count = torch.maximum(current_source_count, torch.ones_like(current_source_count))  # L133
        current_positive_target_count = torch.maximum(current_target_count, torch.ones_like(current_target_count))  # L134

        current_source_centroid = _unsorted_segment_sum(source_feature, source_result, self.num_classes) / current_positive_source_count  # L136
        current_target_centroid = _unsorted_segment_sum(target_feature, target_result, self.num_classes) / current_positive_target_count  # L137

        decay = self.decay  # L139

        # DEVIATION: L142 (commented upstream) instead of L143 -- see the file header.
        target_centroid = (decay) * current_target_centroid + (1. - decay) * self.target_moving_centroid
        source_centroid = (decay) * current_source_centroid + (1. - decay) * self.source_moving_centroid  # L144

        self.semantic_loss = protoloss(source_centroid, target_centroid)  # L146

        # L149-150: target is the "real" class (label 1), source the "fake" one (label 0).
        D_real_loss = torch.mean(F.binary_cross_entropy_with_logits(target_logits, torch.ones_like(target_logits)))
        D_fake_loss = torch.mean(F.binary_cross_entropy_with_logits(source_logits, torch.zeros_like(source_logits)))
        D_loss = D_real_loss + D_fake_loss  # L151
        G_loss = -D_loss  # L152

        G_loss = 0.1 * G_loss  # L156
        D_loss = 0.1 * D_loss  # L157
        return G_loss, D_loss, source_centroid, target_centroid

    def _detached_D(self, x):
        params = {name: p.detach() for name, p in self.D.named_parameters()}
        return torch.func.functional_call(self.D, params, (x,))

    def discriminator_regularizer(self):
        """model.py L83: Dregloss = 0.0005 * reduce_mean([l2_loss(w) for D weights]),
        tf.nn.l2_loss(w) = sum(w**2)/2. Added to D_loss in D's own objective (L84-85)."""
        ws = [self.D.weights, self.D.weights2, self.D.weights3]
        return 0.0005 * torch.stack([(w ** 2).sum() / 2 for w in ws]).mean()

    def discriminator_param_groups(self, lr, momentum=0.9):
        """model.py L84-85 (`adoptimize`): weights at `lr`, biases at 2*lr, momentum 0.9
        (MomentumOptimizer). No weight_decay here: upstream regularises through
        `discriminator_regularizer()` inside the loss, not through the optimiser."""
        weights = [self.D.weights, self.D.weights2, self.D.weights3]
        biases = [self.D.biases, self.D.biases2, self.D.biases3]
        return [
            {"params": weights, "lr": lr, "momentum": momentum, "weight_decay": 0.0},
            {"params": biases, "lr": 2.0 * lr, "momentum": momentum, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def update_centroids(self, source_centroid, target_centroid):
        """model.py L194-195: assign after the optimiser step (tf.control_dependencies)."""
        self.source_moving_centroid.copy_(source_centroid.detach())
        self.target_moving_centroid.copy_(target_centroid.detach())


def cos_distance(vector1, vector2):
    """train.py L33-44 (despite the name, a cosine SIMILARITY); None on a zero norm."""
    normA = float(torch.sum(vector1 ** 2))
    normB = float(torch.sum(vector2 ** 2))
    if normA == 0.0 or normB == 0.0:
        return None
    return float(torch.sum(vector1 * vector2)) / ((normA * normB) ** 0.5)


@torch.no_grad()
def ehts_select(target_features, source_prototypes, top_k=3):
    """EHTS as the code does it: train.py L203-214 (most similar source prototype per
    target sample, by `cos_distance`) then pseudo.py L22-33 (keep the `top_k`=3 most
    similar targets of each pseudo-class). Returns (indices, pseudo_labels).

    `source_prototypes` is (C, D): train.py L180-182 averages source features per class
    (`dic_s`). A per-round procedure, not a per-batch loss.
    """
    preds, sims = [], []
    for j in range(target_features.size(0)):
        sim_list = [cos_distance(target_features[j], source_prototypes[k]) for k in range(source_prototypes.size(0))]
        max_sim = max(sim_list)
        preds.append(sim_list.index(max_sim))
        sims.append(max_sim)
    indices, labels = [], []
    for k in range(source_prototypes.size(0)):
        members = [(sims[j], j) for j in range(len(preds)) if preds[j] == k]
        if not members:
            continue
        ans = sorted(members, reverse=True)
        for _, j in ans[:top_k]:
            indices.append(j)
            labels.append(k)
    return torch.tensor(indices, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
