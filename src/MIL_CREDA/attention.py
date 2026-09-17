"""Instance relevance inside a bag, and the bag representation it induces.

A subject is a bag whose instances carry no label of their own and do not
contribute equally. The logit of each instance (Eq. 15) combines two bounded
components: a learned relevance R_phi(h), shared across domains, and a
consensus term built from the instance kernel kappa^I of Eq. (14) -- the
average affinity of an instance with the other instances of its OWN bag.
Eq. (16) turns those logits into weights normalized strictly WITHIN the bag
via an exponential with temperature, and Eq. (19) collapses the bag into a
convex combination of its instance embeddings for the classifier.

The learned relevance selector is Ilse-Tomczak-Welling attention with its
output vector reparametrized onto the closed unit l1 ball: the parameter that
is learned, `v_R`, carries no constraint of its own; the selector always
applies the normalized `v_R_tilde = v_R / max(1, ||v_R||_1)`, which coincides
with `v_R` when `||v_R||_1 <= 1` and rescales it to the ball's edge otherwise.
This holds for ANY value of `v_R`, so `|R_phi(h)| <= ||v_R_tilde||_1 <= 1` by
construction rather than by anything the optimizer has to respect.
"""

from __future__ import annotations

import torch

from MIL_CREDA import as_matrix, as_tensor
from MIL_CREDA.kernels import gaussian_kernel

__provenance__ = {
    "revision": "research-concept-r21.md",
    "sections": ["3"],
    "equations": ["15", "16", "19"],
    "invariants": [
        "bag_weights_on_simplex",
        "bag_embedding_permutation_invariant",
        "relevance_logit_matches_eq15_definition",
        "relevance_component_bounded_by_l1_normalization",
        "bag_weights_permutation_equivariant",
        "bag_weights_singleton_is_one",
        "attention_logits_depend_only_on_own_bag",
        "consensus_component_in_unit_interval",
        "consensus_component_bandwidth_limits",
        "relevance_logit_reduces_to_relevance_at_gamma_zero",
        "attention_logit_ratio_bounded_by_temperature",
        "relevance_logit_reduces_to_l1_normalized_relevance_at_neutral_hyperparameters",
        "effective_bag_size_in_range",
        "uniform_self_similarity_in_range",
        "separation_condition_implies_majority_consensus",
    ],
}


def _l1_ball_reparametrization(v_R: torch.Tensor) -> torch.Tensor:
    """v_R_tilde = v_R / max(1, ||v_R||_1), the un-numbered definition just
    before Eq. (15): the raw learned parameter is `v_R`, unconstrained; the
    selector always uses this normalized form, which coincides with `v_R`
    inside the unit l1 ball and is rescaled to its edge outside it, so
    ||v_R_tilde||_1 <= 1 for ANY value of the parameters.
    """
    v_R = as_tensor(v_R).reshape(-1)
    norm = torch.linalg.vector_norm(v_R, ord=1)
    return v_R / torch.clamp(norm, min=1.0)


def relevance_component(
    H: torch.Tensor, V_R: torch.Tensor, b_R: torch.Tensor, v_R: torch.Tensor
) -> torch.Tensor:
    """R_phi(h) = v_R_tilde^T tanh(V_R h + b_R), applied per instance.

    Since every coordinate of tanh lies in [-1, 1], Hoelder's inequality gives
    |R_phi(h)| <= ||v_R_tilde||_1 <= 1 for every h -- the bound the proposal
    states right after defining v_R_tilde. The same parameters serve source
    and target, so an embedding produces the same value regardless of its
    domain or bag.
    """
    H = as_matrix(H)
    v_R_tilde = _l1_ball_reparametrization(v_R)
    hidden = torch.tanh(H @ as_tensor(V_R).transpose(0, 1) + as_tensor(b_R))
    return hidden @ v_R_tilde


def consensus_component(H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """The consensus term of Eq. (15): (1/m) sum_a' kappa^I(h_a, h_a'), for
    every instance a of ONE bag, over its own instances only.

    Reuses the instance kernel of Eq. (14) and its own bandwidth sigma, never
    a second similarity. Because kappa^I(h, h) = 1, this lies in (0, 1] and
    equals 1/m in the limit sigma -> 0 (only self-terms survive) and 1 in the
    limit sigma -> infinity (the kernel saturates at one everywhere).
    """
    H = as_matrix(H)
    K = gaussian_kernel(H, H, sigma)
    return K.sum(dim=1) / H.shape[0]


def relevance_logits(
    H: torch.Tensor,
    V_R: torch.Tensor,
    b_R: torch.Tensor,
    v_R: torch.Tensor,
    gamma: float,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Implement Eq. (15): nu_a = R_phi(h_a) + gamma * consensus_a, gamma >= 0.

    `gamma` and `sigma` are explicit hyperparameters fixed during training --
    the proposal gives them no value, so nothing here defaults them. Because
    the consensus term lies in (0, 1] and R_phi in [-1, 1], both are bounded
    and comparable scales, and neither can cancel the other by growth of its
    own parameters. With gamma = 0 the logit reduces to R_phi alone. The
    logit is a function of the instance and of its own bag -- the consensus
    term never reads another bag's instances -- which is what keeps the bag
    representation of Eq. (17) a function of the bag alone.
    """
    if gamma < 0:
        raise ValueError("the consensus weight gamma must be non-negative")
    relevance = relevance_component(H, V_R, b_R, v_R)
    consensus = consensus_component(H, sigma)
    return relevance + gamma * consensus


def bag_weights(logits: torch.Tensor, tau_att: float) -> torch.Tensor:
    """Implement Eq. (16): beta_a = exp(nu_a / tau_att) / sum exp(nu_a' / tau_att).

    Never across bags and never mixing domains: the weights express relative
    relevance among the instances of a single subject. `tau_att` is a fixed
    hyperparameter with no default -- the proposal gives none -- that
    controls how concentrated the weights are; dividing both components of
    the logit alike, it does not disturb the balance gamma already fixed
    between them.
    """
    if tau_att <= 0:
        raise ValueError("the attention temperature tau_att must be strictly positive")
    return torch.softmax(as_tensor(logits).reshape(-1) / tau_att, dim=0)


def bag_embedding(H: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Implement Eq. (19): z = sum_a beta_a h_a, a convex combination in R^d.

    Invariant under permutations of the bag's instances, since the sum pairs
    each weight with its own embedding.
    """
    H = as_matrix(H)
    weights = as_tensor(weights).reshape(-1)
    if H.shape[0] != weights.shape[0]:
        raise ValueError("one weight per instance is required")
    return weights @ H


# --------------------------------------------------------------- diagnostics
#
# Neither quantity is wired into training or into any report; both are
# read-only diagnostics for the un-demonstrated possibility the proposal
# names right after Eq. (16): that end-to-end training could degrade the
# attention toward uniform weights. Read together rather than alone.


def effective_bag_size(weights: torch.Tensor) -> torch.Tensor:
    """m_eff = 1 / sum_a beta_a^2, the exponential of the quadratic Renyi
    entropy of the weights. m_eff in [1, m] and equals m only at uniform
    weights.
    """
    weights = as_tensor(weights).reshape(-1)
    return 1.0 / (weights * weights).sum()


def uniform_self_similarity(H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """||Psi_bar(B)||^2 = (1/m^2) sum_a sum_a' kappa^I(h_a, h_a'), the average
    consensus of the bag. In [1/m, 1], and equals 1 only if every embedding
    of the bag coincides.
    """
    H = as_matrix(H)
    K = gaussian_kernel(H, H, sigma)
    m = H.shape[0]
    return K.sum() / (m * m)
