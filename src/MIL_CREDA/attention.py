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
output vector CONSTRAINED to the closed unit l1 ball: `||v_R||_1 <= 1`, imposed
as a max-norm constraint. The selector reads `v_R` raw, and
`project_onto_l1_ball` re-imposes the constraint after each optimizer step, so
`|R_phi(h)| <= ||v_R||_1 <= 1` holds because the parameter is kept feasible.

**This replaced a reparametrization, and the difference is not cosmetic.**
Through r30 the parameter was unconstrained and the selector divided by
`max(1, ||v_R||_1)` inside the forward pass. The two produce the same bound and
NOT the same training: r30's division sat in the graph, so every update carried
the gradient of the normalization and r30's differentiability paragraph had to
carry an exception for it. r31 moves the constraint out of the graph entirely --
the optimizer computes its update from the raw parameter and the projection then
moves the result onto the ball -- so the exception is gone and `v_R_tilde` stops
existing as an object. What the optimizer sees therefore changed, which is why
this is a change to the training loop and not only to a function.

Because the constraint is now the parameter's rather than the forward pass's,
it has to be ESTABLISHED as well as maintained: see `Arm.__init__`, which
projects once at construction. `nn.init.normal_(v_R, std=0.1)` over 128
coordinates draws a norm near 11, so a tree that projected only after steps
would run its first forward with the bound above already false.
"""

from __future__ import annotations

import torch

from MIL_CREDA import as_matrix, as_tensor
from MIL_CREDA.kernels import gaussian_kernel

__provenance__ = {
    "revision": "research-concept-r33.md",
    "sections": ["3"],
    "equations": ["15", "16", "19"],
    "invariants": [
        "bag_weights_on_simplex",
        "bag_embedding_permutation_invariant",
        "relevance_logit_matches_eq15_definition",
        "project_onto_l1_ball_imposes_the_max_norm_constraint",
        "relevance_component_bounded_by_the_l1_constraint",
        "bag_weights_permutation_equivariant",
        "bag_weights_singleton_is_one",
        "attention_logits_depend_only_on_own_bag",
        "consensus_component_in_unit_interval",
        "consensus_component_bandwidth_limits",
        "relevance_logit_reduces_to_relevance_at_gamma_zero",
        "attention_logit_ratio_bounded_by_temperature",
        "relevance_logit_reduces_to_the_raw_relevance_at_neutral_hyperparameters",
        "effective_bag_size_in_range",
        "uniform_self_similarity_in_range",
        "separation_condition_implies_majority_consensus",
    ],
}


def project_onto_l1_ball(v_R: torch.Tensor) -> bool:
    """Impose Eq. (15)'s constraint as a max norm, IN PLACE, after a step.

    r31: "La restriccion se impone como norma maxima: tras cada paso de
    optimizacion, si `||v_R||_1 > 1`, se reescala `v_R <- v_R / ||v_R||_1`."

    Returns whether it fired, so a caller can tell a constraint that is doing
    work from one that never binds -- if the norm never exceeds one, r31's
    change is a no-op on that run and that is a fact worth recording rather
    than assuming either way.

    **In place, and on purpose.** The optimizer holds a reference to this
    parameter. A version returning a new tensor would leave the trained object
    untouched, the constraint would silently never apply, and every bound
    Eq. (15) states would quietly stop holding while the code read as though
    it enforced them.

    **Outside the graph, and that is the half r31 changed.** r30 divided inside
    the forward pass, so the normalization was a differentiable step and its
    differentiability paragraph carried an exception for it ("lo es excepto en
    un conjunto de medida nula"). r31 drops that exception because nothing in
    the forward normalizes any more: this runs under `no_grad` on a leaf the
    optimizer owns, between one step and the next forward, so it is a
    modification of the feasible set and not a term of the objective. What the
    optimizer sees therefore changes -- it computes its update from the raw
    parameter and the projection then moves the result, where r30 had the
    gradient of the division folded into every update.

    Interior points are left exactly alone: `max(1, ||v_R||_1)` and this rule
    agree there, and rescaling an interior point out to the edge would impose a
    constraint r31 does not state.
    """
    with torch.no_grad():
        norm = torch.linalg.vector_norm(v_R.reshape(-1), ord=1)
        if float(norm) <= 1.0:
            return False
        v_R.div_(norm)
    return True


def relevance_component(
    H: torch.Tensor, V_R: torch.Tensor, b_R: torch.Tensor, v_R: torch.Tensor
) -> torch.Tensor:
    """R_phi(h) = v_R^T tanh(V_R h + b_R), applied per instance.

    Since every coordinate of tanh lies in [-1, 1], Hoelder's inequality gives
    |R_phi(h)| <= ||v_R||_1 <= 1 for every h -- the bound Eq. (15) states.

    **`v_R` is read RAW, and the bound is a precondition rather than a
    construction.** Through r30 this function divided by `max(1, ||v_R||_1)`,
    so the bound held for ANY value of the parameter and the caller had nothing
    to respect. r31 imposes `||v_R||_1 <= 1` as a constraint on the parameter
    itself, maintained by `project_onto_l1_ball` after each optimizer step. So
    the bound now holds because the parameter is kept feasible, and this
    function would happily return something larger if handed an infeasible
    `v_R` -- which is why the projection has to be wired into the training loop
    and not merely available to it.

    The same parameters serve source and target, so an embedding produces the
    same value regardless of its domain or bag.
    """
    H = as_matrix(H)
    hidden = torch.tanh(H @ as_tensor(V_R).transpose(0, 1) + as_tensor(b_R))
    return hidden @ as_tensor(v_R).reshape(-1)


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
