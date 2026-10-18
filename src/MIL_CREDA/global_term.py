"""The global dependency score and the bounded loss it becomes.

Section 5 contrasts, class by class, the conditional geometry of both domains.
Eq. (32) builds three density matrices -- the two trace-normalized marginals
and the BALANCED joint, a congruence that puts half the mass on each domain;
Eq. (33) averages the two marginal entropies against the joint's; Eqs. (34)-(35)
bound that score by `-ln 2` below and `2 ln cosh((H_t - H_s)/4)` above, both read
off the same matrices; and Eq. (36) divides the corrected score `U_c - J_c` by
the constant `ln 2` that r30 proved bounds it, so minimizing the loss maximizes
`J_c - U_c`. Eq. (37) aggregates over EVERY class with the observed source
proportions `n_c^s / N_s`, which are non-negative and sum to one, so `L_glob` is
a CONVEX combination; a class with no active pair is charged the maximum loss of
one rather than skipped, and an empty active set therefore reads one and not
zero. The confidences do not enter this aggregation at all.

**That paragraph said the opposite until 2026-10-18 and the function did not.**
It read "aggregates the active classes SUB-convexly, weighted by the confidences
of the bags routed to each, and yields zero when no class is active" -- r31's
form, whose weights were `q_c / sum n^t` with `q_c = sum_j w_j^t`, summed to at
most one, and let the shortfall gate the term. r32 rewrote it and `global_loss`
below has implemented r32's form since; the three clauses above it survived
their own mechanism, which is the defect this repository keeps finding in prose
rather than in code.

**What r31 changed here, and what it did not.** Eq. (36)'s denominator was the
per-class span `U_c - L_c` through r29, with that span explicitly excluded from
the gradient. r30 proved `0 <= U_c - J_c <= ln 2`, so r31 divides by `ln 2` and
drops the gradient exception -- a literal constant has nothing to detach. The
lower end `L_c = -ln 2` stays in the formulation as Eq. (34)'s bound on `J_c`
and `entropy_cosh_bounds` still returns it; it left the denominator only.
Minimizing `l_glob,c` now reads as maximizing `J_c - U_c`, where r29 wrote
maximizing `J_c`.

**One revision lives here, and `__provenance__` names it.** r29 replaced
Eq. (32)'s normalization and Eqs. (34)-(35)'s ends, and the functions that
answer them -- `balanced_mixture`, `balanced_dependency_score`,
`entropy_cosh_bounds`, `class_global_loss_cosh` -- are what every consumer in
this repository calls.

**Three r25 functions stood beside them until 2026-10-18 and were removed.**
`dependency_score`, `conservative_bounds` and `class_global_loss` were kept
byte-identical as r25's retained trio, because `src/CREDA/bounds.py` held
copies of the latter two and correcting prior work was a separate piece of
work. What ended that is r33: it states one pair of ends, r25's cardinality
pair is nowhere in the formulation, and code for an equation the bound revision
does not contain makes the declaration above false about part of the file. The
CREDA copies went with them, and `models.BOUNDS_FORMS` keeps the keyword with a
single legal value so a run that asks for `conservative` is told the map is
gone rather than handed another one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from MIL_CREDA import DTYPE, as_tensor
from MIL_CREDA.renyi import quadratic_entropy, trace_normalize

#: Eq. (34)'s lower end, `L_c = -ln 2`. A constant, unlike r25's
#: `-ln(n_s + n_t)`: it is the value `J_c` approaches when the cross block
#: vanishes with the two purities equal, and that configuration says nothing
#: about how many bags either domain contributed.
#:
#: **r31 removed this from Eq. (36) and did not remove it from the formulation.**
#: Eq. (34) still states it as the lower bound on `J_c` and
#: `entropy_cosh_bounds` still returns it; what it stopped being is Eq. (36)'s
#: denominator.
ENTROPY_COSH_LOWER = -math.log(2.0)

#: r31 Eq. (36)'s denominator, `ln 2`. **Not `-ENTROPY_COSH_LOWER`, though it
#: is the same number.** This one is the bound r30 proved on the SPAN of the
#: corrected score: with `r_c = ln(rho_c^s / rho_c^t)`, the lower bound on
#: `rho_c^mix` gives `J_c >= ln cosh(r_c/2) - ln 2`, Eq. (35) is
#: `U_c = 2 ln cosh(r_c/4)`, and `cosh(r_c/2) = 2 cosh^2(r_c/4) - 1 >=
#: cosh^2(r_c/4)`, so `0 <= U_c - J_c <= ln 2`.
#:
#: The two constants coincide because both descend from the same `ln 2` in the
#: arithmetic-geometric mean step, and they are written separately because they
#: are claims about different objects -- one bounds `J_c`, the other bounds
#: `U_c - J_c`. A reader who saw `-ENTROPY_COSH_LOWER` in the quotient would
#: reasonably conclude the lower end was still in the denominator, which is
#: exactly what r31 changed.
GLOBAL_LOSS_SPAN = math.log(2.0)

__provenance__ = {
    "revision": "research-concept-r34.md",
    "sections": ["5"],
    "equations": ["32", "33", "34", "35", "36", "37"],
    "invariants": [
        # r31 Eq. (32) -- the trace balance as a congruence, introduced at
        # r29 and unchanged since.
        "balanced_mixture_is_a_unit_trace_congruence",
        "balanced_mixture_halves_the_mass_of_each_domain",
        "balanced_mixture_withdraws_a_bag_whose_confidence_vanishes",
        "balanced_score_is_invariant_to_the_common_confidence_scale",
        # r31 Eqs. (33)-(35) -- the score and the entropy-cosh ends. r29
        # introduced them; r31 narrowed Eq. (35) to its upper end, and the
        # lower end these invariants also cover is Eq. (34)'s.
        "entropy_cosh_bounds_are_the_two_ends_of_equation_thirty_five",
        "balanced_dependency_score_within_the_entropy_cosh_bounds",
        "the_global_loss_is_zero_exactly_at_alignment",
        "entropy_cosh_bounds_order_alignment_above_the_collapse",
        "cardinality_bounds_the_upper_end_without_being_attained",
        "cosh_bounds_agree_across_the_two_packages",
        # The axis `balanced_dependency_score` used to pin with no name.
        "detach_target_entropy_removes_the_dispersive_force_and_not_the_value",
        "detach_target_entropy_defaults_to_the_axis_value_creda_declares",
        # The two LAWS of this module -- pinned here with no flag on purpose,
        # and asserted to be the settings CREDA's own declaration chooses
        # rather than left as the mere absence of an alternative.
        "the_balance_is_credas_trace_mass_and_not_its_cardinality_one",
        "the_bounds_are_credas_entropy_cosh_choice_and_not_r25s_pair",
        # r31 Eq. (36) -- the denominator is the constant `ln 2`.
        "class_global_loss_cosh_divides_by_the_constant_span",
        # r33 Eq. (37) -- the aggregation weights are the observed source
        # proportions, and a class the routing abandoned costs the maximum.
        "global_loss_is_convex_combination_of_source_proportions",
        "global_loss_charges_one_for_an_inactive_class",
        "global_loss_inactive_charge_carries_no_gradient",
    ],
}


def is_active(
    K_s_c: torch.Tensor, K_t_weighted: torch.Tensor, K_mix_weighted: torch.Tensor
) -> bool:
    """The membership test for C_act of Eq. (37).

    A class participates only when bags exist in both domains and all three
    matrices that must be normalized have positive trace; otherwise the score
    is undefined for it and Eq. (37) excludes it rather than defaulting it.

    Returns a Python bool, not a tensor: membership of C_act selects which terms
    enter the sum, and a selection is not a quantity to differentiate.
    """
    matrices = [as_tensor(M) for M in (K_s_c, K_t_weighted, K_mix_weighted)]
    if any(M.numel() == 0 for M in matrices):
        return False
    return all(bool(torch.trace(M) > 0.0) for M in matrices)
def balanced_mixture(
    K_s_c: torch.Tensor, K_st_weighted: torch.Tensor, K_t_weighted: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implement Eq. (32): the three density matrices the global term works on.

    Returns `(K̆_s, K̆̃_t, K̆̃_mix)`, which is the triple r29 itself names --
    "en suma, el término global opera sobre tres matrices de densidad: las dos
    marginales normalizadas por traza y la conjunta balanceada". The two
    marginals are each divided by their own trace, as Eq. (9) requires of
    anything an entropy is taken of; the joint is the balanced congruence

        K̆̃_mix = ½ Q K̃_mix Q,
        Q = diag(tr(K_s)^{-1/2} I_{n_s}, tr(K̃_t)^{-1/2} I_{n_t}),

    assembled here block by block from `K̆_s`, `K̆̃_t` and the cross block
    rescaled coherently by `sqrt(tr(K_s) tr(K̃_t))`.

    **What this replaces.** r25 normalized the assembled mixed matrix by its own
    trace. That leaves the source block's weight inside the joint at
    `tr(K_s) / tr(K̃_mix)`, which moves with the batch's routing counts and with
    the COMMON scale of the confidences. When one block dominates, the quadratic
    entropy of the joint collapses to that block's and the score stops demanding
    alignment at all: a large misaligned group can outscore perfect alignment,
    and a uniform fall in every confidence can RAISE `J_c`. Pinning each
    domain's mass at ½ removes both.

    **The leading ½ is written because it is load-bearing here**, and that is a
    real difference from `src/CREDA/bounds.py`'s side of the same equation. In
    `CREDALoss` the factor is a genuine no-op, because that code divides by the
    trace of whatever it is handed and `½M / tr(½M) = M / tr(M)`. Nothing
    divides again here: this function returns a matrix that is ALREADY a
    density, and `quadratic_entropy` consumes it directly, so
    `H_2(½M) = H_2(M) + ln 4`. The factor is also in r29's own bound
    derivation, which reads
    `ρ_mix = ¼ρ_s + ¼ρ_t + ½‖K̆̃_st‖_F²` -- precisely the Frobenius norm of a
    matrix carrying it. Dropping it would leave the trace at 2, which is not a
    density, and would move every bound in Eqs. (34)-(35).

    **What survives the balance, and what does not.** The common level of the
    confidences cancels: `K̃_t = D K_t D` and its trace both scale by the square
    of a uniform factor, and the cross block and its `sqrt(tr · tr)` both scale
    by the factor itself. What does NOT cancel is their RELATIVE structure,
    which keeps acting inside `K̆̃_t` and `K̆̃_st` -- so a bag whose confidence
    goes to zero is withdrawn, exactly as r29 states, while a uniform drop
    changes nothing the score can see. That absolute level enters OUTSIDE the
    score instead, through Eq. (37)'s weights.

    Positive semidefiniteness survives because `Q` is a positive diagonal and a
    congruence `Q K Q` preserves the sign of the spectrum; the trace is
    `½ + ½ = 1` by construction, so Eq. (9) has the valid density it needs.

    **The balance is a LAW here, not an axis, and the law is asserted.** CREDA
    reaches this same construction through two flags -- `balance_mix_blocks`
    and `balance_mass` -- and either could be set otherwise; here there is no
    unbalanced branch and no cardinality branch to select. Until
    `the_balance_is_credas_trace_mass_and_not_its_cardinality_one`, the only
    evidence that the pinned construction was the one CREDA CHOSE was the
    absence of a flag, which is not evidence. That invariant reads CREDA's own
    declared choice and the congruence CREDA's `_mix_kernel_concat` builds
    under it, so the pin reddens if either side moves.
    """
    K_s = as_tensor(K_s_c)
    K_st_w = as_tensor(K_st_weighted)
    K_t_w = as_tensor(K_t_weighted)
    trace_s = torch.trace(K_s)
    trace_t = torch.trace(K_t_w)
    if trace_s <= 0.0 or trace_t <= 0.0:
        raise ValueError("both domains must contribute positive trace to the class")
    breve_s = K_s / trace_s
    breve_t = K_t_w / trace_t
    breve_st = K_st_w / torch.sqrt(trace_s * trace_t)
    top = torch.cat((breve_s, breve_st), dim=1)
    bottom = torch.cat((breve_st.transpose(0, 1), breve_t), dim=1)
    breve_mix = 0.5 * torch.cat((top, bottom), dim=0)
    return breve_s, breve_t, breve_mix


def balanced_dependency_score(
    K_s_c: torch.Tensor,
    K_st_weighted: torch.Tensor,
    K_t_weighted: torch.Tensor,
    *,
    detach_target_entropy: bool = False,
) -> torch.Tensor:
    """Implement Eq. (33) over Eq. (32)'s densities: J_c = (H_s + H_t)/2 - H_mix.

    The same contrast r25's `dependency_score` USED TO compute -- that function
    was removed on 2026-10-18 and nothing under `src/` defines it any more --
    over the three matrices Eq. (32) now defines rather than three independent
    trace normalizations. The signature is why this replaced it rather than
    becoming a keyword on it: the r25 function's third argument was the
    ASSEMBLED mixed matrix, and the balance cannot be applied to one — it needs
    `tr(K_s)` and `tr(K̃_t)` separately, so
    the blocks have to arrive separately. A keyword would have left the third
    positional argument meaning one thing under one setting and another under
    the other, which is worse than two names.

    A surrogate score for structural dependency, not a conventional mutual
    information: the joint structure is the conditional mixed matrix, not a
    Hadamard product of paired marginals. Both marginals carry the same sign
    and the same coefficient, so neither is privileged. It is MAXIMIZED, and
    Eq. (36)'s decreasing map is what turns it into something to minimize.

    **`detach_target_entropy` is an axis, and it used to be pinned here with no
    name.** `False` -- the default -- is what r29 introduced, what r31 still
    states, and what is
    published: `H_2(K̆̃_t)` enters the score attached, so the score's gradient
    carries the DISPERSIVE force the target block exerts on itself. `True`
    turns that force off and leaves everything else intact: `H_mix` still
    carries the target inside it, so the alignment the term asks for is
    untouched, and Eq. (35)'s ends read `H_2(K̆̃_t)` through `h_target` of
    `class_global_loss_cosh` rather than through here, so the bounds are
    untouched as well. It is an instrument, not a correction, and the parameter
    exists so a run can say which setting produced its numbers.

    **It changes no value this function returns**, only where the gradient
    flows -- `x.detach()` and `x` are the same number. So a test comparing
    scores or losses passes with the axis on AND with it broken, and the only
    test that can see it reads the GRADIENT. The difference is exact rather
    than approximate: against the attached score, detaching removes precisely
    `0.5 ∂H_2(K̆̃_t)/∂θ` from `∂J_c/∂θ`.

    **Which force it removes, and on whom.** `K̃_t = D K_t D` carries the
    confidences, so the entry pairing bags `j` and `k` enters weighted by
    `w_j^2 w_k^2` -- under r29's declared weighting the force concentrates on
    pairs of CONFIDENT bags, and what it does there is separate each class's
    confident core, which is the half the alignment term does not ask for.
    """
    breve_s, breve_t, breve_mix = balanced_mixture(
        K_s_c, K_st_weighted, K_t_weighted
    )
    h_target = quadratic_entropy(breve_t)
    if detach_target_entropy:
        h_target = h_target.detach()
    return 0.5 * (
        quadratic_entropy(breve_s) + h_target
    ) - quadratic_entropy(breve_mix)


def entropy_cosh_bounds(
    h_source: torch.Tensor, h_target: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """Return Eq. (35)'s `U_c = 2 ln cosh((H_t - H_s)/4)` and Eq. (34)'s `L_c`.

    **The two ends come from two equations, and r31 is why that is worth a
    sentence.** r28's Eq. (35) stated both: `L_c = -ln 2` and the `cosh` upper
    end, together. r31 narrowed Eq. (35) to the upper end alone, because `L_c`
    had stopped being needed where Eq. (35) was used -- Eq. (36) divides by the
    literal `ln 2` now, not by the span. The lower end did not leave the
    formulation: Eq. (34), byte-identical in r28 and r31, still states
    `-ln 2 <= J_c <= 2 ln cosh((H_t - H_s)/4)`.

    So this function keeps returning BOTH ends and its signature is unchanged;
    what changed is the attribution. The correction landed in
    `src/CREDA/bounds.py`' copy first and this one kept the old summary, which
    is why the test that holds it now reads both packages. See
    `ENTROPY_COSH_LOWER` for the constant itself.

    **What this replaces, and why the cardinality version could not stay.**
    r25's ends, removed from this module on 2026-10-18, combined the permitted
    extremes of each entropy separately, so they were constants once the two
    counts were fixed. An affine map with constant ends preserves the order of the scores
    it maps, and the score it is handed does not reach its own maximum at
    alignment when the marginal entropies differ: with `K̆_s = I / n_s` and a
    target collapsed far enough that the cross block vanishes,
    `J_c = ln((1 + 1/n_s)/4) + ½ ln n_s`, which is +0.52 at `n_s = 43` --
    above the 0 of perfect alignment. So r25's map prefers that collapse to the
    alignment, and no choice of constants repairs it.

    `U_c` depending on the two marginal entropies is what removes it, because
    `U_c - J_c >= 0` with equality only at alignment. Both ends are evaluated on
    the SAME normalized matrices that build `J_c` (Eq. 32), which is what makes
    that a statement about the score actually computed rather than about a bound
    on some other object.

    **Where it comes from.** With purities `ρ = tr(K̆²)` and `H_2 = -ln ρ`, the
    score reads `J_c = ln ρ_mix - ½ ln ρ_s - ½ ln ρ_t`, and the Frobenius norm
    splits over the blocks as `ρ_mix = ¼ρ_s + ¼ρ_t + ½‖K̆̃_st‖_F²`. The cross
    block is bounded by `0 <= ‖K̆̃_st‖_F² <= sqrt(ρ_s ρ_t)`; AM-GM gives the
    lower end `J_c >= ln ½ = -ln 2`, and
    `(sqrt(ρ_s) + sqrt(ρ_t))² / (4 sqrt(ρ_s ρ_t)) = cosh²(¼ ln(ρ_s/ρ_t))`
    gives the upper, with `ln(ρ_s/ρ_t) = H_2(K̆̃_t) - H_2(K̆_s)`.

    **The order of the subtraction is the target's minus the source's**, as
    Eq. (35) writes it. `cosh` is even, so the two orders return the same number
    and no test reading `U_c` alone can see the difference; the order is kept
    because the equation states it, not because this function would misbehave.

    **The lower end is a float and the upper end is a tensor**, and the
    asymmetry is the point. `L_c` is a constant of the formulation. `U_c` is a
    function of two entropies that are themselves functions of the features
    being aligned, so it carries a gradient and must stay attached -- it is half
    of the corrected score `U_c - J_c`. Casting it to a float here would
    silently delete that half of the gradient, and the loss would still look
    perfectly plausible.

    Cardinality survives only as an a priori observation: since `0 <= H_2 <=
    ln N`, `U_c <= 2 ln cosh(¼ ln max(n_s, n_t))`. That constant is NOT attained
    for `max >= 2` -- reaching it would need the maximal entropy difference and
    equality in the cross block's Cauchy-Schwarz bound at once, and the latter
    forces coinciding structures and hence equal entropies -- so it is not used
    to normalize anything.

    **Which Eq. (35) a run uses is a LAW here, and the law is asserted.** Here
    the normalization ALWAYS happens: this package carries one pair of ends and
    no flag selects another, which is why `wiring.py` reaches
    `class_global_loss_cosh` unconditionally.

    **r25's pair is gone from both packages and this sentence used to say
    otherwise.** It read that `conservative_bounds` "is retained only for the
    copies CREDA pins, never reached by a consumer"; that function was removed
    on 2026-10-18, here and in `CREDA.bounds`, because r33 states one pair of
    ends and r25's appears nowhere in the formulation. CREDA's own
    `normalize_by_bounds`/`bounds_form` keywords survive -- the second with a
    single admissible value -- so that a caller naming the retired map is
    refused by name instead of meeting `TypeError: unexpected keyword`.
    `the_bounds_are_credas_entropy_cosh_choice_and_not_r25s_pair` measures that
    the pinned reading is the one CREDA's declaration selects and that r25's
    pair would read something else, so the pin is evidence rather than silence.
    """
    difference = as_tensor(h_target) - as_tensor(h_source)
    upper = 2.0 * torch.log(torch.cosh(difference / 4.0))
    return ENTROPY_COSH_LOWER, upper


def class_global_loss_cosh(
    score: float | torch.Tensor,
    h_source: torch.Tensor,
    h_target: torch.Tensor,
) -> torch.Tensor:
    """Implement r31's Eq. (36): l_glob,c = (U_c - J_c) / ln 2, in [0, 1].

    `score` is `J_c`, the quantity that is MAXIMIZED. That was the convention
    r25's `class_global_loss` took too, and it is stated here rather than by
    reference because that function was removed on 2026-10-18 and a reader can
    no longer check the convention against it. `h_source` and `h_target` are
    `H_2` of
    Eq. (32)'s two trace-normalized marginals, which is exactly what
    `balanced_mixture` returns as its first two values.

    **The denominator is a literal constant, and that is the whole change.**
    r29 divided by the per-class span `U_c - L_c`. r30 proved the span of the
    corrected score is bounded by `ln 2` for every class -- see
    `GLOBAL_LOSS_SPAN` for the three lines of it -- so r31 divides by that
    constant, and the quotient still lands in [0, 1] by the theorem rather than
    by construction from the endpoints it divides by.

    **Two consequences, both invisible to a test comparing one class to
    itself.** First, r31: "su escala no depende del lote, de la clase ni de las
    cardinalidades: un mismo valor representa la misma diferencia `U_c - J_c`
    en cualquier caso". Under r29 two classes with the same corrected score and
    different marginal entropies reported different losses, so the per-class
    normalization made the numbers incomparable across the very axis Eq. (37)
    then summed over. Second, the top of the interval becomes reachable: under
    r29 the worst case evaluated to `ln 2 / (U_c + ln 2)`, strictly below 1
    whenever the marginal entropies differ.

    **Nothing is detached, because there is nothing left to detach.** r29's
    backpropagation section listed `U_c - L_c` as excluded from the gradient;
    r31's does not, because `ln 2` is a literal and carries no gradient to
    remove. `U_c` keeps its gradient in the NUMERATOR, where it always
    belonged -- it is half of the corrected score `U_c - J_c`, non-negative and
    zero only at alignment. What the detach used to suppress was a SECOND
    appearance of `U_c`, in the denominator, whose gradient rewarded widening
    the entropy gap; that appearance is gone rather than suppressed, so the
    numerator's `U_c` is now the only route and it penalizes the gap.
    """
    _lower, upper = entropy_cosh_bounds(h_source, h_target)
    return (upper - as_tensor(score)) / GLOBAL_LOSS_SPAN
def global_loss(
    class_losses: Sequence[torch.Tensor | float | None],
    source_counts: Sequence[int],
) -> torch.Tensor:
    """Implement Eq. (37): the convex aggregation over EVERY class.

    One entry per class in `C`, both arguments in the same class order.
    `class_losses[c]` is that class's `l_glob,c` when `c` is in `C_act`, and
    `None` when it is not; `source_counts[c]` is `n_c^s`, how many SOURCE bags
    of that class the batch holds. The weights are `n_c^s / N_s` with
    `N_s = sum_c n_c^s`, and `l~_c` is `1` for every class that is not active.

    **This replaced a confidence gate, and the two are opposite in kind.**
    Until r31 the weights were `omega_c = q_c / sum_c' n_c'^t` with
    `q_c = sum_{j in T_c} w_j^t` -- built from Eq. (24)'s confidences, summing
    to at most one, so the absolute level of the pseudo-label confidence scaled
    the whole term. r32 removed it: a weight the model itself determines can be
    lowered instead of earned, and the revision now says the weights "proceden
    de las etiquetas fuente, que son observadas". `n_c^s` is fixed by the data.

    **An inactive class costs the maximum, and that is the point.** r31 excluded
    it from the sum, so a target encoder that collapsed until its hard
    pseudo-label (Eq. 22) named one class left a sum over one class -- fewer
    terms, each in [0, 1], and the term went DOWN while nothing was aligned.
    Charging `1` makes that abandonment read as a rise. The charge carries no
    gradient and nothing here detaches it: the routing that decides membership
    is an `argmax`, so there is no path to differentiate, and `1` is a literal.
    Its whole job is the VALUE, so collapse is visible in the loss rather than
    rewarded by it.

    A class with `n_c^s = 0` cannot be active -- Eq. (37)'s membership asks for
    bags on both sides -- and its weight is `0 / N_s`, so it contributes
    nothing either way and needs no special case.

    The weights are convex, `sum_c n_c^s / N_s = 1` exactly, so `L_glob` is a
    convex combination of values in [0, 1] and stays in [0, 1]. A caller
    building weights by hand can no longer break that, because it does not hand
    weights: it hands counts.
    """
    if len(class_losses) != len(source_counts):
        raise ValueError(
            f"one source count per class is required: {len(class_losses)} "
            f"losses against {len(source_counts)} counts")
    if len(class_losses) == 0:
        raise ValueError("Eq. (37) sums over C, which is never empty")
    counts = [int(n) for n in source_counts]
    if any(n < 0 for n in counts):
        raise ValueError("a source bag count is non-negative")
    total = sum(counts)
    if total == 0:
        raise ValueError(
            "N_s is zero: the batch holds no source bag, so Eq. (37)'s weights "
            "are undefined")
    present = [loss for loss in class_losses if loss is not None]
    if not present:
        # Every class inactive: `l~_c = 1` throughout and the weights sum to
        # one, so the term is exactly 1. Built here rather than through
        # `_stack` because there is no per-class tensor to take dtype or
        # device from.
        return torch.ones((), dtype=DTYPE)
    # `(sum_c n_c^s l~_c) / N_s`, and NOT `sum_c (n_c^s / N_s) * l~_c`. Same
    # equation, different arithmetic: forming each weight first multiplies by
    # an inexact reciprocal and rounds once per class. Integer counts are exact
    # in float32 and there is one division, so a balanced batch with every
    # class active reproduces the plain mean bit for bit.
    reference = as_tensor(present[0]).reshape(())
    inactive_mass = sum(
        n for loss, n in zip(class_losses, counts) if loss is None)
    active = [
        as_tensor(loss).reshape(()) * n
        for loss, n in zip(class_losses, counts) if loss is not None
    ]
    weighted = torch.stack(active).sum() + inactive_mass
    return weighted / total


def _stack(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """One flat tensor out of a per-class sequence, however the caller built it.

    Each class loss arrives as its own 0-dim tensor when Eq. (36) is applied
    class by class, so the sequence has to be stacked rather than cast: casting
    would drop every one of those gradients on the floor.
    """
    if isinstance(values, torch.Tensor):
        return values.reshape(-1)
    items = [as_tensor(value).reshape(()) for value in values]
    return torch.stack(items) if items else torch.zeros(0, dtype=DTYPE)
