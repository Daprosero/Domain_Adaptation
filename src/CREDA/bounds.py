"""The bounds and the supervised scale, copied onto the CREDA side.

**These are deliberate copies, not imports, and the duplication is the point.**
The same functions already exist in `src/MIL_CREDA/global_term.py` and
`src/MIL_CREDA/objective.py`, and nothing here would be shorter for importing
them. What an import would cost is independence: CREDA is the baseline that
MIL-CREDA is measured against, so a CREDA module that reached into
`MIL_CREDA` would point the dependency backwards -- prior work depending on
the method it is the control for -- and the comparison would be contaminated
at the import line, before a single number was produced. A baseline that has
to be installed alongside the thing it baselines is not a baseline.

**What keeps a copy from drifting is a test, not a promise.**
`tests/test_creda_sample.py` imports both this module and its originals and
asserts, function by function, that they agree -- on ordinary inputs and on
the inputs each original refuses. A copy that is asserted equal to its
original is verified rather than trusted, so the separation costs nothing in
fidelity; a copy whose justification is not written down is just a duplicate
nobody dares touch, which is why this paragraph exists.

**Where the originals say "bag", this says "unit" or "sample".** Eq. (21) is
where that substitution lives: one row of `Y_source` per labelled unit, and at
sample level that unit is a sample. Eq. (35)'s ends no longer read a count at
all -- they are functions of the two marginal entropies -- so nothing is
substituted there.

`as_tensor` and `as_matrix` are copied too, for the same reason and with the
same test treatment: they are `MIL_CREDA.__init__`'s plumbing, and importing
plumbing would re-establish exactly the dependency this module exists to
avoid.

**One revision lives here, and `__provenance__` names it: r34.** r34 moved
Eq. 15's consensus term to a leave-one-out average, which is MIL-CREDA's
attention over the instances of a bag; this module holds Eqs. 21, 35 and 36,
all three byte-identical in r33 and r34, so the rebind moved no arithmetic and
every claim below about r33 holds at r34 unchanged. It named r25
until 2026-10-18, because `conservative_bounds` and `class_global_loss` sat
beside the two functions below and implemented Eq. (35)/(36) as r25 wrote them:
ends from cardinality alone, `L_c = -ln(n_s + n_t)` and
`U_c = (1/2)(ln n_s + ln n_t)`, and a loss divided by the per-class span
`U_c - L_c`. A declaration is a claim about everything it covers, so while both
maps were reachable the older one decided the stamp.

**Both of those functions were removed on 2026-10-18, and that is why the stamp
could move.** r33 states one pair of ends and r25's pair is nowhere in the
formulation, so keeping code for it meant declaring a revision whose
mathematics the module no longer held. What is left implements r33:

- `entropy_cosh_bounds` -- Eqs. (34)-(35)'s two ends, the ones that REPLACED
  the cardinality pair. r28 introduced them, deriving the upper end from the
  two marginal entropies of the same normalized matrices that build the score,
  and r33 states it unchanged.
- `class_global_loss_cosh` -- Eq. (36) over that upper end:
  `(U_c - J_c) / ln 2`, dividing by the constant r30 proved bounds the span.
- `source_bound` and `source_loss` -- Eq. (21), byte-identical in r25 and r33,
  and still held equal to `src/MIL_CREDA/objective.py`'s originals by
  `tests/test_creda_sample.py`.

**The r33 pair arrived in two steps, and the scale is the second one.** r28
introduced both functions with Eq. (36) still dividing by the per-class span
`U_c - L_c` and that span explicitly excluded from the gradient. r30 proved
`0 <= U_c - J_c <= ln 2`, and r31 therefore divides by `ln 2` and drops the
gradient exception, because a literal constant has nothing to detach. That is
the step that made the loss comparable across batches: with `U_c - L_c` in the
denominator the scale moved with every class's own entropies, so the same value
meant a different degree of alignment in two classes. The lower end
`L_c = -ln 2` did not disappear from the formulation -- Eq. (34) still states it
and `entropy_cosh_bounds` still returns it -- it left the denominator.
`GLOBAL_LOSS_SPAN` carries the theorem; `ENTROPY_COSH_LOWER` carries the lower
end, and they are separate names on purpose.

**What this costs, stated rather than left to be discovered.** Everything
`Plots_Generator` and `Results_Generator` published was measured under r25's
two ends, and no code here computes them any more. Those numbers are reachable
in the artefacts that recorded them; they are not reproducible by re-running
`CREDA.training_pipeline.train_creda`, which no longer names a bounds form.
`models.BOUNDS_FORMS` keeps the keyword with a single legal value so a run that
asks for `conservative` is told the map is gone.
"""

from __future__ import annotations

import math

import torch

#: The dtype this module uses for anything it has to materialize itself.
#: `MIL_CREDA.DTYPE`'s value, copied: the invariants these bounds take part in
#: are checked near 1e-12 and single precision cannot resolve them.
DTYPE = torch.float64

#: Eq. (34)'s lower end, `L_c = -ln 2`. A constant, unlike r25's
#: `-ln(n_s + n_t)`: it is the value `J_c` approaches when the cross block
#: vanishes with the two purities equal, and that configuration says nothing
#: about how many units either domain contributed.
#:
#: **r31 removed this from Eq. (36) and did not remove it from the formulation.**
#: It is still the lower end `entropy_cosh_bounds` returns and still the bound
#: Eq. (34) states on `J_c`; what it stopped being is Eq. (36)'s denominator.
ENTROPY_COSH_LOWER = -math.log(2.0)

#: r31 Eq. (36)'s denominator, `ln 2`. **Not `-ENTROPY_COSH_LOWER`, though it
#: is the same number.** This one is the bound r30 proved on the SPAN of the
#: corrected score: with `r_c = ln(rho_s / rho_t)`, the lower bound on
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
    "sections": ["3", "5"],
    "equations": ["21", "35", "36"],
    "invariants": [
        "source_bound_copy_agrees_with_the_original",
        "source_loss_copy_agrees_with_the_original",
        "entropy_cosh_bounds_are_equation_thirty_fives_two_ends",
        "class_global_loss_cosh_divides_by_the_constant_span",
    ],
}


def as_tensor(value: object, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Accept whatever the caller has; return a tensor without disturbing one.

    A tensor is returned untouched. Re-casting it would silently detach it from
    the graph that produced it, or pull a device-resident term back to the host,
    and either one turns a trainable implementation back into a calculator.
    Anything else -- a list, an array, a Python float -- becomes a tensor.
    """
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=DTYPE if dtype is None else dtype)


def as_matrix(value: object) -> torch.Tensor:
    """`as_tensor`, then at least two dimensions: one row per sample."""
    tensor = as_tensor(value)
    return tensor.reshape(1, -1) if tensor.dim() < 2 else tensor


# ------------------------------- Eq. (35), byte-identical from revision 33 on
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

    So this function keeps returning BOTH ends and its signature is unchanged.
    What changed is the attribution, and the attribution is the whole content
    of a docstring like this one: a reader sent to Eq. (35) in r31 looking for
    `L_c` does not find it there and has no way to tell whether the code or the
    equation is wrong. See `ENTROPY_COSH_LOWER` for the constant itself.

    **What this replaces, and why the cardinality version could not stay.**
    r25's ends, removed from this module on 2026-10-18, combined the permitted
    extremes of
    each entropy separately, so they are constants once the two counts are
    fixed. An affine map with constant ends preserves the order of the scores
    it maps, and the score it is handed does not reach its own maximum at
    alignment when the marginal entropies differ: with `K_s = I / n_s` and a
    target collapsed far enough that the cross block vanishes,
    `J_c = ln((1 + 1/n_s)/4) + (1/2) ln n_s`, which is +0.52 at `n_s = 43` --
    above the 0 of perfect alignment. So the r25 map prefers that collapse to
    the alignment, and no choice of constants repairs it.

    `U_c` depending on the two marginal entropies is what removes it, because
    `U_c - J_c >= 0` with equality only at alignment. The ends are evaluated on
    the SAME trace-normalized matrices that build `J_c` (Eq. 32), which is what
    makes that statement about the score actually computed rather than about a
    bound on a different object.

    **The order of the subtraction is the target's minus the source's**, as
    r28 writes it. `cosh` is even, so the two orders return the same number and
    no test reading `U_c` alone can see the difference; the order is kept
    because the equation states it, not because this function would misbehave.

    **The lower end is a float and the upper end is a tensor**, and the
    asymmetry is the point. `L_c` is a constant of the formulation. `U_c` is a
    function of two entropies that are themselves functions of the features
    being aligned, so it carries a gradient and must stay attached -- it is
    half of the corrected score `U_c - J_c`. Casting it to a float here would
    silently delete that half of the gradient, and the loss would still look
    perfectly plausible.
    """
    difference = as_tensor(h_target) - as_tensor(h_source)
    upper = 2.0 * torch.log(torch.cosh(difference / 4.0))
    return ENTROPY_COSH_LOWER, upper


# ------------------------------- Eq. (36), byte-identical from revision 33 on


def class_global_loss_cosh(
    score: float | torch.Tensor,
    h_source: torch.Tensor,
    h_target: torch.Tensor,
) -> torch.Tensor:
    """Implement r31's Eq. (36): l_glob,c = (U_c - J_c) / ln 2, in [0, 1].

    `score` is `J_c`, the quantity that is MAXIMIZED -- the same convention
    `class_global_loss` above takes, and the reason `CREDALoss` passes it
    `-creda_c`: this repository computes `creda_c = h_mix - (h_s + h_t)/2`,
    which is `-J_c`.

    `h_source` and `h_target` are `H_2` of the two trace-normalized marginals,
    which is exactly what `CREDALoss._renyi_entropy_order_2` returns for
    `K_s_c` and the weighted `K_t_c`: that function divides by the trace of
    whatever it is handed, so its output is already the entropy of the
    normalized matrix and needs no further normalization here.

    **The denominator is a literal constant, and that is the whole change.**
    r29 divided by the span `U_c - L_c`, which varies with the class. r30
    proved the span of the corrected score is bounded by `ln 2` regardless --
    see `GLOBAL_LOSS_SPAN` for the three lines of it -- so r31 divides by that
    constant and the quotient still lands in [0, 1], now by the theorem rather
    than by construction from the endpoints.

    **Two consequences, both of which a test comparing one class to itself
    cannot see.** First, the scale stops depending on the batch, the class or
    the cardinalities: a given value means the same `U_c - J_c` in every class,
    where r29's per-class denominator made two classes with equal corrected
    scores report different losses. Second, the top of the interval becomes
    reachable -- under r29, `ln 2 / (U_c + ln 2)` is strictly below 1 for every
    class whose marginal entropies differ, so the worst case never read as 1.

    **Nothing is detached, because there is nothing left to detach.** r29
    excluded `U_c - L_c` from the gradient by design, and r31 drops that
    exception from its backpropagation section: `ln 2` is a literal and carries
    no gradient to remove. `U_c` keeps its gradient in the NUMERATOR, where it
    always belonged -- it is half of the corrected score. The removed term is
    the one that rewarded widening the entropy gap through the denominator; the
    numerator's `U_c` still penalizes it, which is what the upper end is for.
    """
    _lower, upper = entropy_cosh_bounds(h_source, h_target)
    return (upper - as_tensor(score)) / GLOBAL_LOSS_SPAN


# ---------------------------- Eq. (21), byte-identical from revision 25 on


def source_bound(epsilon: float = 1e-8) -> float:
    """Implement B_src of Eq. (21): the exact supremum of a single unit's loss.

    The per-unit quantity is monotone in the observed class's score, so its
    extremes sit at the extremes of g in [0, 1]. All the mass on the correct
    class costs zero; all of it elsewhere costs ln(1 + 1/eps), whatever the
    remaining C - 1 classes do, because only the observed class enters.

    The stabilizer is what makes that finite: without it the logarithm diverges
    and there is no ceiling to divide by. So eps is not a purely numerical knob
    -- it fixes the scale of the supervised term, and moving it from 1e-8 to
    1e-6 takes this bound from about 18.42 to about 13.82.
    """
    if epsilon <= 0:
        raise ValueError("the source stabilizer must be strictly positive")
    return math.log(1.0 + 1.0 / epsilon)


def source_loss(
    G_source: torch.Tensor, Y_source: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Implement Eq. (21): the per-unit averaged negative log-likelihood over B_src.

    `Y_source` holds one-hot labels, one row per labelled unit -- a bag where
    the original caller feeds it bags, a sample here -- so the inner sum selects
    the observed class. The stabilizer is normalized by its own maximum,
    (g + eps)/(1 + eps), so the argument never exceeds one: the logarithm stays
    non-positive, the term stays non-negative, and a unit predicted with
    certainty costs exactly zero. Adding eps inside the logarithm alone would
    make a certain, correct prediction score -ln(1 + eps) < 0.

    Dividing by `source_bound` puts every unit's loss in [0, 1) and so the
    average too, which is what lets the terms of Eq. (39) be read on a common
    scale. It leaves the minimizer alone: B_src is a constant fixed by eps, so
    dividing by it is the same as rescaling the free coefficients. The upper end
    is a supremum and not a maximum, because the classifier produces strictly
    positive probabilities and never reaches g = 0.
    """
    bound = source_bound(epsilon)
    G = as_matrix(G_source)
    Y = as_matrix(Y_source)
    if G.shape != Y.shape:
        raise ValueError("one label vector per unit, over the same classes, is required")
    if G.shape[0] == 0:
        raise ValueError("the supervised term needs at least one source unit")
    per_unit = -(Y * torch.log((G + epsilon) / (1.0 + epsilon))).sum() / bound
    return per_unit / G.shape[0]
