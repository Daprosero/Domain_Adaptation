"""The arms declared in `config.ARMS`: what each one computes, and nothing
about how it is trained.

Every arm shares the same linear head, the same data, the same schedules and
the same optimizer. The encoder is where Eq. (13) splits: the floor (`B`)
holds exactly one, and every arm that declares an adaptation term (`E`, `F`,
`G`) holds two -- a source encoder and a target encoder, sharing no
parameter and sharing only the codomain `R^d`. What separates the arms
otherwise is written in `config.ARMS` and read here: which statistical unit
the arm trains on, which adaptation term it carries, whether the target
blocks are weighted by confidence, whether the local correspondence is on,
how a bag becomes a representation, and which of a bag's instances the arm
is allowed to look at.

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
from MIL_CREDA.attention import (
    bag_embedding,
    bag_weights,
    project_onto_l1_ball,
    relevance_logits,
)
from MIL_CREDA.bag_kernel import bag_kernel, bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import confidences, pseudolabel
from MIL_CREDA.global_term import (
    balanced_dependency_score,
    balanced_mixture,
    class_global_loss_cosh,
    global_loss,
    is_active,
)
from MIL_CREDA.local_term import local_distance, local_loss, total_correspondence
from MIL_CREDA.objective import source_loss, total_objective
from MIL_CREDA.renyi import quadratic_entropy, trace_normalize
from MIL_CREDA_Benchmark import config
from MIL_CREDA_Benchmark.mlp import MLP_BACKBONES, build_mlp

#: The two domains Eq. (13) names, and the only two values any caller may
#: state at an embedding boundary (spec `per-domain-encoders`, "No embedding
#: without a declared domain"). Not named `DOMAINS` -- `config.DOMAINS`
#: already means the three datasets M/U/S (`harness.py:873`), and a second
#: `DOMAINS` here would be a collision a reader has to work out.
SIDES = ("source", "target")


def _target_bag_entropy(K_tt: torch.Tensor) -> float:
    """r25:1087's prescribed target-degeneracy diagnostic (spec
    `target-degeneracy-diagnostic`): `quadratic_entropy(trace_normalize(K_tt))`
    over the UNCONDITIONAL target bag Gram, with no new mathematics -- Eq. (9)
    and Eq. (10) as `MIL_CREDA.renyi` already implements them, read off the
    matrix `_milcreda_term` already builds.

    Reports rather than raises at both of the boundaries the composition can
    reach: `trace_normalize`'s `ValueError` on a non-positive trace, and
    `quadratic_entropy`'s own `ValueError` on a zero Frobenius norm. A
    diagnostic that crashes exactly when the failure mode it exists to catch
    is at its most severe is the opposite of what a diagnostic is for.
    `float("nan")` is this repository's established absent-reading sentinel
    (`harness.accuracy`, `latent.geometry`), not `0.0` -- `0.0` is exactly the
    collapsed value this diagnostic reports, so it cannot also mean "no
    reading available" without becoming indistinguishable from the failure it
    exists to flag.

    Read-only: `.detach()` before leaving `torch`, exactly as every other
    reported (never optimized) float in `training_step`'s own return dict
    already does -- this reading is never a term of Eq. (39).
    """
    try:
        A = trace_normalize(K_tt)
    except ValueError:
        return float("nan")
    try:
        return float(quadratic_entropy(A).detach())
    except ValueError:
        return float("nan")


def _off_diagonal_mean(K: torch.Tensor) -> float:
    """Decision 10's instrument -- the mean of every entry off the main
    diagonal, applied identically to `K_ss`, `K_st` and `K_tt` rather than
    written three times (the file's own rule against two copies where one
    differs from the other only in which matrix it reads). A bag's kernel
    with itself sits on the diagonal of `K_ss`/`K_tt` and is always close to
    the kernel's own ceiling regardless of how the cloud is spread, which
    would bias the mean toward "healthy" independently of the reading this
    exists to make visible.
    """
    n, m = K.shape[0], K.shape[1]
    K = K.detach()
    if n <= 1 or m <= 1:
        return float(K.mean()) if K.numel() else float("nan")
    off_diagonal = ~torch.eye(n, m, dtype=torch.bool, device=K.device)
    return float(K[off_diagonal].mean())


def _off_diagonal_mean_live(K: torch.Tensor) -> torch.Tensor:
    """`_off_diagonal_mean`'s identical computation, with no `.detach()` and
    no cast to `float` -- a LIVE tensor, gradient attached.

    A second spelling of one number is normally the defect this repository
    calls out repeatedly, and that rule still holds for every DIAGNOSTIC
    reading in this file -- `sourceKernelMean`/`crossKernelMean`/
    `targetKernelMean` have no business carrying gradient, since nothing
    trains on a number only ever printed. `_target_dispersion` is not one of
    those: once it is a divisor the loss itself uses, it is part of the
    objective, and `_off_diagonal_mean`'s own `.detach()` -- correct for
    every OTHER caller, which is why it is not removed there -- would sever
    the one gradient path this function exists to keep open. So the
    computation is kept twice on purpose, named apart, each serving the
    caller its own contract demands: `_off_diagonal_mean` for every
    diagnostic reader, this one for `_target_dispersion` alone.
    """
    n, m = K.shape[0], K.shape[1]
    if n <= 1 or m <= 1:
        return K.mean() if K.numel() else torch.tensor(float("nan"))
    off_diagonal = ~torch.eye(n, m, dtype=torch.bool, device=K.device)
    return K[off_diagonal].mean()


def _target_dispersion(K_tt: torch.Tensor) -> torch.Tensor:
    """The PROBE's own reading (`config.NORMALIZE_LOCAL_BY_DISPERSION`): the
    mean pairwise squared distance among the target bag representations
    Psi(B_j^t) in H_I,

        mean over i != j of ( K_tt[i,i] + K_tt[j,j] - 2*K_tt[i,j] )

    read off `K_tt` -- already built by `_milcreda_term` -- rather than
    recomputed from scratch. For `n` bags there are `n*(n-1)` ordered
    off-diagonal pairs `(i, j)`; each diagonal entry `K_tt[i,i]` occurs in
    exactly `n-1` of them (once per choice of the other index), so the first
    two terms' average over those pairs is `K_tt`'s own diagonal mean, taken
    twice over, and the third term's average is exactly `_off_diagonal_
    mean_live(K_tt)`. Hence:

        dispersion = 2 * (diagonal_mean(K_tt) - _off_diagonal_mean_live(K_tt))

    **A LIVE tensor, deliberately NOT detached -- this is the correction to
    an earlier version of this probe, which detached it.** `L = d^2 / D`
    with `D` detached gives `dL/dtheta = (1/D) * d(d^2)/dtheta`: descent
    lowers `d^2` with a step scaled by `1/D`, and as the cloud collapses
    `D -> 0` so that scale explodes -- a detached denominator is a POSITIVE
    FEEDBACK LOOP into the exact collapse this probe exists to resist, not
    merely "no anti-collapse force." With `D` live, `dL/dtheta =
    (1/D) * d(d^2)/dtheta - (d^2/D^2) * dD/dtheta`; the second term is the
    entire remedy -- it pushes to INCREASE `D`, i.e. to spread the target
    representations apart, which is the one thing a detached divisor can
    never do. `tests/test_local_dispersion.py`'s own
    `test_the_dispersion_denominator_carries_gradient_and_resists_collapse`
    measures exactly this: a gradient step must increase the measured
    dispersion when `d_j^2` is held fixed.

    `K_tt` itself must reach here with `requires_grad` still true -- i.e.
    computed in the ordinary forward pass, never under `torch.no_grad()` or
    behind an earlier `.detach()` in the call chain -- or this correction is
    inert. `_milcreda_term` builds it in the ordinary forward path with no
    such break.

    The diagnostic `self._diagnostics["targetDispersion"]` still wants a
    plain, detached number to report; `_dispersion_normalized_local_term`
    reads this same live tensor twice -- once live, for the loss, once
    `.detach()`ed, for the diagnostic -- rather than computing it twice.
    """
    diagonal_mean = K_tt.diagonal().mean()
    return 2.0 * (diagonal_mean - _off_diagonal_mean_live(K_tt))


def _dispersion_normalized_local_term(
    distances: torch.Tensor, K_tt: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """The PROBE's own local term: Eq. (38)'s aggregation, with `d_j^2`
    divided by `_target_dispersion(K_tt)` instead of the revision's constant 2.

    Mirrors `MIL_CREDA.local_term.local_loss`'s aggregation precisely -- the
    uniform mean, over losses read off the measured dispersion rather than
    Eq. (38)'s constant bound.

    **It mirrored a confidence-weighted aggregation until 2026-10-18, and
    following r33 here is not cosmetic.** This probe exists to isolate ONE
    change, the divisor, which is the same reason
    `config.NORMALIZE_LOCAL_BY_DISPERSION` is a run-level flag and not a
    per-arm one: "a per-arm flag would make two adjacent rungs of the ladder
    differ in the component under test AND in the normalization, so no
    difference between them could be attributed to either one". A probe left on
    r31's aggregation would reintroduce exactly that confound at the moment
    somebody turns it on -- the divisor and the weights would move together and
    neither could be read off the result. `target_weights` left the signature
    with the weights, and `w.sum()`'s stabilizer with the denominator it held
    up; `config.EPSILON_LOCAL` remains below, now stabilizing only the measured
    dispersion, which is a different divisor with a real risk of reaching zero.

    `dispersion` is LIVE here -- see `_target_dispersion`'s own docstring
    for why a detached divisor amplifies collapse instead of resisting it --
    so `local_term` carries a gradient path to `K_tt` through the
    denominator as well as through `distances`. The returned second element
    is a `.detach()`ed COPY of the same number, for `_milcreda_term`'s own
    `self._diagnostics["targetDispersion"]`: a diagnostic has no business
    holding a graph open, and reading the live tensor once rather than
    recomputing it is what keeps the live and the reported numbers from
    ever being able to disagree.

    **No `[0, 1]` guarantee survives this division** -- see `config.
    NORMALIZE_LOCAL_BY_DISPERSION`'s own docstring. The ratio is unbounded
    above whenever the measured dispersion is small, which is exactly the
    regime a probe of collapse has to visit, and nothing here clamps it to
    recover a bound Eq. (38) never claimed for this divisor.
    """
    dispersion = _target_dispersion(K_tt)
    normalized = distances / (dispersion + config.EPSILON_LOCAL)
    local_term = normalized.mean()
    return local_term, float(dispersion.detach())


@dataclass
class Pool:
    """One domain's training bags, already on the device the run uses."""

    images: torch.Tensor      # (n_images, 3, 32, 32)
    members: torch.Tensor     # (n_bags, 30) indices into `images`
    labels: torch.Tensor      # (n_bags,) observed classes; target labels are never read

    def take(self, positions: torch.Tensor) -> torch.Tensor:
        """The instances of the chosen bags, as (B, 30, 3, 32, 32)."""
        return self.images[self.members[positions]]


def _resolve_encoder(reference_images: torch.Tensor) -> nn.Module:
    """Which encoder `config.BACKBONE` builds, resolved through the plain
    module-level `FeatureExtractor` name on the resnet/vit path -- never a
    local alias, a second import, or a call through this module's own name --
    so the six existing test files that `monkeypatch.setattr(wiring,
    "FeatureExtractor", ...)` keep intercepting construction exactly as they
    did before this resolver existed.

    `reference_images` is read only on the MLP path, to size its input layer
    to this transfer's own instance dimensionality (2 or 3 for a toy domain);
    the resnet/vit path ignores it completely, exactly as construction did
    before `config.BACKBONE` could ever name an MLP.
    """
    if config.BACKBONE in MLP_BACKBONES:
        return build_mlp(config.BACKBONE, reference_images.shape[-1])
    return FeatureExtractor(backbone=config.BACKBONE, pretrained=config.PRETRAINED)


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

        self.encoder = _resolve_encoder(source.images)
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
            # r31 states `||v_R||_1 <= 1` as a CONSTRAINT of Eq. (15) and then
            # says how it is maintained -- a projection after each optimizer
            # step. Maintenance is not establishment: `std=0.1` over
            # `ATTENTION_WIDTH = 128` coordinates draws a norm near 11, so
            # without this line the FIRST forward would run at `|R_phi| <= 11`
            # and the bound Eq. (15) claims would be false exactly where no
            # step has yet had the chance to restore it.
            #
            # It also keeps r31 agreeing with r30 where the two should agree:
            # r30's forward divided by that same norm, so the selector sat at
            # the ball's edge from the start. Projecting here leaves the
            # intended difference -- that the optimizer now sees the raw
            # parameter -- as the only one between the revisions.
            #
            # Consumes no generator state, so the draw order every other
            # parameter depends on is untouched.
            project_onto_l1_ball(self.v_R)
        # Selection arms (SU, SA, SK) are removed. Every declared arm's
        # `spec["selection"]` is `None`, so `select` always returns every
        # instance of a bag unchanged; this branch stays only as the
        # construction that would feed a future arm declaring one again.

        # How often r31's Eq. (15) projection was asked, and how often it bound.
        # Plain ints, assigned unconditionally so every arm answers the question
        # (an arm with no learned selector answers 0/0 rather than raising), and
        # deliberately NOT tensors or parameters: they are a tally the harness
        # keeps, not a quantity any equation reads. Nothing here consumes the
        # global generator, which the `target_encoder` note below depends on.
        self.projection_calls = 0
        self.projection_firings = 0

        if spec["adaptation"] == "creda":
            # lambda_creda is one because the coefficient is applied outside, from
            # the shared schedule; leaving it here as well would apply it twice.
            # No declared arm (B, E, F, G, SU, SA, SK) carries "creda" here as
            # `spec["adaptation"]` -- this is this benchmark's own harness code,
            # not `src/CREDA/`'s prior work, and no declared arm reaches this
            # branch today. It is kept rather than deleted because a future arm
            # could still declare `"adaptation": "creda"` and land here.
            #
            # `bounds_form` and `balance_mass` are deliberately NOT named here,
            # which is the opposite of what `CREDA.training_pipeline` does. That
            # is a ruling, not an oversight: the image benchmark is going to be
            # replaced by the toyset benchmark, so freezing it at r25 is not
            # worth it, and an arm landing here should compute whatever the
            # constructor currently defaults to -- r31's `entropy-cosh`/`trace`
            # since 2026-10-17. Whatever this branch computes is meant to move
            # with the formulation.
            #
            # What that move costs here, measured rather than assumed, is
            # nothing on two independent counts. First, this construction leaves
            # `normalize_by_bounds` and `balance_mix_blocks` at `False`, and
            # those are the only flags under which either keyword is read: on a
            # fixed seeded batch the loss is bit-identical under both defaults,
            # at `-0.6323614716529846` for `weighting=True` and
            # `0.058492064476013184` for `weighting=False`. Second, every
            # `adaptation` value declared anywhere -- `config.ARMS`,
            # `config.SEARCH_ARMS`, `toy_run.TOY_ARMS` -- is `"milcreda"` or
            # `None`, so no arm constructs this at all. Nothing in any reported
            # number moved.
            self.creda = CREDALoss(sigma="auto", lambda_creda=1.0,
                                   use_entropy_weighting=spec["weighting"])

        # Eq. (13)'s second encoder, theta_t -- constructed LAST, through the
        # same module-level `FeatureExtractor` name `self.encoder` used above
        # (never a local alias, a second import, or `copy.deepcopy`: six test
        # files patch this exact name and both constructions must see the
        # patch). Only where an adaptation term exists to train it: the floor
        # (`B`) carries no adaptation term and gets `None` rather than an
        # untrained, frozen-at-initialization module.
        #
        # LAST matters (design Decision 3): constructing a `FeatureExtractor`
        # consumes the global generator, so building this one before the head
        # or the attention parameters above would shift their draw and put a
        # SECOND difference on the `B->E` rung (`config.py:944-946`'s
        # attributability rule) beside the encoder count itself. Constructing
        # it last means every parameter `B` and an adapted arm share is drawn
        # from identical generator state, and only the module `B` does not
        # have consumes state afterwards.
        self.target_encoder = (
            _resolve_encoder(source.images)
            if spec["adaptation"] is not None else None
        )

    # ------------------------------------------------------------ representation

    def instance_embeddings(self, bags: torch.Tensor) -> torch.Tensor:
        """Encode every instance of every bag: (B, m, ...) -> (B, m, d).

        The encoder never sees a bag. It is applied to each instance separately,
        as Eq. (13) states, and identically in both units.
        """
        B, m = bags.shape[0], bags.shape[1]
        flat = self.encoder(bags.reshape(B * m, *bags.shape[2:]))
        return flat.reshape(B, m, -1)

    def _kernel_embedding(self, H: torch.Tensor) -> torch.Tensor:
        """The embedding the instance kernel kappa^I (Eq. 14) evaluates on --
        the one fork `config.NORMALIZE_KERNEL_EMBEDDING` governs.

        Every caller that feeds an embedding to `gaussian_kernel`, whether
        directly (`bag_kernel`, through `bag_kernel_matrix`) or through
        `relevance_logits`'s own consensus term, routes it through here
        first. Every caller that feeds an embedding to `bag_embedding` --
        and so to `self.head` -- never does: Eq. (19)'s convex combination is
        what the head trains on, and the head is the one consumer this flag
        must leave alone (see `config.NORMALIZE_KERNEL_EMBEDDING`'s own
        docstring for why). Routing happens at each kernel call site rather
        than once where the embedding is produced, precisely so the two
        consumers can keep seeing different views of the same `H`.

        With the flag off this returns `H` itself, unmodified -- not even a
        clone -- so every path that never calls this method differently is
        bit-identical to before this fork existed.
        """
        if not config.NORMALIZE_KERNEL_EMBEDDING:
            return H
        return F.normalize(H, p=2, dim=-1)

    def _kernel_pairs(self, pairs: list[tuple]) -> list[tuple]:
        """A `bags_of`-shaped list of `(H, weights)` with every embedding
        routed through `_kernel_embedding`; the weights travel unchanged,
        since `_kernel_embedding` never changes which instance a weight
        belongs to. Used wherever a whole bag list reaches `bag_kernel_matrix`
        or a per-bag `bag_kernel` call rather than `bag_embedding`.
        """
        return [(self._kernel_embedding(H), w) for H, w in pairs]

    def weights_for(self, H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """The in-bag weights beta of Eq. (16), over whatever instances survive.

        `sigma` is Decision 1's one constant bandwidth -- `self.sigma` at every
        call site, resolved once at construction from the search's override or
        `config.KERNEL_SIGMA` -- passed explicitly and with no default, so a
        caller supplying a different value is a visible, deliberate choice
        rather than something this method could quietly default to.

        `relevance_logits`'s consensus half evaluates the instance kernel, so
        the embedding handed to it is `self._kernel_embedding(H)` rather than
        `H` itself -- the fork `config.NORMALIZE_KERNEL_EMBEDDING` governs.
        """
        if self.spec["attention"] == "learned":
            logits = relevance_logits(
                self._kernel_embedding(H), self.V_R, self.b_R, self.v_R,
                self.attention_gamma, sigma,
            )
            return bag_weights(logits, self.attention_temperature)
        # uniform: Eq. (19) with beta = 1/m
        return torch.full((H.shape[0],), 1.0 / H.shape[0],
                          dtype=H.dtype, device=H.device)

    def select(self, H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """The instances of ONE bag this arm is allowed to look at: (m, d) -> (k, d).

        **Selection happens HERE and nowhere else**, which is the property the
        whole comparison rests on: the kernels, the attention, the bag
        representation and the decision at evaluation all see the same
        instances, so a difference between two rules is the CHOICE and never
        two halves of the model disagreeing about which bag they hold.

        The three rules were removed on 2026-10-12 with the `SU`/`SA`/`SK` arms
        and restored on 2026-10-20 for the aggregation section, which asks
        whether the attention knows which instances matter. The seam itself
        never left, so this is a declaration and not a rewiring.

        - `topk` keeps the `k` instances the arm's OWN weights rank highest.
        - `arbitrary` keeps `k` at random. It is the control: at the same `k`,
          `topk` has to beat it or the attention is scoring nothing.
        - `regular` keeps `k` evenly spaced, which separates "top-k wins" from
          "fewer instances already helps".

        `None` keeps the whole bag and is what every arm declared before this
        date does. The budget is the arm's own `selectK`; it is declared before
        the run at 5, 10 and 15 of 30, and all three are reported.
        """
        rule = self.spec["selection"]
        if rule is None:
            return H
        budget = self.spec.get("selectK")
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(
                f"the selection rule {rule!r} needs a positive integer "
                f"`selectK`, and this arm declares {budget!r}")
        m = int(H.shape[0])
        if budget >= m:
            # Not a silent no-op: a budget at or above the bag size means the
            # comparison this arm exists for is not running, and reading the
            # same number under three rules would read as three agreeing
            # measurements of one thing.
            raise ValueError(
                f"`selectK` is {budget} and the bag holds {m} instances, so "
                "this rule selects nothing and the three rules would be "
                "byte-identical")
        if rule == "topk":
            # The arm's OWN weights, which is the whole question: if the
            # attention scores anything, its best `k` has to beat `k` at
            # random. `weights_for` is called rather than reimplemented, so
            # the ranking is the one the bag representation would have used.
            order = torch.argsort(self.weights_for(H, sigma).reshape(-1),
                                  descending=True)
            keep = order[:budget]
        elif rule == "arbitrary":
            # The control. Its generator is derived from the bag's own
            # contents and never from a global seed, so two bags of one batch
            # draw differently while the same bag draws the same instances
            # every time it is seen -- a draw that moved between steps would
            # add a second difference to every comparison that reads it.
            seed = int(torch.sum(H.detach().reshape(-1)[:8].double() * 1e6)) % (2**31)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            keep = torch.randperm(m, generator=generator)[:budget].to(H.device)
        elif rule == "regular":
            # Evenly spaced over the bag's own order: the third foot that
            # separates "top-k wins" from "fewer instances already helps".
            step = m / budget
            keep = torch.tensor([int(i * step) for i in range(budget)],
                                device=H.device, dtype=torch.long)
        else:
            raise ValueError(
                f"unknown selection rule {rule!r}: the declared rules are "
                "`topk`, `arbitrary` and `regular`, or None to keep the bag")
        # Sorted, so the kept instances arrive in the bag's own order whichever
        # rule chose them. Unsorted, `topk` would hand the kernels a
        # permutation the other two never produce, and a difference between
        # rules could be the ORDER rather than the choice.
        return H[torch.sort(keep).values]

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

    def _scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Bag-level scores from already-encoded instances, whatever unit the
        arm trains on -- the bag/instance branch `forward` and `score_target`
        share, so Eq. (19) and the instance-averaging rule live in exactly one
        place regardless of which domain's encoder produced `embeddings`.

        The comparison happens at the bag because it is the only unit where both
        families decide without inventing a rule. An instance-unit arm reaches it
        by averaging the class distributions of its 30 instances; the logarithm of
        that average is returned so cross-entropy over it is exactly the negative
        log-likelihood of the averaged decision.
        """
        if self.spec["unit"] == "bag":
            Z, _ = self.bag_representations(embeddings, self.sigma)
            return self.head(Z)
        probabilities = F.softmax(self.head(embeddings), dim=-1).mean(dim=1)
        return torch.log(probabilities + config.EPSILON)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """Bag-level scores through the SOURCE encoder (`theta_s`) --
        `harness.accuracy`'s path, and every `nn.Module.__call__` caller's.

        Signature unchanged from before Eq. (13) split the encoder: a domain
        argument here would either default (reintroducing the exact failure
        this change removes) or make `model(x)` a `TypeError`, breaking `Arm`
        as a plain `nn.Module` and editing the floor guard at
        `tests/test_arm_objectives.py:429`/`:443`. Evaluation through the
        target encoder gets its own entry point instead: `score_target`.
        """
        return self._scores(self.instance_embeddings(bags))

    def score_target(self, bags: torch.Tensor) -> torch.Tensor:
        """Bag-level scores through the TARGET encoder (`theta_t`) --
        `harness.target_accuracy`'s path.

        On the floor (`B`), `target_embeddings` falls back to `self.encoder`,
        so this reads exactly what `forward` reads: the floor's guarantee
        (Decision 2) is unchanged, and it predicts on target through the same
        source-trained encoder it always did.
        """
        return self._scores(self.target_embeddings(bags))

    # ------------------------------------------------------------- adaptation

    def _draw_target(self, generator: torch.Generator) -> torch.Tensor:
        """The step's unlabelled target bags. Plain: nothing is stratified by a
        label the method is not allowed to see."""
        order = torch.randperm(self.target.members.shape[0], generator=generator)
        # The generator stays on the host so a run is reproducible whatever the
        # device is; the indices move to wherever the bags actually live.
        return order[: config.BAGS_PER_STEP].to(self.target.members.device)

    def project_constraints(self) -> bool:
        """Re-impose Eq. (15)'s `||v_R||_1 <= 1` after an optimizer step.

        r31 states the constraint as a max norm applied "tras cada paso de
        optimizacion", so this is called by the training loop right after
        `optimizer.step()` -- the same position, and for the same reason, as
        `sync_target_from_source` above: the rule counts OPTIMIZER steps, and a
        projection applied before the update would constrain the parameter the
        step is about to move and leave the moved value unconstrained.

        **The rescale itself is not implemented here.** Eq. (15) lives in
        `MIL_CREDA.attention`, and an arm that wrote the projection inline
        would have forked the formulation: the module could then move to a
        later revision and this copy would not, which is invisible to every
        check that reads the two declarations separately. This method decides
        only WHETHER there is a selector to project and keeps the tally.

        Returns whether the constraint bound on this call. An arm with no
        learned selector returns `False` without touching a tensor -- the
        pooling arms (`max`, `mean`) have no `v_R`, so there is nothing whose
        norm could exceed one.
        """
        self.projection_calls += 1
        if not hasattr(self, "v_R"):
            return False
        fired = project_onto_l1_ball(self.v_R)
        if fired:
            self.projection_firings += 1
        return fired

    def sync_target_from_source(self) -> None:
        """Transmit `theta_s`'s parameters into `theta_t`. A no-op on the floor.

        **Procedure, never a term.** Under `no_grad` and in place: the copy
        builds no graph, so the adaptation term's gradient never reaches
        `theta_s` through `theta_t`, and the optimizer keeps its own state
        because the tensors are the same objects it was built over. Eq. (13)
        is untouched -- the two encoders remain distinct tensors sharing none,
        and they diverge again from the next step on.

        **Only parameters. The buffers stay the target's own.** Measured: at
        the shortest interval, carrying the normalization statistics over as
        well was WORSE (0.7222 against 1.0000). Running statistics are a
        reading of the material each domain actually saw, not a parameter of
        the map -- `named_parameters()` is what this walks, and that is the
        whole distinction.

        Lives on the `Arm` rather than in `harness.run_one` because whether
        there is anything to transmit into is the arm's own shape: a floor
        declares no adaptation and therefore no `target_encoder`, and a loop
        that had to ask would be carrying a fact the model already holds.
        """
        if self.target_encoder is None:
            return
        source = dict(self.encoder.named_parameters())
        with torch.no_grad():
            for name, parameter in self.target_encoder.named_parameters():
                parameter.copy_(source[name])

    def target_embeddings(self, bags: torch.Tensor) -> torch.Tensor:
        """Encode a batch through the TARGET encoder: (B, m, ...) -> (B, m, d).

        Promoted from the private training-time seam `_target_embeddings`
        (design Decision 2): it is now the place the encoder CHOICE lives, not
        merely a named place to reintroduce a freeze at. Routes through
        `self.target_encoder`, falling back to `self.encoder` on the floor
        (`B`), where `target_encoder is None` -- the floor's settled
        behaviour: it holds one encoder, and both its entry points reach it.

        Every adapted arm passes the target batch through its own target
        encoder exactly as `instance_embeddings` passes the source batch
        through the source one: normalization layers stay in training mode
        and learn from this forward like the rest of the model does. `GN` is
        retired, and with it the one branch that used to stand here and the
        three helpers that served only it (`_source_embeddings`,
        `_encode_with_frozen_stats`, `_batchnorm_modules`). The
        `spec["normalization"]` key went with them: nothing read it once the
        branch was gone, and a declared field nobody reads is a field the next
        reader has to work out and discard. No declared arm normalizes its
        target forward with anything but its own encoder's training-mode
        statistics.
        """
        encoder = self.target_encoder if self.target_encoder is not None else self.encoder
        B, m = bags.shape[0], bags.shape[1]
        flat = encoder(bags.reshape(B * m, *bags.shape[2:]))
        return flat.reshape(B, m, -1)

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
        H_t = self.target_embeddings(target_bags)

        # Decision 1: one constant bandwidth for the whole method -- the
        # consensus term, the top-k ranking, every kernel block below and
        # `local_distance` all read `self.sigma` (`config.KERNEL_SIGMA`, or the
        # ceiling search's override) and nothing else. r25 Eq. (23): "Un unico
        # ancho de banda sigma gobierna los tres bloques, ya que los tres
        # derivan del mismo kernel de instancia."
        sigma = self.sigma
        bags_s = self.bags_of(H_s, sigma)
        bags_t = self.bags_of(H_t, sigma)

        # The fork `config.NORMALIZE_KERNEL_EMBEDDING` governs (see
        # `Arm._kernel_embedding`): every kernel block below -- the three
        # Grams and the local term's own self-similarity further down --
        # reads the kernel's own view of each bag, built once here and
        # reused. `bags_s`/`bags_t` themselves are NEVER routed through it:
        # their raw `H` is what `Z_t` just below, and `bag_representations`
        # elsewhere, hand to `self.head`.
        kernel_bags_s = self._kernel_pairs(bags_s)
        kernel_bags_t = self._kernel_pairs(bags_t)

        K_ss = bag_kernel_matrix(kernel_bags_s, kernel_bags_s, sigma)
        K_st = bag_kernel_matrix(kernel_bags_s, kernel_bags_t, sigma)
        K_tt = bag_kernel_matrix(kernel_bags_t, kernel_bags_t, sigma)

        Z_t = torch.stack([bag_embedding(H, w) for H, w in bags_t])
        G_t = F.softmax(self.head(Z_t), dim=1)                    # Eq. (20)
        w_t = confidences(G_t)                                    # Eq. (24)
        pseudo = torch.tensor([pseudolabel(g) for g in G_t],      # Eq. (22)
                              device=K_ss.device)

        # r33 Eq. (37) sums over EVERY class, not only the active ones, so
        # these two lists carry one entry per class in `C` rather than one per
        # active class: `None` marks a class the aggregation charges `1` for,
        # and `source_counts` carries its `n_c^s`. Both are built here because
        # this loop is the only place the routing and the class index sets
        # exist.
        class_losses = []
        source_counts = []
        for class_id in range(self.classes):
            s_idx = torch.nonzero(source_labels == class_id, as_tuple=False).reshape(-1)
            t_idx = torch.nonzero(pseudo == class_id, as_tuple=False).reshape(-1)
            # `n_c^s` for Eq. (37)'s weight, recorded for every class before
            # any exclusion: a class with no source bag weighs `0 / N_s`, so it
            # contributes nothing whichever branch it takes and needs no
            # special case. A class the SOURCE batch has and the target routing
            # abandoned is the one the charge below is for.
            source_counts.append(int(s_idx.numel()))
            if s_idx.numel() == 0 or t_idx.numel() == 0:
                class_losses.append(None)       # not in C_act (Eq. 37)
                continue
            K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, s_idx, t_idx)
            if self.spec["weighting"]:
                K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w_t[t_idx])
            else:
                K_st_w, K_t_w = K_st_c, K_t_c
            K_mix = mixed_matrix(K_s_c, K_st_w, K_t_w)
            # Unchanged under r29: Eq. (37)'s active set still asks that the
            # three matrices which must be normalized have positive trace, and
            # Eq. (32) divides by `tr(K_s)`, `tr(K~_t)` and their geometric
            # mean -- the same two traces this already reads.
            if not is_active(K_s_c, K_t_w, K_mix):
                class_losses.append(None)       # not in C_act (Eq. 37)
                continue
            # r29 Eqs. (32)-(35): the balanced congruence, the score over its
            # three densities, and the ends read off the same two marginals.
            # This replaces r25's `dependency_score`/`conservative_bounds`
            # pair, whose constant ends ordered a collapsed target ABOVE
            # perfect alignment (see `entropy_cosh_bounds`' docstring).
            breve_s, breve_t, _ = balanced_mixture(K_s_c, K_st_w, K_t_w)
            score = balanced_dependency_score(K_s_c, K_st_w, K_t_w)
            class_losses.append(class_global_loss_cosh(
                score, quadratic_entropy(breve_s), quadratic_entropy(breve_t)))

        # Spec `target-degeneracy-diagnostic`: six readings and one count, all
        # free -- read off matrices this method already builds, with no new
        # mathematics. Stashed on `self` rather than widening this method's
        # return (a 2-tuple four call sites in `tests/test_arm_objectives.py`
        # already unpack) -- `training_step` reads it back immediately after
        # calling this method, on the one path that reaches here.
        self._diagnostics = {
            # r25:1087's prescribed instrument, over the UNCONDITIONAL target
            # Gram -- before any per-class conditioning narrows it, so a
            # collapse uniform across classes (Requirement: "The per-class
            # weighted H_target...MUST NOT substitute") is still visible even
            # once the loop above has at most one active class left to read.
            "targetBagEntropy": _target_bag_entropy(K_tt),
            # Eq. (24)'s magnitude -- neither half of r25's mechanism is
            # diagnostic alone (Decision 9): a collapsed cloud can produce
            # either a vanished mean weight or a maximal one (a confidently
            # wrong single class), and `targetConfidenceMin` catches the
            # onset a mean over ten bags would hide.
            "targetConfidenceMean": float(w_t.mean().detach()),
            "targetConfidenceMin": float(w_t.min().detach()),
            # |C_act| (Eq. 37) -- r25's third named observable, already
            # computed as the length of this loop's own accumulator.
            "activeClasses": sum(1 for loss in class_losses if loss is not None),
            # Decision 10: whether one sigma still describes both clouds,
            # read as the off-diagonal means of the three kernels this method
            # already built above.
            "sourceKernelMean": _off_diagonal_mean(K_ss),
            "crossKernelMean": _off_diagonal_mean(K_st),
            "targetKernelMean": _off_diagonal_mean(K_tt),
            # The PROBE's own reading (`config.NORMALIZE_LOCAL_BY_DISPERSION`):
            # nan whenever this run never computed one -- the local term is
            # off, or the flag is off and Eq. (38)'s constant bound governed
            # instead -- so a reader can never mistake "unmeasured" for "the
            # cloud is a single point" (which is `0.0`, a real reading).
            "targetDispersion": float("nan"),
        }

        # r33 Eq. (37): `L_glob = sum_{c in C} (n_c^s / N_s) * l~_c`, with
        # `l~_c = 1` for every class that is not active. Not gated on the arm's
        # confidence axis, and that is the change: r29's weights were built
        # from `w_j^t` and this call sat behind `spec["weighting"]` for that
        # reason, while r33's come from the OBSERVED source labels and have
        # nothing to do with the confidences. The axis still means what it
        # always meant -- Eq. (26)'s `D`, applied above -- and arm `E`
        # (`MIL-CREDA**`, "lacks the local term and the weighting") still
        # differs from the arms that declare it, at that one site.
        #
        # The empty-`C_act` branch went with the weights. r31 answered zero
        # there, so an encoder that collapsed until one class was left reported
        # a SMALLER term for having abandoned the rest; `global_loss` now
        # charges `1` per abandoned class, which has no gradient and is not
        # meant to: it makes the collapse visible in the value.
        global_term = global_loss(class_losses, source_counts)
        # **The gate the ablation needs, symmetric to `local` below.** An arm
        # that declares `global: False` computes Eq. (37) and then drops it,
        # rather than skipping the class loop: the loop is also what fills
        # `self._diagnostics`, so skipping it would leave a local-only arm
        # reporting `nan` for every kernel reading and the ablation could not
        # be compared against the arms that do report them.
        #
        # `spec.get` and not `spec[...]`: every arm declared before 2026-10-20
        # carries no `global` key and must keep computing it, which is what
        # every number measured so far ran under. A missing key is the old
        # behaviour, never an absent term.
        if not self.spec.get("global", True):
            global_term = torch.zeros_like(global_term)
        local_term = torch.zeros_like(global_term)

        if self.spec["local"]:
            distances = []
            for column in range(len(bags_t)):
                cross = K_st[:, column]
                pi = total_correspondence(cross, source_labels, G_t[column],
                                          self.tau_local)
                # The local term's own kernel evaluation (Eq. 31's
                # self-similarity term): the kernel's view, not the raw one --
                # `bags_t[column]` would hand `bag_kernel` the same raw `H`
                # `Z_t` above hands the head, which is exactly the consumer
                # this fork must not reach.
                H_j, w_j = kernel_bags_t[column]
                distances.append(
                    local_distance(bag_kernel(H_j, w_j, H_j, w_j, sigma), cross, K_ss, pi)
                )
            distances_tensor = torch.stack(distances)
            if config.NORMALIZE_LOCAL_BY_DISPERSION:
                # The PROBE: divide `d_j^2` by the target bags' own measured
                # dispersion in H_I instead of Eq. (38)'s constant 2 -- see
                # `_target_dispersion` and `_dispersion_normalized_local_term`
                # above for the derivation and the aggregation, and
                # `config.NORMALIZE_LOCAL_BY_DISPERSION`'s own docstring for
                # why Eq. (38)'s `[0, 1]` guarantee does not survive it.
                local_term, dispersion = _dispersion_normalized_local_term(
                    distances_tensor, K_tt)
                self._diagnostics["targetDispersion"] = dispersion
            else:
                # r33 Eq. (38): the uniform mean. `w_t` left this call with
                # r32 -- a weight the classifier computes can be lowered on the
                # worst-aligned subject instead of aligning it -- and
                # `config.EPSILON_LOCAL` left with the denominator it
                # stabilized.
                local_term = local_loss(distances_tensor)

        return global_term, local_term

    def _creda_term(self, H_s, source_labels, target_bags):
        """CREDA's own loss, over instances, exactly as it was written."""
        instances = H_s.reshape(-1, self.encoder.output_dim)
        H_t = self.target_embeddings(target_bags).reshape(-1, self.encoder.output_dim)
        y_s = source_labels.repeat_interleave(H_s.shape[1])
        g_t = F.softmax(self.head(H_t), dim=1)
        return self.creda(instances, H_t, y_s, g_t)

    # ------------------------------------------------------------------- step

    def training_step(self, bags: torch.Tensor, labels: torch.Tensor,
                      ramp: float, generator: torch.Generator,
                      ramp_local: float | None = None) -> dict:
        """The arm's own objective, and nothing the arm does not have.

        The supervised term is the arm's. A bag-unit arm calls `source_loss`,
        which is Eq. (21) as the revision states it, normalized by its own
        supremum B_src; an instance-unit arm keeps CREDA's per-instance
        cross-entropy, because prior work is used exactly as it was written.

        `total_objective` (Eq. (39)) always took two coefficients apart --
        `lambda_global` and `lambda_local`, its own signature and docstring say
        so. Until this stretch of work `ramp` alone funded both, passed twice
        (`total_objective(..., ramp, ramp)`): one schedule, one ceiling, applied
        to two terms the equation itself never tied together. That was the
        harness under-applying its own equation, not a property of it.
        `ramp_local` is the missing second coefficient, and it is OPTIONAL
        precisely so every caller that never knew a second coefficient existed
        keeps training exactly as before: omitted, it defaults to `ramp`, and
        `total_objective(..., ramp, ramp)` runs unchanged --
        `test_the_bag_unit_arms_assemble_the_objective_and_never_write_a_term_
        inline` (`tests/test_arm_objectives.py`) pins exactly this default.
        Only MIL-CREDA's own branch reads it; CREDA's one-term objective and the
        floor's supervised-only one have nothing for a second coefficient to
        multiply.

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
        # The default that keeps every caller unaware of the second
        # coefficient training exactly as before -- see the docstring above.
        coefficient_local = ramp if ramp_local is None else ramp_local
        adaptation = torch.zeros((), device=logits.device, dtype=logits.dtype)
        contribution_global = torch.zeros((), device=logits.device, dtype=logits.dtype)
        contribution_local = torch.zeros((), device=logits.device, dtype=logits.dtype)
        # Spec `target-degeneracy-diagnostic` / design Decision 8: every one of
        # these reports `float("nan")`, NEVER `0.0`, wherever the reading is
        # structurally unavailable -- `0.0` is exactly the collapsed value the
        # diagnostic exists to report, so a floor (or CREDA's own one-term
        # objective, which no declared arm reaches) defaulting to `0.0` would
        # read as maximally degenerate on every run forever.
        diagnostics = {
            "targetBagEntropy": float("nan"),
            "targetConfidenceMean": float("nan"),
            "targetConfidenceMin": float("nan"),
            "activeClasses": float("nan"),
            "sourceKernelMean": float("nan"),
            "crossKernelMean": float("nan"),
            "targetKernelMean": float("nan"),
            "targetDispersion": float("nan"),
        }
        if self.spec["adaptation"] == "milcreda":
            target_bags = self.target.take(target_indices)
            global_term, local_term = self._milcreda_term(
                embeddings, labels, target_bags
            )
            diagnostics = self._diagnostics
            adaptation = global_term + local_term
            total = total_objective(                                  # Eq. (39)
                supervised, global_term, local_term, coefficient, coefficient_local
            )
            contribution_global = coefficient * global_term
            contribution_local = coefficient_local * local_term
        elif self.spec["adaptation"] == "creda":
            # CREDA's objective, not Eq. (39): one term, one coefficient, as its
            # own code writes it. `coefficient_local` is never read here --
            # CREDA has no local term for a second coefficient to multiply --
            # and the whole contribution is booked under the "global" slot
            # purely so the sum below still equals it; no declared arm reaches
            # this branch (see `Arm.__init__`'s own note on `"creda"`).
            # `_creda_term` builds no bag Gram at all (CREDA is instance-unit,
            # not bag-unit) -- there is nothing here for the diagnostic to
            # read, so it stays at the nan default declared above.
            target_bags = self.target.take(target_indices)
            adaptation = self._creda_term(embeddings, labels, target_bags)
            total = supervised + coefficient * adaptation
            contribution_global = coefficient * adaptation
        else:
            # The floor: no target image passes through the encoder, ever, in
            # training (Decision 2). `target_indices` is drawn above and
            # deliberately unused past this point. Decision 8: the floor
            # structurally cannot build `K_tt` -- doing so would mean encoding
            # a target image, which Decision 2 forbids -- so it stays at the
            # nan default declared above.
            total = supervised

        return {
            "logits": logits,
            "loss": total,
            # Reported as its own column: if the coefficient scaled the term to
            # irrelevance, the run is talking about the scale and not the methods,
            # and that has to be visible rather than inferred.
            "supervised": float(supervised.detach()),
            "adaptation": float(adaptation.detach()),
            # The two terms' contributions, each on its own coefficient, so
            # "the global term did the work and the local did nothing" reads
            # apart from its reverse instead of collapsing into one number.
            "contributionGlobal": float(contribution_global.detach()),
            "contributionLocal": float(contribution_local.detach()),
            # Kept as the derivable SUM of the two above -- never a smaller,
            # differently-scoped number -- because `MIL_CREDA_Benchmark/
            # __init__.py`'s own `components`/`dimensions` declarations and
            # `tables.py`'s renderers already read `contribution` as a bare
            # total, and neither file is in this stretch's authorized scope to
            # repoint at the split.
            "contribution": float((contribution_global + contribution_local).detach()),
            # Spec `target-degeneracy-diagnostic`: readable during training,
            # not only after -- see `harness.run_one`'s per-epoch aggregation.
            "targetBagEntropy": diagnostics["targetBagEntropy"],
            "targetConfidenceMean": diagnostics["targetConfidenceMean"],
            "targetConfidenceMin": diagnostics["targetConfidenceMin"],
            "activeClasses": diagnostics["activeClasses"],
            "sourceKernelMean": diagnostics["sourceKernelMean"],
            "crossKernelMean": diagnostics["crossKernelMean"],
            "targetKernelMean": diagnostics["targetKernelMean"],
            "targetDispersion": diagnostics["targetDispersion"],
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
