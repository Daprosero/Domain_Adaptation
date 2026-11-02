"""The MIL-CREDA toyset runtime the tutorial notebook trains with.

`MIL-CREDA/Notebooks/Toyset.ipynb` writes its own two training loops and
evaluation, and `MIL_CREDA_Benchmark.tutorial_visuals` draws its figures. What
this module owes them is the rest: CREDA's three hard families of
`CREDA.toy_experiment.DURAS` cut into bags by `toy_bags` (`build_family`,
`material_for`), the bound globals the bag split reads (`material_bound`), the
one declared arm and its hyperparameters (`TOY_ARMS`, `declared_hyper`,
`build_arm`), the protocol pieces the loops call (`supervised_loss`,
`shared_phase_one_state`, `frozen_source_sigma`, `copy_source_to_target`,
`set_phase_two_modes`, `phase_two_objective`, `effective_classes`,
`steps_of_one_epoch`), and the readings the figures are drawn from
(`bag_positions`, `lectura_de_bolsas`, `caminata_de_w`, `_base_pca`,
`_limites`, `_rasterizar`).

The benchmark this module used to run -- the phase runners, the searches, the
sweep and its stored artefact, the section 2-5 experiments, the ablation,
mechanism and selection arms and the old renderers -- was retired with the
split toyset notebooks. Nothing here trains a whole run: the tutorial does.

**One arm, with the confidence weighting ON and Eq. (39)'s two balance
coefficients off.** `TOY_ARMS` declares the toyset's single arm, and reads
nothing of `config.ARMS`/`config.ARMS_BY_ID`, which are the digit benchmark's
declaration of a different experiment. Both phase-2 terms enter at
`FIXED_COEFFICIENT`. What is NOT re-implemented here is a single equation:
every term comes from `MIL_CREDA.*` through `wiring.Arm`, exactly as every
other arm of this benchmark reaches it.

**The protocol is CREDA's two phases.** Phase 1 fits the source encoder, the
attention parameters of Eq. (15) and the head on the supervised term ALONE
(Eq. 21, `supervised_loss`) and freezes them. `copy_source_to_target` then
makes theta_t bit-identical to theta_s -- once -- and phase 2 lets the global
and local terms move theta_t and nothing else, selecting its epoch by
`effective_classes` (CREDA's `cef`, maximized, read once per pass of the
training role and only from `MIN_EPOCHS_2` passes on). Four retired search
dimensions are refused by name: see `refuse_retired_hyper`.

**No `__provenance__`, deliberately**, for `toy_bags.py`'s reason: this is
benchmark-side plumbing and implements no equation of the formulation.

----------------------------------------------------------------------------
Two scales, and what separates them
----------------------------------------------------------------------------

`FULL` declares CREDA's own lengths, measured off `CREDA.toy_experiment`.
`PILOT` is that configuration with the lengths lowered and the MATERIAL
unchanged; the tutorial replaces `PILOT`'s lengths with its own.
`epochs_1`, `epochs_2` and `patience_1` are the lengths, and `patience_1` is
the only criterion among them: it is phase 1's own stop, over SOURCE valid
accuracy. Phase 2 has no patience and runs `epochs_2` whole.

----------------------------------------------------------------------------
Two globals are bent, and this is a bend rather than a parameter
----------------------------------------------------------------------------

`config.CLASSES` is 10 and these families have 3 classes, so that one still has
to be bent. `config.BAGS_PER_CLASS` is 12 and this toyset cuts 27, so that one
is bent too and the bend is no longer a no-op: the two numbers used to agree
because this toyset borrowed the digit rule, and `Scale.bags_per_class` records
why it stopped. `config.INSTANCES_PER_BAG` is 30 and the operator fixed the same
number here, so that one IS bent to the value it already holds -- the code does
not special-case that, because a binding that skipped a global whenever it
happened to agree would be a second code path nothing exercises.
`config.py` is not editable on this path, and `bags.build`/`bags._select` read
every one of them as module globals, so
`material_bound` overrides them -- and `TRAIN_BAGS`, `VALID_BAGS`, `EVAL_BAGS`,
`INSTANCES_PER_BAG`, `BAGS_PER_STEP`, `BACKBONE`, `DOMAINS` and
`bags._raw_dataset` with them -- and restores every one in a `finally`. This is
the pattern `harness.run_one` and `wiring._resolve_encoder` already rely on for
`config.BACKBONE`.

**Where it breaks:** a module global is process-wide. Two of these runs
overlapping in one interpreter -- two threads, or a digit campaign in the same
process -- would read each other's override, and the failure is silent: the
numbers come out of a draw nobody declared. Nothing here can prevent that;
`material_bound` asserts the globals it is about to bend still hold their
declared values on entry, so an overlap is at least caught rather than measured.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from CREDA.toy_experiment import ASPECTO_MAX, DURAS, sigma_de_la_fuente
from MIL_CREDA.confidence import confidences                   # Eq. (24)
from MIL_CREDA.objective import source_loss                       # Eq. (21)
from MIL_CREDA_Benchmark import (bags, config, harness, latent, toy, toy_bags,
                                 wiring)

# --------------------------------------------------------------------- scale


@dataclass(frozen=True)
class Scale:
    """Every number that defines a run of this programme, in one place.

    `name` names the scale. The two instances below differ in the lengths and
    in `points_per_domain`, and the tutorial replaces the lengths with its own.
    """

    name: str
    #: Lengths, BOTH IN EPOCHS since 2026-10-19. `epochs_1` is a CEILING on
    #: phase 1, not a length: the stopping rule ends it earlier, exactly as
    #: `EPOCAS_1` was in CREDA's retired toy runner. `epochs_2` IS the length -- phase 2
    #: has no stopping rule, so it runs its ceiling whole.
    epochs_1: int
    epochs_2: int
    patience_1: int
    #: Material: how many points each domain's generator draws. The ONLY
    #: material field a reduction may lower, and what it has to stay large
    #: enough for is `instances_per_bag` below -- `_instances_per_bag` refuses
    #: a draw whose smallest bag cannot fund it, rather than quietly handing
    #: `bags._select` a bag it will refuse mid-draw.
    points_per_domain: int
    #: Everything below is the programme and is equal at both scales.
    lr_1: float = 1e-2
    lr_2: float = 1e-3
    classes: int = 3
    #: THIS toyset's own number, and no longer the digit benchmark's
    #: `config.BAGS_PER_CLASS`: 27 bags per class, so 81 bags per domain over
    #: CREDA's three classes. It was 4, then 12 (which was the digit rule), and
    #: what each change buys is resolution -- see `realized_split` for the
    #: roles that follow and `toy_bags.BAGS_PER_CLASS_TOY` for the measurement
    #: over several counts that makes 27 the efficient point rather than
    #: simply a larger one.
    #:
    #: **It is a programme field, so both scales carry it**, which is what
    #: makes the pilot a rehearsal of this instrument and not of a coarser one.
    bags_per_class: int = 27
    #: How many instances each bag contributes, DECLARED and not derived. The
    #: digit benchmark's own `config.INSTANCES_PER_BAG`, and a programme field
    #: rather than a repetition one: a bag is the same size at both scales, in
    #: every family, so the unit of work per step is fixed and declared instead
    #: of varying with whichever quantile band came out smallest.
    instances_per_bag: int = 30
    #: CREDA's `FRAC_TRAIN`/`FRAC_VALID`, as REQUESTED. They are fractions of
    #: POINTS there and this benchmark splits BAGS, so what is realized is in
    #: `realized_split` -- reported beside the request rather than quietly in
    #: place of it.
    frac_train: float = 0.6
    frac_valid: float = 0.2

    def realized_split(self) -> dict:
        """How many bags per class each role actually gets, and the fractions
        that follow.

        `frac_train`/`frac_valid` cannot be honoured at bag granularity: with 27
        bags per class, 0.6/0.2/0.2 asks for 16.2/5.4/5.4 bags. The attainable
        split nearest the request with every role covering every class is
        16/5/6, and both numbers travel so nobody reads the request as the
        measurement.

        **What that realizes, as the number a reader compares two cells
        with: 18 test bags and a step of 5.56 pp.** The published accuracy is
        read over `test` in every class, so it takes one of 19 values and two
        cells cannot differ by less than 100/18 points. The previous cut of 12
        realized 7/2/3, so 9 test bags and 11.11 pp.

        **Which fractions these are matters and is not the digit benchmark's.**
        `frac_train`/`frac_valid` are CREDA's `FRAC_TRAIN`/`FRAC_VALID`. The digit campaign's own bag split is a different pair --
        `TRAIN_BAGS`/`VALID_BAGS`/`EVAL_BAGS` of 64/20/36 over 120 bags, so
        0.5333/0.1667/0.3000 -- and at 27 bags per class those would ask for
        14.4/4.5/8.1 and land on 14/5/8. The two readings differ by two bags
        per class in train and in test, and this path keeps CREDA's because
        that is what this scale declares and what the pin holds it to.

        **The third role is named `test`, which is CREDA's own word for it.**
        It is the role the tutorial reads its test accuracies on. The bagset
        this split feeds keeps `bags.BagSet.eval_idx` -- that attribute is the
        digit benchmark's as much as this one's and renaming it is not this
        toyset's to do -- so the role's name and that attribute's name differ
        on purpose.

        **This role absorbs the rounding, and it is computed last for that
        reason.** `train` and `valid` round to the nearest whole bag and `test`
        is the complement, so the 0.4 of a bag `valid`'s 5.4 gave up and the
        0.2 `train`'s 16.2 gave up both land on `test` -- which is why `test`
        gets 6 where the request asked for 5.4. `bags._split_counts`' own docstring describes the same ordering
        and reads as though the complement were the role PROTECTED from the
        remainder; it is the role that receives it. Said plainly here because the
        verdict rests on that role and a reader is owed which direction the
        rounding went.

        `bags._share` never sees a remainder on this path at all: `material_bound`
        sets `config.TRAIN_BAGS`/`VALID_BAGS`/`EVAL_BAGS` to a PER-CLASS count
        times `classes`, so each total divides by the class count exactly and
        every class gets the identical 16/5/6. The per-class draw `_split_counts`
        performs -- the reason the local correspondence is defined at all, since
        it is undefined for a class with no source bag -- therefore cannot drop
        a class here even in principle.
        """
        train = max(1, round(self.bags_per_class * self.frac_train))
        valid = max(1, round(self.bags_per_class * self.frac_valid))
        evaluation = self.bags_per_class - train - valid
        if evaluation < 1:
            raise ValueError(
                f"the requested split {self.frac_train}/{self.frac_valid} "
                f"leaves {evaluation} evaluation bags per class out of "
                f"{self.bags_per_class}; every role must cover every class")
        return {
            "requested": {"train": self.frac_train, "valid": self.frac_valid,
                          "test": 1.0 - self.frac_train - self.frac_valid},
            "bagsPerClass": {"train": train, "valid": valid, "test": evaluation},
            "realized": {"train": train / self.bags_per_class,
                         "valid": valid / self.bags_per_class,
                         "test": evaluation / self.bags_per_class},
        }


#: The scale CREDA's retired toy runner declared -- `EPOCAS_1`, `EPOCAS_2`,
#: `LR_1`, `LR_2`, `PACIENCIA_1`, `FRAC_TRAIN`, `FRAC_VALID` -- written here as
#: values, because those constants left `CREDA.toy_experiment` with that
#: runner. Declared, not run: nothing here releases it.
#:
#: **`points_per_domain` is the ONE field of this scale that is no longer
#: CREDA's, and that departure is forced rather than chosen.** It read
#: `toy_bags.PUNTOS_POR_DOMINIO` -- CREDA's own 1800, the number its
#: three-families figure draws -- and 27 bags per class of 30 declared
#: instances cannot be funded by it: measured over the thirteen seeds either
#: scale uses, 1800 leaves a smallest quantile band of 20 points and
#: `_instances_per_bag` refuses on all 39 (family, seed) pairs. The floor is
#: 2586, also measured over those thirteen seeds.
#:
#: What is written here instead is the DENSITY this scale has always drawn,
#: carried to the new bag count: 1800 over 36 bags is 50 points per bag, and
#: 81 bags at 50 is 4050. Measured at 4050, the smallest band over all 39
#: pairs is 46, against the 30 the programme declares. CREDA has no bags at
#: all -- it works on points -- so there is no bagged number of its own to
#: read here, and the density is the strongest tie that survives the cut.
FULL = Scale(
    name="full",
    epochs_1=100, epochs_2=100,
    patience_1=30,
    points_per_domain=4050,
)

#: The same programme, short enough to be wrong in minutes. What makes the
#: restore path run here is `epochs_2` holding enough passes for the criterion
#: to choose; the paragraph on it below derives the number.
#:
#: **`points_per_domain` is 2835 and the declared bag size is what forced it.**
#: 81 bags of 30 instances need 2430 instances per domain, and the bands are
#: quantiles of a draw rather than equal shares, so the arithmetic is not the
#: constraint -- the smallest band is. Measured over the thirteen seeds either
#: scale uses, with the same procedure that settled the previous 1260: the
#: floor is 2586 (2585 leaves a smallest bag of 29 on `rollo` and `toroide` at
#: seed 0 and refuses), and 2835 leaves 32 on the worst of all 39
#: (family, seed) pairs -- the identical margin the previous 1260 held at 36
#: bags. 2835 is this scale's own density carried to the new cut: 1260 over 36
#: bags is 35 points per bag, and 81 bags at 35 is 2835.
#:
#: So the cost of declaring the bag size is that `points_per_domain` stops
#: being a usable reduction: it may only fall as far as still funds the
#: programme's bag, and below that `_instances_per_bag` refuses instead of
#: running something smaller.
#:
#: What that buys is a pilot whose MATERIAL IS THE FULL RUN'S: a bag holds 30
#: instances at both scales and a domain holds 81 of them at both, so an epoch
#: here processes exactly what an epoch there processes.
#:
#: **`epochs_2` is 4 and the number is DERIVED rather than chosen.** The
#: phase-2 criterion reads once per PASS of the training role, so a run needs
#: enough passes for the criterion to become legal and then choose:
#: `MIN_EPOCHS_2 + 1`. The tutorial refuses fewer than `MIN_EPOCHS_2`.
PILOT = Scale(
    name="pilot",
    epochs_1=6, epochs_2=4,
    patience_1=3,
    points_per_domain=2835,
)


# ------------------------------------------- the phase-2 criterion, in epochs
#
# The phase-2 criterion is read once per pass of the training role, never per
# step. `CREDA.toy_experiment` reads its own phase-2 criterion per epoch; this
# is the same rule with this programme's own epoch, and the sampling needs no
# change --
# `harness.balanced_batches` already keeps a shuffled queue per class and
# refills it when it empties, so with 16 training bags per class 16 steps IS
# one pass.


def steps_of_one_epoch(targets) -> int:
    """How many steps ONE pass of a role has, as `balanced_batches` draws it.

    That sampler yields exactly one bag of every class per step and refills
    each class's queue independently, so a pass is as long as the PER-CLASS
    count -- not as long as the role.

    **It REFUSES when the classes do not agree**, rather than returning the
    first or the largest. CREDA's `pasos_de_una_epoca` refuses the same way
    when its two domains disagree in blocks, and for the same reason: an
    unequal role has no step at which every bag has come out exactly once, so
    wrapping the short queue would leave the property this whole change exists
    for false with nothing failing. The refusal is reachable -- an unequal role
    runs happily and silently forever through that sampler.
    """
    counts: dict[int, int] = {}
    for label in targets:
        counts[int(label)] = counts.get(int(label), 0) + 1
    if not counts:
        raise ValueError("the role holds no bags, so it has no pass")
    distinct = sorted(set(counts.values()))
    if len(distinct) != 1:
        raise ValueError(
            f"the classes of this role do not agree in bag count ({counts}), "
            "so there is no step at which every bag has come out exactly "
            "once; deciding which class defines the pass is a decision about "
            "the experiment and is not taken here")
    return distinct[0]


#: THE MINIMUM, DECLARED. It was DERIVED from `min_steps_2 = 40` -- 40 steps of
#: this programme's pass is `ceil(40 / 16) = 3` of them -- and that input was
#: retired with the step ceiling, so the derivation would now read a number
#: nobody maintains. The VALUE did not move: 3 is what the derivation gave.
#:
#: **It does not equal CREDA's 5, and that is deliberate rather than an
#: oversight.** Both were derived from the same 40-step floor over different
#: passes (16 steps here, 9 there), and declaring them did not reconcile the
#: two numbers, because reconciling them is a decision about the guard and not
#: about the unit. The cross-package pin asserts both are declared in epochs
#: and publishes both values; it does not assert they are equal.
#:
#: Measured, this guard DECIDES rather than guards: over the 90 runs of
#: `Results/Toyset/full`, 38 chose EXACTLY the first legal step -- 42%.
MIN_EPOCHS_2 = 3


# ----------------------------------------------------------------- the toyset

#: One single-character domain code per (family, role). Single characters are
#: not a style choice: `bags.build` seeds its generator with
#: `seed * 1000 + ord(code)`, and `ord` refuses anything longer.
FAMILY_CODES = {
    "lunas": ("l", "L"),
    "toroide": ("o", "O"),
    "rollo": ("w", "W"),
}

#: What each bent `config.DOMAINS` entry is called while a run is bound.
DOMAIN_NAMES = {
    "l": "CredaLunasSource", "L": "CredaLunasTarget",
    "o": "CredaToroideSource", "O": "CredaToroideTarget",
    "w": "CredaRolloSource", "W": "CredaRolloTarget",
}

#: CREDA's `NORMAS`, carried over WHOLE: `ninguna`, `batchnorm`,
#: `layernorm`, in that order, each mapped to the backbone `mlp.py` builds for
#: it. The order is the reading and not an accident of a dict -- every row
#: count, every figure row and every table block downstream is derived from
#: THIS, so a fourth backbone grows the grid without anybody editing a literal.
#:
#: The tutorial trains one of them, `batchnorm`, and the other two stay
#: declared because they are CREDA's own rows and `mlp.py` builds all three.
NORMALIZATIONS = {"ninguna": "mlp-nobn", "batchnorm": "mlp",
                  "layernorm": "mlp-layernorm"}


#: ---------------------------------------------------------------------------
#: The toyset's OWN arm declaration, and why it has to be its own
#: ---------------------------------------------------------------------------
#:
#: `config.ARMS` (`B`, `E`, `F`, `G`) is the DIGIT benchmark's declaration -- a
#: different experiment, over a different material, answering a different
#: question. Until this stretch this module read `tuple(config.ARM_ORDER)` and
#: so borrowed that declaration whole, and the borrowing was the defect rather
#: than merely untidy. Two things make it structural:
#:
#: - `tests/test_benchmark_declarations.py::test_the_declared_arms_and_rungs_
#:   are_exactly_the_operators_four_and_three` PINS those four ids and those
#:   three rungs, and its own docstring says the pin exists "to catch the NEXT
#:   silent reversal of whichever decision is current". Adding an arm there to
#:   serve this toyset would change the digit campaign and trip a pin that is
#:   doing its job.
#: - The protocol is not theirs. `config.ARMS` is a declaration of ARMS, and
#:   the arms are only half of what separates these two runs: the digit
#:   benchmark trains one phase through `harness.run_one` and reads the LAST
#:   epoch, under `harness.ramp`'s warmup to a searched ceiling. This path
#:   trains two phases, restores the best step, and enters Eq. (39) at a FIXED
#:   coefficient of 1 with no schedule at all. Borrowing the declaration would
#:   have tied this protocol's arm list to the other one's.
#:
#: **ONE arm, and the absence is the declaration.** A floor arm `TB` stood here
#: -- the same unit, the same everything, with the adaptation term switched off
#: -- and the operator retired it. So this list has one entry and there is no
#: pair to subtract: what the run reports is this arm's accuracy and nothing
#: says what it is better or worse than. That absence is written down here
#: rather than compensated for, because a replacement comparison invented to
#: fill the hole would answer a question nobody asked, and a floor kept "for
#: reference" would be quoted as one.
#:
#: **The three flags came out equal to digit `G`'s, and the independence is
#: unchanged by that.** With the confidence weighting back on (see the module
#: docstring's note on the mis-read `sin sopesado`) the adapted arm declares
#: `adaptation="milcreda"`, `weighting=True`, `local=True` -- which is exactly
#: `config.ARMS_BY_ID["G"]`'s three. An earlier version of this comment argued
#: the independence from the OPPOSITE fact: that no digit arm carried
#: "local without weighting", so the shape this toyset needed could not be
#: borrowed. That sentence is now false and is NOT what holds the line. What
#: holds it is the two reasons above: the pin, and the protocol.
#:
#: The operator's own framing of what this toyset is for: "eso es de un
#: benchmark que no es el nuestro, despues los conectamos, pero es necesario
#: mirar primero si funciona o no bajo las condiciones que usamos en creda".
#: Connecting the two benchmarks is later work and is not declared here.
#:
#: What this independence does NOT mean: everything the toyset genuinely shares
#: with the digit benchmark it keeps sharing -- `wiring.Arm` itself, the Pools,
#: the material bridge, every equation under `MIL_CREDA.*`. It is the ARM
#: DECLARATION, and only that, that is this module's own.
TOY_ARMS = [
    {
        "id": "TGL", "name": "Toy-MIL-CREDA (global + local, weighted)",
        "label": "global + local, confidence weighting on, both coefficients 1",
        "unit": "bag", "adaptation": "milcreda", "weighting": True,
        # `global` is written out although `_milcreda_term` defaults it to
        # True, so the arm reads what it computes without a default.
        # Behaviourally identical -- `spec.get("global", True)` with the key
        # present and True is the same branch.
        "global": True,
        "local": True,
        "attention": "learned", "selection": None, "normalization": "shared",
        "why": ("what this toyset exists to find out: whether MIL-CREDA works "
                "at all under the conditions CREDA uses. Phase 2 computes BOTH "
                "declared terms -- the global one (Eqs. 32-37) and the local "
                "correspondence (Eqs. 30-31, aggregated by Eq. 38) -- with the "
                "confidence weighting ON: Eq. (26)'s `D`, Eq. (37)'s `omega_c` "
                "and Eq. (38)'s own aggregation all run, and nothing about the "
                "confidence is switched off anywhere. What IS off is Eq. (39)'s "
                "pair of balance coefficients, which the operator's `sin "
                "sopesado los dos loss de la fase 2` was about: both are "
                "exactly 1, fixed, with no ramp (`FIXED_COEFFICIENT`). The "
                "`weighting: False` an earlier stretch declared here came from "
                "reading `sopesado` as the confidence weighting; see the module "
                "docstring."),
    },
]

TOY_ARMS_BY_ID = {arm["id"]: arm for arm in TOY_ARMS}

#: The arms this toyset runs, in declaration order. **Not `LADDER_ARMS`, which
#: is what this used to be called**: with the floor arm retired there is no
#: ladder for it to be the arms of, and a name describing a mechanism the module
#: no longer has is read and believed. `build_arm` resolves every id through
#: `TOY_ARMS_BY_ID` -- never through `config.ARMS_BY_ID`, which is what
#: `wiring.build` would have done.
TOY_ARM_IDS = tuple(arm["id"] for arm in TOY_ARMS)

#: Named rather than indexed at every call site, so the arm carrying the method
#: is read off the declaration above instead of spelled out again. With one arm
#: declared this IS that arm; `TOY_FLOOR`, which named the other end of the
#: retired rung, is gone rather than left pointing at this one.
TOY_ADAPTED = TOY_ARM_IDS[-1]

#: The globals `material_bound` bends, with the value each must still hold on
#: entry. An override that found one already moved would be measuring somebody
#: else's draw, so entry asserts rather than assumes.
BENT_GLOBALS = (
    "CLASSES", "BAGS_PER_CLASS", "INSTANCES_PER_BAG",
    "TRAIN_BAGS", "VALID_BAGS", "EVAL_BAGS", "BAGS_PER_STEP", "BACKBONE",
    # `MLP_OUTPUT_DIM` joined this tuple when `DIM` became a searched
    # dimension. `mlp.MLP` reads the global at CONSTRUCTION, so a searched
    # value can only reach the encoder by being bent for the duration of a
    # binding -- and being in this tuple is what makes the bend asserted on
    # entry and restored on the way out, exactly like the other seven.
    "MLP_OUTPUT_DIM",
)


def family_generators() -> dict:
    """CREDA's three hard families by name, straight from `DURAS`."""
    return dict(DURAS)


def build_family(name: str, seed: int, scale: Scale) -> toy_bags.Familia:
    """One family's two domains, their labels and their bag indices.

    `toy_bags.construir` is the one call: it draws the source from `seed`, the
    target independently from `seed + DESPLAZAMIENTO_DE_SEMILLA`, cuts the bags
    off the UNPERTURBED target, and only then perturbs. It returns a frozen
    `Familia` dataclass -- not a 6-tuple -- and carries `Xt_crudo` as a seventh
    field.
    """
    generators = family_generators()
    if name not in generators:
        raise KeyError(f"no family {name!r}; CREDA declares {sorted(generators)}")
    return toy_bags.construir(name, generators[name], semilla=seed,
                              n=scale.points_per_domain,
                              bolsas_por_clase=scale.bags_per_class)


def datasets_of(familia: toy_bags.Familia) -> dict:
    """The bridge, and the whole of it: each domain as a `toy.ToyDataset` whose
    `.subjects` ARE the bag indices `toy_bags.bolsas` computed.

    `ToyDataset`'s own docstring states the contract `bags.build` needs:
    `.targets` is what `bags._labels` reads, `.subjects` is what `bags._select`
    draws from, and `__getitem__` returns `(tensor, label)`. `_select` then
    walks a class subject by subject in sorted order and hands subject `k` to
    bag `k` of that class, which is why the bag a point lands in downstream is
    the band `bolsas` put it in.
    """
    return {
        "source": toy.ToyDataset(familia.Xs, familia.ys, familia.bs),
        "target": toy.ToyDataset(familia.Xt, familia.yt, familia.bt),
    }


@contextlib.contextmanager
def material_bound(familia: toy_bags.Familia, scale: Scale, backbone: str,
                   hyper: dict | None = None):
    """Bend the globals this toyset needs, and restore every one on the way out.

    **`hyper` is read for the SEARCHED globals only.** Today that is `dim`,
    CREDA's `DIM`: `mlp.MLP` reads `config.MLP_OUTPUT_DIM` at construction, so
    the only way a trial's choice can reach the encoder is for the global to
    hold it while `build_arm` runs. Omitted, every global keeps the value
    `config` declares and this is byte-identical to the pre-search binding --
    which is why every caller that searches nothing passes nothing.

    See the module docstring for why this is a bend and where it breaks. Two
    things make it auditable rather than merely reversible: entry asserts every
    name in `BENT_GLOBALS` still holds the value `config` declares, and
    `bags._raw_dataset` is replaced at the ONE seam that decides which dataset a
    code resolves to -- never `toy.TOY_CODES` and never `toy.build_toy_dataset`,
    which would leave two patches to keep in agreement.
    """
    # The same guard every other entry taking a hyper dict runs: this one
    # READS that dict (`dim`), so a hyper carrying a retired key is refused.
    refuse_retired_hyper(hyper)
    declared = {name: getattr(config, name) for name in BENT_GLOBALS}
    unexpected = {name: value for name, value in declared.items()
                  if value != _DECLARED_GLOBALS[name]}
    if unexpected:
        raise RuntimeError(
            f"these `config` globals are not at their declared values on entry: "
            f"{unexpected}; another run of this module is probably bound in this "
            f"same process, and its draw and this one would be each other's")

    split = scale.realized_split()["bagsPerClass"]
    prepared = datasets_of(familia)
    source_code, target_code = FAMILY_CODES[familia.nombre]
    by_code = {source_code: prepared["source"], target_code: prepared["target"]}

    raw_dataset = bags._raw_dataset
    names_added = []
    try:
        config.CLASSES = scale.classes
        config.BAGS_PER_CLASS = scale.bags_per_class
        config.INSTANCES_PER_BAG = _instances_per_bag(familia, scale)
        config.TRAIN_BAGS = split["train"] * scale.classes
        config.VALID_BAGS = split["valid"] * scale.classes
        # The KEY is the record's role name (`test`, CREDA's word, see
        # `realized_split`); the DESTINATION is the digit benchmark's own field
        # name and stays `EVAL_BAGS`. Two vocabularies meet on this line and
        # neither renames the other.
        config.EVAL_BAGS = split["test"] * scale.classes
        config.BAGS_PER_STEP = scale.classes
        config.BACKBONE = backbone
        config.MLP_OUTPUT_DIM = int(
            (hyper or {}).get("dim") or _DECLARED_GLOBALS["MLP_OUTPUT_DIM"])
        for code, dataset in by_code.items():
            if code not in config.DOMAINS:
                config.DOMAINS[code] = DOMAIN_NAMES[code]
                names_added.append(code)

        def _bound_raw_dataset(code: str, root: Path):
            """The declared families for the two bound codes, and a refusal for
            anything else: a code this binding does not name reaching here would
            otherwise fall through to a digit download."""
            del root
            if code not in by_code:
                raise ValueError(
                    f"domain {code!r} is not bound by this run; this binding "
                    f"carries {sorted(by_code)} for family "
                    f"{familia.nombre!r}")
            return by_code[code]

        bags._raw_dataset = _bound_raw_dataset
        yield by_code
    finally:
        bags._raw_dataset = raw_dataset
        for code in names_added:
            config.DOMAINS.pop(code, None)
        for name, value in declared.items():
            setattr(config, name, value)


#: The values `BENT_GLOBALS` must hold outside a binding, read once at import so
#: a nested or overlapping binding is caught instead of inherited.
_DECLARED_GLOBALS = {name: getattr(config, name) for name in BENT_GLOBALS}


def _instances_per_bag(familia: toy_bags.Familia, scale: Scale) -> int:
    """The scale's DECLARED instances per bag, once the draw is shown to fund it.

    It used to return the smallest bag it found, which made the material's unit
    a measurement: a bag held 150 points in `lunas` and 138 in `toroide` at the
    full scale, 30 at a 360-point pilot, and the number travelled nowhere a
    reader would meet it beside the accuracy it shaped. Two arms fed different
    amounts per step are being compared on their optimizer, and so are two
    families and two scales -- so the unit is declared (`Scale.instances_per_bag`)
    and this measures only whether the draw can pay for it.

    `bags._select` takes exactly `config.INSTANCES_PER_BAG` from every subject
    and refuses a subject that cannot fund it, mid-draw, naming the subject
    rather than the scale. The refusal below is that same fact asked EARLY and
    answered in the scale's own terms: quantile bands differ by several points
    when a class does not divide, so the smallest bag and not the arithmetic
    is what decides, and a scale whose
    smallest bag falls short is a scale that cannot run rather than one that
    runs smaller.
    """
    counts = []
    for labels, indices in ((familia.ys, familia.bs), (familia.yt, familia.bt)):
        for class_id in np.unique(labels):
            for bag in range(scale.bags_per_class):
                counts.append(int(((labels == class_id) & (indices == bag)).sum()))
    smallest = min(counts)
    if smallest < scale.instances_per_bag:
        raise ValueError(
            f"the smallest bag of {familia.nombre!r} holds {smallest} points "
            f"and the {scale.name!r} scale declares "
            f"{scale.instances_per_bag} instances per bag over "
            f"{scale.bags_per_class} bags per class; raise "
            f"`points_per_domain` above {scale.points_per_domain} or lower the "
            f"declared bag, because `bags._select` would refuse this draw "
            f"subject by subject partway through it")
    return scale.instances_per_bag


def material_for(familia: toy_bags.Familia, seed: int, scale: Scale,
                 backbone: str) -> dict:
    """Both domains as `bags.BagSet`s, drawn through `bags.build` itself.

    Nothing about the draw is reimplemented: the permutation, the three disjoint
    roles, the manifest and the contamination seam are `bags.build`'s, and what
    this adds is only which dataset a code resolves to.
    """
    with material_bound(familia, scale, backbone):
        root = Path("/nonexistent-toy-root")   # never touched: see `_raw_dataset`
        source = bags.build(FAMILY_CODES[familia.nombre][0], root, seed)
        target = bags.build(FAMILY_CODES[familia.nombre][1], root, seed)
    return {"source": source, "target": target}


# ------------------------------------------- a measurement that trains nothing

def measuring(measurement):
    """A forward pass taken to MEASURE runs on a deep copy, so it leaves the
    model bit-identical.

    The defect this closes, measured: the retired runner called
    `embedding_scales` -- a diagnostic -- on a freshly built model before
    training that same object in phase 1. A fresh `nn.Module` is in `train`
    mode, so the
    diagnostic's forward re-estimated the encoder's running statistics. The
    doctrine is CREDA's own, at `toy_experiment.py`: a BatchNorm re-estimates
    `running_mean`/`running_var` in the FORWARD, without gradient, so
    `no_grad` is not enough -- it measured the CE over the full source drifting
    3.5e-02 with `||dtheta|| = 0` exact, an encoder frozen in weights and
    moving in behaviour.

    **And the channel is the early stop, not the training.** In `train` mode a
    BatchNorm normalizes by the BATCH statistics, so the running ones never
    enter the training forward and the training-loss column came back
    bit-identical. What they do enter is `harness.accuracy`, which reads in
    `eval` -- so phase 1's early stop chose its best epoch on a criterion a
    diagnostic had contributed to. Measured on `toroide`/`mlp`: `bestEpoch`
    1 -> 0, validation accuracy at epoch 0 0.8667 -> 0.9333, and the weights
    phase 1 then freezes and shares with phase 2 differ by
    `||dtheta|| = 2.548`. Absent under `mlp-nobn` and `mlp-layernorm`, which
    keep no such buffer at all.

    **A copy and NOT `eval`-and-restore, and the choice was measured rather
    than preferred.** Both were authorized. `eval` also leaves the model
    bit-identical, and it additionally RESTATES the reading: these measurements
    are called on a model in `train` mode, where a BatchNorm normalizes by the
    batch, and reading the same rows in `eval` on a freshly built encoder --
    whose running statistics are still `(0, 1)` -- is a different number.
    Measured on `toroide`/`mlp` at `embedding_scales`' own call site:
    `withinBag` 1.6038 -> 0.3570 (0.223x), `betweenBags` 0.8450 -> 0.3886
    (0.460x), `pooled` 1.6926 -> 0.5128 (0.303x). The tutorial's phase-1
    bandwidth is read off `withinBag`, so moving it to `eval` would change it.
    That is a decision about the experiment and not a tidy-up, exactly as `embedding_scales`' own docstring
    says of its `limit`, and it is not this work's to take. On a copy every one
    of the three comes back bit-identical, at both moments the two decorated
    functions are called at -- fresh and in `train`, and after phase 1 in
    `eval`, where the two routes agree anyway.

    **What the copy is NOT a fix for, said out loud.** Reading these in `train`
    mode makes a bag's embedding depend on which other bags shared its chunk --
    the very thing `bag_positions`' docstring refuses for the positions it
    draws. The copy makes the reading harmless; it does not make it
    chunk-independent. Moving it to `eval` is the exit and is the operator's.

    A decorator and not a `with` inside each body, so the claim is visible at
    the signature and the copy has one definition. The model has to be the
    first positional argument and that is CHECKED rather than assumed: a later
    edit that reordered the signature would otherwise leave a decorator that
    silently copies the wrong object and measures the live model anyway.
    """
    @functools.wraps(measurement)
    def wrapper(model, *args, **kwargs):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"`measuring` expects the model as the first positional "
                f"argument of {measurement.__name__} and received "
                f"{type(model).__name__}; a decorator that cannot find the "
                f"model copies nothing and measures the caller's own object")
        return measurement(copy.deepcopy(model), *args, **kwargs)
    return wrapper


# -------------------------------------------------------- the sigma calibration

SIGMA_BAND = (0.05, 0.95)


def _off_diagonal(d: torch.Tensor) -> torch.Tensor:
    return d[~torch.eye(d.shape[0], dtype=torch.bool, device=d.device)]


def _within_bag_median(embeddings: torch.Tensor) -> float:
    """The pairwise median over the set Eq. (14)'s instance kernel evaluates:
    within-bag pairs, pooled over the bags handed in.

    **One definition, two callers**, which is the whole reason this is a
    function rather than two inlined `median()` calls: `embedding_scales`
    reports this as the scale the bandwidth has to serve and
    `frozen_source_sigma` freezes it as the bandwidth itself, and two
    definitions of one statistic are two things nothing obliges to agree.

    **It is CREDA's own formula, restricted to CREDA's own scope.**
    `CREDALoss._median_pairwise` is `sqrt(median(d^2) + 1e-6)` over ONE set, and
    the median commutes with the monotone square root, so over a single bag --
    where "within-bag pairs" and "all pairs" are the same set -- this and
    CREDA's function agree to that epsilon. `test_the_frozen_sigma_reads_the_
    set_the_instance_kernel_evaluates` measures the agreement per bag rather
    than leaving the sentence to be believed. What differs is the SCOPE: the
    median is taken over the UNION of the per-bag pair sets, which is the set
    the kernel reads, and never over the pool of every instance against every
    other, which mixes pairs from different bags and different classes.
    """
    return float(torch.cat([_off_diagonal(torch.cdist(H, H))
                            for H in embeddings]).median())


def _weight_spread(model: wiring.Arm, embeddings: torch.Tensor,
                   sigma: float, device: torch.device) -> float:
    """How far Eq. (16)'s weights sit from uniform, at one bandwidth, averaged
    over the bags handed in.

    The mechanism-level reading the band exists to protect. A bandwidth that
    saturates the instance kernel drives Eq. (15)'s consensus term to the same
    value on every instance of a bag, and a softmax is invariant to a constant
    added to every logit -- so the attention becomes numerically mean pooling
    while every flag still says it is on.

    **It only moves when `attentionGamma` is non-zero**, because the consensus
    term is the only path the bandwidth takes into the logit.
    `config.ATTENTION_GAMMA` is `0.0`, so at the declared hyper this number is
    bit-identical at every bandwidth -- measured, across four orders of
    magnitude. A reader finding this equal across two bandwidths should check
    gamma before concluding the bandwidth does not matter.
    """
    with torch.no_grad():
        return float(torch.stack([
            model.weights_for(H.to(device), sigma).std()
            for H in embeddings]).mean())


@measuring
def embedding_scales(model: wiring.Arm, bagset: bags.BagSet,
                     positions: torch.Tensor, device: torch.device,
                     limit: int = 16) -> dict:
    """The three distances Decision 1's ONE bandwidth has to serve, measured.

    This is where the first version of this function was WRONG, and the error is
    recorded rather than quietly fixed: it measured the median distance over all
    instances pooled together, and that is not the distance the instance kernel
    evaluates. Eq. (14)'s consensus half reads instances WITHIN one bag, and a
    bag is a band of one class, so its spread is much smaller than the pooled
    one -- measured on `lunas`, 0.88 against 1.59. A bandwidth centred on the
    pooled median leaves the within-bag kernel near one and so the attention
    weights of Eq. (16) at a fraction of a percent of uniform: the mechanism is
    numerically off while every flag says it is on.

    - `withinBag` is what Eq. (14) and so Eq. (15)'s consensus evaluate.
    - `betweenBags` is what the bag Gram of Eqs. (32-37) evaluates.
    - `pooled` is every instance against every instance, reported because it is
      what a naive calibration would have used.

    The two the bandwidth has to serve differ by a factor this returns as
    `scaleRatio`, and ONE constant cannot be in band for both. That is Decision
    1's consequence, not a defect to engineer around: it is reported.

    **It goes through `latent.represent` and never through
    `Arm.instance_embeddings` directly.** `tests/test_domain_mixup_guard.py`
    derives, from the syntax tree, that only `wiring.py` and `latent.py` may
    call that method at all -- a target-side reading routed through the SOURCE
    encoder is the one bug no shape check catches, so the call site roster is
    structural. `latent.represent` is the declared seam that states its domain
    and refuses without one, which is exactly what this measurement needs: it
    reads the source bags through theta_s and says so.

    The instance rows come back flattened, so the bag grouping is recovered by
    reshaping -- and that reshape is CHECKED rather than trusted, against
    `represent`'s own bag-unit reading of the same positions. If the flattening
    order were not bag-major the two would disagree, and a within-bag distance
    computed over the wrong grouping is exactly the silent defect this whole
    function exists to have found once already.

    **`limit` is a cap on how many bags are read, and at the declared cut it
    BITES HARD.** It was undocumented while the training role held 6 bags, so
    it never truncated anything and read as a formality. At 12 bags per class
    the role held 21 and the median was taken over the first 16. At the
    declared 27 the role holds 48, so the median is taken over the first 16 of
    48 -- a third of the role, in the role's own order. The value is left where
    it was on purpose -- raising it would move every bandwidth this module has
    ever measured, which is a decision about the experiment and not a tidy-up --
    and the truncation is REPORTED instead: `bagsOffered` is how many were
    handed in, `bagsRead` how many were used, and `bagsTruncated` says whether
    the two differ. A cap that silently starts biting is a measurement that
    quietly changed what it is a median of.
    """
    chosen = positions[:limit]
    rows, _ = latent.represent(model, bagset, chosen, device,
                               domain="source", unit="instance")
    count = int(len(chosen))
    if rows.shape[0] % count:
        raise ValueError(
            f"{rows.shape[0]} instance rows over {count} bags does not divide; "
            "the grouping cannot be recovered")
    per_bag = rows.shape[0] // count
    embeddings = rows.reshape(count, per_bag, -1)

    reference, _ = latent.represent(model, bagset, chosen, device,
                                    domain="source", unit="bag")
    with torch.no_grad():
        Z, _ = model.bag_representations(embeddings.to(device),
                                        config.KERNEL_SIGMA)
    gap = float((Z.float().cpu() - reference.float()).abs().max())
    if gap > 1e-5:
        raise ValueError(
            f"the reshaped grouping disagrees with `latent.represent`'s own "
            f"bag reading by {gap}; the instance rows are not bag-major")

    with torch.no_grad():
        pooled = _off_diagonal(torch.cdist(rows, rows))
        between = _off_diagonal(torch.cdist(reference, reference))
    within_median = _within_bag_median(embeddings)
    # The weights at the bandwidth the grid is about to be centred on, not
    # at the declared constant: what travels is how far Eq. (16) is from
    # uniform where this run will actually put it.
    spread = _weight_spread(model, embeddings, within_median, device)
    uniform = 1.0 / per_bag
    return {
        "withinBag": within_median,
        "betweenBags": float(between.median()),
        "pooled": float(pooled.median()),
        "bagsRead": count,
        # What the cap did, said out loud. `bagsRead` alone cannot tell a role
        # of sixteen bags from a role of twenty-one read sixteen deep.
        "bagsOffered": int(len(positions)),
        "bagsTruncated": bool(len(positions) > count),
        "bagLimit": int(limit),
        "instancesPerBag": per_bag,
        "groupingCheckedAgainstBagUnit": gap,
        # How far Eq. (16)'s weights are from uniform at this bandwidth. A
        # spread of 1e-4 against a uniform 1/m says the attention is numerically
        # mean pooling, whatever the flags say, so the number travels.
        "weightSpread": spread,
        "uniformWeight": uniform,
        "weightSpreadRelativeToUniform": spread / uniform if uniform else 0.0,
    }


#: What multiplies the frozen bandwidth: the tutorial runs at this default.
#: Neutral and NOT `config.KERNEL_SIGMA`: that constant is 36.135, measured on
#: digits under resnet18, and the whole point of deriving the bandwidth is that
#: a number from another experiment's scale decides nothing here. The tutorial
#: runs at the statistic itself.
SIGMA_MULTIPLIER_NEUTRAL = 1.0

#: Where the frozen bandwidth's statistic comes from, as the record names it.
#: One string, so the record says which function produced the number instead of
#: leaving a reader to infer it from the value.
FROZEN_SIGMA_STATISTIC = "CREDA.toy_experiment.sigma_de_la_fuente"

#: The SET that statistic is taken over, and the half that was WRONG until this
#: stretch. CREDA takes its pairwise median over the set its own kernel sees --
#: points. Here Eq. (14)'s instance kernel evaluates pairs of instances WITHIN a
#: bag, and the first version of this function took the median over the whole
#: training role pooled, which mixes pairs from different bags and different
#: classes and so reads 17x to 77x the distance the kernel actually measures.
#: Measured on the pilot record, 4 of the 9 cells then fell outside
#: `SIGMA_BAND`, all four saturated, at the smallest multiplier the grid offers.
#:
#: The statistic is CREDA's and stays CREDA's; only the scope is corrected. Both
#: scopes travel in the record, with the ratio, so the correction is readable
#: rather than a number that silently moved.
FROZEN_SIGMA_SCOPE = ("within-bag instance pairs, pooled over the source "
                      "training role's bags")
FROZEN_SIGMA_POOLED_SCOPE = ("every instance against every instance over the "
                             "source training role, which is NOT the set "
                             "Eq. (14) evaluates")


@measuring
def frozen_source_sigma(model: wiring.Arm, bagset: bags.BagSet,
                        positions: torch.Tensor,
                        device: torch.device,
                        multiplier: float = SIGMA_MULTIPLIER_NEUTRAL) -> dict:
    """CREDA's own bandwidth, over the set CREDA's own kernel sees: the pairwise
    median of WITHIN-BAG instance pairs across the source's full train.

    **CREDA's statistic, with its scope corrected -- not a second statistic.**
    See `FROZEN_SIGMA_SCOPE` for what was wrong and what it cost. The formula is
    `CREDALoss._median_pairwise`'s, and `_within_bag_median` carries the one
    definition both this and `embedding_scales` read; over a single bag, where
    the two scopes are the same set, the two agree to the epsilon CREDA's own
    formula carries, and a test measures that per bag rather than asserting it.
    `sigma_de_la_fuente` is still CALLED here, on the pooled scope, because the
    reading this correction left has to stay visible beside the one that
    governs: `pooledSigma` is the number every pre-correction run ran at.

    **Source only, and that is what makes it worth deriving.** Nothing injected
    into the target -- an intruder, a relocation, contamination -- can move it,
    so damage entering through the class block stops being confused with damage
    entering through the unit of measurement. `readsTargetRows` is returned as a
    flat `False` rather than left implicit, because "the statistic is immune" is
    a claim and a reader is owed the field that carries it.

    **The FULL train, and the cap is refused rather than tolerated.**
    `embedding_scales` takes a `limit` of 16 bags, which at the declared cut is
    a third of the training role; this hands it the whole role and raises if the
    reading comes back truncated. A frozen bandwidth measured over a third of
    its own role, in the role's own order, is a bandwidth whose value depends on
    a cap nobody chose.

    **Measured AFTER phase 1 and frozen, which is where CREDA measures it.**
    CREDA's tutorial calls `sigma_de_la_fuente` on the source train the moment
    phase 1 ends and the source encoder is frozen; after phase 2 it could no
    longer be computed at all. This is that moment in this
    benchmark's unit: `positions` is the source training role and `theta_s` is
    frozen, so the number cannot move again. The scope changed in this stretch;
    the moment and the freeze did not.

    **`multiplier` is read, never applied.** What it buys is that the two
    readings below -- the kernel value and Eq. (16)'s weight spread -- are
    reported at the bandwidth phase 2 will ACTUALLY run at rather than at the
    bare statistic. The returned `sigma` is the statistic itself, and the
    caller is what multiplies it; a function that applied its own multiplier
    would be a second place the phase-2 bandwidth is decided.

    It reads through `latent.represent` and never through
    `Arm.instance_embeddings` directly, for the reason `embedding_scales` gives:
    the call-site roster is structural, and `represent` is the declared seam
    that states its domain and refuses without one.
    """
    scales = embedding_scales(model, bagset, positions, device,
                              limit=int(len(positions)))
    if scales["bagsTruncated"]:
        raise ValueError(
            f"the frozen bandwidth was measured over {scales['bagsRead']} of "
            f"{scales['bagsOffered']} bags; a cap must not reach this reading")
    rows, _ = latent.represent(model, bagset, positions, device,
                               domain="source", unit="instance")
    detached = rows.detach()
    embeddings = detached.reshape(scales["bagsRead"],
                                  scales["instancesPerBag"], -1)

    within = scales["withinBag"]
    pooled = sigma_de_la_fuente(detached)
    at_within = multiplier * within
    at_pooled = multiplier * pooled
    uniform = scales["uniformWeight"]

    def kernel(sigma: float) -> float:
        return math.exp(-(within ** 2) / (2.0 * sigma ** 2))

    def relative_spread(sigma: float) -> float:
        spread = _weight_spread(model, embeddings, sigma, device)
        return spread / uniform if uniform else 0.0

    within_relative = relative_spread(at_within)
    pooled_relative = relative_spread(at_pooled)
    return {
        "sigma": within,
        "statistic": FROZEN_SIGMA_STATISTIC,
        "scope": FROZEN_SIGMA_SCOPE,
        # The reading this correction left, computed on the same rows at the
        # same moment, so what the scope bought is a subtraction a reader can
        # perform rather than a claim they have to take.
        "pooledSigma": pooled,
        "pooledScope": FROZEN_SIGMA_POOLED_SCOPE,
        "pooledToWithinRatio": pooled / within if within else float("nan"),
        "multiplierRead": multiplier,
        # What the instance kernel reads at each of the two bandwidths. The
        # first is `exp(-1/(2m^2))` and so in band for every multiplier the grid
        # offers -- an identity, and said so where it is recorded, because a
        # band satisfied by construction proves the scope and never the
        # mechanism.
        "kernelWithinBag": kernel(at_within),
        "inBand": SIGMA_BAND[0] <= kernel(at_within) <= SIGMA_BAND[1],
        "pooledKernelWithinBag": kernel(at_pooled),
        "pooledInBand": SIGMA_BAND[0] <= kernel(at_pooled) <= SIGMA_BAND[1],
        "bandIsAnIdentityNote": (
            "the within-bag kernel at m times the within-bag median is "
            "exp(-1/(2 m^2)) whatever the measured distance is, so `inBand` "
            "holds by construction once the scope is right. It reports that "
            "the SCOPE is right and says nothing about the mechanism; "
            "`weightSpreadGainOverPooled` is the reading with a reachable red"),
        # The mechanism the band exists to protect, measured by taking the
        # credit away: the same weights on the same embeddings at the scope
        # this correction left.
        "weightSpreadRelativeToUniform": within_relative,
        "pooledWeightSpreadRelativeToUniform": pooled_relative,
        "weightSpreadGainOverPooled": (within_relative / pooled_relative
                                       if pooled_relative else float("nan")),
        "weightSpreadNote": (
            "zero gain means `attentionGamma` is zero, not that the bandwidth "
            "does not matter: Eq. (15)'s consensus term is the only path the "
            "bandwidth takes into Eq. (16), and config.ATTENTION_GAMMA is 0.0"),
        "attentionGamma": float(model.attention_gamma),
        "bags": int(len(positions)),
        "instances": int(detached.shape[0]),
        "instancesPerBag": int(scales["instancesPerBag"]),
        "dims": int(detached.shape[1]),
        "bagsTruncated": bool(scales["bagsTruncated"]),
        # Not implicit. The immunity is the reason this statistic is worth
        # computing, so the record carries the fact rather than the argument.
        "readsTargetRows": False,
        "measuredAt": "after phase 1, with theta_s frozen",
        # The full within-bag reading this was taken from, on the TRAINED
        # encoder over the whole role -- distinct from `sigma`'s own `scales`
        # block in the record, which is the pre-phase-1 bootstrap reading
        # capped at 16 bags.
        "withinBagReading": scales,
    }


# ------------------------------------------------------------- the eight ranges


#: The four declared ranges this path does NOT search, each with the thing it
#: parameterizes and the reason it decides nothing here.
#:
#: All four are Eq. (39)'s schedule: `ceiling` IS `lambda_glob` and
#: `ceilingLocal` IS `lambda_loc` -- `harness.ramp` multiplies its warmup curve
#: by the ceiling, so the ceiling is the coefficient the curve approaches --
#: and the two deltas are how fast each curve gets there. With both
#: coefficients fixed at exactly 1 and no curve, a value for any of the four
#: changes no number that this path computes.
#:
#: Naming them is the whole point. A key that decides nothing and is silently
#: accepted is indistinguishable from one that decides something, so
#: `refuse_retired_hyper` raises on each of these BY NAME.
RETIRED_RANGES = {
    "ceiling": {
        "range": config.CEILING_RANGE,
        "was": "lambda_glob of Eq. (39), as harness.ramp's ceiling",
        "why": ("the coefficient is fixed at 1 (`FIXED_COEFFICIENT`), so there "
                "is no ceiling for a curve to approach"),
    },
    "ceilingLocal": {
        "range": config.CEILING_LOCAL_RANGE,
        "was": "lambda_loc of Eq. (39), as harness.ramp's ceiling",
        "why": ("the same, for the local term's coefficient: also fixed at 1"),
    },
    "rampDelta": {
        "range": config.RAMP_DELTA_RANGE,
        "was": "how fast lambda_glob grows, as harness.ramp's delta",
        "why": ("nothing grows: no ramp is evaluated on this path, so the "
                "steepness of a curve that does not exist is not a dimension"),
    },
    "rampDeltaLocal": {
        "range": config.RAMP_DELTA_LOCAL_RANGE,
        "was": "how fast lambda_loc grows, as harness.ramp's delta",
        "why": ("the same, for the local term's curve"),
    },
}

#: Eq. (39)'s two coefficients on this path: `L = L_src + lambda_glob * L_glob
#: + lambda_loc * L_loc` with BOTH lambdas exactly 1, fixed, with no ramp.
#:
#: **A fixed 1 is not a ramp evaluated at its ceiling, which is why
#: `harness.ramp` is not called with this value instead of being dropped.**
#: `milcreda_ramp` is a WARMUP: `(1 - e^{-delta p}) / (1 + e^{-delta p})` times
#: the ceiling, which is ~0 at the first step and approaches the ceiling from
#: below without reaching it. Handing it `ceiling=1.0` would train under a
#: schedule from 0 to just under 1 -- a schedule nobody declared -- and would
#: report it as a fixed coefficient. So the two calls are gone and this
#: constant reaches `wiring.Arm.training_step` directly, which is what that
#: method wants: two plain floats, `ramp` and `ramp_local`.
FIXED_COEFFICIENT = 1.0


def refuse_retired_hyper(hyper: dict | None) -> None:
    """Raise if a hyper dict carries one of `RETIRED_RANGES`' four names.

    CREDA's own guard against `TECHO` and `DELTA`, built against this path's
    names: a hyper dict from an older search of this toyset carries `ceiling`,
    `ceilingLocal`, `rampDelta` and `rampDeltaLocal` -- the retired
    `MIL-CREDA/Notebooks/toy_search.json` did, with `ceiling` at 0.01568 -- and
    a caller who passes one in believes it is reproducing the run those values
    came from.

    **Refused, not ignored.** Ignoring is the failure mode: the key arrives,
    changes nothing, and nothing anywhere says so, so the run reads as the
    searched-ceiling run it is not. The name is in the message because the
    caller's next question is which key, and a message naming only "a retired
    key" sends them to read this source to find out.

    Called from both public entries on this path that take a hyper dict --
    `build_arm` and `material_bound` -- because each is reachable on its own
    from the notebook.
    """
    for name in RETIRED_RANGES:
        if name in (hyper or {}):
            retired = RETIRED_RANGES[name]
            raise ValueError(
                f"{name} is retired on this path: it was {retired['was']}, and "
                f"{retired['why']}. Both of Eq. (39)'s coefficients are fixed "
                f"at {FIXED_COEFFICIENT} with no ramp, so this value decides "
                f"nothing; received {(hyper or {})[name]!r}. A hyper dict "
                f"carrying it comes from a search made when it did decide "
                f"something, so the rest of that dict is from another "
                f"programme too. Drop the key rather than setting it to a "
                f"value that looks harmless.")


def declared_hyper(scale: Scale) -> dict:
    """Every searchable dimension at the value `config` declares for it.

    What the tutorial trains at: nothing is searched. `scale` is taken so a
    caller cannot assemble this without having one in hand.

    **None of `RETIRED_RANGES`' four names appears here**, and that is load
    bearing rather than tidy: a retired key emitted here would be handed
    straight into `refuse_retired_hyper` and nothing would run at all.
    """
    del scale
    return {
        # The PHASE-1 bootstrap only, and the last resort for it: the tutorial
        # replaces it with the measured within-bag median before phase 1 and
        # with the frozen statistic the moment phase 1 ends.
        "kernelSigma": config.KERNEL_SIGMA,
        "attentionGamma": config.ATTENTION_GAMMA,
        "attentionTemperature": config.ATTENTION_TEMPERATURE,
        "tauLocal": config.TAU_LOCAL,
        "lr1": None,
        "weightDecay1": 0.0,
        # `None` for the two rates and `0.0` for the two decays:
        # `scale.lr_1`/`scale.lr_2` take over on `None`.
        "dim": config.MLP_OUTPUT_DIM,
        "lr2": None,
        "weightDecay2": 0.0,
    }


# ------------------------------------------------------------------- the phases

def build_arm(arm_id: str, material: dict, scale: Scale, hyper: dict,
              device: torch.device, seed: int) -> wiring.Arm:
    """One arm of THIS toyset's own declaration, on this toyset's material.

    `wiring.Arm` is constructed directly, with a spec out of `TOY_ARMS_BY_ID`.
    `wiring.build` is deliberately not called: it resolves its `arm_id` through
    `config.ARMS_BY_ID` and so cannot build an arm the DIGIT benchmark does not
    declare -- which is the whole of what made this module borrow that
    declaration. `Arm.__init__` takes the spec dict as its first argument and
    reads every key it needs off it, so handing it this module's own is not a
    workaround: it is the seam the class already offers.

    Seeded immediately before construction so every arm of a cell draws
    identical shared parameters: `wiring.Arm.__init__` constructs theta_t LAST
    precisely so the encoder, the head and Eq. (15)'s parameters come out
    bit-identical between any two arms built with one seed: the tutorial's
    phase-1 probe and phase-1 model draw the same shared parameters.
    """
    refuse_retired_hyper(hyper)
    if arm_id not in TOY_ARMS_BY_ID:
        raise ValueError(
            f"unknown toyset arm {arm_id!r}; this module declares "
            f"{list(TOY_ARMS_BY_ID)}. The digit benchmark's own arms "
            f"({list(config.ARMS_BY_ID)}) are not this declaration and are not "
            f"reachable from here")
    torch.manual_seed(seed)
    spec = TOY_ARMS_BY_ID[arm_id]
    source, target = material["source"], material["target"]
    pools = (harness.pool_of(source, source.train_idx, device),
             harness.pool_of(target, target.train_idx, device))
    return wiring.Arm(spec, scale.classes, *pools, hyper=hyper).to(device)


def shared_phase_one_state(model: wiring.Arm) -> dict:
    """The parameters phase 1 fits: everything but theta_t.

    Snapshotted by name so the identical tensors can be loaded into the model
    phase 2 adapts: the tutorial keeps phase 1's best epoch with it and loads
    it into the phase-2 model built at the frozen bandwidth.
    """
    return {name: value.detach().clone()
            for name, value in model.state_dict().items()
            if not name.startswith("target_encoder.")}


def supervised_loss(model: wiring.Arm, x: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:
    """Eq. (21) over bag scores, which is phase 1's whole objective.

    `MIL_CREDA.objective.source_loss` is the equation and it is called, never
    rewritten. The three lines around it mirror `wiring.Arm.training_step`'s own
    bag branch; `training_step` itself is not usable here because it also builds
    the adaptation term, which would pass target bags through theta_t and let
    phase 1 train the encoder phase 2 is supposed to start from.
    """
    logits = model(x)
    return source_loss(
        F.softmax(logits, dim=1),
        F.one_hot(y, model.classes).to(logits.dtype),
        config.EPSILON,
    )


def copy_source_to_target(model: wiring.Arm) -> None:
    """The one copy: theta_t starts bit-identical to theta_s.

    `Arm.sync_target_from_source` carries the PARAMETERS -- it deliberately
    leaves the buffers alone, because carrying normalization statistics over at
    every step of a leash measured worse. This is not that leash: it is CREDA's
    `clone_for_target`, which happens exactly once and does carry the buffers,
    and it has to, or a BatchNorm theta_t would begin at `running_mean = 0` and
    `running_var = 1` and so would NOT be bit-identical to the encoder it is
    supposed to start as.
    """
    if model.target_encoder is None:
        return
    model.sync_target_from_source()
    source = dict(model.encoder.named_buffers())
    with torch.no_grad():
        for name, buffer in model.target_encoder.named_buffers():
            buffer.copy_(source[name])


def set_phase_two_modes(model: wiring.Arm) -> None:
    """Source frozen and in `eval`, theta_t the only module in `train`.

    `eval` and not merely `no_grad`: `no_grad` stops the gradient and does not
    stop BatchNorm from re-estimating its running statistics in the forward. A
    source encoder left in `train` would have its buffers move during phase 2,
    and the tutorial's `buffersFuenteMovidosFaseDos` check would report a
    freeze that did not hold -- which is exactly what CREDA's own buffer
    assertion exists to catch.
    """
    model.eval()
    if model.target_encoder is not None:
        model.target_encoder.train()


def effective_classes(model: wiring.Arm, device: torch.device) -> float:
    """CREDA's `cef`: the EFFECTIVE number of classes in the target's routing.

    `exp(H(p))`, with `p` the normalized histogram of the bag-level `argmax`
    over the target TRAIN role and `H` natural. It is `1` when every target bag
    lands in ONE class -- the collapse -- and `classes` when it spreads evenly,
    so it lives in `[1, classes]`.

    **The histogram is taken over `classes` FIXED bins and not over the classes
    present.** Counting only the present ones would make a collapse to one
    class and a perfect split between the two that remain give numbers that
    cannot be compared across epochs, which is exactly the comparison the
    criterion makes.

    **It reads NO target label, and that is what qualifies it as a criterion.**
    It is the histogram of the model's OWN predictions.

    The role is `model.target` -- the pool `build_arm` fills from
    `target.train_idx` -- which is the same material phase 2 trains on, already
    on the device. The per-step draw is a SAMPLE of it
    (`config.BAGS_PER_STEP` bags with replacement across steps), so a histogram
    taken from a step's draw would not be comparable between epochs; this walks
    the whole role.

    Read in `eval` and restored with `set_phase_two_modes`: `no_grad` stops
    the gradient and does NOT stop a BatchNorm from re-estimating its running
    statistics in the forward, so a reading taken in `train` would move the
    very buffers phase 2 promises to leave alone.
    """
    positions = torch.arange(model.target.members.shape[0],
                             device=model.target.members.device)
    model.eval()
    counts = torch.zeros(model.classes, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, int(positions.numel()), config.BAGS_PER_STEP):
            chunk = positions[start:start + config.BAGS_PER_STEP]
            routed = model.score_target(model.target.take(chunk)).argmax(dim=1)
            counts += torch.bincount(routed.cpu(), minlength=model.classes)
    set_phase_two_modes(model)
    total = int(counts.sum())
    assert total == int(positions.numel()), (total, int(positions.numel()),
                                             "a bag was routed twice or not at all")
    assert total > 0, "an empty routing has no effective number of classes"
    assert len(counts) == model.classes, (
        len(counts), model.classes, "the routing named a class out of range")
    share = counts.double() / float(total)
    nonzero = share[share > 0.0]
    return float(torch.exp(-(nonzero * nonzero.log()).sum()))


def phase_two_objective(record: dict) -> torch.Tensor:
    """Eq. (39) minus its first summand: what phase 2 actually minimizes.

    **A subtraction and not a re-composition, because the terms come back
    detached.** `wiring.training_step` returns `loss` as the differentiable
    three-term total and `contributionGlobal`/`contributionLocal` as plain
    floats, so the two-term objective cannot be rebuilt from the dict -- there
    is no tensor in it for the two summands. Subtracting the supervised FLOAT
    from the total tensor gives the right value and, because the subtrahend is
    a constant, a gradient that is bit-identical to the total's. That identity
    is the whole reason this is sound, and it is measured rather than claimed.

    **Why it is done here and not in `wiring.py`.** That file assembles Eq. (39)
    for every arm of the digit benchmark too, which the operator deferred, so
    changing what it returns would change an experiment outside this stretch.
    The objective is a property of THIS protocol's phase 2, and this is the
    module that owns it.

    `supervised` is read off the same record, so a caller cannot pair the loss
    of one step with the reading of another.
    """
    return record["loss"] - record["supervised"]


# ------------------------------------------------------- the latent positions


#: How many epochs after epoch 0 the adaptation animation captures at most:
#: `tutorial_visuals.epocas_a_capturar` spaces them evenly, and with no more
#: epochs than this it captures every one.
#:
#: **Twenty is the frame count CREDA's retired toy runner declared:** its
#: `CUADROS` was `range(0, EPOCAS_2 + 1, 5)` over 100 epochs, a stride of 5
#: epochs and twenty-one frames.
TRAJECTORY_FRAMES = 20


@torch.no_grad()
def bag_positions(model: wiring.Arm, bagset: bags.BagSet,
                  device: torch.device, domain: str) -> dict:
    """Eq. (19)'s `z` for EVERY bag of one domain, with its class and its index.

    Every bag of the domain and not a role: the figure is about where the two
    clouds sit, and a cloud drawn from one role would be the role's own
    sampling shown as the method's geometry. It also keeps the two domains
    equal in how many points they contribute, which is the one thing a reader
    reads as coverage without being told.

    `domain` states which encoder embeds `bagset` -- `"source"` for
    `model.instance_embeddings` (`theta_s`), `"target"` for
    `model.target_embeddings` (`theta_t`) -- and has no default anywhere in the
    call chain, the same refusal `latent.represent` already makes: a caller that
    cannot say which domain it holds gets an error rather than a plausible
    number from whichever encoder happened to be first.

    **`model.sigma` and never `config.KERNEL_SIGMA`, which is why this is not
    `latent.represent`.** That function does exactly this for the digit
    campaign and evaluates Eq. (19)'s weights at `config.KERNEL_SIGMA`. On this
    toyset that constant is out of band, because the bandwidth here is
    measured per family and handed to `build_arm` as `kernelSigma`, which
    lands on `Arm.sigma`. Reusing it would draw a space at a bandwidth this run
    never computed in. `latent.py` is not on this work's
    authorized surface, so the duplication is reported rather than removed.

    In `eval` and under `no_grad`: a BatchNorm encoder read in `train` mode
    would make a bag's position depend on which other bags shared its batch,
    and then the same bag would land somewhere else at a different chunk size.
    The mode is restored to the one phase 2 runs under before returning.
    """
    if domain not in wiring.SIDES:
        raise ValueError(f"unknown domain {domain!r}; known: {wiring.SIDES}")
    embed = model.instance_embeddings if domain == "source" else model.target_embeddings
    total = int(bagset.members.shape[0])

    model.eval()
    rows, classes, decided = [], [], []
    for start in range(0, total, config.BAGS_PER_STEP):
        chunk = torch.arange(start, min(start + config.BAGS_PER_STEP, total))
        instances = bagset.images[bagset.members[chunk]].to(device)
        embeddings = embed(instances)
        Z, _ = model.bag_representations(embeddings, model.sigma)
        rows.append(Z.float().cpu())
        classes.append(bagset.labels[chunk].cpu())
        # The head's own decision for each bag, off the SAME `embeddings` the
        # positions came from, so column two of the grid is the same cloud
        # recoloured and never a second reading taken somewhere else.
        #
        # `model._scores` and not `model.head(Z)`, which is the obvious line and
        # is wrong for an arm that does not predict on the bag: `_scores` is
        # where the unit branch lives (a bag goes through Eq. (19), an
        # instance-unit arm averages its instances' class distributions and
        # takes the log), and writing one of its two halves here would fork that
        # rule. Promoting it out of `wiring.Arm`'s private surface is the
        # alternative and `wiring.py` is not on this work's authorized surface,
        # so the private call is used and said out loud rather than forked.
        decided.append(model._scores(embeddings).argmax(dim=1).cpu())
    set_phase_two_modes(model)

    return {
        "positions": torch.cat(rows).numpy().astype(np.float32),
        "classes": torch.cat(classes).numpy().astype(np.int16),
        # What the head decided, per bag, at the state the model is in RIGHT
        # NOW. On the source it cannot move -- neither `theta_s` nor the head is
        # in the optimizer -- and on the target it moves with every step, which
        # is the whole of what column two animates.
        "predictions": torch.cat(decided).numpy().astype(np.int16),
        # The bag's own index, written rather than left to row order. The
        # figure pairs each target bag's BEFORE position with its AFTER one to
        # draw the displacement, and pairing by row order is exactly the
        # implicit contract that breaks in silence: the arrows would still be
        # drawn, between the wrong pairs, and nothing in the picture would say
        # so. With the index on both views the pairing is a join the drawing
        # code asserts. It is not a drawing channel.
        "bags": np.arange(total, dtype=np.int16),
    }


def dispositivo_del_toyset() -> "torch.device":
    """La CPU, porque en las DOS maquinas medidas le gana al acelerador.

    `harness.resolve_device()` prefiere `cuda > mps > cpu`, que es el orden de
    `CREDA.training_pipeline` y es el correcto para el banco de digitos: ahi un
    paso pasa un lote de imagenes por un resnet18 y hay matmuls grandes. Acá no
    hay ninguno. Un paso de la fase 2 de este toyset procesa TRES bolsas de 30
    instancias --- 90 puntos --- por un MLP chico, asi que el despacho a un acelerador cuesta mas que la aritmetica que
    lleva.

    Medido el 2026-10-21 sobre la MISMA celda (un brazo `TGL`, lunas,
    batchnorm, semilla 0, los hiperparametros del artefacto completo), con la
    fase 2 corriendo sus 100 epocas enteras en los cuatro casos:

    | dispositivo             | s/paso de fase 2 | el flujo entero |
    | M2 cpu, 8 nucleos       |          0.00967 |          6.16 h |
    | M2 mps, la GPU de Apple |          0.11182 |         66.34 h |
    | Kaggle cpu, 4 nucleos   |          0.05059 |         31.52 h |
    | Kaggle T4 (`sm_75`)     |          0.07521 |         46.48 h |

    El acelerador pierde en las dos maquinas: 11.6x en el M2 y 1.5x en Kaggle.
    El cruce del matmul cuadrado esta en n~512 en el M2 y en n~128 en Kaggle, y
    ninguna matriz que este toyset forma llega a n=100 --- los bloques de Gram
    son de 16x16, una bolsa de entrenamiento por clase.

    **Esto NO es una medicion de velocidad sola: el dispositivo mueve el
    numero.** La misma celda y la misma semilla dieron `targetAccuracy` 0.5556 /
    0.3333 / 0.6111 / 0.5000 en esos cuatro dispositivos, que es divergencia
    float32 entre backends compuesta a lo largo de 100 epocas por 16 pasos.
    Ninguno de los cuatro es el correcto y no se afirma que lo sea; lo que se
    sigue es que la campana entera corre en UNO, y esta funcion es donde ese uno
    queda declarado en vez de salir de un orden de preferencia heredado.

    `harness.resolve_device()` queda intacto: lo leen cinco sitios del banco de
    digitos y `CREDA.training_pipeline`, y moverlo cambiaria experimentos que
    esta decision no abarca.
    """
    return torch.device("cpu")


# ----------------------------------------------- lo que se lee del modelo ya entrenado

def codificador_de(model: wiring.Arm, domain: str):
    """El encoder que embebe `domain`, con la negativa que la Ec. (13) pide.

    Un solo lugar donde este modulo resuelve dominio -> encoder, para que la
    regla "ningun embedding sin un dominio declarado" (spec
    `per-domain-encoders`) sea una puerta y no tres copias de un condicional.
    `bag_positions` tiene su propia copia de esta linea y no se toca: ese
    docstring ya declara que `latent.py` no esta en la superficie autorizada de
    ese trabajo, y unificarlas es otra tarea.

    **Y hay que decir que el guardia solo ve llamadas directas.**
    `tests/test_domain_mixup_guard.py::
    test_no_module_besides_wiring_and_latent_calls_instance_embeddings` recorre
    el arbol de sintaxis buscando una LLAMADA cuyo atributo sea
    `instance_embeddings`, asi que un `embed = model.instance_embeddings`
    seguido de `embed(...)` --- que es lo que hace `bag_positions`, y ahora
    esto --- pasa el guardia por indireccion. El guardia atrapo mi primera
    version, que llamaba derecho; no habria atrapado esta. Queda reportado en
    vez de arreglado: endurecerlo es tocar una prueba del banco de digitos y
    su alcance no es el de este trabajo.
    """
    if domain not in wiring.SIDES:
        raise ValueError(f"dominio {domain!r} desconocido; conocidos: {wiring.SIDES}")
    return (model.instance_embeddings if domain == "source"
            else model.target_embeddings)


@measuring
def lectura_de_bolsas(model: wiring.Arm, bagset: bags.BagSet,
                      device: torch.device, domain: str) -> dict:
    """La Ec. (20) `g_j` y la Ec. (24) `w_j`, bolsa por bolsa, de un dominio.

    Es la lectura que la caminata de `w` mira moverse. Las dos cantidades salen del MISMO paso adelante y por
    la misma composicion que corre la perdida (`_milcreda_term`: `Z` por la Ec.
    (19), `softmax` de la cabeza, `confidences`), no por una segunda definicion
    puesta aca: `w_j` esta definido sobre `g_j` y leerlo de otro lado seria
    inventar un segundo `w`.

    Se niega sobre un brazo que no decide en la bolsa. La Ec. (24) esta escrita
    sobre la Ec. (20), que es la distribucion de la CABEZA sobre la
    representacion de bolsa; un brazo de unidad instancia llega a su decision
    promediando las distribuciones de sus instancias, y meter eso por este
    lector devolveria un numero que la Ec. (24) no nombra.

    Todas las bolsas del dominio y no un rol, por la razon de `bag_positions`:
    un rol es un muestreo y leerlo como la geometria del metodo es leer el
    muestreo. Las posiciones que devuelve son las del BAGSET.
    """
    if model.spec["unit"] != "bag":
        raise ValueError(
            f"el brazo {model.spec['id']!r} decide en la unidad "
            f"{model.spec['unit']!r}, y la Ec. (24) esta escrita sobre la Ec. "
            "(20), la distribucion de la cabeza sobre la representacion de "
            "bolsa. Un promedio de instancias por este lector seria otro `w`")
    embed = codificador_de(model, domain)
    total = int(bagset.members.shape[0])

    model.eval()
    puntajes, pesos, clases = [], [], []
    with torch.no_grad():
        for inicio in range(0, total, config.BAGS_PER_STEP):
            tramo = torch.arange(inicio, min(inicio + config.BAGS_PER_STEP, total))
            instancias = bagset.images[bagset.members[tramo]].to(device)
            H = embed(instancias)
            Z, _ = model.bag_representations(H, model.sigma)
            g = F.softmax(model.head(Z), dim=1)                   # Ec. (20)
            puntajes.append(g.float().cpu())
            pesos.append(confidences(g).float().cpu())            # Ec. (24)
            clases.append(bagset.labels[tramo].cpu())
    set_phase_two_modes(model)

    return {
        "scores": torch.cat(puntajes).numpy().astype(np.float64),
        "w": torch.cat(pesos).numpy().astype(np.float64),
        "classes": torch.cat(clases).numpy().astype(np.int16),
        "bags": np.arange(total, dtype=np.int64),
    }


# ------------------------------------------------- la caminata construida de w

#: Cuantos puntos tiene una CAMINATA CONSTRUIDA, y su unidad es el BARRIDO.
#:
#: Ni paso ni epoca, y el operador lo nombro asi el 2026-10-19: estos no son
#: updates del optimizador ni pasadas sobre los datos. Son las posiciones de un
#: camino que nosotros construimos, y el gif no entrena nada --- lee un modelo
#: que ya existe en posiciones que todavia no existian.
BARRIDOS = 24


#: Los TRES caminos de la caminata de `w`, en la forma exacta de CREDA: por cada uno,
#: las clases que EMPATAN en el arranque y la clase a cuyo centro camina.
#:
#: Es la declaracion de `CREDA.tutorial_visuals.WALK_STARTS`, que escribe
#: literalmente `(((0, 1), 0), ((1, 2), 2), ((0, 1, 2), 1))`. Aca es lo mismo UN NIVEL ARRIBA: alla el punto
#: que camina es una instancia del latente, aca es una BOLSA en el espacio de
#: representacion.
#:
#: **Cada camino arranca en una FRONTERA y termina en una clase, y eso es lo
#: que la figura demuestra.** El punto sale de donde la cabeza NO sabe --- `w`
#: a medio camino en un empate doble, casi cero en el triple --- y llega a
#: donde el posterior satura y `w` va a uno. Un arranque que no fuera frontera
#: no tendria de donde subir.
#:
#: **Antes eran tres arranques hacia los TRES destinos**, nueve caminos en
#: paneles separados, y uno de los arranques --- el centro de la clase 1 --- ni
#: siquiera era una frontera. Ruling del operador del 2026-10-22: un solo
#: barrido, cada punto a un centro distinto, como en CREDA.
#:
#: Medidos en CREDA sobre su propia figura: frontera doble `w = 0.369`,
#: frontera TRIPLE `w = 0.000003`.
CAMINOS_W = (((0, 1), 0), ((1, 2), 2), ((0, 1, 2), 1))


def _nombre_del_arranque(clases) -> str:
    """Como se llama la frontera donde empatan esas clases.

    Derivado de las clases y no escrito al lado: dos listas que digan lo mismo
    se separan, y la que se separara seria la que nadie volvio a mirar.
    """
    clases = tuple(int(c) for c in clases)
    return ("frontera-triple" if len(clases) > 2
            else "frontera-" + "-".join(str(c) for c in clases))


#: Los nombres de los tres arranques, DERIVADOS de `CAMINOS_W`.
ARRANQUES_W = tuple(_nombre_del_arranque(cl) for cl, _ in CAMINOS_W)


def _pasos_del_barrido(barridos: int, como: torch.Tensor) -> torch.Tensor:
    """Los `barridos` pasos de 0 a 1, EN EL DEVICE Y DTYPE de `como`.

    **El device explicito es el arreglo de un defecto medido, no un adorno.**
    `torch.linspace` sin `device=` nace en CPU, y un tensor 0-dim de CPU
    multiplicando uno de MPS devuelve el ORIGINAL en lugar de fallar: medido el
    2026-10-20 sobre `lunas`, `inicio + t * (final - inicio)` dio exactamente
    `inicio` para los 24 barridos con `||final - inicio|| = 23.85`. Las dos
    curvas salian constantes --- brecha 0.0025, `d_j^2` 0.8131, `pi` maximo
    0.0658 en los 24 --- y una caminata plana se lee como "el termino no
    responde", que es un resultado, en vez de como lo que era.

    """
    return torch.linspace(0.0, 1.0, barridos,
                          dtype=como.dtype, device=como.device)


def _centros_de_clase(Z: torch.Tensor, etiquetas: torch.Tensor,
                      clases: int) -> torch.Tensor:
    """El centro de cada clase en el espacio de REPRESENTACION de bolsa."""
    return torch.stack([Z[etiquetas == c].mean(dim=0) for c in range(clases)])


def _empate_en_el_segmento(head, Z_a: torch.Tensor, Z_b: torch.Tensor,
                           a: int, b: int, vueltas: int = 60) -> torch.Tensor:
    """El punto del segmento `Z_a -> Z_b` donde los logits de `a` y `b` empatan.

    Biseccion sobre `logit_a - logit_b`, que vale `>0` en `Z_a` y `<0` en `Z_b`
    si la cabeza decide cada centro como su propia clase. Si no cambia de
    signo, el empate NO esta en este segmento y esto se niega en vez de
    devolver un extremo: un extremo pasaria por frontera y `w` ahi no es el que
    la figura dice.
    """
    def diferencia(t: torch.Tensor) -> float:
        Z = Z_a + t * (Z_b - Z_a)
        logits = head(Z.unsqueeze(0))[0]
        return float(logits[a] - logits[b])

    lo, hi = torch.tensor(0.0), torch.tensor(1.0)
    if diferencia(lo) * diferencia(hi) >= 0:
        raise ValueError(
            f"la diferencia de logits entre las clases {a} y {b} no cambia de "
            f"signo sobre el segmento entre sus centros ({diferencia(lo):+.4f} "
            f"y {diferencia(hi):+.4f}), asi que no hay empate que encontrar; "
            "la cabeza no decide cada centro como su propia clase")
    for _ in range(vueltas):
        medio = (lo + hi) / 2
        if diferencia(lo) * diferencia(medio) <= 0:
            hi = medio
        else:
            lo = medio
    t = (lo + hi) / 2
    return Z_a + t * (Z_b - Z_a)


def _empate_a_tres(head, centros: torch.Tensor, pasos: int = 200
                   ) -> tuple[torch.Tensor, float]:
    """El punto del triangulo de los tres centros donde los TRES logits estan
    lo mas parejos que una rejilla baricentrica de `pasos` alcanza.

    Devuelve `(Z, residuo)` con el residuo = `max(logit) - min(logit)` ahi. El
    residuo VIAJA: un empate a tres que en realidad no empata daria una curva
    que arranca en un `w` que no es cero, y el lector tiene que poder verlo sin
    recalcular nada. Se busca en rejilla y no resolviendo el sistema porque la
    cabeza no tiene por que ser lineal, y una rejilla con su residuo al lado es
    honesta sobre lo que encontro.
    """
    if centros.shape[0] < 3:
        raise ValueError(f"hacen falta tres centros, hay {centros.shape[0]}")
    pesos = torch.tensor(
        [[i, j, pasos - i - j] for i in range(pasos + 1)
         for j in range(pasos + 1 - i)],
        dtype=centros.dtype, device=centros.device) / pasos
    # En LOTE y no una cabeza por punto: la rejilla tiene ~20k posiciones y
    # llamarla de a una tarda minutos para un numero que un solo forward da.
    Z = pesos @ centros[:3]
    logits = head(Z)
    margenes = logits.max(dim=1).values - logits.min(dim=1).values
    cual = int(torch.argmin(margenes))
    return Z[cual], float(margenes[cual])


def _bolsa_mas_indecisa(g: torch.Tensor, clases: tuple[int, ...],
                        prohibidas: frozenset[int] = frozenset()
                        ) -> tuple[int, bool]:
    """La bolsa del blanco mas indecisa entre las `clases` nombradas.

    **No lee ninguna etiqueta.** Todo sale de `g_j^t`, la Ec. (20), que es lo
    unico que la formulacion tiene en el blanco: la Ec. (22) deriva la
    pseudoetiqueta de ahi y la Ec. (24) deriva `w` de ahi. Agrupar por
    `target.labels` seria meter un oraculo que el metodo no tiene.

    Dos nombradas: la que minimiza `|g_a - g_b|` entre las bolsas cuyas DOS
    clases mas probables son exactamente `{a, b}`. El filtro por el par de
    arriba es lo que hace que sea esa frontera: sin el, una bolsa que apuesta
    todo a una tercera clase y reparte una miga entre `a` y `b` tiene la
    brecha mas chica de todas y ganaria la eleccion.

    Tres nombradas: la de `w` mas bajo, que es la mas cercana a la uniforme ---
    el argmin de la Ec. (24), la indecision a tres literal.

    `prohibidas` son las bolsas que otro camino ya tomo. Ver
    `_bolsas_que_arrancan`, que es quien las junta.

    Devuelve `(indice, relajado)`. `relajado` es `True` cuando ninguna bolsa
    LIBRE tenia ese par arriba y la eleccion cayo sobre todas: VIAJA porque una
    frontera que ninguna bolsa duda no es la frontera que la figura nombra, y
    el lector tiene que poder verlo sin recalcular nada.
    """
    clases = tuple(int(c) for c in clases)
    libres = torch.ones(g.shape[0], dtype=torch.bool, device=g.device)
    for i in prohibidas:
        libres[int(i)] = False
    if not bool(libres.any()):
        raise ValueError(
            "no quedan bolsas del blanco libres para este camino: hay menos "
            f"bolsas ({g.shape[0]}) que caminos a dibujar")
    if len(clases) > 2:
        w = confidences(g).masked_fill(~libres, float("inf"))
        return int(torch.argmin(w)), False
    a, b = clases
    brecha = (g[:, a] - g[:, b]).abs()
    par = torch.tensor(sorted(clases), device=g.device, dtype=torch.long)
    dos_mas = torch.topk(g, 2, dim=1).indices
    arriba = (torch.sort(dos_mas, dim=1).values == par).all(dim=1) & libres
    if bool(arriba.any()):
        candidatas = torch.nonzero(arriba, as_tuple=False).reshape(-1)
        return int(candidatas[torch.argmin(brecha[candidatas])]), False
    return int(torch.argmin(brecha.masked_fill(~libres, float("inf")))), True


def _bolsas_que_arrancan(g: torch.Tensor) -> dict[str, tuple[int, bool]]:
    """Una bolsa del blanco por camino de `CAMINOS_W`, y las TRES distintas.

    **Sin esto dos caminos eligen la misma bolsa y la figura dibuja el mismo
    punto dos veces.** No es hipotetico: medido el 2026-10-22 a escala
    completa, en `rollo` la frontera (1,2) y la TRIPLE eligieron las dos la
    bolsa 41, y en `toroide` la (0,1) y la TRIPLE eligieron las dos la 0. Pasa
    porque una bolsa casi uniforme es a la vez la mas pareja entre dos clases
    y la mas pareja entre las tres, asi que gana las dos elecciones.

    Se resuelve por el ORDEN DECLARADO de `CAMINOS_W` y no por un optimo
    global: cada camino toma su mejor bolsa todavia libre y el que llega
    ultimo paga el desempate. Un optimo global haria que la bolsa de un camino
    dependiera de los otros dos, y entonces agregar un cuarto camino moveria
    los tres que ya estaban --- que es lo contrario de lo que una figura por
    camino promete.
    """
    elegidas: dict[str, tuple[int, bool]] = {}
    tomadas: set[int] = set()
    for cl, _ in CAMINOS_W:
        j, relajado = _bolsa_mas_indecisa(g, cl, frozenset(tomadas))
        tomadas.add(j)
        elegidas[_nombre_del_arranque(cl)] = (j, relajado)
    return elegidas


@measuring
def caminata_de_w(model: wiring.Arm, source: bags.BagSet,
                  target: bags.BagSet, device: torch.device,
                  rol: str = "train_idx", rol_blanco: str = "train_idx",
                  barridos: int = BARRIDOS) -> dict:
    """La caminata de `w`: tres bolsas del BLANCO caminan hacia el centro de
    clase de la FUENTE que les toca, y `w` se lee exacto a lo largo.

    **No entrena nada.** `w = 1 - H2(g)/ln C` es una
    funcion del posterior y de nada mas, asi que la curva es una propiedad de
    la Ec. (24) y de la cabeza YA entrenada: exacta, sin semillas y sin
    varianza. Lo unico que hace es preguntarle a la cabeza por posiciones que
    todavia no existian.

    **El punto que camina es una bolsa del BLANCO, y eso es el arreglo de un
    defecto.** La Ec. (24) define `w_j^t` sobre `g_j^t` --- superindice `t`,
    blanco --- y la Ec. (25) la mete en `D_c^t`, que pondera bloques de bolsas
    OBJETIVO: el parrafo que sigue a la Ec. (24) dice literal que las bolsas
    fuente no se ponderan "porque su etiqueta es observada y no estimada", y
    `MIL_CREDA.conditional` lo tiene como invariante (`source_block_never_
    weighted`). Esta figura caminaba un punto construido en la geometria de la
    FUENTE, asi que dibujaba `w` sobre algo que la perdida no multiplica nunca.

    **Los centros de llegada SI son de la fuente.** Ahi la etiqueta es
    observada y la formulacion la tiene, asi que un centro de clase de la
    fuente es un objeto legitimo: la bolsa del blanco es empujada hacia el
    grupo de sujetos de la fuente.

    **La frontera dejo de ser el arranque y es el CRITERIO DE SELECCION.**
    Antes el arranque era el punto donde la cabeza empata --- exacto, pero
    FABRICADO: ninguna bolsa lo ocupa, y `w` ahi no es el `w` de nadie. Ahora
    se elige la bolsa del blanco mas indecisa entre las dos clases que nombra
    el camino (`_bolsa_mas_indecisa`, sin etiquetas), asi que `w` arranca donde
    arranca de verdad. Los puntos de frontera se siguen calculando y VIAJAN
    como referencia, con el residuo del empate triple al lado, para que la
    figura pueda mostrar cuanto le falto a la bolsa real para estar sobre la
    frontera.

    La unidad de la caminata es un **barrido**. Devuelve, por arranque, el
    camino en representacion (para dibujarlo proyectado) y `w` barrido por
    barrido, mas el `w` del arranque, que es el numero que CREDA reporta.
    """
    clases = int(model.head.out_features) if hasattr(model.head, "out_features") \
        else int(config.CLASSES)
    if clases < 3:
        raise ValueError(
            f"la frontera TRIPLE necesita tres clases y hay {clases}")
    s_idx = getattr(source, rol)
    t_idx = getattr(target, rol_blanco)
    codificar = codificador_de(model, "source")
    codificar_blanco = codificador_de(model, "target")

    model.eval()
    with torch.no_grad():
        # La FUENTE: la nube, el plano del dibujo y los centros de llegada. Su
        # etiqueta es observada, asi que agrupar por ella es legitimo.
        H = codificar(source.images[source.members[s_idx]].to(device))
        Z, _ = model.bag_representations(H, model.sigma)
        etiquetas = source.labels[s_idx].to(Z.device)
        centros = _centros_de_clase(Z, etiquetas, clases)

        # El BLANCO: de aca sale el punto que camina. `g_t` es lo unico que se
        # le pregunta --- nunca `target.labels`.
        H_t = codificar_blanco(target.images[target.members[t_idx]].to(device))
        Z_t, _ = model.bag_representations(H_t, model.sigma)
        g_t = F.softmax(model.head(Z_t), dim=1)

        # Las fronteras, de REFERENCIA. El triple sale de `_empate_a_tres`; un
        # empate de dos sale del segmento entre los dos centros, que es donde
        # esa frontera lo corta.
        triple, residuo = _empate_a_tres(model.head, centros)
        fronteras = {}
        for cl, _ in CAMINOS_W:
            nombre = _nombre_del_arranque(cl)
            fronteras[nombre] = (triple if len(cl) > 2 else
                                 _empate_en_el_segmento(
                                     model.head, centros[cl[0]], centros[cl[1]],
                                     cl[0], cl[1]))
        destinos = tuple(d for _, d in CAMINOS_W)

        reparto = _bolsas_que_arrancan(g_t)
        caminos: dict[str, dict[str, dict]] = {}
        elegidas: dict[str, int] = {}
        relajados: dict[str, bool] = {}
        for cl, suyo in CAMINOS_W:
            nombre = _nombre_del_arranque(cl)
            j, relajado = reparto[nombre]
            elegidas[nombre] = int(t_idx[j].item())
            relajados[nombre] = bool(relajado)
            inicio = Z_t[j]
            # `device=` explicito, y no es cosmetico. Un `linspace` sin device
            # nace en CPU, y un tensor 0-dim de CPU multiplicando uno de MPS
            # DEVUELVE EL ORIGINAL sin fallar: medido, `inicio + t*(final -
            # inicio)` dio `inicio` para los 24 barridos, asi que la caminata
            # salia perfectamente plana y se leia como "el termino no responde"
            # en vez de como un defecto.
            ts = _pasos_del_barrido(barridos, inicio)
            por_destino: dict[str, dict] = {}
            # UNO, el suyo: la figura de CREDA empareja cada frontera con una
            # clase. Se deja el mapa anidado porque es la forma que
            # `tutorial_visuals.gif_de_caminata` y el cuaderno recorren.
            for clase in (suyo,):
                destino = centros[clase]
                camino = (inicio.unsqueeze(0)
                          + ts[:, None] * (destino - inicio).unsqueeze(0))
                g = F.softmax(model.head(camino), dim=1)
                # La clave es el ENTERO como texto porque este diccionario
                # viaja por JSON, que no tiene claves enteras: un `int` sale
                # como `"0"` y volveria como `"0"`, asi que se escribe asi
                # desde el principio y el lector no tiene dos convenciones.
                por_destino[str(int(clase))] = {
                    "camino": camino.float().cpu().numpy().astype(np.float64),
                    "g": g.float().cpu().numpy().astype(np.float64),
                    "w": confidences(g).float().cpu().numpy().astype(np.float64),
                }
            caminos[nombre] = por_destino
    set_phase_two_modes(model)

    claves_de = {_nombre_del_arranque(cl): [str(int(d))] for cl, d in CAMINOS_W}
    return {
        "barridos": int(barridos),
        "clasesDestino": [int(c) for c in destinos],
        "arranques": list(ARRANQUES_W),
        "caminos": caminos,
        # El `w` con el que arranca cada camino y con el que termina, por
        # arranque Y por destino. El inicial no depende del destino --- es el
        # arranque --- y se reporta igual por par, para que una tabla no tenga
        # que saber cual de los dos indices lo determina.
        "wInicial": {n: {c: float(caminos[n][c]["w"][0]) for c in claves_de[n]}
                     for n in ARRANQUES_W},
        "wFinal": {n: {c: float(caminos[n][c]["w"][-1]) for c in claves_de[n]}
                   for n in ARRANQUES_W},
        # Que bolsa del blanco camina en cada arranque, en la numeracion del
        # BAGSET.
        "bolsaDelBlanco": elegidas,
        # `True` cuando ninguna bolsa tenia ese par como sus dos clases mas
        # probables: la eleccion cayo sobre todas y esa frontera no es una
        # frontera que el blanco dude. Ver `_bolsa_mas_indecisa`.
        "eleccionRelajada": relajados,
        # Las fronteras EXACTAS de la cabeza, de referencia: la bolsa real no
        # esta sobre ellas y la figura muestra cuanto le falta.
        "fronteras": {n: v.float().cpu().numpy().astype(np.float64)
                      for n, v in fronteras.items()},
        # **Si esto no es ~0, la frontera triple no empata.** Viaja para que no
        # haya que recalcularlo.
        "residuoDelEmpateTriple": float(residuo),
        # La base de PCA con la que el dibujo proyecta los caminos, calculada
        # sobre las bolsas de la FUENTE para que las tres caminatas y la nube
        # compartan plano.
        "nube": Z.float().cpu().numpy().astype(np.float64),
        "clasesDeLaNube": etiquetas.cpu().numpy().astype(np.int16),
        "centros": centros.float().cpu().numpy().astype(np.float64),
    }


# ------------------------- the latent space: the basis, the box, the readings

def _base_pca(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The 2-D basis this panel projects on, fitted on the SOURCE positions.

    **Fitted on the source alone, and the same basis then projects all three
    views.** The figure's question is where phase 2 moved the target relative
    to a frozen reference, and a basis refitted per view would move the frame
    along with the cloud -- the displacement would then be partly the
    projection's own, and nothing in the picture would separate the two. The
    source is the reference because it is the half that does not move.

    PCA and not the UMAP `latent._embed` uses for the digit grid, for a reason
    that is about this figure rather than a preference: a before/after pair has
    to live in ONE frame, a linear basis is the only projection that can be
    fitted once and applied to a cloud it was not fitted on, and a non-linear
    embedding refitted over the union would place the two moments by their
    mutual neighbourhoods -- which is the very thing being measured. It is also
    the choice CREDA's retired toy renderer made for exactly this figure (an
    SVD of the source, "clavada a la fuente").
    """
    mu = positions.mean(0)
    _, singular, Vt = np.linalg.svd(positions - mu, full_matrices=False)
    total = float((singular ** 2).sum())
    explained = float((singular[:2] ** 2).sum() / total) if total > 0 else float("nan")
    return mu, Vt[:2].T, explained


def _limites(puntos: np.ndarray) -> tuple[list[tuple[float, float]], float, bool]:
    """A CENTRED box, square while the cloud is nearly isotropic.

    The rule of CREDA's retired toy renderer (`limites_fila`), carried over with
    its reasons rather than re-derived: centred because both clouds move and a min/max box pins its
    extreme on one of them, square because an alignment is judged by DISTANCES
    and two axes at different scales show distances that are not there. Past
    `ASPECTO_MAX` -- CREDA's own 3.0, imported rather than copied -- the box
    goes per axis and the panel's title SAYS so, because beyond that ratio a
    square box leaves most of the panel empty and the cloud reads as a line.
    """
    centro = (puntos.max(0) + puntos.min(0)) / 2
    rango = puntos.max(0) - puntos.min(0)
    razon = float(max(rango) / max(float(min(rango)), 1e-9))
    if razon <= ASPECTO_MAX:
        medio = float(np.max(rango)) / 2 * 1.06
        return ([(float(centro[d] - medio), float(centro[d] + medio))
                 for d in range(2)], razon, True)
    return ([(float(centro[d] - rango[d] / 2 * 1.06),
              float(centro[d] + rango[d] / 2 * 1.06))
             for d in range(2)], razon, False)


# ---------------------------------------------------------- the GIF frames

#: The step of a frame and the pause on the LAST one, in MILLISECONDS, as
#: `tutorial_visuals.gif_en_memoria` writes them. CREDA's own two numbers and
#: its own units: the seconds-shaped `0.30` that reads right rounded to 0 and
#: wrote GIFs with no time assigned at all.
MS_FRAME, MS_PAUSE = 140, 2500


def _rasterizar(fig) -> np.ndarray:
    """La figura como arreglo RGB, cerrada despues de leerla."""
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    imagen = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return imagen
