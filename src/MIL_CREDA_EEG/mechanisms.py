"""The mechanism experiments, read from the stored models: K1, K2 and K3.

Experiments plan v60, Secs. 6.1-6.3, over proposal r44 Eqs. 14-19 (attention
and bag kernel), 28-31 (local alignment) and 34-38 (bounds of the normalized
terms). K1 and K2 train nothing: they read a model that `model_store` saved
(Secs. 6.3 and 7.3) or a history that training already recorded. K3 trains its
variants (`train_ablation_variant`, `train_aggregation_variant`). There is one
report seed (Sec. 4.6): every per-subject quantity is that seed's value, and a
subject that appears twice for one model and transfer is refused, never
averaged (`once_per_subject`).

**K1, the attention selects something meaningful (Sec. 6.1).** Measured per
arm on its core transfers, on the target test bags of each rotation, with the
saved phase-2 models; each arm is analysed separately, so no subject enters
one analysis twice. The task and device arms run the marked-window contrast,
the level arm only the behavioural one (`K1_CONTRASTS`): the high-density
bags of the levels carry no marked window. Per window: the attention weight
(`attention_weights`, read from the model's own attention: Eqs. 15-16 for
MIL-CREDA, its variant without consensus and the common wrapper, the native
aggregator for AdaptMIL); whether it is marked by sample loss
(Sec. 2.3, `integrity_mark`) or bad contact (Sec. 2.5, any channel's
`contact_bad`), which `marked_windows` reads from the bag's own marks; and, in
the level arm, whether it holds a behavioural event (`event_windows`, the
`event_window` mark that preprocessing writes from `k1_events`; refused as
`NotRecorded` while it does not exist). The marked-window contrast reads only
EMOTIV recordings (`is_emotiv`, from `bag_data.CONDITIONS`), so the device arm
measures only its Robots side, and only the subjects whose bag holds both
marked and clean windows; the others are counted per arm. Per subject: the
mean weight of the marked windows minus that of the clean ones in the task and
device arms and, in the level arm, the mean weight of the event windows minus
that of the rest. Tested with Wilcoxon over subjects (`k1_tests`). The
criterion is three tests across arms, Holm over the three: in the task arm,
marked vs clean with MIL-CREDA and MIL-CREDA vs the own attention without
consensus (Sec. 6.3, `K1_WITHOUT_CONSENSUS`) paired per subject; in the level
arm, event vs rest with MIL-CREDA. The device arm reports the two marked-window
tests as support, without criterion. Every other attention -- the published
and gated attentions of Sec. 6.3, the common attention of Sec. 7.2 and
AdaptMIL -- is reported with the same contrasts and without criterion. The
mean is excluded (`K1_EXCLUDED`).

**K2, local matching aligns compatible subjects (Sec. 6.2).** Over the probe
subjects of each rotation (`assignment.Assignment.probe_subjects`: the test
fold's subjects holding both conditions of the transfer), the target bag is the
subject's bag in the target condition through the target trunk, and its own
source bag is its bag in the source condition through the source trunk; the
condition the assignment did not read comes only through
`assignment.probe_bag`. Distances are the bag-kernel (RKHS) distances at the
frozen sigma (`kernel_distances`, Eqs. 17-19) for MIL-CREDA and its
global-term-only variant, and Euclidean distances between bag representations
for the comparators (`euclidean_distances`). Per subject the separation is the
mean distance to the other probe subjects' source bags minus the distance to
its own (`separation`), tested with Wilcoxon over the probe subjects on the two
declared transfers (`K2_DECLARED`): separation > 0 with MIL-CREDA, and
MIL-CREDA vs the global-term-only variant of the Sec. 6.3 ablation
(`K2_NO_LOCAL`) paired per subject, Holm over four.

**Every Wilcoxon row of K1 and K2 (Secs. 6.1, 6.2)** carries the
matched-pairs rank-biserial correlation as its effect size and, beside it,
the Hodges-Lehmann estimator of the per-subject difference (`hodges_lehmann`)
with its Wilcoxon-based 95% confidence interval in `ci_low` and `ci_high`:
the exact signed-rank interval on the sorted Walsh averages, from the k-th to
the (M + 1 - k)-th, with k the largest integer whose null probability
P(T+ <= k - 1) is at most 0.025 (`statistics.hodges_lehmann`, which states
the method and reports the achieved coverage). `k1_test_table` and
`k2_test_table` lay the rows out under the column headers of Secs. 6.1 and
6.2, verbatim (`K1_TABLE_COLUMNS`, `K2_TABLE_COLUMNS`).

**K3, aggregation and ablation (Sec. 6.3).** The implementation control of
the normalized terms reads the per-epoch minimum and maximum of the global
and local terms that `two_phase.adapt_target_phase` records on every step
(`bound_control`). The two tables and their contrasts (`variant_table`,
`k3_contrasts`) read the pooled test predictions of each variant. No variant has
a search of its own: every one runs with full MIL-CREDA's selected
hyperparameters of the same transfer. The ablation variants (`ABLATIONS`,
`two_phase.TERM_SELECTIONS`) share phase 1 with full MIL-CREDA: they start from
its stored phase-1 checkpoint at its sigma and run phase 2 with one term
dropped, built exactly as `search.train_mil_creda_phase_two` builds full
MIL-CREDA's phase 2 ("both terms" reproduces it bit for bit). The aggregation
variants (`aggregation_variants`) change phase 1, so each trains both phases:
phase 1 as `search.train_phase_one` trains the MIL-CREDA group, phase 2 from
that checkpoint as full MIL-CREDA's. The own attention without consensus takes
full MIL-CREDA's selection with gamma overridden to 0; the published poolings
record `attentionGamma` and `attentionTemperature` as not applicable (NaN in the
hyperparameters, listed in `not_applicable`), never reused. Each variant's
history is tagged `MIL-CREDA [<variant>]:phase<n>`, so the bound control of full
MIL-CREDA never reads it.

**Implementation choice the plan does not fix.** "The common attention of the
non-MIL comparators" (Sec. 7.2) is frozen from the common phase-1 group, but its
weights on a target bag go through each comparator's own phase-2 target trunk,
so they differ by comparator and the plan names none. K1 reports one reference
per non-MIL comparator whose phase-2 model is stored, named
`common attention [<comparator>]` (`common_attention_name`), all without
criterion; it never picks one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from MIL_CREDA.bag_kernel import bag_kernel_matrix
from MIL_CREDA.confidence import pseudolabel
from MIL_CREDA.local_term import class_correspondence
from MIL_CREDA_Benchmark.poc_math import entropy_ratio
from MIL_CREDA_EEG import aggregation_variants
from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, model_store, search
from MIL_CREDA_EEG import statistics as st
from MIL_CREDA_EEG import two_phase
from MIL_CREDA_EEG.two_phase import EVAL_CHUNK

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md",
    "sections": ["4.6", "6.1", "6.2", "6.3", "7.2", "7.3"],
    "experiments": ["K1-atencion", "K2-emparejamiento", "K3-ablacion"],
    "equations": ["14", "15", "16", "17", "18", "19", "33", "38"],
    "invariants": [
        "the_event_contrast_reads_only_the_slackline_levels_of_the_level_arm",
        "a_marked_window_is_an_integrity_or_contact_mark",
        "an_unrecorded_event_mark_is_refused_not_guessed",
        "the_attention_weights_are_the_models_own_and_sum_to_one",
        "the_weight_contrast_is_marked_minus_clean_and_skips_subjects_without_marks",
        "the_marked_window_contrast_reads_only_emotiv_recordings",
        "k1_reads_one_report_seed_and_holms_three_criterion_tests_across_task_and_level",
        "k2_reads_the_other_condition_only_through_the_probe",
        "the_kernel_distance_is_the_rkhs_distance_at_the_frozen_sigma",
        "the_separation_is_others_minus_own",
        "k2_tests_the_two_declared_transfers_with_holm_over_four",
        "every_k1_and_k2_wilcoxon_row_carries_hodges_lehmann_and_its_interval",
        "the_k1_and_k2_test_tables_carry_the_plans_headers",
        "the_bound_control_reads_every_recorded_step",
        "a_stored_phase_two_model_is_rebuilt_with_every_parameter",
        "a_stored_k3_variant_model_is_rebuilt_for_its_attention",
        "the_variant_trainers_refuse_an_unknown_variant",
        "the_both_terms_ablation_is_full_mil_creda",
        "the_ablation_starts_from_full_mil_credas_phase_one_and_its_hyperparameters",
        "the_parameters_a_variant_lacks_are_not_applicable",
        "an_aggregation_variant_trains_both_phases",
        "an_aggregation_variant_resumes_from_its_stored_phase_one",
    ],
}

#: Sec. 6.1: the conditions with timed behavioural events, the Slackline levels' TAGS.
EVENT_CONDITIONS = ("L1", "L6", "L11")
#: The per-window mark of a behavioural event (`k1_events`); preprocessing writes it.
EVENT_MARK = "event_window"
#: Sec. 2.3 and 2.5: the marks K1 reads.
INTEGRITY_MARK = "integrity_mark"
CONTACT_MARK = "contact_bad"
#: Sec. 6.1: the contrasts each arm runs, in the order of its test table.
K1_CONTRASTS = {"task": ("contact",), "level": ("event",), "device": ("contact",)}
#: Sec. 6.1: the device of the only recordings with marked windows (`bag_data.CONDITIONS`).
EMOTIV_DEVICE = bag_data.CONDITIONS["Robots"][1]
#: Sec. 6.1: the three criterion tests, (arm, contrast), Holm over the three.
K1_CRITERION_TESTS = (("task", "marked"), ("task", "paired"), ("level", "event"))
K1_CRITERION = "task marked-window tests and level event test, Holm over three (Sec. 6.1, v56)"
K1_OWN = two_phase.MODEL
#: Sec. 6.1: the rival that carries the criterion, the K3 variant without consensus.
K1_WITHOUT_CONSENSUS = model_store.variant_name(aggregation_variants.WITHOUT_CONSENSUS)
#: Sec. 6.1: the K3 attentions reported without criterion.
K1_REFERENCE_VARIANTS = tuple(model_store.variant_name(v) for v in
                              ("published attention", "gated attention"))
K1_EXCLUDED = {"mean": "its weights are uniform, so its contrast is zero by construction "
                       "(Sec. 6.1)"}
#: Sec. 6.2: chosen before any result, by structure.
K2_DECLARED = {"smaller shift": assignment_mod.Transfer("level", "L1", "L6"),
               "larger shift": assignment_mod.Transfer("device", "L6", "Robots")}
#: Sec. 6.2: MIL-CREDA without its local term is the Sec. 6.3 global-term-only variant.
K2_NO_LOCAL = model_store.variant_name("global term only")
#: Sec. 6.1's test table, headers verbatim.
K1_TABLE_COLUMNS = ("Contraste", "Sujetos", "Estadístico", "Valor de p",
                    "Tamaño del efecto, correlación biserial de rangos",
                    "Estimador de Hodges-Lehmann", "Intervalo de confianza al 95 % de Wilcoxon")
#: Sec. 6.2's test table, headers and rows verbatim, in the plan's order.
K2_TABLE_COLUMNS = ("Prueba de Wilcoxon", "Estadístico", "Valor de p crudo",
                    "Valor de p ajustado por Holm",
                    "Tamaño del efecto, correlación biserial de rangos",
                    "Estimador de Hodges-Lehmann", "Intervalo de confianza al 95 % de Wilcoxon",
                    "Sujetos de la sonda")
K2_TABLE_ROWS = (
    ("smaller shift", "separation", "Corrimiento menor: separación de MIL-CREDA mayor que cero"),
    ("larger shift", "separation", "Corrimiento mayor: separación de MIL-CREDA mayor que cero"),
    ("smaller shift", "no local", "Corrimiento menor: MIL-CREDA contra sin término local"),
    ("larger shift", "no local", "Corrimiento mayor: MIL-CREDA contra sin término local"))
#: Sec. 6.2's reading gate (v56): the verdict a transfer carries when the local
#: term had no room, never "not supported".
K2_INCONCLUSIVE = "no concluyente (el mecanismo no tuvo margen)"
#: Sec. 6.2's gate thresholds (v56): read where entropy < 0.9 OR room >= 1.0.
K2_ENTROPY_MAX = 0.9
K2_ROOM_MIN = 1.0
#: A diagnostic whose stored model is missing: recorded with its reason, never gated.
K2_NOT_COMPUTED = "no calculado"
#: Sec. 6.2's diagnostics table, headers verbatim, in the plan's order.
K2_DIAGNOSTICS_COLUMNS = (
    "Transferencia", "Sujetos de la sonda", "Mediana de la razón de entropía",
    "Fracción con la propia bolsa fuente como la más cercana", "Mediana de la razón de margen",
    "Empuje con término local", "Empuje sin término local", "Condición de lectura")
#: Sec. 6.3.
AGGREGATIONS = ("own", *aggregation_variants.VARIANTS)
ABLATIONS = tuple(two_phase.TERM_SELECTIONS)
TERMS = ("global", "local")


def common_attention_name(comparator: str) -> str:
    """K1's name for the common attention read through one non-MIL comparator's model."""
    return f"common attention [{comparator}]"


class MechanismError(RuntimeError):
    """A mechanism analysis was asked for something Sec. 6 does not define."""


class NotRecorded(MechanismError):
    """A quantity the analysis needs is not recorded by preprocessing or training."""


# ------------------------------------------------------------------ stored models


def load_phase_two(path: str | Path, *, partition_path: str | Path | None = None,
                   allow_code_mismatch: bool = False):
    """A saved phase-2 model rebuilt from its phase-1 group's architecture.

    Every parameter and buffer of the architecture must be in the stored state
    (the store saves the whole model); keys the architecture does not hold are the
    strategy's auxiliary training structures and are returned, not loaded. A CLUDA
    model is rebuilt whole by its own unit (`cluda.load_phase_two`: the projector
    restored and the comparator's source digest checked), so nothing is returned unloaded.
    """
    state, meta = model_store.load(path, partition_path=partition_path,
                                   allow_code_mismatch=allow_code_mismatch)
    if meta["phase"] != 2:
        raise MechanismError(f"{path} is not a phase-2 model")
    name = meta["name"]
    if name == "CLUDA":
        from MIL_CREDA_EEG.comparators.cluda import training as cluda
        model, meta = cluda.load_phase_two(path, partition_path=partition_path,
                                           allow_code_mismatch=allow_code_mismatch)
        return model, meta, []
    group = name if name in model_store.VARIANTS else search.PHASE_ONE_GROUP_OF[name]
    model = model_store.build_stored(group, meta["channels"]["source"],
                                     meta["channels"]["target"], meta["hyper"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise MechanismError(f"{path}: the stored state lacks {missing[:5]} of the {group} "
                             f"architecture")
    if meta.get("sigma") is not None and hasattr(model, "fix_sigma"):
        model.fix_sigma(meta["sigma"])
    model.eval()
    return model, meta, list(unexpected)


# ------------------------------------------------------------------ K1 (Sec. 6.1)


def marked_windows(bag: bag_data.Bag) -> np.ndarray:
    """Per window: an integrity mark (Sec. 2.3) or bad contact on any channel (Sec. 2.5)."""
    marks = bag.marks()
    if INTEGRITY_MARK not in marks:
        raise NotRecorded(f"{bag.subject}/{bag.condition}: no {INTEGRITY_MARK} in its marks")
    marked = np.asarray(marks[INTEGRITY_MARK], dtype=bool).reshape(-1)
    if CONTACT_MARK in marks:
        contact = np.asarray(marks[CONTACT_MARK], dtype=bool)
        marked = marked | (contact.reshape(len(marked), -1).any(axis=1) if contact.size
                           else np.zeros_like(marked))
    if len(marked) != bag.n_windows:
        raise MechanismError(f"{bag.subject}/{bag.condition}: {len(marked)} marks for "
                             f"{bag.n_windows} windows")
    return marked


def is_emotiv(condition: str) -> bool:
    """Sec. 6.1: the condition is recorded with EMOTIV, read from `bag_data.CONDITIONS`."""
    if condition not in bag_data.CONDITIONS:
        raise MechanismError(f"unknown condition {condition!r}; known: "
                             f"{list(bag_data.CONDITIONS)}")
    return bag_data.CONDITIONS[condition][1] == EMOTIV_DEVICE


def event_windows(bag: bag_data.Bag) -> np.ndarray:
    """Per window: it contains the instant of a behavioural event (Sec. 6.1)."""
    if bag.condition not in EVENT_CONDITIONS:
        raise MechanismError(f"{bag.condition} has no timed event stream (Sec. 6.1: only the "
                             f"Slackline levels {list(EVENT_CONDITIONS)})")
    marks = bag.marks()
    if EVENT_MARK not in marks:
        raise NotRecorded(f"{bag.subject}/{bag.condition}: preprocessing records no "
                          f"{EVENT_MARK!r}; the event contrast of Sec. 6.1 needs it")
    events = np.asarray(marks[EVENT_MARK], dtype=bool).reshape(-1)
    if len(events) != bag.n_windows:
        raise MechanismError(f"{bag.subject}/{bag.condition}: {len(events)} event marks for "
                             f"{bag.n_windows} windows")
    return events


@torch.no_grad()
def attention_weights(model, windows: torch.Tensor, domain: str = "target") -> np.ndarray:
    """The model's own in-bag weights for one bag, in eval mode."""
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        H = model.embed([windows], domain, chunk=EVAL_CHUNK)
        if hasattr(model, "bag_pairs"):
            weights = model.bag_pairs(H)[0][1]
        elif hasattr(model, "aggregator"):
            weights = model.aggregator.weights(model.aggregator.project(H[0]))
        else:
            raise MechanismError(f"{type(model).__name__} exposes no attention")
    finally:
        for m, mode in modes.items():
            m.train(mode)
    return weights.detach().to(torch.float64).cpu().numpy()


def weight_contrast(weights: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean weight where `mask` minus mean weight elsewhere; None if either side is empty."""
    mask = np.asarray(mask, dtype=bool)
    if mask.all() or not mask.any():
        return None
    return float(weights[mask].mean() - weights[~mask].mean())


def k1_subject_rows(model, bags: Sequence[bag_data.Bag], *, model_name: str,
                    transfer: assignment_mod.Transfer, rotation: int, seed: int,
                    domain: str = "target") -> list[dict]:
    """Per target test subject: the arm's K1 quantities for one model and rotation."""
    if transfer.arm not in K1_CONTRASTS:
        raise MechanismError(f"K1 is measured per arm {list(K1_CONTRASTS)}; got {transfer.arm}")
    with_events = "event" in K1_CONTRASTS[transfer.arm]
    with_marked = "contact" in K1_CONTRASTS[transfer.arm]
    rows = []
    for bag in bags:
        weights = attention_weights(model, torch.from_numpy(np.ascontiguousarray(bag.windows()))
                                    .float(), domain)
        emotiv = is_emotiv(bag.condition)
        marked = marked_windows(bag) if with_marked and emotiv else None
        event, event_note = None, None
        if with_events:
            try:
                event = weight_contrast(weights, event_windows(bag))
            except NotRecorded as error:
                event_note = str(error)
        rows.append({"model": model_name, "arm": transfer.arm, "transfer": transfer.name,
                     "pair": "|".join(transfer.pair), "rotation": int(rotation),
                     "seed": int(seed), "subject": bag.subject,
                     "emotiv": emotiv,
                     "marked_minus_clean": None if marked is None
                     else weight_contrast(weights, marked),
                     "n_marked": None if marked is None else int(marked.sum()),
                     "n_windows": int(bag.n_windows),
                     "event_minus_rest": event, "event_not_recorded": event_note})
    return rows


def once_per_subject(rows: Iterable[Mapping]) -> list[Mapping]:
    """Sec. 4.6: one report seed, so each (model, transfer, subject) appears once."""
    rows = list(rows)
    twice = [k for k, n in Counter((r.get("model"), r["transfer"], r["subject"])
                                   for r in rows).items() if n > 1]
    if twice:
        raise MechanismError(f"a subject enters each model's analysis once (one report seed, "
                             f"one test rotation); repeated: {twice[:5]}")
    return rows


def _wilcoxon_row(values: Sequence[float | None], *, comparison: str, name: str,
                  experiment: str, extra: Mapping | None = None) -> dict:
    finite = [v for v in values if v is not None]
    if not finite:
        return st.significance_row(comparison=comparison, test=st.TEST_WILCOXON,
                                   statistic=None, p_value=None, effect_size=None,
                                   effect_size_name=name, ci_low=None, ci_high=None, n=0,
                                   hypothesis=experiment, experiment=experiment,
                                   **(extra or {}))
    w = st.wilcoxon_signed_rank(finite)
    return st.significance_row(comparison=comparison, test=st.TEST_WILCOXON,
                               statistic=w["statistic"], p_value=w["p_value"],
                               effect_size=w["rank_biserial"], effect_size_name=name,
                               ci_low=w["ci_low"], ci_high=w["ci_high"], n=w["n"],
                               hypothesis=experiment, experiment=experiment,
                               median=w["median"], n_nonzero=w["n_nonzero"],
                               hodges_lehmann=w["hodges_lehmann"],
                               ci_name="Hodges-Lehmann, exact Wilcoxon signed-rank 95% interval",
                               ci_coverage=w["ci_coverage"], ci_note=w["ci_note"],
                               **(extra or {}))


def _not_recorded(members: Sequence[Mapping]) -> str | None:
    return next((r["event_not_recorded"] for r in members if r.get("event_not_recorded")),
                None)


def _k1_single(members: Sequence[Mapping], arm: str, model: str,
               criterion: str | None) -> list[dict]:
    """The arm's one-model contrasts: marked vs clean (task, device) or event vs rest (level).

    `criterion` None marks the own model: each row takes the criterion of its
    (arm, contrast) slot in `K1_CRITERION_TESTS`, or support otherwise.
    """
    experiment = "K1-atencion"
    prefix = f"{experiment}:{arm}"
    rows = []
    if "contact" in K1_CONTRASTS[arm]:
        marked = [r["marked_minus_clean"] for r in members]
        rows.append(_wilcoxon_row(
            marked, comparison=f"{prefix}:marked vs clean, {model}",
            name="matched-pairs rank-biserial (marked - clean weight)", experiment=experiment,
            extra={"criterion": criterion or _k1_criterion(arm, "marked"),
                   "subjects": sum(v is not None for v in marked),
                   "subjects_with_both_window_kinds": sum(v is not None for v in marked),
                   "subjects_not_emotiv": sum(not r.get("emotiv", True) for r in members),
                   "subjects_without_marked_windows": sum(v is None for v in marked)}))
    if "event" in K1_CONTRASTS[arm]:
        events = [r["event_minus_rest"] for r in members]
        rows.append(_wilcoxon_row(
            events, comparison=f"{prefix}:event vs rest, {model}",
            name="matched-pairs rank-biserial (event - rest weight)", experiment=experiment,
            extra={"criterion": criterion or _k1_criterion(arm, "event"),
                   "subjects": sum(v is not None for v in events),
                   "not_computed": _not_recorded(members)}))
    return rows


def _k1_criterion(arm: str, contrast: str) -> str:
    return K1_CRITERION if (arm, contrast) in K1_CRITERION_TESTS else "none (support, Sec. 6.1)"


def k1_tests(rows_by_model: Mapping[str, Sequence[Mapping]], *,
             own: str = K1_OWN, without: str = K1_WITHOUT_CONSENSUS) -> list[dict]:
    """Sec. 6.1's tables, per arm, from per-subject rows at the report seed.

    The criterion rows of every arm present form one family, Holm over them
    (three when the task and level arms are both measured).
    """
    experiment = "K1-atencion"
    by_model = {m: once_per_subject(rs) for m, rs in rows_by_model.items()}
    out = []
    for arm in [a for a in K1_CONTRASTS if any(r["arm"] == a for r in by_model.get(own, []))]:
        members = [r for r in by_model[own] if r["arm"] == arm]
        family = _k1_single(members, arm, own, None)
        if "contact" in K1_CONTRASTS[arm]:
            rivals = [r for r in by_model.get(without, []) if r["arm"] == arm]
            paired, note = [], None
            if not rivals:
                note = (f"the {without} model is not stored for the {arm} arm (Sec. 6.3 "
                        f"aggregation table: run the k3 step)")
            else:
                rival_by = {(r["transfer"], r["subject"]): r for r in rivals}
                for r in members:
                    c = rival_by.get((r["transfer"], r["subject"]))
                    if c is None:
                        raise MechanismError(f"{r['subject']} of {r['transfer']} has no "
                                             f"{without} row")
                    if r["marked_minus_clean"] is not None and \
                            c["marked_minus_clean"] is not None:
                        paired.append(r["marked_minus_clean"] - c["marked_minus_clean"])
            family.insert(1, _wilcoxon_row(
                paired, comparison=f"{experiment}:{arm}:marked vs clean, {own} vs {without}",
                name=f"matched-pairs rank-biserial ({own} - {without})", experiment=experiment,
                extra={"criterion": _k1_criterion(arm, "paired"), "subjects": len(paired),
                       "subjects_with_both_window_kinds": len(paired), "not_computed": note}))
        out += family
        for model in sorted(m for m in by_model if m not in (own, without)):
            mine = [r for r in by_model[model] if r["arm"] == arm]
            if mine:
                out += _k1_single(mine, arm, model, "none (reference, Sec. 6.1)")
    st.apply_holm([r for r in out if r["criterion"] == K1_CRITERION],
                  f"{experiment} criterion tests")
    return out


# ------------------------------------------------------------------ K2 (Sec. 6.2)


def probe_pairs(a: assignment_mod.Assignment, bags: bag_data.BagSet,
                rotation: int) -> list[tuple[str, bag_data.Bag, bag_data.Bag]]:
    """(subject, target-condition bag, source-condition bag) for each probe subject."""
    out = []
    for subject in sorted(a.probe_subjects(rotation)):
        assigned = bags.get(subject, a.conditions[subject])
        other = assignment_mod.probe_bag(a, bags, subject, rotation)
        if assigned.condition == a.transfer.target:
            out.append((subject, assigned, other))
        else:
            out.append((subject, other, assigned))
    return out


def _tensor(bag: bag_data.Bag) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(bag.windows())).float()


@torch.no_grad()
def kernel_geometry(model, target: Sequence[torch.Tensor],
                    source: Sequence[torch.Tensor]) -> dict:
    """At the model's sigma: K_st[j, k] = kappa^B(target j, source k), the RKHS distances
    D_ts[j, k] and D_ss[k, l] between source bags, and the target pseudo-labels (Eq. 22)."""
    model.eval()
    sigma = model.sigma
    pt = model.bag_pairs(model.embed(list(target), "target", chunk=EVAL_CHUNK))
    ps = model.bag_pairs(model.embed(list(source), "source", chunk=EVAL_CHUNK))
    K_ts = bag_kernel_matrix(pt, ps, sigma)
    K_ss = bag_kernel_matrix(ps, ps, sigma)
    k_tt = torch.stack([bag_kernel_matrix([p], [p], sigma)[0, 0] for p in pt])
    k_ss = torch.stack([bag_kernel_matrix([p], [p], sigma)[0, 0] for p in ps])
    d2 = k_tt[:, None] + k_ss[None, :] - 2 * K_ts
    d2_ss = k_ss[:, None] + k_ss[None, :] - 2 * K_ss
    G_t = model.classify(model.represent(pt))
    return {"K_st": K_ts.cpu().numpy(), "D_ts": torch.sqrt(d2.clamp_min(0)).cpu().numpy(),
            "D_ss": torch.sqrt(d2_ss.clamp_min(0)).cpu().numpy(),
            "pseudo": np.array([pseudolabel(g) for g in G_t])}


@torch.no_grad()
def kernel_distances(model, target: Sequence[torch.Tensor],
                     source: Sequence[torch.Tensor]) -> np.ndarray:
    """D[j, k]: RKHS distance between target bag j and source bag k at the model's sigma."""
    model.eval()
    sigma = model.sigma
    pt = model.bag_pairs(model.embed(list(target), "target", chunk=EVAL_CHUNK))
    ps = model.bag_pairs(model.embed(list(source), "source", chunk=EVAL_CHUNK))
    K_ts = bag_kernel_matrix(pt, ps, sigma)
    k_tt = torch.stack([bag_kernel_matrix([p], [p], sigma)[0, 0] for p in pt])
    k_ss = torch.stack([bag_kernel_matrix([p], [p], sigma)[0, 0] for p in ps])
    d2 = k_tt[:, None] + k_ss[None, :] - 2 * K_ts
    return torch.sqrt(d2.clamp_min(0)).cpu().numpy()


@torch.no_grad()
def bag_representations(model, bags: Sequence[torch.Tensor], domain: str) -> torch.Tensor:
    model.eval()
    if hasattr(model, "representations"):
        return model.representations(list(bags), domain, chunk=EVAL_CHUNK)
    return model.represent(model.bag_pairs(model.embed(list(bags), domain, chunk=EVAL_CHUNK)))


@torch.no_grad()
def euclidean_distances(model, target: Sequence[torch.Tensor],
                        source: Sequence[torch.Tensor]) -> np.ndarray:
    zt = bag_representations(model, target, "target").to(torch.float64)
    zs = bag_representations(model, source, "source").to(torch.float64)
    return torch.cdist(zt, zs).cpu().numpy()


def separation(D: np.ndarray, subjects: Sequence[str]) -> list[dict]:
    """Per subject j: mean over k != j of D[j, k] minus D[j, j]."""
    D = np.asarray(D, dtype=float)
    n = len(subjects)
    if D.shape != (n, n) or n < 2:
        raise MechanismError(f"separation needs a square matrix over >= 2 probe subjects; "
                             f"got {D.shape} for {n}")
    out = []
    for j, s in enumerate(subjects):
        others = float(np.mean(np.delete(D[j], j)))
        out.append({"subject": s, "own": float(D[j, j]), "others": others,
                    "separation": others - float(D[j, j])})
    return out


def k2_subject_rows(model, a: assignment_mod.Assignment, bags: bag_data.BagSet, *,
                    rotation: int, seed: int, model_name: str, metric: str) -> list[dict]:
    """The probe subjects' separations for one model, rotation and seed."""
    pairs = probe_pairs(a, bags, rotation)
    if len(pairs) < 2:
        return []
    subjects = [s for s, _, _ in pairs]
    target = [_tensor(t) for _, t, _ in pairs]
    source = [_tensor(s) for _, _, s in pairs]
    if metric == "kernel":
        D = kernel_distances(model, target, source)
    elif metric == "euclidean":
        D = euclidean_distances(model, target, source)
    else:
        raise MechanismError(f"metric is 'kernel' (MIL-CREDA) or 'euclidean'; got {metric!r}")
    return [{"model": model_name, "transfer": a.transfer.name,
             "pair": "|".join(a.transfer.pair), "rotation": int(rotation), "seed": int(seed),
             "metric": metric, **row} for row in separation(D, subjects)]


def k2_tests(full_rows: Sequence[Mapping], no_local_rows: Sequence[Mapping] | None, *,
             full: str = two_phase.MODEL, no_local: str = K2_NO_LOCAL,
             diagnostics: Mapping[str, Mapping] | None = None) -> list[dict]:
    """Sec. 6.2: separation > 0 and full > no-local on the two declared transfers, Holm/4.

    All four tests are always computed and Holm'd over four. When `diagnostics`
    carries a transfer's reading gate (v56), its two rows carry
    `inconclusive_no_room` and, where the local term had no room, the verdict
    `K2_INCONCLUSIVE` -- never "not supported"; the tests are reported either way.
    """
    experiment = "K2-emparejamiento"
    full_rows = once_per_subject(full_rows)
    no_local_rows = once_per_subject(no_local_rows or [])
    family = []
    for label, transfer in K2_DECLARED.items():
        mine = [r for r in full_rows if r["transfer"] == transfer.name]
        gate = _k2_gate_extra(diagnostics, transfer.name)
        family.append(_wilcoxon_row(
            [r["separation"] for r in mine],
            comparison=f"{experiment}:{transfer.name} ({label}):separation, {full}",
            name="matched-pairs rank-biserial (others - own distance)", experiment=experiment,
            extra={"declared": label, "k2_test": "separation", "transfer": transfer.name,
                   **gate}))
        other = {r["subject"]: r for r in no_local_rows if r["transfer"] == transfer.name}
        diffs = [r["separation"] - other[r["subject"]]["separation"] for r in mine
                 if r["subject"] in other]
        note = None if other else (f"the global-term-only variant (Sec. 6.3 ablation) is not "
                                   f"trained and stored for {transfer.name}")
        family.append(_wilcoxon_row(
            diffs, comparison=f"{experiment}:{transfer.name} ({label}):{full} vs {no_local}",
            name=f"matched-pairs rank-biserial ({full} - {no_local} separation)",
            experiment=experiment, extra={"declared": label, "k2_test": "no local",
                                          "transfer": transfer.name, **gate,
                                          **({"not_computed": note} if note else {})}))
    return st.apply_holm(family, f"{experiment} criterion tests")


def k2_has_room(median_entropy_ratio: float, median_room_ratio: float) -> bool:
    """Sec. 6.2's reading gate (v56): read where entropy < 0.9 OR room >= 1.0."""
    # None is a nan read back from JSON
    entropy = float("nan") if median_entropy_ratio is None else float(median_entropy_ratio)
    room = float("nan") if median_room_ratio is None else float(median_room_ratio)
    low_entropy = not np.isnan(entropy) and entropy < K2_ENTROPY_MAX
    enough_room = not np.isnan(room) and room >= K2_ROOM_MIN
    return bool(low_entropy or enough_room)


def _k2_gate_extra(diagnostics: Mapping[str, Mapping] | None, transfer: str) -> dict:
    """The `inconclusive_no_room` flag and, when it fires, the K2 verdict."""
    if not diagnostics or transfer not in diagnostics or \
            diagnostics[transfer].get("not_computed"):
        return {}
    d = diagnostics[transfer]
    read = k2_has_room(d["median_entropy_ratio"], d["median_room_ratio"])
    return {"inconclusive_no_room": not read,
            **({} if read else {"verdict": K2_INCONCLUSIVE})}


def _same_class_spread(D_ss: np.ndarray, members: np.ndarray) -> float:
    """Median pairwise H distance among the same-class source bags; nan for < 2."""
    if members.size < 2:
        return float("nan")
    block = D_ss[np.ix_(members, members)]
    iu = np.triu_indices(members.size, k=1)
    return float(np.median(block[iu]))


def k2_transfer_diagnostics(K_st, source_labels, target_classes, own_index,
                            D_start, D_start_ss, tau_loc, *, transfer: str | None = None,
                            D_end_full=None, D_end_no_local=None) -> dict:
    """Sec. 6.2's mechanism diagnostics on one transfer's probe target bags (v56).

    Measured with the frozen sigma, the phase-1 source encoder and the target
    encoder at the start of phase 2 (the phase-1 checkpoint), before the
    criterion verdict is read. For each target bag `j`, `S_c` is the source
    bags of its pseudo-label `target_classes[j]` and `own_index[j]` is its own
    source bag:

    - median over bags of the entropy ratio of Eq. 34's pi over `S_c`
      (`class_correspondence` at `tau_loc`, `entropy_ratio`): 1 uniform, 0 one-hot;
    - fraction of bags whose own source bag is the kappa^B-nearest of `S_c`;
    - median room ratio: own-partner H distance at start over the median
      same-class source-bag spread;
    - push toward the own source bag, start minus end, with the local term
      (`D_end_full`) and without it (`D_end_no_local`), when each is given.
    """
    K_st = np.asarray(K_st, dtype=float)
    D_start = np.asarray(D_start, dtype=float)
    D_start_ss = np.asarray(D_start_ss, dtype=float)
    source_labels = np.asarray(source_labels)
    target_classes = np.asarray(target_classes)
    own_index = np.asarray(own_index, dtype=int)
    ratios, nearest_hits, rooms, push_full, push_no_local = [], [], [], [], []
    for j in range(K_st.shape[0]):
        members = np.nonzero(source_labels == target_classes[j])[0]
        if members.size == 0:
            continue
        own = int(own_index[j])
        pi = class_correspondence(torch.as_tensor(K_st[j]),
                                  torch.as_tensor(members), float(tau_loc))
        ratios.append(entropy_ratio(pi))
        nearest = int(members[int(np.argmax(K_st[j, members]))])
        nearest_hits.append(float(nearest == own))
        spread = _same_class_spread(D_start_ss, members)
        if not np.isnan(spread) and spread > 0:
            rooms.append(float(D_start[j, own]) / spread)
        if D_end_full is not None:
            push_full.append(float(D_start[j, own]) - float(np.asarray(D_end_full)[j, own]))
        if D_end_no_local is not None:
            push_no_local.append(float(D_start[j, own])
                                 - float(np.asarray(D_end_no_local)[j, own]))
    return _k2_aggregate(transfer, int(K_st.shape[0]), {
        "entropy_ratio": ratios, "own_nearest": nearest_hits, "room_ratio": rooms,
        "push_with_local": push_full, "push_without_local": push_no_local})


def _k2_aggregate(transfer: str | None, probe_subjects: int,
                  per_bag: Mapping[str, Sequence[float]]) -> dict:
    ratios, hits, rooms = per_bag["entropy_ratio"], per_bag["own_nearest"], per_bag["room_ratio"]
    full, no_local = per_bag["push_with_local"], per_bag["push_without_local"]
    return {"transfer": transfer,
            "probe_subjects": int(probe_subjects),
            "median_entropy_ratio": float(np.nanmedian(ratios)) if ratios else float("nan"),
            "fraction_own_nearest": float(np.mean(hits)) if hits else float("nan"),
            "median_room_ratio": float(np.median(rooms)) if rooms else float("nan"),
            "push_with_local": float(np.mean(full)) if full else None,
            "push_without_local": float(np.mean(no_local)) if no_local else None,
            "per_bag": {k: list(v) for k, v in per_bag.items()}}


def k2_pool_diagnostics(transfer: str, parts: Sequence[Mapping]) -> dict:
    """One transfer's diagnostics over its rotations: the per-bag values of every part
    (`k2_transfer_diagnostics`) pooled, then the medians and means taken once."""
    keys = ("entropy_ratio", "own_nearest", "room_ratio", "push_with_local",
            "push_without_local")
    pooled = {k: [v for p in parts for v in p["per_bag"][k]] for k in keys}
    out = _k2_aggregate(transfer, sum(p["probe_subjects"] for p in parts), pooled)
    out["rotations"] = len(parts)
    return out


def k2_phase_two_start(record):
    """The MIL-CREDA model at the start of phase 2: the stored phase-1 checkpoint
    `(state, meta)` -- source encoder, attention and head -- and the target encoder as the
    bridge initialises it from the source encoder, at the frozen sigma."""
    state, meta = record
    if meta.get("phase") != 1 or meta.get("name") != two_phase.MODEL:
        raise MechanismError(f"the start of phase 2 is MIL-CREDA's phase-1 checkpoint; got "
                             f"phase {meta.get('phase')} of {meta.get('name')}")
    if meta.get("sigma") is None:
        raise MechanismError("the MIL-CREDA phase-1 checkpoint carries no sigma")
    model = model_store.build_stored(two_phase.MODEL, meta["channels"]["source"],
                                     meta["channels"]["target"], meta["hyper"])
    _load_phase_one_state(model, state, float(meta["sigma"]))
    two_phase.bridge(model)
    model.eval()
    return model


def k2_rotation_diagnostics(start, full, no_local, pairs, *, tau_loc: float,
                            transfer: str) -> dict:
    """Sec. 6.2's diagnostics of one transfer, rotation and seed on its probe pairs
    `(subject, target bag, source bag)`: `start` (`k2_phase_two_start`) measures the
    geometry at the start of phase 2; `full` and `no_local` (None when not stored) the end."""
    target = [_tensor(t) for _, t, _ in pairs]
    source = [_tensor(s) for _, _, s in pairs]
    geometry = kernel_geometry(start, target, source)
    own = np.arange(len(pairs))
    return k2_transfer_diagnostics(
        geometry["K_st"], [two_phase.CLASSES.index(s.label) for _, _, s in pairs],
        geometry["pseudo"], own,
        geometry["D_ts"], geometry["D_ss"], tau_loc, transfer=transfer,
        D_end_full=kernel_distances(full, target, source),
        D_end_no_local=None if no_local is None else kernel_distances(no_local, target,
                                                                       source))


def k2_diagnostics_table(diagnostics: Sequence[Mapping]) -> list[dict]:
    """Sec. 6.2's diagnostics table: one row per transfer, headers verbatim."""
    out = []
    for d in diagnostics:
        if d.get("not_computed"):
            reading = f"{K2_NOT_COMPUTED}: {d['not_computed']}"
        elif not k2_has_room(d["median_entropy_ratio"], d["median_room_ratio"]):
            reading = K2_INCONCLUSIVE
        else:
            reading = "leída"
        out.append(dict(zip(K2_DIAGNOSTICS_COLUMNS,
                            (d.get("transfer"), d.get("probe_subjects"),
                             d.get("median_entropy_ratio"), d.get("fraction_own_nearest"),
                             d.get("median_room_ratio"), d.get("push_with_local"),
                             d.get("push_without_local"), reading))))
    return out


def both_vs_global_only_inconclusive(direction_diagnostics: Sequence[Mapping]) -> bool | None:
    """Sec. 6.3 (v56): the both-terms vs global-only row of a core pair is
    inconclusive only when NEITHER of its two directions meets the Sec. 6.2 gate.
    Directions not computed are left out; None when no direction was computed.
    """
    diags = [d for d in direction_diagnostics if not d.get("not_computed")]
    if not diags:
        return None
    return not any(k2_has_room(d["median_entropy_ratio"], d["median_room_ratio"])
                   for d in diags)


def _interval(row: Mapping) -> str:
    lo, hi = row.get("ci_low"), row.get("ci_high")
    return "" if lo is None or hi is None else f"[{lo:.4f}, {hi:.4f}]"


def k1_test_table(rows: Sequence[Mapping]) -> list[dict]:
    """Sec. 6.1's test table from `k1_tests` rows, one per contrast, in their order."""
    return [dict(zip(K1_TABLE_COLUMNS, (r["comparison"], r.get("subjects", r["n"]),
                                        r["statistic"], r["p_value"],
                                        r["effect_size"], r.get("hodges_lehmann"),
                                        _interval(r)))) for r in rows]


def k2_test_table(rows: Sequence[Mapping]) -> list[dict]:
    """Sec. 6.2's test table from `k2_tests` rows, in the plan's row order."""
    by = {(r["declared"], r["k2_test"]): r for r in rows}
    out = []
    for declared, test, label in K2_TABLE_ROWS:
        r = by.get((declared, test))
        if r is None:
            raise MechanismError(f"no K2 test row for {declared}, {test}")
        out.append(dict(zip(K2_TABLE_COLUMNS, (label, r["statistic"], r["p_value"],
                                               r["p_adjusted"], r["effect_size"],
                                               r.get("hodges_lehmann"), _interval(r),
                                               r["n"]))))
    return out


# ------------------------------------------------------------------ K3 (Sec. 6.3)


def bound_control(history: Iterable[Mapping], model: str = "MIL-CREDA:phase2") -> list[dict]:
    """Declared bound and observed min and max of each normalized term over every step."""
    records = [r for r in history if r.get("model") == model]
    if not records:
        raise NotRecorded(f"no {model} history to read the term bounds from")
    out = []
    for term in TERMS:
        lows = [r[f"{term}_min"] for r in records]
        highs = [r[f"{term}_max"] for r in records]
        bound = (records[0][f"{term}_bound_low"], records[0][f"{term}_bound_high"])
        lo, hi = float(min(lows)), float(max(highs))
        out.append({"term": term, "bound_low": bound[0], "bound_high": bound[1],
                    "observed_min": lo, "observed_max": hi,
                    "within": bool(bound[0] <= lo <= hi <= bound[1]),
                    "epochs_read": len(records)})
    return out


def variant_table(predictions: Mapping[str, Sequence[Mapping]]) -> list[dict]:
    """Per variant: pooled BA, AUC, sensitivity and specificity at the report seed."""
    rows = []
    for name, records in predictions.items():
        summary = st.pooled(records)
        rows.append({"variant": name, **{m: summary[m] for m in
                                         ("balanced_accuracy", "auc", "sensitivity",
                                          "specificity")}})
    return rows


def k3_contrasts(table: str, pair: str, reference: str, predictions: Mapping[str, Sequence],
                 rivals: Sequence[str], *, test_train_ratio: float,
                 resamples: int = st.BOOTSTRAP_RESAMPLES) -> list[dict]:
    """The bootstrap contrasts of one table and core pair (Holm is applied per table)."""
    rows = []
    for rival in rivals:
        rows += st.pair_contrast(predictions[reference], predictions[rival], name_a=reference,
                                 name_b=rival, scope=f"{table}:{pair}",
                                 test_train_ratio=test_train_ratio, hypothesis="K3-ablacion",
                                 experiment="K3-ablacion", resamples=resamples)
    return rows


#: Full MIL-CREDA's phase-2 keys: the only phase-2 hyperparameters a variant receives.
MIL_CREDA_PHASE_TWO = ("lr2", "wd2", "tauLocal")


def _exactly(params: Mapping, keys: Sequence[str], what: str) -> dict:
    if set(params) != set(keys):
        raise MechanismError(f"a K3 variant has no search of its own (Sec. 6.3): it takes "
                             f"exactly full MIL-CREDA's selected {what} {sorted(keys)}; got "
                             f"{sorted(params)}")
    return dict(params)


def _variant_history(records: Sequence[Mapping], variant: str) -> list[dict]:
    """Tag `MIL-CREDA:phase<n>` as `MIL-CREDA [<variant>]:phase<n>`."""
    out = []
    for r in records:
        phase = str(r["model"]).rsplit(":", 1)[-1]
        out.append({**r, "model": f"{two_phase.MODEL} [{variant}]:{phase}"})
    return out


def _load_phase_one_state(model, state: Mapping, sigma: float) -> None:
    """A phase-1 state into a MIL-CREDA model, at its sigma; only the target trunk may lack."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise MechanismError(f"the phase-1 state does not fit (missing {missing}, unexpected "
                             f"{unexpected})")
    model.fix_sigma(sigma)


def _adapt_from_phase_one(model, state: Mapping, sigma: float, data: search.SearchData,
                          rotation: int, seed: int, hyper: Mapping, device: torch.device,
                          term_selection: str) -> dict:
    """Phase 2 from a phase-1 state, as `search.train_mil_creda_phase_two` runs it."""
    _load_phase_one_state(model, state, sigma)
    generator = torch.Generator().manual_seed(seed)
    joint = two_phase.bridge(model)
    d = search._rotation(data, rotation)
    result = two_phase.adapt_target_phase(
        model, {"checkpoint": {"sigma": sigma}}, joint["frozen_state"], d["source_train"],
        d["target_train"], d["target_valid"], hyper, generator, device, seed=seed,
        fold=rotation, transfer=data.transfer.name, term_selection=term_selection)
    return {"result": result, "layer_copy": joint["layer_copy"]}


def train_ablation_variant(data: search.SearchData, rotation: int, seed: int, record,
                           params: Mapping, schedule: Mapping, term_selection: str,
                           device: torch.device | str = "cpu") -> dict:
    """Sec. 6.3's ablation: full MIL-CREDA's phase 1, phase 2 with `term_selection`.

    `record` is full MIL-CREDA's one stored phase-1 checkpoint `(state, meta)` of this
    transfer and rotation, at `search.PHASE_ONE_SEED`, whatever seed the variant trains
    at; `params` are full MIL-CREDA's selected phase-2
    hyperparameters of this transfer.
    """
    two_phase.term_coefficients(term_selection)
    state, meta = record
    expected = {"phase": 1, "name": two_phase.MODEL, "transfer": data.transfer.name,
                "rotation": int(rotation), "seed": search.PHASE_ONE_SEED}
    wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        raise MechanismError(f"the ablation starts from full MIL-CREDA's one phase-1 "
                             f"checkpoint of {data.transfer.name}, rotation {rotation}, seed "
                             f"{search.PHASE_ONE_SEED}; the record differs in {wrong}")
    if meta.get("sigma") is None:
        raise MechanismError("the MIL-CREDA phase-1 checkpoint carries no sigma; phase 2 "
                             "runs at exactly that value")
    device = torch.device(device)
    params = _exactly(params, MIL_CREDA_PHASE_TWO, "phase-2 hyperparameters")
    hyper = two_phase.check_hyper({**{k: meta["hyper"][k] for k in two_phase.PHASE_ONE_KEYS},
                                   **params, **search.check_schedule(schedule)})
    channels = data.channels()
    torch.manual_seed(seed)
    model = model_store.build_model(two_phase.MODEL, channels["source"], channels["target"],
                                    hyper).to(device)
    out = _adapt_from_phase_one(model, state, float(meta["sigma"]), data, rotation, seed,
                                hyper, device, term_selection)
    return {"variant": term_selection, "term_selection": term_selection, "model": model,
            "hyper": hyper, "channels": channels, "result": out["result"],
            "layer_copy": out["layer_copy"],
            "history": _variant_history(out["result"]["history"], term_selection)}


def aggregation_hyper(phase_one: Mapping, phase_two: Mapping, schedule: Mapping,
                      aggregation: str) -> dict:
    """Full MIL-CREDA's selected hyperparameters as `aggregation` reads them: gamma 0 for
    the own attention without consensus, NaN for what a published pooling lacks."""
    aggregation_variants.check_aggregation(aggregation)
    phase_one = _exactly(phase_one, tuple(search.PHASE_ONE_SPACES[two_phase.MODEL]),
                         "phase-1 hyperparameters")
    phase_two = _exactly(phase_two, MIL_CREDA_PHASE_TWO, "phase-2 hyperparameters")
    return two_phase.check_hyper(aggregation_variants.variant_hyper(
        aggregation, {**phase_one, **phase_two, **search.check_schedule(schedule)}))


def train_aggregation_variant(data: search.SearchData, rotation: int, seed: int,
                              aggregation: str, phase_one: Mapping, phase_two: Mapping,
                              schedule: Mapping, device: torch.device | str = "cpu", *,
                              phase_one_record=None) -> dict:
    """Sec. 6.3's aggregation table: one variant, both phases, at full MIL-CREDA's choice.

    `phase_one_record` is the variant's own stored phase-1 checkpoint `(state, meta)`
    of this transfer and rotation, at `search.PHASE_ONE_SEED`; given, phase 1 is not trained again and phase 2
    runs from it exactly as it would have run from the fresh checkpoint.
    """
    aggregation_variants.check_aggregation(aggregation)
    hyper = aggregation_hyper(phase_one, phase_two, schedule, aggregation)
    device = torch.device(device)
    channels = data.channels()
    d = search._rotation(data, rotation)

    def build():
        return aggregation_variants.build(aggregation, channels["source"], channels["target"],
                                          hyper).to(device)

    if phase_one_record is None:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = build()
        phase1 = two_phase.train_source_phase(model, d["source_train"], d["source_valid"],
                                              hyper, generator, device, seed=seed,
                                              fold=rotation, transfer=data.transfer.name)
        state, sigma = phase1["checkpoint"]["state"], phase1["checkpoint"]["sigma"]
    else:
        state, meta = phase_one_record
        expected = {"phase": 1, "name": model_store.variant_name(aggregation),
                    "transfer": data.transfer.name, "rotation": int(rotation),
                    "seed": search.PHASE_ONE_SEED}
        wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
        if wrong or meta.get("sigma") is None:
            raise MechanismError(f"the {aggregation} phase 2 starts from its own phase-1 "
                                 f"checkpoint of {data.transfer.name}, rotation {rotation}, "
                                 f"seed {search.PHASE_ONE_SEED}, with its sigma; the record "
                                 f"differs in {wrong}")
        phase1, sigma = None, float(meta["sigma"])
    torch.manual_seed(seed)
    model = build()
    out = _adapt_from_phase_one(model, state, sigma, data, rotation, seed, hyper, device,
                                two_phase.BOTH_TERMS)
    return {"variant": aggregation, "aggregation": aggregation,
            "form": aggregation_variants.FORMS.get(
                aggregation, aggregation_variants.WITHOUT_CONSENSUS_FORM),
            "model": model, "hyper": hyper,
            "not_applicable": list(aggregation_variants.not_applicable(aggregation)),
            "channels": channels, "phase1": phase1, "result": out["result"],
            "layer_copy": out["layer_copy"],
            "history": _variant_history((phase1["history"] if phase1 else [])
                                        + out["result"]["history"], aggregation)}
