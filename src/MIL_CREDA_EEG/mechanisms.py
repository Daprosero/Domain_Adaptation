"""The mechanism experiments, read from the stored models: K1, K2 and K3.

Experiments plan v40, Secs. 6.1-6.3, over proposal r34 Eqs. 14-19 (attention
and bag kernel), 28-31 (local alignment) and 34-38 (bounds of the normalized
terms). K1 and K2 train nothing: they read a model that `model_store` saved
(Sec. 7.3) or a history that training already recorded. K3 trains its variants
(`train_ablation_variant`, `train_aggregation_variant`).

**K1, the attention selects something meaningful (Sec. 6.1).** Only in the
conditions with an event stream (`EVENT_CONDITIONS`: the three high-density
levels and Robots), on the target test bags of each rotation, with the saved
phase-2 models. Per window: the attention weight (`attention_weights`, read
from the model's own attention: Eqs. 15-16 for MIL-CREDA and the common
wrapper, the native aggregator for AdaptMIL and HASD); whether it is marked by
sample loss (Sec. 2.3, `integrity_mark`) or bad contact (Sec. 2.5, any
channel's `contact_bad`), which `marked_windows` reads from the bag's own
marks; whether it holds a behavioural event (`event_windows`); and the ratio of
the clinical marker's two bands (`band_ratio`). Per subject: the mean weight
of the marked windows minus that of the clean ones, the mean weight of the
event windows minus that of the rest, and the Spearman correlation between
weight and band ratio. Each is averaged over the seeds first and then tested
with Wilcoxon over subjects (`k1_tests`), Holm over the three criterion tests;
the spectral row has no criterion and no correction. Subjects without marked
windows do not enter the first contrast and are counted.

**K2, local matching aligns compatible subjects (Sec. 6.2).** Over the probe
subjects of each rotation (`assignment.Assignment.probe_subjects`: the test
fold's subjects holding both conditions of the transfer), the target bag is the
subject's bag in the target condition through the target trunk, and its own
source bag is its bag in the source condition through the source trunk; the
condition the assignment did not read comes only through
`assignment.probe_bag`. Distances are the bag-kernel (RKHS) distances at the
frozen sigma (`kernel_distances`, Eqs. 17-19) for MIL-CREDA and Euclidean
distances between bag representations for the comparators
(`euclidean_distances`). Per subject the separation is the mean distance to
the other probe subjects' source bags minus the distance to its own
(`separation`), averaged over seeds and tested with Wilcoxon over the probe
subjects on the two declared transfers (`K2_DECLARED`), with Holm over four.

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
that checkpoint as full MIL-CREDA's, with `attentionGamma` and
`attentionTemperature` recorded as not applicable (NaN in the hyperparameters,
listed in `not_applicable`), never reused. Each variant's history is tagged
`MIL-CREDA [<variant>]:phase<n>`, so the bound control of full MIL-CREDA never
reads it.

**Implementation choices the plan does not fix.** The K1 band edges are a
required input (`bands`), because Sec. 2.7 records two conventions for the
high band and the plan does not pick one for the ratio. Which comparator's
model carries "the common attention" in K1 is a required input too: in the
device arm a phase-1 model cannot read the target, so a phase-2 model of one of
the six non-MIL comparators is needed and the plan does not name one. Whether
K1 pools its subjects over every K1 transfer or tests each core pair apart is a
required input (`K1_SCOPES`); within one pair every assigned subject is
target exactly once, across pairs a subject can repeat. The weight-ratio
contrast is the per-subject Spearman correlation tested against zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

from MIL_CREDA.bag_kernel import bag_kernel_matrix
from MIL_CREDA_EEG import aggregation_variants
from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, model_store, search
from MIL_CREDA_EEG import statistics as st
from MIL_CREDA_EEG import two_phase
from MIL_CREDA_EEG.two_phase import EVAL_CHUNK

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["6.1", "6.2", "6.3", "7.3"],
    "experiments": ["K1-atencion", "K2-emparejamiento", "K3-ablacion"],
    "equations": ["14", "15", "16", "17", "18", "19", "37", "38"],
    "invariants": [
        "k1_reads_only_conditions_with_an_event_stream",
        "a_marked_window_is_an_integrity_or_contact_mark",
        "an_unrecorded_event_mark_is_refused_not_guessed",
        "the_attention_weights_are_the_models_own_and_sum_to_one",
        "the_weight_contrast_is_marked_minus_clean_and_skips_subjects_without_marks",
        "the_band_ratio_reads_the_declared_bands_and_has_no_default",
        "k1_averages_over_seeds_before_testing_and_holms_the_three_criterion_tests",
        "k2_reads_the_other_condition_only_through_the_probe",
        "the_kernel_distance_is_the_rkhs_distance_at_the_frozen_sigma",
        "the_separation_is_others_minus_own",
        "k2_tests_the_two_declared_transfers_with_holm_over_four",
        "the_bound_control_reads_every_recorded_step",
        "a_stored_phase_two_model_is_rebuilt_with_every_parameter",
        "the_variant_trainers_refuse_an_unknown_variant",
        "the_both_terms_ablation_is_full_mil_creda",
        "the_ablation_starts_from_full_mil_credas_phase_one_and_its_hyperparameters",
        "the_parameters_a_variant_lacks_are_not_applicable",
        "an_aggregation_variant_trains_both_phases",
    ],
}

#: Sec. 6.1: the conditions with an event stream; CogniFit has none.
EVENT_CONDITIONS = ("L1", "L6", "L11", "Robots")
#: The per-window mark of a behavioural event; preprocessing does not produce it yet.
EVENT_MARK = "event_window"
#: Sec. 2.3 and 2.5: the marks K1 reads.
INTEGRITY_MARK = "integrity_mark"
CONTACT_MARK = "contact_bad"
RATE = 128.0
K1_SCOPES = ("pooled", "per-pair")
#: Sec. 6.2: chosen before any result, by structure.
K2_DECLARED = {"smaller shift": assignment_mod.Transfer("level", "L1", "L6"),
               "larger shift": assignment_mod.Transfer("device", "L6", "Robots")}
#: Sec. 6.3.
AGGREGATIONS = ("own", *aggregation_variants.VARIANTS)
ABLATIONS = tuple(two_phase.TERM_SELECTIONS)
TERMS = ("global", "local")


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
    strategy's auxiliary training structures and are returned, not loaded.
    """
    state, meta = model_store.load(path, partition_path=partition_path,
                                   allow_code_mismatch=allow_code_mismatch)
    if meta["phase"] != 2:
        raise MechanismError(f"{path} is not a phase-2 model")
    group = search.PHASE_ONE_GROUP_OF[meta["name"]]
    model = model_store.build_model(group, meta["channels"]["source"],
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


def event_windows(bag: bag_data.Bag) -> np.ndarray:
    """Per window: it contains the instant of a behavioural event (Sec. 6.1)."""
    if bag.condition not in EVENT_CONDITIONS:
        raise MechanismError(f"{bag.condition} has no event stream (Sec. 6.1)")
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


def band_ratio(windows: np.ndarray, bands: Sequence[Sequence[float]],
               rate: float = RATE) -> np.ndarray:
    """Per window: low-band power over high-band power (periodogram sums), channels averaged."""
    if len(bands) != 2 or any(len(b) != 2 or not 0 <= b[0] < b[1] for b in bands):
        raise MechanismError(f"bands is ((low_lo, low_hi), (high_lo, high_hi)); got {bands}")
    x = np.asarray(windows, dtype=np.float64)
    power = np.abs(np.fft.rfft(x - x.mean(axis=-1, keepdims=True), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / rate)

    def band(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            raise MechanismError(f"no frequency bin in [{lo}, {hi}) Hz")
        return power[..., sel].sum(axis=-1).mean(axis=-1)

    low, high = band(*bands[0]), band(*bands[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return low / high


def k1_subject_rows(model, bags: Sequence[bag_data.Bag], *, model_name: str,
                    transfer: assignment_mod.Transfer, rotation: int, seed: int,
                    bands: Sequence[Sequence[float]], domain: str = "target") -> list[dict]:
    """Per target test subject: the three K1 quantities for one model, rotation and seed."""
    if transfer.target not in EVENT_CONDITIONS:
        raise MechanismError(f"K1 reads only targets with an event stream, not "
                             f"{transfer.target}")
    rows = []
    for bag in bags:
        windows = bag.windows()
        weights = attention_weights(model, torch.from_numpy(np.ascontiguousarray(windows))
                                    .float(), domain)
        marked = marked_windows(bag)
        try:
            event = weight_contrast(weights, event_windows(bag))
            event_note = None
        except NotRecorded as error:
            event, event_note = None, str(error)
        ratio = band_ratio(windows, bands)
        finite = np.isfinite(ratio)
        rho = (float(sps.spearmanr(weights[finite], ratio[finite]).statistic)
               if finite.sum() > 2 else None)
        rows.append({"model": model_name, "transfer": transfer.name,
                     "pair": "|".join(transfer.pair), "rotation": int(rotation),
                     "seed": int(seed), "subject": bag.subject,
                     "marked_minus_clean": weight_contrast(weights, marked),
                     "n_marked": int(marked.sum()), "n_windows": int(len(marked)),
                     "event_minus_rest": event, "event_not_recorded": event_note,
                     "weight_ratio_spearman": rho if rho is None or math.isfinite(rho)
                     else None})
    return rows


def average_over_seeds(rows: Iterable[Mapping], values: Sequence[str],
                       by: Sequence[str] = ("model", "transfer", "subject")) -> list[dict]:
    """Mean of each value over the seeds of each unit, ignoring seeds where it is None."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        row = dict(zip(by, key))
        row["pair"] = rs[0].get("pair")
        row["seeds"] = sorted({int(r["seed"]) for r in rs})
        notes = [r["event_not_recorded"] for r in rs if r.get("event_not_recorded")]
        if notes:
            row["event_not_recorded"] = notes[0]
        for v in values:
            xs = [r[v] for r in rs if r.get(v) is not None]
            row[v] = float(np.mean(xs)) if xs else None
        out.append(row)
    return out


K1_VALUES = ("marked_minus_clean", "event_minus_rest", "weight_ratio_spearman")


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
                               ci_low=None, ci_high=None, n=w["n"], hypothesis=experiment,
                               experiment=experiment, median=w["median"],
                               n_nonzero=w["n_nonzero"], **(extra or {}))


def k1_tests(own_rows: Sequence[Mapping], common_rows: Sequence[Mapping], *, own: str,
             common: str, scope: str) -> list[dict]:
    """Sec. 6.1's table from per-subject rows already averaged over seeds."""
    if scope not in K1_SCOPES:
        raise MechanismError(f"the K1 scope is a required choice among {list(K1_SCOPES)}")
    experiment = "K1-atencion"
    groups = ({"all": list(own_rows)} if scope == "pooled"
              else {p: [r for r in own_rows if r["pair"] == p]
                    for p in sorted({r["pair"] for r in own_rows})})
    common_by = {(r["transfer"], r["subject"]): r for r in common_rows}
    rows = []
    for label, members in groups.items():
        prefix = f"{experiment}:{label}"
        marked = [r["marked_minus_clean"] for r in members]
        skipped = sum(v is None for v in marked)
        paired = []
        for r in members:
            c = common_by.get((r["transfer"], r["subject"]))
            if c is None:
                raise MechanismError(f"{r['subject']} of {r['transfer']} has no {common} row")
            if r["marked_minus_clean"] is not None and c["marked_minus_clean"] is not None:
                paired.append(r["marked_minus_clean"] - c["marked_minus_clean"])
        events = [r["event_minus_rest"] for r in members]
        family = [
            _wilcoxon_row(marked, comparison=f"{prefix}:marked vs clean, {own}",
                          name="matched-pairs rank-biserial (marked - clean weight)",
                          experiment=experiment,
                          extra={"subjects_without_marked_windows": skipped}),
            _wilcoxon_row(paired, comparison=f"{prefix}:marked vs clean, {own} vs {common}",
                          name=f"matched-pairs rank-biserial ({own} - {common})",
                          experiment=experiment, extra={"common_attention_model": common}),
            _wilcoxon_row(events, comparison=f"{prefix}:event vs rest, {own}",
                          name="matched-pairs rank-biserial (event - rest weight)",
                          experiment=experiment,
                          extra={"not_computed": next((r["event_not_recorded"] for r in members
                                                       if r.get("event_not_recorded")), None)}),
        ]
        rows += st.apply_holm(family, f"{prefix} criterion tests")
        rows.append(_wilcoxon_row(
            [r["weight_ratio_spearman"] for r in members],
            comparison=f"{prefix}:attention weight vs band ratio, {own}",
            name="matched-pairs rank-biserial of per-subject Spearman rho",
            experiment=experiment, extra={"criterion": "none (reported both ways)"}))
    return rows


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


K2_VALUES = ("own", "others", "separation")


def k2_tests(full_rows: Sequence[Mapping], no_local_rows: Sequence[Mapping] | None, *,
             full: str = "MIL-CREDA", no_local: str = "MIL-CREDA, global term only") -> list[dict]:
    """Sec. 6.2: separation > 0 and full > no-local on the two declared transfers, Holm/4."""
    experiment = "K2-emparejamiento"
    family = []
    for label, transfer in K2_DECLARED.items():
        mine = [r for r in full_rows if r["transfer"] == transfer.name]
        family.append(_wilcoxon_row(
            [r["separation"] for r in mine],
            comparison=f"{experiment}:{transfer.name} ({label}):separation, {full}",
            name="matched-pairs rank-biserial (others - own distance)", experiment=experiment))
        if no_local_rows is None:
            family.append(_wilcoxon_row(
                [], comparison=f"{experiment}:{transfer.name} ({label}):{full} vs {no_local}",
                name=f"matched-pairs rank-biserial ({full} - {no_local} separation)",
                experiment=experiment,
                extra={"not_computed": "the global-term-only variant is not trained yet "
                                       "(Sec. 6.3 ablation)"}))
            continue
        other = {r["subject"]: r for r in no_local_rows if r["transfer"] == transfer.name}
        diffs = [r["separation"] - other[r["subject"]]["separation"] for r in mine
                 if r["subject"] in other]
        family.append(_wilcoxon_row(
            diffs, comparison=f"{experiment}:{transfer.name} ({label}):{full} vs {no_local}",
            name=f"matched-pairs rank-biserial ({full} - {no_local} separation)",
            experiment=experiment))
    return st.apply_holm(family, f"{experiment} criterion tests")


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
    """Per variant: pooled BA, AUC, sensitivity and specificity, mean and sd over seeds."""
    rows = []
    for name, records in predictions.items():
        summary = st.summarize_seeds(st.pooled_by_seed(records))
        rows.append({"variant": name, **{f"{m}_{k}": summary[m][k] for m in
                                         ("balanced_accuracy", "auc", "sensitivity",
                                          "specificity") for k in ("mean", "sd")}})
    return rows


def k3_contrasts(table: str, pair: str, reference: str, predictions: Mapping[str, Sequence],
                 rivals: Sequence[str], *, reduction: str, test_train_ratio: float,
                 resamples: int = st.BOOTSTRAP_RESAMPLES) -> list[dict]:
    """The bootstrap contrasts of one table and core pair (Holm is applied per table)."""
    rows = []
    for rival in rivals:
        rows += st.pair_contrast(predictions[reference], predictions[rival], name_a=reference,
                                 name_b=rival, scope=f"{table}:{pair}", reduction=reduction,
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


def _adapt_from_phase_one(model, state: Mapping, sigma: float, data: search.SearchData,
                          rotation: int, seed: int, hyper: Mapping, device: torch.device,
                          term_selection: str) -> dict:
    """Phase 2 from a phase-1 state, as `search.train_mil_creda_phase_two` runs it."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise MechanismError(f"the phase-1 state does not fit (missing {missing}, unexpected "
                             f"{unexpected})")
    model.fix_sigma(sigma)
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

    `record` is full MIL-CREDA's stored phase-1 checkpoint `(state, meta)` of this
    transfer, rotation and seed; `params` are full MIL-CREDA's selected phase-2
    hyperparameters of this transfer.
    """
    two_phase.term_coefficients(term_selection)
    state, meta = record
    expected = {"phase": 1, "name": two_phase.MODEL, "transfer": data.transfer.name,
                "rotation": int(rotation), "seed": int(seed)}
    wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        raise MechanismError(f"the ablation starts from full MIL-CREDA's phase-1 checkpoint "
                             f"of {data.transfer.name}, rotation {rotation}, seed {seed}; the "
                             f"record differs in {wrong}")
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


def aggregation_hyper(phase_one: Mapping, phase_two: Mapping, schedule: Mapping) -> dict:
    """Full MIL-CREDA's selected hyperparameters, with what a variant lacks set to NaN."""
    phase_one = _exactly(phase_one, tuple(search.PHASE_ONE_SPACES[two_phase.MODEL]),
                         "phase-1 hyperparameters")
    phase_two = _exactly(phase_two, MIL_CREDA_PHASE_TWO, "phase-2 hyperparameters")
    nan = float("nan")
    return two_phase.check_hyper({**phase_one,
                                  **{k: nan for k in aggregation_variants.NOT_APPLICABLE},
                                  **phase_two, **search.check_schedule(schedule)})


def train_aggregation_variant(data: search.SearchData, rotation: int, seed: int,
                              aggregation: str, phase_one: Mapping, phase_two: Mapping,
                              schedule: Mapping, device: torch.device | str = "cpu") -> dict:
    """Sec. 6.3's aggregation table: one variant, both phases, at full MIL-CREDA's choice."""
    aggregation_variants.check_aggregation(aggregation)
    hyper = aggregation_hyper(phase_one, phase_two, schedule)
    device = torch.device(device)
    channels = data.channels()
    d = search._rotation(data, rotation)

    def build():
        return aggregation_variants.AggregationVariantModel(
            channels["source"], channels["target"], aggregation=aggregation).to(device)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = build()
    phase1 = two_phase.train_source_phase(model, d["source_train"], d["source_valid"], hyper,
                                          generator, device, seed=seed, fold=rotation,
                                          transfer=data.transfer.name)
    checkpoint = phase1["checkpoint"]
    torch.manual_seed(seed)
    model = build()
    out = _adapt_from_phase_one(model, checkpoint["state"], checkpoint["sigma"], data,
                                rotation, seed, hyper, device, two_phase.BOTH_TERMS)
    return {"variant": aggregation, "aggregation": aggregation,
            "form": aggregation_variants.FORMS[aggregation], "model": model, "hyper": hyper,
            "not_applicable": list(aggregation_variants.NOT_APPLICABLE),
            "channels": channels, "phase1": phase1, "result": out["result"],
            "layer_copy": out["layer_copy"],
            "history": _variant_history(phase1["history"] + out["result"]["history"],
                                        aggregation)}
