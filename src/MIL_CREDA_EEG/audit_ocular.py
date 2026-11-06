"""Ocular artifact: MEASURE, never act.

Experiments plan v57, Sec. 2.8. The plan declares a defect ("a record whose
fraction of contaminated epochs exceeds the declared threshold") but no
threshold, and no definition of a contaminated epoch. Before any rule exists
the question the operator asked comes first: is ocular activity in the EEG
associated with the label, with sex or with age? If it is, an encoder can learn
eye movement instead of physiology (Sec. 2.8, third paragraph).

**Nothing here modifies the signal or excludes anything.** Every four-second
window of every record is measured, including the ones touching the device
rail; how many touch it is reported next to the measure, so a reader can see
when saturation, not the eyes, drives a number. A flat channel (zero variance,
the Sec. 2.4 definition) enters neither measure; when every posterior channel
is flat the ratio is undefined and reads as missing, not as a huge number.

The bipolar externals cannot carry the measurement: on every CGX record they
sit constant at the rail (Sec. 2.4 audit). The proxy is the scalp itself, read
where ocular activity is largest and slowest:

1. **Frontal peak-to-peak** per window, the largest over the frontal channels:
   a blink or a saccade is a deflection of tens to hundreds of microvolts over
   the front of the head.
2. **Low-frequency ratio** per window: power in 0.5-4 Hz averaged over the
   frontal channels, over the same power averaged over the posterior ones.
   Ocular activity is frontal and slow, so it raises this ratio; slow activity
   at the back lowers it.

The signal is band-passed with the band Sec. 2.7 declares (0.2-45 Hz), on the
continuous recording and only then cut into windows, as Sec. 2.7 orders. The
filter output is a copy used for measurement; the record is never rewritten.

Per record, each measure is summarised by its median and its upper quantiles
over windows. No cut is chosen: the distributions and their association with
label, sex and age are the deliverable.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, rankdata, spearmanr, t as student_t, wilcoxon

from MIL_CREDA_EEG import inventory
from MIL_CREDA_EEG.audit_channels import BAND, RAILS, bandpass
from MIL_CREDA_EEG.audit_rate import effective_rate
from MIL_CREDA_EEG.defect_log import _clean
from MIL_CREDA_EEG.modality import (
    CGX_NOMINAL_RATE, cgx_name, emotiv_nominal_rate, family_of, read_columns,
    read_emotiv_header,
)

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md",
    "sections": ["2.8"],
    "experiments": ["2.8"],
    "invariants": [
        "a_slow_frontal_blink_raises_the_frontal_posterior_ratio",
        "posterior_only_slow_activity_does_not_raise_the_ratio",
        "measures_never_alter_the_input_array",
        "every_window_is_measured_and_none_is_dropped",
        "a_flat_channel_carries_no_measure_and_a_flat_posterior_set_leaves_the_ratio_undefined",
        "the_channels_used_are_named_and_absent_ones_are_reported",
        "the_label_cross_keeps_undetermined_apart_and_tests_yes_against_no",
        "an_age_effect_is_not_read_as_a_label_effect_after_adjustment",
        "holm_adjusts_upward_and_keeps_order",
    ],
}

WINDOW_S = 4.0
#: Where blinks and saccades put their power (delta band).
LOW_BAND = (0.5, 4.0)
#: Upper quantiles reported next to the median, over a record's windows.
QUANTILES = (0.90, 0.99)

#: Frontal channels: Emotiv's four most anterior; CGX's prefrontal and
#: anterior-frontal row (Fp1, Fpz, Fp2, AF7, AF8 in its Quick-32r montage).
FRONTAL = {"EMOTIV": ("AF3", "AF4", "F7", "F8"),
           "CGX": ("Fp1", "Fp2", "Fpz", "AF7", "AF8")}
#: Posterior reference: occipital channels (Emotiv has no Oz).
POSTERIOR = {"EMOTIV": ("O1", "O2"), "CGX": ("O1", "O2", "Oz")}

#: Measures crossed with demographics (per record, then averaged per subject).
MEASURES = ("frontal_ptp_median", "frontal_ptp_p90",
            "low_ratio_median", "low_ratio_p90")


def channels_used(available: list[str], family: str) -> dict[str, list[str]]:
    """The frontal and posterior channels this record carries, and the ones it lacks."""
    frontal = [c for c in FRONTAL[family] if c in available]
    posterior = [c for c in POSTERIOR[family] if c in available]
    missing = [c for c in FRONTAL[family] + POSTERIOR[family] if c not in available]
    return {"frontal": frontal, "posterior": posterior, "missing": missing}


def window_measures(signal: np.ndarray, channels: list[str], rate: float,
                    frontal: list[str] | tuple[str, ...],
                    posterior: list[str] | tuple[str, ...],
                    rails: tuple[float, float] = RAILS["EMOTIV"],
                    window_s: float = WINDOW_S) -> dict[str, np.ndarray]:
    """Per window: frontal peak-to-peak, frontal/posterior low-band power, rail touch.

    Works on a copy; missing samples are filled with the channel median only in
    that copy, so the filter can run over the continuous recording.
    """
    index = {c: k for k, c in enumerate(channels)}
    f_idx = [index[c] for c in frontal if c in index]
    p_idx = [index[c] for c in posterior if c in index]
    x = np.array(np.asarray(signal)[:, f_idx + p_idx], dtype=float, copy=True)
    finite = np.isfinite(x)
    usable = finite.any(axis=0)
    medians = np.where(usable, np.nanmedian(np.where(finite, x, np.nan), axis=0)
                       if usable.any() else 0.0, 0.0)
    x = np.where(finite, x, medians)
    # A flat channel (zero variance, Sec. 2.4's definition) carries no measure:
    # its filtered power is numerical residue, and a ratio over it is undefined.
    usable &= x.std(axis=0) > 0
    at_rail = (x <= rails[0]) | (x >= rails[1])
    filtered = bandpass(x, rate, BAND)

    size = int(round(rate * window_s))
    n = len(filtered) // size
    nf = len(f_idx)
    empty = {"frontal_ptp": np.array([]), "low_ratio": np.array([]),
             "rail_windows": np.array([], dtype=bool)}
    if n == 0 or nf == 0 or len(p_idx) == 0:
        return empty
    windows = filtered[: n * size].reshape(n, size, -1)
    ptp = windows.max(axis=1) - windows.min(axis=1)
    ptp[:, ~usable] = np.nan
    frontal_ptp = np.nanmax(ptp[:, :nf], axis=1) if usable[:nf].any() \
        else np.full(n, np.nan)

    centred = windows - windows.mean(axis=1, keepdims=True)
    taper = np.hanning(size)[None, :, None]
    power = np.abs(np.fft.rfft(centred * taper, axis=1)) ** 2
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    low = power[:, (freqs >= LOW_BAND[0]) & (freqs <= LOW_BAND[1]), :].sum(axis=1)
    low[:, ~usable] = np.nan
    front_low = np.nanmean(low[:, :nf], axis=1) if usable[:nf].any() else np.full(n, np.nan)
    back_low = np.nanmean(low[:, nf:], axis=1) if usable[nf:].any() else np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(back_low > 0, front_low / back_low, np.nan)
    rail_windows = at_rail[: n * size].reshape(n, size, -1).any(axis=(1, 2))
    return {"frontal_ptp": frontal_ptp, "low_ratio": ratio, "rail_windows": rail_windows}


def summarize(measures: dict[str, np.ndarray]) -> dict:
    """Median and upper quantiles per measure over a record's windows."""
    out = {"n_windows": int(len(measures["frontal_ptp"]))}
    for key in ("frontal_ptp", "low_ratio"):
        v = np.asarray(measures[key], dtype=float)
        v = v[np.isfinite(v)]
        out[f"{key}_median"] = float(np.median(v)) if len(v) else None
        for q in QUANTILES:
            out[f"{key}_p{int(round(q * 100))}"] = float(np.quantile(v, q)) if len(v) else None
    rails = np.asarray(measures["rail_windows"], dtype=bool)
    out["rail_window_fraction"] = float(rails.mean()) if len(rails) else None
    return out


# ------------------------------------------------------------- reading


def _read(path: str) -> tuple[str, np.ndarray, list[str], np.ndarray, float]:
    """Only the timestamps and the frontal/posterior columns of one EEG file."""
    family, _ = family_of(path)
    columns = read_columns(path)
    if family == "EMOTIV":
        wanted = {f"EEG.{c}": c for c in FRONTAL[family] + POSTERIOR[family]}
        time_column = "Timestamp"
        nominal = emotiv_nominal_rate(read_emotiv_header(path))
    else:
        wanted = {col: cgx_name(col) for col in columns
                  if cgx_name(col) in FRONTAL[family] + POSTERIOR[family]}
        time_column = "timestamps"
        nominal = CGX_NOMINAL_RATE
    present = [c for c in columns if c in wanted]
    frame = pd.read_csv(path, skiprows=1 if family == "EMOTIV" else 0,
                        usecols=present + [time_column], low_memory=False)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    timestamps = frame[time_column].to_numpy(dtype=float)
    rate = nominal or effective_rate(timestamps)
    return (family, frame[present].to_numpy(dtype=float), [wanted[c] for c in present],
            timestamps, float(rate))


def ocular_record(path: str, meta: dict) -> dict:
    """Both measures over one EEG file; an unreadable file reports its reason."""
    out = dict(meta)
    try:
        family, signal, channels, timestamps, rate = _read(path)
        if len(timestamps) < 2:
            raise ValueError(f"too few samples: {len(timestamps)}")
    except Exception as error:  # noqa: BLE001 -- the reason is the deliverable
        out.update({"readable": False, "error": f"{type(error).__name__}: {error}"})
        return _clean(out)
    used = channels_used(channels, family)
    measures = window_measures(signal, channels, rate, used["frontal"],
                               used["posterior"], RAILS[family])
    out.update({"readable": True, "error": None, "rate_used": rate,
                "frontal_channels": used["frontal"],
                "posterior_channels": used["posterior"],
                "missing_channels": used["missing"], **summarize(measures)})
    return _clean(out)


def _work(args: tuple[str, dict, str]) -> dict:
    path, meta, cache = args
    result = ocular_record(path, meta)
    Path(cache).write_text(json.dumps(result), encoding="utf-8")
    return result


def run_ocular(raw_root: str | Path, out_dir: str | Path, workers: int = 4,
               force: bool = False) -> list[dict]:
    """One pass over every EEG record, cached per record under `records/ocular/`."""
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    cache_dir = out_dir / "records" / "ocular"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eeg = inventory.eeg_records(inventory.build_index(raw_root))
    keys = ["record_id", "subject", "task", "level", "task_label", "condition",
            "family", "device", "session", "path"]
    done, todo = [], []
    for row in eeg.to_dict("records"):
        meta = _clean({k: row[k] for k in keys})
        if meta["level"] is not None:
            meta["level"] = int(meta["level"])
        cache = cache_dir / f"{row['record_id']}.json"
        if cache.is_file() and not force:
            done.append(json.loads(cache.read_text(encoding="utf-8")))
        else:
            todo.append((str(raw_root / row["path"]), meta, str(cache)))
    if todo:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                done.extend(pool.map(_work, todo, chunksize=1))
        else:
            done.extend(map(_work, todo))
    return sorted(done, key=lambda r: r["record_id"])


# ------------------------------------------------------------- crosses


def ocular_frame(records: list[dict]) -> pd.DataFrame:
    """One row per record; the channel lists joined as text."""
    frame = pd.DataFrame(records)
    for key in ("frontal_channels", "posterior_channels", "missing_channels"):
        if key in frame:
            frame[key] = frame[key].map(lambda v: ",".join(v) if isinstance(v, list) else v)
    return frame


def cohorts(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each recording condition, plus the three Slackline levels pooled."""
    ok = frame[frame["readable"]]
    out = {c: ok[ok["condition"] == c] for c in
           ("Slackline-L1/CGX", "Slackline-L6/CGX", "Slackline-L11/CGX",
            "Robots/EMOTIV", "Cognifit/EMOTIV")}
    out["Slackline (3 niveles)/CGX"] = ok[ok["family"] == "CGX"]
    return out


def per_subject(rows: pd.DataFrame, measure: str) -> pd.Series:
    """A subject's value in a cohort: the mean over its records there."""
    return rows.groupby("subject")[measure].mean().dropna()


def _p(test, a, b):
    return float(test(a, b).pvalue) if len(a) and len(b) else None


def _rho(x, y):
    if len(x) < 3:
        return None, None
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def label_p_age_adjusted(value: pd.Series, label: pd.Series, age: pd.Series) -> float | None:
    """Label coefficient of `rank(value) ~ 1 + [label == yes] + age`, two-sided.

    The diagnosed children are younger in every cohort, and age moves EEG
    amplitude: this reads the label with age held fixed, in both directions --
    an age effect is not read as a label effect, nor does it hide one.
    """
    y = rankdata(np.asarray(value, dtype=float))
    x = np.column_stack([np.ones(len(y)), (np.asarray(label) == "yes").astype(float),
                         np.asarray(age, dtype=float)])
    dof = len(y) - x.shape[1]
    if dof < 1 or np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    sigma2 = residual @ residual / dof
    se = np.sqrt(sigma2 * np.linalg.inv(x.T @ x)[1, 1])
    return float(2 * student_t.sf(abs(beta[1] / se), dof)) if se > 0 else None


def cross_measure(values: pd.Series, demo: pd.DataFrame) -> dict:
    """One per-subject measure against label (yes vs no), sex and age.

    Undetermined subjects are counted and summarised but enter no test. Sex is
    tested twice: Mann-Whitney on the values, and Fisher on the subjects above
    vs below the cohort median (the median split is a reading aid, not a cut).
    """
    table = demo.set_index("subject").loc[values.index].assign(value=values.astype(float))
    groups = {k: table.loc[table["label"] == k, "value"] for k in ("yes", "no", "undetermined")}
    tested = table[table["label"].isin(["yes", "no"])]
    m, f = tested.loc[tested["sex"] == "M", "value"], tested.loc[tested["sex"] == "F", "value"]
    above = tested["value"] > tested["value"].median()
    grid = [[int((above & (tested["sex"] == s)).sum()), int((~above & (tested["sex"] == s)).sum())]
            for s in ("M", "F")]
    fisher_p = float(fisher_exact(grid)[1]) if min(map(sum, grid)) > 0 else None
    rho, rho_p = _rho(tested["age"], tested["value"])
    within = {k: _rho(tested.loc[tested["label"] == k, "age"],
                      tested.loc[tested["label"] == k, "value"]) for k in ("yes", "no")}
    out = {}
    for k, g in groups.items():
        out[f"n_{k}"] = int(len(g))
        out[f"median_{k}"] = float(g.median()) if len(g) else None
    out.update({
        "label_mannwhitney_p": _p(mannwhitneyu, groups["yes"], groups["no"]),
        "label_p_age_adjusted": label_p_age_adjusted(tested["value"], tested["label"],
                                                     tested["age"]),
        "median_M": float(m.median()) if len(m) else None,
        "median_F": float(f.median()) if len(f) else None,
        "sex_mannwhitney_p": _p(mannwhitneyu, m, f),
        "sex_median_split_fisher_p": fisher_p,
        "age_spearman_rho": rho, "age_spearman_p": rho_p,
        "age_spearman_rho_within_yes": within["yes"][0],
        "age_spearman_p_within_yes": within["yes"][1],
        "age_spearman_rho_within_no": within["no"][0],
        "age_spearman_p_within_no": within["no"][1],
    })
    return out


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjustment; missing p-values stay missing."""
    p = pd.Series(p, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    adjusted = (valid * (m - np.arange(m))).cummax().clip(upper=1.0)
    return adjusted.reindex(p.index)


def cross_table(frame: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    """Every cohort x measure, crossed with label, sex and age; Holm over label tests."""
    rows = []
    for cohort, sub in cohorts(frame).items():
        for measure in MEASURES:
            values = per_subject(sub, measure)
            if values.empty:
                continue
            rows.append({"cohort": cohort, "measure": measure,
                         **cross_measure(values, demo)})
    table = pd.DataFrame(rows)
    table["label_p_holm"] = holm(table["label_mannwhitney_p"])
    table["label_p_age_adjusted_holm"] = holm(table["label_p_age_adjusted"])
    return table


def confounds(frame: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    """Label against age and sex in each cohort, so an age or sex effect is not
    read as a label effect; for Emotiv also label against device model."""
    rows = []
    for cohort, sub in cohorts(frame).items():
        table = demo[demo["subject"].isin(sub["subject"].unique())]
        table = table[table["label"].isin(["yes", "no"])]
        yes, no = table[table["label"] == "yes"], table[table["label"] == "no"]
        sex_grid = [[int((g["sex"] == s).sum()) for s in ("M", "F")] for g in (yes, no)]
        row = {"cohort": cohort, "n_yes": len(yes), "n_no": len(no),
               "median_age_yes": float(yes["age"].median()) if len(yes) else None,
               "median_age_no": float(no["age"].median()) if len(no) else None,
               "age_label_mannwhitney_p": _p(mannwhitneyu, yes["age"], no["age"]),
               "male_fraction_yes": float((yes["sex"] == "M").mean()) if len(yes) else None,
               "male_fraction_no": float((no["sex"] == "M").mean()) if len(no) else None,
               "sex_label_fisher_p": float(fisher_exact(sex_grid)[1]) if len(yes) and len(no) else None}
        if (sub["family"] == "EMOTIV").all() and len(sub):
            device = sub.groupby("subject")["device"].first()
            lab = demo.set_index("subject")["label"]
            dev = pd.DataFrame({"device": device, "label": device.index.map(lab)})
            dev = dev[dev["label"].isin(["yes", "no"])]
            grid = [[int(((dev["label"] == k) & (dev["device"] == d)).sum())
                     for d in ("EPOCX", "EPOCPLUS")] for k in ("yes", "no")]
            row["epocx_fraction_yes"] = grid[0][0] / max(sum(grid[0]), 1)
            row["epocx_fraction_no"] = grid[1][0] / max(sum(grid[1]), 1)
            row["device_label_fisher_p"] = float(fisher_exact(grid)[1])
        rows.append(row)
    return pd.DataFrame(rows)


def task_contrast(frame: pd.DataFrame) -> pd.DataFrame:
    """Emotiv game task (Robots) against the task without game events (Cognifit),
    paired within the subjects that have both: Wilcoxon signed-rank per measure."""
    ok = frame[frame["readable"] & (frame["family"] == "EMOTIV")]
    rows = []
    for measure in MEASURES:
        robots = per_subject(ok[ok["task"] == "Robots"], measure)
        cognifit = per_subject(ok[ok["task"] == "Cognifit"], measure)
        both = robots.index.intersection(cognifit.index)
        rows.append({"measure": measure, "n_paired": int(len(both)),
                     "median_robots": float(robots[both].median()) if len(both) else None,
                     "median_cognifit": float(cognifit[both].median()) if len(both) else None,
                     "wilcoxon_p": (float(wilcoxon(robots[both], cognifit[both]).pvalue)
                                    if len(both) > 1 else None)})
    return pd.DataFrame(rows)
