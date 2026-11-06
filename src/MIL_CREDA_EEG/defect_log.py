"""The defect log: the audit pass and the per-record log of its marks.

Experiments plan v59, Sec. 2.10. The plan's flow table -- initial, rate (2.2),
record without EEG (2.4), contact (2.5), undetermined diagnosis and EPOC+
records (2.10), bag floor (3.2), final -- and its crosses by exclusion type are
produced by `exclusions`, which applies those exclusions. This module is the
audit that feeds them: the defects of the earlier steps, measured once per
record, and the checks of every mark against label, sex and age
(`cross_with_demographics`, which `exclusions` reuses for its crosses). Sex
starts unbalanced in this release -- far more boys among the diagnosed than
among the controls -- so the sex check is not a formality.

Hard defects and incompatibilities are reported apart, each with its own cross
table: a corrupt file is plausibly random; a sound record incompatible with the
working configuration is plausibly systematic, because it tends to come from a
different period or montage. Mixing them ruins the reading.

**What is excluded here: nothing.** `flow_table` is the audit's own mark table,
one row per audit step: it states how many subjects WOULD remain if the hard
defects whose rule the audit reads directly were applied (an unreadable file;
a rate outside the menu or different from the nominal, Sec. 2.2). Every other
row is a mark: a gap longer than a window (2.3), a marked or missing channel
(2.4), a contact window below the maximum (2.5), and a restarted or
concatenated session (2.6, `audit_duration`), which v56 marks and reports and
never excludes. A mark is counted only when its definition needs no number the
plan has not given, and the continuous distributions behind every mark are
crossed with the label as well, so a later cut can be chosen without looking at
accuracy.

This module also drives the single pass over the release: each EEG record is
read once, every audit step runs on it, and its summary is cached as JSON under
`MIL-CREDA-EEG/Results/audit/records/`, so a notebook never re-reads 58 GB. Its
`records.csv` is the per-record table Sec. 2.6 cites and `exclusions` checks
its excluded sets against.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

from MIL_CREDA_EEG import inventory
from MIL_CREDA_EEG.audit_channels import (
    FRONTOPOLAR, OCCIPITAL, RAILS, bipolar_metrics, channel_metrics, mark_channels,
    missing_vs_cohort, reference_metrics,
)
from MIL_CREDA_EEG.audit_contact import emotiv_quality, line_noise_ratio
from MIL_CREDA_EEG.audit_duration import RESTART_MARK, audit_duration, nominal_duration
from MIL_CREDA_EEG.audit_integrity import audit_integrity
from MIL_CREDA_EEG.audit_rate import CGX_MENU, EMOTIV_MENU, audit_rate
from MIL_CREDA_EEG.modality import emotiv_export_streams, read_record

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v59.md",
    "sections": ["2.6", "2.10"],
    "experiments": ["2.6", "2.10"],
    "invariants": [
        "hard_defects_and_incompatibilities_are_kept_apart",
        "flow_table_counts_subjects_left_after_declared_hard_defects",
        "cross_with_label_counts_marked_subjects_per_group",
        "audit_record_of_an_unreadable_file_is_a_hard_defect_not_a_crash",
        "flow_table_counts_the_restart_mark_and_excludes_nothing_for_it",
        "records_frame_carries_the_duration_mark",
    ],
}

#: Steps of the audit's own mark table, in plan order (the v56 flow table of
#: exclusions is `exclusions.STEPS`).
STEPS = ("inicial", "lectura", "2.2 tasa", "2.3 integridad", "2.4 canales",
         "2.5 contacto", "2.6 duración")


def _clean(value):
    """JSON-safe copy: numpy scalars to Python, NaN and inf to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def audit_record(path: str, meta: dict, flags_info: dict) -> dict:
    """Every audit step over one EEG file. A file that cannot be read is a
    hard defect with its reason, never a crash of the whole pass."""
    out = dict(meta)
    try:
        record = read_record(path)
        if len(record.timestamps) < 2 or record.signal.size == 0:
            raise ValueError(f"too few samples: {len(record.timestamps)}")
    except Exception as error:  # noqa: BLE001 -- the reason is the deliverable
        out.update({"readable": False, "error": f"{type(error).__name__}: {error}"})
        return _clean(out)
    menu = EMOTIV_MENU if record.family == "EMOTIV" else CGX_MENU
    rate = audit_rate(record.timestamps, record.counter, record.nominal_rate, menu)
    used = record.nominal_rate or rate["effective_rate"]
    metrics = channel_metrics(record.signal, record.channels, used, RAILS[record.family])
    out.update({
        "readable": True, "error": None, "family": record.family,
        "device": record.device, "channels": record.channels,
        "n_columns": len(record.columns), "n_samples": int(len(record.timestamps)),
        "header": record.header,
        "export_streams": (emotiv_export_streams(path)
                           if record.family == "EMOTIV" else None),
        "rate": rate,
        "integrity": audit_integrity(record.timestamps, record.counter,
                                     record.interpolated, used),
        "channels_audit": metrics,
        "channel_marks": mark_channels(metrics),
        "duration": audit_duration(record.timestamps, used,
                                   nominal_duration(meta.get("task"), meta.get("level"),
                                                    flags_info)),
    })
    if record.family == "EMOTIV":
        out["contact"] = emotiv_quality(record.timestamps, record.quality,
                                        channels=tuple(record.channels))
    else:
        index = {ch: k for k, ch in enumerate(record.channels)}
        front = record.signal[:, [index[c] for c in FRONTOPOLAR if c in index]]
        back = record.signal[:, [index[c] for c in OCCIPITAL if c in index]]
        out["reference"] = (reference_metrics(record.extras["A2"], record.signal, used)
                            if "A2" in record.extras else None)
        out["bipolar"] = {name: bipolar_metrics(record.extras[name], front, back, used)
                          for name in ("ExG 1", "ExG 2") if name in record.extras}
        out["line_noise"] = dict(zip(record.channels,
                                     line_noise_ratio(record.signal, used).tolist()))
    return _clean(out)


def _work(args: tuple[str, dict, dict, str]) -> dict:
    path, meta, flags, cache = args
    result = audit_record(path, meta, flags)
    Path(cache).write_text(json.dumps(result), encoding="utf-8")
    return result


def run_audit(raw_root: str | Path, out_dir: str | Path, workers: int = 4,
              force: bool = False) -> list[dict]:
    """One pass over every EEG record, cached per record; returns the summaries."""
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    eeg = inventory.eeg_records(inventory.build_index(raw_root))
    flags_path = raw_root / "slackline_flags_info.json"
    flags = json.loads(flags_path.read_text(encoding="utf-8")) if flags_path.is_file() else {}
    keys = ["record_id", "subject", "task", "level", "task_label", "condition",
            "family", "device", "session", "path", "size_bytes"]
    done, todo = [], []
    for row in eeg.to_dict("records"):
        meta = _clean({k: row[k] for k in keys})
        if meta["level"] is not None:
            meta["level"] = int(meta["level"])
        cache = records_dir / f"{row['record_id']}.json"
        if cache.is_file() and not force:
            done.append(json.loads(cache.read_text(encoding="utf-8")))
        else:
            todo.append((str(raw_root / row["path"]), meta, flags, str(cache)))
    if todo:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                done.extend(pool.map(_work, todo, chunksize=1))
        else:
            done.extend(map(_work, todo))
    return sorted(done, key=lambda r: r["record_id"])


# ------------------------------------------------------------- tables


def records_frame(records: list[dict]) -> pd.DataFrame:
    """One row per record with the scalars every table and figure reads."""
    readable = [r for r in records if r.get("readable")]
    missing = {}
    for family in {r["family"] for r in readable}:
        missing.update(missing_vs_cohort({r["record_id"]: r["channels"]
                                          for r in readable if r["family"] == family}))
    rows = []
    for r in records:
        row = {k: r.get(k) for k in ("record_id", "subject", "task", "level",
                                     "task_label", "condition", "family", "device",
                                     "size_bytes", "readable", "error")}
        if r.get("readable"):
            rate, integ, dur = r["rate"], r["integrity"], r["duration"]
            marks = r["channel_marks"]
            row.update({
                "n_samples": r["n_samples"], "n_channels": len(r["channels"]),
                "nominal_rate": rate["nominal_rate"],
                "effective_rate": rate["effective_rate"],
                "counter_corrected_rate": rate["counter_corrected_rate"],
                "median_step_rate": rate.get("median_step_rate"),
                "relative_deviation": rate["relative_deviation"],
                "nearest_menu_rate": rate["nearest_menu_rate"],
                "counter_contiguous": rate["counter_contiguous"],
                "rate_hard_defect": rate["hard_defect"],
                "rate_ratio_class": rate["ratio_class"],
                "half_of_nominal": rate["half_of_nominal"],
                "lost_fraction": integ["lost_fraction"], "n_lost": integ["n_lost"],
                "interpolated_fraction": integ["interpolated_fraction"],
                "longest_gap_s": integ["longest_gap_s"],
                "gap_exceeds_window": integ["gap_exceeds_window"],
                "windows_touched_by_loss": integ["windows_touched_by_loss"],
                "windows_touched_by_interpolation":
                    integ["windows_touched_by_interpolation"],
                "n_windows": dur["n_windows"], "duration_s": dur["duration_s"],
                "nominal_s": dur["nominal_s"], "ratio_to_nominal": dur["ratio_to_nominal"],
                "duration_marks": ";".join(dur.get("marks", [])),
                "restarted_or_concatenated": RESTART_MARK in dur.get("marks", []),
                "n_flat_channels": sum("flat" in m for m in marks.values()),
                "n_saturated_channels": sum("saturated" in m for m in marks.values()),
                "n_marked_channels": sum(bool(m) for m in marks.values()),
                "n_missing_vs_cohort": len(missing.get(r["record_id"], [])),
                "missing_vs_cohort": ";".join(missing.get(r["record_id"], [])),
                "max_abs_log_std_z": _max_abs(r["channels_audit"], "log_std_robust_z"),
                "min_high_band_z": _min(r["channels_audit"], "high_band_robust_z"),
            })
            contact = r.get("contact") or {}
            eq = contact.get("EQ.OVERALL") or {}
            cq = contact.get("CQ.Overall") or {}
            row.update({
                "eq_overall_median": eq.get("median"),
                "eq_window_fraction_below": eq.get("window_fraction_below"),
                "cq_overall_median": cq.get("median"),
                "cq_window_fraction_below": cq.get("window_fraction_below"),
                "quality_any_window_below": (bool(eq["window_fraction_below"])
                                             if eq.get("window_fraction_below")
                                             is not None else None),
            })
            if r["family"] == "CGX":
                ref = r.get("reference") or {}
                ln = [v for v in (r.get("line_noise") or {}).values() if v is not None]
                row.update({
                    "a2_raw_std": ref.get("raw_std"),
                    "a2_raw_median_abs": ref.get("raw_median_abs"),
                    "a2_band_std_ratio": ref.get("band_std_ratio_to_scalp"),
                    "a2_band_corr_scalp_mean": ref.get("band_corr_with_scalp_mean"),
                    "line_noise_median": float(np.median(ln)) if ln else None,
                })
                for name, b in (r.get("bipolar") or {}).items():
                    key = name.replace(" ", "").lower()
                    row[f"{key}_constant"] = b["constant"]
                    row[f"{key}_at_rail_fraction"] = b["at_rail_fraction"]
                    row[f"{key}_front_minus_back_corr"] = b["frontal_minus_occipital_corr"]
        rows.append(row)
    return pd.DataFrame(rows)


def _max_abs(metrics: dict, key: str):
    values = [abs(m[key]) for m in metrics.values() if m.get(key) is not None]
    return max(values) if values else None


def _min(metrics: dict, key: str):
    values = [m[key] for m in metrics.values() if m.get(key) is not None]
    return min(values) if values else None


def channels_frame(records: list[dict]) -> pd.DataFrame:
    """One row per record and channel: the per-channel metrics and marks."""
    rows = []
    for r in records:
        if not r.get("readable"):
            continue
        for ch, m in r["channels_audit"].items():
            rows.append({"record_id": r["record_id"], "subject": r["subject"],
                         "family": r["family"], "device": r["device"],
                         "condition": r["condition"], "channel": ch, **m,
                         "marks": ";".join(r["channel_marks"][ch]),
                         "line_noise": (r.get("line_noise") or {}).get(ch)})
    return pd.DataFrame(rows)


def _truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def classify_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `hard_defects` and `incompatibilities`: two lists, never merged."""
    frame = frame.copy()
    unreadable = ~_truthy(frame["readable"])
    rate = _truthy(frame["rate_hard_defect"])
    missing = frame["n_missing_vs_cohort"].fillna(0) > 0
    frame["hard_defects"] = [
        (["unreadable"] if u else []) + (["rate_hard_defect"] if r else [])
        for u, r in zip(unreadable, rate)]
    frame["incompatibilities"] = [["channel_missing_vs_cohort"] if m else []
                                  for m in missing]
    return frame


def step_marks(frame: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Per step: (records marked, records with a declared hard defect)."""
    none = pd.Series(False, index=frame.index)
    unreadable = ~_truthy(frame["readable"])
    rate = _truthy(frame["rate_hard_defect"])
    channels = (frame["n_marked_channels"].fillna(0) > 0) | \
        (frame["n_missing_vs_cohort"].fillna(0) > 0)
    contact = (_truthy(frame["quality_any_window_below"])
               if "quality_any_window_below" in frame else none)
    restart = (_truthy(frame["restarted_or_concatenated"])
               if "restarted_or_concatenated" in frame else none)
    return {"lectura": (unreadable, unreadable), "2.2 tasa": (rate, rate),
            "2.3 integridad": (_truthy(frame["gap_exceeds_window"]), none),
            "2.4 canales": (channels, none), "2.5 contacto": (contact, none),
            "2.6 duración": (restart, none)}


def flow_table(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    """Flow of one cohort: marks per step and subjects left under hard defects."""
    frame = frame[frame["family"] == family].reset_index(drop=True)
    rows = [{"step": "inicial", "records": len(frame), "records_marked": 0,
             "subjects_marked": 0, "records_hard_defect": 0,
             "subjects_left": frame["subject"].nunique()}]
    dropped = pd.Series(False, index=frame.index)
    for step, (marked, hard) in step_marks(frame).items():
        dropped = dropped | hard
        rows.append({"step": step, "records": len(frame),
                     "records_marked": int(marked.sum()),
                     "subjects_marked": int(frame.loc[marked, "subject"].nunique()),
                     "records_hard_defect": int(hard.sum()),
                     "subjects_left": int(frame.loc[~dropped, "subject"].nunique())})
    return pd.DataFrame(rows)


def cross_with_demographics(marked: pd.Series, demo: pd.DataFrame) -> dict:
    """Marked vs unmarked subjects by label, sex and age, with exact tests."""
    table = demo.set_index("subject").loc[marked.index].assign(marked=marked.astype(bool))

    def counts(by: str) -> pd.DataFrame:
        grouped = table.groupby(by)["marked"].agg(marked="sum", total="count")
        grouped["unmarked"] = grouped["total"] - grouped["marked"]
        grouped["fraction_marked"] = grouped["marked"] / grouped["total"]
        return grouped.reset_index()

    def fisher(rows: pd.DataFrame, a: str, b: str, by: str):
        rows = rows.set_index(by)
        if a not in rows.index or b not in rows.index:
            return None
        grid = [[int(rows.loc[a, "marked"]), int(rows.loc[a, "unmarked"])],
                [int(rows.loc[b, "marked"]), int(rows.loc[b, "unmarked"])]]
        return float(fisher_exact(grid)[1])

    label, sex = counts("label"), counts("sex")
    ages_m, ages_u = table.loc[table["marked"], "age"], table.loc[~table["marked"], "age"]
    age_p = (float(mannwhitneyu(ages_m, ages_u).pvalue)
             if len(ages_m) and len(ages_u) else None)
    return {
        "label": label, "sex": sex,
        "age": pd.DataFrame({"group": ["marked", "unmarked"],
                             "median_age": [ages_m.median(), ages_u.median()],
                             "n": [len(ages_m), len(ages_u)]}),
        "tests": {"label_yes_vs_no_fisher_p": fisher(label, "yes", "no", "label"),
                  "sex_fisher_p": fisher(sex, "M", "F", "sex"),
                  "age_mannwhitney_p": age_p},
    }


def continuous_cross(values: pd.Series, demo: pd.DataFrame) -> dict:
    """A per-subject measurement against label (yes/no), sex and age."""
    table = demo.set_index("subject").loc[values.index].assign(value=values.astype(float))
    table = table[np.isfinite(table["value"])]
    yes, no = table.loc[table["label"] == "yes", "value"], table.loc[table["label"] == "no", "value"]
    m, f = table.loc[table["sex"] == "M", "value"], table.loc[table["sex"] == "F", "value"]
    rho = spearmanr(table["age"], table["value"]) if len(table) > 2 else None

    def mw(a, b):
        return float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else None

    return {"n": int(len(table)),
            "median_yes": float(yes.median()) if len(yes) else None,
            "median_no": float(no.median()) if len(no) else None,
            "label_mannwhitney_p": mw(yes, no),
            "median_M": float(m.median()) if len(m) else None,
            "median_F": float(f.median()) if len(f) else None,
            "sex_mannwhitney_p": mw(m, f),
            "age_spearman_rho": float(rho.statistic) if rho is not None else None,
            "age_spearman_p": float(rho.pvalue) if rho is not None else None}


def subject_flags(frame: pd.DataFrame, family: str, mask: pd.Series) -> pd.Series:
    """A subject is marked when any of its records in this cohort is."""
    rows = frame["family"] == family
    return mask[rows].groupby(frame.loc[rows, "subject"]).any()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BALLADEER audit, one pass.")
    parser.add_argument("raw_root")
    parser.add_argument("out_dir")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    records = run_audit(args.raw_root, args.out_dir, args.workers, args.force)
    out = Path(args.out_dir)
    records_frame(records).to_csv(out / "records.csv", index=False)
    channels_frame(records).to_csv(out / "channels.csv", index=False)
    print(f"{len(records)} records audited into {out}")


if __name__ == "__main__":
    main()
