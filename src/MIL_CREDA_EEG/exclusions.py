"""What preprocessing excludes, and the flow log that keeps it honest.

Experiments plan v55, Secs. 2.2, 2.4, 2.5, 2.10 and 3.2. Three rules drop a
record as a hard defect of the record, each with its reason:

* **2.2 rate**: an effective rate outside the manufacturer's menu or different
  from the nominal one -- `audit_rate`'s own classification;
* **2.4 a record without EEG**: more than half of its scalp channels flat or
  saturated in every one of its windows. The plan names the one record this
  is (UB0058, Slackline level 6, 25 of 29 channels); the rule is stated so the
  record is derived and not typed in, and `check_against_audit` proves the
  derivation selects that record and no other;
* **2.5 contact**: the operator-approved majority rule,
  `audit_contact.contact_rule`, unchanged.

Then two exclusions by design (Sec. 2.10), each its own step and its own kind:

* **2.10 undetermined**: every record of a subject whose diagnosis is
  undetermined (the reason is Sec. 4.3's);
* **2.10 EPOC+**: every record of the EPOC+ model of the low-cost device. The
  subject keeps its high-density (CGX) records and, if it has them, its EPOC X
  ones -- the rule reads the record's device, never the subject.

And the bag floor (Sec. 3.2): a record whose useful windows, already without
the first and the partial last, fall below `windows.BAG_FLOOR` leaves. On the
published release it removes nothing; it stays as a guard.

Nothing else is excluded. A defective channel stays (Sec. 2.4) and a marked
window stays (Secs. 2.3, 2.5).

The flow table (Sec. 2.10) runs per device cohort from the initial records and
subjects to the final ones, step by step in plan order; a record meeting two
rules leaves at the first. Each kind -- hard defects, incompatibilities, each
design exclusion and the floor -- is crossed with label, sex and age apart, in
its own rows. No rule here is an incompatibility, so that table is written and
is empty: stated, not omitted.

This module also drives the single preprocessing pass: each record is read
once, its exclusion decided, and when kept its windows are written to
`MIL-CREDA/Data/processed/signals/<record_id>.npz` beside a JSON summary under
`records/`, so a rerun only computes what is missing.

The npz holds `signal`, `window_index`, `window_start_s`, every per-window
side array of `windows.WINDOW_ARRAYS` the record has, and the record-level
`channels`, `record_id`, `subject`, `condition`, `device`, `family`, `rate`,
`target_rate`, `gap_exceeds_window` and `contact_meets_rule`. A Slackline
record also holds `event_window` (plan v42, Sec. 6.1): its session's sibling
`*_TAGS_*.csv` is read and every K1 event instant marks the kept window that
contains it. A Slackline record whose TAGS file is missing, ambiguous or
unparseable gets no `event_window` -- never an all-False one -- and one entry
in the defect log, the manifest's `event_defect` column. It is a defect of the
mark, never an exclusion: the record and the partition are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from MIL_CREDA_EEG import inventory, k1_events
from MIL_CREDA_EEG.audit_channels import RAILS
from MIL_CREDA_EEG.audit_contact import contact_rule
from MIL_CREDA_EEG.audit_rate import CGX_MENU, EMOTIV_MENU, audit_rate
from MIL_CREDA_EEG.defect_log import _clean, cross_with_demographics
from MIL_CREDA_EEG.modality import read_columns, read_record
from MIL_CREDA_EEG.preprocess_filter import scalp_signal
from MIL_CREDA_EEG.windows import (
    BAG_FLOOR, EVENT_MARK, WINDOW_S, channel_marks, complete_windows, emotiv_cq, record_windows,
)

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md",
    "sections": ["2.2", "2.4", "2.5", "2.7", "2.10", "3.2", "6.1"],
    "experiments": ["2.2", "2.4", "2.5", "2.10", "3.2", "K1-atencion"],
    "invariants": [
        "a_record_with_most_channels_dead_in_every_window_has_no_eeg",
        "a_channel_dead_in_all_but_one_window_is_not_dead",
        "a_minority_of_dead_channels_is_a_channel_defect_not_a_record_without_eeg",
        "reasons_follow_plan_order_and_only_the_three_declared_rules_exclude",
        "the_excluded_set_is_exactly_the_records_with_a_reason",
        "undetermined_subjects_leave_with_all_their_records",
        "epocplus_records_leave_and_their_subject_keeps_cgx_and_epocx",
        "a_bag_below_the_floor_leaves_at_its_own_row",
        "the_floor_excludes_no_bag_of_the_processed_data",
        "flow_table_counts_records_and_subjects_left_after_each_step",
        "each_kind_of_exclusion_is_crossed_apart",
        "a_slackline_record_writes_its_event_window_and_counts_its_events",
        "a_slackline_record_without_a_readable_tags_file_gets_no_mark_and_one_defect",
        "another_condition_gets_no_event_key_and_no_defect",
        "every_existing_npz_member_is_byte_for_byte_unchanged",
        "the_partition_inputs_do_not_change",
    ],
}

#: Reason -> flow step, in plan order. `lectura` guards an unreadable file.
REASON_STEP = {
    "lectura unreadable": "lectura",
    "2.2 rate_hard_defect": "2.2 tasa",
    "2.4 no_eeg": "2.4 registro sin EEG",
    "2.5 majority_bad_contact": "2.5 contacto",
    "2.10 undetermined_diagnosis": "2.10 indeterminado",
    "2.10 epocplus_record": "2.10 EPOC+",
    "3.2 bag_floor": "3.2 piso de bolsa",
}
STEPS = tuple(dict.fromkeys(REASON_STEP.values()))
#: The defects are of the record itself; the design exclusions and the floor
#: each keep their own kind, so no cross mixes them. None is an incompatibility.
REASON_KIND = {
    "lectura unreadable": "hard_defect",
    "2.2 rate_hard_defect": "hard_defect",
    "2.4 no_eeg": "hard_defect",
    "2.5 majority_bad_contact": "hard_defect",
    "2.10 undetermined_diagnosis": "design_undetermined",
    "2.10 epocplus_record": "design_epocplus",
    "3.2 bag_floor": "bag_floor",
}
KINDS = ("hard_defect", "incompatibility", "design_undetermined", "design_epocplus",
         "bag_floor")

#: Sec. 2.10: the low-cost model whose records leave by design.
EXCLUDED_DEVICE = "EPOCPLUS"

#: Sec. 2.4: a record without EEG has MORE than half its channels dead.
NO_EEG_MAJORITY = 0.5

#: The proprietary focus score kept aside as audit metadata (Sec. 2.1). The
#: release writes it under two names: 82 files carry `PM.Focus.*` and 180 carry
#: `PM.Attention.*`, with the same five sub-columns. Read as the same metric
#: under a later EmotivPRO name -- an assumption, so the column actually read
#: is recorded per record in the manifest.
FOCUS_COLUMNS = ("PM.Focus.Scaled", "PM.Attention.Scaled")

#: Plan v42, Sec. 6.1: only the Slackline levels log K1 events with an instant.
EVENT_TASK = "Slackline"
TAGS_GLOB = "*_TAGS_*.csv"
#: Per-record event fields of the summary and the manifest; None when not marked.
EVENT_FIELDS = ("event_tags", "event_defect", "events_hit", "events_error",
                "events_correct_no_press", "events_outside_kept", "windows_event_marked")


# ------------------------------------------------------------- rules


def dead_channels(flat: np.ndarray, saturated: np.ndarray) -> int:
    """Channels flat or saturated in every one of the record's windows."""
    dead = np.asarray(flat, dtype=bool) | np.asarray(saturated, dtype=bool)
    if dead.shape[0] == 0:
        return 0
    return int(dead.all(axis=0).sum())


def record_without_eeg(n_dead: int, n_channels: int) -> bool:
    return bool(n_dead > n_channels * NO_EEG_MAJORITY)


def exclusion_reasons(rate_hard_defect: bool, n_dead: int, n_channels: int,
                      contact_meets_rule: bool | None) -> list[str]:
    """Every declared rule the record meets, in plan order."""
    reasons = []
    if rate_hard_defect:
        reasons.append("2.2 rate_hard_defect")
    if record_without_eeg(n_dead, n_channels):
        reasons.append("2.4 no_eeg")
    if contact_meets_rule:
        reasons.append("2.5 majority_bad_contact")
    return reasons


def design_reasons(undetermined: bool, device: str) -> list[str]:
    """Sec. 2.10's two exclusions by design the record meets, in plan order."""
    reasons = []
    if undetermined:
        reasons.append("2.10 undetermined_diagnosis")
    if device == EXCLUDED_DEVICE:
        reasons.append("2.10 epocplus_record")
    return reasons


def bag_floor_reasons(n_windows: int) -> list[str]:
    """Sec. 3.2: a bag with fewer useful windows than the floor leaves."""
    return ["3.2 bag_floor"] if n_windows < BAG_FLOOR else []


# ------------------------------------------------------------- one record


def read_focus(path: str) -> tuple[np.ndarray | None, str | None]:
    """The per-sample focus score of an Emotiv file, and the column it came from."""
    columns = read_columns(path)
    column = next((c for c in FOCUS_COLUMNS if c in columns), None)
    if column is None:
        return None, None
    frame = pd.read_csv(path, skiprows=1, usecols=[column], low_memory=False)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float), column


def _summary_counts(out: dict) -> dict:
    def count(key, reduce_any=False):
        if key not in out:
            return None
        value = np.asarray(out[key])
        return int(value.any(axis=1).sum() if reduce_any else value.sum())

    def median(key):
        if key not in out:
            return None
        v = np.asarray(out[key], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else None

    return {"windows_integrity_marked": count("integrity_mark"),
            "windows_long_gap": count("long_gap"),
            "windows_any_flat": count("flat", True),
            "windows_any_saturated": count("saturated", True),
            "windows_any_contact_bad": count("contact_bad", True),
            "windows_rail_ocular": count("rail_window"),
            "frontal_ptp_median": median("frontal_ptp"),
            "low_ratio_median": median("low_ratio"),
            "focus_windows_with_reading": (int(np.isfinite(out["focus_median"]).sum())
                                           if "focus_median" in out else None)}


def session_events(path: str) -> tuple[list[dict] | None, str | None, str | None]:
    """The logged rows of the EEG file's sibling TAGS file, its name, and the defect.

    Exactly one `*_TAGS_*.csv` beside the EEG file is read; none, more than
    one, or one `k1_events.parse_tags` refuses returns no rows and the defect.
    """
    found = sorted(Path(path).parent.glob(TAGS_GLOB))
    if not found:
        return None, None, "tags_missing"
    if len(found) > 1:
        return None, None, "tags_ambiguous: " + ", ".join(p.name for p in found)
    try:
        rows = k1_events.parse_tags(found[0].read_text(encoding="utf-8-sig"))
        k1_events.event_instants(rows)
    except (k1_events.K1EventError, UnicodeDecodeError, csv.Error) as error:
        return None, found[0].name, f"tags_unparseable: {error}"
    return rows, found[0].name, None


def preprocess_record(path: str, meta: dict, signals_dir: str) -> dict:
    """Read once, decide the exclusion, and when kept write the windows."""
    out = dict(meta)
    try:
        record = read_record(path)
        if len(record.timestamps) < 2 or record.signal.size == 0:
            raise ValueError(f"too few samples: {len(record.timestamps)}")
        scalp, channels = scalp_signal(record)
    except Exception as error:  # noqa: BLE001 -- the reason is the deliverable
        out.update({"readable": False, "error": f"{type(error).__name__}: {error}",
                    "reasons": ["lectura unreadable"]})
        return _clean(out)
    menu = EMOTIV_MENU if record.family == "EMOTIV" else CGX_MENU
    rate_report = audit_rate(record.timestamps, record.counter, record.nominal_rate, menu)
    rate = float(record.nominal_rate)
    n_windows = complete_windows(len(scalp), rate)
    marks = channel_marks(scalp, rate, RAILS[record.family], n_windows)
    n_dead = dead_channels(marks["flat"], marks["saturated"])
    cq = emotiv_cq(record, channels) if record.family == "EMOTIV" else None
    rule = contact_rule(cq, rate) if cq is not None else None
    reasons = exclusion_reasons(rate_report["hard_defect"], n_dead, len(channels),
                                rule["meets_rule"] if rule else None)
    reasons += design_reasons(bool(meta.get("undetermined")), meta["device"])
    out.update({
        "readable": True, "error": None, "nominal_rate": rate,
        "effective_rate": rate_report["effective_rate"],
        "rate_hard_defect_reasons": rate_report["hard_defect_reasons"],
        "n_samples_raw": int(len(scalp)), "n_channels": len(channels),
        "n_windows_total": int(n_windows), "n_dead_channels": n_dead,
        "contact_n_bad_channels": rule["n_bad_channels"] if rule else None,
        "contact_meets_rule": rule["meets_rule"] if rule else None,
        "reasons": reasons, "npz": None, "npz_bytes": None,
        "n_windows": 0, "bag_windows": None, **dict.fromkeys(EVENT_FIELDS),
    })
    if reasons:
        return _clean(out)
    focus, focus_column = read_focus(path) if record.family == "EMOTIV" else (None, None)
    rows, instants = None, None
    if meta.get("task") == EVENT_TASK:
        rows, out["event_tags"], out["event_defect"] = session_events(path)
        instants = k1_events.event_instants(rows) if rows is not None else None
    windows = record_windows(record, cq=cq, focus=focus, events=instants)
    out["bag_windows"] = windows["n_windows"]
    floor = bag_floor_reasons(windows["n_windows"])
    if floor:
        out["reasons"] = floor
        return _clean(out)
    target = Path(signals_dir) / f"{meta['record_id']}.npz"
    arrays = {key: value for key, value in windows.items()
              if isinstance(value, np.ndarray)}
    np.savez(target, **arrays, channels=np.array(windows["channels"]),
             record_id=np.array(meta["record_id"]), subject=np.array(meta["subject"]),
             condition=np.array(meta["condition"]), device=np.array(meta["device"]),
             family=np.array(meta["family"]), rate=np.array(windows["rate"]),
             target_rate=np.array(windows["target_rate"]),
             gap_exceeds_window=np.array(windows["gap_exceeds_window"]),
             contact_meets_rule=np.array(bool(windows.get("contact_meets_rule", False))))
    out.update({"n_windows": windows["n_windows"], "focus_column": focus_column,
                "npz": f"signals/{target.name}", "npz_bytes": target.stat().st_size,
                "gap_exceeds_window": windows["gap_exceeds_window"],
                **_summary_counts(windows)})
    if EVENT_MARK in windows:
        counts = k1_events.event_counts(rows)
        out.update({"events_hit": counts["hit"], "events_error": counts["error"],
                    "events_correct_no_press": counts["correct no-press"],
                    "events_outside_kept": k1_events.instants_outside(
                        instants, windows["window_start_s"], WINDOW_S),
                    "windows_event_marked": int(windows[EVENT_MARK].sum())})
    return _clean(out)


def _work(args: tuple[str, dict, str, str]) -> dict:
    path, meta, signals_dir, cache = args
    result = preprocess_record(path, meta, signals_dir)
    Path(cache).write_text(json.dumps(result), encoding="utf-8")
    return result


def _cached(cache: Path, out_dir: Path, meta: dict) -> dict | None:
    if not cache.is_file():
        return None
    summary = json.loads(cache.read_text(encoding="utf-8"))
    if summary.get("npz") and not (out_dir / summary["npz"]).is_file():
        return None
    if any(summary.get(key) != value for key, value in meta.items()):
        return None         # decided under another label or another floor
    if summary.get("readable") and "event_defect" not in summary:
        return None         # written before the event mark existed (plan v42, Sec. 6.1)
    return summary


def run_preprocess(raw_root: str | Path, out_dir: str | Path, workers: int = 4,
                   force: bool = False, record_ids: list[str] | None = None) -> list[dict]:
    """One pass over every EEG record, cached per record; returns the summaries."""
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    records_dir, signals_dir = out_dir / "records", out_dir / "signals"
    records_dir.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(parents=True, exist_ok=True)
    eeg = inventory.eeg_records(inventory.build_index(raw_root))
    demo = inventory.load_demographics(raw_root)
    undetermined = set(demo.loc[demo["label"] == "undetermined", "subject"])
    if record_ids is not None:
        eeg = eeg[eeg["record_id"].isin(record_ids)]
    keys = ["record_id", "subject", "task", "level", "task_label", "condition",
            "family", "device", "session", "path"]
    done, todo = [], []
    for row in eeg.to_dict("records"):
        meta = _clean({k: row[k] for k in keys})
        if meta["level"] is not None:
            meta["level"] = int(meta["level"])
        meta["undetermined"] = meta["subject"] in undetermined
        meta["bag_floor"] = BAG_FLOOR
        cache = records_dir / f"{row['record_id']}.json"
        summary = None if force else _cached(cache, out_dir, meta)
        if summary is not None:
            done.append(summary)
        else:
            todo.append((str(raw_root / row["path"]), meta, str(signals_dir), str(cache)))
    if todo:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                done.extend(pool.map(_work, todo, chunksize=1))
        else:
            done.extend(map(_work, todo))
    return sorted(done, key=lambda r: r["record_id"])


# ------------------------------------------------------------- manifest and flow


def finalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Add excluded, reason, first step, kind and the undetermined flag."""
    frame = frame.copy()
    reasons = frame["reasons"].map(lambda r: list(r) if isinstance(r, (list, tuple)) else [])
    frame["excluded"] = reasons.map(bool)
    frame["exclusion_reason"] = reasons.map(";".join)
    frame["exclusion_step"] = reasons.map(lambda r: REASON_STEP[r[0]] if r else "")
    frame["exclusion_kind"] = reasons.map(lambda r: REASON_KIND[r[0]] if r else "")
    frame["undetermined"] = frame["label"] == "undetermined"
    return frame


MANIFEST_COLUMNS = (
    "record_id", "subject", "condition", "task_label", "family", "device", "label",
    "sex", "age", "undetermined", "nominal_rate", "effective_rate", "n_samples_raw",
    "n_channels", "n_windows_total", "n_windows", "bag_windows", "excluded", "exclusion_step",
    "exclusion_reason", "exclusion_kind", "n_dead_channels", "contact_n_bad_channels",
    "contact_meets_rule", "gap_exceeds_window", "windows_integrity_marked",
    "windows_long_gap", "windows_any_flat", "windows_any_saturated",
    "windows_any_contact_bad", "windows_rail_ocular", "frontal_ptp_median",
    "low_ratio_median", "focus_column", "focus_windows_with_reading", *EVENT_FIELDS,
    "npz", "npz_bytes", "error",
)


def manifest_frame(summaries: list[dict], demo: pd.DataFrame) -> pd.DataFrame:
    """One row per record, joined with the label, sex and age of its subject."""
    frame = pd.DataFrame(summaries).merge(demo[["subject", "label", "sex", "age"]],
                                          on="subject", how="left")
    frame = finalize_manifest(frame)
    return frame[[c for c in MANIFEST_COLUMNS if c in frame.columns]]


def flow_table(manifest: pd.DataFrame, family: str) -> pd.DataFrame:
    """Sec. 2.10 flow of one cohort: records and subjects left after each step."""
    frame = manifest[manifest["family"] == family].reset_index(drop=True)
    reasons = frame["exclusion_reason"].fillna("").map(lambda s: s.split(";") if s else [])
    left = pd.Series(True, index=frame.index)

    def row(step: str, meeting: int, excluded: int) -> dict:
        subjects = frame.loc[left, "subject"]
        return {"step": step, "records_meeting_rule": meeting,
                "records_excluded": excluded, "records_left": int(left.sum()),
                "subjects_left": int(subjects.nunique()),
                "undetermined_subjects_flagged": int(
                    frame.loc[left & frame["undetermined"].astype(bool), "subject"].nunique())}

    rows = [row("inicial", 0, 0)]
    for step in STEPS:
        meets = reasons.map(lambda r: any(REASON_STEP[x] == step for x in r))
        newly = left & (frame["exclusion_step"] == step)
        left = left & ~newly
        rows.append(row(step, int(meets.sum()), int(newly.sum())))
    rows.append(row("final", 0, int((~left).sum())))
    return pd.DataFrame(rows)


def exclusion_crosses(manifest: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    """Per cohort and kind, excluded subjects by label, sex and age, with exact tests.

    A subject counts as excluded in a cohort when any of its records there left
    for a reason of that kind. One long table; `kind` keeps the two apart.
    """
    rows = []
    for kind in KINDS:
        for family in sorted(manifest["family"].unique()):
            frame = manifest[manifest["family"] == family]
            hit = frame["exclusion_kind"] == kind
            marked = hit.groupby(frame["subject"]).any()
            cross = cross_with_demographics(marked, demo)
            tests = cross["tests"]
            for by in ("label", "sex"):
                for r in cross[by].to_dict("records"):
                    rows.append({"kind": kind, "family": family, "by": by,
                                 "value": r[by], "excluded": int(r["marked"]),
                                 "total": int(r["total"]),
                                 "fraction_excluded": float(r["fraction_marked"]),
                                 "median_age": None, **tests})
            for r in cross["age"].to_dict("records"):
                rows.append({"kind": kind, "family": family, "by": "age",
                             "value": "excluded" if r["group"] == "marked" else "kept",
                             "excluded": None, "total": int(r["n"]),
                             "fraction_excluded": None,
                             "median_age": (float(r["median_age"])
                                            if pd.notna(r["median_age"]) else None),
                             **tests})
    return pd.DataFrame(rows)


def check_against_audit(manifest: pd.DataFrame, audit_dir: str | Path) -> dict:
    """The excluded sets against what the audit itself found, rule by rule.

    Every value is the symmetric difference; all empty means each rule
    selected exactly the records the audit and the plan name.
    """
    audit_dir = Path(audit_dir)
    records = pd.read_csv(audit_dir / "records.csv")
    rule = pd.read_csv(audit_dir / "contact_rule.csv")
    channels = pd.read_csv(audit_dir / "channels.csv")
    dead = (channels["flat_window_fraction"] == 1) | (channels["saturated_window_fraction"] == 1)
    per_record = dead.groupby(channels["record_id"]).sum()
    size = channels.groupby("record_id").size()
    no_eeg = set(per_record[per_record > size * NO_EEG_MAJORITY].index)

    def ids(reason: str) -> set:
        hit = manifest["exclusion_reason"].fillna("").str.contains(reason, regex=False)
        return set(manifest.loc[hit, "record_id"])

    expected = {
        "2.2 rate_hard_defect": set(records.loc[records["rate_hard_defect"] == True,  # noqa: E712
                                                "record_id"]),
        "2.4 no_eeg": no_eeg,
        "2.5 majority_bad_contact": set(rule.loc[rule["meets_rule"] == True,  # noqa: E712
                                                 "record_id"]),
    }
    return {reason: sorted(ids(reason) ^ want) for reason, want in expected.items()}


def write_outputs(summaries: list[dict], demo: pd.DataFrame, out_dir: str | Path) -> dict:
    """manifest.csv, one flow table per cohort, and the exclusion crosses."""
    out_dir = Path(out_dir)
    manifest = manifest_frame(summaries, demo)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    flows = {family: flow_table(manifest, family) for family in ("CGX", "EMOTIV")}
    for family, flow in flows.items():
        flow.to_csv(out_dir / f"flow_{family}.csv", index=False)
    crosses = exclusion_crosses(manifest, demo)
    crosses.to_csv(out_dir / "exclusion_crosses.csv", index=False)
    return {"manifest": manifest, "flows": flows, "crosses": crosses}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BALLADEER preprocessing, one pass.")
    parser.add_argument("raw_root")
    parser.add_argument("out_dir")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summaries = run_preprocess(args.raw_root, args.out_dir, args.workers, args.force)
    demo = inventory.load_demographics(args.raw_root)
    outputs = write_outputs(summaries, demo, args.out_dir)
    manifest = outputs["manifest"]
    print(f"{len(manifest)} records, {int(manifest['excluded'].sum())} excluded, "
          f"written into {args.out_dir}")


if __name__ == "__main__":
    main()
