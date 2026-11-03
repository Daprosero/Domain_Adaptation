"""Experiments plan v42, Sec. 6.1: preprocessing writes K1's `event_window` mark.

The EEG of a Slackline level starts at the level start (data paper: «Level startup
coincides with the start of the backend session. Events are linked to the EEG by
converting the elapsed seconds into sample indices»), so an event at `generalTime`
seconds sits in raw window `floor(generalTime / 4)` and in kept window
`floor(generalTime / 4) - 1`, the first window being the discarded filter transient
(Sec. 2.7). Every record here is synthetic, written with a fixed seed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from MIL_CREDA_EEG import bag_data, exclusions, k1_events, mechanisms, partition, windows
from MIL_CREDA_EEG.modality import read_record
from test_balladeer_preprocess_entry import _cgx_file, _emotiv_file

SECONDS = 284                       # 71 complete windows, 70 kept
CGX_NAME = "UB0001_EEG_CGX_2023_01_01T00.00.00.csv"
TAGS_NAME = "UB0001_TAGS_2023_01_01T00.00.00.csv"


def _entry(reacted, correct, t, duplicated="False"):
    return {"reacted": reacted, "reactionTime": 0.8, "correct": correct,
            "duplicated": duplicated, "flagType": [0], "generalTime": t, "focus": "Target"}


#: generalTime -> raw window floor(t / 4): 0.5 -> 0 (discarded), 4.0 -> 1, 7.99 -> 1,
#: 9.0 -> 2, 9.5 duplicated (dropped), 30.0 -> 7, 283.0 -> 70 (last kept),
#: 284.5 -> 71 (the partial last, dropped).
ENTRIES = [_entry("True", "True", 0.5), _entry("True", "True", 4.0),
           _entry("True", "False", 7.99), _entry("False", "True", 9.0),
           _entry("True", "True", 9.5, duplicated="True"), _entry("True", "False", 30.0),
           _entry("True", "True", 283.0), _entry("False", "True", 284.5)]
KEPT_MARKED = [0, 1, 6, 69]


def _tags_text(entries, header=" timestamp,label,value,port", bom=True) -> str:
    lines = [header]
    for i, e in enumerate(entries):
        value = str({"reactionOrOmission": [e]}).replace('"', '""')
        lines.append(f'{1.6e12 + i * 1000.0},Marcador,"{value}",python-endpoint')
    return ("﻿" if bom else "") + "\n".join(lines) + "\n"


def _meta(record_id, task="Slackline", level=1, family="CGX", device="CGX", subject="UB0001"):
    condition = f"{task}-L{level}/{device}" if task == "Slackline" else f"{task}/{family}"
    return {"record_id": record_id, "subject": subject, "task": task, "level": level,
            "task_label": f"{task}-L{level}", "condition": condition, "family": family,
            "device": device, "session": "1", "path": "x", "undetermined": False,
            "bag_floor": windows.BAG_FLOOR}


@pytest.fixture(scope="module")
def eeg(tmp_path_factory):
    """One synthetic Slackline EEG file, copied into sibling folders with and without TAGS."""
    root = tmp_path_factory.mktemp("events")
    source = root / "source" / CGX_NAME
    _cgx_file(source, SECONDS, seed=7)
    return {"root": root, "bytes": source.read_bytes(), "path": source}


def _session(eeg, name, tags=None, extra_tags=None) -> Path:
    folder = eeg["root"] / name
    folder.mkdir()
    path = folder / CGX_NAME
    path.write_bytes(eeg["bytes"])
    if tags is not None:
        (folder / TAGS_NAME).write_text(tags, encoding="utf-8")
    if extra_tags is not None:
        (folder / "UB0001_TAGS_CGX_2023_01_01T00.00.01.csv").write_text(extra_tags,
                                                                        encoding="utf-8")
    return path


def _run(path, meta, out_name, root) -> tuple[dict, Path]:
    signals = root / out_name
    signals.mkdir()
    summary = exclusions.preprocess_record(str(path), meta, str(signals))
    return summary, signals / f"{meta['record_id']}.npz"


# ------------------------------------------------------------- parsing the TAGS file


def test_the_tags_file_parses_in_both_release_header_shapes():
    for text in (_tags_text(ENTRIES), _tags_text(ENTRIES, "timestamp,label,value,port", False)):
        rows = k1_events.parse_tags(text)
        assert [r["generalTime"] for r in rows] == [e["generalTime"] for e in ENTRIES]
    assert k1_events.event_counts(k1_events.parse_tags(_tags_text(ENTRIES))) == {
        "hit": 3, "error": 2, "correct no-press": 2}


def test_an_unparseable_tags_file_is_refused_not_read_as_no_events():
    for bad in ("timestamp,label,port\n1,Marcador,x\n",              # no value column
                _tags_text([]),                                      # no row at all
                ' timestamp,label,value,port\n1,Marcador,"{not python",p\n',
                ' timestamp,label,value,port\n1,Marcador,"{\'other\': 1}",p\n'):
        with pytest.raises(k1_events.K1EventError):
            k1_events.parse_tags(bad)


# ------------------------------------------------------------- the window an event falls in


def test_an_event_marks_the_kept_window_holding_its_general_time(eeg):
    record = read_record(eeg["path"])
    instants = k1_events.event_instants(ENTRIES)
    out = windows.record_windows(record, events=instants)
    mark = out[windows.EVENT_MARK]
    assert mark.dtype == np.bool_ and mark.shape == (out["n_windows"],) == (70,)
    assert np.flatnonzero(mark).tolist() == KEPT_MARKED
    # The formula: kept window = floor(generalTime / 4) - DISCARDED_LEADING.
    inside = [t for t in instants if 4.0 <= t < 71 * 4.0]
    assert sorted({int(t // windows.WINDOW_S) - windows.DISCARDED_LEADING
                   for t in inside}) == KEPT_MARKED
    # With the transient window kept, the event at 0.5 s marks raw window 0.
    first = windows.record_windows(record, events=instants, keep_first=True)
    assert np.flatnonzero(first[windows.EVENT_MARK]).tolist() == [0] + [k + 1 for k in KEPT_MARKED]
    assert k1_events.instants_outside(instants, out["window_start_s"], windows.WINDOW_S) == 2
    assert windows.EVENT_MARK == mechanisms.EVENT_MARK
    assert windows.EVENT_MARK in windows.WINDOW_ARRAYS
    assert windows.EVENT_MARK not in bag_data.NOT_MARKS


def test_the_event_mark_never_touches_another_array(eeg):
    record = read_record(eeg["path"])
    before = windows.record_windows(record)
    after = windows.record_windows(record, events=k1_events.event_instants(ENTRIES))
    assert set(after) - set(before) == {windows.EVENT_MARK}
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            assert after[key].dtype == value.dtype and after[key].tobytes() == value.tobytes()
        else:
            assert after[key] == value


# ------------------------------------------------------------- preprocess_record


def test_a_slackline_record_writes_its_event_window_and_counts_its_events(eeg):
    path = _session(eeg, "with_tags", tags=_tags_text(ENTRIES))
    summary, npz = _run(path, _meta("rec_tags"), "out_tags", eeg["root"])
    with np.load(npz) as z:
        mark = z[windows.EVENT_MARK]
        assert mark.dtype == np.bool_ and len(mark) == summary["n_windows"] == len(z["signal"])
        assert np.flatnonzero(mark).tolist() == KEPT_MARKED
    assert summary["event_tags"] == TAGS_NAME
    assert summary["event_defect"] is None
    assert (summary["events_hit"], summary["events_error"],
            summary["events_correct_no_press"]) == (3, 2, 2)
    assert summary["events_outside_kept"] == 2
    assert summary["windows_event_marked"] == len(KEPT_MARKED)


@pytest.mark.parametrize("case, defect", [("missing", "tags_missing"),
                                          ("unparseable", "tags_unparseable"),
                                          ("ambiguous", "tags_ambiguous")])
def test_a_slackline_record_without_a_readable_tags_file_gets_no_mark_and_one_defect(
        eeg, case, defect):
    tags = {"missing": None, "unparseable": "timestamp,label,port\n1,Marcador,x\n",
            "ambiguous": _tags_text(ENTRIES)}[case]
    path = _session(eeg, f"defect_{case}", tags=tags,
                    extra_tags=_tags_text(ENTRIES) if case == "ambiguous" else None)
    summary, npz = _run(path, _meta(f"rec_{case}"), f"out_{case}", eeg["root"])
    with np.load(npz) as z:
        assert windows.EVENT_MARK not in z.files
    assert summary["event_defect"].split(":")[0] == defect
    assert summary["reasons"] == []                 # a defect of the mark, never an exclusion
    for key in ("events_hit", "events_error", "events_correct_no_press",
                "events_outside_kept", "windows_event_marked"):
        assert summary[key] is None
    manifest = exclusions.manifest_frame([summary], _demo())
    assert manifest["event_defect"].iloc[0].startswith(defect)


def test_another_condition_gets_no_event_key_and_no_defect(tmp_path):
    path = tmp_path / "rob" / "UB0001_EPOCX_1_2023.md.pm.bp.csv"
    _emotiv_file(path, SECONDS, seed=2, cq_level=4.0)
    (path.parent / TAGS_NAME).write_text(_tags_text(ENTRIES), encoding="utf-8")
    meta = _meta("rec_robots", task="Robots", level=None, family="EMOTIV", device="EPOCX")
    signals = tmp_path / "out"
    signals.mkdir()
    summary = exclusions.preprocess_record(str(path), meta, str(signals))
    with np.load(signals / "rec_robots.npz") as z:
        assert windows.EVENT_MARK not in z.files
    assert summary["event_defect"] is None and summary["event_tags"] is None


# ------------------------------------------------------------- the invariant


def _pre_change_npz(path: Path, meta: dict, target: Path) -> None:
    """The npz exactly as preprocessing wrote it before the event mark existed."""
    record = read_record(path)
    out = windows.record_windows(record)            # no events: the pre-change path
    arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    np.savez(target, **arrays, channels=np.array(out["channels"]),
             record_id=np.array(meta["record_id"]), subject=np.array(meta["subject"]),
             condition=np.array(meta["condition"]), device=np.array(meta["device"]),
             family=np.array(meta["family"]), rate=np.array(out["rate"]),
             target_rate=np.array(out["target_rate"]),
             gap_exceeds_window=np.array(out["gap_exceeds_window"]),
             contact_meets_rule=np.array(bool(out.get("contact_meets_rule", False))))


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {name: z.read(name) for name in z.namelist()}


def _demo() -> pd.DataFrame:
    return pd.DataFrame({"subject": ["UB0001"], "label": ["adhd"], "sex": ["M"], "age": [9]})


def test_every_existing_npz_member_is_byte_for_byte_unchanged(eeg, tmp_path):
    meta = _meta("rec_invariant")
    path = _session(eeg, "invariant", tags=_tags_text(ENTRIES))
    summary, npz = _run(path, meta, "out_invariant", eeg["root"])
    reference = tmp_path / "reference.npz"
    _pre_change_npz(path, meta, reference)
    before, after = _members(reference), _members(npz)
    assert set(after) - set(before) == {f"{windows.EVENT_MARK}.npy"}
    assert set(before) <= set(after)
    for name, data in before.items():
        assert after[name] == data, name
    # Without the TAGS file the npz is the pre-change one, member for member.
    bare, bare_npz = _run(_session(eeg, "invariant_bare"), meta, "out_invariant_bare",
                          eeg["root"])
    assert _members(bare_npz) == before
    # Every summary field other than the event fields is unchanged.
    event_fields = {"event_tags", "event_defect", "events_hit", "events_error",
                    "events_correct_no_press", "events_outside_kept", "windows_event_marked",
                    "npz_bytes"}
    assert ({k: v for k, v in summary.items() if k not in event_fields}
            == {k: v for k, v in bare.items() if k not in event_fields})


def test_the_partition_inputs_do_not_change(eeg):
    marked, _ = _run(_session(eeg, "survivors_tags", tags=_tags_text(ENTRIES)),
                     _meta("rec_s"), "out_s", eeg["root"])
    bare, _ = _run(_session(eeg, "survivors_bare"), _meta("rec_s"), "out_s_bare", eeg["root"])
    with_marks = exclusions.manifest_frame([marked], _demo())
    without = exclusions.manifest_frame([bare], _demo())
    pd.testing.assert_frame_equal(partition.survivors(with_marks), partition.survivors(without))
    event_columns = [c for c in with_marks.columns if c.startswith(("event", "windows_event"))]
    assert set(event_columns) == {"event_tags", "event_defect", "events_hit", "events_error",
                                  "events_correct_no_press", "events_outside_kept",
                                  "windows_event_marked"}
    pd.testing.assert_frame_equal(with_marks.drop(columns=event_columns + ["npz_bytes"]),
                                  without.drop(columns=event_columns + ["npz_bytes"]))
    assert json.dumps(marked["reasons"]) == json.dumps(bare["reasons"]) == "[]"


def test_a_summary_cached_before_the_event_mark_is_recomputed(tmp_path):
    meta = _meta("rec_cache")
    cache = tmp_path / "rec_cache.json"
    old = {**meta, "readable": True, "npz": None, "reasons": []}
    cache.write_text(json.dumps(old), encoding="utf-8")
    assert exclusions._cached(cache, tmp_path, meta) is None
    cache.write_text(json.dumps({**old, **dict.fromkeys(exclusions.EVENT_FIELDS)}),
                     encoding="utf-8")
    assert exclusions._cached(cache, tmp_path, meta) is not None
    unreadable = {**meta, "readable": False, "reasons": ["lectura unreadable"]}
    cache.write_text(json.dumps(unreadable), encoding="utf-8")
    assert exclusions._cached(cache, tmp_path, meta) is not None
