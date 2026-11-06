"""What the BALLADEER release holds: subjects, tasks, sessions, files, labels.

Experiments plan v58, Secs. 1, 3.2 and 10.3. Before any audit step measures a
record, something has to say which records exist and whom they belong to. This
module walks the release tree `<subject>/<task>/<session>/<file>` without
opening a single data file, and reads `users_demographics.json`, the source of
the label.

A **condition** is the plan's recording context (Sec. 3.2): task, level and
device family. A subject has up to five EEG conditions, and the overlap between
them decides how many subjects each subject-disjoint partition can hold
(Sec. 10.3).

The label is `diagnosed` (`yes` / `no` / `undetermined`), not `group`: the
data paper assigns `group` by the psychologists' criterion, and the release
itself shows the two disagree (diagnosed children sit in the control group).
Both are kept, so a reader can see the disagreement instead of inheriting it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md",
    "sections": ["3.2", "10.3"],
    "experiments": ["3.2", "10.3"],
    "invariants": [
        "index_has_one_row_per_session_file_and_skips_finder_debris",
        "indeterminate_count_and_the_group_it_landed_in",
        "condition_overlap_counts_subjects_shared_by_two_conditions",
    ],
}

#: Release-relative directory of the raw data, inside the product folder.
RAW_RELATIVE = Path("MIL-CREDA") / "Data" / "raw" / "BALLADEER ADHD DATASET"

#: Task folder -> (task, level). Levels exist only for Slackline.
TASKS = {
    "SlacklineLvl1": ("Slackline", 1),
    "SlacklineLvl6": ("Slackline", 6),
    "SlacklineLvl11": ("Slackline", 11),
    "AttentionRobotsDesktop": ("Robots", None),
    "Cognifit": ("Cognifit", None),
}

#: Data paper, Table 1: participants with at least one valid recording.
DATA_PAPER_TABLE1 = {
    ("Slackline-L1", "CGX"): 140, ("Slackline-L6", "CGX"): 140,
    ("Slackline-L11", "CGX"): 140,
    ("Robots", "EPOCPLUS"): 59, ("Robots", "EPOCX"): 73,
    ("Cognifit", "EPOCPLUS"): 59, ("Cognifit", "EPOCX"): 70,
}

_IGNORED = {".DS_Store"}


def file_kind(name: str) -> str:
    """What a session file is, read off its name alone."""
    if "_EEG_CGX_" in name or re.search(r"_EPOC(X|PLUS)_", name):
        return "EEG"
    if "_GAME_DATA_" in name:
        return "GAME_DATA"
    if "_EYE_TRACKING_DATA_" in name:
        return "EYE_TRACKING"
    if "_TAGS_" in name:
        return "TAGS"
    return "OTHER"


def task_label(task: str, level: int | None) -> str:
    return f"{task}-L{level}" if level is not None else task


def build_index(root: str | Path) -> pd.DataFrame:
    """One row per session file. Never opens a file; only names and sizes."""
    root = Path(root)
    rows = []
    for subject_dir in sorted(p for p in root.iterdir()
                              if p.is_dir() and p.name.startswith("UB")):
        for task_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            task, level = TASKS.get(task_dir.name, (task_dir.name, None))
            for session_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                for file in sorted(session_dir.iterdir()):
                    if not file.is_file() or file.name in _IGNORED:
                        continue
                    kind = file_kind(file.name)
                    device = None
                    if kind == "EEG":
                        match = re.search(r"_(EPOCX|EPOCPLUS)_", file.name)
                        device = match.group(1) if match else "CGX"
                    rows.append({
                        "subject": subject_dir.name, "task_dir": task_dir.name,
                        "task": task, "level": level,
                        "task_label": task_label(task, level),
                        "session": session_dir.name, "kind": kind,
                        "device": device, "file": file.name,
                        "path": str(file.relative_to(root)),
                        "size_bytes": file.stat().st_size,
                    })
    return pd.DataFrame(rows)


def eeg_records(index: pd.DataFrame) -> pd.DataFrame:
    """The EEG files of the index, with family, condition and a stable id."""
    eeg = index[index["kind"] == "EEG"].copy()
    eeg["family"] = eeg["device"].map(lambda d: "CGX" if d == "CGX" else "EMOTIV")
    eeg["condition"] = eeg["task_label"] + "/" + eeg["family"]
    eeg["record_id"] = (eeg["subject"] + "__" + eeg["task_dir"] + "__"
                        + eeg["session"] + "__" + eeg["device"])
    return eeg.reset_index(drop=True)


def load_demographics(root: str | Path) -> pd.DataFrame:
    """Subject, label (`diagnosed`), sex, age and group."""
    rows = json.loads((Path(root) / "users_demographics.json").read_text(encoding="utf-8"))
    sex = {1: "M", 2: "F", "1": "M", "2": "F"}
    return pd.DataFrame([{
        "subject": row["user"], "label": row["diagnosed"],
        "sex": sex.get(row["gender"], str(row["gender"])),
        "age": int(row["age"]), "group": int(row["group"]),
    } for row in rows]).sort_values("subject").reset_index(drop=True)


def indeterminate_summary(demo: pd.DataFrame) -> dict:
    """How many subjects are `undetermined`, and which group they were put in."""
    undetermined = demo[demo["label"] == "undetermined"]
    return {
        "count": int(len(undetermined)),
        "by_group": {str(k): int(v) for k, v in
                     undetermined["group"].value_counts().sort_index().items()},
        "by_sex": {str(k): int(v) for k, v in
                   undetermined["sex"].value_counts().sort_index().items()},
    }


def subjects_per_condition(eeg: pd.DataFrame) -> pd.DataFrame:
    """Unique subjects per task (rows) and device (columns)."""
    return (eeg.groupby(["task_label", "device"])["subject"].nunique()
            .unstack(fill_value=0))


def condition_overlap(eeg: pd.DataFrame) -> pd.DataFrame:
    """Subjects shared by every pair of conditions; the diagonal is each size."""
    sets = eeg.groupby("condition")["subject"].apply(set)
    names = list(sets.index)
    return pd.DataFrame([[len(sets[a] & sets[b]) for b in names] for a in names],
                        index=names, columns=names)
