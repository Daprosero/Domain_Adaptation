"""The modality filter: where every column of every EEG file goes.

Experiments plan v02, Sec. 2.1. Only EEG is kept from BALLADEER, and inside an
EEG file there are three destinations, not two:

* **signal** -- scalp EEG channels; the only columns the encoder will ever see;
* **audit** -- metadata steps 2.2-2.6 need and the encoder never sees: sample
  counters, timestamps, per-channel quality, the interpolation flag, the
  sample-rate quality, the reference and bipolar external channels of the
  high-density device, and the game logs;
* **discard** -- motion, triggers, battery, and the quantities the Emotiv
  software derives on its own (performance metrics and band power), one of
  which is a proprietary attention score.

**What the `.md.pm.bp` suffix is.** EmotivPRO names an export after the streams
it merged into the raw EEG file: `md` motion data (`MOT.*`), `pm` performance
metrics (`PM.*`) and `bp` band power (`POW.*`). Each row is one EEG sample; the
slower streams are written on the rows where they update and left empty
elsewhere. One file in the release is `.md.bp` -- it simply carries no `PM.*`.

A column no rule recognises is `unclassified`, never kept by default: a new
header variant must be read before it is trusted.

This module also reads a record, because reading IS the split: the arrays a
caller gets back are already sorted by destination, and discarded columns are
never loaded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v02.md",
    "sections": ["2.1"],
    "experiments": ["2.1"],
    "invariants": [
        "scalp_channels_go_to_signal_and_only_they_do",
        "reference_and_bipolar_externals_go_to_audit_not_discard",
        "motion_triggers_battery_and_proprietary_derivations_are_discarded",
        "an_unknown_column_is_unclassified_never_silently_kept",
    ],
}

SIGNAL = "signal"
AUDIT = "audit"
DISCARD = "discard"
UNCLASSIFIED = "unclassified"

#: The 14 Emotiv EPOC X / EPOC+ scalp channels, in file order.
EMOTIV_CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8",
                   "T8", "FC6", "F4", "F8", "AF4")

#: The 29 CGX Quick-32r scalp channels, in file order (README, Table 2).
CGX_CHANNELS = ("AF7", "Fpz", "F7", "Fz", "T7", "FC6", "Fp1", "F4", "C4", "Oz",
                "CP6", "Cz", "PO8", "CP5", "O2", "O1", "P3", "P4", "P7", "P8",
                "Pz", "PO7", "T8", "C3", "Fp2", "F3", "F8", "FC5", "AF8")

#: CGX additional channels: the mastoid reference and two bipolar externals.
CGX_EXTRAS = ("A2", "ExG 1", "ExG 2")

#: CGX nominal rate, declared by the data paper ("500 samples per second").
CGX_NOMINAL_RATE = 500.0

_STREAMS = {"md": "motion_data", "pm": "performance_metrics", "bp": "band_power"}


def _emotiv(column: str) -> tuple[str, str]:
    if column in ("Timestamp", "OriginalTimestamp"):
        return AUDIT, "timestamp"
    if column == "EEG.Counter":
        return AUDIT, "sample_counter"
    if column == "EEG.Interpolated":
        return AUDIT, "interpolation_flag"
    if column.startswith("EEG.") and column[4:] in EMOTIV_CHANNELS:
        return SIGNAL, "scalp_eeg"
    if column == "EEG.RawCq":
        return AUDIT, "channel_quality"
    if column in ("EEG.Battery", "EEG.BatteryPercent"):
        return DISCARD, "battery"
    if column in ("MarkerIndex", "MarkerType", "MarkerValueInt", "EEG.MarkerHardware"):
        return DISCARD, "trigger"
    if column == "EQ.SampleRateQuality":
        return AUDIT, "sample_rate_quality"
    if column.startswith(("CQ.", "EQ.")):
        return AUDIT, "channel_quality"
    if column.startswith("MOT."):
        return DISCARD, "motion"
    if column.startswith(("PM.Attention.", "PM.Focus.")):
        return DISCARD, "proprietary_attention_score"
    if column.startswith(("PM.", "POW.")):
        return DISCARD, "proprietary_derived"
    return UNCLASSIFIED, "unknown"


def _cgx(column: str) -> tuple[str, str]:
    name = cgx_name(column)
    if column == "timestamps":
        return AUDIT, "timestamp"
    if column == "Packet Counter":
        return AUDIT, "sample_counter"
    if name in CGX_CHANNELS:
        return SIGNAL, "scalp_eeg"
    if name == "A2":
        return AUDIT, "reference"
    if name in ("ExG 1", "ExG 2"):
        return AUDIT, "bipolar_external"
    if name.startswith("ACC"):
        return DISCARD, "motion"
    if column == "TRIGGER":
        return DISCARD, "trigger"
    return UNCLASSIFIED, "unknown"


def cgx_name(column: str) -> str:
    """`AF7(uV)` -> `AF7`; columns without a unit come back unchanged."""
    return re.sub(r"\([^)]*\)$", "", column)


def classify_column(family: str, column: str) -> tuple[str, str]:
    """`(destination, category)` of one column of one device family."""
    if family == "EMOTIV":
        return _emotiv(column)
    if family == "CGX":
        return _cgx(column)
    raise ValueError(f"unknown device family {family!r}")


_FILES = {
    "EEG": (SIGNAL, "eeg_recording"),
    "GAME_DATA": (AUDIT, "game_log"),
    "TAGS": (AUDIT, "game_log"),
    "DEMOGRAPHICS": (AUDIT, "label_source"),
    "SLACKLINE_FLAGS": (AUDIT, "game_log"),
    "EYE_TRACKING": (DISCARD, "other_modality"),
    "EMBRACEPLUS": (DISCARD, "other_modality"),
    "README": (DISCARD, "documentation"),
}


def classify_file(kind: str) -> tuple[str, str]:
    """Destination of a whole file, by the kind `inventory.file_kind` assigns."""
    return _FILES.get(kind, (UNCLASSIFIED, "unknown"))


def emotiv_export_streams(name: str) -> list[str]:
    """The streams an EmotivPRO export carries, read off its `.md.pm.bp` suffix."""
    stem = Path(name).name
    if stem.endswith(".csv"):
        stem = stem[:-4]
    tokens = []
    for token in reversed(stem.split(".")):
        if token not in _STREAMS:
            break
        tokens.append(token)
    return [_STREAMS[t] for t in reversed(tokens)]


def family_of(path: str | Path) -> tuple[str, str]:
    """`(family, device)` from a file name: CGX, EPOCX or EPOCPLUS."""
    name = Path(path).name
    if "_EEG_CGX_" in name:
        return "CGX", "CGX"
    match = re.search(r"_(EPOCX|EPOCPLUS)_", name)
    if match:
        return "EMOTIV", match.group(1)
    raise ValueError(f"{name} is not an EEG file of a known device")


def read_emotiv_header(path: str | Path) -> dict[str, str]:
    """The `key:value, ...` metadata line EmotivPRO writes above the header."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        line = handle.readline()
    header = {}
    for part in line.strip().split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            header[key.strip()] = value.strip()
    return header


def emotiv_nominal_rate(header: dict[str, str]) -> float | None:
    match = re.search(r"eeg_(\d+(?:\.\d+)?)", header.get("sampling rate", ""))
    return float(match.group(1)) if match else None


def read_columns(path: str | Path) -> list[str]:
    """The column header of an EEG file, without reading its body."""
    family, _ = family_of(path)
    with open(path, encoding="utf-8", errors="replace") as handle:
        if family == "EMOTIV":
            handle.readline()
        return handle.readline().strip().split(",")


def column_table(path: str | Path) -> list[dict[str, str]]:
    """One row per header column: its destination and category."""
    family, device = family_of(path)
    rows = []
    for column in read_columns(path):
        destination, category = classify_column(family, column)
        rows.append({"family": family, "device": device, "column": column,
                     "destination": destination, "category": category})
    return rows


@dataclass
class EEGRecord:
    """One EEG file, already split by destination. Discarded columns never load."""

    path: str
    family: str
    device: str
    columns: list[str]
    channels: list[str]
    signal: np.ndarray
    timestamps: np.ndarray
    counter: np.ndarray | None
    interpolated: np.ndarray | None
    nominal_rate: float | None
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    quality: dict[str, np.ndarray] = field(default_factory=dict)
    header: dict[str, str] = field(default_factory=dict)


def read_record(path: str | Path) -> EEGRecord:
    """Load one EEG file: signal and audit columns only."""
    family, device = family_of(path)
    columns = read_columns(path)
    keep = [c for c in columns if classify_column(family, c)[0] in (SIGNAL, AUDIT)]
    frame = pd.read_csv(path, skiprows=1 if family == "EMOTIV" else 0,
                        usecols=keep, low_memory=False)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    signal_columns = [c for c in columns if classify_column(family, c)[0] == SIGNAL]
    if family == "EMOTIV":
        header = read_emotiv_header(path)
        channels = [c[4:] for c in signal_columns]
        quality = {c: frame[c].to_numpy(dtype=float) for c in keep
                   if classify_column(family, c)[1] in ("channel_quality",
                                                         "sample_rate_quality")}
        return EEGRecord(
            path=str(path), family=family, device=device, columns=columns,
            channels=channels, signal=frame[signal_columns].to_numpy(dtype=float),
            timestamps=frame["Timestamp"].to_numpy(dtype=float),
            counter=(frame["EEG.Counter"].to_numpy(dtype=float)
                     if "EEG.Counter" in frame else None),
            interpolated=(frame["EEG.Interpolated"].to_numpy(dtype=float)
                          if "EEG.Interpolated" in frame else None),
            nominal_rate=emotiv_nominal_rate(header), quality=quality, header=header)
    extras = {cgx_name(c): frame[c].to_numpy(dtype=float) for c in keep
              if cgx_name(c) in CGX_EXTRAS}
    return EEGRecord(
        path=str(path), family=family, device=device, columns=columns,
        channels=[cgx_name(c) for c in signal_columns],
        signal=frame[signal_columns].to_numpy(dtype=float),
        timestamps=frame["timestamps"].to_numpy(dtype=float),
        counter=(frame["Packet Counter"].to_numpy(dtype=float)
                 if "Packet Counter" in frame else None),
        interpolated=None, nominal_rate=CGX_NOMINAL_RATE, extras=extras)
