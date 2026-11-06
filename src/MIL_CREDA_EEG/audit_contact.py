"""Contact-quality audit, asymmetric across devices by necessity.

Experiments plan v55, Sec. 2.5. Emotiv writes quality indicators per channel
and per session; CGX writes none, so its criterion has to come from the signal.

**Emotiv.** The defect is falling below the acquisition threshold the data
paper itself declares: "an acceptable threshold being no lower than 90%" of
signal quality. `EQ.OVERALL` is that signal-quality percentage; `CQ.Overall`,
the contact percentage the setup turned green before recording, is measured
against the same number so both readings are visible. The threshold is the
paper's; how a record-level verdict aggregates windows is NOT declared, so this
module reports the median and the fraction of four-second windows whose median
falls below, and turns neither into an exclusion.

**CGX.** No column and no declared threshold exist. The signal-derived
measurement is mains pick-up: dry electrodes with poor contact have high
impedance and pick up more 50 Hz (the acquisition country's mains, Sec. 2.7).
The ratio of 49-51 Hz power to in-band power is reported per channel, with no
cut.

The audit of this audit -- whether its marks correlate with the label, because
a child with the disorder plausibly moves and touches the sensors more -- is
run over these distributions in `defect_log`, measured and declared, not fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md",
    "sections": ["2.5"],
    "experiments": ["2.5"],
    "invariants": [
        "emotiv_quality_below_the_declared_ninety_is_counted_per_window",
        "cgx_contact_has_no_quality_column_so_it_is_read_from_the_signal",
        "two_models_with_the_same_proxy_per_level_read_as_the_same_scale",
        "a_model_reporting_a_rescaled_level_reads_as_a_different_scale",
        "contact_scale_measures_never_alter_the_input",
        "eight_of_fourteen_channels_bad_in_most_windows_meets_the_rule",
        "seven_of_fourteen_channels_bad_does_not_meet_the_rule",
        "a_channel_bad_in_exactly_half_the_windows_is_not_bad",
    ],
}

#: Data paper, Technical Validation: signal quality "no lower than 90%".
EMOTIV_QUALITY_THRESHOLD = 90.0
WINDOW_S = 4.0
LINE_HZ = 50.0
BAND = (0.2, 45.0)


def emotiv_quality(timestamps: np.ndarray, quality: dict[str, np.ndarray],
                   threshold: float = EMOTIV_QUALITY_THRESHOLD,
                   window_s: float = WINDOW_S,
                   channels: tuple[str, ...] = ()) -> dict:
    """Median and windowed fraction below the threshold, overall and per channel."""
    timestamps = np.asarray(timestamps, dtype=float)
    windows = np.floor((timestamps - timestamps[0]) / window_s).astype(np.int64)
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
    n_windows = int(np.floor((timestamps[-1] - timestamps[0] + step) / window_s + 1e-9))
    report: dict = {"threshold": float(threshold), "window_s": float(window_s)}
    for key in ("EQ.OVERALL", "CQ.Overall"):
        values = quality.get(key)
        if values is None or not np.isfinite(values).any():
            report[key] = None
            continue
        values = np.asarray(values, dtype=float)
        ok = np.isfinite(values) & (windows < max(n_windows, 1))
        per_window = {}
        for w, v in zip(windows[ok], values[ok]):
            per_window.setdefault(int(w), []).append(v)
        medians = np.array([np.median(v) for v in per_window.values()])
        finite = values[np.isfinite(values)]
        report[key] = {
            "median": float(np.median(finite)),
            "sample_fraction_below": float(np.mean(finite < threshold)),
            "windows_with_reading": int(len(medians)),
            "window_fraction_below": (float(np.mean(medians < threshold))
                                      if len(medians) else None),
        }
    report["per_channel_contact_median"] = {
        ch: float(np.nanmedian(quality[f"CQ.{ch}"])) for ch in channels
        if f"CQ.{ch}" in quality and np.isfinite(quality[f"CQ.{ch}"]).any()}
    report["per_channel_signal_quality_median"] = {
        ch: float(np.nanmedian(quality[f"EQ.{ch}"])) for ch in channels
        if f"EQ.{ch}" in quality and np.isfinite(quality[f"EQ.{ch}"]).any()}
    return report


def line_noise_ratio(signal: np.ndarray, rate: float, line: float = LINE_HZ,
                     window_s: float = WINDOW_S) -> np.ndarray:
    """Per channel: power within 1 Hz of the mains line over in-band power."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal, axis=0)
    signal = np.nan_to_num(signal)
    freqs, power = welch(signal, fs=rate, nperseg=min(len(signal), int(rate * window_s)),
                         axis=0)
    line_power = power[(freqs >= line - 1) & (freqs <= line + 1)].sum(axis=0)
    in_band = power[(freqs >= BAND[0]) & (freqs <= BAND[1])].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(in_band > 0, line_power / in_band, np.nan)


# ------------------------------------------------- contact scale per model
#
# Experiments plan v04, Sec. 2.5. Both Emotiv models write `CQ.*` columns, but
# the fraction of windows below the threshold differs sharply by model. Before
# any cut is set, the question is whether the two models report contact on the
# same scale. Two readings answer it, neither of which excludes anything:
#
# 1. The value set each model writes (distinct values and their frequencies).
# 2. A calibration against the signal itself: per four-second window and
#    channel, mains pick-up (49-51 Hz over 1-45 Hz) and broadband
#    high-frequency power (35-45 Hz over 1-45 Hz), on the raw signal at its
#    native rate. Under a shared scale, the same reported level carries the
#    same proxy on both models. Windows are consecutive blocks of samples; only
#    windows whose level is constant ("pure") enter the calibration, and the
#    record, not the window, is the unit of every test.

#: Highest per-channel contact level Emotiv writes (green).
CQ_TOP = 4.0
REFERENCE_BAND = (1.0, 45.0)
HIGH_BAND = (35.0, 45.0)
PROXIES = ("line_ratio", "hf_ratio")


def value_counts(values: np.ndarray, decimals: int = 4) -> dict[float, int]:
    """Distinct finite values (rounded, to merge float32 spellings) and their counts."""
    values = np.asarray(values, dtype=float)
    values = np.round(values[np.isfinite(values)], decimals)
    distinct, counts = np.unique(values, return_counts=True)
    return {float(v): int(c) for v, c in zip(distinct, counts)}


def _blocks(array: np.ndarray, rate: float, window_s: float) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    n = int(round(rate * window_s))
    n_windows = len(array) // n
    return array[:n_windows * n].reshape(n_windows, n, *array.shape[1:])


def window_proxies(signal: np.ndarray, rate: float,
                   window_s: float = WINDOW_S) -> dict[str, np.ndarray]:
    """Per window and channel: mains and high-frequency power over 1-45 Hz power."""
    blocks = np.nan_to_num(_blocks(signal, rate, window_s))
    freqs, power = welch(blocks, fs=rate, nperseg=min(blocks.shape[1], int(rate * 2)),
                         axis=1)
    def band(lo: float, hi: float) -> np.ndarray:
        return power[:, (freqs >= lo) & (freqs <= hi)].sum(axis=1)
    reference = band(*REFERENCE_BAND)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"line_ratio": np.where(reference > 0, band(LINE_HZ - 1, LINE_HZ + 1) / reference,
                                       np.nan),
                "hf_ratio": np.where(reference > 0, band(*HIGH_BAND) / reference, np.nan)}


def window_levels(cq: np.ndarray, rate: float,
                  window_s: float = WINDOW_S) -> tuple[np.ndarray, np.ndarray]:
    """Per window and channel: the median contact level, and whether it never changed."""
    blocks = _blocks(cq, rate, window_s)
    with np.errstate(invalid="ignore"):
        pure = (np.all(np.isfinite(blocks), axis=1)
                & (np.nanmax(blocks, axis=1) == np.nanmin(blocks, axis=1)))
        levels = np.nanmedian(blocks, axis=1)
    return levels, pure


def calibration_rows(proxies: dict[str, np.ndarray], levels: np.ndarray,
                     pure: np.ndarray) -> list[dict]:
    """Per contact level, over pure channel-windows: their count and proxy medians."""
    rows = []
    for level in np.unique(levels[pure]):
        mask = pure & (levels == level)
        row = {"level": float(level), "n_windows": int(mask.sum())}
        for name, values in proxies.items():
            v = values[mask]
            v = v[np.isfinite(v)]
            row[f"{name}_median"] = float(np.median(v)) if len(v) else None
        rows.append(row)
    return rows


def overall_vs_channels(cq: np.ndarray, overall: np.ndarray) -> dict[int, dict[float, int]]:
    """Per sample, `CQ.Overall` against how many channels sit at the top level."""
    cq, overall = np.asarray(cq, dtype=float), np.asarray(overall, dtype=float)
    ok = np.isfinite(overall) & np.all(np.isfinite(cq), axis=1)
    at_top = (cq[ok] == CQ_TOP).sum(axis=1)
    pairs, counts = np.unique(np.column_stack([at_top, np.round(overall[ok], 4)]),
                              axis=0, return_counts=True)
    table: dict[int, dict[float, int]] = {}
    for (n, value), c in zip(pairs, counts):
        table.setdefault(int(n), {})[float(value)] = int(c)
    return table


def _holm(p: list) -> list:
    p = np.array([np.nan if v is None else v for v in p], dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, p[i] * (m - k)))
        out[i] = running
    return [None if not np.isfinite(v) else float(v) for v in out]


def _f_test(y: np.ndarray, full: np.ndarray, reduced: np.ndarray) -> float | None:
    from scipy.stats import f as f_dist
    rss = []
    for design in (full, reduced):
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        rss.append(float(((y - design @ beta) ** 2).sum()))
    rank_full = np.linalg.matrix_rank(full)
    df_num = rank_full - np.linalg.matrix_rank(reduced)
    df_den = len(y) - rank_full
    if df_num <= 0 or df_den <= 0 or rss[0] <= 0:
        return None
    return float(f_dist.sf(((rss[1] - rss[0]) / df_num) / (rss[0] / df_den), df_num, df_den))


def scale_comparison(table: pd.DataFrame, proxy: str, model_col: str = "device",
                     level_col: str = "level", alpha: float = 0.05) -> dict:
    """Same level, same proxy? Per-level Mann-Whitney (Holm) and a rank-regression interaction.

    `table` holds one row per record and level. The verdict reads "same" only
    when no level differs between models after Holm AND the model x level
    interaction is not significant; the numbers behind it are returned.
    """
    from scipy.stats import mannwhitneyu, rankdata
    data = table[[model_col, level_col, proxy]].dropna()
    models = sorted(data[model_col].unique())
    if len(models) != 2:
        raise ValueError(f"need exactly two models, got {models}")
    per_level, p_values = [], []
    for level in sorted(data[level_col].unique()):
        groups = [data.loc[(data[level_col] == level) & (data[model_col] == m), proxy]
                  for m in models]
        row = {"level": float(level)}
        for m, g in zip(models, groups):
            row[f"n_{m}"] = int(len(g))
            row[f"median_{m}"] = float(g.median()) if len(g) else None
        p = (float(mannwhitneyu(*groups, alternative="two-sided").pvalue)
             if all(len(g) >= 2 for g in groups) else None)
        row["mannwhitney_p"] = p
        p_values.append(p)
        per_level.append(row)
    for row, adjusted in zip(per_level, _holm(p_values)):
        row["p_holm"] = adjusted
    y = rankdata(data[proxy].to_numpy())
    level_dummies = pd.get_dummies(data[level_col].astype(str), drop_first=True).to_numpy(float)
    model = (data[model_col] == models[1]).to_numpy(float)[:, None]
    ones = np.ones((len(y), 1))
    interaction = level_dummies * model
    base = np.hstack([ones, level_dummies])
    interaction_p = _f_test(y, np.hstack([base, model, interaction]), np.hstack([base, model]))
    model_p = _f_test(y, np.hstack([base, model]), base)
    differs = any(r["p_holm"] is not None and r["p_holm"] < alpha for r in per_level) \
        or (interaction_p is not None and interaction_p < alpha)
    return {"proxy": proxy, "models": models, "per_level": per_level,
            "interaction_p": interaction_p, "model_main_effect_p": model_p,
            "verdict": "different" if differs else "same"}


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def contact_scale_record(path: str, meta: dict) -> dict:
    """Value sets, calibration rows and header fields of one Emotiv file."""
    from MIL_CREDA_EEG.modality import emotiv_nominal_rate, read_columns, read_emotiv_header
    out = dict(meta)
    try:
        header = read_emotiv_header(path)
        rate = emotiv_nominal_rate(header)
        columns = read_columns(path)
        channels = [c[4:] for c in columns if c.startswith("EEG.") and f"CQ.{c[4:]}" in columns]
        quality = [c for c in columns if c.startswith(("CQ.", "EQ."))]
        frame = pd.read_csv(path, skiprows=1, low_memory=False,
                            usecols=[f"EEG.{c}" for c in channels] + quality)
        frame = frame.apply(pd.to_numeric, errors="coerce")
    except Exception as error:  # noqa: BLE001 -- the reason is the deliverable
        out.update({"readable": False, "error": f"{type(error).__name__}: {error}"})
        return _clean(out)
    signal = frame[[f"EEG.{c}" for c in channels]].to_numpy(float)
    cq = frame[[f"CQ.{c}" for c in channels]].to_numpy(float)
    levels, pure = window_levels(cq, rate)
    rows = calibration_rows(window_proxies(signal, rate), levels, pure)
    out.update({
        "readable": True, "error": None, "rate": rate, "header": header,
        "channels": channels,
        "value_counts": {c: value_counts(frame[c].to_numpy(float)) for c in quality},
        "calibration": rows, "pure_window_fraction": float(pure.mean()) if pure.size else None,
        "overall_vs_channels": (overall_vs_channels(cq, frame["CQ.Overall"].to_numpy(float))
                                if "CQ.Overall" in frame else None),
    })
    return _clean(out)


def _work(args: tuple[str, dict, str]) -> dict:
    import json
    path, meta, cache = args
    result = contact_scale_record(path, meta)
    with open(cache, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return result


def run_contact_scale(raw_root, out_dir, workers: int = 4, force: bool = False) -> list[dict]:
    """One pass over every Emotiv record, cached per record under `records/contact_scale/`."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    from MIL_CREDA_EEG import inventory
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    cache_dir = out_dir / "records" / "contact_scale"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eeg = inventory.eeg_records(inventory.build_index(raw_root))
    eeg = eeg[eeg.family == "EMOTIV"]
    keys = ["record_id", "subject", "task", "device", "session", "path"]
    done, todo = [], []
    for row in eeg.to_dict("records"):
        meta = _clean({k: row[k] for k in keys})
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


def scale_frames(records: list[dict]) -> dict[str, pd.DataFrame]:
    """Long tables: value counts, record-level calibration, overall map, header fields."""
    values, calib, overall, headers = [], [], [], []
    for r in records:
        if not r.get("readable"):
            continue
        base = {"record_id": r["record_id"], "device": r["device"], "task": r["task"]}
        for column, counts in r["value_counts"].items():
            for value, count in counts.items():
                values.append({**base, "column": column, "value": float(value), "count": count})
        for row in r["calibration"]:
            calib.append({**base, **row})
        for n, counts in (r["overall_vs_channels"] or {}).items():
            for value, count in counts.items():
                overall.append({**base, "channels_at_top": int(n), "overall": float(value),
                                "count": count})
        headers.append({**base, "rate": r["rate"], "pure_window_fraction": r["pure_window_fraction"],
                        **{f"header:{k}": v for k, v in r["header"].items()
                           if k not in ("title", "start timestamp", "stop timestamp", "samples",
                                        "headset serial")}})
    return {"values": pd.DataFrame(values), "calibration": pd.DataFrame(calib),
            "overall": pd.DataFrame(overall), "headers": pd.DataFrame(headers)}


# ------------------------------------------------- record rule (count only)
#
# Experiments plan v04, Sec. 2.5, record rule approved by the operator on
# 2026-10-27: "se excluye una grabación solo cuando la mayoría de sus 14
# electrodos tiene mal contacto en la mayoría de sus ventanas". This section
# only counts which records meet it; nothing here excludes anything.

#: The rule's majorities are strict: more than half.
RULE_MAJORITY = 0.5


def contact_rule(cq: np.ndarray, rate: float, window_s: float = WINDOW_S,
                 level: float = CQ_TOP) -> dict:
    """Whether one Emotiv record meets the operator-approved contact rule.

    Operationalization, verbatim: per Emotiv record, per channel, a window is
    bad-contact for that channel when its median `CQ.<channel>` in the window
    is below level 4 (the green level; CQ values are {0,1,2,4}); a channel is
    bad-contact for the record when more than half of the record's 4 s windows
    are bad for it; the record meets the rule when more than half of its 14
    channels (i.e. >= 8) are bad-contact.

    Windows are the consecutive sample blocks of `window_levels`; a window with
    no finite reading for a channel is not bad for it. `cq` is never altered.
    """
    levels, _ = window_levels(cq, rate, window_s)
    n_windows, n_channels = levels.shape
    with np.errstate(invalid="ignore"):
        bad = np.isfinite(levels) & (levels < level)
    fraction = bad.mean(axis=0) if n_windows else np.zeros(n_channels)
    bad_channels = fraction > RULE_MAJORITY
    return {"n_windows": int(n_windows), "n_channels": int(n_channels),
            "windows_without_reading": int((~np.isfinite(levels)).sum()),
            "bad_window_fraction": [float(f) for f in fraction],
            "n_bad_channels": int(bad_channels.sum()),
            "meets_rule": bool(bad_channels.sum() > n_channels * RULE_MAJORITY)}


def contact_rule_record(path: str, meta: dict) -> dict:
    """The rule on one Emotiv file, reading only its per-channel `CQ.*` columns."""
    from MIL_CREDA_EEG.modality import emotiv_nominal_rate, read_columns, read_emotiv_header
    out = dict(meta)
    try:
        rate = emotiv_nominal_rate(read_emotiv_header(path))
        columns = read_columns(path)
        channels = [c[4:] for c in columns if c.startswith("EEG.") and f"CQ.{c[4:]}" in columns]
        frame = pd.read_csv(path, skiprows=1, low_memory=False,
                            usecols=[f"CQ.{c}" for c in channels])
        cq = frame[[f"CQ.{c}" for c in channels]].apply(pd.to_numeric, errors="coerce")
    except Exception as error:  # noqa: BLE001 -- the reason is the deliverable
        out.update({"readable": False, "error": f"{type(error).__name__}: {error}"})
        return _clean(out)
    out.update({"readable": True, "error": None, "rate": rate, "channels": channels,
                **contact_rule(cq.to_numpy(float), rate)})
    return _clean(out)


def _rule_work(args: tuple[str, dict, str]) -> dict:
    import json
    path, meta, cache = args
    result = contact_rule_record(path, meta)
    with open(cache, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return result


def run_contact_rule(raw_root, out_dir, workers: int = 4, force: bool = False) -> pd.DataFrame:
    """The rule over every Emotiv record, cached per record under `records/contact_rule/`."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    from MIL_CREDA_EEG import inventory
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    cache_dir = out_dir / "records" / "contact_rule"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eeg = inventory.eeg_records(inventory.build_index(raw_root))
    eeg = eeg[eeg.family == "EMOTIV"]
    keys = ["record_id", "subject", "task", "device", "session", "path"]
    done, todo = [], []
    for row in eeg.to_dict("records"):
        meta = _clean({k: row[k] for k in keys})
        cache = cache_dir / f"{row['record_id']}.json"
        if cache.is_file() and not force:
            done.append(json.loads(cache.read_text(encoding="utf-8")))
        else:
            todo.append((str(raw_root / row["path"]), meta, str(cache)))
    if todo:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                done.extend(pool.map(_rule_work, todo, chunksize=1))
        else:
            done.extend(map(_rule_work, todo))
    frame = pd.DataFrame(sorted(done, key=lambda r: r["record_id"]))
    return frame.drop(columns=["bad_window_fraction", "channels"], errors="ignore")


def fisher_2x2(flag: pd.Series, group: pd.Series, a, b) -> dict:
    """Fisher exact test of a boolean flag between two values of `group`."""
    from scipy.stats import fisher_exact
    keep = group.isin([a, b]) & flag.notna()
    flag, group = flag[keep].astype(bool), group[keep]
    table = [[int((flag & (group == g)).sum()), int((~flag & (group == g)).sum())] for g in (a, b)]
    odds, p = fisher_exact(table)
    return {"groups": f"{a} vs {b}", f"meets_{a}": table[0][0], f"n_{a}": sum(table[0]),
            f"meets_{b}": table[1][0], f"n_{b}": sum(table[1]),
            "odds_ratio": float(odds), "fisher_p": float(p)}
