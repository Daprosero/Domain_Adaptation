"""Domain separability: how far apart two recording conditions are, MEASURED.

Experiments plan v58, Secs. 2.7 and 5. Two questions come before any transfer
is run. (1) The Emotiv headset model (EPOC X vs EPOC+) is confounded with the
label -- is it visible in the preprocessed EEG at all? (2) For every candidate
transfer of Sec. 5 (level, task, device), how separable are its two domains?
A domain classifier answers both: if a plain classifier tells condition A from
condition B well above chance, the shift between them is large; if it cannot,
the shift is small.

**This is a measuring instrument, not the paper's method.** It never trains on
the diagnosis label, never modifies a file, never excludes a record beyond the
two exclusions the plan already declares, and chooses no transfer. Every choice
below is the instrument's and is declared here and in the notebook:

- **Signal (Sec. 2.7).** Scalp channels only. Missing samples are filled with
  the channel median in a copy (as `audit_ocular` does) so the filter can run
  over the continuous recording; band-pass 0.2-45 Hz (`audit_channels.bandpass`,
  zero-phase Butterworth order 4) on the continuous signal at the native rate,
  THEN polyphase resampling to 128 Hz, THEN 4 s non-overlapping windows. The
  partial last window is dropped (Sec. 3.1). The first window is dropped too:
  Sec. 2.7 discards "the first seconds" of the filter transient without a
  number, and one window (4 s, several time constants of a 0.2 Hz high-pass)
  is this instrument's reading of it.
- **Flat channels** (zero variance, the Sec. 2.4 definition) carry no feature
  (NaN), as in `audit_ocular`. Inside the classifier a NaN is imputed with the
  training-fold median; the test fold never informs it.
- **Rate.** The nominal rate the file declares (Emotiv header; CGX 500 Hz), as
  `audit_ocular` uses it.
- **Features.** Per window and channel, log10 band power (Welch, 2 s Hann
  segments, 50 % overlap, 0.5 Hz resolution) in delta 0.5-4, theta 4-8, alpha
  8-13, beta 13-30 and low gamma 30-45 Hz. Absolute power: the two families
  have different amplitude scales, and that difference is part of the shift.
  Record-level feature = median over the record's windows.
- **Channels.** A same-family pair uses its family's full montage. A CGX vs
  Emotiv pair uses only the channels both montages name identically in the
  10-20 system (`common_channels`). That intersection is for measurement only;
  Sec. 3.3 forbids it for the model.
- **Classifier.** Median imputation, standardisation, L2 logistic regression
  (C = 1, lbfgs), all fitted inside the training fold. Cross-validation is
  `StratifiedGroupKFold(5)` grouped by subject, so a subject never sits in
  training and test at once: where the same subjects appear on both sides,
  each subject's records stay in one fold. Metric: test-fold ROC AUC, mean and
  standard deviation over the five folds.
- **Null.** The domain label is permuted inside the grouped design
  (`permute_domain`): a subject with records on both sides keeps one record per
  side and has its two labels swapped with probability 1/2; subjects on one
  side only have their subject-level labels shuffled among themselves. The fold
  split is held fixed. p = (1 + #{null >= observed}) / (1 + n_permutations).
- **Seed** 20261025 for folds and permutations.
- **Window level** is reported too, same folds, no permutation null (cost).
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch
from scipy.stats import mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from MIL_CREDA_EEG import inventory
from MIL_CREDA_EEG.audit_channels import BAND, bandpass
from MIL_CREDA_EEG.audit_rate import effective_rate
from MIL_CREDA_EEG.defect_log import _clean
from MIL_CREDA_EEG.modality import (
    CGX_CHANNELS, CGX_NOMINAL_RATE, EMOTIV_CHANNELS, cgx_name, emotiv_nominal_rate,
    family_of, read_columns, read_emotiv_header,
)

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md",
    "sections": ["2.7", "5"],
    "experiments": ["2.7", "5"],
    "invariants": [
        "domains_differing_in_alpha_power_are_separable",
        "identical_domains_are_not_separable",
        "grouped_folds_never_put_one_subject_in_train_and_test",
        "features_never_alter_the_input_array",
        "the_band_pass_reaches_the_features_and_windows_are_non_overlapping",
        "a_flat_channel_carries_no_feature",
        "the_permutation_keeps_the_grouped_design",
        "the_common_channels_are_named_identically_in_both_montages",
    ],
}

COMMON_RATE = 128.0
WINDOW_S = 4.0
#: Windows dropped at the start for the filter transient (Sec. 2.7).
TRANSIENT_WINDOWS = 1
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma_low": (30.0, 45.0)}
WELCH_SEGMENT_S = 2.0
SEED = 20261025
N_SPLITS = 5
N_PERMUTATIONS = 200
MONTAGES = {"EMOTIV": EMOTIV_CHANNELS, "CGX": CGX_CHANNELS}
#: Record the plan excludes as a hard defect (Sec. 2.4).
EXCLUDED_RECORDS = (("UB0058", "Slackline-L6/CGX"),)


def common_channels() -> list[str]:
    """Channels both montages name identically, in Emotiv order. Measurement only."""
    return [c for c in EMOTIV_CHANNELS if c in CGX_CHANNELS]


# ------------------------------------------------------------- signal


def preprocess(signal: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass on the continuous copy, then resample to 128 Hz.

    Returns the resampled signal (samples x channels) and the per-channel
    `usable` mask (finite somewhere and non-zero variance). The input array is
    never written to.
    """
    x = np.array(np.asarray(signal, dtype=float), copy=True)
    finite = np.isfinite(x)
    usable = finite.any(axis=0)
    medians = np.zeros(x.shape[1])
    if usable.any():
        medians[usable] = np.nanmedian(np.where(finite, x, np.nan)[:, usable], axis=0)
    x = np.where(finite, x, medians)
    usable &= x.std(axis=0) > 0
    filtered = bandpass(x, rate, BAND)
    ratio = Fraction(COMMON_RATE / rate).limit_denominator(1000)
    if ratio != 1:
        filtered = resample_poly(filtered, ratio.numerator, ratio.denominator, axis=0)
    return filtered, usable


def window_features(x: np.ndarray, usable: np.ndarray, rate: float = COMMON_RATE,
                    window_s: float = WINDOW_S,
                    transient_windows: int = TRANSIENT_WINDOWS) -> np.ndarray:
    """log10 band power per window, channel and band: (windows, channels, bands)."""
    size = int(round(rate * window_s))
    n = len(x) // size - transient_windows
    if n <= 0:
        return np.empty((0, x.shape[1], len(BANDS)))
    start = transient_windows * size
    windows = np.asarray(x)[start: start + n * size].reshape(n, size, -1)
    nperseg = int(round(rate * WELCH_SEGMENT_S))
    freqs, psd = welch(windows, fs=rate, nperseg=nperseg, axis=1)
    df = freqs[1] - freqs[0]
    out = np.empty((n, x.shape[1], len(BANDS)))
    for k, (lo, hi) in enumerate(BANDS.values()):
        sel = (freqs >= lo) & (freqs < hi) if hi < BAND[1] else (freqs >= lo) & (freqs <= hi)
        power = psd[:, sel, :].sum(axis=1) * df
        with np.errstate(divide="ignore"):
            out[:, :, k] = np.where(power > 0, np.log10(power), np.nan)
    out[:, ~np.asarray(usable, dtype=bool), :] = np.nan
    return out


def record_features(windows: np.ndarray) -> np.ndarray:
    """Median over windows: (channels, bands). All-NaN channels stay NaN."""
    if len(windows) == 0:
        return np.full(windows.shape[1:], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN channel
        return np.nanmedian(windows, axis=0)


def feature_names(channels: list[str]) -> list[str]:
    return [f"{c}:{b}" for c in channels for b in BANDS]


# ------------------------------------------------------------- reading


def _read(path: str) -> tuple[str, np.ndarray, list[str], float]:
    """The scalp columns of one EEG file and its nominal rate."""
    family, _ = family_of(path)
    columns = read_columns(path)
    if family == "EMOTIV":
        wanted = {f"EEG.{c}": c for c in EMOTIV_CHANNELS}
        time_column = "Timestamp"
        nominal = emotiv_nominal_rate(read_emotiv_header(path))
    else:
        wanted = {col: cgx_name(col) for col in columns if cgx_name(col) in CGX_CHANNELS}
        time_column = "timestamps"
        nominal = CGX_NOMINAL_RATE
    present = [c for c in columns if c in wanted]
    frame = pd.read_csv(path, skiprows=1 if family == "EMOTIV" else 0,
                        usecols=present + [time_column], low_memory=False)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    rate = nominal or effective_rate(frame[time_column].to_numpy(dtype=float))
    return family, frame[present].to_numpy(dtype=float), [wanted[c] for c in present], float(rate)


def domain_record(path: str, meta: dict) -> tuple[dict, np.ndarray | None]:
    """Window features of one file; an unreadable file reports its reason."""
    out = dict(meta)
    try:
        family, signal, channels, rate = _read(path)
        x, usable = preprocess(signal, rate)
        windows = window_features(x, usable)
    except Exception as error:  # noqa: BLE001 -- the reason is the deliverable
        out.update({"readable": False, "error": f"{type(error).__name__}: {error}"})
        return _clean(out), None
    out.update({"readable": True, "error": None, "rate_used": rate, "channels": channels,
                "n_flat": int((~usable).sum()), "n_windows": int(len(windows))})
    return _clean(out), windows.astype(np.float32)


def _work(args: tuple[str, dict, str]) -> dict:
    path, meta, cache = args
    result, windows = domain_record(path, meta)
    if windows is not None:
        np.save(cache + ".npy", windows)
    Path(cache + ".json").write_text(json.dumps(result), encoding="utf-8")
    return result


def run_domains(raw_root: str | Path, out_dir: str | Path, workers: int = 4,
                force: bool = False) -> list[dict]:
    """Window features of every EEG record, cached under `records/domains/`."""
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    cache_dir = out_dir / "records" / "domains"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eeg = inventory.eeg_records(inventory.build_index(raw_root))
    keys = ["record_id", "subject", "task", "level", "task_label", "condition",
            "family", "device", "session", "path"]
    done, todo = [], []
    for row in eeg.to_dict("records"):
        meta = _clean({k: row[k] for k in keys})
        if meta["level"] is not None:
            meta["level"] = int(meta["level"])
        cache = cache_dir / row["record_id"]
        if Path(str(cache) + ".json").is_file() and not force:
            done.append(json.loads(Path(str(cache) + ".json").read_text(encoding="utf-8")))
        else:
            todo.append((str(raw_root / row["path"]), meta, str(cache)))
    if todo:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                done.extend(pool.map(_work, todo, chunksize=1))
        else:
            done.extend(map(_work, todo))
    return sorted(done, key=lambda r: r["record_id"])


def load_windows(out_dir: str | Path, record_id: str) -> np.ndarray:
    return np.load(Path(out_dir) / "records" / "domains" / f"{record_id}.npy")


def features_on(windows: np.ndarray, channels: list[str], wanted: list[str]) -> np.ndarray:
    """Reindex (windows, channels, bands) onto `wanted`; absent channels are NaN."""
    index = {c: k for k, c in enumerate(channels)}
    out = np.full((windows.shape[0], len(wanted), windows.shape[2]), np.nan)
    for j, c in enumerate(wanted):
        if c in index:
            out[:, j, :] = windows[:, index[c], :]
    return out


# ------------------------------------------------------------- classifier


def grouped_folds(y: np.ndarray, groups: np.ndarray, seed: int = SEED,
                  n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subject-grouped, domain-stratified folds; one subject never in both halves."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def _model():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def fold_aucs(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Test-fold ROC AUC of the domain classifier, one per fold."""
    aucs = []
    for train, test in folds:
        if len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
            aucs.append(np.nan)
            continue
        model = _model().fit(X[train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))
    return np.asarray(aucs, dtype=float)


def permute_domain(y: np.ndarray, groups: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """A domain labelling the grouped design could have produced.

    Subjects with records on both sides have their labels swapped with
    probability 1/2; one-side subjects have their labels shuffled among them.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    out = y.copy()
    subjects = pd.unique(groups)
    single = []
    for s in subjects:
        idx = np.flatnonzero(groups == s)
        if len(np.unique(y[idx])) > 1:
            if rng.random() < 0.5:
                out[idx] = 1 - y[idx]
        else:
            single.append(s)
    if single:
        labels = np.array([y[groups == s][0] for s in single])
        shuffled = rng.permutation(labels)
        for s, lab in zip(single, shuffled):
            out[groups == s] = lab
    return out


def separability(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 n_permutations: int = N_PERMUTATIONS, seed: int = SEED) -> dict:
    """Grouped-CV AUC, its fold spread, and a permutation null inside the design."""
    y = np.asarray(y, dtype=int)
    folds = grouped_folds(y, groups, seed)
    aucs = fold_aucs(X, y, folds)
    observed = float(np.nanmean(aucs))
    rng = np.random.default_rng(seed)
    null = np.array([np.nanmean(fold_aucs(X, permute_domain(y, groups, rng), folds))
                     for _ in range(n_permutations)])
    return {"auc_mean": observed, "auc_sd": float(np.nanstd(aucs, ddof=1)),
            "fold_aucs": aucs.tolist(),
            "perm_p": float((1 + np.sum(null >= observed)) / (1 + n_permutations))
            if n_permutations else None,
            "null_q025": float(np.quantile(null, 0.025)) if n_permutations else None,
            "null_q975": float(np.quantile(null, 0.975)) if n_permutations else None,
            "n_permutations": int(n_permutations)}


# ------------------------------------------------------------- pairs


def eligible(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Readable records minus the declared exclusions (Secs. 2.4, 4.3)."""
    ok = frame[frame["readable"]].copy()
    ok["label"] = ok["subject"].map(labels)
    drop = ok.apply(lambda r: (r["subject"], r["condition"]) in EXCLUDED_RECORDS, axis=1)
    return ok[~drop & (ok["label"] != "undetermined")].reset_index(drop=True)


def pairs(ok: pd.DataFrame) -> list[dict]:
    """Every candidate pair: (name, group, side A mask, side B mask, channels)."""
    em, cg = ok["family"] == "EMOTIV", ok["family"] == "CGX"
    x, p = ok["device"] == "EPOCX", ok["device"] == "EPOCPLUS"
    rob, cog = ok["task"] == "Robots", ok["task"] == "Cognifit"
    lv = {k: ok["condition"] == f"Slackline-L{k}/CGX" for k in (1, 6, 11)}
    emo, cgx, common = list(EMOTIV_CHANNELS), list(CGX_CHANNELS), common_channels()
    out = [
        ("EPOC X vs EPOC+ · Robots", "casco", x & rob, p & rob, emo),
        ("EPOC X vs EPOC+ · CogniFit", "casco", x & cog, p & cog, emo),
        ("EPOC X vs EPOC+ · ambas tareas", "casco", x & em, p & em, emo),
        ("EPOC X vs EPOC+ · ambas · solo TDAH", "casco (dentro de etiqueta)",
         x & em & (ok["label"] == "yes"), p & em & (ok["label"] == "yes"), emo),
        ("EPOC X vs EPOC+ · ambas · solo control", "casco (dentro de etiqueta)",
         x & em & (ok["label"] == "no"), p & em & (ok["label"] == "no"), emo),
        ("Slackline L1 vs L6", "nivel (5.1)", lv[1], lv[6], cgx),
        ("Slackline L1 vs L11", "nivel (5.1)", lv[1], lv[11], cgx),
        ("Slackline L6 vs L11", "nivel (5.1)", lv[6], lv[11], cgx),
        ("Robots vs CogniFit (Emotiv)", "tarea (5.2)", rob, cog, emo),
        ("CGX (3 niveles) vs Emotiv Robots", "dispositivo (5.3)", cg, rob, common),
        ("CGX (3 niveles) vs Emotiv CogniFit", "dispositivo (5.3)", cg, cog, common),
    ]
    return [{"pair": n, "arm": g, "a": a.to_numpy(), "b": b.to_numpy(), "channels": c}
            for n, g, a, b, c in out]


def pair_design(ok: pd.DataFrame, pair: dict, out_dir: str | Path,
                level: str = "record") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, y (1 = side A), subject groups for one pair, at record or window level."""
    rows, ys, gs = [], [], []
    for side, mask in ((1, pair["a"]), (0, pair["b"])):
        for r in ok[mask].itertuples():
            w = features_on(load_windows(out_dir, r.record_id), list(r.channels),
                            pair["channels"])
            if level == "record":
                rows.append(record_features(w).ravel()[None, :])
            else:
                rows.append(w.reshape(len(w), -1))
            n = 1 if level == "record" else len(w)
            ys += [side] * n
            gs += [r.subject] * n
    return np.vstack(rows), np.asarray(ys), np.asarray(gs)


def separability_table(ok: pd.DataFrame, out_dir: str | Path,
                       n_permutations: int = N_PERMUTATIONS, seed: int = SEED,
                       window_level: bool = True) -> pd.DataFrame:
    rows = []
    for pair in pairs(ok):
        X, y, g = pair_design(ok, pair, out_dir)
        res = separability(X, y, g, n_permutations, seed)
        a, b = ok[pair["a"]], ok[pair["b"]]
        row = {"pair": pair["pair"], "arm": pair["arm"],
               "n_records_a": len(a), "n_subjects_a": a["subject"].nunique(),
               "n_records_b": len(b), "n_subjects_b": b["subject"].nunique(),
               "n_subjects_both_sides": len(set(a["subject"]) & set(b["subject"])),
               "n_channels": len(pair["channels"]), "n_features": X.shape[1],
               "channels": ",".join(pair["channels"]), **res,
               "fold_aucs": ",".join(f"{v:.3f}" for v in res["fold_aucs"])}
        if window_level:
            Xw, yw, gw = pair_design(ok, pair, out_dir, level="window")
            wa = fold_aucs(Xw, yw, grouped_folds(yw, gw, seed))
            row.update({"window_n": len(yw), "window_auc_mean": float(np.nanmean(wa)),
                        "window_auc_sd": float(np.nanstd(wa, ddof=1))})
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- confounds


def headset_tables(ok: pd.DataFrame, records_csv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Descriptive only: headset vs label, vs nominal rate, vs contact quality."""
    em = ok[ok["family"] == "EMOTIV"].merge(
        records_csv[["record_id", "nominal_rate", "cq_window_fraction_below"]],
        on="record_id", how="left")
    label = (em.groupby(["task", "device"])["label"].value_counts().unstack(fill_value=0)
             .reindex(columns=["yes", "no"], fill_value=0))
    rate = em.groupby(["task", "device"])["nominal_rate"].value_counts().unstack(fill_value=0)
    contact = []
    for task, sub in [("ambas", em)] + list(em.groupby("task")):
        vx = sub.loc[sub["device"] == "EPOCX", "cq_window_fraction_below"].dropna()
        vp = sub.loc[sub["device"] == "EPOCPLUS", "cq_window_fraction_below"].dropna()
        contact.append({"task": task, "n_epocx": len(vx), "n_epocplus": len(vp),
                        "median_epocx": float(vx.median()), "median_epocplus": float(vp.median()),
                        "mean_epocx": float(vx.mean()), "mean_epocplus": float(vp.mean()),
                        "mannwhitney_p": float(mannwhitneyu(vx, vp).pvalue)})
    return {"label": label, "rate": rate, "contact": pd.DataFrame(contact)}
