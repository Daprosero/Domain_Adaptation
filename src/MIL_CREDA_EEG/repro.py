"""Shared plumbing for the reproducibility notebooks (MIL-CREDA-EEG/Notebooks).

* `ieee_root()` gives the folder that holds `ADHD_group/` and `Control_group/` of the public
  Kaggle dataset `daprosero/mi-tdah-dataset` (CC0). Inside a Kaggle notebook with the dataset
  attached it uses the mounted copy; anywhere else (local, Colab) it downloads only the 121 IEEE
  `.mat` files (about 33 MB) listed in `Manifests/ieee_files.json` with `kagglehub`, and exports
  `IEEE_ROOT` so every loader that calls `xds_pilot.find_ieee` reads the same copy.
* `portion(items, split)` is the one rule every stage uses to split its work over accounts:
  `split=None` runs everything, `split=(k, n)` runs items k, k+n, k+2n, ... (0 <= k < n).
* `results_dir(stage)` is `MIL-CREDA-EEG/Results/paper/<stage>/`, where a stage notebook leaves
  its tables, figures and summaries.
* `compare(rows)` prints the stage's numbers next to the paper's and says which differ by more
  than the stated tolerance; results may vary slightly from run to run.
"""
from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

DATASET = "daprosero/mi-tdah-dataset"


def repo_root() -> Path:
    """The repository root (the folder that holds `src/` and `MIL-CREDA-EEG/`)."""
    return Path(__file__).resolve().parents[2]


def eeg_dir() -> Path:
    return repo_root() / "MIL-CREDA-EEG"


def manifest() -> dict:
    return json.loads((eeg_dir() / "Manifests" / "ieee_files.json").read_text())


def _mounted() -> str | None:
    hits = sorted(glob.glob("/kaggle/input/**/TDAH/ieee/ADHD_group", recursive=True))
    return str(Path(hits[0]).parent) if hits else None


def ieee_root(cache: str | os.PathLike | None = None) -> str:
    """Folder holding the IEEE `ADHD_group/` and `Control_group/` folders; sets `IEEE_ROOT`."""
    given = os.environ.get("IEEE_ROOT")
    if given and glob.glob(f"{given}/ADHD_group/*.mat"):
        return given
    root = _mounted()
    if root is None:
        root = download_ieee(cache)
    os.environ["IEEE_ROOT"] = root
    return root


def download_ieee(cache: str | os.PathLike | None = None) -> str:
    """Download the IEEE files of the public dataset with kagglehub and return their folder."""
    import kagglehub  # imported here: not needed inside Kaggle

    if cache is not None:
        os.environ["KAGGLEHUB_CACHE"] = str(cache)
    m = manifest()
    handle = f"{m['dataset']}/versions/{m['version']}"
    last = None
    for rel in m["files"]:
        last = kagglehub.dataset_download(handle, path=f"{m['prefix']}/{rel}")
    root = Path(last).parents[1]
    missing = [r for r in m["files"] if not (root / r).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} IEEE files missing after download, e.g. {missing[0]}")
    return str(root)


def parse_split(split) -> tuple[int, int] | None:
    """None, a (k, n) pair or a "k/n" string; returns None or (k, n) with 0 <= k < n."""
    if split is None or split == "" or split == "None":
        return None
    if isinstance(split, str):
        k, n = (int(x) for x in split.split("/"))
    else:
        k, n = (int(x) for x in split)
    if not (n >= 1 and 0 <= k < n):
        raise ValueError(f"split must satisfy 0 <= k < n, got k={k}, n={n}")
    return k, n


def portion(items: Sequence, split=None) -> list:
    """The share of `items` this run executes: all of them, or items k, k+n, ... for (k, n)."""
    s = parse_split(split)
    items = list(items)
    if s is None:
        return items
    k, n = s
    return items[k::n]


def results_dir(stage: str) -> Path:
    d = eeg_dir() / "Results" / "paper" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def compare(rows: Iterable[tuple[str, float, float, float]]) -> list[dict]:
    """rows of (name, value obtained, value in the paper, tolerance); prints and returns them."""
    out = []
    for name, got, paper, tol in rows:
        diff = abs(float(got) - float(paper))
        out.append({"name": name, "obtained": float(got), "paper": float(paper),
                    "tolerance": float(tol), "within": bool(diff <= tol)})
    width = max((len(r["name"]) for r in out), default=4)
    print(f"{'quantity':<{width}}  obtained    paper   within tolerance")
    for r in out:
        print(f"{r['name']:<{width}}  {r['obtained']:8.3f} {r['paper']:8.3f}   "
              f"{'yes' if r['within'] else 'NO'} (+/-{r['tolerance']:g})")
    return out
