"""The reproducible entry point: raw BALLADEER in, `Data/processed/` out.

Experiments plan v54, Secs. 2.1-2.10, 3.1, 3.2 and 4.2. One function,
`build_processed`, takes the raw release and writes every processed record,
the manifest, the two flow tables and the exclusion crosses; then it freezes
the current partition over the surviving subjects (`partition`, Sec. 4.2)
and writes the sha256 of every processed file to a manifest kept in git. The
notebook `BALLADEER_03_Preprocesamiento.ipynb` calls it, and so does the
command line, from the repository root:

    PYTHONPATH=src python -m MIL_CREDA_EEG.preprocess \\
        --raw "MIL-CREDA/Data/raw/BALLADEER ADHD DATASET" --out MIL-CREDA/Data/processed

The command line writes the partition to `partition.PARTITION_RELATIVE` and
the hash manifest to `MIL-CREDA/Manifests/processed_sha256.csv` unless told
otherwise; the function writes them only where it is told to. A partition
that already exists is recomputed and compared, never overwritten
(`partition.freeze_partition`).

Nothing here reads an audit output, a scratch file or anything git-ignored
except the raw release itself: every exclusion and every mark is recomputed
from raw by the package's own functions. The only cache is the per-record one
under the output folder, and `rebuild=True` deletes it first -- only the files
this pipeline writes, never anything else in the folder.

**Where the raw data comes from.** Figshare, DOI 10.6084/m9.figshare.28676042,
file "BALLADEER ADHD DATASET.zip", MD5 c7e7b5d57855ada4a380f667d1a5cf85,
extracted to `MIL-CREDA/Data/raw/`. `check_raw` refuses a tree that does not
look like that release, with a message in Spanish because the notebooks are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from MIL_CREDA_EEG import exclusions, inventory, partition

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.7", "2.8", "2.10", "3.1", "3.2", "4.2"],
    "experiments": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.7", "2.8", "2.10", "3.1", "3.2",
                    "4.2"],
    "invariants": [
        "the_cli_and_the_notebook_entry_function_write_identical_outputs",
        "rebuild_clears_only_the_files_this_pipeline_writes",
        "a_raw_tree_without_demographics_or_with_the_wrong_subject_count_refuses",
        "a_zip_whose_md5_differs_refuses",
        "the_hash_manifest_lists_every_processed_file",
        "a_changed_partition_input_stops_the_entry_point",
    ],
}

DOI = "10.6084/m9.figshare.28676042"
ZIP_NAMES = ("BALLADEER ADHD DATASET.zip", "BALLADEER_ADHD_DATASET.zip")
ZIP_MD5 = "c7e7b5d57855ada4a380f667d1a5cf85"
#: Subject folders `UB*` in the published release.
EXPECTED_SUBJECTS = 158

#: Every file name this pipeline writes at the top of the output folder.
OUTPUT_FILES = ("manifest.csv", "flow_CGX.csv", "flow_EMOTIV.csv", "exclusion_crosses.csv")

#: Kept in git, outside the data folder (Sec. 4.2), relative to the repository root.
PARTITION_RELATIVE = partition.PARTITION_RELATIVE
SHA_MANIFEST_RELATIVE = Path("MIL-CREDA") / "Manifests" / "processed_sha256.csv"


class RawDataError(RuntimeError):
    """The raw release is missing or is not the published one."""


def check_raw(raw_root: str | Path, expected_subjects: int | None = EXPECTED_SUBJECTS) -> dict:
    """The raw release exists and has the published structure; refuses otherwise."""
    raw_root = Path(raw_root)
    fix = (f"Descargá 'BALLADEER ADHD DATASET.zip' de Figshare (DOI {DOI}) y "
           f"descomprimilo en MIL-CREDA/Data/raw/.")
    if not raw_root.is_dir():
        raise RawDataError(f"No existe la carpeta de datos crudos {raw_root}. {fix}")
    if not (raw_root / "users_demographics.json").is_file():
        raise RawDataError(f"Falta users_demographics.json en {raw_root}: no es el release "
                           f"publicado o quedó a medio descomprimir. {fix}")
    subjects = sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("UB"))
    if expected_subjects is not None and len(subjects) != expected_subjects:
        raise RawDataError(f"Se esperaban {expected_subjects} carpetas de sujeto UB* en "
                           f"{raw_root} y hay {len(subjects)}. {fix}")
    eeg = inventory.eeg_records(inventory.build_index(raw_root))
    if eeg.empty:
        raise RawDataError(f"No hay ningún archivo EEG bajo {raw_root}. {fix}")
    return {"subjects": len(subjects), "eeg_records": int(len(eeg)),
            "by_device": {k: int(v) for k, v in eeg["device"].value_counts().sort_index().items()}}


def file_md5(path: str | Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha_manifest(out_dir: str | Path, path: str | Path) -> pd.DataFrame:
    """Every file under `out_dir` (hidden files aside): its path, size and sha256."""
    out_dir, path = Path(out_dir), Path(path)
    files = sorted((p for p in out_dir.rglob("*")
                    if p.is_file() and not p.name.startswith(".")),
                   key=lambda p: p.relative_to(out_dir).as_posix())
    table = pd.DataFrame([{"file": p.relative_to(out_dir).as_posix(),
                           "bytes": p.stat().st_size, "sha256": file_sha256(p)}
                          for p in files], columns=["file", "bytes", "sha256"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    return table


def check_zip(data_dir: str | Path, expected: str = ZIP_MD5) -> dict:
    """The release zip's MD5 when the zip sits in `data_dir`; refuses on a mismatch."""
    for name in ZIP_NAMES:
        path = Path(data_dir) / name
        if path.is_file():
            found = file_md5(path)
            if found != expected:
                raise RawDataError(f"El MD5 de {name} es {found} y el publicado es {expected}: "
                                   f"el archivo no es el release de Figshare (DOI {DOI}).")
            return {"zip": name, "md5": found, "verified": True}
    return {"zip": None, "md5": None, "verified": False}


def clear_outputs(out_dir: str | Path) -> int:
    """Delete only what this pipeline writes: its CSVs, records/*.json, signals/*.npz."""
    out_dir = Path(out_dir)
    removed = 0
    targets = [out_dir / name for name in OUTPUT_FILES]
    targets += sorted((out_dir / "records").glob("*.json"))
    targets += sorted((out_dir / "signals").glob("*.npz"))
    for path in targets:
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def build_processed(raw_root: str | Path, out_dir: str | Path, workers: int = 4,
                    rebuild: bool = True,
                    expected_subjects: int | None = EXPECTED_SUBJECTS,
                    partition_path: str | Path | None = None,
                    sha_manifest_path: str | Path | None = None,
                    partition_seed: int = partition.PARTITION_SEED,
                    regenerate_partition: bool = False,
                    new_partition_name: str | None = None) -> dict:
    """Raw release -> processed records, manifest, flows, crosses, partition, hashes.

    The partition is frozen at `partition_path` and the hash manifest written
    at `sha_manifest_path`; either step is skipped when its path is None.
    """
    raw = check_raw(raw_root, expected_subjects)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleared = clear_outputs(out_dir) if rebuild else 0
    summaries = exclusions.run_preprocess(raw_root, out_dir, workers=workers, force=rebuild)
    demo = inventory.load_demographics(raw_root)
    outputs = exclusions.write_outputs(summaries, demo, out_dir)
    frozen = None
    if partition_path is not None:
        frozen = partition.freeze_partition(
            partition.survivors(outputs["manifest"]), partition_path, seed=partition_seed,
            regenerate=regenerate_partition, new_name=new_partition_name)
    hashes = (write_sha_manifest(out_dir, sha_manifest_path)
              if sha_manifest_path is not None else None)
    return {**outputs, "raw": raw, "cleared": cleared, "summaries": summaries,
            "partition": frozen, "sha_manifest": hashes}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="BALLADEER preprocessing (plan v18): raw release -> Data/processed/, "
                    "then the frozen partition and the hash manifest.")
    parser.add_argument("--raw", required=True, help="the extracted release folder")
    parser.add_argument("--out", required=True, help="the processed output folder")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-rebuild", action="store_true",
                        help="reuse the per-record cache under --out instead of recomputing")
    parser.add_argument("--expected-subjects", type=int, default=EXPECTED_SUBJECTS,
                        help="UB* folders the release must hold; 0 skips the count")
    parser.add_argument("--partition", default=str(PARTITION_RELATIVE),
                        help="the frozen current partition; recomputed and compared, never overwritten")
    parser.add_argument("--sha-manifest", default=str(SHA_MANIFEST_RELATIVE),
                        help="where the sha256 of every processed file is written")
    parser.add_argument("--regenerate-partition", metavar="NEW_NAME", default=None,
                        help="write a new partition under this file name, next to --partition")
    args = parser.parse_args(argv)
    result = build_processed(args.raw, args.out, args.workers, not args.no_rebuild,
                             args.expected_subjects or None,
                             partition_path=args.partition, sha_manifest_path=args.sha_manifest,
                             regenerate_partition=args.regenerate_partition is not None,
                             new_partition_name=args.regenerate_partition)
    manifest = result["manifest"]
    frozen = result["partition"]
    print(json.dumps({"records": int(len(manifest)),
                      "excluded": int(manifest["excluded"].sum()),
                      "windows": int(manifest["n_windows"].sum()),
                      "partition": {"status": frozen["status"], "path": str(frozen["path"]),
                                    "sha256": frozen["doc"][partition.HASH_FIELD]},
                      "raw": result["raw"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
