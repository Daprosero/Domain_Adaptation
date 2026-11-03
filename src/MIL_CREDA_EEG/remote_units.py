"""The campaign's units of work, run one remote session at a time and merged back.

Experiments plan v43, Secs. 4.2, 4.6, 7.3, 7.6 and 7.7, under the forge's remote
contract (`remote-execution`: the session-chain rule, job inputs and the shard
stamp; `remote-parallelism`: the reproducibility checklist). Code review blocker
B5: the campaign had no shard-aware contract.

**Units** (`enumerate_units`), in one deterministic order, each with a stable id
and the ids it needs first (`after`):

1. `search.phase1.<transfer>.<group>` -- one phase-1 study of Sec. 7.7 and its
   search-seed checkpoint in every rotation (`campaign.phase_one_study`, then the
   checkpoint `campaign.run_search` stores);
2. `search.phase2.<transfer>.<strategy>` -- one phase-2 study, from the search-seed
   checkpoints of the strategy's group (`campaign.phase_two_study`);
3. `final.<transfer>` -- the `final` step on that transfer (`campaign.run_final`):
   the report-seed models of Sec. 7.3, the controls and the upper bound;
4. `k3.<transfer>` -- the `k3` step on that transfer (`campaign.run_k3`).

Transfers follow `assignment.TRANSFERS`, groups `model_store.GROUPS` and strategies
`model_store.STRATEGIES`, whatever order the config lists them in. A study trains
every rotation inside each trial (its objective is mean - sd over the four), so a
search unit is never split by rotation. `statistics`, `mechanisms`, `results` and
the K3 tables read every transfer at once: they run after the merge, as campaign
steps, and train nothing that a merged unit already stored.

**Running** (`run_units`). The units named by `FORGE_RUN_UNITS` (a JSON list of
ids), or by the caller, run in that order through the campaign's own functions,
each in its own temporary product seeded with the product's stored models,
studies and runs and with every completed unit not merged yet. Before the first
unit, `set_deterministic` sets the deterministic torch flags. Before each unit,
`FORGE_SESSION_DEADLINE` (epoch seconds) is read: when the unit may not finish
before it, `partial.json` is written in the working directory as
`{"remaining": [ids]}` and the call returns. A unit already complete is skipped,
and a unit whose `after` ids are neither complete, merged nor earlier in the run
is refused before anything runs.

**Inputs** (`resolve_inputs`). `config["remoteInputs"]` names the dataset ref of
the processed data and of the partition; `FORGE_INPUT_DATASETS` maps each ref to
its mounted directory, where the processed `manifest.csv` and the partition file
are looked for. A ref mapped to null is refused: this target has no download.

**Outputs.** Each unit writes `<shards>/<unit id>/` (default `Results/shards/`):
`outputs/` with the files it created under `Models/`, `Results/hpo/` and
`Results/*/runs/` (the lines it appended to a runs file, for those), `runs.jsonl`
with one record naming the unit, and `shard.json` last, holding the environment
stamp (python, torch, CUDA, cuDNN, accelerator, commit, requirements digest, code
version, deterministic flags, the plan's seeds, the data checksum and preparation
version and the partition hash) and the completion stamp (unit, every output with
its sha256, the derived files it discarded). Derived files (tables, figures,
summaries) are discarded: the steps after the merge rewrite them over every
transfer.

**Merge** (`merge`). Every `shard.json` under the given roots is checked before
anything is written: a duplicate unit, a unit already merged, an environment
field missing or null, shards that disagree on any field of `IDENTICAL_FIELDS`, an
expected identity that differs, an output whose bytes no longer match its sha256,
or a new file already in the product -- each refuses the whole merge. Otherwise
the outputs are written into the product layout in unit order and the units are
recorded in `Results/shards/merged.jsonl`.

**Seeds.** The units carry no seed of their own: the campaign functions train at
the plan's scientific seeds, the search seed 0 and the report seed 1 (Secs. 4.6
and 7.7). The forge checklist's seed derived from the unit id is not applied,
because it would replace the plan's declared seeds.

**Local only.** Nothing here submits, schedules or contacts a remote service.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, campaign, model_store, partition, preprocess, results, search

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v43.md",
    "sections": ["4.2", "4.6", "7.3", "7.6", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K3-ablacion"],
    "invariants": [
        "units_are_enumerated_in_a_deterministic_order_with_stable_ids",
        "units_name_their_dependencies",
        "selected_units_come_from_forge_run_units",
        "inputs_resolve_from_forge_input_datasets",
        "mounted_data_must_match_the_repository_manifest",
        "each_unit_writes_isolated_outputs_with_completion_and_environment_stamps",
        "the_deadline_stops_before_a_unit_and_writes_partial",
        "completed_units_are_skipped_on_resume",
        "a_unit_whose_dependencies_are_absent_is_refused_before_anything_runs",
        "deterministic_flags_are_set_before_any_unit",
        "merge_writes_the_product_layout_once",
        "merge_refuses_an_environment_mismatch_and_writes_nothing",
        "merge_refuses_a_duplicate_unit",
        "merge_refuses_a_missing_field_or_a_tampered_output",
        "units_run_the_campaign_functions_with_the_plans_seeds",
        "requirements_pin_torch_optuna_and_scipy_to_the_installed_versions",
    ],
}

#: The forge runner's variables (`remote-execution`, assets/runner_invoke.py).
UNITS_ENV = "FORGE_RUN_UNITS"
DEADLINE_ENV = "FORGE_SESSION_DEADLINE"
INPUTS_ENV = "FORGE_INPUT_DATASETS"
CLONE_COMMIT_ENV = "FORGE_CLONE_COMMIT"
PARTIAL_MARKER = "partial.json"
STAMP_FILE = "shard.json"
RUNS_FILE = "runs.jsonl"
OUTPUTS_DIR = "outputs"
FORMAT = "balladeer-remote-unit/v1"
SHARDS_RELATIVE = Path("Results") / "shards"
MERGED_RECORD = SHARDS_RELATIVE / "merged.jsonl"
REQUIREMENTS = bag_data.REPOSITORY / "requirements.txt"
SYNTHETIC = "synthetic"
#: How far below a mounted input directory a file is looked for (the runner's depth).
INPUT_SEARCH_DEPTH = 4

#: The deterministic flags of the forge checklist (`remote-parallelism`).
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

#: Fields every shard stamps and the merge requires identical: the forge's
#: `stamp-fields`, plus the partition hash, the code version and the flags.
IDENTICAL_FIELDS = ("python", "torch", "cuda", "cudnn", "accelerator", "commit",
                    "requirementsDigest", "dataChecksum", "dataPreparationVersion",
                    "partitionSha256", "codeVersion", "deterministic", "seeds")
#: Of those, the ones a stamp may never leave null (CUDA and cuDNN are null on CPU).
NON_NULL_FIELDS = ("python", "torch", "accelerator", "commit", "requirementsDigest",
                   "dataChecksum", "dataPreparationVersion", "partitionSha256",
                   "codeVersion", "deterministic", "seeds")

#: What a unit is given from the product, and what of its writes it keeps.
SEED_PATTERNS = ("Models/*", "Results/hpo/*", "Results/separability/*", "Results/*/runs/*")
OUTPUT_PATTERNS = ("Models/*", "Results/hpo/*", "Results/*/runs/*")
APPEND_PATTERNS = ("Results/*/runs/*.jsonl",)


class UnitError(RuntimeError):
    """A unit cannot be selected or run as asked."""


class InputUnavailable(UnitError):
    """A declared input is not where the session says, or is not the recorded data."""


class MergeError(RuntimeError):
    """The shards cannot be merged; nothing was written."""


@dataclass(frozen=True)
class Unit:
    id: str
    step: str
    transfer: str
    index: int
    phase: int | None = None
    name: str | None = None
    after: tuple[str, ...] = ()


# ------------------------------------------------------------------ enumeration


def _phase_one_id(transfer: assignment_mod.Transfer, group: str) -> str:
    return f"search.phase1.{model_store.transfer_slug(transfer)}.{model_store.name_slug(group)}"


def _phase_two_id(transfer: assignment_mod.Transfer, strategy: str) -> str:
    return (f"search.phase2.{model_store.transfer_slug(transfer)}."
            f"{model_store.name_slug(strategy)}")


def enumerate_units(config: Mapping, product: str | Path) -> list[Unit]:
    """Every unit of the campaign the config declares, in the one deterministic order."""
    cfg = campaign.check_config(config)
    names = set(cfg["transfers"])
    transfers = [t for t in assignment_mod.TRANSFERS if t.name in names]
    strategies = [s for s in model_store.STRATEGIES if s in cfg["strategies"]]
    groups = [g for g in model_store.GROUPS
              if any(search.PHASE_ONE_GROUP_OF[s] == g for s in strategies)]
    k3 = campaign.k3_plan(cfg, campaign.core_transfers(cfg, Path(product)))
    specs: list[dict] = []
    for t in transfers:
        specs += [dict(id=_phase_one_id(t, g), step="search", transfer=t.name, phase=1, name=g)
                  for g in groups]
    for t in transfers:
        specs += [dict(id=_phase_two_id(t, s), step="search", transfer=t.name, phase=2, name=s,
                       after=(_phase_one_id(t, search.PHASE_ONE_GROUP_OF[s]),))
                  for s in strategies]
    for t in transfers:
        specs.append(dict(id=f"final.{model_store.transfer_slug(t)}", step="final",
                          transfer=t.name,
                          after=tuple(_phase_one_id(t, g) for g in groups)
                          + tuple(_phase_two_id(t, s) for s in strategies)))
    for t in transfers:
        if t.name in k3:
            slug = model_store.transfer_slug(t)
            specs.append(dict(id=f"k3.{slug}", step="k3", transfer=t.name,
                              after=(f"final.{slug}",)))
    return [Unit(index=i, **spec) for i, spec in enumerate(specs)]


def select_units(units: Sequence[Unit], requested: Sequence[str] | None = None, *,
                 environ: Mapping[str, str] | None = None) -> list[Unit]:
    """The units to run: `requested`, else `FORGE_RUN_UNITS`, else every unit."""
    environ = os.environ if environ is None else environ
    if requested is None:
        raw = environ.get(UNITS_ENV)
        if raw is None:
            return list(units)
        requested = json.loads(raw)
        if not isinstance(requested, list) or not all(isinstance(i, str) for i in requested):
            raise UnitError(f"{UNITS_ENV} must be a JSON list of unit ids; got {raw!r}")
    by_id = {u.id: u for u in units}
    unknown = [i for i in requested if i not in by_id]
    if unknown:
        raise UnitError(f"{unknown} is not a unit of this campaign; the units are "
                        f"{list(by_id)}")
    return [by_id[i] for i in dict.fromkeys(requested)]


def _check_ready(chosen: Sequence[Unit], done: set[str]) -> None:
    available = set(done)
    for unit in chosen:
        missing = [d for d in unit.after if d not in available]
        if missing:
            raise UnitError(f"unit {unit.id} needs {missing} first: complete in the shards "
                            f"root, merged into the product, or earlier in this run")
        available.add(unit.id)


# ------------------------------------------------------------------ inputs and identity


def _find(root: Path, name: str) -> Path:
    for depth in range(INPUT_SEARCH_DEPTH + 1):
        found = sorted(p for p in root.glob("/".join(["*"] * depth + [name])) if p.is_file())
        if found:
            return found[0]
    raise InputUnavailable(f"no {name} within {INPUT_SEARCH_DEPTH} levels of {root}")


def resolve_inputs(config: Mapping, *, environ: Mapping[str, str] | None = None) -> Mapping:
    """The config with `processedRoot` and `partitionPath` read from the mounted inputs."""
    environ = os.environ if environ is None else environ
    raw = environ.get(INPUTS_ENV)
    refs = config.get("remoteInputs") or {}
    if raw is None or not refs:
        return config
    mounted = json.loads(raw)
    out = dict(config)
    for role, key in (("processed", "processedRoot"), ("partition", "partitionPath")):
        ref = refs.get(role)
        if ref is None:
            continue
        if ref not in mounted:
            raise InputUnavailable(f"the {role} input {ref} is not among the declared inputs "
                                   f"{sorted(mounted)}")
        if mounted[ref] is None:
            raise InputUnavailable(f"the {role} input {ref} was not mounted ({INPUTS_ENV} maps "
                                   f"it to null) and this target has no download for it")
        root = Path(mounted[ref])
        if role == "processed":
            out[key] = str(_find(root, "manifest.csv").parent)
        else:
            name = Path(config.get("partitionPath") or partition.PARTITION_RELATIVE).name
            out[key] = str(_find(root, name))
    return out


def data_identity(cfg: Mapping) -> dict:
    """The data checksum, the preparation version and the partition hash a unit reads."""
    processed = Path(cfg.get("processedRoot")
                     or bag_data.REPOSITORY / bag_data.PROCESSED_RELATIVE)
    checksum = preprocess.file_sha256(processed / "manifest.csv")
    if cfg.get("synthetic"):
        version = SYNTHETIC
    else:
        recorded_path = bag_data.REPOSITORY / preprocess.SHA_MANIFEST_RELATIVE
        if not recorded_path.is_file():
            raise InputUnavailable(f"no processed-data manifest at {recorded_path}")
        import pandas as pd
        recorded = pd.read_csv(recorded_path).set_index("file")["sha256"].to_dict()
        if recorded.get("manifest.csv") != checksum:
            raise InputUnavailable(
                f"{processed / 'manifest.csv'} has sha256 {checksum}; the repository's "
                f"{preprocess.SHA_MANIFEST_RELATIVE} records {recorded.get('manifest.csv')}")
        version = preprocess.file_sha256(recorded_path)
    path = Path(cfg.get("partitionPath") or bag_data.default_partition_path())
    return {"dataChecksum": checksum, "dataPreparationVersion": version,
            "partitionSha256": partition.load_partition(path)[partition.HASH_FIELD]}


# ------------------------------------------------------------------ environment


def set_deterministic() -> dict:
    """The forge checklist's deterministic torch flags, set process-wide; their state."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"useDeterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnnBenchmark": torch.backends.cudnn.benchmark,
            "cudnnDeterministic": torch.backends.cudnn.deterministic,
            "cublasWorkspaceConfig": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def source_commit(environ: Mapping[str, str] | None = None) -> str | None:
    """`git rev-parse HEAD` of the repository, else `FORGE_CLONE_COMMIT` (a code bundle)."""
    environ = os.environ if environ is None else environ
    try:
        out = subprocess.run(["git", "-C", str(bag_data.REPOSITORY), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=60, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return environ.get(CLONE_COMMIT_ENV) or None


def _accelerator(device: str) -> str:
    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.get_device_name(d)
    return d.type


def environment_stamp(cfg: Mapping, deterministic: Mapping, identity: Mapping,
                      environ: Mapping[str, str] | None = None) -> dict:
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": cudnn,
            "accelerator": _accelerator(cfg["device"]), "commit": source_commit(environ),
            "requirementsDigest": preprocess.file_sha256(REQUIREMENTS),
            "codeVersion": model_store.code_version(), "deterministic": dict(deterministic),
            "seeds": {"search": search.SEARCH_SEED, "report": list(search.REPORT_SEEDS)},
            **identity}


# ------------------------------------------------------------------ files


def _matches(rel: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def _files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file()}


def _sha(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read() if limit is None else handle.read(limit))
    return digest.hexdigest()


def _read_stamp(folder: Path) -> dict:
    return json.loads((folder / STAMP_FILE).read_text(encoding="utf-8"))


def completed_units(shards_root: Path) -> dict[str, Path]:
    """Unit id -> its folder, for every folder directly under `shards_root` with a stamp."""
    found = {}
    for path in sorted(Path(shards_root).glob(f"*/{STAMP_FILE}")):
        stamp = json.loads(path.read_text(encoding="utf-8"))
        unit = stamp.get("completion", {}).get("unit")
        if stamp.get("completion", {}).get("complete") is True and path.parent.name == unit:
            found[unit] = path.parent
    return found


def merged_units(product: Path) -> set[str]:
    path = Path(product) / MERGED_RECORD
    if not path.exists():
        return set()
    return {json.loads(x)["unit"] for x in path.read_text(encoding="utf-8").splitlines() if x}


def _apply(folder: Path, stamp: Mapping, dest: Path, *, tolerate_identical: bool) -> None:
    for entry in stamp["completion"]["outputs"]:
        source, target = folder / OUTPUTS_DIR / entry["path"], dest / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry["mode"] == "append":
            with target.open("ab") as handle:
                handle.write(source.read_bytes())
        elif target.exists():
            if not (tolerate_identical and _sha(target) == entry["sha256"]):
                raise UnitError(f"{target} exists and differs from unit "
                                f"{stamp['completion']['unit']}'s output")
        else:
            shutil.copyfile(source, target)


def _seed(unit_product: Path, product: Path, overlay: Sequence[tuple[Path, dict]]) -> None:
    for rel, path in _files(product).items():
        if _matches(rel, SEED_PATTERNS) and not rel.startswith(SHARDS_RELATIVE.as_posix()):
            target = unit_product / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for folder, stamp in overlay:
        _apply(folder, stamp, unit_product, tolerate_identical=True)


def _diff(unit: Unit, root: Path, before: Mapping[str, tuple[int, str]]):
    after = _files(root)
    removed = sorted(set(before) - set(after))
    if removed:
        raise UnitError(f"unit {unit.id} removed files it was given: {removed}")
    outputs, discarded = [], []
    for rel, path in after.items():
        if rel not in before:
            if _matches(rel, OUTPUT_PATTERNS):
                outputs.append({"path": rel, "mode": "new", "offset": 0})
            else:
                discarded.append(rel)
            continue
        size, digest = before[rel]
        if _sha(path) == digest:
            continue
        if (_matches(rel, APPEND_PATTERNS) and path.stat().st_size > size
                and _sha(path, size) == digest):
            outputs.append({"path": rel, "mode": "append", "offset": size})
            continue
        raise UnitError(f"unit {unit.id} rewrote {rel}, a file it was given")
    return outputs, discarded


def _write_shard(unit: Unit, root: Path, outputs: Sequence[dict], discarded: Sequence[str],
                 environment: Mapping, result: object, shards_root: Path) -> Path:
    final = shards_root / unit.id
    staging = shards_root / f".{unit.id}.writing"
    for stale in (staging, final):
        if stale.exists():
            shutil.rmtree(stale)
    recorded = []
    for entry in outputs:
        source, target = root / entry["path"], staging / OUTPUTS_DIR / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as handle:
            handle.seek(entry["offset"])
            target.write_bytes(handle.read())
        recorded.append({"path": entry["path"], "mode": entry["mode"],
                         "bytes": target.stat().st_size, "sha256": _sha(target)})
    staging.mkdir(parents=True, exist_ok=True)
    (staging / RUNS_FILE).write_text(json.dumps(
        {"unit": unit.id, "step": unit.step, "transfer": unit.transfer, "complete": True},
        ensure_ascii=False) + "\n", encoding="utf-8")
    stamp = {"format": FORMAT, "environment": dict(environment),
             "completion": {"unit": unit.id, "index": unit.index, "step": unit.step,
                            "transfer": unit.transfer, "phase": unit.phase, "name": unit.name,
                            "after": list(unit.after), "complete": True,
                            "outputs": recorded, "discarded": list(discarded),
                            "result": results._plain(result)}}
    (staging / STAMP_FILE).write_text(json.dumps(stamp, indent=1, ensure_ascii=False,
                                                 default=str), encoding="utf-8")
    staging.rename(final)
    return final


# ------------------------------------------------------------------ running


def execute_unit(unit: Unit, cfg: Mapping, product: Path,
                 bags_of: Callable[[], bag_data.BagSet]) -> object:
    """One unit through the campaign's own functions, writing into `product`."""
    bags = bags_of()
    transfer = {t.name: t for t in assignment_mod.TRANSFERS}[unit.transfer]
    if unit.step != "search":
        return campaign.RUNNERS[unit.step]({**cfg, "transfers": [unit.transfer]}, product, bags)
    campaign.register(cfg["strategies"])
    data = search.search_data(transfer, bags)
    if unit.phase == 1:
        study = campaign.phase_one_study(unit.name, data, cfg, product)
        for r in sorted(data.rotations):
            campaign._phase_one_checkpoint(unit.name, data, r, search.SEARCH_SEED,
                                           study.best_params, cfg, product)
        return {"bestValue": study.best_value}
    group = search.PHASE_ONE_GROUP_OF[unit.name]
    records = {r: model_store.load(model_store.model_path(
        campaign._models(product), transfer, r, search.SEARCH_SEED, 1, group),
        partition_path=campaign._partition_path(cfg)) for r in sorted(data.rotations)}
    study = campaign.phase_two_study(unit.name, data, records, cfg, product)
    return {"bestValue": study.best_value}


def run_units(config: Mapping, product: str | Path, *, units: Sequence[str] | None = None,
              shards_root: str | Path | None = None, workdir: str | Path | None = None,
              deadline: float | None = None, estimate: Callable[[Unit], float] | None = None,
              execute: Callable | None = None,
              environ: Mapping[str, str] | None = None) -> dict:
    """Run the selected units in order, each into its own shard folder, until the deadline."""
    environ = os.environ if environ is None else environ
    product = Path(product)
    cfg = resolve_inputs(campaign.check_config(config), environ=environ)
    shards_root = Path(shards_root) if shards_root else product / SHARDS_RELATIVE
    workdir = Path(workdir) if workdir else Path.cwd()
    chosen = select_units(enumerate_units(cfg, product), units, environ=environ)
    deterministic = set_deterministic()
    merged = merged_units(product)
    done = set(completed_units(shards_root)) | merged
    _check_ready(chosen, done)
    if deadline is None:
        deadline = float(environ.get(DEADLINE_ENV, "inf"))
    estimate = estimate or (lambda unit: 0.0)
    execute = execute or execute_unit
    environment = environment_stamp(cfg, deterministic, data_identity(cfg), environ)
    marker = workdir / PARTIAL_MARKER
    marker.unlink(missing_ok=True)
    cache: dict[str, bag_data.BagSet] = {}

    def bags_of() -> bag_data.BagSet:
        if "bags" not in cache:
            cache["bags"] = campaign.load_bags(cfg)
        return cache["bags"]

    ran, skipped = [], []
    for position, unit in enumerate(chosen):
        if unit.id in done:
            skipped.append(unit.id)
            continue
        if time.time() + estimate(unit) > deadline:
            remaining = [u.id for u in chosen[position:] if u.id not in done]
            workdir.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"remaining": remaining}), encoding="utf-8")
            return {"ran": ran, "skipped": skipped, "remaining": remaining,
                    "partial": str(marker)}
        complete = completed_units(shards_root)
        overlay = sorted(((folder, _read_stamp(folder)) for uid, folder in complete.items()
                          if uid not in merged),
                         key=lambda item: item[1]["completion"]["index"])
        with tempfile.TemporaryDirectory(prefix="remote-unit-") as scratch:
            root = Path(scratch) / "product"
            root.mkdir()
            _seed(root, product, overlay)
            before = {rel: (p.stat().st_size, _sha(p)) for rel, p in _files(root).items()}
            result = execute(unit, cfg, root, bags_of)
            outputs, discarded = _diff(unit, root, before)
            shards_root.mkdir(parents=True, exist_ok=True)
            _write_shard(unit, root, outputs, discarded, environment, result, shards_root)
        done.add(unit.id)
        ran.append(unit.id)
    return {"ran": ran, "skipped": skipped, "remaining": [], "partial": None}


# ------------------------------------------------------------------ merge


def merge(product: str | Path, roots: Sequence[str | Path] | None = None, *,
          expected: Mapping | None = None) -> dict:
    """Validate every shard under `roots` against the others, then write the product."""
    product = Path(product)
    roots = [Path(r) for r in roots] if roots else [product / SHARDS_RELATIVE]
    shards = []
    for root in roots:
        for path in sorted(root.rglob(STAMP_FILE)) if root.is_dir() else []:
            stamp = json.loads(path.read_text(encoding="utf-8"))
            unit = stamp.get("completion", {}).get("unit")
            if stamp.get("format") != FORMAT or stamp["completion"].get("complete") is not True:
                raise MergeError(f"{path}: not a complete {FORMAT} stamp")
            if path.parent.name != unit:
                raise MergeError(f"{path.parent} holds unit {unit}: a folder is named after "
                                 f"its unit; remove a staging or renamed folder")
            shards.append((path.parent, stamp))
    if not shards:
        raise MergeError(f"no {STAMP_FILE} under {[str(r) for r in roots]}")
    counts = Counter(s["completion"]["unit"] for _, s in shards)
    duplicates = {u: [str(f) for f, s in shards if s["completion"]["unit"] == u]
                  for u, n in counts.items() if n > 1}
    if duplicates:
        raise MergeError(f"duplicate units, each must come back exactly once: {duplicates}")
    again = sorted(set(counts) & merged_units(product))
    if again:
        raise MergeError(f"units already merged into {product}: {again}")
    for folder, stamp in shards:
        env = stamp.get("environment", {})
        missing = [f for f in IDENTICAL_FIELDS if f not in env]
        missing += [f for f in NON_NULL_FIELDS if f in env and env[f] is None]
        if missing:
            raise MergeError(f"{folder}: environment fields missing or null: {missing}")
    for field in IDENTICAL_FIELDS:
        seen: dict[str, list[str]] = {}
        for _, stamp in shards:
            seen.setdefault(json.dumps(stamp["environment"][field], sort_keys=True),
                            []).append(stamp["completion"]["unit"])
        if len(seen) > 1:
            raise MergeError(f"shards disagree on {field}: {seen}")
    for field, value in (expected or {}).items():
        actual = shards[0][1]["environment"].get(field)
        if actual != value:
            raise MergeError(f"the shards' {field} is {actual!r}; expected {value!r}")
    for folder, stamp in shards:
        for entry in stamp["completion"]["outputs"]:
            path = folder / OUTPUTS_DIR / entry["path"]
            if not path.is_file() or _sha(path) != entry["sha256"]:
                raise MergeError(f"{folder}: output {entry['path']} does not match its "
                                 f"recorded sha256")
            if entry["mode"] == "new" and (product / entry["path"]).exists():
                raise MergeError(f"{folder}: output {entry['path']} already exists in "
                                 f"{product}")
    shards.sort(key=lambda item: item[1]["completion"]["index"])
    for folder, stamp in shards:
        _apply(folder, stamp, product, tolerate_identical=False)
    record = product / MERGED_RECORD
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open("a", encoding="utf-8") as handle:
        for folder, stamp in shards:
            handle.write(json.dumps({"unit": stamp["completion"]["unit"],
                                     "commit": stamp["environment"]["commit"],
                                     "shard": str(folder)}, ensure_ascii=False) + "\n")
    return {"merged": [s["completion"]["unit"] for _, s in shards],
            "files": sum(len(s["completion"]["outputs"]) for _, s in shards)}


# ------------------------------------------------------------------ command line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m MIL_CREDA_EEG.remote_units",
                                     description="The campaign's remote units (plan v43).")
    parser.add_argument("verb", choices=("list", "run", "merge"))
    parser.add_argument("--config", required=True, help="campaign config (JSON)")
    parser.add_argument("--product", default=str(campaign.DEFAULT_PRODUCT))
    parser.add_argument("--unit", action="append", help="run: a unit id (repeatable); "
                        f"default {UNITS_ENV}, else every unit")
    parser.add_argument("--shards", action="append",
                        help="run: the shards root; merge: a root to merge (repeatable); "
                             "default <product>/Results/shards")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        if args.verb == "list":
            outcome = [u.__dict__ for u in enumerate_units(config, args.product)]
        elif args.verb == "run":
            outcome = run_units(config, args.product, units=args.unit,
                                shards_root=args.shards[0] if args.shards else None)
        else:
            cfg = campaign.check_config(config)
            path = Path(cfg.get("partitionPath") or bag_data.default_partition_path())
            current = partition.load_partition(path)[partition.HASH_FIELD]
            outcome = merge(args.product, args.shards, expected={"partitionSha256": current})
    except (UnitError, MergeError, campaign.CampaignError, search.SearchError) as error:
        print(json.dumps({"status": "refused", "error": type(error).__name__,
                          "detail": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(results._plain(outcome), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
