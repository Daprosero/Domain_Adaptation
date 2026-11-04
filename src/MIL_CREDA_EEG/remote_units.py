"""The campaign's units of work, run one remote session at a time and merged back.

Experiments plan v43, Secs. 4.2, 4.6, 7.3, 7.6 and 7.7, under the forge's remote
contract (`remote-execution`: the session-chain rule, job inputs and the shard
stamp; `remote-parallelism`: the reproducibility checklist). Code review blocker
B5: the campaign had no shard-aware contract; the second review's R1-R4 (an active
unit the deadline could not stop, shards checked only against each other, signals
outside the data identity, no rehearsal stamp) and its device and pin items.

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
each in its own working product, `<shards>/.progress/<unit id>/product`, seeded once
with the product's stored models, studies and runs and with every completed unit
not merged yet. It is not temporary: a study's SQLite and each rotation's models
and predictions stay there as they are written, so the shards root is the carry
path a continued session needs. Before the first unit, the declared device is
placed (`device_placement`: a remote session must declare it, and a declared CUDA
device that does not land on CUDA is refused), and `set_deterministic` sets the
deterministic torch flags.

**Deadline.** No unit starts without a duration estimate: the caller's measured
`estimate`, else the timing probe's `Results/timing.json` (`timing_estimate`);
without either the run is refused, naming what to measure. `FORGE_SESSION_DEADLINE`
(epoch seconds) is read and compared with the estimate of the next step before each
unit, between trials (search, `search.run_phase_*_study(should_stop=...)`), before
each phase-1 checkpoint and between rotations (final, k3). When the next step may not
finish before it, the unit stops cleanly and `partial.json` is written in the working
directory as `{"remaining": [ids], "progress": {id: ...}}` (completed trials, or
completed rotations); a later call with the same unit continues its working product
(a study of the same identity resumes). A unit already complete is skipped, and a
unit whose `after` ids are neither complete, merged nor earlier in the run is
refused before anything runs.

**Currency.** Every completed shard a run skips or resumes from, every working
product it continues and every shard a merge reads is compared with the CURRENT
campaign's identity (`campaign_identity`: commit, requirements digest, code version,
data checksum and preparation version, partition hash, plan revision, configuration
digest, seeds, deterministic flags and the torch release); a mismatch is refused,
naming the field.

**Inputs** (`resolve_inputs`). `config["remoteInputs"]` names the dataset ref of
the processed data and of the partition; `FORGE_INPUT_DATASETS` maps each ref to
its mounted directory, where the processed `manifest.csv` and the partition file
are looked for. A ref mapped to null is refused: this target has no download. Before
anything is loaded, every consumed processed file (`manifest.csv` and each kept
record's NPZ) is verified against `MIL-CREDA/Manifests/processed_sha256.csv`, once
per session (`verify_processed`).

**Smoke** (`smoke`). One unit, the first by default, through `run_units` on the
mounted data; under the runner it also writes the rehearsal `shard.json` into its
working directory (`mode`, `evidence.commit`, `evidence.sourcesDigest`, the actual
`environment.device` and the stamp fields), apart from the per-unit stamps.

**Outputs.** Each unit writes `<shards>/<unit id>/` (default `Results/shards/`):
`outputs/` with the files it created under `Models/`, `Results/hpo/` and
`Results/*/runs/` (the lines it appended to a runs file, for those), `runs.jsonl`
with one record naming the unit, and `shard.json` last, holding the environment
stamp (python, torch, CUDA, cuDNN, the actual device and accelerator, commit,
requirements digest, code version, deterministic flags, the plan's seeds, the data
checksum and preparation version, the partition hash, the plan revision and the
configuration digest) and the completion stamp (unit, every output with
its sha256, the derived files it discarded). Derived files (tables, figures,
summaries) are discarded: the steps after the merge rewrite them over every
transfer.

**Merge** (`merge`). Every `shard.json` under the given roots is checked before
anything is written: a duplicate unit, a unit already merged, an environment
field missing or null, shards that disagree on any field of `IDENTICAL_FIELDS`, a
shard not of the current campaign, an output whose bytes no longer match its sha256,
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
import contextlib
import fnmatch
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, campaign, model_store, partition, preprocess, results, search

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v44.md",
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
        "a_unit_without_a_measured_duration_estimate_is_refused_naming_what_to_measure",
        "the_deadline_stops_a_study_between_trials_and_the_rerun_completes_it",
        "the_deadline_stops_final_between_rotations_and_the_rerun_completes_it",
        "the_deadline_before_a_unit_with_no_time_left_writes_partial_and_runs_nothing",
        "merge_refuses_a_shard_whose_stamp_is_not_the_current_campaigns",
        "resume_refuses_a_completed_shard_whose_stamp_is_not_the_current_campaigns",
        "every_consumed_processed_file_is_verified_against_the_recorded_sha256",
        "the_processed_files_are_verified_once_per_session",
        "a_run_on_tampered_signals_is_refused_before_any_unit",
        "the_smoke_writes_the_rehearsal_shard_json_in_its_working_directory",
        "a_remote_run_must_declare_its_device",
        "a_declared_cuda_device_that_is_not_placed_on_cuda_is_refused_before_training",
        "the_stamp_records_the_actual_device_placement",
        "requirements_pin_every_installed_runtime_dependency_to_its_installed_version",
        "every_pinned_requirement_equals_its_installed_version",
    ],
}

#: The forge runner's variables (`remote-execution`, assets/runner_invoke.py).
UNITS_ENV = "FORGE_RUN_UNITS"
DEADLINE_ENV = "FORGE_SESSION_DEADLINE"
INPUTS_ENV = "FORGE_INPUT_DATASETS"
CLONE_COMMIT_ENV = "FORGE_CLONE_COMMIT"
RUN_MODE_ENV = "FORGE_RUN_MODE"
CLONE_ROOT_ENV = "FORGE_CLONE_ROOT"
#: Any of these in the environment means the forge's runner started this session.
REMOTE_ENVS = (UNITS_ENV, DEADLINE_ENV, INPUTS_ENV, RUN_MODE_ENV, CLONE_ROOT_ENV)
PARTIAL_MARKER = "partial.json"
STAMP_FILE = "shard.json"
RUNS_FILE = "runs.jsonl"
OUTPUTS_DIR = "outputs"
FORMAT = "balladeer-remote-unit/v1"
SHARDS_RELATIVE = Path("Results") / "shards"
MERGED_RECORD = SHARDS_RELATIVE / "merged.jsonl"
#: Under the shards root: one working product per unit not finished yet (a carry path).
PROGRESS_DIR = ".progress"
#: The measured duration record (`remote-parallelism`'s timing probe), under the product.
TIMING_RELATIVE = Path("Results") / "timing.json"
#: The recorded sha256 of every processed file (`preprocess`), read at call time.
SHA_MANIFEST = bag_data.REPOSITORY / preprocess.SHA_MANIFEST_RELATIVE
REQUIREMENTS = bag_data.REPOSITORY / "requirements.txt"
SYNTHETIC = "synthetic"
#: How far below a mounted input directory a file is looked for (the runner's depth).
INPUT_SEARCH_DEPTH = 4

#: The deterministic flags of the forge checklist (`remote-parallelism`).
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
#: The state `set_deterministic` leaves, which every shard must have run under.
DETERMINISTIC = {"useDeterministicAlgorithms": True, "cudnnBenchmark": False,
                 "cudnnDeterministic": True, "cublasWorkspaceConfig": CUBLAS_WORKSPACE_CONFIG}

#: Fields every shard stamps and the merge requires identical: the forge's
#: `stamp-fields`, plus the partition hash, the code version, the flags, the plan
#: revision and the configuration digest.
IDENTICAL_FIELDS = ("python", "torch", "cuda", "cudnn", "accelerator", "commit",
                    "requirementsDigest", "dataChecksum", "dataPreparationVersion",
                    "partitionSha256", "codeVersion", "deterministic", "seeds", "revision",
                    "configDigest")
#: Of those, the ones a stamp may never leave null (CUDA and cuDNN are null on CPU).
NON_NULL_FIELDS = ("python", "torch", "accelerator", "commit", "requirementsDigest",
                   "dataChecksum", "dataPreparationVersion", "partitionSha256",
                   "codeVersion", "deterministic", "seeds", "revision", "configDigest")
#: Fields compared with the CURRENT campaign (`campaign_identity`), not only between
#: shards. `torch` is compared on its release (`2.13.0` of `2.13.0+cu126`). Python,
#: CUDA, cuDNN and the accelerator describe the worker, not the campaign: shards must
#: agree on them, and the merging machine may differ.
CAMPAIGN_FIELDS = ("commit", "requirementsDigest", "codeVersion", "dataChecksum",
                   "dataPreparationVersion", "partitionSha256", "revision", "configDigest",
                   "seeds", "deterministic", "torch")
#: Config keys that locate the inputs or name the device, left out of `configDigest`:
#: a mounted path differs per session and the device is stamped on its own.
CONFIG_DIGEST_EXCLUDES = ("processedRoot", "partitionPath", "device")

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


class DeadlineReached(Exception):
    """Raised inside a unit at a trial or rotation boundary; `progress` says how far."""

    def __init__(self, progress: Mapping):
        super().__init__(f"the session deadline is reached: {dict(progress)}")
        self.progress = dict(progress)


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


#: (processed root, sha256 of the record) pairs already verified in this session.
_VERIFIED: set[tuple[str, str]] = set()


def verify_processed(processed: str | Path, sha_manifest: str | Path) -> None:
    """Every processed file a run consumes -- `manifest.csv` and the NPZ of each kept
    record -- has the sha256 the record lists; checked once per session and record."""
    import pandas as pd
    processed, sha_manifest = Path(processed), Path(sha_manifest)
    key = (str(processed.resolve()), preprocess.file_sha256(sha_manifest))
    if key in _VERIFIED:
        return
    recorded = pd.read_csv(sha_manifest).set_index("file")["sha256"].to_dict()

    def check(rel: str) -> None:
        if rel not in recorded:
            raise InputUnavailable(f"{rel} is consumed but not recorded in {sha_manifest}")
        path = processed / rel
        if not path.is_file():
            raise InputUnavailable(f"{path} ({rel}) is recorded in {sha_manifest} but absent")
        actual = preprocess.file_sha256(path)
        if actual != recorded[rel]:
            raise InputUnavailable(f"{path} ({rel}) has sha256 {actual}; {sha_manifest} "
                                   f"records {recorded[rel]}")

    check("manifest.csv")
    manifest = pd.read_csv(processed / "manifest.csv", low_memory=False)
    kept = manifest[~manifest["excluded"].astype(bool)]
    for rel in dict.fromkeys(str(x) for x in kept["npz"]):
        check(rel)
    _VERIFIED.add(key)


def data_identity(cfg: Mapping) -> dict:
    """The data checksum, the preparation version and the partition hash a unit reads.

    Outside a synthetic set every consumed processed file is first verified against
    `SHA_MANIFEST` (`verify_processed`), so the preparation version -- that record's
    own sha256 -- vouches for the signals, not only for the manifest."""
    processed = Path(cfg.get("processedRoot")
                     or bag_data.REPOSITORY / bag_data.PROCESSED_RELATIVE)
    if cfg.get("synthetic"):
        version = SYNTHETIC
    else:
        recorded_path = Path(SHA_MANIFEST)
        if not recorded_path.is_file():
            raise InputUnavailable(f"no processed-data manifest at {recorded_path}")
        verify_processed(processed, recorded_path)
        version = preprocess.file_sha256(recorded_path)
    checksum = preprocess.file_sha256(processed / "manifest.csv")
    path = Path(cfg.get("partitionPath") or bag_data.default_partition_path())
    return {"dataChecksum": checksum, "dataPreparationVersion": version,
            "partitionSha256": partition.load_partition(path)[partition.HASH_FIELD]}


def config_digest(config: Mapping) -> str:
    """sha256 of the checked campaign config, without `CONFIG_DIGEST_EXCLUDES`."""
    cfg = {k: v for k, v in campaign.check_config(config).items()
           if k not in CONFIG_DIGEST_EXCLUDES}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def campaign_identity(cfg: Mapping, environ: Mapping[str, str] | None = None) -> dict:
    """What a shard of the CURRENT campaign is stamped with: `CAMPAIGN_FIELDS`, the
    fields `merge` and a resumed run compare."""
    identity = {"commit": source_commit(environ),
            "requirementsDigest": preprocess.file_sha256(REQUIREMENTS),
            "codeVersion": model_store.code_version(), **data_identity(cfg),
            "revision": campaign.REVISION, "configDigest": config_digest(cfg),
            "seeds": {"search": search.SEARCH_SEED, "report": list(search.REPORT_SEEDS)},
            "deterministic": dict(DETERMINISTIC), "torch": torch.__version__}
    return {field: identity[field] for field in CAMPAIGN_FIELDS}


def identity_mismatches(stamped: Mapping, expected: Mapping) -> list[str]:
    """`field: stamped X, current Y` for every field of `expected` the stamp differs in."""
    out = []
    for field, want in expected.items():
        have = stamped.get(field)
        if field == "torch":
            have = have.split("+")[0] if isinstance(have, str) else have
            want = want.split("+")[0]
        if json.dumps(have, sort_keys=True) != json.dumps(want, sort_keys=True):
            out.append(f"{field}: stamped {have!r}, current {want!r}")
    return out


# ------------------------------------------------------------------ duration estimate


def _measure_first(path: Path) -> str:
    return (f"measure one epoch with the timing probe (remote-parallelism "
            f"scripts/timing_probe.py --out {path}); the pilot provides it. Or pass an "
            f"explicit measured estimate")


def timing_estimate(path: str | Path, config: Mapping,
                    product: str | Path) -> Callable[[Unit], float]:
    """Seconds of a unit's next step from a measured schema-1 timing record.

    A step is one trial (search) or one rotation (final, k3). The epoch time is the
    largest `secondsPerUnit` the record measured (`unit` `epoch`); the step's epochs
    are their caps, so the estimate bounds the step from above: a trial trains every
    rotation for `epochs1` (phase 1) or `epochs2` (phase 2) epochs, and so does a
    phase-1 checkpoint pass; a final rotation trains each phase-1 group and the upper
    bound for `epochs1` and each strategy twice (primary and label-free) for `epochs2`;
    a K3 rotation trains each planned variant for `epochs1 + epochs2`.
    """
    path, product = Path(path), Path(product)
    if not path.is_file():
        raise UnitError(f"no duration estimate for the units: no {path}; "
                        + _measure_first(path))
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise UnitError(f"{path} is not JSON ({error}); " + _measure_first(path)) from error
    rows = [m for m in (record.get("measurements") or [] if isinstance(record, dict) else [])
            if isinstance(m, dict) and m.get("unit") == "epoch"
            and isinstance(m.get("secondsPerUnit"), (int, float))]
    if not isinstance(record, dict) or record.get("schema") != 1 or not rows:
        raise UnitError(f"{path} holds no measured epoch (a schema-1 record with "
                        f"measurements of unit epoch and their secondsPerUnit); "
                        + _measure_first(path))
    per_epoch = max(float(m["secondsPerUnit"]) for m in rows)
    cfg = campaign.check_config(config)
    schedule = cfg["schedule"]
    rotations = len(partition.load_partition(campaign._partition_path(cfg))["rotations"])
    strategies = list(cfg["strategies"])
    groups = {search.PHASE_ONE_GROUP_OF[s] for s in strategies}

    def epochs(unit: Unit) -> int:
        if unit.step == "search":
            return rotations * schedule["epochs1" if unit.phase == 1 else "epochs2"]
        if unit.step == "final":
            return ((len(groups) + 1) * schedule["epochs1"]
                    + 2 * len(strategies) * schedule["epochs2"])
        plan = campaign.k3_plan(cfg, campaign.core_transfers(cfg, product))[unit.transfer]
        return (sum(len(v) for v in plan.values())
                * (schedule["epochs1"] + schedule["epochs2"]))

    return lambda unit: epochs(unit) * per_epoch


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


def device_placement(config: Mapping, environ: Mapping[str, str] | None = None) -> dict:
    """The declared device and where a tensor placed on it actually lands.

    A session the forge's runner started (`REMOTE_ENVS`) must declare `device` in its
    config. A declared device a tensor cannot be placed on (a CUDA device on a session
    without one) is refused here, before any training; the stamp records the actual
    placement."""
    environ = os.environ if environ is None else environ
    if any(k in environ for k in REMOTE_ENVS) and config.get("device") is None:
        raise UnitError("a remote run must declare its device in the campaign config "
                        "(\"device\": \"cuda\" or \"cpu\"); the campaign's CPU default "
                        "applies only to a local run")
    declared = str(config.get("device") or "cpu")
    try:
        actual, reason = torch.empty(0, device=declared).device, ""
    except (RuntimeError, AssertionError) as error:
        actual, reason = None, str(error)
    if actual is None:     # torch places on the declared CUDA device or raises
        raise UnitError(f"the config declares device {declared!r} but no tensor can be "
                        f"placed there ({reason}): refused before any training")
    if actual.type == "cuda":
        name = torch.cuda.get_device_name(actual)
    else:
        name = platform.processor() or platform.machine() or actual.type
    return {"declared": declared, "kind": actual.type, "name": name}


def environment_stamp(deterministic: Mapping, expected: Mapping, placement: Mapping) -> dict:
    """The worker (python, torch, CUDA, cuDNN, the actual device and accelerator), the
    deterministic flags in force, and the campaign identity the unit ran under."""
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    accelerator = placement["name"] if placement["kind"] == "cuda" else placement["kind"]
    return {**expected, "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": cudnn, "accelerator": accelerator,
            "device": dict(placement), "deterministic": dict(deterministic)}


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


@contextlib.contextmanager
def _rotation_boundaries(unit: Unit, rotations: Sequence[int], seeds: int,
                         guard: Callable[[], bool]):
    """Ask `guard` after each rotation of a final or k3 step; stop between two of them.

    `campaign.run_final` and `campaign.run_k3` take no hook, so the boundary is read
    off the call that ends each rotation and report seed: `results.append_predictions`
    with the label-free rule (final) or for K3 (k3). The wrapper is installed only for
    the call and removed afterwards; everything a rotation wrote is already in the
    working product, and the campaign skips stored predictions and reuses stored
    models when the unit is continued."""
    real = results.append_predictions
    total, seen = len(rotations) * seeds, [0]

    def ends_rotation(experiment: str, rule: str) -> bool:
        if unit.step == "final":
            return rule == "label-free"
        return experiment == campaign.K3

    def wrapped(root, experiment, records, *, identity, rule="primary"):
        path = real(root, experiment, records, identity=identity, rule=rule)
        if ends_rotation(experiment, rule):
            seen[0] += 1
            if seen[0] < total and guard():
                raise DeadlineReached({"completedRotations": list(rotations[:seen[0] // seeds]),
                                       "rotations": list(rotations)})
        return path

    results.append_predictions = wrapped
    try:
        yield
    finally:
        results.append_predictions = real


def _completed_trials(study) -> int:
    import optuna
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def execute_unit(unit: Unit, cfg: Mapping, product: Path,
                 bags_of: Callable[[], bag_data.BagSet],
                 guard: Callable[[], bool] | None = None) -> object:
    """One unit through the campaign's own functions, writing into `product`.

    `guard()` answers whether the next step may not finish before the deadline; it is
    asked between trials (search), before each phase-1 checkpoint and between rotations
    (final, k3), and a true answer raises `DeadlineReached` with how far the unit got.
    A search unit calls `search.run_phase_*_study` with the arguments
    `campaign.phase_*_study` passes, plus the hook those wrappers do not forward."""
    guard = guard or (lambda: False)
    bags = bags_of()
    transfer = {t.name: t for t in assignment_mod.TRANSFERS}[unit.transfer]
    if unit.step != "search":
        rotations = sorted(int(r) for r in bags.rotations)
        with _rotation_boundaries(unit, rotations, len(cfg["reportSeeds"]), guard):
            return campaign.RUNNERS[unit.step]({**cfg, "transfers": [unit.transfer]},
                                               product, bags)
    campaign.register(cfg["strategies"])
    data = search.search_data(transfer, bags)
    n_trials = cfg["nTrials"]
    common = dict(root=product, n_trials=n_trials, device=cfg["device"],
                  should_stop=lambda study: guard())
    if unit.phase == 1:
        study = search.run_phase_one_study(unit.name, data, cfg["schedule"], **common)
        if _completed_trials(study) < n_trials:
            raise DeadlineReached({"completedTrials": _completed_trials(study),
                                   "nTrials": n_trials})
        rotations = sorted(data.rotations)
        for position, r in enumerate(rotations):
            path = model_store.model_path(campaign._models(product), transfer, r,
                                          search.SEARCH_SEED, 1, unit.name)
            if not path.exists() and guard():
                raise DeadlineReached({"completedTrials": n_trials, "nTrials": n_trials,
                                       "completedCheckpoints": list(rotations[:position]),
                                       "rotations": list(rotations)})
            campaign._phase_one_checkpoint(unit.name, data, r, search.SEARCH_SEED,
                                           study.best_params, cfg, product)
        return {"bestValue": study.best_value}
    group = search.PHASE_ONE_GROUP_OF[unit.name]
    records = {r: model_store.load(model_store.model_path(
        campaign._models(product), transfer, r, search.SEARCH_SEED, 1, group),
        partition_path=campaign._partition_path(cfg)) for r in sorted(data.rotations)}
    study = search.run_phase_two_study(unit.name, data, records, cfg["schedule"], **common)
    if _completed_trials(study) < n_trials:
        raise DeadlineReached({"completedTrials": _completed_trials(study),
                               "nTrials": n_trials})
    return {"bestValue": study.best_value}


def _working_product(unit: Unit, product: Path, shards_root: Path, merged: set[str],
                     expected: Mapping) -> tuple[Path, dict]:
    """The unit's working product under `<shards>/.progress/<unit>/`, and what it was given.

    Created and seeded once (the product's stored files and every completed unit not
    merged yet); a later session continues in it, so a study's SQLite and a rotation's
    models and predictions persist as they are written. Progress stamped by another
    campaign is refused, naming what differs."""
    folder = shards_root / PROGRESS_DIR / unit.id
    root, given, stamped = folder / "product", folder / "given.json", folder / "identity.json"
    if given.is_file():
        differ = identity_mismatches(json.loads(stamped.read_text(encoding="utf-8")),
                                     expected)
        if differ:
            raise UnitError(f"{folder} holds progress of another campaign and is not "
                            f"continued: " + "; ".join(differ)
                            + "; remove it to start the unit again")
        return root, {k: tuple(v) for k, v in
                      json.loads(given.read_text(encoding="utf-8")).items()}
    if folder.exists():
        shutil.rmtree(folder)          # a seeding cut before it finished
    root.mkdir(parents=True)
    complete = completed_units(shards_root)
    overlay = sorted(((f, _read_stamp(f)) for uid, f in complete.items() if uid not in merged),
                     key=lambda item: item[1]["completion"]["index"])
    _seed(root, product, overlay)
    before = {rel: (p.stat().st_size, _sha(p)) for rel, p in _files(root).items()}
    stamped.write_text(json.dumps(dict(expected), sort_keys=True), encoding="utf-8")
    given.write_text(json.dumps(before), encoding="utf-8")       # last: seeding is done
    return root, before


def _check_completed(complete: Mapping[str, Path], merged: set[str],
                     expected: Mapping) -> None:
    for uid, folder in complete.items():
        if uid in merged:
            continue
        differ = identity_mismatches(_read_stamp(folder).get("environment", {}), expected)
        if differ:
            raise UnitError(f"{folder} was stamped by another campaign and is not skipped or "
                            f"resumed from: " + "; ".join(differ)
                            + "; remove it to run the unit again")


def run_units(config: Mapping, product: str | Path, *, units: Sequence[str] | None = None,
              shards_root: str | Path | None = None, workdir: str | Path | None = None,
              deadline: float | None = None, estimate: Callable[[Unit], float] | None = None,
              timing: str | Path | None = None, clock: Callable[[], float] = time.time,
              execute: Callable | None = None,
              environ: Mapping[str, str] | None = None) -> dict:
    """Run the selected units in order, each into its own shard folder, until the deadline.

    `estimate(unit)` is the seconds of the unit's next step (a trial or a rotation); when
    the caller passes none it is read from the measured `timing` record (default
    `<product>/Results/timing.json`), and a run without one is refused."""
    environ = os.environ if environ is None else environ
    product = Path(product)
    placement = device_placement(config, environ)
    cfg = resolve_inputs(campaign.check_config(config), environ=environ)
    shards_root = Path(shards_root) if shards_root else product / SHARDS_RELATIVE
    workdir = Path(workdir) if workdir else Path.cwd()
    chosen = select_units(enumerate_units(cfg, product), units, environ=environ)
    if estimate is None:
        estimate = timing_estimate(Path(timing) if timing else product / TIMING_RELATIVE,
                                   cfg, product)
    deterministic = set_deterministic()
    expected = campaign_identity(cfg, environ)
    merged = merged_units(product)
    complete = completed_units(shards_root)
    _check_completed(complete, merged, expected)
    done = set(complete) | merged
    _check_ready(chosen, done)
    if deadline is None:
        deadline = float(environ.get(DEADLINE_ENV, "inf"))
    execute = execute or execute_unit
    environment = environment_stamp(deterministic, expected, placement)
    marker = workdir / PARTIAL_MARKER
    marker.unlink(missing_ok=True)
    cache: dict[str, bag_data.BagSet] = {}

    def bags_of() -> bag_data.BagSet:
        if "bags" not in cache:
            cache["bags"] = campaign.load_bags(cfg)
        return cache["bags"]

    def stop(position: int, progress: Mapping) -> dict:
        remaining = [u.id for u in chosen[position:] if u.id not in done]
        workdir.mkdir(parents=True, exist_ok=True)
        payload = {"remaining": remaining, **({"progress": dict(progress)} if progress else {})}
        marker.write_text(json.dumps(payload), encoding="utf-8")
        return {"ran": ran, "skipped": skipped, "remaining": remaining,
                "partial": str(marker), "environment": environment}

    ran, skipped = [], []
    for position, unit in enumerate(chosen):
        if unit.id in done:
            skipped.append(unit.id)
            continue

        def guard(unit: Unit = unit) -> bool:
            return clock() + estimate(unit) > deadline

        if guard():
            return stop(position, {})
        root, before = _working_product(unit, product, shards_root, merged, expected)
        try:
            result = execute(unit, cfg, root, bags_of, guard=guard)
        except DeadlineReached as reached:
            return stop(position, {unit.id: reached.progress})
        outputs, discarded = _diff(unit, root, before)
        shards_root.mkdir(parents=True, exist_ok=True)
        _write_shard(unit, root, outputs, discarded, environment, result, shards_root)
        shutil.rmtree(shards_root / PROGRESS_DIR / unit.id)
        done.add(unit.id)
        ran.append(unit.id)
    return {"ran": ran, "skipped": skipped, "remaining": [], "partial": None,
            "environment": environment}


# ------------------------------------------------------------------ smoke


def smoke(config: str | Path | Mapping, product: str | Path | None = None, *,
          unit: str | None = None, shards_root: str | Path | None = None,
          workdir: str | Path | None = None, timing: str | Path | None = None,
          estimate: Callable[[Unit], float] | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
    """The rehearsal callable: one unit (default the first one) on the mounted data.

    It runs through `run_units`, under the same deadline, estimate, device and data
    checks, and leaves its per-unit stamp in the shards root as any unit does. Under the
    forge's runner (`FORGE_RUN_MODE` set) it also writes `shard.json` into its working
    directory, the rehearsal evidence `smoke record` reads: `mode`, `evidence` (`commit`,
    `sourcesDigest` -- the code version --, `codeVersion`, `unit`, `complete`),
    `environment` (the unit stamp's environment: the actual `device` `{declared, kind,
    name}`, `accelerator` and every identity field) and `runs` (the outcome)."""
    environ = os.environ if environ is None else environ
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    product = Path(product) if product else campaign.DEFAULT_PRODUCT
    workdir = Path(workdir) if workdir else Path.cwd()
    if unit is None:
        cfg = resolve_inputs(campaign.check_config(config), environ=environ)
        unit = enumerate_units(cfg, product)[0].id
    outcome = run_units(config, product, units=[unit], shards_root=shards_root,
                        workdir=workdir, timing=timing, estimate=estimate, environ=environ)
    mode = environ.get(RUN_MODE_ENV)
    if mode:
        env = outcome["environment"]
        runs = {k: v for k, v in outcome.items() if k != "environment"}
        payload = {"mode": mode,
                   "evidence": {"commit": env["commit"], "sourcesDigest": env["codeVersion"],
                                "codeVersion": env["codeVersion"], "unit": unit,
                                "complete": unit in outcome["ran"] + outcome["skipped"]},
                   "environment": env, "runs": runs}
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / STAMP_FILE).write_text(json.dumps(payload, indent=1, sort_keys=True,
                                                     ensure_ascii=False, default=str),
                                          encoding="utf-8")
    return outcome


# ------------------------------------------------------------------ merge


def merge(product: str | Path, roots: Sequence[str | Path] | None = None, *,
          config: Mapping, expected: Mapping | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
    """Validate every shard under `roots` against the others and against the CURRENT
    campaign `config` describes (`campaign_identity`), then write the product."""
    product = Path(product)
    roots = [Path(r) for r in roots] if roots else [product / SHARDS_RELATIVE]
    current = {**campaign_identity(campaign.check_config(config), environ),
               **(expected or {})}
    shards = []
    for root in roots:
        for path in sorted(root.rglob(STAMP_FILE)) if root.is_dir() else []:
            if PROGRESS_DIR in path.relative_to(root).parts:
                continue                   # an unfinished unit's working product
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
    for folder, stamp in shards:
        differ = identity_mismatches(stamp["environment"], current)
        if differ:
            raise MergeError(f"{folder} is not a shard of the current campaign: "
                             + "; ".join(differ))
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
    parser.add_argument("verb", choices=("list", "run", "merge", "smoke"))
    parser.add_argument("--config", required=True, help="campaign config (JSON)")
    parser.add_argument("--product", default=str(campaign.DEFAULT_PRODUCT))
    parser.add_argument("--unit", action="append", help="run: a unit id (repeatable); "
                        f"default {UNITS_ENV}, else every unit")
    parser.add_argument("--shards", action="append",
                        help="run: the shards root; merge: a root to merge (repeatable); "
                             "default <product>/Results/shards")
    parser.add_argument("--timing", help="run/smoke: the measured timing record; default "
                                         f"<product>/{TIMING_RELATIVE}")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        if args.verb == "list":
            outcome = [u.__dict__ for u in enumerate_units(config, args.product)]
        elif args.verb == "run":
            outcome = run_units(config, args.product, units=args.unit, timing=args.timing,
                                shards_root=args.shards[0] if args.shards else None)
        elif args.verb == "smoke":
            outcome = smoke(config, args.product, unit=(args.unit or [None])[0],
                            timing=args.timing,
                            shards_root=args.shards[0] if args.shards else None)
        else:
            outcome = merge(args.product, args.shards, config=config)
    except (UnitError, MergeError, campaign.CampaignError, search.SearchError) as error:
        print(json.dumps({"status": "refused", "error": type(error).__name__,
                          "detail": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(results._plain(outcome), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
