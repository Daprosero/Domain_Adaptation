"""Every final model, saved per transfer, rotation, seed and group or strategy.

Experiments plan v40, Sec. 7.3 (last paragraph of "Fase 2"), with Sec. 4.2's
partition rule. Saved: the selected phase-1 checkpoint of each of the four
phase-1 groups (`GROUPS`), with its sigma for MIL-CREDA, and the selected
phase-2 model of each of the nine strategies (`STRATEGIES`). Beside each model
go the chosen hyperparameters, the partition hash and the code version, so the
mechanism experiments of Sec. 6 and any later test run on the saved models
without training again, and an interrupted remote run resumes without losing
what it already trained.

**Format.** A plain `torch.save` of the state dict (`<name>.pt`, read back
with `weights_only=True`) and a JSON sidecar (`<name>.json`) with the keys
above, the channel counts and `state_sha256`, a digest of every tensor's key,
dtype, shape and bytes. Loading recomputes the digest and refuses a state that
changed on disk.

**Refusals on load (Sec. 7.3).** A model whose `partition_sha256` is not the
partition in force is refused through `partition.check_result`
(`PartitionMismatch`); a model written by other code is refused
(`CodeVersionMismatch`) unless the caller passes `allow_code_mismatch=True`,
and then the sidecar returned says so. The store never overwrites a saved
model (`ModelExists`).

**Code version: this implementation's rule.** The plan names "the code
version" without fixing it. It is the sha256 over the source bytes of the files
that define the models and their training (`CODE_FILES`): the EEGNet trunk and
head, the MIL-CREDA model, the common wrapper, the two-phase training and the
`MIL_CREDA` package of the proposal's equations. A content digest, not a git
commit, because a remote run executes a working tree, and the commit would not
change when an uncommitted edit did.

**Layout.** `<root>/<source>-to-<target>/rotation<r>/seed<s>/phase<p>__<name>.pt`,
`<root>` defaulting to `MIL-CREDA/Models/Balladeer/` (`DEFAULT_ROOT`).

**Rebuilding a phase-1 model.** `BUILDERS` maps each phase-1 group to the
constructor of its architecture; `common` and `MIL-CREDA` are registered here,
AdaptMIL and HASD are placeholders that their comparator unit registers. A
builder takes `(source_channels, target_channels, hyper)` and returns a module
whose `predict(bags, domain)` gives the bag predictions, as `two_phase.evaluate`
reads them.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import torch

from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import partition
from MIL_CREDA_EEG.bag_data import REPOSITORY, default_partition_path

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["4.2", "7.3"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K1-atencion",
                    "K2-emparejamiento", "K3-ablacion"],
    "invariants": [
        "a_saved_model_round_trips_bit_identically",
        "load_refuses_a_foreign_partition_hash",
        "load_refuses_a_code_version_mismatch_unless_allowed",
        "a_mil_creda_model_is_stored_with_its_sigma",
        "the_store_never_overwrites_a_saved_model",
        "a_state_that_changed_on_disk_is_refused",
        "every_final_model_has_its_own_path_per_transfer_rotation_seed_and_name",
        "an_unregistered_phase_one_group_cannot_be_rebuilt",
    ],
}

FORMAT = "balladeer-model/v1"
#: Sec. 7.3: the four phase-1 groups.
GROUPS = ("common", "AdaptMIL", "HASD", "MIL-CREDA")
#: Sec. 7.1: the nine strategies of phase 2 (eight comparators and MIL-CREDA).
STRATEGIES = ("AdaptMIL", "UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS",
              "CLUST — variante source-free", "HASD", "MIL-CREDA")
#: Sec. 7.3: "con su sigma en el caso de MIL-CREDA".
SIGMA_REQUIRED = "MIL-CREDA"
DEFAULT_ROOT = REPOSITORY / "MIL-CREDA" / "Models" / "Balladeer"
_SRC = Path(__file__).resolve().parents[1]
CODE_FILES = (
    *sorted(p.relative_to(_SRC).as_posix() for p in (_SRC / "MIL_CREDA").glob("*.py")),
    "MIL_CREDA_EEG/eegnet.py", "MIL_CREDA_EEG/mil_creda_model.py",
    "MIL_CREDA_EEG/mil_wrapper.py", "MIL_CREDA_EEG/two_phase.py",
)


class StoreError(RuntimeError):
    """A model cannot be saved or loaded as the plan requires."""


class ModelExists(StoreError):
    """A model is already saved under that transfer, rotation, seed and name."""


class CodeVersionMismatch(StoreError):
    """The model was written by other code than the code that would read it."""


class GroupNotRegistered(StoreError):
    """No architecture is registered for this phase-1 group yet."""


def _build_common(source_channels: int, target_channels: int, hyper: Mapping):
    from MIL_CREDA_EEG.mil_wrapper import CommonMILModel
    return CommonMILModel(source_channels, target_channels)


def _build_mil_creda(source_channels: int, target_channels: int, hyper: Mapping):
    from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel
    return MILCREDAModel(source_channels, target_channels,
                         attention_gamma=hyper["attentionGamma"],
                         attention_temperature=hyper["attentionTemperature"])


#: Phase-1 group -> builder. AdaptMIL and HASD are registered by their own unit.
BUILDERS: dict[str, Callable] = {"common": _build_common, "MIL-CREDA": _build_mil_creda}


def require_registered(group: str) -> Callable:
    if group not in GROUPS:
        raise StoreError(f"{group!r} is not a phase-1 group; the groups are {list(GROUPS)}")
    if group not in BUILDERS:
        raise GroupNotRegistered(f"no architecture is registered for the phase-1 group "
                                 f"{group!r} yet")
    return BUILDERS[group]


def build_model(group: str, source_channels: int, target_channels: int, hyper: Mapping):
    return require_registered(group)(source_channels, target_channels, hyper)


def code_version() -> str:
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((_SRC / name).read_bytes() + b"\0")
    return digest.hexdigest()


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        digest.update(f"{key}|{t.dtype}|{tuple(t.shape)}|".encode("utf-8"))
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()


def transfer_slug(transfer: assignment_mod.Transfer) -> str:
    return f"{transfer.source}-to-{transfer.target}"


def name_slug(name: str) -> str:
    """A file-name form: `CS+CCS` -> `CS-CCS`, `CLUST — variante source-free` ->
    `CLUST-variante-source-free`."""
    return "-".join("".join(c if c.isalnum() or c == "-" else " " for c in name)
                    .replace("-", " ").split())


def _check_name(phase: int, name: str) -> None:
    if phase == 1 and name not in GROUPS:
        raise StoreError(f"phase 1 saves a group {list(GROUPS)}, not {name!r}")
    if phase == 2 and name not in STRATEGIES:
        raise StoreError(f"phase 2 saves a strategy {list(STRATEGIES)}, not {name!r}")
    if phase not in (1, 2):
        raise StoreError(f"phase must be 1 or 2, got {phase!r}")


def model_path(root: str | Path, transfer: assignment_mod.Transfer, rotation: int, seed: int,
               phase: int, name: str) -> Path:
    _check_name(phase, name)
    return (Path(root) / transfer_slug(transfer) / f"rotation{int(rotation)}"
            / f"seed{int(seed)}" / f"phase{phase}__{name_slug(name)}.pt")


def save(root: str | Path, *, transfer: assignment_mod.Transfer, rotation: int, seed: int,
         phase: int, name: str, state: Mapping[str, torch.Tensor], hyper: Mapping,
         partition_sha256: str, channels: Mapping[str, int], sigma: float | None = None,
         extra: Mapping | None = None) -> Path:
    """Write the state dict and its sidecar; refuse to overwrite."""
    path = model_path(root, transfer, rotation, seed, phase, name)
    if name == SIGMA_REQUIRED and sigma is None:
        raise StoreError(f"a {name} model is stored with its sigma (Sec. 7.3)")
    if not partition_sha256:
        raise StoreError("a model is stored with the hash of its partition (Sec. 7.3)")
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        raise ModelExists(f"{path} is already saved; the store never overwrites a model")
    state = {k: v.detach().cpu().clone() for k, v in state.items()}
    meta = {"format": FORMAT, "transfer": transfer.name, "arm": transfer.arm,
            "rotation": int(rotation), "seed": int(seed), "phase": int(phase), "name": name,
            "hyper": dict(hyper), "sigma": None if sigma is None else float(sigma),
            partition.HASH_FIELD: partition_sha256, "code_version": code_version(),
            "channels": {k: int(v) for k, v in channels.items()},
            "state_sha256": state_digest(state),
            "revision": __provenance__["revision"], **(dict(extra) if extra else {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".pt.partial")
    torch.save(state, partial)
    os.replace(partial, path)
    sidecar.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load(path: str | Path, *, partition_path: str | Path | None = None,
         allow_code_mismatch: bool = False) -> tuple[dict, dict]:
    """The state dict and its sidecar, after every refusal of Sec. 7.3."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    partition.check_result(meta, partition_path or default_partition_path())
    current = code_version()
    if meta.get("code_version") != current:
        if not allow_code_mismatch:
            raise CodeVersionMismatch(f"{path} was written by code {meta.get('code_version')} "
                                      f"and the current code is {current}")
        meta["code_version_mismatch"] = {"stored": meta.get("code_version"),
                                         "current": current}
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_digest(state) != meta["state_sha256"]:
        raise StoreError(f"{path}: the stored state does not match its sidecar's "
                         f"state_sha256")
    return state, meta


def load_model(path: str | Path, *, partition_path: str | Path | None = None,
               allow_code_mismatch: bool = False):
    """A phase-1 model rebuilt from its group's architecture, sigma fixed when stored."""
    state, meta = load(path, partition_path=partition_path,
                       allow_code_mismatch=allow_code_mismatch)
    if meta["phase"] != 1:
        raise StoreError("load_model rebuilds phase-1 checkpoints; phase-2 models are read "
                         "by the strategy that trained them")
    model = build_model(meta["name"], meta["channels"]["source"], meta["channels"]["target"],
                        meta["hyper"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise StoreError(f"{path}: the state does not fit the {meta['name']} architecture "
                         f"(missing {missing}, unexpected {unexpected})")
    if meta["sigma"] is not None and hasattr(model, "fix_sigma"):
        model.fix_sigma(meta["sigma"])
    model.eval()
    return model, meta
