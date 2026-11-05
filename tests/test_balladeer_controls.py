"""Experiments plan v50, Secs. 4.5 and 7.5: source-only controls and the target-supervised
upper bound."""

from __future__ import annotations

import json
import numbers

import numpy as np
import pytest
import torch

from MIL_CREDA_EEG import (assignment, bag_data, controls, eegnet, mil_wrapper, model_store,
                           partition, two_phase as tp)
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel

CPU = torch.device("cpu")
LEVEL = assignment.Transfer("level", "L1", "L11")
DEVICE = assignment.Transfer("device", "L6", "Robots")
SCHEDULE = {"epochs1": 1, "epochs2": 1, "bagsPerStep": 2}


def _bagset(root, sha: str = "x" * 64) -> bag_data.BagSet:
    """24 subjects over 4 folds; L1 and L11 holders, some holding both (3 channels)."""
    rng = np.random.default_rng(0)
    bags = []
    for i in range(24):
        subject, fold, label = f"UB{i:04d}", i % 4, ("yes", "no")[(i // 4) % 2]
        held = {0: ("L1",), 1: ("L11",), 2: ("L1", "L11")}[i % 3]
        for condition in held:
            path = root / f"{subject}_{condition}.npz"
            np.savez(path, signal=rng.normal(size=(3, 3, 512)).astype(np.float32))
            bags.append(bag_data.Bag(subject=subject, condition=condition,
                                     record_id=f"{subject}-{condition}", label=label, sex="M",
                                     fold=fold, device="CGX", channels=("a", "b", "c"),
                                     n_windows=3, path=path))
    rotations = {str(r): partition.rotation_roles(r) for r in range(4)}
    return bag_data.BagSet(bags=tuple(bags), partition_sha256=sha, rotations=rotations)


def _target_bags(channels: int, n: int = 4) -> list[tp.BagTensor]:
    g = torch.Generator().manual_seed(5)
    return [tp.BagTensor(f"t{i}", "c", i % 2, torch.randn(3, channels, eegnet.SAMPLES,
                                                          generator=g)) for i in range(n)]


class _Untouchable(torch.nn.Module):
    def predict(self, *args, **kwargs):
        raise AssertionError("a device-arm control must not run the model")


def _numbers(value):
    if isinstance(value, bool) or isinstance(value, str) or value is None:
        return []
    if isinstance(value, numbers.Number):
        return [value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _numbers(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _numbers(v)]
    return [value]


def test_the_device_arm_controls_return_the_na_record_and_never_a_number():
    assert controls.NA_RECORD == "N/A — espacios de entrada heterogéneos"
    for group in model_store.GROUPS:
        out = controls.source_only(DEVICE, group, _Untouchable(), _target_bags(2),
                                   rotation=1, seed=2)
        assert out["record"] == controls.NA_RECORD
        assert out["applicable"] is False and out["in_statistics"] is False
        assert _numbers({k: v for k, v in out.items() if k not in ("rotation", "seed")}) == []
        assert "balanced_accuracy" not in out and "predictions" not in out


def test_the_main_reference_is_the_common_checkpoint_applied_to_the_target():
    torch.manual_seed(0)
    model = mil_wrapper.CommonMILModel(3, 3)
    bags = _target_bags(3)
    out = controls.source_only(LEVEL, "common", model, bags, rotation=0, seed=1)
    assert out["reference"] == "main source-only reference" and out["applicable"] is True
    G = tp.evaluate(model, bags, "source", CPU)
    assert [p["probability"] for p in out["predictions"]] == [float(g[1]) for g in G]
    assert [p["subject"] for p in out["predictions"]] == [b.subject for b in bags]
    assert out["balanced_accuracy"] == tp.balanced_accuracy([b.label for b in bags],
                                                            G.argmax(dim=1).tolist())
    with pytest.raises(controls.ControlError, match="channels"):
        controls.source_only(LEVEL, "common", model, _target_bags(2), rotation=0, seed=1)


def test_each_other_phase_one_group_has_a_matched_source_only_control():
    assert controls.MAIN_REFERENCE == "common"
    assert controls.MATCHED_GROUPS == ("AdaptMIL", "HASD", "MIL-CREDA")
    torch.manual_seed(0)
    model = MILCREDAModel(3, 3, attention_gamma=0.4, attention_temperature=1.0)
    model.fix_sigma(2.0)
    out = controls.source_only(LEVEL, "MIL-CREDA", model, _target_bags(3), rotation=0, seed=1)
    assert out["reference"] == "matched source-only control" and out["group"] == "MIL-CREDA"
    with pytest.raises(controls.ControlError):
        controls.source_only(LEVEL, "UDA-DDA", model, _target_bags(3), rotation=0, seed=1)


def _partition_file(root):
    doc = {"format": "tiny"}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    path = root / "partition.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, doc[partition.HASH_FIELD]


def test_a_placeholder_group_control_accepts_any_registered_phase_one_checkpoint(
        tmp_path, monkeypatch):
    path_p, sha = _partition_file(tmp_path)
    torch.manual_seed(0)
    model = mil_wrapper.CommonMILModel(3, 3)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("target_trunk.")}
    saved = model_store.save(tmp_path / "Models", transfer=LEVEL, rotation=0, seed=1, phase=1,
                             name="HASD", state=state, hyper={"lr1": 1e-3},
                             partition_sha256=sha, channels={"source": 3, "target": 3})
    with pytest.raises(model_store.GroupNotRegistered):
        controls.source_only_from_store(saved, LEVEL, _target_bags(3), partition_path=path_p)
    monkeypatch.setitem(model_store.BUILDERS, "HASD",
                        lambda s, t, hyper: mil_wrapper.CommonMILModel(s, t))
    out = controls.source_only_from_store(saved, LEVEL, _target_bags(3), partition_path=path_p)
    assert out["group"] == "HASD" and out["applicable"] is True and out["rotation"] == 0
    with pytest.raises(controls.ControlError, match="transfer"):
        controls.source_only_from_store(saved, assignment.Transfer("level", "L11", "L1"),
                                        _target_bags(3), partition_path=path_p)


def test_the_upper_bound_never_sees_source_labels_nor_target_test_labels(tmp_path, monkeypatch):
    bags = _bagset(tmp_path)
    rotation = 1
    a = assignment.assign_from_bags(LEVEL, bags)
    view = a.rotation(rotation)
    forbidden = view["source_nontest"] | view["source_test"] | view["target_test"]
    for s in forbidden:                       # their bags cannot be read at all any more
        bags.get(s, a.conditions[s]).path.unlink()
    calls = []
    real = assignment.side_bags

    def spy(a_, bags_, side, r, role, **kwargs):
        calls.append((side, role, kwargs.get("split")))
        return real(a_, bags_, side, r, role, **kwargs)

    monkeypatch.setattr(controls.assignment_mod, "side_bags", spy)
    out = controls.train_upper_bound(a, bags, rotation, 3, {"lr1": 1e-2, "wd1": 1e-4},
                                     SCHEDULE, CPU)
    assert sorted(calls) == sorted([("target", "inner_train", i) for i in range(3)]
                                   + [("target", "inner_validation", i) for i in range(3)]
                                   + [("target", "nontest", None)])
    assert out["in_ranking"] is False and out["reference"] == "target-supervised upper bound"
    non_test = view["target_nontest"]
    assert out["subjects"]["train"] == sorted(non_test)
    assert sorted(out["subjects"]["inner_validation"]) == sorted(non_test)


def test_the_upper_bound_is_the_common_architecture_on_the_target_channels(tmp_path):
    bags = _bagset(tmp_path)
    a = assignment.assign_from_bags(LEVEL, bags)
    out = controls.train_upper_bound(a, bags, 0, 3, {"lr1": 1e-2, "wd1": 1e-4}, SCHEDULE, CPU)
    model = out["model"]
    assert isinstance(model, mil_wrapper.CommonMILModel)
    assert model.source_trunk.channels == 3
    assert {r["model"] for r in out["history"]} == {"upper-bound:target-supervised"}
    with pytest.raises(KeyError):
        controls.train_upper_bound(a, bags, 0, 3, {"lr1": 1e-2}, SCHEDULE, CPU)
    with pytest.raises(ValueError, match="schedule"):
        controls.train_upper_bound(a, bags, 0, 3, {"lr1": 1e-2, "wd1": 1e-4},
                                   {"epochs1": 1}, CPU)


def test_the_upper_bound_epochs_are_chosen_on_the_pooled_inner_splits_and_refit_without_validation(
        tmp_path, monkeypatch):
    bags = _bagset(tmp_path)
    a = assignment.assign_from_bags(LEVEL, bags)
    runs = []
    real = tp.train_source_phase

    def spy(model, train, valid, hyper, generator, device, **kwargs):
        out = real(model, train, valid, hyper, generator, device, **kwargs)
        runs.append({"valid": len(valid), "epochs": hyper["epochs1"],
                     "seed": kwargs["seed"], "ran": out["epochs"]})
        return out

    monkeypatch.setattr(controls.two_phase, "train_source_phase", spy)
    pooled = [0.5, 0.75, 0.75]
    monkeypatch.setattr(controls.search, "pooled_series",
                        lambda per_split, label_free: {"pooledBalancedAccuracy": pooled})
    out = controls.train_upper_bound(a, bags, 2, 3, {"lr1": 1e-2, "wd1": 1e-4},
                                     {**SCHEDULE, "epochs1": 3}, CPU)
    # three inner splits at the search seed over the whole cap, each with validation ...
    assert [(r["epochs"], r["seed"], r["ran"]) for r in runs[:3]] == [(3, 0, 3)] * 3
    assert all(r["valid"] > 0 for r in runs[:3])
    # ... then one refit at the report seed, the earliest best epoch count, no validation
    assert runs[3] == {"valid": 0, "epochs": 2, "seed": 3, "ran": 2} and len(runs) == 4
    assert out["best_epoch"] == 1 and out["selection"]["epochs"] == 2
    assert out["hyper"]["epochs1"] == 2 and out["best_valid_balanced_accuracy"] == 0.75
    assert len(out["history"]) == 2
