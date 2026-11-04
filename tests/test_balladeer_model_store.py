"""Experiments plan v43, Secs. 6.3 and 7.3: every final model saved, and refused on a foreign partition."""

from __future__ import annotations

import json

import pytest
import torch

from MIL_CREDA_EEG import assignment, model_store, partition
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel

LEVEL = assignment.Transfer("level", "L1", "L11")
HYPER1 = {"lr1": 1e-3, "wd1": 1e-5, "attentionGamma": 0.3, "attentionTemperature": 0.7,
          "sigmaMultiplier": 0.75, "epochs1": 3, "epochs2": 3,
          "bagsPerStep": 4}


def _partition(root, tag: str = "a"):
    doc = {"format": "tiny", "tag": tag}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    path = root / f"partition_{tag}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, doc[partition.HASH_FIELD]


def _model():
    torch.manual_seed(0)
    model = MILCREDAModel(3, 3, attention_gamma=0.3, attention_temperature=0.7)
    model.fix_sigma(1.2345678901234567)
    return model


def _phase_one_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("target_trunk.")}


def _save(root, model, sha, **overrides):
    kwargs = dict(transfer=LEVEL, rotation=2, seed=1, phase=1, name="MIL-CREDA",
                  state=_phase_one_state(model), hyper=HYPER1, partition_sha256=sha,
                  sigma=model.sigma, channels={"source": 3, "target": 3})
    kwargs.update(overrides)
    return model_store.save(root, **kwargs)


def test_a_saved_model_round_trips_bit_identically(tmp_path):
    path_p, sha = _partition(tmp_path)
    model = _model()
    path = _save(tmp_path / "Models", model, sha)
    state, meta = model_store.load(path, partition_path=path_p)
    original = _phase_one_state(model)
    assert state.keys() == original.keys()
    for k in original:
        assert state[k].dtype == original[k].dtype and torch.equal(state[k], original[k])
    assert meta["sigma"] == model.sigma
    assert meta["hyper"] == HYPER1 and meta["partition_sha256"] == sha
    assert meta["code_version"] == model_store.code_version()
    loaded, meta2 = model_store.load_model(path, partition_path=path_p)
    assert loaded.sigma == model.sigma
    assert loaded.attention_gamma == 0.3 and loaded.attention_temperature == 0.7
    for k, v in original.items():
        assert torch.equal(loaded.state_dict()[k], v)


def test_load_refuses_a_foreign_partition_hash(tmp_path):
    _, sha_a = _partition(tmp_path, "a")
    path_b, _ = _partition(tmp_path, "b")
    path = _save(tmp_path / "Models", _model(), sha_a)
    with pytest.raises(partition.PartitionMismatch):
        model_store.load(path, partition_path=path_b)


def test_load_refuses_a_code_version_mismatch_unless_allowed(tmp_path, monkeypatch):
    path_p, sha = _partition(tmp_path)
    path = _save(tmp_path / "Models", _model(), sha)
    monkeypatch.setattr(model_store, "code_version", lambda: "0" * 64)
    with pytest.raises(model_store.CodeVersionMismatch):
        model_store.load(path, partition_path=path_p)
    _, meta = model_store.load(path, partition_path=path_p, allow_code_mismatch=True)
    assert meta["code_version_mismatch"] == {"stored": meta["code_version"],
                                             "current": "0" * 64}


def test_load_refuses_a_model_written_under_another_plan_revision(tmp_path):
    path_p, sha = _partition(tmp_path)
    path = _save(tmp_path / "Models", _model(), sha)
    # a model of the revision in force loads, and through load_model too
    assert model_store.load(path, partition_path=path_p)[1]["revision"] == \
        model_store.__provenance__["revision"]
    model_store.load_model(path, partition_path=path_p)
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    for stored in ("experiments-plan-experimental-mil-creda-sobre-balladeer-v43.md", None):
        if stored is None:
            meta.pop("revision")
        else:
            meta["revision"] = stored
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
        # allow_code_mismatch covers the code version only, never the revision
        for allow in (False, True):
            with pytest.raises(model_store.RevisionMismatch, match="revision"):
                model_store.load(path, partition_path=path_p, allow_code_mismatch=allow)
            with pytest.raises(model_store.RevisionMismatch, match="revision"):
                model_store.load_model(path, partition_path=path_p,
                                       allow_code_mismatch=allow)


def test_a_mil_creda_model_is_stored_with_its_sigma(tmp_path):
    _, sha = _partition(tmp_path)
    with pytest.raises(model_store.StoreError, match="sigma"):
        _save(tmp_path / "Models", _model(), sha, sigma=None)
    with pytest.raises(model_store.StoreError, match="sigma"):
        _save(tmp_path / "Models", _model(), sha, phase=2, sigma=None)


def test_the_store_never_overwrites_a_saved_model(tmp_path):
    _, sha = _partition(tmp_path)
    _save(tmp_path / "Models", _model(), sha)
    with pytest.raises(model_store.ModelExists):
        _save(tmp_path / "Models", _model(), sha)


def test_a_state_that_changed_on_disk_is_refused(tmp_path):
    path_p, sha = _partition(tmp_path)
    path = _save(tmp_path / "Models", _model(), sha)
    state = torch.load(path, weights_only=True)
    key = next(iter(state))
    state[key] = state[key] + 1
    torch.save(state, path)
    with pytest.raises(model_store.StoreError, match="state"):
        model_store.load(path, partition_path=path_p)


def test_every_final_model_has_its_own_path_per_transfer_rotation_seed_and_name(tmp_path):
    root = tmp_path / "Models"
    paths = {model_store.model_path(root, t, r, s, p, n)
             for t in (LEVEL, assignment.Transfer("level", "L11", "L1"))
             for r in range(4) for s in (1, 2, 3)
             for p, names in ((1, model_store.GROUPS), (2, model_store.STRATEGIES))
             for n in names}
    assert len(paths) == 2 * 4 * 3 * (len(model_store.GROUPS) + len(model_store.STRATEGIES))
    assert len(model_store.STRATEGIES) == 9 and len(model_store.GROUPS) == 4
    with pytest.raises(model_store.StoreError):
        model_store.model_path(root, LEVEL, 0, 1, 1, "UDA-DDA")       # not a phase-1 group
    with pytest.raises(model_store.StoreError):
        model_store.model_path(root, LEVEL, 0, 1, 2, "common")        # not a strategy
    assert model_store.DEFAULT_ROOT.parts[-3:] == ("MIL-CREDA", "Models", "Balladeer")
    # Sec. 6.3: the K3 variants are stored like the final models; the aggregation variants
    # train both phases, the ablation variants only phase 2.
    assert model_store.VARIANTS == {
        "MIL-CREDA [own without consensus]": (1, 2), "MIL-CREDA [published attention]": (1, 2),
        "MIL-CREDA [gated attention]": (1, 2), "MIL-CREDA [mean]": (1, 2),
        "MIL-CREDA [global term only]": (2,), "MIL-CREDA [local term only]": (2,)}
    variant_paths = {model_store.model_path(root, LEVEL, 0, 1, p, n)
                     for n, phases in model_store.VARIANTS.items() for p in phases}
    assert len(variant_paths) == 4 * 2 + 2
    assert not variant_paths & {model_store.model_path(root, LEVEL, 0, 1, p, n)
                                for p, names in ((1, model_store.GROUPS),
                                                 (2, model_store.STRATEGIES)) for n in names}
    with pytest.raises(model_store.StoreError):
        model_store.model_path(root, LEVEL, 0, 1, 1, "MIL-CREDA [global term only]")
    with pytest.raises(model_store.StoreError):
        model_store.model_path(root, LEVEL, 0, 1, 2, "MIL-CREDA [max]")


def test_an_unregistered_phase_one_group_cannot_be_rebuilt(tmp_path):
    path_p, sha = _partition(tmp_path)
    model = _model()
    path = _save(tmp_path / "Models", model, sha, name="AdaptMIL", sigma=None)
    with pytest.raises(model_store.GroupNotRegistered):
        model_store.load_model(path, partition_path=path_p)


def test_the_code_version_covers_every_comparator_module(tmp_path, monkeypatch):
    src = model_store._SRC
    comparators = sorted(p.relative_to(src).as_posix()
                         for p in (src / "MIL_CREDA_EEG" / "comparators").rglob("*.py"))
    assert comparators and set(comparators) <= set(model_store.CODE_FILES)
    import shutil
    copy = tmp_path / "src"
    for name in model_store.CODE_FILES:
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / name, copy / name)
    monkeypatch.setattr(model_store, "_SRC", copy)
    before = model_store.code_version()
    edited = copy / comparators[-1]
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert model_store.code_version() != before


@pytest.mark.parametrize("selection_module", ["MIL_CREDA_EEG/search.py",
                                              "MIL_CREDA_EEG/campaign.py"])
def test_the_code_version_covers_the_selection_code(tmp_path, monkeypatch,
                                                    selection_module):
    """Sec. 4.5: the code that selects a stored model's configuration and epoch is part of
    the code version, so a change to the selection marks stored records stale."""
    assert selection_module in model_store.CODE_FILES
    src = model_store._SRC
    import shutil
    copy = tmp_path / "src"
    for name in model_store.CODE_FILES:
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / name, copy / name)
    monkeypatch.setattr(model_store, "_SRC", copy)
    before = model_store.code_version()
    edited = copy / selection_module
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert model_store.code_version() != before


def test_the_k3_variant_models_are_stored_and_rebuilt(tmp_path):
    from MIL_CREDA_EEG import aggregation_variants as av
    path_p, sha = _partition(tmp_path)
    hyper = av.variant_hyper("published attention", HYPER1)
    torch.manual_seed(0)
    published = av.build("published attention", 3, 3, hyper)
    published.fix_sigma(0.9)
    name = model_store.variant_name("published attention")
    with pytest.raises(model_store.StoreError, match="sigma"):
        _save(tmp_path / "Models", published, sha, name=name, hyper=hyper, sigma=None,
              state=_phase_one_state(published))
    path = _save(tmp_path / "Models", published, sha, name=name, hyper=hyper, sigma=0.9,
                 state=_phase_one_state(published))
    model, meta = model_store.load_model(path, partition_path=path_p)
    assert isinstance(model, av.AggregationVariantModel) and model.sigma == 0.9
    assert model.aggregation == "published attention" and meta["name"] == name
    without = model_store.build_stored(model_store.variant_name("own without consensus"), 3, 3,
                                       av.variant_hyper("own without consensus", HYPER1))
    assert type(without) is MILCREDAModel and without.attention_gamma == 0.0
    ablation = model_store.build_stored(model_store.variant_name("global term only"), 3, 3,
                                        HYPER1)
    assert type(ablation) is MILCREDAModel and ablation.attention_gamma == 0.3


def test_the_code_version_covers_the_k3_variant_module(tmp_path, monkeypatch):
    assert "MIL_CREDA_EEG/aggregation_variants.py" in model_store.CODE_FILES
    import shutil
    src, copy = model_store._SRC, tmp_path / "src"
    for name in model_store.CODE_FILES:
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / name, copy / name)
    monkeypatch.setattr(model_store, "_SRC", copy)
    before = model_store.code_version()
    edited = copy / "MIL_CREDA_EEG" / "aggregation_variants.py"
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert model_store.code_version() != before
