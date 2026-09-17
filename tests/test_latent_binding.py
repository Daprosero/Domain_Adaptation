"""The checkpoints and the record they are read beside describe one run, or neither.

Phase two globs `config.MODELS` and measures whatever is there. A directory full
of valid checkpoints from an earlier, smaller run loads, measures and renders
exactly like the right ones — under the current record's stamp, with every other
check green. That is what these tests refuse.
"""

from __future__ import annotations

import pytest
import torch

from MIL_CREDA_Benchmark import latent


def _checkpoint(seed, **reduction):
    base = {"epochs": 20, "ceilings": {"creda": 1e-4}, "backbone": "resnet18"}
    base.update(reduction)
    return {"seed": seed, "manifest": f"/models/G_M-U_seed{seed}.manifest.json",
            "reduction": base}


def _summary(seeds, **reduction):
    base = {"epochs": 20, "ceilings": {"creda": 1e-4}, "seeds": list(seeds)}
    base.update(reduction)
    return {"reduction": base}


def test_checkpoints_from_the_run_the_record_describes_pass_through():
    found = [_checkpoint(3), _checkpoint(17)]
    assert latent.bound(found, _summary(range(30))) == found


def test_an_earlier_shorter_run_is_refused_under_the_current_stamp():
    """The failure this exists for: seed 0 is a seed the campaign ran too.

    Only `epochs` tells the two apart, which is why the comparison is over every
    shared field and not over the seed alone.
    """
    pilot = [_checkpoint(0, epochs=3)]
    with pytest.raises(latent.CheckpointsDisagree) as raised:
        latent.bound(pilot, _summary(range(30)))
    assert "epochs" in str(raised.value)


def test_a_seed_the_record_never_ran_is_refused():
    with pytest.raises(latent.CheckpointsDisagree) as raised:
        latent.bound([_checkpoint(41)], _summary(range(30)))
    assert "seed" in str(raised.value)


def test_the_shard_seed_list_is_not_compared_and_a_distributed_run_still_passes():
    """A manifest carries its shard's three seeds; the record carries all thirty.

    Requiring equality there would refuse every distributed campaign — which is
    every campaign this repository actually runs.
    """
    shard = _checkpoint(17, seeds=[15, 16, 17])
    assert latent.bound([shard], _summary(range(30))) == [shard]


def test_a_record_with_nothing_in_common_refuses_rather_than_passing_quietly():
    """An unprovable precondition is not a satisfied one.

    With no shared field the loop compares nothing and finds nothing, which reads
    exactly like agreement — in the one state where nothing at all is known.
    """
    with pytest.raises(latent.CheckpointsDisagree) as raised:
        latent.bound([{"seed": 3, "manifest": "m", "reduction": {"onlyMine": 1}}],
                     {"reduction": {"onlyTheirs": 2}})
    assert "cannot be established" in str(raised.value)


def test_no_checkpoints_at_all_refuses():
    with pytest.raises(latent.CheckpointsDisagree):
        latent.bound([], _summary(range(30)))


def _environment(**overrides):
    base = {"interpreter": "/repo/.venv", "python": "3.12.13",
            "platform": "macos-arm64", "torch": "2.13.0", "selfHosted": True,
            "power": {"source": "mains", "charge": 44},
            "device": {"name": "mps", "kind": "mps"}}
    base.update(overrides)
    return base


def test_the_charger_moving_mid_run_is_not_a_different_run():
    """A campaign takes minutes and the laptop gets plugged in halfway through.

    The run's own later checkpoints then carry `battery/70` where its record
    carries `mains/44`, and the two describe the same machine running the same
    campaign. Refusing there is a refusal about the wall socket.
    """
    plugged = _checkpoint(3, environment=_environment())
    on_battery = _checkpoint(17, environment=_environment(
        power={"source": "battery", "charge": 70}))
    summary = _summary(range(30), environment=_environment())

    assert latent.disagreements([plugged, on_battery], summary) == []
    assert latent.bound([plugged, on_battery], summary) == [plugged, on_battery]


def test_the_rest_of_the_environment_is_still_identity():
    """The exclusion is one sub-key wide, not `environment` wide.

    Each of these on its own is a checkpoint produced somewhere else, and each
    still has to be refused — otherwise the previous test bought agreement by
    hollowing the guard out.
    """
    summary = _summary(range(30), environment=_environment())
    for field, value in [("python", "3.11.9"), ("torch", "2.4.0"),
                         ("device", {"name": "T4", "kind": "cuda"}),
                         ("interpreter", "/elsewhere/.venv")]:
        elsewhere = _checkpoint(3, environment=_environment(**{field: value}))
        with pytest.raises(latent.CheckpointsDisagree) as raised:
            latent.bound([elsewhere], summary)
        assert "environment" in str(raised.value), field


def test_the_environment_finding_quotes_what_was_actually_compared():
    """A message carrying `power` would send the reader to the field that was
    excluded from the decision it is explaining."""
    elsewhere = _checkpoint(3, environment=_environment(python="3.11.9"))
    clashes = latent.disagreements([elsewhere],
                                   _summary(range(30), environment=_environment()))

    assert len(clashes) == 1
    assert clashes[0]["field"] == "environment"
    assert "power" not in clashes[0]["checkpoint_says"]
    assert "power" not in clashes[0]["record_says"]
    assert clashes[0]["checkpoint_says"]["python"] == "3.11.9"
    assert clashes[0]["record_says"]["python"] == "3.12.13"


def test_an_environment_that_is_not_a_mapping_is_still_compared():
    """Excusing a shape this does not recognise would be an exclusion nobody
    wrote: the field is still identity, whatever form it was recorded in."""
    older = _checkpoint(3, environment="macos-arm64/py3.11")
    with pytest.raises(latent.CheckpointsDisagree):
        latent.bound([older], _summary(range(30), environment="macos-arm64/py3.12"))


def test_the_disagreement_names_the_checkpoint_the_field_and_both_values():
    """A refusal that says only "they disagree" sends the reader to find out."""
    clashes = latent.disagreements([_checkpoint(0, epochs=3)], _summary(range(30)))
    assert clashes == [{"checkpoint": "G_M-U_seed0.manifest.json", "field": "epochs",
                        "checkpoint_says": 3, "record_says": 20}]


def _write_checkpoint(directory, arm, transfer="M-U", seed=0):
    """One manifest plus the weights beside it, which is what `available` globs."""
    import json
    (directory / f"{arm}_{transfer}_seed{seed}.pt").write_bytes(b"")
    (directory / f"{arm}_{transfer}_seed{seed}.manifest.json").write_text(
        json.dumps({"arm": arm, "transfer": transfer, "seed": seed,
                    "reduction": {"epochs": 20}}), encoding="utf-8")


def test_a_checkpoint_of_an_undeclared_arm_is_tagged_and_never_dropped(tmp_path,
                                                                      monkeypatch):
    """A directory outlives the declaration that filled it.

    Weights for an arm the bench has since undeclared stay on disk and glob
    exactly like the current ones. Dropping them here would let a whole previous
    campaign leave the tree with nothing saying so, which is the failure the
    `median` tag already exists to avoid one question over. So both come back and
    `declared` is what tells them apart.
    """
    from MIL_CREDA_Benchmark import config

    vigente = next(iter(config.ARMS_BY_ID))
    ajeno = next(a for a in ("A", "C", "D", "ZZ") if a not in config.ARMS_BY_ID)
    _write_checkpoint(tmp_path, vigente)
    _write_checkpoint(tmp_path, ajeno)
    monkeypatch.setattr(config, "models_for", lambda *a, **k: tmp_path)

    found = latent.available(0.0, True)

    assert len(found) == 2, "etiquetado, no filtrado: los dos tienen que volver"
    por_brazo = {entry["arm"]: entry["declared"] for entry in found}
    assert por_brazo[vigente] is True
    assert por_brazo[ajeno] is False


def test_a_checkpoint_stamped_under_an_earlier_revision_is_tagged_and_never_dropped(
        tmp_path, monkeypatch):
    """Defect (f): a manifest carrying an earlier managed revision (r17 pilot
    checkpoints are on disk under r21 today) comes back tagged rather than
    silently analysed as though it were current -- the same `tag, do not
    drop` choice `declared` already makes, applied to the revision a
    checkpoint's own manifest carries.
    """
    import json

    from MIL_CREDA_Benchmark import config

    vigente = next(iter(config.ARMS_BY_ID))
    (tmp_path / f"{vigente}_M-U_seed0.pt").write_bytes(b"")
    (tmp_path / f"{vigente}_M-U_seed0.manifest.json").write_text(
        json.dumps({"arm": vigente, "transfer": "M-U", "seed": 0,
                    "reduction": {"epochs": 20, "revision": "research-concept-r17.md"}}),
        encoding="utf-8")
    (tmp_path / f"{vigente}_M-S_seed0.pt").write_bytes(b"")
    (tmp_path / f"{vigente}_M-S_seed0.manifest.json").write_text(
        json.dumps({"arm": vigente, "transfer": "M-S", "seed": 0,
                    "reduction": {"epochs": 20, "revision": config.REVISION}}),
        encoding="utf-8")
    monkeypatch.setattr(config, "models_for", lambda *a, **k: tmp_path)

    found = latent.available(0.0, True)

    assert len(found) == 2, "etiquetado, no filtrado: los dos tienen que volver"
    by_transfer = {entry["transfer"]: entry["currentRevision"] for entry in found}
    assert by_transfer["M-U"] is False
    assert by_transfer["M-S"] is True

    summary = latent.stale_revisions(found)
    assert summary == {"count": 1, "byRevision": {"research-concept-r17.md": 1}}


def test_load_refuses_a_checkpoint_stamped_under_an_earlier_revision():
    """`load()` is where a stale-revision checkpoint would otherwise be
    measured and rendered under today's stamp without a word; this is the
    refusal half of defect (f), beside the tag `available()` already adds.
    """
    from MIL_CREDA_Benchmark import config

    stale = {"reduction": {"revision": "research-concept-r17.md"},
             "source": {}, "target": {}}
    with pytest.raises(latent.StaleCheckpointRevision) as raised:
        latent.load(stale, device=None)
    assert "research-concept-r17.md" in str(raised.value)
    assert config.REVISION in str(raised.value)


def test_a_checkpoint_with_no_revision_at_all_is_tagged_and_load_refuses_it_too():
    """Defect (g): a manifest carrying NO `revision` field at all -- an even
    older manifest, from before this field was recorded -- used to read
    inconsistently between the two tools: `available()` already tagged it
    `currentRevision: False` (`None != config.REVISION`), but `load()`'s own
    guard only fired when `revision is not None`, so it loaded the very
    checkpoint `available()` had just flagged as not current. Both now agree:
    tagged, and refused.
    """
    from MIL_CREDA_Benchmark import config

    vigente = next(iter(config.ARMS_BY_ID))
    revisionless = {"arm": vigente, "transfer": "M-U", "seed": 0,
                    "reduction": {"epochs": 20}, "source": {}, "target": {}}

    assert revisionless["reduction"].get("revision") is None

    with pytest.raises(latent.StaleCheckpointRevision):
        latent.load(revisionless, device=None)


def test_available_tags_a_checkpoint_with_no_revision_field_not_current(
        tmp_path, monkeypatch):
    """The `available()` half of the same consistency: a manifest missing
    `revision` entirely comes back tagged `currentRevision: False`, exactly
    like one naming an explicitly earlier revision -- never treated as
    current because nothing contradicts it.
    """
    import json

    from MIL_CREDA_Benchmark import config

    vigente = next(iter(config.ARMS_BY_ID))
    (tmp_path / f"{vigente}_M-U_seed0.pt").write_bytes(b"")
    (tmp_path / f"{vigente}_M-U_seed0.manifest.json").write_text(
        json.dumps({"arm": vigente, "transfer": "M-U", "seed": 0,
                    "reduction": {"epochs": 20}}),
        encoding="utf-8")
    monkeypatch.setattr(config, "models_for", lambda *a, **k: tmp_path)

    found = latent.available(0.0, True)
    assert len(found) == 1
    assert found[0]["currentRevision"] is False


def _checkpoint_with_reduction(vigente: str, **reduction):
    base = {"epochs": 20}
    base.update(reduction)
    return {"arm": vigente, "transfer": "M-U", "seed": 0, "reduction": base}


def test_available_tags_a_checkpoint_whose_bandwidth_or_attention_hyperparameters_drifted(
        tmp_path, monkeypatch):
    """Defect (i): `KERNEL_SIGMA`, `ATTENTION_GAMMA` and `ATTENTION_TEMPERATURE`
    are stamped in every `Reduction`, and a checkpoint whose manifest recorded
    a different one is tagged `currentHyperparameters: False` -- the same
    `tag, do not drop` choice `currentRevision` already makes, applied to
    Decision 1's bandwidth and Eq. (16)'s two hyperparameters rather than to
    the managed revision.
    """
    import json

    from MIL_CREDA_Benchmark import config

    vigente = next(iter(config.ARMS_BY_ID))
    current = _checkpoint_with_reduction(
        vigente, revision=config.REVISION, kernelSigma=config.KERNEL_SIGMA,
        attentionGamma=config.ATTENTION_GAMMA,
        attentionTemperature=config.ATTENTION_TEMPERATURE)
    drifted = _checkpoint_with_reduction(
        vigente, revision=config.REVISION,
        kernelSigma=config.KERNEL_SIGMA * 3,
        attentionGamma=config.ATTENTION_GAMMA,
        attentionTemperature=config.ATTENTION_TEMPERATURE)

    (tmp_path / f"{vigente}_M-U_seed0.pt").write_bytes(b"")
    (tmp_path / f"{vigente}_M-U_seed0.manifest.json").write_text(
        json.dumps(current), encoding="utf-8")
    (tmp_path / f"{vigente}_M-S_seed0.pt").write_bytes(b"")
    (tmp_path / f"{vigente}_M-S_seed0.manifest.json").write_text(
        json.dumps({**drifted, "transfer": "M-S"}), encoding="utf-8")
    monkeypatch.setattr(config, "models_for", lambda *a, **k: tmp_path)

    found = latent.available(0.0, True)
    assert len(found) == 2
    by_transfer = {entry["transfer"]: entry["currentHyperparameters"] for entry in found}
    assert by_transfer["M-U"] is True
    assert by_transfer["M-S"] is False


def test_an_older_manifest_missing_the_three_hyperparameter_fields_is_drift(
        tmp_path, monkeypatch):
    """A manifest missing one or more of `HYPERPARAMETER_FIELDS` was not
    produced by today's `harness.Reduction`, which stamps all three on every
    checkpoint. Under the current revision that absence IS drift -- treating
    it as "nothing to compare" would load and analyse the checkpoint silently
    under today's config. It may still also be caught by `currentRevision` if
    it is genuinely from an earlier revision; this test only isolates the
    hyperparameter axis.
    """
    import json

    from MIL_CREDA_Benchmark import config

    vigente = next(iter(config.ARMS_BY_ID))
    (tmp_path / f"{vigente}_M-U_seed0.pt").write_bytes(b"")
    (tmp_path / f"{vigente}_M-U_seed0.manifest.json").write_text(
        json.dumps(_checkpoint_with_reduction(vigente, revision=config.REVISION)),
        encoding="utf-8")
    monkeypatch.setattr(config, "models_for", lambda *a, **k: tmp_path)

    found = latent.available(0.0, True)
    assert found[0]["currentHyperparameters"] is False


def test_load_refuses_a_checkpoint_stamped_under_a_different_kernel_sigma():
    """`load()`'s own refusal, beside the tag `available()` adds -- the same
    shape as `StaleCheckpointRevision`'s refusal, for `kernelSigma` alone.
    """
    from MIL_CREDA_Benchmark import config

    drifted = {"reduction": {"revision": config.REVISION,
                             "kernelSigma": config.KERNEL_SIGMA * 3},
               "source": {}, "target": {}}
    with pytest.raises(latent.StaleCheckpointHyperparameters) as raised:
        latent.load(drifted, device=None)
    assert "kernelSigma" in str(raised.value)


def test_load_refuses_a_checkpoint_stamped_under_a_different_gamma_or_temperature():
    from MIL_CREDA_Benchmark import config

    drifted_gamma = {"reduction": {"revision": config.REVISION,
                                   "attentionGamma": config.ATTENTION_GAMMA + 1.0},
                     "source": {}, "target": {}}
    with pytest.raises(latent.StaleCheckpointHyperparameters) as raised:
        latent.load(drifted_gamma, device=None)
    assert "attentionGamma" in str(raised.value)

    drifted_tau = {"reduction": {"revision": config.REVISION,
                                 "attentionTemperature":
                                     config.ATTENTION_TEMPERATURE + 1.0},
                   "source": {}, "target": {}}
    with pytest.raises(latent.StaleCheckpointHyperparameters) as raised:
        latent.load(drifted_tau, device=None)
    assert "attentionTemperature" in str(raised.value)


def test_load_never_confuses_hyperparameter_drift_with_revision_drift():
    """The two refusals are distinct exceptions: a checkpoint under the
    current revision but a moved bandwidth raises
    `StaleCheckpointHyperparameters`, never `StaleCheckpointRevision` -- a
    caller that only caught the revision exception would otherwise let this
    one through silently.
    """
    from MIL_CREDA_Benchmark import config

    drifted = {"reduction": {"revision": config.REVISION,
                             "kernelSigma": config.KERNEL_SIGMA * 3},
               "source": {}, "target": {}}
    with pytest.raises(latent.StaleCheckpointHyperparameters):
        latent.load(drifted, device=None)
    try:
        latent.load(drifted, device=None)
    except latent.StaleCheckpointRevision:
        pytest.fail("hyperparameter drift raised as a revision refusal")
    except latent.StaleCheckpointHyperparameters:
        pass


def test_keep_median_stamps_the_three_hyperparameters_the_run_actually_used(
        tmp_path, monkeypatch):
    """Round-trip through the real manifest writer, not a hand-built fixture.

    `harness.keep_median` is what actually stamps a checkpoint's manifest —
    `asdict(reduction)` carries `kernelSigma`/`attentionGamma`/
    `attentionTemperature` straight from the `Reduction` those fields are
    `init=False` on, i.e. straight from `config` at the moment the checkpoint
    was written. Reading that manifest back through `latent.available()` has
    to see all three fields present and agreeing, or `Reduction` stamping a
    value the run never used would pass unnoticed.
    """
    from MIL_CREDA_Benchmark import config, harness

    arm = next(iter(config.ARMS_BY_ID))
    transfer = "M->U"
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(config, "MODELS", root)
    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    monkeypatch.setattr(config, "models_for", lambda *a, **k: root)

    (root / f"{arm}_M-U_seed0.pt").write_bytes(b"w")
    runs = [{"arm": arm, "transfer": transfer, "seed": 0, "env": "e1",
             "targetAccuracy": 0.5, "sourceAccuracy": 1.0, "seconds": 1.0,
             "contribution": 0.1, "supervised": 0.2, "adaptationShare": 0.3,
             "peakMiB": 50.0, "parameters": 11247434}]
    manifests = {(transfer, 0): {"source": "M", "target": "U"}}
    harness.keep_median(runs, arm, transfer, manifests, harness.Reduction())

    found = latent.available(0.0, True)
    assert len(found) == 1
    entry = found[0]
    reduction = entry["reduction"]
    assert reduction["kernelSigma"] == config.KERNEL_SIGMA
    assert reduction["attentionGamma"] == config.ATTENTION_GAMMA
    assert reduction["attentionTemperature"] == config.ATTENTION_TEMPERATURE
    assert entry["currentHyperparameters"] is True


def test_load_succeeds_on_a_checkpoint_whose_stamp_matches_the_current_config(
        monkeypatch):
    """`load()`'s positive path, mirrored against the refusals above it.

    Every earlier test in this file drives `load()` into one of its two
    refusals. None of them proves the refusal fires *only* when it should —
    a `load()` that refused unconditionally would pass every one of them.
    This is the case where the revision and all three hyperparameters agree
    with `config`, and neither `StaleCheckpointRevision` nor
    `StaleCheckpointHyperparameters` may fire.
    """
    from MIL_CREDA_Benchmark import bags, config, wiring

    class _FakeBagSet:
        def __init__(self):
            self.images = torch.zeros(1)
            self.members = torch.zeros(1)
            self.labels = torch.zeros(1, dtype=torch.long)
            self.train_idx = torch.tensor([0])

    class _FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(bags, "rebuild", lambda manifest, root: _FakeBagSet())
    monkeypatch.setattr(wiring, "build", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(latent.torch, "load", lambda *a, **k: {})

    record = {
        "reduction": {"revision": config.REVISION,
                      "kernelSigma": config.KERNEL_SIGMA,
                      "attentionGamma": config.ATTENTION_GAMMA,
                      "attentionTemperature": config.ATTENTION_TEMPERATURE},
        "arm": next(iter(config.ARMS_BY_ID)),
        "source": {}, "target": {}, "weights": "irrelevant.pt",
    }

    model, source, target = latent.load(record, device=None)
    assert isinstance(model, _FakeModel)


def test_a_floor_the_bench_no_longer_declares_is_an_undefined_question():
    """Not "the checkpoints are missing". The two read the same and mean opposite
    things: one is a file to go and produce, the other is a comparison that has
    stopped existing. Without this guard the weights are still on disk, `load`
    reaches `wiring.build` and the whole notebook dies on an unknown arm — which
    is the least informative possible answer to "are the two floors redundant?".
    """
    import torch
    from MIL_CREDA_Benchmark import config, latent

    ajeno = next(a for a in ("A", "C", "D", "ZZ") if a not in config.ARMS_BY_ID)
    vigente = next(iter(config.ARMS_BY_ID))

    answer = latent.floors_agree(["M-U"], 0, torch.device("cpu"),
                                 left=ajeno, right=vigente)

    assert answer["agree"] is None
    assert ajeno in answer["detail"]
    assert "declara un solo piso" in answer["detail"]
