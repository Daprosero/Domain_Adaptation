"""The checkpoints and the record they are read beside describe one run, or neither.

Phase two globs `config.MODELS` and measures whatever is there. A directory full
of valid checkpoints from an earlier, smaller run loads, measures and renders
exactly like the right ones — under the current record's stamp, with every other
check green. That is what these tests refuse.
"""

from __future__ import annotations

import pytest

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
