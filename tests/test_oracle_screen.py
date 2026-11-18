"""The oracle screening's evaluation pool and search space (no training)."""
import pytest
import torch

from MIL_CREDA_EEG import oracle_screen as O

UNIT = {"setting": "device", "d_target": 1.5, "shift": 1.5, "artifact_rate": 3.0, "event_fraction": 0.2,
        "paired": True, "n_subjects": 60}


def test_eval_pool_is_new_target_subjects_and_deterministic():
    a = O.eval_pool({"unitKwargs": UNIT}, 1.0, 4)
    b = O.eval_pool({"unitKwargs": UNIT}, 1.0, 4)
    assert [x.subject for x in a] == ["TGX-S000", "TGX-S001", "TGX-S002", "TGX-S003"]
    assert sorted(x.label for x in a) == [0, 0, 1, 1]
    assert all(torch.equal(x.windows, y.windows) for x, y in zip(a, b))


def test_eval_pool_refuses_other_settings():
    with pytest.raises(ValueError, match="device-shift"):
        O.eval_pool({"unitKwargs": {**UNIT, "setting": "subject"}}, 1.0, 2)


def test_mil_creda_space_fixes_nothing_in_advance():
    assert {"wd2", "gamma", "tau_att", "lr2", "lambda_global", "lambda_local", "tau_local"} == set(O.MIL_SPACE)
    assert O.MIL_SPACE["lr2"][1] == (1e-4, 1e-2) and O.MIL_SPACE["wd2"][1] == (1e-7, 1e-2)
    assert O.EPOCHS2 == (10, 20, 40)
    assert "AdaptMIL" not in O.METHODS and O.METHODS[0] == "MIL-CREDA" and len(O.METHODS) == 8
