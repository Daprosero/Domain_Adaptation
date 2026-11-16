"""The criteria pool's search spaces, its random draw and its sharding (no training)."""
from MIL_CREDA_EEG import criteria_pool as C


def test_spaces_search_everything_and_keep_gamma_and_tau_local_to_mil_creda():
    mil = C.space("MIL-CREDA")
    assert set(mil) == {"lr2", "wd2", "lambda_global", "lambda_local", "tau_local", "gamma", "tau_att"}
    assert (mil["tau_local"].low, mil["tau_local"].high, mil["tau_local"].log) == (1e-3, 1.0, True)
    assert (mil["tau_att"].low, mil["tau_att"].high) == (1.0, 10.0)
    for m in C.METHODS[1:]:
        s = C.space(m)
        assert (s["tau_att"].low, s["tau_att"].high, s["tau_att"].log) == (1.0, 10.0, True)
        assert "gamma" not in s and "tau_local" not in s and "epochs2" not in s
        assert {"lr2", "wd2"} <= set(s)
    assert C.EPOCHS2 == 100 and C.TRIALS == 100 and C.SEEDS == (1, 2, 3, 4, 5)


def test_draw_is_deterministic_and_differs_across_seeds_and_methods():
    a, b = C.draw("COT", 1), C.draw("COT", 1)
    assert a == b and len(a) == C.TRIALS
    assert a != C.draw("COT", 2)
    assert a[0]["lr2"] != C.draw("CLUDA", 1)[0]["lr2"]


def test_shards_cover_every_trial_once():
    covered = [i for s in range(C.SHARDS) for i in C.shard_trials(s)]
    assert covered == list(range(C.TRIALS))


def test_units_are_the_twelve_dev_units_and_the_six_d1_boundary_units():
    assert len(C.UNITS) == 18 and len(set(C.UNITS)) == 18
    assert sum(u.startswith("d1-") for u in C.UNITS) == 6
    assert all(C.base_file(u).name == f"{u}-s1.pt" for u in C.UNITS)


def test_training_seed_depends_on_seed_and_trial_only():
    assert C.train_seed(2, 37) == C.train_seed(2, 37) != C.train_seed(3, 37)
