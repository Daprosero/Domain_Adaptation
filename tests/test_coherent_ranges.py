"""Protocol v81 ranges: every searched range follows the method's own scales, in both phases.

Phase 1 (shared base): lr1 log [1e-4, 1e-2] and l2 log [1e-4, 1e-1] as in v80, tauAtt1 log [0.01, 1]. Phase 2: lr2 log [1e-4, 1e-2]
for every method; tau_att log [0.01, 1] for MIL-CREDA and the comparators' common MIL wrapper; MIL-CREDA's
eta_glob and eta_loc log [0.1, 1] and tau_local log [1e-3, 0.3]. Everything else keeps its v80 range.
"""
from optuna.distributions import FloatDistribution

from MIL_CREDA_EEG import criteria_pool as C
from MIL_CREDA_EEG import search
from MIL_CREDA_EEG import spectral_search as SS


def test_phase_one_space():
    assert SS.SPACE["lr1"] == ("log", 1e-4, 1e-2)
    assert SS.SPACE["l2"] == ("log", 1e-4, 1e-1)
    assert SS.SPACE["tauAtt1"] == ("log", 0.01, 1.0)
    assert SS.SPACE["width"] == ("cat", (32, 64, 128)) and SS.SPACE["epochs"] == ("cat", (50, 100, 200))


def test_mil_creda_phase_two_space():
    sp = C.space("MIL-CREDA")
    assert sp["lr2"] == FloatDistribution(1e-4, 1e-2, log=True)
    assert sp["wd2"] == FloatDistribution(1e-7, 1e-2, log=True)
    assert sp["lambda_global"] == sp["lambda_local"] == FloatDistribution(0.1, 1.0, log=True)
    assert sp["tau_local"] == FloatDistribution(1e-3, 0.3, log=True)
    assert sp["tau_att"] == FloatDistribution(0.01, 1.0, log=True)
    assert sp["gamma"] == FloatDistribution(0.0, 1.0)


def test_comparators_share_lr2_and_the_wrapper_temperature():
    for m in C.METHODS:
        if m == "MIL-CREDA":
            continue
        sp = C.space(m)
        assert sp["lr2"] == FloatDistribution(1e-4, 1e-2, log=True)
        assert sp["wd2"] == FloatDistribution(1e-7, 1e-2, log=True)
        assert sp["tau_att"] == FloatDistribution(0.01, 1.0, log=True)
    assert search.LR2 == FloatDistribution(1e-4, 1e-2, log=True)


def test_every_draw_lies_inside_its_space():
    for m in C.METHODS:
        sp = C.space(m)
        for p in C.draw(m, 1):
            for k, d in sp.items():
                if isinstance(d, FloatDistribution):
                    assert d.low <= p[k] <= d.high, (m, k, p[k])
