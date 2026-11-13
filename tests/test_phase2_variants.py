"""variant=: K3 ablations of MIL-CREDA's phase 2 only (S8, v70 8.3). Phase 1 and the shared base are unchanged."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest
import torch

from MIL_CREDA_EEG import campaign_unit as CU
from MIL_CREDA_EEG import mil_creda_model as MM
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


def test_sum_and_mean_pooling_give_unit_and_uniform_weights():
    m = MM.MILCREDAModel.__new__(MM.MILCREDAModel)
    w = torch.tensor([0.1, 2.0, -1.0])
    assert torch.allclose(MM.pooled_weights(w, 1.0, "sum"), torch.ones(3, dtype=w.dtype))
    assert torch.allclose(MM.pooled_weights(w, 1.0, "mean"), torch.full((3,), 1 / 3, dtype=w.dtype))
    assert torch.allclose(MM.pooled_weights(w, 1.0, "attention"), torch.softmax(w, 0))


def _run(tmp, variant, seen):
    config = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    real = XP.run_unit

    def spy(*a, **k):
        seen.append(dict(P.PHASE_TWO_AGGREGATION)); return real(*a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
        mp.setitem(XP.SPACE2, "epochs2", ("cat", (1,)))
        mp.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])
        mp.setattr(XP, "run_unit", spy)
        CU.run(config, seeds=(1,), trials=1, trials2=2, methods=(), workdir=tmp, environ={}, pick="cef",
               variant=variant, full_trace=False, experiment="sim", rotation=0, setting="subject", shift=1.0,
               n_subjects=20)
    (xp,) = glob.glob(str(tmp / "xpilot-*.json"))
    return json.loads(Path(xp).read_text())


@pytest.mark.parametrize("variant,agg,fixed", [
    ("nocons", {"gamma": 0.0, "temperature": 1.0, "pooling": "attention"}, {}),
    ("mean", {"gamma": 0.0, "temperature": 1.0, "pooling": "mean"}, {}),
    ("sum", {"gamma": 0.0, "temperature": 1.0, "pooling": "sum"}, {}),
    ("global", {"gamma": 0.5, "temperature": 1.0, "pooling": "attention"}, {"lambda_local": 0.0}),
    ("local", {"gamma": 0.5, "temperature": 1.0, "pooling": "attention"}, {"lambda_global": 0.0}),
])
def test_each_variant_changes_phase_two_only_and_is_restored(tmp_path, variant, agg, fixed):
    before, space = dict(P.PHASE_TWO_AGGREGATION), dict(XP.SPACE2)
    seen = []
    rec = _run(tmp_path, variant, seen)
    assert seen and seen[0] == agg
    s = rec["phase2Search"]
    assert s["variant"] == variant
    for k, v in fixed.items():
        assert all(t["params"][k] == v for t in s["trials"]) and s["params"][k] == v
    assert P.PHASE_TWO_AGGREGATION == before and dict(XP.SPACE2) == space
