"""Plan v63 timing entry for remote-parallelism's `timing_probe.py`.

`epoch(settings, wrap)` trains ONE epoch of one arm on inner split 0 of rotation 0 of one
transfer, exactly as a search trial of `v63_campaign` trains it (spectral encoder, v63
standardization, MIL-CREDA as run). The arm and the transfer come from the environment:

    V63_TIMING_ARM       p1:<group> or p2:<strategy>   (default p1:MIL-CREDA)
    V63_TIMING_TRANSFER  one of the 16 transfers        (default L1->L11)

A phase-2 arm starts from a phase-1 checkpoint of its group trained for two epochs once per
process (setup, not timed). CCA-LSC cannot train fewer than 4 phase-2 epochs (3 warm-up and
1 calibrated), so its timed unit is those 4 epochs; every other arm's unit is one epoch.
The data are the real processed bags (`bag_data.load_bags`); no label chooses anything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from MIL_CREDA_EEG import bag_data, campaign, model_store, search  # noqa: E402
from MIL_CREDA_EEG import v63_campaign as C  # noqa: E402
from MIL_CREDA_EEG import v63_setup as V  # noqa: E402

_CACHE: dict = {}
SCHEDULE = {"epochs1": 1, "epochs2": 1, "bagsPerStep": 8}
PARAMS1 = {"lr1": 3e-3, "l2": 1e-2}


def _setup() -> dict:
    key = (os.environ.get("V63_TIMING_ARM", "p1:MIL-CREDA"),
           os.environ.get("V63_TIMING_TRANSFER", "L1->L11"))
    if key in _CACHE:
        return _CACHE[key]
    arm, name = key
    phase, _, who = arm.partition(":")
    campaign.register(list(model_store.STRATEGIES))
    bags = bag_data.load_bags(None, None)
    data = V.search_data(V.transfer(name), bags, rotations=[0])
    state = {"phase": phase, "who": who, "data": data, "view": data.inner_view(0, 0)}
    if phase == "p2":
        group = search.PHASE_ONE_GROUP_OF[who]
        with V.installed():
            h = V.phase_one_hyper(group, PARAMS1, {**SCHEDULE, "epochs1": 2})
            with V.phase_one_l2(PARAMS1["l2"]):
                out = V.train_phase_one(group, data, 0, 1, h)
        ck = out["result"]["checkpoint"]
        meta = {"phase": 1, "name": group, "transfer": name, "rotation": 0, "seed": 1,
                "partition_sha256": data.partition_sha256, "channels": out["channels"],
                "hyper": out["hyper"], "sigma": ck.get("sigma")}
        state["record"] = (ck["state"], meta)
        space = C.phase_two_space(who)
        state["params"] = {k: (getattr(d, "choices", None) or [None])[0]
                           if hasattr(d, "choices") else (d.low * d.high) ** 0.5
                           for k, d in space.items()}
        state["epochs2"] = search.MIN_TRAINED_EPOCHS.get(who, 1)
    _CACHE[key] = state
    return state


def epoch(settings: dict, wrap) -> None:
    if settings.get("threads"):
        torch.set_num_threads(int(settings["threads"]))
    s = _setup()
    for _ in wrap(range(1)):
        with V.installed():
            if s["phase"] == "p1":
                h = V.phase_one_hyper(s["who"], PARAMS1, SCHEDULE)
                with V.phase_one_l2(PARAMS1["l2"]):
                    V.train_phase_one(s["who"], s["view"], 0, 0, h, sigma=False)
            else:
                trainer = C.phase_two_trainer(s["who"], gate=1.0)
                trainer(s["view"], 0, 0, s["record"], s["params"],
                        {**SCHEDULE, "epochs2": s["epochs2"]}, "cpu")
