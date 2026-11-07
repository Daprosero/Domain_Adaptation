"""Stored phase-1 models scored with their BatchNorm statistics and with recalibrated ones.

Diagnostic only. No training and no write to the model store; statistics are recomputed
from the source training bags of each rotation. Target labels are only read.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import torch

from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, two_phase
from MIL_CREDA_EEG.phase_one_sweep import _side, _summary, recalibrate

torch.set_num_threads(4)
run = json.load(open("tools/kaggle/v59-k3/run-config.json"))["run"]["kwargs"]["config"]
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != "remoteInputs"}, "device": "cpu"})
campaign.register(cfg["strategies"])
bags = bag_data.load_bags(None, None)
root = Path("MIL-CREDA-EEG/Models/Balladeer")
device = torch.device("cpu")
transfers = {t.name: t for t in A.TRANSFERS}
for name in sys.argv[1:]:
    t = transfers[name]
    a = A.assign_from_bags(t, bags)
    folder = name.replace("->", "-to-")
    sides = {k: [] for k in ("source", "target", "sourceRecal", "targetRecal")}
    for r in range(4):
        path = root / folder / f"rotation{r}" / "seed1" / "phase1__MIL-CREDA.pt"
        model, _ = model_store.load_model(path, allow_code_mismatch=True)
        model = model.to(device)
        train = two_phase.bag_tensors(A.side_bags(a, bags, "source", r, A.REFIT_ROLE))
        source = two_phase.bag_tensors(A.side_bags(a, bags, "source", r, "test"))
        target = two_phase.bag_tensors(A.side_bags(a, bags, "target", r, "test"))
        sides["source"] += _side(model, source, device)
        sides["target"] += _side(model, target, device)
        recalibrate(model, train, device)
        sides["sourceRecal"] += _side(model, source, device)
        sides["targetRecal"] += _side(model, target, device)
        del model, train, source, target
    print(name, {k: _summary(v) for k, v in sides.items()}, flush=True)
