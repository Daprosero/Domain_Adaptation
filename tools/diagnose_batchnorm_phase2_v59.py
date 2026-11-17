"""Stored phase-2 models on target test subjects, with recalibrated target BatchNorm.

Diagnostic only. No training and no write to the model store. Statistics come from the
unlabeled target training bags of each rotation; target labels are used only to score.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import torch

from MIL_CREDA_EEG import assignment as A, bag_data, campaign, mechanisms, two_phase
from MIL_CREDA_EEG.phase_one_sweep import _summary, recalibrate

torch.set_num_threads(4)
run = json.load(open("tools/kaggle/v59-k3/run-config.json"))["run"]["kwargs"]["config"]
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != "remoteInputs"}, "device": "cpu"})
campaign.register(cfg["strategies"])
bags = bag_data.load_bags(None, None)
root = Path("MIL-CREDA-EEG/Models/Balladeer")
device = torch.device("cpu")
transfers = {t.name: t for t in A.TRANSFERS}


def side(model, group):
    p = two_phase.evaluate(model, group, "target", device)[:, 1].detach().numpy()
    return [{"label": b.label, "probability": float(q)} for b, q in zip(group, p)]


for name in sys.argv[1:]:
    t = transfers[name]
    a = A.assign_from_bags(t, bags)
    stored, fresh = [], []
    for r in range(4):
        path = root / name.replace("->", "-to-") / f"rotation{r}" / "seed1" / "phase2__MIL-CREDA.pt"
        model = mechanisms.load_phase_two(path)[0]
        model = model.to(device)
        unlabeled = two_phase.bag_tensors(A.side_bags(a, bags, "target", r, A.REFIT_ROLE))
        test = two_phase.bag_tensors(A.side_bags(a, bags, "target", r, "test"))
        stored += side(model, test)
        recalibrate(model, unlabeled, device, trunk="target_trunk")
        fresh += side(model, test)
        del model, unlabeled, test
    print(name, "stored", _summary(stored), "recalibrated", _summary(fresh), flush=True)
