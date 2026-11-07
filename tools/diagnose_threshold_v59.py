"""Stored models with the decision threshold set from unlabeled target bags.

Diagnostic only. No training and no write to the model store. Per rotation, the threshold
makes the share of positive target training bags equal the source training class share.
Target test labels are used only to score.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from MIL_CREDA_EEG import assignment as A, bag_data, campaign, mechanisms, model_store, two_phase

torch.set_num_threads(4)
run = json.load(open("tools/kaggle/v59-k3/run-config.json"))["run"]["kwargs"]["config"]
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != "remoteInputs"}, "device": "cpu"})
campaign.register(cfg["strategies"])
bags = bag_data.load_bags(None, None)
device = torch.device("cpu")
root = Path("MIL-CREDA-EEG/Models/Balladeer")
transfers = {t.name: t for t in A.TRANSFERS}


def probabilities(model, group, domain):
    return two_phase.evaluate(model, group, domain, device)[:, 1].detach().numpy()


def balanced(y, hat):
    y, hat = np.array(y), np.array(hat)
    return float(np.mean([(hat[y == c] == c).mean() for c in (0, 1)]))


for name in sys.argv[1:]:
    t = transfers[name]
    a = A.assign_from_bags(t, bags)
    out = {k: {"y": [], "p": [], "fixed": [], "moved": []} for k in ("predict", "milCreda")}
    for r in range(4):
        folder = root / name.replace("->", "-to-") / f"rotation{r}" / "seed1"
        share = np.mean([b.label == "yes" for b in A.side_bags(a, bags, "source", r, A.REFIT_ROLE)])
        unlabeled = two_phase.bag_tensors(A.side_bags(a, bags, "target", r, A.REFIT_ROLE))
        test = two_phase.bag_tensors(A.side_bags(a, bags, "target", r, "test"))
        models = (("predict", model_store.load_model(folder / "phase1__MIL-CREDA.pt", allow_code_mismatch=True)[0], "source"),
                  ("milCreda", mechanisms.load_phase_two(folder / "phase2__MIL-CREDA.pt")[0], "target"))
        for key, model, domain in models:
            cut = np.quantile(probabilities(model, unlabeled, domain), 1 - share)
            p = probabilities(model, test, domain)
            o = out[key]
            o["y"] += [b.label for b in test]
            o["p"] += p.tolist()
            o["fixed"] += (p >= 0.5).astype(int).tolist()
            o["moved"] += (p >= cut).astype(int).tolist()
        del models, unlabeled, test
    print(name, {k: {"at.5": round(balanced(o["y"], o["fixed"]), 3), "moved": round(balanced(o["y"], o["moved"]), 3),
                     "auc": round(float(roc_auc_score(o["y"], o["p"])), 3)} for k, o in out.items()}, flush=True)
