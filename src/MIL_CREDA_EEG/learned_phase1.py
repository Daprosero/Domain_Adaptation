"""Phase 1 with a learned instance encoder (TSMNet-style or T-GARNet), unseen-source generalization.

Diagnostic, outside the plan's protocol; writes nothing to the model store. Same windows as the
spectral check (1-45 Hz, common average reference, at most 60 per bag), inputs standardized per
channel by the source training windows, neutral MIL aggregation (no consensus, flat attention),
explicit L2 .01. Unseen source test AUC and BA, pooled over the four outer rotations.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, learned_trunks, model_store, remote_units, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2

HYPER = {**sc.HYPER, "lr1": 1e-3, "epochs1": 100}


def _blocks(raw, filtered=True):
    """The "blocks" arm: 50%-overlap windows cut into ordered blocks of 15
    (v64_pilot.ordered_blocks); each block is one training unit. `filtered` adds the 1-45 Hz
    band-pass and common average reference; False keeps the stored 0.2-45 Hz windows."""
    from MIL_CREDA_EEG import v64_pilot
    return [blk for b in raw for blk in v64_pilot.ordered_blocks(b, None, filtered=filtered)]


def _stored_whole(bags):
    """The "whole" arm on the stored 0.2-45 Hz windows: at most WINDOWS evenly spaced per bag."""
    return [two_phase.BagTensor(b.subject, b.condition, b.label,
                                b.windows[::max(1, len(b.windows) // sc.WINDOWS)].float())
            for b in bags]


def euclidean_aligned(units):
    """Euclidean alignment (He & Wu, IEEE TBME 2020) per real subject, label-free: every unit of a
    subject (its bag, or all its blocks) is re-centred by that subject's mean window covariance,
    X -> R^{-1/2} X, as tools/diagnostics/ea_check.py does; test subjects use their own windows."""
    by = {}
    for u in units:
        by.setdefault(u.subject.split("#")[0], []).append(u)
    inv = {}
    for subject, us in by.items():
        x = torch.cat([u.windows for u in us]).double()
        R = torch.einsum("nct,ndt->cd", x, x) / (x.shape[0] * x.shape[2])
        w, V = torch.linalg.eigh(R + 1e-6 * torch.eye(R.shape[0], dtype=R.dtype) * R.diagonal().mean())
        inv[subject] = V @ torch.diag(w.clamp_min(1e-12) ** -0.5) @ V.T
    return [two_phase.BagTensor(u.subject, u.condition, u.label,
                                torch.einsum("cd,ndt->nct", inv[u.subject.split("#")[0]],
                                             u.windows.double()).float()) for u in units]


def _cleaned_whole(raw, rule, channels=None):
    """The "whole" arm on the stored 0.2-45 Hz windows after a label-free window selection.
    `rule` names one or both of: "contact" drops windows with bad contact on any electrode
    (Sec. 2.5's per-window rule; only where the record carries the indicator, EMOTIV);
    "ocular" drops, within each record, windows whose frontal peak-to-peak exceeds the record's
    median + 3 MAD (Sec. 2.8's measure); joined by "+". `channels` keeps only those channels
    (the shared montage of a device transfer). Then at most WINDOWS evenly spaced per bag."""
    rules = set((rule or "").split("+")) - {""}
    out = []
    for b in raw:
        w, mk = b.windows(), b.marks()
        keep = np.ones(len(w), dtype=bool)
        if "contact" in rules and "contact_bad" in mk:
            keep &= ~np.asarray(mk["contact_bad"]).any(axis=1)
        if "ocular" in rules:
            fp = np.asarray(mk["frontal_ptp"], dtype=float)
            med = np.nanmedian(fp)
            mad = 1.4826 * np.nanmedian(np.abs(fp - med))
            keep &= ~(fp > med + 3.0 * mad)
        if not keep.any():
            keep[:] = True
        w = w[keep]
        if channels is not None:
            names = list(b.channels)
            w = w[:, [names.index(c) for c in channels]]
        out.append(two_phase.BagTensor(b.subject, b.condition, two_phase.CLASSES.index(b.label),
                                       torch.from_numpy(np.ascontiguousarray(
                                           w[::max(1, len(w) // sc.WINDOWS)])).float()))
    return out


@contextlib.contextmanager
def _trunk(kind):
    if kind == "eegnet":                     # EEGNet-8,2 as defined: the plan's own trunk
        yield
    elif kind == "spectral":
        from MIL_CREDA_EEG import spectral_trunk
        with spectral_trunk.installed():
            yield
    else:
        with learned_trunks.installed(kind):
            yield


class _Reverse(torch.autograd.Function):
    """Gradient reversal (Ganin & Lempitsky, ICML 2015): identity forward, -lambda backward."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def train_phase_one(model, train, hyper, generator, device, adversary: bool = False):
    """Phase 1 exactly as two_phase.train_source_phase (Adam on trunk, attention and head,
    interleaved class-balanced batches, constraints after each step, Eq. 21 with the active
    explicit L2), optionally with a class-conditional subject adversary: one head per class
    predicts, from every window embedding, which training subject of that class it comes
    from; the trunk receives the reversed gradient, scaled by lambda = 2/(1+exp(-10 p)) - 1
    over training progress p. It uses subject identities only, never a target label."""
    hyper = two_phase.check_hyper(hyper)
    two_phase.phase_one_sigma(model, train, hyper["sigmaMultiplier"], device)
    params = [*model.source_trunk.parameters(), *model.attention_parameters(), *model.head.parameters()]
    heads, index = None, {}
    if adversary:
        d = model.source_trunk.output_dim
        by = {}
        for b in train:
            by.setdefault(int(b.label), []).append(b.subject.split("#")[0])
        heads = torch.nn.ModuleDict()
        for c, subs in by.items():
            subs = sorted(set(subs))
            index.update({(c, s): i for i, s in enumerate(subs)})
            heads[str(c)] = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(),
                                                torch.nn.Linear(64, len(subs)))
        heads = heads.to(device)
        params += list(heads.parameters())
    optimizer = torch.optim.Adam(params, lr=hyper["lr1"], weight_decay=hyper["wd1"])
    epochs = int(hyper["epochs1"])
    for epoch in range(epochs):
        model.train()
        lam = 2.0 / (1.0 + np.exp(-10.0 * epoch / max(1, epochs - 1))) - 1.0
        for batch in two_phase.interleaved_batches(train, hyper["bagsPerStep"], generator):
            bags = [train[i] for i in batch]
            labels = torch.tensor([b.label for b in bags], device=device)
            H = model.embed([b.windows.to(device) for b in bags], "source")
            G = model.classify(model.represent(model.bag_pairs(H)))
            loss = two_phase.source_term(model, G, labels)
            if adversary:
                ce = []
                for b, h in zip(bags, H):
                    c = int(b.label)
                    target = torch.full((h.shape[0],), index[(c, b.subject.split("#")[0])],
                                        device=device, dtype=torch.long)
                    logits = heads[str(c)](_Reverse.apply(h.float(), float(lam)))
                    ce.append(torch.nn.functional.cross_entropy(logits, target))
                loss = loss + torch.stack(ce).mean().to(loss.dtype)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.constrain("source_trunk", "head", "attention")


def run_source(transfer, cfg, bags, device, kind, seed, epochs, arm="whole", ea=False,
               stored=False, rotations=(0, 1, 2, 3), clean=None, threshold="fixed", target=False,
               adversary=None):
    """`threshold` "fixed" scores at 0.5 after training on every non-test source subject;
    "source-validation" trains on inner split 0's training subjects, picks the cut that
    maximizes balanced accuracy on its validation subjects (source labels only) and applies
    that cut, unchanged, to the test subjects. `target` also scores the same model and cut on
    the target domain's test subjects (no target label is read before scoring; no adaptation:
    predict), on the shared channels of a device transfer."""
    from MIL_CREDA_EEG import v63_setup
    tr = v63_setup.transfer(transfer)
    channels = v63_setup.channels_of(tr) if target else None
    ty_all, tp_all, tcuts = [], [], []
    from MIL_CREDA_EEG import v63_setup
    a = A.assign_from_bags(v63_setup.transfer(transfer), bags)
    hyper = two_phase.check_hyper({**HYPER, "epochs1": epochs, "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    y, p, cuts = [], [], []
    train_y, train_p = [], []
    kept = []
    for r in rotations:
        tuned = threshold == "source-validation"
        raw = lambda role, side_="source": (A.side_bags(a, bags, side_, r, role, split=0)
                                            if role.startswith("inner")
                                            else A.side_bags(a, bags, side_, r, role))
        if arm == "cleanblocks":
            from MIL_CREDA_EEG import search_pilot
            prep = lambda role, side_="source": [blk for b in raw(role, side_)
                                                 for blk in search_pilot.clean_blocks(b, channels)]
        elif clean or target:
            prep = lambda role, side_="source": _cleaned_whole(raw(role, side_), clean, channels)
        elif arm == "blocks":
            prep = lambda role: _blocks(raw(role), filtered=not stored)
        elif stored:
            prep = lambda role: _stored_whole(two_phase.bag_tensors(raw(role)))
        else:
            prep = lambda role: sc.prepared(two_phase.bag_tensors(raw(role)))
        side = (lambda role: euclidean_aligned(prep(role))) if ea else prep
        train, test = side("inner_train" if tuned else A.REFIT_ROLE), side("test")
        valid = side("inner_validation") if tuned else []
        with _trunk(kind):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            c = int(train[0].windows.shape[1])
            m = model_store.build_model("MIL-CREDA", c, c, hyper).to(device)
            if kind == "spectral":
                m.source_trunk.norm = sc.standard_of(m.source_trunk, train)
            elif kind != "eegnet":
                m.source_trunk.norm = learned_trunks.input_standard_of(m.source_trunk, train)
            kept.append(float(np.mean([len(b.windows) for b in train])))
            renyi = learned_trunks.renyi_loss() if kind == "tgarstf" else contextlib.nullcontext()
            # The explicit L2 enters every step; scale it so each real subject gets the same
            # regularization per epoch whether it trains as one bag or as several blocks.
            l2 = sc.L2 * len({b.subject.split("#")[0] for b in train}) / max(1, len(train))
            with renyi, explicit_l2(l2):
                if adversary is None:
                    two_phase.train_source_phase(m, train, [], hyper, g, device, seed=seed, fold=r, transfer=transfer)
                else:                         # one loop for every arm of the adversary comparison
                    train_phase_one(m, train, hyper, g, device, adversary=bool(adversary))
            def subject_scores(units):
                q = np.nan_to_num(two_phase.evaluate(m, units, "source", device)[:, 1].detach().cpu().numpy(), nan=.5)
                per = {}                      # blocks count as subjects (v64); whole bags are subjects
                for b, v in zip(units, q):
                    key = b.subject if arm in ("blocks", "cleanblocks") else b.subject.split("#")[0]
                    per.setdefault(key, [b.label, []])[1].append(float(v))
                return [lab for lab, _ in per.values()], [float(np.mean(vs)) for _, vs in per.values()]
            ty, tp = subject_scores(test)
            ry, rp = subject_scores(train)
            train_y += ry
            train_p += rp
            cut = 0.5
            if tuned:
                vy, vp = subject_scores(valid)
                grid = sorted(set(vp))
                cands = [(grid[i] + grid[i + 1]) / 2 for i in range(len(grid) - 1)] or [0.5]
                cut = max(cands, key=lambda t: (balanced_accuracy_score(vy, np.asarray(vp) > t), -abs(t - .5)))
            y += ty
            p += tp
            cuts += [cut] * len(ty)
            if target:
                gy, gp = subject_scores(prep("test", "target"))
                ty_all += gy
                tp_all += gp
                tcuts += [cut] * len(gy)
        print(kind, transfer, seed, "rotation", r, "done", flush=True)
    return {"source": transfer.split("->")[0], "kind": kind, "arm": arm, "ea": bool(ea),
            "filter": "stored 0.2-45 Hz" if stored else "1-45 Hz + CAR", "clean": clean,
            "rotations": list(rotations), "meanTrainWindows": round(float(np.mean(kept)), 1),
            "seed": seed, "epochs": epochs,
            "threshold": threshold, "cuts": sorted(set(round(c, 4) for c in cuts)),
            "l2": round(float(l2), 6), "adversary": adversary, "trainAUC": round(float(roc_auc_score(train_y, train_p)), 3),
            "predictedPositive": round(float(np.mean(np.asarray(p) > .5)), 3),
            "AUC": round(float(roc_auc_score(y, p)), 3),
            "BA": round(float(balanced_accuracy_score(y, np.asarray(p) > .5)), 3),
            "BAatCut": round(float(balanced_accuracy_score(y, np.asarray(p) > np.asarray(cuts))), 3),
            "predictedPositiveAtCut": round(float(np.mean(np.asarray(p) > np.asarray(cuts))), 3),
            "n": len(y),
            **({"target": transfer.split("->")[1], "targetN": len(ty_all),
                "targetAUC": round(float(roc_auc_score(ty_all, tp_all)), 3),
                "targetBA": round(float(balanced_accuracy_score(ty_all, np.asarray(tp_all) > .5)), 3),
                "targetPredictedPositive": round(float(np.mean(np.asarray(tp_all) > .5)), 3),
                "targetBAatCut": round(float(balanced_accuracy_score(ty_all, np.asarray(tp_all) > np.asarray(tcuts))), 3),
                "targetPredictedPositiveAtCut": round(float(np.mean(np.asarray(tp_all) > np.asarray(tcuts))), 3)}
               if target else {})}


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, kind: str, seed: int = 1,
          epochs: int = 100, arms=("whole",), eas=(False,), stored: bool = False,
          rotations=(0, 1, 2, 3), cleans=(None,), threshold: str = "fixed", target: bool = False,
          adversaries=(None,),
          workdir: str | Path | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register(cfg["strategies"])
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ), placement)
    device = torch.device(cfg.get("device", "cpu"))
    bags = campaign.load_bags(cfg)
    mode = environ.get(remote_units.RUN_MODE_ENV)
    started = time.time()
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness:
        results = []
        for arm in arms:
            for ea in eas:
                for clean in cleans:
                  for adv in adversaries:
                    results.append(run_source(transfer, cfg, bags, device, kind, int(seed),
                                              int(epochs), arm, bool(ea), bool(stored),
                                              tuple(int(r) for r in rotations), clean, threshold,
                                              bool(target), adv))
                    print(json.dumps(results[-1]), flush=True)
    result = results[0] if len(results) == 1 else {"arms": results}
    (workdir / "learned-phase1.json").write_text(json.dumps({**result, "seconds": round(time.time() - started),
                                                             "environment": environment}, default=str), encoding="utf-8")
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"], "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": transfer, "units": [transfer],
                                "complete": True, "gpu": witness.evidence(), "estimates": {}},
                   "environment": environment, "runs": {"ran": [transfer], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return {"ran": [transfer], "skipped": [], "environment": environment}
