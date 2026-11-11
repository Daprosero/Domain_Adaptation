"""Instrumented EEGNet phase 1: where does training differ between a source where EEGNet
generalizes (Robots) and one where it does not (L11)? Diagnostic, outside the plan's protocol.

The training step is two_phase.train_source_phase's, unchanged in effect (same sigma, Adam,
class-interleaved batches, Eq. 21 plus the explicit L2, constraints after every step); the
only difference is that the gradient is assembled as grad(Eq. 21) + grad(L2) so the two can be
measured apart. Recorded:

- per epoch: Eq. 21 and L2 values, per-layer gradient norms of each part, the update-to-weight
  ratio, weight norms, the head row norm and how many depthwise rows sit at their max-norm;
- at checkpoint epochs, with running BatchNorm statistics: per-layer activation spread on the
  training and validation bags, the bag representation norm, the output range, BA at 0.5 and
  AUC on train, validation and test, a logistic probe on the bag representations (fitted on
  train, scored on validation and test), and the share of window-embedding variance explained
  by subject and by class;
- once: per-subject input amplitude and its AUC for the class, and at the end the temporal
  filters' peak frequencies.
Writes trace-check.json; nothing goes to the model store.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, eegnet, model_store, remote_units, two_phase, v63_setup
from MIL_CREDA_EEG import learned_phase1 as L
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.mil_creda_model import source_term as plain_source_term
from MIL_CREDA_EEG.patience_exercise import GpuWitness

CHECKPOINTS = (0, 1, 2, 5, 10, 20, 30, 50, 75, 99)
LAYERS = ("conv_temporal", "bn1", "depthwise", "bn2", "pool1", "separable_point", "bn3", "pool2")
FS = 128.0


def _r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def _safe_auc(y, s):
    return _r(roc_auc_score(y, s), 3) if len(set(y)) > 1 else None


def _groups(model):
    t = model.source_trunk
    return {"conv_temporal": [t.conv_temporal.weight], "depthwise": [t.depthwise.weight],
            "separable": [t.separable_depth.weight, t.separable_point.weight],
            "batchnorm": [p for bn in (t.bn1, t.bn2, t.bn3) for p in bn.parameters()],
            "head": list(model.head.parameters()),
            "attention": list(model.attention_parameters())}


def _l2_weights(model, batchnorm=True):
    """The explicit L2's weights, as phase_one_sweep.explicit_l2; `batchnorm` False leaves the
    BatchNorm scales out."""
    t = model.source_trunk
    skip = set() if batchnorm else {id(p) for bn in (t.bn1, t.bn2, t.bn3) for p in bn.parameters()}
    return [p for n, p in t.named_parameters() if "weight" in n and id(p) not in skip] + [model.head.linear.weight]


def prepared(units, channel_scale=False, highpass=None):
    """Label-free, per recording. `highpass` (Hz): zero-phase 4th-order Butterworth high-pass on
    every window. `channel_scale`: every channel of a subject divided by its robust amplitude
    (0.7413 x interquartile range over all of that subject's windows and samples)."""
    if not channel_scale and not highpass:
        return units
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, float(highpass), btype="highpass", fs=FS, output="sos") if highpass else None
    by = {}
    for u in units:
        by.setdefault(u.subject.split("#")[0], []).append(u)
    out = {}
    for subject, us in by.items():
        xs = [u.windows.double().numpy() for u in us]
        if sos is not None:
            xs = [sosfiltfilt(sos, x, axis=-1).copy() for x in xs]
        if channel_scale:
            flat = np.concatenate(xs).transpose(1, 0, 2).reshape(xs[0].shape[1], -1)
            q75, q25 = np.percentile(flat, [75, 25], axis=1)
            amp = np.maximum(0.7413 * (q75 - q25), 1e-6)[None, :, None]
            xs = [x / amp for x in xs]
        for u, x in zip(us, xs):
            out[id(u)] = two_phase.BagTensor(u.subject, u.condition, u.label, torch.from_numpy(x).float())
    return [out[id(u)] for u in units]


def window_peaks(units):
    """Each window's peak in robust units: max over channels and samples of |x| divided by that
    channel's robust amplitude (0.7413 x interquartile range) over the subject's own windows."""
    by = {}
    for u in units:
        by.setdefault(u.subject.split("#")[0], []).append(u)
    out = {}
    for us in by.values():
        flat = torch.cat([u.windows for u in us]).double().numpy()
        f = flat.transpose(1, 0, 2).reshape(flat.shape[1], -1)
        q75, q25 = np.percentile(f, [75, 25], axis=1)
        amp = np.maximum(0.7413 * (q75 - q25), 1e-6)[None, :, None]
        for u in us:
            out[id(u)] = (np.abs(u.windows.double().numpy()) / amp).max(axis=(1, 2))
    return [out[id(u)] for u in units]


def rejected(units, threshold, keep_min=10):
    """Label-free amplitude rejection per recording: drop every window with any channel above
    `threshold` robust units (window_peaks). A subject keeps at least its `keep_min` windows with
    the lowest peaks, so nobody leaves the data."""
    if not threshold:
        return units, None
    out, kept = [], []
    for u, pk in zip(units, window_peaks(units)):
        idx = np.where(pk <= threshold)[0]
        if len(idx) < min(keep_min, len(pk)):
            idx = np.sort(np.argsort(pk)[:min(keep_min, len(pk))])
        kept.append(len(idx) / len(pk))
        out.append(two_phase.BagTensor(u.subject, u.condition, u.label, u.windows[torch.as_tensor(idx)]))
    return out, kept


def _attention_vs_peak(model, units, device):
    """Spearman correlation, per bag, between a window's attention weight and its peak in robust
    units; the mean over bags (negative: artifact windows weigh less)."""
    from scipy.stats import spearmanr
    H = model.embed([u.windows.to(device) for u in units], "source", two_phase.EVAL_CHUNK)
    rs = []
    for (_, w), pk in zip(model.bag_pairs(H), window_peaks(units)):
        w = w.detach().cpu().numpy().ravel()
        if np.std(w) > 0 and np.std(pk) > 0:
            rs.append(spearmanr(w, pk).correlation)
    return _r(np.mean(rs), 3) if rs else None


def _norm(ts):
    ts = [t for t in ts if t is not None]
    return float(torch.sqrt(sum((t.detach().double() ** 2).sum() for t in ts))) if ts else 0.0


def _variance_share(H, subjects, labels):
    """Share of the total variance of window embeddings explained by subject and by class."""
    X = H - H.mean(0)
    total = float((X ** 2).sum())
    if total <= 0:
        return 0.0, 0.0
    def between(keys):
        out = 0.0
        for k in set(keys):
            idx = np.asarray(keys) == k
            out += idx.sum() * float((X[idx].mean(0) ** 2).sum())
        return out / total
    return between(subjects), between(labels)


@torch.no_grad()
def _snapshot(model, splits, device):
    model.eval()
    acts = {}
    hooks = []
    trunk = model.source_trunk
    for name in LAYERS:
        def hook(_m, _i, out, name=name):
            acts.setdefault(name, []).append(out.detach().float())
        hooks.append(getattr(trunk, name).register_forward_hook(hook))
    out = {}
    reps = {}
    scores = {}
    try:
        for split, units in splits.items():
            acts.clear()
            H = model.embed([u.windows.to(device) for u in units], "source", two_phase.EVAL_CHUNK)
            Z = model.represent(model.bag_pairs(H))
            G = model.classify(Z)[:, 1].cpu().numpy()
            y = [u.label for u in units]
            Zn = Z.cpu().numpy()
            reps[split] = (Zn, y)
            scores[split] = (np.asarray(y), G)
            rec = {"above": _r(np.mean(G > .5), 3), "pMin": _r(G.min()), "pMax": _r(G.max()),
                   "BA": _r(balanced_accuracy_score(y, G > .5), 3), "AUC": _safe_auc(y, G),
                   "zNorm": _r(np.linalg.norm(Zn, axis=1).mean(), 3)}
            if split in ("train", "valid"):
                rec["layerStd"] = {k: _r(torch.cat(v).std().item(), 4) for k, v in acts.items()}
                Hw = torch.cat(H).cpu().numpy()
                subj = [u.subject.split("#")[0] for u, h in zip(units, H) for _ in range(h.shape[0])]
                lab = [u.label for u, h in zip(units, H) for _ in range(h.shape[0])]
                s, c = _variance_share(Hw, subj, lab)
                rec["subjectVarShare"], rec["classVarShare"] = _r(s, 3), _r(c, 3)
                rec["windowEmbeddingStd"] = _r(Hw.std(), 4)
                rec["attentionMaxWeight"] = _r(float(np.mean([w.max().item() for _, w in model.bag_pairs(H)])), 4)
                rec["attentionVsPeak"] = _attention_vs_peak(model, units, device)
            out[split] = rec
    finally:
        for h in hooks:
            h.remove()
    vy, vp = scores["valid"]                 # the cut that maximizes BA on the validation subjects
    grid = np.sort(np.unique(vp))
    cands = list((grid[:-1] + grid[1:]) / 2) or [0.5]
    cut = max(cands, key=lambda t: (balanced_accuracy_score(vy, vp > t), -abs(t - .5)))
    ty, tp = scores["test"]
    out["calibrated"] = {"cut": _r(cut), "validBA": _r(balanced_accuracy_score(vy, vp > cut), 3),
                         "testBA": _r(balanced_accuracy_score(ty, tp > cut), 3),
                         "testPositive": _r(np.mean(tp > cut), 3)}
    out["testScores"] = [[u.subject.split("#")[0], int(u.label), _r(float(q), 5)]
                         for u, q in zip(splits["test"], tp)]
    Ztr, ytr = reps["train"]
    sc_ = StandardScaler().fit(Ztr)
    probe = LogisticRegression(C=1.0, max_iter=2000).fit(sc_.transform(Ztr), ytr)
    for split in ("valid", "test"):
        Z, y = reps[split]
        q = probe.predict_proba(sc_.transform(Z))[:, 1]
        out[split]["probeAUC"], out[split]["probeBA"] = _safe_auc(y, q), _r(balanced_accuracy_score(y, q > .5), 3)
    return out


def _peak_frequencies(model):
    w = model.source_trunk.conv_temporal.weight.detach().cpu().double().numpy().reshape(model.source_trunk.f1, -1)
    n = 512
    spec = np.abs(np.fft.rfft(w, n=n, axis=1))
    freqs = np.fft.rfftfreq(n, 1 / FS)
    return [round(float(freqs[i]), 1) for i in spec.argmax(1)]


def _input_amplitude(splits):
    out = {}
    for split, units in splits.items():
        amp = [float(u.windows.std()) for u in units]
        y = [u.label for u in units]
        out[split] = {"meanStd": _r(np.mean(amp), 3), "classAUC": _safe_auc(y, amp),
                      "class0": _r(np.mean([a for a, l in zip(amp, y) if l == 0]), 3),
                      "class1": _r(np.mean([a for a, l in zip(amp, y) if l == 1]), 3)}
    return out


def trace(transfer, cfg, bags, device, seed, epochs, rotation=0, l2=sc.L2, channel_scale=False,
          highpass=None, l2_batchnorm=True, arm="", reject=None, attention=None, trunk_lr_scale=1.0, dropout=None):
    """`reject`: amplitude threshold in robust units (rejected), applied to all of a subject's
    windows before the evenly spaced selection; `attention`: {"temperature", "gamma"} for the
    phase-1 attention (default: the neutral aggregation, temperature 10, gamma 0)."""
    a = A.assign_from_bags(v63_setup.transfer(transfer), bags)
    att = attention or {}
    hyper = two_phase.check_hyper({**L.HYPER, "epochs1": epochs, "bagsPerStep": cfg["schedule"]["bagsPerStep"],
                                   "attentionTemperature": float(att.get("temperature", L.HYPER["attentionTemperature"])),
                                   "attentionGamma": float(att.get("gamma", L.HYPER["attentionGamma"]))})
    kept = {}

    def role(r):
        full = prepared(two_phase.bag_tensors(
            A.side_bags(a, bags, "source", rotation, r, split=0) if r.startswith("inner")
            else A.side_bags(a, bags, "source", rotation, r)), channel_scale, highpass)
        full, k = rejected(full, reject)
        if k is not None:
            kept[r] = {"median": _r(np.median(k), 3), "min": _r(np.min(k), 3)}
        return L._stored_whole(full)
    splits = {"train": role("inner_train"), "valid": role("inner_validation"), "test": role("test")}
    train = splits["train"]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    c = int(train[0].windows.shape[1])
    with eegnet.configured(dropout=eegnet.DROPOUT if dropout is None else float(dropout)):
        m = model_store.build_model("MIL-CREDA", c, c, hyper).to(device)
    sigma = two_phase.phase_one_sigma(m, train, hyper["sigmaMultiplier"], device)
    params = [*m.source_trunk.parameters(), *m.attention_parameters(), *m.head.parameters()]
    trunk_ids = {id(p) for p in m.source_trunk.parameters()}
    groups_ = [{"params": [p for p in params if id(p) not in trunk_ids]}]
    if trunk_lr_scale > 0:                   # 0 freezes the trunk at its initialization
        groups_.append({"params": [p for p in params if id(p) in trunk_ids], "lr": hyper["lr1"] * float(trunk_lr_scale)})
    opt = torch.optim.Adam(groups_, lr=hyper["lr1"], weight_decay=hyper["wd1"])
    groups = _groups(m)
    l2w = _l2_weights(m, l2_batchnorm)
    l2ids = {id(p) for p in l2w}
    record = {"arm": arm, "dropout": float(m.source_trunk.dropout), "trunkLrScale": float(trunk_lr_scale), "reject": reject, "windowsKept": kept,
              "attentionTemperature": hyper["attentionTemperature"], "attentionGamma": hyper["attentionGamma"],
              "channelScale": bool(channel_scale), "highpass": highpass,
              "l2BatchNorm": bool(l2_batchnorm), "transfer": transfer, "seed": seed, "rotation": rotation, "l2": l2, "epochs": epochs,
              "lr1": hyper["lr1"], "bagsPerStep": hyper["bagsPerStep"], "sigma": _r(sigma["sigma"]),
              "subjects": {k: len(v) for k, v in splits.items()},
              "positives": {k: _r(np.mean([u.label for u in v]), 3) for k, v in splits.items()},
              "windowsPerBag": _r(np.mean([u.windows.shape[0] for u in train]), 1),
              "channels": c, "input": _input_amplitude(splits), "epochsLog": [], "snapshots": {}}
    for epoch in range(epochs):
        m.train()
        acc = {"ce": 0.0, "l2": 0.0, "steps": 0}
        gce = {k: 0.0 for k in groups}
        gl2 = {k: 0.0 for k in groups}
        before = {k: [p.detach().clone() for p in v] for k, v in groups.items()}
        for batch in two_phase.interleaved_batches(train, hyper["bagsPerStep"], g):
            windows = [train[i].windows.to(device) for i in batch]
            labels = torch.tensor([train[i].label for i in batch], device=device)
            G = m.predict(windows, "source")
            ce = plain_source_term(m, G, labels)
            pen = l2 * sum((w ** 2).sum() for w in l2w)
            grads = torch.autograd.grad(ce, params, allow_unused=True)
            opt.zero_grad()
            for p, gr in zip(params, grads):
                gp = torch.zeros_like(p) if gr is None else gr
                if id(p) in l2ids and l2:
                    gp = gp + 2 * l2 * p.detach()
                p.grad = gp
            ids = {id(p): gr for p, gr in zip(params, grads)}
            for k, ps in groups.items():
                gce[k] += _norm([ids.get(id(p)) for p in ps])
                gl2[k] += _norm([2 * l2 * p.detach() for p in ps if id(p) in l2ids]) if l2 else 0.0
            opt.step()
            m.constrain("source_trunk", "head", "attention")
            if m.sigma != sigma["sigma"]:
                raise two_phase.SigmaChanged("sigma moved during phase 1")
            acc["ce"] += float(ce.detach())
            acc["l2"] += float(pen.detach())
            acc["steps"] += 1
        n = max(acc["steps"], 1)
        dw = {k: _norm([p.detach() - q for p, q in zip(v, before[k])]) / max(_norm(v), 1e-12)
              for k, v in groups.items()}
        dep = m.source_trunk.depthwise.weight.detach().flatten(1).norm(dim=1)
        record["epochsLog"].append({
            "epoch": epoch, "steps": acc["steps"], "ce": _r(acc["ce"] / n, 5), "l2": _r(acc["l2"] / n, 5),
            "gradCE": {k: _r(v / n, 6) for k, v in gce.items()},
            "gradL2": {k: _r(v / n, 6) for k, v in gl2.items()},
            "updateRatio": {k: _r(v, 6) for k, v in dw.items()},
            "weightNorm": {k: _r(_norm(v), 4) for k, v in groups.items()},
            "headRowNorm": _r(float(m.head.linear.weight.detach().norm(dim=1).max()), 4),
            "depthwiseAtCap": _r(float((dep >= 0.999 * float(dep.max())).float().mean()), 3)})
        if epoch in CHECKPOINTS or epoch == epochs - 1:
            record["snapshots"][str(epoch)] = _snapshot(m, splits, device)
            print(arm, transfer, "epoch", epoch, json.dumps({k: (v["BA"], v["AUC"]) if "BA" in v else v for k, v in
                                                        record["snapshots"][str(epoch)].items()}), flush=True)
    record["temporalPeaksHz"] = _peak_frequencies(m)
    return record


def smoke(config: Mapping, product: str | Path | None = None, *, transfers=("Robots->CogniFit", "L11->L1"),
          seed: int = 1, epochs: int = 100, rotation: int = 0, l2: float = sc.L2, arms=(None,),
          rotations=None,
          workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
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
        runs = []
        for arm in arms:                     # each {"name", "l2", "channelScale", "highpass", "l2BatchNorm"}
            arm = arm or {}
            for t, rot in [(t, r) for t in transfers for r in (rotations or [rotation])]:
                runs.append(trace(t, cfg, bags, device, int(seed), int(epochs), int(rot),
                                  float(arm.get("l2", l2)), bool(arm.get("channelScale", False)),
                                  arm.get("highpass"), bool(arm.get("l2BatchNorm", True)), arm.get("name", ""),
                                  arm.get("reject"), arm.get("attention"), float(arm.get("trunkLrScale", 1.0)), arm.get("dropout")))
    (workdir / "trace-check.json").write_text(json.dumps({"runs": runs, "seconds": round(time.time() - started),
                                                         "environment": environment}, default=str), encoding="utf-8")
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"], "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": transfers[0], "units": list(transfers),
                                "complete": True, "gpu": witness.evidence(), "estimates": {}},
                   "environment": environment, "runs": {"ran": list(transfers), "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return {"ran": list(transfers), "skipped": [], "environment": environment}
