"""Plan v63: read what the campaign left, for the mechanism and qualitative analyses.

Reads the per-(method, seed) refit records (`v63_campaign.run_record_path`), rebuilds saved
models from `Models/v63` for inference only, and reloads the processed windows the records
point at. Nothing here trains, adapts or selects.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import model_store, two_phase
from MIL_CREDA_EEG import v63_campaign as C
from MIL_CREDA_EEG import v63_setup as V

__provenance__ = {
    "revision": V.REVISION,
    "sections": ["5.3", "6.1", "6.2", "6.4"],
    "experiments": ["C-dispositivo", "K1-atencion", "K2-emparejamiento", "K4-conectividad"],
    "invariants": ["a_record_set_holds_each_target_subject_once_per_seed"],
}

MIL_CREDA = "MIL-CREDA"
NO_CONSENSUS = "MIL-CREDA [own without consensus]"
GLOBAL_ONLY = "MIL-CREDA [global term only]"
LOCAL_ONLY = "MIL-CREDA [local term only]"


def records(product: Path, t, method: str) -> dict:
    """(rotation, seed) -> record of `method` on transfer `t`; failed records left out."""
    root = Path(product) / C.RESULTS / V.experiment_of(t) / "runs" / model_store.transfer_slug(t)
    out = {}
    for path in sorted(root.glob(f"outer*/{model_store.name_slug(method)}__seed*.json")):
        r = json.loads(path.read_text())
        if r.get("failed") or r.get("method") != method:
            continue
        out[(int(r["rotation"]), int(r["seed"]))] = r
    return out


def by_subject(recs: Mapping) -> dict:
    """subject -> {"seeds": {seed: index into its record}, "rotation", "label", ...}."""
    out: dict = {}
    for (r, seed), rec in recs.items():
        for k, s in enumerate(rec["subjects"]):
            entry = out.setdefault(s, {"rotation": r, "label": int(rec["labels"][k]),
                                       "condition": rec["conditions"][k], "seeds": {}})
            if entry["rotation"] != r or seed in entry["seeds"]:
                raise V.V63Error(f"{s} is a target test subject of two rotations or twice in "
                                 f"one seed")
            entry["seeds"][seed] = (rec, k)
    return out


def seed_mean(entry: Mapping, key: str) -> np.ndarray:
    return np.mean([np.asarray(rec[key][k], float) for rec, k in entry["seeds"].values()], 0)


def predicted_class(entry: Mapping) -> int:
    return int(float(np.mean([rec["probabilities"][k] for rec, k in entry["seeds"].values()]))
               > 0.5)


class Store:
    """Saved models and processed windows of one product and its bag set."""

    def __init__(self, product: Path, bags):
        self.product, self.bags = Path(product), bags
        self.prep = V.Prepared(bags)

    def raw_windows(self, subject: str, condition: str, index: Sequence[int],
                    channels: Sequence[str] | None) -> np.ndarray:
        raw = self.bags.get(subject, condition)
        w = np.asarray(raw.windows(), float)[list(index)]
        if channels is not None:
            names = list(raw.channels)
            w = w[:, [names.index(c) for c in channels]]
        return w

    def prepared(self, t, subject: str, condition: str) -> two_phase.BagTensor:
        return self.prep([self.bags.get(subject, condition)], V.channels_of(t))[0]

    def phase_one(self, t, rotation: int, seed: int, group: str = "MIL-CREDA"):
        path = C.model_path(self.product, t, rotation, seed, 1, group)
        state, meta = V.load_state(path, self.bags.partition_sha256)
        return C.rebuild_phase_one(group, (state, meta)), meta

    def mil_creda(self, t, rotation: int, seed: int, method: str = MIL_CREDA,
                  knobs: Mapping | None = None):
        """A saved phase-2 MIL-CREDA (or K3 variant) model, at its phase-1 sigma."""
        path = C.model_path(self.product, t, rotation, seed, 2, method)
        state, meta = V.load_state(path, self.bags.partition_sha256)
        _, p1 = self.phase_one(t, rotation, seed)
        with V.standardization(V.standard_of_state(state)):
            model = model_store.build_model("MIL-CREDA", meta["channels"]["source"],
                                            meta["channels"]["target"], meta["hyper"])
        model.load_state_dict(state)
        model.fix_sigma(p1["sigma"])
        model.attention_gamma = float((knobs or {}).get("gamma", meta["hyper"]["attentionGamma"]))
        model.attention_temperature = float(meta["hyper"]["attentionTemperature"])
        return model.eval()

    def bridged(self, t, rotation: int, seed: int):
        """Phase 2's start: the MIL-CREDA phase-1 checkpoint, target trunk copied from the
        source trunk, with the phase-2 aggregation (Sec. 7.3)."""
        model, _ = self.phase_one(t, rotation, seed)
        model.attention_gamma = V.PHASE_TWO_AGGREGATION["attentionGamma"]
        model.attention_temperature = V.PHASE_TWO_AGGREGATION["attentionTemperature"]
        two_phase.bridge(model)
        return model.eval()

    def probe(self, t, rotation: int) -> dict:
        """Sec. 6.2's probe subjects of `rotation` (test-fold subjects holding both
        conditions) -> (their target-condition bag, their source-condition bag). The other
        condition is read through `assignment.probe_bag`, the guarded reader."""
        a = A.assign_from_bags(t, self.bags)
        out = {}
        for s in sorted(a.probe_subjects(rotation)):
            own = self.bags.get(s, a.conditions[s])
            other = A.probe_bag(a, self.bags, s, rotation)
            tgt, src = (own, other) if a.conditions[s] == t.target else (other, own)
            out[s] = tuple(self.prep([b], V.channels_of(t))[0] for b in (tgt, src))
        return out


@torch.no_grad()
def kernel_distances(model, target: Sequence[two_phase.BagTensor],
                     source: Sequence[two_phase.BagTensor]) -> np.ndarray:
    """(target, source) distances in the bag-kernel space at the model's frozen sigma:
    sqrt(K(t,t) + K(s,s) - 2 K(t,s))."""
    pt = model.bag_pairs(model.embed([b.windows for b in target], "target", two_phase.EVAL_CHUNK))
    ps = model.bag_pairs(model.embed([b.windows for b in source], "source", two_phase.EVAL_CHUNK))
    from MIL_CREDA_EEG import mil_creda_model as M
    Ktt = torch.diagonal(M.bag_kernel_matrix(pt, pt, model.sigma))
    Kss = torch.diagonal(M.bag_kernel_matrix(ps, ps, model.sigma))
    Kts = M.bag_kernel_matrix(pt, ps, model.sigma)
    d2 = Ktt[:, None] + Kss[None, :] - 2 * Kts
    return torch.sqrt(torch.clamp(d2, min=0)).double().cpu().numpy()


def wilcoxon_row(comparison: str, differences: Sequence[float], *, hypothesis: str,
                 experiment: str, family: str, **extra) -> dict:
    from MIL_CREDA_EEG import statistics
    w = statistics.wilcoxon_signed_rank(differences)
    return statistics.significance_row(
        comparison=comparison, test="Wilcoxon signed-rank over subjects",
        statistic=w["statistic"], p_value=w["p_value"], effect_size=w["rank_biserial"],
        effect_size_name="rank-biserial correlation", ci_low=w["ci_low"], ci_high=w["ci_high"],
        n=w["n"], hypothesis=hypothesis, experiment=experiment, family=family,
        hodges_lehmann=w["hodges_lehmann"], median=w["median"], **extra)


def holm_family(rows: list[dict], family: str) -> list[dict]:
    from MIL_CREDA_EEG import statistics
    return statistics.apply_holm(rows, family) if rows else rows


def median_rule(values: Mapping[str, float]) -> str | None:
    """The subject whose value is the median (the lower middle on an even count); fixed
    before any figure is looked at."""
    finite = sorted((v, s) for s, v in values.items() if v is not None and np.isfinite(v))
    return finite[(len(finite) - 1) // 2][1] if finite else None


def group(items: Sequence[Mapping], key: str) -> dict:
    out = defaultdict(list)
    for x in items:
        out[x[key]].append(x)
    return out
