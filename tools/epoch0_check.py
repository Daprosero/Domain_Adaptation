"""Plan v58 Sec. 4.5: the epoch-0 toy-set check of the phase-2 centroid rule.

Protocol (verbatim, experiments plan v58, Sec. 4.5): "Antes de la corrida completa se
comprueba en el toy set que la regla castiga la falta de adaptación. En cada transferencia
del toy set se calcula el puntaje de la regla en la época 0 de la fase 2, cuando el
codificador objetivo todavía es la copia del codificador fuente, y en la época que la regla
elige. El criterio se declara antes de medir: la regla se considera adecuada en ese conjunto
si el puntaje de la época 0 es menor que el de la época elegida en la mayoría de las
transferencias. Si no lo es, se dice así antes de lanzar la corrida completa y la regla
vuelve a deliberación."

What this adapter does, and what it leaves alone:

- The toy set is the one the proof of concept trained on: `toy_run`'s three families
  (`lunas`, `toroide`, `rollo`; each one a source -> target transfer), the `FULL` scale,
  the BatchNorm MLP, the single declared arm `TGL`, and the two phases of
  `MIL_CREDA_Benchmark.poc._two_phases`, run unchanged. Each family trains at the archived
  search's configuration for it (`MIL-CREDA/Results/Toyset/full/barrido.json`, `hp1` and
  `hp2`; for `rollo` these are exactly `poc.ARCHIVED_ROLLO_HYPER` and
  `poc.SIGMA_MULTIPLIER`).
- Nothing in the toy's training changes. Two read-only hooks observe it:
  `poc.train_phase_two` is wrapped to score epoch 0 (theta_t is still the copy of
  theta_s, before any phase-2 optimizer step), and `toy_run.effective_classes`, which the
  phase-2 loop calls once at the end of every epoch, is wrapped to score that epoch and
  then return the toy's own `cef` unchanged. The toy still restores its own `max(cef)`
  epoch; the centroid series is a reading beside it.
- The score is the EEG campaign's own centroid rule, `MIL_CREDA_EEG.two_phase.
  centroid_counts` + `balanced_accuracy_of_counts`, on the toy's representations: the
  adapted model (theta_t, the frozen attention and head) predicts the target VALIDATION
  role's bags (pseudo-label = argmax) and keeps each bag's Z, the vector the head reads;
  one centroid per pseudo-class; the source VALIDATION role's bags go through the frozen
  source encoder and the same attention; each is assigned to the nearest centroid and
  compared with its real source label. The toy has one validation role, not three inner
  splits, so the counts are those of that one split. No target label is read.
- The chosen epoch is the rule's argmax over the phase-2 epochs 1..E (ties to the
  earliest epoch, as in `search.select`), the epochs the campaign's search scores.

Declared before measuring (this module's `CRITERION`): one row per (transfer, seed); a
transfer passes when "epoch-0 score < chosen-epoch score" holds in the majority of its
seeds; the rule is "adecuada" on the toy set when the majority of the transfers pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import multiprocessing
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from MIL_CREDA_Benchmark import config, poc, toy_run  # noqa: E402
from MIL_CREDA_EEG import two_phase  # noqa: E402

PROTOCOL = "experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md, Sec. 4.5"
FAMILIES = ("lunas", "toroide", "rollo")
#: The toy set's declared report seeds (`barrido.json`, `escala.report_seeds`).
SEEDS = tuple(range(10))
ARM = "TGL"
ARCHIVE = ROOT / "MIL-CREDA" / "Results" / "Toyset" / "full" / "barrido.json"
OUT = ROOT / "MIL-CREDA" / "Results" / "epoch0_check"
CRITERION = ("Declared before measuring: one row per (transfer, seed); a transfer passes when "
             "the epoch-0 centroid score is strictly lower than the score at the epoch the "
             "rule chooses (argmax over phase-2 epochs 1..E, earliest on ties) in the "
             "majority of its seeds; the rule is 'adecuada' on the toy set when the majority "
             "of the transfers pass (plan v58, Sec. 4.5).")


# ------------------------------------------------------------------ the score

@torch.no_grad()
def _bag_tensor(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    items = [dataset[i] for i in range(len(dataset))]
    return (torch.stack([item[0] for item in items]),
            torch.tensor([int(item[1]) for item in items]))


@torch.no_grad()
def toy_representations(model, bags: torch.Tensor, domain: str) -> tuple[np.ndarray, np.ndarray]:
    """(Z, probabilities) of `bags` through the `domain` encoder, the shared attention
    and the head: Z is the bag representation the head reads (Eq. 19)."""
    if domain == "target":
        embeddings = model.target_embeddings(bags)
    elif domain == "source":
        embeddings = model.instance_embeddings(bags)
    else:
        raise ValueError(f"domain must be 'source' or 'target', got {domain!r}")
    Z, _ = model.bag_representations(embeddings, model.sigma)
    probabilities = F.softmax(model.head(Z), dim=-1)
    return Z.double().cpu().numpy(), probabilities.double().cpu().numpy()


def centroid_score(model, target_bags: torch.Tensor, source_bags: torch.Tensor,
                   source_labels, classes: int) -> dict:
    """The v58 centroid rule on one split: `two_phase.centroid_counts` over the toy
    model's representations. Read in `eval`, then the toy's phase-2 modes restored, as
    `toy_run.effective_classes` does. Reads no target label."""
    model.eval()
    try:
        Zt, Pt = toy_representations(model, target_bags, "target")
        Zs, _ = toy_representations(model, source_bags, "source")
    finally:
        toy_run.set_phase_two_modes(model)
    counts = two_phase.centroid_counts(Zt, Pt, Zs, np.asarray(source_labels), classes)
    return {"score": two_phase.balanced_accuracy_of_counts(counts),
            "pseudoClasses": int(len(set(Pt.argmax(axis=1).tolist()))),
            "counts": counts.tolist()}


def chosen_epoch(series) -> int:
    """The rule's epoch: argmax of the score over epochs 1..E, the earliest on a tie."""
    candidates = [(row["epoch"], row["score"]) for row in series if row["epoch"] >= 1]
    if not candidates:
        raise ValueError("no phase-2 epoch past epoch 0 was scored")
    best = max(score for _, score in candidates)
    return min(epoch for epoch, score in candidates if score == best)


@contextlib.contextmanager
def recording(series: list, material: dict, classes: int, device: torch.device):
    """Score epoch 0 and the end of every phase-2 epoch, without touching the toy's
    training: the wrapped functions call the originals and return what they return."""
    _, target_valid, _ = toy_run.bags.roles(material["target"])
    _, source_valid, _ = toy_run.bags.roles(material["source"])
    target_bags, _ = _bag_tensor(target_valid)  # labels discarded: never read
    source_bags, source_labels = _bag_tensor(source_valid)
    target_bags, source_bags = target_bags.to(device), source_bags.to(device)
    original_train, original_cef = poc.train_phase_two, toy_run.effective_classes

    def score(model, epoch):
        series.append({"epoch": epoch, **centroid_score(model, target_bags, source_bags,
                                                        source_labels.tolist(), classes)})

    def train_phase_two(model, *args, **kwargs):
        score(model, 0)
        return original_train(model, *args, **kwargs)

    def effective_classes(model, device_):
        value = original_cef(model, device_)
        score(model, len(series))
        series[-1]["cef"] = value
        return value

    poc.train_phase_two, toy_run.effective_classes = train_phase_two, effective_classes
    try:
        yield
    finally:
        poc.train_phase_two, toy_run.effective_classes = original_train, original_cef


@contextlib.contextmanager
def sigma_multiplier(value: float):
    declared = poc.SIGMA_MULTIPLIER
    poc.SIGMA_MULTIPLIER = value
    try:
        yield
    finally:
        poc.SIGMA_MULTIPLIER = declared


# --------------------------------------------------------------- one transfer

def family_hyper(family: str, archive: Path = ARCHIVE) -> tuple[dict, float]:
    """The archived search's configuration for `family` over `declared_hyper`, and its
    `sigmaMultiplier`."""
    stored = json.loads(Path(archive).read_text())
    hp1, hp2 = dict(stored["hp1"][family]), dict(stored["hp2"][family])
    multiplier = float(hp1.pop("sigmaMultiplier"))
    hp1.pop("kernelSigma", None)  # measured again by the run itself
    return {**toy_run.declared_hyper(poc.SCALE), **hp1, **hp2}, multiplier


def run_one(family: str, seed: int, scale=None, archive: Path = ARCHIVE) -> dict:
    """Both phases of the toy's `TGL` arm on (family, seed), with the centroid series."""
    torch.set_num_threads(1)
    scale = scale or poc.SCALE
    start = time.perf_counter()
    device = poc._device()
    hyper, multiplier = family_hyper(family, archive)
    familia = toy_run.build_family(family, seed, scale)
    material = toy_run.material_for(familia, seed, scale, poc.BACKBONE)
    series: list = []
    with toy_run.material_bound(familia, scale, poc.BACKBONE, hyper), \
            sigma_multiplier(multiplier), \
            recording(series, material, scale.classes, device):
        _, record, _ = poc._two_phases(ARM, material, familia, scale, hyper, device, seed)
    epochs = record["phaseTwo"]["steps"] // record["phaseTwo"]["stepsPerEpoch"]
    if [row["epoch"] for row in series] != list(range(epochs + 1)):
        raise RuntimeError(f"the centroid series is not epochs 0..{epochs}: "
                           f"{[row['epoch'] for row in series]}")
    chosen = chosen_epoch(series)
    return {"transfer": f"{family} source -> {family} target", "family": family,
            "seed": seed, "scale": scale.name, "arm": ARM, "sigmaMultiplier": multiplier,
            "epoch0Score": series[0]["score"], "chosenEpoch": chosen,
            "chosenScore": series[chosen]["score"],
            "epoch0Lower": bool(series[0]["score"] < series[chosen]["score"]),
            "cefEpoch": record["phaseTwo"]["bestEpoch"],
            "sourceOnlyTargetAccuracy": record["sourceOnlyTargetAccuracy"],
            "targetAccuracy": record["targetAccuracy"], "phaseOne": record["phaseOne"],
            "series": series, "seconds": time.perf_counter() - start}


def verdict(rows) -> dict:
    """`CRITERION` applied to the rows."""
    transfers = {}
    for row in rows:
        transfers.setdefault(row["family"], []).append(bool(row["epoch0Lower"]))
    per_transfer = {family: {"seeds": len(flags), "epoch0Lower": sum(flags),
                             "passes": sum(flags) > len(flags) / 2}
                    for family, flags in transfers.items()}
    passing = sum(v["passes"] for v in per_transfer.values())
    adequate = passing > len(per_transfer) / 2
    return {"perTransfer": per_transfer, "transfersPassing": passing,
            "transfers": len(per_transfer),
            "rowsEpoch0Lower": sum(bool(r["epoch0Lower"]) for r in rows), "rows": len(rows),
            "adequate": adequate,
            "statement": ("adecuada: the epoch-0 score is lower than the chosen epoch's in "
                          f"the majority of the transfers ({passing}/{len(per_transfer)})"
                          if adequate else
                          "NOT adecuada: the epoch-0 score is not lower than the chosen "
                          f"epoch's in the majority of the transfers ({passing}/"
                          f"{len(per_transfer)}); per plan v58 Sec. 4.5 this is said before "
                          "the full run is launched and the rule returns to deliberation")}


# ------------------------------------------------------------------- writing

def table(rows) -> str:
    lines = ["| Transfer | Seed | Score epoch 0 | Chosen epoch | Score at chosen epoch | "
             "Epoch 0 < chosen? |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['transfer']} | {r['seed']} | {r['epoch0Score']:.4f} | "
                     f"{r['chosenEpoch']} | {r['chosenScore']:.4f} | "
                     f"{'yes' if r['epoch0Lower'] else 'no'} |")
    return "\n".join(lines)


def write(rows, out: Path, meta: dict) -> dict:
    rows = sorted(rows, key=lambda r: (FAMILIES.index(r["family"])
                                       if r["family"] in FAMILIES else 99, r["seed"]))
    result = verdict(rows)
    out.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "criterion": CRITERION, "verdict": result, "rows": rows}
    (out / "epoch0_check.json").write_text(json.dumps(payload, indent=1) + "\n")
    per = "\n".join(f"| {f} | {v['epoch0Lower']}/{v['seeds']} | "
                    f"{'yes' if v['passes'] else 'no'} |"
                    for f, v in result["perTransfer"].items())
    md = (f"# Epoch-0 toy-set check of the phase-2 centroid rule\n\n"
          f"Protocol: {PROTOCOL}. Arm `{ARM}`, scale `{meta['scale']}`, device "
          f"`{meta['device']}`, torch {meta['torch']}, seeds {meta['seeds']}. Measured "
          f"{meta['measuredAt']}.\n\n{CRITERION}\n\n## Per (transfer, seed)\n\n{table(rows)}"
          f"\n\n## Per transfer\n\n| Transfer | Seeds with epoch 0 < chosen | Passes |\n"
          f"|---|---|---|\n{per}\n\n## Verdict\n\n{result['statement']}.\n")
    (out / "epoch0_check.md").write_text(md)
    return payload


def _job(args):
    family, seed = args
    return run_one(family, seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    families = [f for f in args.families.split(",") if f]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    jobs = [(f, s) for f in families for s in seeds]
    start = time.perf_counter()
    if args.workers > 1:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]
    meta = {"protocol": PROTOCOL, "measuredAt": datetime.now(timezone.utc).isoformat(),
            "scale": poc.SCALE.name, "device": str(poc._device()), "torch": torch.__version__,
            "platform": platform.platform(), "families": families, "seeds": seeds,
            "workers": args.workers, "wallSeconds": time.perf_counter() - start,
            "archive": str(ARCHIVE.relative_to(ROOT))}
    payload = write(rows, Path(args.out), meta)
    print(json.dumps({"verdict": payload["verdict"], "wallSeconds": meta["wallSeconds"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
