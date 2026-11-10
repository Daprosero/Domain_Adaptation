"""Pilot: which v63 fixes does MIL-CREDA need on EEGNet? One transfer, leave-one-out.

Diagnostic outside any plan, on the transfer with the largest v64 gain (L11->Robots, which
becomes a development transfer). Everything is `v64_pilot` (EEGNet exactly as defined,
fixed values, the "blocks" data arm, 4 rotations x 3 seeds); kept in every variant: the
neutral phase-1 aggregation (gamma 0, temperature 10), the between-bag sigma and L2-SP
(mu 10). The six fixes under test:

  detConf   target confidences (Eq. 24) enter Eq. 26 without gradient
  detG      g enters the correspondence (Eq. 35) without gradient
  rowTau    tau_loc of a target bag is the spread of its kernel row (else tauLocal 1.0)
  median    Eq. 22 cuts at the median of the bridged model's target scores (else 0.5)
  gate      the v63 shift gate multiplies the adaptation terms (else 1)
  preproc   1-45 Hz band-pass and common average reference (else the stored windows, v60)

Variants: "v63" (all six), "faithful" (none), "-<fix>" (all but one) and "faithfulAtt"
(faithful, with phase 2 also training the attention under the full Eq. 39). The filtered and
the stored data each train their own phase 1 (predict and MIL-CREDA), shared by every
variant on that data, so all phase-2 variants of one (rotation, seed) are paired.
A unit id is "<transfer>:<rotation>".
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from MIL_CREDA_EEG import mil_creda_model as M, remote_units, search, two_phase
from MIL_CREDA_EEG import v63_campaign as C
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import v64_pilot as P

RESULTS = Path("Results") / "ablation-pilot"
FIXES = ("detConf", "detG", "rowTau", "median", "gate", "preproc")
VARIANTS = {"v63": frozenset(FIXES), "faithful": frozenset(),
            **{f"-{f}": frozenset(FIXES) - {f} for f in FIXES},
            "faithfulAtt": frozenset()}
#: Variants whose phase 2 also trains the attention phi with the full Eq. 39.
TRAINS_ATTENTION = frozenset({"faithfulAtt"})
ATTENTION_KEYS = ("V_R", "b_R", "v_R")


@contextlib.contextmanager
def attention_rules():
    """Phase 2 trains phi as well (theta_s and the head stay frozen) under the full Eq. 39:
    L_src now carries gradient (to phi only) beside the adaptation terms (to theta_t and
    phi); phi joins the target trunk's Adam (same lr2, wd2) and keeps its l1 projection."""
    held = {}
    bridge_plain, step_plain, equal_plain = (two_phase.bridge, two_phase.phase_two_step,
                                             two_phase._states_equal)
    adam_plain = torch.optim.Adam

    def bridge(model):
        out = bridge_plain(model)
        for p in model.attention_parameters():
            p.requires_grad_(True)
        constrain = model.constrain
        model.constrain = lambda *m: constrain(*m, "attention") if "target_trunk" in m \
            else constrain(*m)
        held["model"] = model
        return out

    def adam(params, *args, **kwargs):
        params = list(params)
        if "model" in held:
            params += held["model"].attention_parameters()
        return adam_plain(params, *args, **kwargs)

    def step(model, source, target, tau_local, device, term_selection=two_phase.BOTH_TERMS):
        labels = torch.tensor([b.label for b in source], device=device)
        pairs_s = model.bag_pairs(model.embed([b.windows.to(device) for b in source], "source"))
        source_ce = M.source_term(model, model.classify(model.represent(pairs_s)), labels)
        pairs_t = model.bag_pairs(model.embed([b.windows.to(device) for b in target], "target"))
        terms = M.adaptation_terms(model, pairs_s, labels, pairs_t, tau_local)
        return {**terms, "source_ce": source_ce.detach(),
                "objective": two_phase.phase_two_objective(terms, term_selection) + source_ce}

    def equal(a, b):
        keep = lambda s: {k: v for k, v in s.items() if k not in ATTENTION_KEYS}
        return equal_plain(keep(a), keep(b))

    two_phase.bridge, two_phase.phase_two_step, two_phase._states_equal = bridge, step, equal
    torch.optim.Adam = adam
    try:
        yield
    finally:
        two_phase.bridge, two_phase.phase_two_step = bridge_plain, step_plain
        two_phase._states_equal = equal_plain
        torch.optim.Adam = adam_plain
        if "model" in held:
            del held["model"].constrain


@contextlib.contextmanager
def fix_rules(on: frozenset):
    """MIL-CREDA's phase-2 rules with only the fixes in `on`; the spectral feature
    standardization is a no-op (EEGNet as defined)."""
    nothing = contextlib.nullcontext

    def confidences(G):
        return V._conf(G).detach() if V._active() and "detConf" in on else V._conf(G)

    def correspondence(cross, labels, g, tau):
        if not V._active():
            return V._corr(cross, labels, g, tau)
        return V._corr(cross, labels, g.detach() if "detG" in on else g,
                       V.row_tau(cross) if "rowTau" in on else tau)

    def pseudolabel(g):
        return V._pseudolabel(g) if "median" in on else V._pl(g)

    patches = [(M, "confidences", confidences), (M, "total_correspondence", correspondence),
               (M, "pseudolabel", pseudolabel), (two_phase, "phase_two_objective", V._objective),
               (V, "standardization", lambda norm, thread=False: nothing()),
               (V, "standard_of", lambda bags: None), (V, "standard_of_state", lambda s: None)]
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, v in patches:
        setattr(o, n, v)
    try:
        yield
    finally:
        for o, n, v in saved:
            setattr(o, n, v)


def phase_one(data, view, test, rotation: int, seed: int, device, sched, out: dict, tag: str):
    """Phase 1 of the common group (predict) and of MIL-CREDA on one data reading; returns
    the MIL-CREDA checkpoint, or None when the encoder collapsed (recorded)."""
    record = None
    for group in ("common", "MIL-CREDA"):
        key = f"predict{tag}" if group == "common" else f"phase1{tag}"
        hyper = V.phase_one_hyper(group, {"lr1": P.FIXED["lr1"], "l2": P.FIXED["l2"]}, sched)
        try:
            with V.phase_one_l2(P.FIXED["l2"]):
                model = V.train_phase_one(group, data, rotation, seed, hyper, device)["model"]
        except (FloatingPointError, V.V63Error) as error:
            out[key] = f"failed: {type(error).__name__}: {error}"
            continue
        out["trunk"] = type(model.source_trunk).__name__
        out[key] = P._rows(model.eval(), test, "source", device)
        if group == "MIL-CREDA":
            sigma = V.between_bag_sigma(model, view["source_train"], seed=seed)
            model._sigma = sigma
            record = (two_phase.phase_one_state(model), {"hyper": dict(hyper), "sigma": sigma})
    return record


def run_one(t, rotation: int, seed: int, bags, preps: Mapping, device,
            epochs1: int = P.EPOCHS1, epochs2: int = P.EPOCHS2,
            variants: Mapping = VARIANTS, mu: float = V.MU) -> dict:
    started = time.time()
    sched = {"epochs1": epochs1, "epochs2": epochs2, "bagsPerStep": P.BAGS_PER_STEP}
    out = {"transfer": t.name, "rotation": rotation, "arm": "blocks", "seed": seed,
           "fixed": P.FIXED, "epochs1": epochs1, "epochs2": epochs2, "mu": float(mu),
           "gateV63": P.GATE_V63[t.name][rotation],
           "variants": {k: sorted(v) for k, v in variants.items()}}
    for filtered, tag in ((True, ""), (False, "Stored")):
        if not any(("preproc" in on) == filtered for on in variants.values()):
            continue
        data = V.search_data(t, bags, preps[filtered], rotations=[rotation])
        test = V.test_bags(t, bags, rotation, preps[filtered])["target"]
        view = data.rotations[rotation]
        out[f"nUnits{tag}"] = {"source": len(view["source_train"]),
                               "target": len(view["target_train"]), "test": len(test)}
        with fix_rules(frozenset()):
            record = phase_one(data, view, test, rotation, seed, device, sched, out, tag)
        for name, on in variants.items():
            if ("preproc" in on) != filtered:
                continue
            if record is None:
                out[name] = "failed: no MIL-CREDA phase-1 checkpoint"
                continue
            gate = P.GATE_V63[t.name][rotation] if "gate" in on else 1.0
            saved, search.PHASE_ONE_SEED = search.PHASE_ONE_SEED, seed
            try:
                trains = attention_rules() if name in TRAINS_ATTENTION else contextlib.nullcontext()
                with fix_rules(on), trains:
                    o2 = V.train_mil_creda_phase_two(data, rotation, seed, record,
                                                     {"lr2": P.FIXED["lr2"], "wd2": P.FIXED["wd2"]},
                                                     sched, device, gate=gate)
                out[name] = P._rows(o2["model"].eval(), test, "target", device)
            except (FloatingPointError, V.V63Error) as error:   # recorded, never retried
                out[name] = f"failed: {type(error).__name__}: {error}"
            finally:
                search.PHASE_ONE_SEED = saved
    out["seconds"] = round(time.time() - started, 1)
    return out


def record_path(product: Path, t, rotation: int, seed: int, folder: Path = RESULTS) -> Path:
    return (product / folder / t.name.replace("->", "-to-") / f"rotation{rotation}"
            / f"seed{seed}.json")


def run(config: Mapping, product: str | Path | None = None, *, units: Sequence[str],
        seeds: Sequence[int] = (1, 2, 3), epochs1: int = P.EPOCHS1, epochs2: int = P.EPOCHS2,
        variants: Sequence[str] | None = None, mu: float | None = None, workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    """Every (unit, seed) in order, resumable; stops before the session deadline. `variants`
    keeps only those names; `mu` overrides L2-SP (v63: 10), results kept apart."""
    environ = os.environ if environ is None else environ
    product = Path(product) if product else C.DEFAULT_PRODUCT
    workdir = Path(workdir) if workdir else Path.cwd()
    config = {"schedule": {"epochs1": epochs1, "epochs2": epochs2,
                           "bagsPerStep": P.BAGS_PER_STEP}, **config}
    cfg = C._setup(config, environ)
    deadline = float(environ.get(remote_units.DEADLINE_ENV, "inf"))
    placement = remote_units.device_placement(cfg, environ)
    deterministic = remote_units.set_deterministic()
    identity = {"commit": remote_units.source_commit(environ), "codeVersion": V.code_version(),
                "revision": "ablation-pilot"}
    environment = remote_units.environment_stamp(deterministic, identity, placement)
    device = cfg["device"]
    bags = C.campaign.load_bags(cfg)
    preps = {True: P.BlockPrepared(bags), False: P.BlockPrepared(bags, filtered=False)}
    ran, skipped, longest, started = [], [], 0.0, time.time()
    chosen = {k: VARIANTS[k] for k in (variants or VARIANTS)}
    mu = V.MU if mu is None else float(mu)
    folder = RESULTS if mu == V.MU and variants is None else Path(
        f"{RESULTS}-{'+'.join(sorted(chosen))}-mu{mu:g}")
    rules_plain = V.mil_creda_rules
    V.mil_creda_rules = lambda model, threshold, gate, mu_=mu: rules_plain(
        model, threshold, gate, mu=mu_)
    todo = [(u, s) for u in units for s in seeds]
    try:
      for u, seed in todo:
          name, rotation = u.rsplit(":", 1)
          t, rotation = V.transfer(name), int(rotation)
          path = record_path(product, t, rotation, seed, folder)
          if path.is_file():
              skipped.append(f"{u}/{seed}")
              continue
          if time.time() + 1.5 * longest + C.DEFAULT_MARGIN > deadline:
              break
          out = run_one(t, rotation, seed, bags, preps, device, epochs1, epochs2, chosen, mu)
          longest = max(longest, out["seconds"])
          C.write_record(path, {**out, "environment": environment})
          ran.append(f"{u}/{seed}")
          print(u, seed, out.get("trunk"), out.get("nUnits", out.get("nUnitsStored")), f"{out['seconds']}s", flush=True)
    finally:
        V.mil_creda_rules = rules_plain
    remaining = [f"{u}/{s}" for u, s in todo if f"{u}/{s}" not in ran + skipped]
    timing = {"longestStepSeconds": longest, "seconds": round(time.time() - started, 1),
              "device": environment.get("device")}
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        workdir.mkdir(parents=True, exist_ok=True)
        payload = {"mode": mode,
                   "evidence": {"commit": environment.get("commit"),
                                "sourcesDigest": identity["codeVersion"],
                                "codeVersion": identity["codeVersion"], "unit": units[0],
                                "units": list(units), "complete": not remaining,
                                "estimates": {}},
                   "environment": environment, "timing": timing,
                   "runs": {"ran": ran, "skipped": skipped, "remaining": remaining}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return {"ran": ran, "skipped": skipped, "remaining": remaining, "timing": timing}
