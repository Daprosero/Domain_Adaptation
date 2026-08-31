"""What each arm computes, and what the campaign hands every arm.

Three claims live here that the declaration tests could only look at from the
outside. That the selecting arms spend a budget of ten is a fact about `select`,
not about the constant it reads. That prior work is used as it was written is a
fact about the objective `training_step` assembles for an instance-unit arm. And
that all ten arms see one contamination draw is a fact about the loop in
`campaign`, where the material is built.

The encoder is stubbed -- it is a pretrained resnet18 and no claim here is about
it -- and everything the claims are about runs for real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from MIL_CREDA_Benchmark import config, harness, wiring

CLASSES = config.CLASSES


class _Encoder(nn.Module):
    """Something with an `output_dim` that maps an instance to a row."""

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder)


def _pool(seed: int) -> wiring.Pool:
    generator = torch.Generator().manual_seed(seed)
    bags_count = config.BAGS_PER_STEP + 2
    images = torch.randn(bags_count * config.INSTANCES_PER_BAG, 3, 8, 8,
                         generator=generator)
    members = torch.arange(images.shape[0]).reshape(bags_count, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags_count) % CLASSES
    return wiring.Pool(images, members, labels)


def _arm(arm_id: str):
    torch.manual_seed(11)
    return wiring.build(arm_id, CLASSES, _pool(1), _pool(2))


# ------------------------------------------------------------- the ten instances

def test_the_three_selecting_arms_spend_a_budget_of_ten(encoder) -> None:
    """`SU`, `SA` and `SK` keep ten of the bag's thirty instances and differ only
    in which ten, so the rung between any two of them is the rule and nothing
    else. The budget is asserted as a number because the trio's whole point is
    that they share it: two arms differing in the rule AND in how much they spend
    have a rung nobody can attribute.

    Reachable red: move `SELECT_K`, or let one rule return a different count --
    `topk` off by one, or the regular stride overrunning the bag.
    """
    assert config.SELECT_K == 10

    H = torch.randn(config.INSTANCES_PER_BAG, 6)
    kept = {}
    for arm_id in ("SU", "SA", "SK"):
        arm = _arm(arm_id)
        rows = arm.select(H)
        assert rows.shape[0] == config.SELECT_K == 10, f"{arm_id} spent another budget"
        kept[arm_id] = {tuple(row.tolist()) for row in rows}

    # and the same ten would make two of the three arms one arm
    assert kept["SU"] != kept["SA"] or kept["SU"] != kept["SK"]

    # the complete method keeps all of them: that is the rung `SK -> G` reads
    assert _arm("G").select(H).shape[0] == config.INSTANCES_PER_BAG


def test_the_three_attention_rungs_are_the_ones_the_ladder_declares() -> None:
    """`SU->SK`, `SA->SK` and `SK->G`, and no other pairing of the selecting arms.

    The first two hold the budget at ten and read what the rule bought; the third
    is the separate question of what the budget itself costs. A fourth pairing --
    `SU->SA`, say -- would compare two arbitrary rules against each other and
    read as if it said something about attention.

    Reachable red: add, drop or repoint any of the three.
    """
    selecting = {arm["id"] for arm in config.ARMS if arm["selection"] is not None}
    assert selecting == {"SU", "SA", "SK"}

    rungs = {(left, right) for left, right, _ in config.LADDER
             if left in selecting or right in selecting}
    assert rungs == {("SU", "SK"), ("SA", "SK"), ("SK", "G")}
    # and the budget rung is the only one leaving the trio
    assert {r for r in rungs if not (r[0] in selecting and r[1] in selecting)} == \
        {("SK", "G")}


# ------------------------------------------------------------ prior work as it is

def test_creda_keeps_its_per_instance_cross_entropy(encoder) -> None:
    """An instance-unit arm's supervised term is CREDA's own cross-entropy over
    instances, never Eq. (18) normalized by `B_src`.

    The two are not interchangeable: `source_loss` divides by its own supremum so
    the three terms of Eq. (39) can be read on one scale, and applying it here
    would edit prior work to make the comparison look tidy. The asymmetry is the
    formulation's and is reported rather than removed.

    Reachable red: call `source_loss` in the instance branch, or drop the
    `repeat_interleave` so the bag's label stops reaching its instances.
    """
    from MIL_CREDA.objective import source_loss

    arm = _arm("D")
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    step = arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    embeddings = arm.instance_embeddings(x)
    logits = arm.head(embeddings)
    per_instance = F.cross_entropy(logits.reshape(-1, CLASSES),
                                   y.repeat_interleave(x.shape[1]))
    assert step["supervised"] == pytest.approx(float(per_instance.detach()), abs=1e-6)

    bag_scores = F.softmax(arm.head(arm.bag_representations(embeddings)[0]), dim=1)
    normalized = source_loss(bag_scores, F.one_hot(y, CLASSES).to(bag_scores.dtype),
                             config.EPSILON)
    assert step["supervised"] != pytest.approx(float(normalized.detach()), abs=1e-6)


def test_creda_carries_one_term_and_computes_it_with_its_own_loss(encoder) -> None:
    """Its own single-term objective: `L_creda` as `CREDALoss` computes it, with
    the coefficient applied once from the shared schedule.

    `lambda_creda` stays at one inside the module because the ramp already
    carries the ceiling; leaving prior work's own coefficient there as well would
    apply it twice and the arm would be running at a scalar nobody declared.

    Reachable red: give `CREDALoss` back its coefficient, or compute the term over
    bag representations instead of over instances.
    """
    arm = _arm("D")
    assert arm.creda.lambda_creda == 1.0

    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    generator = torch.Generator().manual_seed(3)
    step = arm.training_step(x, y, 0.5, generator)

    # the same term, recomputed from prior work's own module over instances
    embeddings = arm.instance_embeddings(x)
    target = arm.target.take(arm._draw_target(torch.Generator().manual_seed(3)))
    H_t = arm.instance_embeddings(target).reshape(-1, arm.encoder.output_dim)
    own = arm.creda(embeddings.reshape(-1, arm.encoder.output_dim), H_t,
                    y.repeat_interleave(x.shape[1]),
                    F.softmax(arm.head(H_t), dim=1))
    assert step["adaptation"] == pytest.approx(float(own.detach()), abs=1e-6)

    # one term and one coefficient: the objective is the sum of exactly those two
    assert float(step["loss"].detach()) == pytest.approx(
        step["supervised"] + step["contribution"], abs=1e-6)
    assert step["contribution"] == pytest.approx(0.5 * step["adaptation"], abs=1e-6)


def test_a_bag_unit_arm_uses_the_normalized_supervised_term_instead(encoder) -> None:
    """The other half of the same claim: the two units do not share a term, and
    which one an arm gets is read from what it declares."""
    from MIL_CREDA.objective import source_loss

    arm = _arm("B")
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    step = arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    Z, _ = arm.bag_representations(arm.instance_embeddings(x))
    scores = F.softmax(arm.head(Z), dim=1)
    expected = source_loss(scores, F.one_hot(y, CLASSES).to(scores.dtype),
                           config.EPSILON)
    assert step["supervised"] == pytest.approx(float(expected.detach()), abs=1e-6)


# ------------------------------------------------- what the campaign hands the arms

def _fake_run_one(seen):
    def run_one(arm_id, transfer, seed, reduction, device, material, **kwargs):
        seen.append({"arm": arm_id, "transfer": harness.transfer_label(transfer),
                     "seed": seed, "source": id(material["source"]),
                     "target": id(material["target"])})
        return {
            "arm": arm_id, "transfer": harness.transfer_label(transfer), "seed": seed,
            "env": "test-env", "targetAccuracy": 0.5 + seed / 100,
            "sourceAccuracy": 0.5, "seconds": 0.01, "peakMiB": 1.0, "parameters": 4,
            "contribution": 0.1, "supervised": 0.2, "adaptationShare": 0.3,
            "curve": [], "epochs": [{"epoch": 0}], "state": None,
        }
    return run_one


def _fake_build(built):
    def build(code, cache, seed, noise=0.0):
        built.append((code, seed, noise))
        return SimpleNamespace(manifest={"code": code, "seed": seed, "noise": noise})
    return build


@pytest.fixture
def campana(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    monkeypatch.setattr(config, "PRODUCT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models")
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")
    seen, built = [], []
    monkeypatch.setattr(harness, "run_one", _fake_run_one(seen))
    monkeypatch.setattr(harness.bags, "build", _fake_build(built))
    return {"runs": seen, "built": built}


def _run_campaign(seeds: list[int], noise: float = 0.0) -> dict:
    return harness.campaign(
        harness.Reduction(seeds=seeds, epochs=1, labelNoise=noise,
                          ceilings={"creda": 1e-4, "milcreda": 1.0},
                          ceilingsByTransfer={}),
        torch.device("cpu"), progress=lambda *a: None)


def test_one_draw_of_the_material_is_shared_by_all_ten_arms(campana) -> None:
    """Arms that saw differently corrupted material differ in the draw as well as
    in what they compute, and no rung between them is attributable.

    Shared by construction and not by agreement: the material is built once per
    (seed, domain) outside the arm loop, so every arm of a cell is handed the
    same object. The count is what says so -- three domains for one seed, however
    many arms ran -- and the identity is what makes the count mean it.

    Reachable red: move `bags.build` inside the arm loop, and the ten arms of a
    cell get ten draws that agree in distribution and in nothing else.
    """
    harness_result = _run_campaign([7], noise=config.NOISE_LEVELS[2])
    built = campana["built"]

    assert len(built) == len(config.DOMAINS), \
        f"the material was drawn {len(built)} times for one seed"
    assert {code for code, _, _ in built} == set(config.DOMAINS)
    assert {rate for _, _, rate in built} == {config.NOISE_LEVELS[2]}

    runs = campana["runs"]
    arms = {run["arm"] for run in runs}
    assert len(arms) == len(config.ARMS) == 10
    for transfer in {run["transfer"] for run in runs}:
        of_cell = [run for run in runs if run["transfer"] == transfer]
        assert len({run["source"] for run in of_cell}) == 1, \
            "two arms of one cell were handed different source material"
        assert len({run["target"] for run in of_cell}) == 1
    assert harness_result["reduction"]["labelNoise"] == config.NOISE_LEVELS[2]


def test_the_campaign_runs_every_one_of_the_six_transfers_and_withholds_none(
        campana) -> None:
    """The verdict is read over all six.

    Withholding the two the search measured bought nothing -- the roles are
    already disjoint by bag -- and cost a third of the units the paired reading
    rests on.

    Reachable red: run `VERDICT_TRANSFERS` minus the searched ones, and both the
    count and the missing labels land here.
    """
    summary = _run_campaign([0, 1, 2])
    declared = {harness.transfer_label(t) for t in config.VERDICT_TRANSFERS}

    assert len(declared) == 6
    assert {run["transfer"] for run in campana["runs"]} == declared
    assert set(summary["grid"]) == declared
    assert set(summary["perTransfer"]) == declared
    searched = {harness.transfer_label(t) for t in config.SEARCH_TRANSFERS}
    assert searched <= set(summary["grid"]), "a searched transfer was withheld"


def test_three_checkpoints_are_kept_per_arm_per_cell_for_every_arm(campana) -> None:
    """Three per arm per cell, so the top three can be selected after the run
    instead of being guessed before it -- and for EVERY arm, because which arms
    the figures need is only known once the campaign has ranked them.

    Reachable red: drop any arm from `CHECKPOINTS`, or keep two instead of three.
    """
    assert set(config.CHECKPOINTS) == {arm["id"] for arm in config.ARMS}
    assert set(config.CHECKPOINTS.values()) == {3}

    summary = _run_campaign([0, 1, 2, 3])
    declared = {harness.transfer_label(t) for t in config.VERDICT_TRANSFERS}
    assert len(summary["checkpoints"]) == len(config.ARMS) * len(declared)
    for key, kept in summary["checkpoints"].items():
        assert len(kept) == 3, f"{key} kept {len(kept)} of four repetitions"

    written = list((config.MODELS).glob("*.manifest.json"))
    assert len(written) == len(config.ARMS) * len(declared) * 3
    one = json.loads(written[0].read_text(encoding="utf-8"))
    assert one["reduction"]["seeds"] == [0, 1, 2, 3]


# ------------------------------------------------------- what funds the third role

def test_the_selection_role_is_funded_by_new_material_and_takes_nothing(
) -> None:
    """Twelve bags per class instead of ten, and the two extra fund the search's
    own role outright: 64 / 20 / 36 where it used to be 64 / -- / 36.

    The relation is the claim. Asserting `VALID_BAGS == 20` beside
    `BAGS_PER_CLASS == 12` would pass just as well if the twenty had been taken
    out of training, which is the thing that was not done.

    Reachable red: raise `VALID_BAGS` without raising `BAGS_PER_CLASS`, or fund
    the role by lowering `TRAIN_BAGS` or `EVAL_BAGS`.
    """
    assert config.BAGS_PER_CLASS == 12
    grew_by = (config.BAGS_PER_CLASS - 10) * config.CLASSES
    assert grew_by == config.VALID_BAGS == 20
    # and neither of the other two paid for it
    assert config.TRAIN_BAGS == 64
    assert config.EVAL_BAGS == 36
    assert config.TRAIN_BAGS + config.EVAL_BAGS == 10 * config.CLASSES
    # the resolution the campaign was sized for is the evaluation role's, untouched
    assert 100 / config.EVAL_BAGS == pytest.approx(2.78, abs=0.01)
