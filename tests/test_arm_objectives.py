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


def _run_one_with_clock(seconds_of):
    """`run_one`'s shape, with the wall time of each run under the test's hand.

    The shared `_fake_run_one` returns a constant `0.01` for every arm, which
    is exactly the fixture that would let a "the slowest arm is named" claim
    pass while naming nothing: with every reading identical, any arm is a
    correct answer. Here the slowest is a different arm in every cell.
    """
    def run_one(arm_id, transfer, seed, reduction, device, material, **kwargs):
        label = harness.transfer_label(transfer)
        return {
            "arm": arm_id, "transfer": label, "seed": seed,
            "env": "test-env", "targetAccuracy": 0.5 + seed / 100,
            "sourceAccuracy": 0.5, "seconds": seconds_of(arm_id, seed),
            "peakMiB": 1.0, "parameters": 4, "contribution": 0.1,
            "supervised": 0.2, "adaptationShare": 0.3,
            "curve": [], "epochs": [{"epoch": 0}], "state": None,
        }
    return run_one


def test_progress_prints_one_line_per_cell_and_names_that_cells_slowest_arm(
        campana, monkeypatch) -> None:
    """One line per (seed, transfer), not one per run, and it names the slowest.

    Six transfers over thirty seeds is 180 lines; the same call inside the arm
    loop prints 1800, and 1800 lines of a run measured in hours is a report
    nobody reads rather than the sign of life it is kept for.

    The slowest arm rides along because the cell granularity is what would
    otherwise hide it. Printed per run, an arm taking ten times its neighbours
    was visible while it was still the only thing that had happened; summarised
    per cell it would surface only once the cell closed, unless the summary
    says which arm spent the time.

    Reachable red, both halves: move `progress` back inside the arm loop and the
    count lands on arms x transfers x seeds; drop the `slowest` clause and the
    arm this cell actually spent its time on is nowhere in the line.
    """
    arm_ids = [arm["id"] for arm in config.ARMS]

    def seconds_of(arm_id: str, seed: int) -> float:
        # A rotation, so no cell shares a slowest arm with the next and a line
        # that named a fixed arm would be wrong five times out of six.
        return 1.0 + (arm_ids.index(arm_id) + seed) % len(arm_ids)

    monkeypatch.setattr(harness, "run_one", _run_one_with_clock(seconds_of))
    lines: list[str] = []
    seeds = [0, 1, 2]
    harness.campaign(
        harness.Reduction(seeds=seeds, epochs=1,
                          ceilings={"creda": 1e-4, "milcreda": 1.0},
                          ceilingsByTransfer={}),
        torch.device("cpu"), progress=lines.append)

    labels = [harness.transfer_label(t) for t in config.VERDICT_TRANSFERS]
    cells = [line for line in lines if " arms  " in line]
    assert len(cells) == len(seeds) * len(labels), (
        f"{len(cells)} progress lines for {len(seeds) * len(labels)} cells of "
        f"{len(arm_ids)} arms -- one per run would be "
        f"{len(seeds) * len(labels) * len(arm_ids)}")

    for seed in seeds:
        slowest = arm_ids[(len(arm_ids) - 1 - seed) % len(arm_ids)]
        spent = seconds_of(slowest, seed)
        for label in labels:
            of_cell = [line for line in cells
                       if f"{label} seed {seed}:" in line]
            assert len(of_cell) == 1, \
                f"{len(of_cell)} lines for the cell {label} seed {seed}"
            assert f"slowest {slowest:>2} {spent:.1f}s" in of_cell[0], \
                (f"the line for {label} seed {seed} does not name {slowest}, "
                 f"which spent {spent:.1f}s of it: {of_cell[0]!r}")


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


# ----------------------------------------------- the objective, assembled or written

def _unit_branch():
    """The two halves of `training_step`, read from the source tree.

    Which branch an arm takes is asserted by running it; which functions a branch
    is ALLOWED to call is a property of the code, and the only place it can be
    read is the code.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(wiring.__file__).read_text(encoding="utf-8"))
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "training_step")
    branch = next(node for node in step.body
                  if isinstance(node, ast.If) and "unit" in ast.dump(node.test))

    def called(nodes) -> set[str]:
        names = set()
        for statement in nodes:
            for node in ast.walk(statement):
                if isinstance(node, ast.Call):
                    function = node.func
                    names.add(function.attr if isinstance(function, ast.Attribute)
                              else getattr(function, "id", ""))
        return names

    return called(branch.body), called(branch.orelse)


def test_the_bag_unit_arms_assemble_the_objective_and_never_write_a_term_inline(
        encoder) -> None:
    """Eq. (18) and Eq. (39) as the revision states them, called and not restated.

    A supervised term written inline in the benchmark is a second copy of an
    equation the proposal already owns: it stops moving when the proposal moves,
    and nothing tells anyone. So the bag branch calls `source_loss` for its
    supervised term and `total_objective` for the sum, and the value each returns
    is asserted against the functions themselves.

    The inline cross-entropy at the other side of the branch is prior work's own,
    and it belongs to the instance unit alone. That is asserted too, because "no
    term written inline" is only meaningful beside the one place a term IS
    written inline on purpose.

    Reachable red: write Eq. (18) or Eq. (39) out by hand in the bag branch, or
    let the bag branch fall through to the instance one's cross-entropy.
    """
    from MIL_CREDA.objective import source_loss, total_objective

    arm = _arm("G")
    assert config.ARMS_BY_ID["G"]["unit"] == "bag"
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    ramp = 0.5
    step = arm.training_step(x, y, ramp, torch.Generator().manual_seed(3))

    embeddings = arm.instance_embeddings(x)
    Z, _ = arm.bag_representations(embeddings)
    scores = F.softmax(arm.head(Z), dim=1)
    supervised = source_loss(scores, F.one_hot(y, CLASSES).to(scores.dtype),
                             config.EPSILON)
    assert step["supervised"] == pytest.approx(float(supervised.detach()), abs=1e-6)

    target = arm.target.take(arm._draw_target(torch.Generator().manual_seed(3)))
    global_term, local_term = arm._milcreda_term(embeddings, y, target)
    assembled = total_objective(supervised, global_term, local_term, ramp, ramp)
    assert float(step["loss"].detach()) == pytest.approx(
        float(assembled.detach()), abs=1e-6)

    # and both equations are called rather than restated
    bag, instance = _unit_branch()
    assert {"source_loss", "one_hot", "softmax"} <= bag
    assert "cross_entropy" not in bag, "a supervised term written inline"
    assert "cross_entropy" in instance, "prior work's own term left the branch"
    assert "source_loss" not in instance, "Eq. (18) applied to prior work"
    assert "total_objective" in _milcreda_calls()


def _milcreda_calls() -> set[str]:
    """Everything the MIL-CREDA arm of `training_step` calls to build its total."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(wiring.__file__).read_text(encoding="utf-8"))
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "training_step")
    branch = next(node for node in ast.walk(step)
                  if isinstance(node, ast.If) and "milcreda" in ast.dump(node.test))
    return {node.func.attr if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", "")
            for statement in branch.body for node in ast.walk(statement)
            if isinstance(node, ast.Call)}


class _SpyF:
    """`torch.nn.functional` with one call recorded. Everything else passes through."""

    def __init__(self, seen):
        self._seen = seen

    def __getattr__(self, name):
        return getattr(F, name)

    def cross_entropy(self, scores, target, *args, **kwargs):
        self._seen.append(target)
        return F.cross_entropy(scores, target, *args, **kwargs)


def test_the_bag_label_reaches_thirty_instances_in_one_unit_and_one_bag_in_the_other(
        encoder, monkeypatch) -> None:
    """One contamination, two perturbations -- asserted as the mechanism and not
    as its consequence.

    `wiring` broadcasts the bag's label to all thirty of its instances for an
    instance-unit arm, so a contaminated instance there carries a genuinely wrong
    label. A bag-unit arm never expands it: the label stays at the bag and the
    contaminants arrive as witnesses inside it. That is one perturbation entering
    two objectives differently, and it is why a robustness table cannot read the
    two families as the same experiment.

    What is asserted is the broadcast itself -- how many supervised targets one
    bag's label becomes. Whether a wrong label and a downweightable witness
    differ in kind is the reading the report makes of this, and it stays an
    argument rather than becoming an assertion.

    Reachable red: drop the `repeat_interleave`, or expand the label in the bag
    branch as well.
    """
    per_instance, per_bag = [], []
    monkeypatch.setattr(wiring, "F", _SpyF(per_instance))

    real_source_loss = wiring.source_loss

    def _spy_source_loss(scores, onehot, epsilon):
        per_bag.append(onehot)
        return real_source_loss(scores, onehot, epsilon)

    monkeypatch.setattr(wiring, "source_loss", _spy_source_loss)

    B = config.BAGS_PER_STEP
    m = config.INSTANCES_PER_BAG

    instance_arm = _arm("D")
    assert config.ARMS_BY_ID["D"]["unit"] == "instance"
    x = instance_arm.source.take(torch.arange(B))
    y = instance_arm.source.labels[:B]
    instance_arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    assert per_instance, "the instance unit computed no supervised term"
    # one target per instance: the bag's single label, thirty times over
    assert per_instance[0].shape == (B * m,)
    assert torch.equal(per_instance[0], y.repeat_interleave(m))
    assert per_bag == [], "the instance unit reached Eq. (18)"

    per_instance.clear()
    bag_arm = _arm("G")
    assert config.ARMS_BY_ID["G"]["unit"] == "bag"
    x = bag_arm.source.take(torch.arange(B))
    y = bag_arm.source.labels[:B]
    bag_arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    assert per_bag, "the bag unit computed no supervised term"
    # one target per bag, unexpanded: the label never leaves the subject
    assert per_bag[0].shape == (B, CLASSES)
    assert torch.equal(per_bag[0].argmax(dim=1), y)
    assert per_instance == [], "the bag unit broadcast the label anyway"

    # thirty supervised targets against one is what "two perturbations" names,
    # and the factor between them is the bag's own cardinality
    assert m == config.INSTANCES_PER_BAG > 1


def test_the_arbitrary_selection_draws_from_a_generator_of_its_own(encoder) -> None:
    """`SA`'s ten positions are drawn, and the draw costs the run nothing.

    Consuming the training generator would shift every later draw of the run --
    the target batches, the shuffling, everything downstream -- and the rung
    `SA->SK` would then be crediting the selection rule with what the offset did.
    A rung that cannot be attributed is not a rung.

    Two things have to hold at once and each passes while the other is broken:
    the positions come out of `SELECTION_SEED` and nothing else, and building the
    arm leaves the surrounding generator exactly where it found it. The second is
    asserted against `SU`, whose positions are computed rather than drawn, so the
    two builds are identical in everything except this draw.

    Reachable red: drop the dedicated generator and let `randperm` fall through
    to the global one.
    """
    def positions_of(arm_id: str, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return wiring.build(arm_id, CLASSES, _pool(1), _pool(2)).positions.clone()

    def after_building(arm_id: str, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        wiring.build(arm_id, CLASSES, _pool(1), _pool(2))
        return torch.randn(4)

    # the draw is reproducible off the declared seed and off nothing else: two
    # different surrounding seeds give the same ten positions
    assert torch.equal(positions_of("SA", 11), positions_of("SA", 4242))

    expected = torch.randperm(
        config.INSTANCES_PER_BAG,
        generator=torch.Generator().manual_seed(config.SELECTION_SEED)
    )[:config.SELECT_K].sort().values
    assert torch.equal(positions_of("SA", 11), expected)

    # and it takes nothing from the generator around it: after building `SA` the
    # next draw is the same one that follows building `SU`, which draws nothing
    assert torch.equal(after_building("SA", 11), after_building("SU", 11))

    # the rule really is a draw and not the even stride `SU` walks
    assert not torch.equal(positions_of("SA", 11), positions_of("SU", 11))
    assert len(positions_of("SA", 11)) == config.SELECT_K
