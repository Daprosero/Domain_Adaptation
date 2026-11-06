"""Plan v59 Sec. 4.5: the epoch-0 toy-set check adapter (`tools/epoch0_check.py`).

Pins, through the adapter's public functions: the score is the EEG centroid rule on the
toy model's representations; the chosen epoch is the argmax over epochs 1..E with ties to
the earliest; the declared majority criterion; and that the two read-only hooks leave the
toy's own phase-2 training unchanged (same record with and without them)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from MIL_CREDA_Benchmark import poc, toy_run
from MIL_CREDA_EEG import two_phase

TOOL = Path(__file__).resolve().parents[1] / "tools" / "epoch0_check.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("epoch0_check", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeArm(torch.nn.Module):
    """Identity encoders, mean bag representation, a fixed linear head."""

    def __init__(self, target_shift: float):
        super().__init__()
        self.target_encoder = torch.nn.Identity()
        self.head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2))
        self.sigma = 1.0
        self.shift = target_shift

    def instance_embeddings(self, bags):
        return bags

    def target_embeddings(self, bags):
        return bags + self.shift

    def bag_representations(self, embeddings, sigma):
        return embeddings.mean(dim=1), None


def test_centroid_score_is_the_eeg_rule_on_the_toy_representations(tool):
    source = torch.tensor([[[3.0, 0.0]], [[0.0, 3.0]], [[2.0, 0.0]], [[0.0, 2.0]]])
    labels = [0, 1, 0, 1]
    target = torch.tensor([[[4.0, 0.0]], [[0.0, 4.0]]])
    model = FakeArm(target_shift=0.0)
    result = tool.centroid_score(model, target, source, labels, classes=2)
    Zt, Pt = tool.toy_representations(model, target, "target")
    Zs, _ = tool.toy_representations(model, source, "source")
    expected = two_phase.centroid_counts(Zt, Pt, Zs, np.asarray(labels), 2)
    assert result["counts"] == expected.tolist()
    assert result["score"] == pytest.approx(1.0)
    assert result["pseudoClasses"] == 2
    assert model.target_encoder.training and not model.head.training


def test_collapse_to_one_pseudo_class_scores_one_half(tool):
    source = torch.tensor([[[3.0, 0.0]], [[0.0, 3.0]]])
    target = torch.tensor([[[5.0, 0.0]], [[4.0, 1.0]]])  # both pseudo-class 0
    result = tool.centroid_score(FakeArm(0.0), target, source, [0, 1], classes=2)
    assert result["pseudoClasses"] == 1
    assert result["score"] == pytest.approx(0.5)


def test_chosen_epoch_is_argmax_over_epochs_one_to_e_earliest_on_tie(tool):
    series = [{"epoch": 0, "score": 0.9}, {"epoch": 1, "score": 0.6},
              {"epoch": 2, "score": 0.8}, {"epoch": 3, "score": 0.8}]
    assert tool.chosen_epoch(series) == 2
    with pytest.raises(ValueError):
        tool.chosen_epoch([{"epoch": 0, "score": 0.5}])


def test_verdict_is_the_declared_majority_of_transfers(tool):
    def rows(family, flags):
        return [{"family": family, "epoch0Lower": f} for f in flags]
    passing = rows("a", [True, True, False]) + rows("b", [True, False, True]) \
        + rows("c", [False, False, True])
    result = tool.verdict(passing)
    assert result["transfersPassing"] == 2 and result["adequate"] is True
    assert result["statement"].startswith("adecuada")
    failing = rows("a", [True, False]) + rows("b", [False, False]) + rows("c", [True, True])
    result = tool.verdict(failing)  # a: 1/2 is not a majority
    assert result["transfersPassing"] == 1 and result["adequate"] is False
    assert "returns to deliberation" in result["statement"]


def test_hooks_leave_the_toy_training_unchanged_and_score_every_epoch(tool, tmp_path):
    scale = toy_run.PILOT
    row = tool.run_one("rollo", 0, scale=scale)
    hyper, multiplier = tool.family_hyper("rollo")
    torch.set_num_threads(1)
    familia = toy_run.build_family("rollo", 0, scale)
    material = toy_run.material_for(familia, 0, scale, poc.BACKBONE)
    with toy_run.material_bound(familia, scale, poc.BACKBONE, hyper), \
            tool.sigma_multiplier(multiplier):
        _, plain, _ = poc._two_phases(tool.ARM, material, familia, scale, hyper,
                                      poc._device(), 0)
    assert row["cefEpoch"] == plain["phaseTwo"]["bestEpoch"]
    assert row["targetAccuracy"] == plain["targetAccuracy"]
    assert row["phaseOne"] == plain["phaseOne"]
    assert [r["epoch"] for r in row["series"]] == list(range(scale.epochs_2 + 1))
    assert row["epoch0Score"] == row["series"][0]["score"]
    assert row["chosenScore"] == row["series"][row["chosenEpoch"]]["score"]
    assert row["epoch0Lower"] == (row["epoch0Score"] < row["chosenScore"])
    assert poc.train_phase_two.__name__ == "train_phase_two"
    assert toy_run.effective_classes.__module__ == "MIL_CREDA_Benchmark.toy_run"
    assert poc.SIGMA_MULTIPLIER == 0.5

    payload = tool.write([row], tmp_path, {"scale": scale.name, "device": "cpu",
                                           "torch": torch.__version__, "seeds": [0],
                                           "measuredAt": "now"})
    stored = json.loads((tmp_path / "epoch0_check.json").read_text())
    assert stored["verdict"] == payload["verdict"]
    assert "| rollo source -> rollo target | 0 |" in (tmp_path / "epoch0_check.md").read_text()
