"""The two MIL-CREDA usage tutorials: their notebooks, the helpers they call and their wiring."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from MIL_CREDA_EEG import criteria_pool as C
from MIL_CREDA_EEG import repro, two_phase
from MIL_CREDA_EEG.reproduce import tutorial as T

ROOT = repro.repo_root()
NOTEBOOKS = {"toyset": ROOT / "MIL-CREDA" / "Notebooks" / "Toyset.ipynb",
             "eeg": ROOT / "MIL-CREDA-EEG" / "Notebooks" / "00_Tutorial_EEG.ipynb"}


def _cells(path):
    return json.loads(Path(path).read_text())["cells"]


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_notebooks_are_stored_small_and_without_outputs(name):
    path = NOTEBOOKS[name]
    assert path.stat().st_size < 2_000_000
    cells = _cells(path)
    code = [c for c in cells if c["cell_type"] == "code"]
    assert code and all(not c.get("outputs") and c.get("execution_count") is None for c in code)
    assert "image/gif" not in path.read_text()


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_first_code_cell_finds_src_locally_and_clones_on_kaggle(name):
    first = "".join(next(c for c in _cells(NOTEBOOKS[name]) if c["cell_type"] == "code")["source"])
    assert 'sys.path.insert(0, str(root / "src"))' in first
    assert "https://github.com/Daprosero/Domain_Adaptation" in first and "/kaggle" in first


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_notebooks_call_the_tutorial_helpers_and_no_superseded_runner(name):
    text = "".join("".join(c["source"]) for c in _cells(NOTEBOOKS[name]))
    assert "from MIL_CREDA_EEG.reproduce import tutorial as T" in text
    for stale in ("toy_run.TOY_ADAPTED", "tutorial_visuals", "gif_de_", "FORGE_CLONE_ROOT", "balladeer"):
        assert stale not in text
    assert 'os.environ.get("REPRO_SMOKE") == "1"' in text


def test_pool_params_are_the_papers_seeded_draw():
    drawn = C.draw("MIL-CREDA", 1, 3)
    assert [T.pool_params(i) for i in range(3)] == drawn
    assert set(drawn[0]) == {"lr2", "wd2", "lambda_global", "lambda_local", "tau_local", "gamma", "tau_att"}


def test_ieee_pick_is_the_tracked_manifest_entry():
    pick = T.ieee_pick()
    assert (pick["trial"], pick["BA"], pick["AUC"], pick["kappa"]) == (21, 0.83333, 0.97222, 0.66667)
    assert pick["cef"] == 2.0          # two classes: no later trial can exceed it, so trials 0..21 decide the pick


def test_recorded_phase_one_settings_lie_in_the_papers_search_space():
    for p in (T.SIM_UNIT["phase1"], T.IEEE_UNIT["phase1"], T.TOY_PHASE1):
        assert 1e-4 <= p["lr1"] <= 1e-2 and 1e-4 <= p["l2"] <= 1e-1
        assert p["width"] in (32, 64, 128) and 0.01 <= p["tauAtt1"] <= 1 and p["epochs"] in (50, 100, 200)


def test_cef_pick_is_the_first_maximum_and_skips_diverged_trials():
    rows = [{"cef": 1.5}, {"cef": None}, {"cef": 2.0}, {"cef": 2.0}]
    assert T.cef_pick(rows) == 2


@pytest.fixture(scope="module")
def lunas():
    return T.toy_unit("lunas", 0)


def test_toy_split_is_subject_disjoint_and_stratified(lunas):
    s = lunas["sides"]
    for a, b in (("source", "source_test"), ("target", "target_test")):
        assert not {x.subject for x in s[a]} & {x.subject for x in s[b]}
    for role in ("source", "source_test", "target", "target_test"):
        counts = np.bincount([x.label for x in s[role]], minlength=3)
        assert counts.min() == counts.max()
    assert all(x.windows.shape[1] == 3 for x in s["source"])


def test_toy_training_moves_only_the_target_encoder_and_attention(lunas):
    base = T.toy_phase_one(lunas, dict(T.TOY_PHASE1, epochs=3))
    assert base["sigma"] > 0 and all(np.isfinite(base["loss"]))
    frozen = {k: v.clone() for k, v in base["base"].state_dict().items()}
    model = T.toy_phase_two(base, lunas, T.pool_params(0), epochs=2, seed=C.train_seed(1, 0))
    assert model.sigma == base["sigma"]
    after = model.state_dict()
    for k, v in frozen.items():
        if k.startswith(("source_trunk.", "head.")):
            assert torch.equal(after[k], v), k
    assert any(not torch.equal(after[k], frozen[k]) for k in after if k.startswith("target_trunk.project"))
    # phase 2 trains a copy: the base every configuration starts from is never trained in place
    assert all(torch.equal(v, frozen[k]) for k, v in base["base"].state_dict().items())
    G = two_phase.evaluate(model, lunas["sides"]["target_test"], "target", T.DEVICE)
    assert G.shape == (18, 3) and torch.isfinite(G).all()
    assert 1.0 <= T.cef(model, lunas["sides"]["target"]) <= 3.0


def _run(path, tmp_path):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, 4)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    return nb


@pytest.mark.slow
def test_toyset_notebook_executes_in_smoke_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_SMOKE", "1")
    nb = _run(NOTEBOOKS["toyset"], tmp_path)
    assert not [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]


IEEE_LOCAL = bool(os.environ.get("IEEE_ROOT"))


@pytest.mark.slow
@pytest.mark.skipif(not IEEE_LOCAL, reason="needs a local copy of the IEEE data (IEEE_ROOT)")
def test_eeg_notebook_executes_in_smoke_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_SMOKE", "1")
    nb = _run(NOTEBOOKS["eeg"], tmp_path)
    assert not [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]


@pytest.mark.slow
@pytest.mark.skipif(not IEEE_LOCAL, reason="needs a local copy of the IEEE data (IEEE_ROOT)")
def test_ieee_split_reproduces_the_papers_cef_pick():
    unit = T.ieee_unit()
    pick = T.ieee_pick()
    with T.running(unit):
        base = T.phase_one(unit, T.IEEE_UNIT["phase1"])
        model = T.phase_two(unit, base, T.pool_params(pick["trial"]), C.train_seed(T.POOL_SEED, pick["trial"]))
        got = T.scores(model, unit["sides"]["target_test"], base["cut"])
    assert abs(got["BA"] - pick["BA"]) <= 0.02 and abs(got["AUC"] - pick["AUC"]) <= 0.02
