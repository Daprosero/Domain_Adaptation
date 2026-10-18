"""The normalized supervised term, the contamination harness and its table.

Three things are held here, and each one has a way of going wrong that leaves
every number looking plausible:

1. **Eq. (21) is what runs.** The toy trained its source with a raw
   `F.cross_entropy`, which starts at `ln C` and has no ceiling. Eq. (21)
   divides the per-unit negative log-likelihood by `B_src = ln(1 + 1/eps)`, so
   the term lives in `[0, 1)` and the panels can carry a fixed axis that is the
   term's own range rather than a convention. A call site left on raw CE would
   draw past the top of that axis and be clipped, silently.

2. **The copy of Eq. (21), never `MIL_CREDA`'s original.** `src/CREDA` is the
   baseline `src/MIL_CREDA` is measured against, so an import pointing that way
   contaminates the comparison at the import line, before a number is produced.
   `src/CREDA/bounds.py` holds a copy that `tests/test_creda_sample.py` asserts
   identical to the original; this file holds the other half -- that the toy
   reaches for the copy.

3. **The contamination reaches the target TRAIN split and nothing else.** The
   term is poisoned where it trains and the metric stays on data nobody moved,
   which is the only reason accuracy is comparable across the ladder. A
   mechanism that also moved the test split would still produce a monotone
   curve, and the curve would be measuring its own perturbation.

Every fixture asserts its own property before the thing under test is called.
A fixture that already equals the mutant turns a passing test into a passing
nothing, and that has happened in this repository.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import stats as _st

from CREDA import toy_experiment as T
from CREDA import toy_plots
from CREDA.bounds import source_bound, source_loss

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE = RAIZ / "src" / "CREDA"

#: The three call sites the task names, by the function they live in. Declared
#: so an emptied enumeration cannot pass: `test_the_enumeration_is_not_empty`
#: reads it back.
FUNCIONES_CON_TERMINO = ("train_source_ce", "corrida")


# --------------------------------------------------------------- AST helpers


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} has no function {name!r}")


def _called_names(node: ast.AST) -> list[str]:
    """Every callee, as the source spells it: `f(...)` and `a.b(...)` alike."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Name):
            out.append(func.id)
        elif isinstance(func, ast.Attribute):
            out.append(func.attr)
    return out


def _imported_modules(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
    return imported


# ================================================= 1. Eq. (21) is what runs


def test_the_enumeration_is_not_empty():
    """A guard on the guard below: an empty tuple passes it while reading nothing."""
    assert len(FUNCIONES_CON_TERMINO) == 2, FUNCIONES_CON_TERMINO
    for name in FUNCIONES_CON_TERMINO:
        assert isinstance(_function(PAQUETE / "toy_experiment.py", name),
                          ast.FunctionDef)


def test_the_supervised_term_is_equation_twenty_ones_normalized_form():
    """No raw `cross_entropy` survives, and `perdida_fuente` runs three times.

    Counted, not merely "present": the task names three sites -- phase 1's
    training loss, phase 1's validation loss and phase 2's recorded CE -- and a
    replacement that reached two of them would leave the third drawing past the
    top of a fixed axis with nothing to say so.
    """
    path = PAQUETE / "toy_experiment.py"

    llamadas_totales = _called_names(_tree(path))
    assert "cross_entropy" not in llamadas_totales, (
        "toy_experiment still calls cross_entropy somewhere: Eq. (21) is the "
        "supervised term now, and a raw CE starts at ln C with no ceiling")
    assert llamadas_totales.count("perdida_fuente") == 3, (
        f"perdida_fuente is called {llamadas_totales.count('perdida_fuente')} "
        "times; the task names three sites")

    # And each named function carries at least one of them, so the three are
    # not all piled into one place.
    for name in FUNCIONES_CON_TERMINO:
        llamadas = _called_names(_function(path, name))
        assert "perdida_fuente" in llamadas, f"{name} does not call perdida_fuente"
        assert "cross_entropy" not in llamadas, f"{name} still calls cross_entropy"


def test_the_toy_reaches_for_the_bounds_copy_and_not_for_the_original():
    """`perdida_fuente` delegates; it does not write Eq. (21) a second time."""
    cuerpo = _function(PAQUETE / "toy_experiment.py", "perdida_fuente")
    assert "source_loss" in _called_names(cuerpo), (
        "perdida_fuente does not call source_loss: an inline copy of Eq. (21) "
        "is a silent fork of the formulation")


def test_the_supervised_term_matches_a_hand_written_equation_twenty_one():
    """The value, against the equation written out here from r28's own text.

    `(g + eps) / (1 + eps)` inside the logarithm and `B_src` outside it. Adding
    `eps` inside alone would make a certain, correct prediction score
    `-ln(1 + eps) < 0`, and dividing by anything other than `B_src` would move
    the scale the fixed axis is drawn against.
    """
    eps = 1e-8
    bound = math.log(1.0 + 1.0 / eps)
    assert T.B_FUENTE == pytest.approx(bound, rel=0, abs=0), T.B_FUENTE
    assert bound == pytest.approx(18.4207, abs=1e-4), bound

    torch.manual_seed(0)
    logits = torch.randn(7, T.CLASES) * 2.5
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0])

    G = F.softmax(logits, 1)
    Y = F.one_hot(y, T.CLASES).to(G.dtype)
    a_mano = float(-(Y * torch.log((G + eps) / (1.0 + eps))).sum() / bound / len(y))

    medido = float(T.perdida_fuente(logits, y))
    assert medido == pytest.approx(a_mano, rel=1e-6), (medido, a_mano)
    # The same value `bounds.source_loss` gives, which is the copy two tests in
    # `test_creda_sample.py` already hold equal to `MIL_CREDA`'s original.
    assert medido == pytest.approx(float(source_loss(G, Y)), rel=1e-12)


def test_the_supervised_term_lives_in_the_unit_interval():
    """`[0, 1]`, and the upper end is CLOSED in floating point.

    `bounds.py`' docstring argues the upper end is a supremum and not a
    maximum, "because the classifier produces strictly positive probabilities
    and never reaches g = 0". That is true of the real line and false of the
    arithmetic: `-ln(eps/(1 + eps))` is EXACTLY `B_src`, so once `g` is small
    enough that `g + eps == eps` the normalized term is exactly `1.0`.
    Measured: that happens from a logit gap of about 40 in float32.

    It is asserted here rather than only reported because the fixed `(0, 1)`
    axis depends on the interval, and the direction of the discrepancy is the
    benign one -- the term reaches the top of the axis and never passes it, so
    nothing is clipped.
    """
    y = torch.tensor([0, 1, 2])
    certero = torch.tensor([[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]])
    equivocado = torch.tensor([[0.0, 0.0, 60.0], [60.0, 0.0, 0.0], [0.0, 60.0, 0.0]])

    assert float(T.perdida_fuente(certero, y)) == pytest.approx(0.0, abs=1e-9)
    peor = float(T.perdida_fuente(equivocado, y))
    assert peor == 1.0, (peor, "the top of the interval is attained exactly")
    assert -math.log(1e-8 / (1 + 1e-8)) == pytest.approx(T.B_FUENTE, rel=0, abs=0)

    # Strictly inside, where the stabilizer does not swallow `g`: a gap of 20
    # leaves `g = 2.06e-09`, which is above `eps`.
    dentro = float(T.perdida_fuente(
        torch.tensor([[0.0, 0.0, 20.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0]]), y))
    assert 0.5 < dentro < 1.0, dentro

    # And the random middle, which is where a raw CE would have started: ln 3 =
    # 1.0986, above the axis. Normalized it is a fraction of the ceiling.
    plano = torch.zeros(3, T.CLASES)
    medio = float(T.perdida_fuente(plano, y))
    assert medio == pytest.approx(math.log(T.CLASES) / T.B_FUENTE, rel=1e-6)
    assert medio < 0.06, (medio, "ln 3 / B_src should be about 0.0596")
    assert math.log(T.CLASES) > 1.0, (
        "a raw CE would have started above the fixed axis; that is why this "
        "term had to change before the axis could")


# =============================== 2. `MIL_CREDA` is never imported by the toy


def test_the_toy_never_imports_the_method_it_is_the_baseline_for():
    """AST, like the no-torch rule next door, and for a stronger reason.

    `src/CREDA` is prior work: the baseline `src/MIL_CREDA` is measured
    against. An import pointing that way makes the baseline depend on the thing
    it is the control for, and the contamination happens at the import line --
    before any number is produced, and invisible to every other check here.
    """
    for nombre in ("toy_experiment.py", "toy_plots.py"):
        importados = _imported_modules(PAQUETE / nombre)
        culpables = sorted(m for m in importados
                           if m == "MIL_CREDA" or m.startswith("MIL_CREDA."))
        assert culpables == [], f"{nombre} imports {culpables}"
        modulo = T if nombre == "toy_experiment.py" else toy_plots
        assert "MIL_CREDA" not in vars(modulo)


# ========================================== 3-6. the contamination mechanism


RHO_FUERTE = 0.30


def _datos():
    return T.datos_semilla(T.FAMILIAS[0], 0)


def _intacto(D, D2, salvo="Xt_tr"):
    """Every split but `salvo`, byte for byte, and the same object besides."""
    for clave, valor in D.items():
        if clave == salvo:
            continue
        otro = D2[clave]
        assert type(otro) is type(valor), (clave, type(otro), type(valor))
        if isinstance(valor, torch.Tensor):
            assert torch.equal(otro, valor), f"{clave} moved"
        else:
            assert np.array_equal(otro, valor), f"{clave} moved"


def test_the_declared_ladder_includes_zero_and_is_the_one_the_sweep_reads():
    assert T.RHOS[0] == 0.0, T.RHOS
    assert T.RHOS == (0.00, 0.10, 0.20, 0.30), T.RHOS


def test_contamination_reaches_the_target_train_split_only():
    """The test split is byte-identical at `rho = 0.30`, and train is not.

    Both halves. "the test split did not move" alone would also pass a
    mechanism that moved nothing at all.
    """
    D = _datos()
    antes = {k: (v.clone() if isinstance(v, torch.Tensor) else np.array(v))
             for k, v in D.items()}

    D2, idx = T.contaminar(D, RHO_FUERTE, 0)

    # `contaminar` did not mutate what it was handed.
    for clave, valor in antes.items():
        assert torch.equal(D[clave], valor), (
            f"contaminar mutated the dict it was given, at {clave}")

    _intacto(D, D2)
    assert not torch.equal(D2["Xt_tr"], D["Xt_tr"]), (
        "the train split did not move at rho=0.30, so nothing was contaminated")

    # Exactly the recorded rows differ, and no others.
    movidas = (~torch.isclose(D2["Xt_tr"], D["Xt_tr"]).all(dim=1)).nonzero().reshape(-1)
    assert sorted(movidas.tolist()) == sorted(np.asarray(idx).tolist()), (
        "the rows that moved are not the rows that were recorded")
    # And the labels travel with the points: the true label is KEPT.
    assert torch.equal(D2["yt_tr"], D["yt_tr"]), (
        "the contaminated points changed label; the mechanism relocates "
        "features and keeps the true label")


def test_a_zero_contamination_fraction_is_a_no_op():
    """Bit-exact, and the index record empty. The zero rung has to reproduce."""
    D = _datos()
    D2, idx = T.contaminar(D, 0.0, 0)
    assert len(idx) == 0, idx
    assert torch.equal(D2["Xt_tr"], D["Xt_tr"])
    _intacto(D, D2)


def test_the_contaminated_indices_are_recorded_with_their_declared_count():
    """`round(rho * n)` per rung, and the record is what the figure reads."""
    D = _datos()
    n = len(D["Xt_tr"])
    assert n > 100, n
    for rho in T.RHOS:
        _, idx = T.contaminar(D, rho, 0)
        esperado = int(round(rho * n))
        assert len(idx) == esperado, (rho, len(idx), esperado, n)
        assert len(set(np.asarray(idx).tolist())) == len(idx), "repeated index"
        assert np.asarray(idx).min(initial=0) >= 0
        assert np.asarray(idx).max(initial=0) < n


def test_the_same_seed_and_fraction_contaminate_identically():
    """And a DIFFERENT seed does not, or determinism would be constancy."""
    D = _datos()
    a, ia = T.contaminar(D, 0.20, 3)
    b, ib = T.contaminar(D, 0.20, 3)
    assert torch.equal(a["Xt_tr"], b["Xt_tr"])
    assert np.array_equal(np.asarray(ia), np.asarray(ib))

    c, ic = T.contaminar(D, 0.20, 4)
    assert not np.array_equal(np.asarray(ia), np.asarray(ic)), (
        "two seeds picked the same points: the draw does not read the seed")


def test_a_contaminated_point_lands_in_another_classs_region():
    """The mechanism EXACTLY, and then how far it actually carries a point.

    The exact half is the mechanism itself: every relocated point is
    `x_j - mu_{y_j} + mu_{c'}` for exactly one foreign `c'`. That is checkable
    without a threshold, so a perturbation of any other shape fails it
    outright, and the figure's caption -- a contaminated point deep inside the
    wrong class gets a HIGH `w` -- rests on it.

    The statistical half is reported rather than demanded at a number somebody
    liked. Measured on `lunas` at `rho = 0.30`: 290 of 324 relocated points are
    nearest a FOREIGN centroid, which is 89.5% and not 100% because the family
    is two interleaved parabolas -- a point's within-class deviation can be
    larger than the gap between the two centroids and carry it back. The floor
    asserted here is 0.80, below the measurement and above "no relocation".
    """
    D = _datos()
    X, y = D["Xt_tr"].numpy(), D["yt_tr"].numpy()
    clases = [int(c) for c in np.unique(y)]
    centros = {c: X[y == c].mean(0) for c in clases}
    D2, idx = T.contaminar(D, RHO_FUERTE, 0)
    X2 = D2["Xt_tr"].numpy()
    idx = np.asarray(idx)

    # The exact mechanism, point by point.
    for j in idx:
        propio = int(y[j])
        candidatos = [c for c in clases
                      if c != propio
                      and np.allclose(X2[j], X[j] - centros[propio] + centros[c],
                                      atol=1e-5)]
        assert len(candidatos) == 1, (
            j, propio, candidatos,
            "the relocated point is not x_j - mu_own + mu_foreign for exactly "
            "one foreign class")

    # And how far that carries it, measured and reported.
    ajenos = sum(
        1 for j in idx
        if min(clases, key=lambda c: float(np.linalg.norm(X2[j] - centros[c])))
        != int(y[j]))
    fraccion = ajenos / len(idx)
    assert fraccion >= 0.80, (
        f"{ajenos} of {len(idx)} ({fraccion:.3f}) relocated points are nearest "
        "a foreign centroid; the relocation is not reaching another region")


# ================================ 7. the uniform-confidence invariance (Eq. 32)


def _g_para_w(objetivo, C, n, dtype=torch.float32):
    """`g` with every row at the SAME entropy, so every `w` equals one `c`.

    The row is `(p, (1-p)/(C-1), ...)` rolled one step per sample, so the
    pseudo-label cycles over the classes while the sum of squares -- and so
    `H_2`, and so `w` -- is identical in every row. A single un-rolled row
    would give every point the same pseudo-label and leave C-1 classes empty.
    """
    lo, hi = 1.0 / C, 1.0 - 1e-15
    for _ in range(300):
        p = (lo + hi) / 2
        fila = np.array([p] + [(1 - p) / (C - 1)] * (C - 1))
        w = 1 - (-math.log((fila ** 2).sum() + 1e-6)) / math.log(C)
        if w < objetivo:
            lo = p
        else:
            hi = p
    p = (lo + hi) / 2
    base = np.array([p] + [(1 - p) / (C - 1)] * (C - 1))
    rows = np.stack([np.roll(base, j % C) for j in range(n)])
    return torch.tensor(rows, dtype=dtype), float(w)


def _perdida_por_clase(C=3, n_s=300, n_t=900):
    torch.manual_seed(0)
    f_s = torch.randn(n_s, 8)
    f_t = torch.randn(n_t, 8) + 0.5
    y_s = torch.arange(n_s) % C
    cr = T.CREDALoss(sigma="auto", sigma_from="batch-source",
                     normalize_by_bounds=T.NORMALIZAR_POR_COTAS,
                     bounds_form=T.FORMA_DE_COTAS, weight_cross_block=True,
                     balance_mix_blocks=True, balance_mass=T.BALANCE_DE_MASA)
    cr.class_mixing = 1.0

    def medir(objetivo):
        g, w = _g_para_w(objetivo, C, n_t)
        # The fixture's own property, before anything is measured over it.
        cuad = (g ** 2).sum(1)
        assert float(cuad.max() - cuad.min()) < 1e-6, "the rows differ in entropy"
        assert len(torch.unique(g.argmax(1))) == C, "a class came out empty"
        with torch.no_grad():
            return cr(f_s, f_t, y_s, g, reduction="none").numpy(), w

    return medir, C, n_t


def test_the_term_is_invariant_to_the_common_confidence_level():
    """r28's claim about Eq. (32), which nothing in the suite checked.

    With every `w` equal to one `c`: `K~_t = c^2 K_t` and `tr(K~_t) = c^2
    tr(K_t)`, so the `c^2` cancels under the trace balance; `K~_st = c K_st`
    over `c sqrt(tr(K_s) tr(K~_t))`, so the `c` cancels too. The per-class
    loss is therefore EXACTLY the same number at every `c`.

    The sweep spans two decades. The assertion at `1e-6` is made over the part
    of it where `tr(K~_t) >= 1`, and that restriction is MEASURED rather than
    chosen: the companion test below shows the residual below it is the
    `1e-6` epsilon inside `_renyi_entropy_order_2` and nothing else.
    """
    medir, C, n_t = _perdida_por_clase()
    cs = [1.0 - 1e-12, 0.5623, 0.3162, 0.1778, 0.1, 0.0562, 0.0316, 0.0178, 0.01]
    # Two decades, measured in decades: the top rung is `1 - 1e-12` rather than
    # `1`, so a bare `>= 100` ratio fails on the last bit of the mantissa and
    # the test would be about floating point instead of about the sweep.
    assert math.log10(cs[0] / cs[-1]) >= 2.0 - 1e-6, (cs[0], cs[-1])

    ref, _ = medir(cs[0])
    assert len(ref) == C
    exactos = 0
    for c in cs:
        l, w = medir(c)
        assert w == pytest.approx(c, rel=1e-6), (w, c)
        traza = w ** 2 * n_t / C          # tr(K~_t) of a class: sum_j w_j^2
        if traza >= 1.0:
            assert np.abs(l - ref).max() <= 1e-6, (
                f"c={c}: the per-class loss moved by "
                f"{np.abs(l - ref).max():.3e} where the epsilon cannot explain it")
            exactos += 1
    assert exactos >= 5, (exactos, "too few rungs reached the exact assertion")


def test_the_residual_of_that_invariance_is_the_entropy_epsilon():
    """Below `tr(K~_t) = 1` the deviation scales as `1/c^2`, which is the epsilon's.

    `_renyi_entropy_order_2` adds `1e-6` to the trace it divides by, so the
    perturbation it introduces is `1e-6 / tr(K~_t)` and `tr(K~_t)` falls as
    `c^2`. If the deviation were the algebra failing it would not follow that
    law, and this is what licenses the restriction in the test above instead of
    leaving it as a tolerance somebody picked.
    """
    medir, C, n_t = _perdida_por_clase()
    ref, _ = medir(1.0 - 1e-12)
    productos = []
    for c in (0.0562, 0.0316, 0.0178, 0.01):
        l, w = medir(c)
        dev = float(np.abs(l - ref).max())
        assert dev > 1e-6, (c, dev, "the restriction above is not measured")
        productos.append(dev * c ** 2)
    assert max(productos) / min(productos) < 1.3, (
        productos, "the residual does not follow the epsilon's 1/c^2 law")


# ======================= 8. both weighting arms share one phase-2 search


def _celda_del_barrido(CONTAM, nombre, rho):
    """A cell of a real sweep, by the TRIPLE key, resolved through the roster.

    This is deliberately not a `(nombre, rho)` lookup with the perturbation
    guessed. The sweep declares its perturbations in `perturbaciones` and names
    one inside every cell key, and reading the key THROUGH the roster is what
    crosses that join: a sweep whose roster and keys disagreed would pass a
    test that only indexed by the key it already knew.
    """
    declaradas = CONTAM["perturbaciones"]
    assert isinstance(declaradas, list), (
        declaradas, "`perturbaciones` is a LIST; a scalar cannot grow to two")
    assert len(declaradas) == 1, (
        declaradas, "one sweep runs one perturbation")
    perturbacion = declaradas[0]
    clave = (nombre, perturbacion, float(rho))
    assert clave in CONTAM["celdas"], (
        clave, sorted(str(k) for k in CONTAM["celdas"]),
        "the sweep's cell key does not carry the perturbation its roster "
        "declares")
    return CONTAM["celdas"][clave]


@pytest.fixture(scope="module")
def barrido_minimo():
    """The real sweep, at the smallest scale that still has two arms and two seeds.

    One family, two seeds, two rungs, one Optuna trial. Small enough to be a
    test and wide enough to see the ragged-length class of defect: a one-arm
    one-seed run cannot.
    """
    return T.barrer_contaminacion(
        familias=[T.FAMILIAS[0]], semillas=[0, 1], rhos=(0.0, 0.20),
        HP1={T.FAMILIAS[0]["nombre"]: dict(T.HP_1)}, trials2=1,
        al_paso=lambda *_: None)


def test_both_weighting_arms_of_a_scenario_share_one_phase_two_search(barrido_minimo):
    """The thing that keeps the comparison controlled, so it is measured.

    Searching per arm would make it "best of `w` on" against "best of `w` off"
    -- a different claim, with the search-versus-term confound back in. The
    recorded `hp2` is read off what each `corrida` actually computed with, not
    off what the sweep intended to pass.
    """
    CONTAM = barrido_minimo
    nombre = T.FAMILIAS[0]["nombre"]
    assert CONTAM["rhos"] == [0.0, 0.20], CONTAM["rhos"]

    for rho in CONTAM["rhos"]:
        celda = _celda_del_barrido(CONTAM, nombre, rho)
        buscado = celda["hp2"]
        assert buscado, (nombre, rho, "the scenario recorded no phase-2 search")
        sin = celda["brazos"][T.BRAZO_SIN_SOP]
        con = celda["brazos"][T.BRAZO_CON_SOP]
        assert len(sin) == len(con) == 2, (len(sin), len(con))
        for a, b in zip(sin, con):
            assert a["hp2"] == b["hp2"], (rho, a["hp2"], b["hp2"])
            assert a["hp2"] == buscado, (rho, a["hp2"], buscado)
        # And the arms really are two different arms.
        assert all(r["sopesado"] is False for r in sin), sin
        assert all(r["sopesado"] is True for r in con), con


def test_the_scenarios_do_not_share_one_phase_two_search(barrido_minimo):
    """One search PER scenario. Reusing the clean-data HP is the other defect.

    Reading the curve at HP tuned for conditions that no longer hold is the
    mirror image of searching per arm, and a single cached `hp2` would pass the
    test above while committing it.
    """
    CONTAM = barrido_minimo
    nombre = T.FAMILIAS[0]["nombre"]
    limpio = _celda_del_barrido(CONTAM, nombre, 0.0)["hp2"]
    sucio = _celda_del_barrido(CONTAM, nombre, 0.20)["hp2"]
    assert set(limpio) == set(sucio), (limpio, sucio)
    # They may coincide by chance at one trial, so what is asserted is that the
    # sweep RAN a search per scenario, which the record says outright.
    assert _celda_del_barrido(CONTAM, nombre, 0.0)["busquedas"] == 1
    assert _celda_del_barrido(CONTAM, nombre, 0.20)["busquedas"] == 1


def test_the_four_arms_are_all_present_and_all_measured_on_test(barrido_minimo):
    """Four arms, every seed, and the floor read on the SAME split as the rest.

    The existing three-arm table reads `.predict` off the VALID split while
    every other arm is on TEST. That is reported as a finding and left alone;
    this table does not inherit it.
    """
    CONTAM = barrido_minimo
    nombre = T.FAMILIAS[0]["nombre"]
    for rho in CONTAM["rhos"]:
        brazos = _celda_del_barrido(CONTAM, nombre, rho)["brazos"]
        assert list(brazos) == list(T.BRAZOS_CONTAMINACION), list(brazos)
        largos = {b: len(v) for b, v in brazos.items()}
        assert set(largos.values()) == {2}, largos
        for b, filas in brazos.items():
            for r in filas:
                assert r["split"] == "test", (b, r.get("split"))
                assert 0.0 <= r["acc_te"] <= 1.0, (b, r["acc_te"])


def test_the_sweep_records_the_weights_and_the_indices_for_the_figure(barrido_minimo):
    """The per-point `w` at each checkpoint, and which points were injected.

    Both, because the figure sizes a marker by one and selects it by the other,
    and either one alone draws something that reads like the other.
    """
    CONTAM = barrido_minimo
    nombre = T.FAMILIAS[0]["nombre"]
    for rho in CONTAM["rhos"]:
        celda = _celda_del_barrido(CONTAM, nombre, rho)
        fig = celda["figura"]
        n_t = len(fig["yt_tr"])
        assert len(fig["w"]) == len(fig["pasos"]), (
            "one weight vector per checkpoint, or the figure draws a step it "
            "did not measure")
        for fila in fig["w"]:
            assert len(fila) == n_t, (len(fila), n_t)
            assert float(np.min(fila)) >= -1e-6 and float(np.max(fila)) <= 1.0 + 1e-6
        idx = np.asarray(celda["indices"][fig["semilla"]])
        assert len(idx) == int(round(rho * n_t)), (rho, len(idx), n_t)


# =============================================== 10. the loss panels' axes


def _hist_fase1():
    """Phase 1's history, with the chosen epoch LAST and its accuracy highest."""
    filas = [(e, 0.9 - 0.05 * e, 0.92 - 0.04 * e, 0.3 + 0.05 * e) for e in range(6)]
    assert all(len(f) == T.CAMPOS_HS for f in filas), T.CAMPOS_HS
    return filas, filas[-1][0]


def _hist_fase2():
    # Seis campos desde el 2026-10-11: la columna del factor del warmup se
    # fue con el warmup, y las dos de despues se corrieron una posicion.
    filas = [(p, 0.10 + 0.001 * p, 0.40 - 0.001 * p,
              (0.10 + 0.001 * p) + (0.40 - 0.001 * p),
              0.42 - 0.001 * p, 0.8 - 0.002 * p) for p in range(0, 60, 10)]
    assert all(len(f) == T.CAMPOS_HT for f in filas), T.CAMPOS_HT
    return filas


def test_the_loss_panels_are_fixed_to_the_terms_own_zero_to_one_range():
    """Linear, `(0, 1)`, both panels. With Eq. (21) that is the term's range.

    Raw CE started at `ln 3 = 1.0986`, which is why this could not be done
    before and why the panel carried a log axis instead. A log axis on a
    quantity that reaches zero is also a panel that cannot draw its own best
    case.
    """
    hist_s, ep_s = _hist_fase1()
    fig, ax = plt.subplots(1, 2)
    try:
        toy_plots.panel_fase1(ax[0], hist_s, "fase 1", ep_s)
        toy_plots.panel_fase2(ax[1], _hist_fase2(), 60, "fase 2", 50)
        for a, nombre in ((ax[0], "panel_fase1"), (ax[1], "panel_fase2")):
            assert a.get_yscale() == "linear", (nombre, a.get_yscale())
            assert a.get_ylim() == (0.0, 1.0), (nombre, a.get_ylim())
    finally:
        plt.close(fig)


def test_no_loss_panel_sets_a_log_axis_anywhere():
    """Read from the source too: a second panel could add one back.

    The rendered check above only covers the two panels it calls. This one is
    the statement about the file, which is what a later panel would break.
    """
    fuente = (PAQUETE / "toy_plots.py").read_text(encoding="utf-8")
    assert "set_yscale" not in fuente, (
        "toy_plots sets a y scale somewhere: every loss axis here is the "
        "term's own [0,1] and a log axis cannot draw its zero")


# ========================= the figure's size mapping, and where it draws


def test_the_marker_size_is_the_declared_affine_map_of_w():
    """`size = s_min + (s_max - s_min) * w`, at both ends and in the middle.

    Declared rather than "proportional to w", because a reader cannot tell
    `w = 0.60` from `w = 0.95` without the map, and a figure whose sizes cannot
    be read says nothing the panel beside it does not.
    """
    assert toy_plots.TAM_MIN_W < toy_plots.TAM_MAX_W
    assert float(toy_plots.tamano_por_w(0.0)) == pytest.approx(toy_plots.TAM_MIN_W)
    assert float(toy_plots.tamano_por_w(1.0)) == pytest.approx(toy_plots.TAM_MAX_W)
    medio = (toy_plots.TAM_MIN_W + toy_plots.TAM_MAX_W) / 2
    assert float(toy_plots.tamano_por_w(0.5)) == pytest.approx(medio)
    # Monotone and clipped, so a `w` off the interval cannot invert the reading.
    assert float(toy_plots.tamano_por_w(-1.0)) == pytest.approx(toy_plots.TAM_MIN_W)
    assert float(toy_plots.tamano_por_w(2.0)) == pytest.approx(toy_plots.TAM_MAX_W)
    # And 0.60 is visibly a different size from 0.95: 0.35 of the full span.
    salto = float(toy_plots.tamano_por_w(0.95) - toy_plots.tamano_por_w(0.60))
    assert salto > 0.3 * (toy_plots.TAM_MAX_W - toy_plots.TAM_MIN_W), salto


def test_the_weight_legend_does_not_land_on_another_panel():
    """The defect the smoke caught: `plt.scatter` draws on `plt.gca()`.

    The legend handles for the size map were built with `plt.scatter([], [])`,
    which attaches to the CURRENT axes rather than to the one the function was
    handed. In a grid that is the last subplot created, so every row's handles
    piled into the bottom-right panel and `panel_fase2`'s own legend adopted
    them -- the loss panel displayed a `w` scale it does not draw, and nothing
    failed.
    """
    fig, ax = plt.subplots(1, 2)
    try:
        rng = np.random.default_rng(0)
        Pt = rng.normal(size=(40, 2))
        w = rng.uniform(size=40)
        # `ax[1]` is the current axes after `subplots`, which is exactly the
        # trap: the call below is made on `ax[0]`.
        toy_plots.panel_contaminacion(
            ax[0], Pt, w, np.arange(0, 40, 4), [(-3, 3), (-3, 3)])
        ax[1].plot([0, 1], [0, 1], label="algo")
        leyenda = ax[1].legend()
        etiquetas = [t.get_text() for t in leyenda.get_texts()]
        assert etiquetas == ["algo"], (
            etiquetas, "the size-map handles landed on a panel that is not "
            "the one they describe")
        # And they ARE on the panel they describe.
        propias = [t.get_text() for t in ax[0].get_legend().get_texts()]
        assert len(propias) == len(toy_plots.REFERENCIAS_W), propias
        assert all(t.startswith("w = ") for t in propias), propias
    finally:
        plt.close(fig)


def test_the_figure_caption_predicts_the_high_weight_of_a_confident_mistake():
    """Written before the run, so the figure reads as a limit and not a bug.

    `w` measures uncertainty, not error. A contaminated point that lands deep
    inside the wrong class is confidently wrong, gets a HIGH `w` and GROWS.
    Measured at smoke scale on `lunas` at `rho = 0.20`: mean `w` of the
    contaminated points 0.859 against 0.739 for the clean ones -- the
    prediction, in the direction stated.
    """
    pie = toy_plots.PIE_CONTAMINACION
    assert "INCERTIDUMBRE" in pie and "no error" in pie, pie
    assert "CRECE" in pie, pie
    assert "ENTRE clases" in pie, pie
    # The map itself, so the caption and the code cannot drift apart.
    assert f"{toy_plots.TAM_MIN_W:.0f}" in pie, pie
    assert f"{toy_plots.TAM_MAX_W - toy_plots.TAM_MIN_W:.0f}" in pie, pie


# ============ 11. the animated scenario: one rho, frames are phase-2 steps
#
# The GIF's subject is the TARGET TRAIN points and how their `w` moves over the
# phase-2 checkpoints. `w` exists nowhere else: there is no term evaluated on
# test, so a `w` on test would be a number with no definition behind it. The
# accuracy tables in the same section report TEST, so the two sit side by side
# and the figure has to say which split it draws without anybody reading code.
#
# Every test below is a declared invariant of `toy_plots`, which is what makes
# the pairing fail-closed: a property tested here with no declaration and a
# declaration with no test look identical to a green run.


def _figura_sintetica(*, F=4, n=12, rho=0.25, mejor_paso=40, semilla=7):
    """One scenario's `figura` block, hand-built, with every claim asserted.

    A fixture that already equals the mutant turns a passing test into a
    passing nothing, so the shapes and the contamination count are checked
    here before anything under test is handed them.
    """
    rng = np.random.default_rng(semilla)
    sucios = int(round(rho * n))
    # NOT a prefix and NOT contiguous. `np.arange(sucios)` would make a
    # "paint the first k points" mutation byte-identical to selecting `idx`,
    # so the probe would survive and the guard would read as held.
    idx = np.sort(rng.choice(n, size=sucios, replace=False)).astype(np.int64)
    hist_s, ep_s = _hist_fase1()
    pasos = [20 * j for j in range(F)]
    f = dict(
        semilla=semilla, k=min(1, F - 1), paso=pasos[min(1, F - 1)],
        pasos=pasos, mejor_paso=mejor_paso, ep_s=ep_s,
        Pt=rng.normal(size=(F, n, 2)),
        PS_t=[rng.normal(size=(n, 2)) for _ in range(F)],
        w=[rng.uniform(size=n) for _ in range(F)],
        rut=rng.integers(0, 3, size=(F, n)),
        idx=idx,
        pred_s=rng.integers(0, 3, size=n),
        ys_tr=rng.integers(0, 3, size=n),
        yt_tr=rng.integers(0, 3, size=n),
        acc_va=[0.30 + 0.05 * j for j in range(F)],
        acc_te=[0.28 + 0.04 * j for j in range(F)],
        acc_s=[0.95] * F,
        pok=[0.40] * F,
        u_t=[0.20] * F,
        hist_s=hist_s, hist_t=_hist_fase2())
    assert len(f["pasos"]) == F == len(f["w"]) == len(f["Pt"])
    assert len(f["idx"]) == int(round(rho * n)), (len(f["idx"]), rho, n)
    assert 0 < len(f["idx"]) < n, len(f["idx"])
    assert not np.array_equal(f["idx"], np.arange(len(f["idx"]))), (
        f["idx"], "the contaminated indices are the first k, so selecting "
        "them and painting a prefix are the same mutation")
    assert max(f["hist_t"])[0] >= mejor_paso, (
        "the fixture's phase-2 history stops before the step it claims to "
        "have chosen, so panel_fase2 would be drawing nothing")
    return f


def _contam_sintetico(*, rhos=(0.10, 0.30), largos=(3, 4), n=12,
                       nombre="lunas", brazo=None, perturbaciones=(None,),
                       padron=None, techo=None):
    """A whole `CONTAM`, keyed by `(familia, rho)` or by the pair, with RAGGED Fs.

    `perturbaciones=(None,)` is the shape on disk today: the record declares no
    perturbation at all. A tuple of names produces the pair-keyed shape, which
    is what the sweep will write once two perturbations run at the same rho.
    """
    brazo = brazo or T.BRAZO_CON_SOP
    assert len(rhos) == len(largos)
    assert len(set(largos)) > 1, (
        largos, "the frame counts are equal, so truncation cannot be seen")
    celdas = {}
    for perturbacion in perturbaciones:
      for rho, F in zip(rhos, largos):
        filas = {b: [dict(acc_te=0.4, split="test")]
                 for b in (padron or T.BRAZOS_CONTAMINACION)}
        for b in (T.BRAZO_SIN_SOP, T.BRAZO_CON_SOP):
            filas[b] = [dict(acc_te=0.45, split="test", acc_s_te=0.95,
                             sopesado=(b == brazo), hp2={}, paso_fin=60,
                             mejor_paso=40, vigia_fin=0.8, pico_te=0.47,
                             tr_e=11.0, tr_d=13.0)]
        if techo is not None:
            # Written by the REAL writer, not by a key this test invents. The
            # field is `techo_no_es_metodo` and not `techo`, so a hand-built
            # fixture agreeing with a hand-built reader would verify both
            # halves and never the join between them.
            filas[techo] = [T._fila_brazo(_nf_techo(), _cu_techo(),
                                          T.PESO_ORACULO)]
            assert any(T.PALABRA_TECHO in k.lower() for k in filas[techo][0]), (
                list(filas[techo][0]), "the writer stopped marking the ceiling")
        clave = ((nombre, float(rho)) if perturbacion is None
                 else (nombre, perturbacion, float(rho)))
        celdas[clave] = dict(
            hp2={}, busquedas=1, brazos=filas,
            indices={7: list(range(int(round(rho * n))))},
            w_resumen={}, figura=_figura_sintetica(F=F, n=n, rho=rho))
    CONTAM = dict(rhos=[float(r) for r in rhos], semillas=[7],
                  pos=T.POS_FILAS[0],
                  brazos=list(padron or T.BRAZOS_CONTAMINACION),
                  hp1={}, hp2={}, celdas=celdas)
    nombradas = [v for v in perturbaciones if v is not None]
    if nombradas:
        CONTAM["perturbaciones"] = list(nombradas)
    return CONTAM


def test_the_contamination_gif_truncates_every_row_to_the_shortest():
    """F = 3, 5, 2 gives 2 frames, and does not raise.

    This is the defect class that has shipped in this notebook three times:
    taking the FIRST row's frame count and indexing every row with it dies
    with an `IndexError` inside the shortest one -- and `nbconvert` exited 0
    anyway. The minimum is the only count every row has.

    The helper has two callers, so it is not a guard kept alive for its own
    test: `animar` truncates three normalizations whose early stopping
    measured [21, 13, 11] frames, and `animar_contaminacion` truncates the
    rows it draws.
    """
    assert toy_plots.cuadros_comunes([3, 5, 2]) == 2
    assert toy_plots.cuadros_comunes([5]) == 5
    assert toy_plots.cuadros_comunes([4, 4]) == 4
    # Never the first, which is what the defect did.
    assert toy_plots.cuadros_comunes([7, 2]) == 2
    assert toy_plots.cuadros_comunes([2, 7]) == 2
    # And a row that produced nothing is a refusal, not a zero-frame GIF.
    with pytest.raises(AssertionError):
        toy_plots.cuadros_comunes([3, 0])
    with pytest.raises(AssertionError):
        toy_plots.cuadros_comunes([])


def test_each_contamination_row_is_labelled_with_its_own_stop_step():
    """Its OWN `mejor_paso`, never a neighbour's or the first row's.

    Measured on `lunas`, the ladder stops at 52, 44, 52 and 102. A label
    carrying one row's number over another's panels is a figure that says the
    criterion fired at a step it did not, and nothing raises.
    """
    corta = _figura_sintetica(F=4, mejor_paso=20)
    larga = _figura_sintetica(F=4, mejor_paso=40)
    a = toy_plots.rotulo_fila_contaminacion((None, 0.10), corta, 1)
    b = toy_plots.rotulo_fila_contaminacion((None, 0.30), larga, 1)
    assert "paró en el paso 20" in a, a
    assert "paró en el paso 40" in b, b
    # The neighbour's stop step, stated exactly, so the assertion cannot pass
    # on an empty slice: a bare `"40" not in a.split("paró")[1]` does, because
    # "disparó" contains "paró" and the split lands past the number.
    assert "paró en el paso 40" not in a, a
    assert "paró en el paso 20" not in b, b
    # The frame's own step is on the label too, so the reader can place the
    # stop relative to what is on screen.
    assert f"paso {corta['pasos'][1]}" in a, a
    # And the label says whether the criterion has already fired by this frame.
    temprano = toy_plots.rotulo_fila_contaminacion((None, 0.30), larga, 0)
    tarde = toy_plots.rotulo_fila_contaminacion((None, 0.30), larga, 3)
    assert temprano != tarde, (temprano, tarde)
    # And the label names the perturbation, because two of them run at the same
    # rho and predict the OPPOSITE thing about the weight pair.
    nombrada = toy_plots.rotulo_fila_contaminacion(("frontera", 0.30), larga, 1)
    assert "frontera" in nombrada, nombrada
    assert toy_plots.SIN_PERTURBACION_DECLARADA in b, b


def test_the_contaminated_points_drawn_are_exactly_the_recorded_indices():
    """Selected by `idx`, never recomputed -- and `len(idx) == round(rho*n)`.

    A recomputed guess would draw a set of the right SIZE in the wrong place,
    which is indistinguishable from the real thing at a glance.
    """
    f = _figura_sintetica(F=3, n=20, rho=0.25)
    Pt = np.asarray(f["Pt"][0], dtype=float)
    idx = np.asarray(f["idx"], dtype=int)
    assert len(idx) == int(round(0.25 * 20)) == 5, len(idx)
    fig, a = plt.subplots()
    try:
        toy_plots.panel_contaminacion(a, Pt, f["w"][0], idx,
                                      [(-4, 4), (-4, 4)])
        # Two collections: the clean background first, then the contaminated.
        assert len(a.collections) == 2, len(a.collections)
        dibujados = np.asarray(a.collections[1].get_offsets())
        assert dibujados.shape == (len(idx), 2), dibujados.shape
        assert np.array_equal(dibujados, Pt[idx]), (dibujados, Pt[idx])
        limpios = np.asarray(a.collections[0].get_offsets())
        assert limpios.shape == (len(Pt) - len(idx), 2), limpios.shape
        # And the panel names the split it draws, because the tables beside it
        # report test and an unlabelled panel reads as test.
        assert toy_plots.SPLIT_DIBUJADO in a.get_title(), a.get_title()
    finally:
        plt.close(fig)


def test_the_marker_size_is_monotone_in_w_between_the_declared_endpoints():
    """Strictly increasing across the interval, and exact at 0 and 1.

    The affine map is already asserted at three points; this is the statement
    a non-monotone replacement would break while still hitting the endpoints.
    """
    ws = np.linspace(0.0, 1.0, 51)
    tam = np.asarray(toy_plots.tamano_por_w(ws), dtype=float)
    assert float(tam[0]) == pytest.approx(toy_plots.TAM_MIN_W)
    assert float(tam[-1]) == pytest.approx(toy_plots.TAM_MAX_W)
    assert np.all(np.diff(tam) > 0), np.diff(tam).min()
    # The legend shows the map, so the reader can read a size rather than
    # believe one.
    assert toy_plots.REFERENCIAS_W[0] == 0.0
    assert toy_plots.REFERENCIAS_W[-1] == 1.0


def test_the_mean_weight_pair_splits_on_the_recorded_indices():
    """Over `idx` and over its complement, against known means.

    This pair IS the figure: the prediction written before the run is that the
    contaminated mean does not fall below the clean one, because `w` measures
    uncertainty and a confident mistake is confident. Computing it over the
    wrong partition would answer a different question with the same shape.
    """
    w = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    idx = np.array([0, 1])
    sucio, limpio = toy_plots.medias_w(w, idx)
    assert sucio == pytest.approx(1.0), sucio
    assert limpio == pytest.approx(0.25), limpio
    # The complement is the complement, not "everything".
    assert limpio != pytest.approx(float(w.mean()))
    # No contamination is no number, not a zero: a zero would plot.
    vacio, todo = toy_plots.medias_w(w, np.array([], dtype=int))
    assert vacio is None, vacio
    assert todo == pytest.approx(float(w.mean()))


def test_the_contamination_figure_names_the_arm_the_artifact_records():
    """Derived from the `sopesado` flag each row recorded, never assumed.

    The npz holds ONE arm's trajectory and no arm key, so the name has to come
    from the half of the record that does say: exactly one arm's rows declare
    `sopesado: True`, and that is the arm whose `w` was saved.
    """
    CONTAM = _contam_sintetico()
    assert toy_plots.brazo_de_la_figura(CONTAM) == T.BRAZO_CON_SOP
    # Move the flag and the answer moves with it.
    movido = _contam_sintetico(brazo=T.BRAZO_SIN_SOP)
    assert toy_plots.brazo_de_la_figura(movido) == T.BRAZO_SIN_SOP
    # Two flagged arms, or none, is a refusal rather than a guess.
    dos = _contam_sintetico()
    for celda in dos["celdas"].values():
        for fila in celda["brazos"][T.BRAZO_SIN_SOP]:
            fila["sopesado"] = True
    with pytest.raises(ValueError):
        toy_plots.brazo_de_la_figura(dos)
    ninguno = _contam_sintetico()
    for celda in ninguno["celdas"].values():
        for b in (T.BRAZO_SIN_SOP, T.BRAZO_CON_SOP):
            for fila in celda["brazos"][b]:
                fila["sopesado"] = False
    with pytest.raises(ValueError):
        toy_plots.brazo_de_la_figura(ninguno)
    # And the case only the CROSS-scenario half can catch: each scenario flags
    # exactly one arm, and they flag DIFFERENT ones. The per-scenario guard
    # sees one flag in each and passes; the artifact still does not say which
    # arm's `w` it saved. Measured by probe: the per-scenario guard is
    # subsumed, so this is the half that is load-bearing for detection.
    cruzado = _contam_sintetico(rhos=(0.10, 0.30), largos=(3, 4))
    celdas = [cruzado["celdas"][("lunas", float(r))] for r in (0.10, 0.30)]
    for fila in celdas[1]["brazos"][T.BRAZO_CON_SOP]:
        fila["sopesado"] = False
    for fila in celdas[1]["brazos"][T.BRAZO_SIN_SOP]:
        fila["sopesado"] = True
    for celda, esperado in zip(celdas, (T.BRAZO_CON_SOP, T.BRAZO_SIN_SOP)):
        flags = {b: [f.get("sopesado") for f in filas]
                 for b, filas in celda["brazos"].items()}
        assert [b for b, v in flags.items() if v and all(v)] == [esperado], flags
    with pytest.raises(ValueError):
        toy_plots.brazo_de_la_figura(cruzado)


def test_the_animated_scenario_is_the_highest_rung_of_the_recorded_ladder():
    """The PAIR `(perturbation, rho)`, both read off the artifact.

    A hardcoded level silently animates a scenario the sweep did not run as
    soon as the ladder moves, and the ladder has already moved once. A rho on
    its own is worse than hardcoded: two perturbations run at the same
    `rho = 0.30` and they predict the OPPOSITE direction for the weight pair,
    so selecting by rho picks one of them without saying which.
    """
    CONTAM = _contam_sintetico(rhos=(0.10, 0.30), largos=(3, 4))
    assert toy_plots.escenario_del_gif(CONTAM) == (None, pytest.approx(0.30))
    otro = _contam_sintetico(rhos=(0.05, 0.15), largos=(3, 4))
    assert toy_plots.escenario_del_gif(otro) == (None, pytest.approx(0.15))
    # Named, and the highest rung WITHIN that perturbation.
    par = _contam_sintetico(rhos=(0.10, 0.30), largos=(3, 4),
                            perturbaciones=("reubicado", "frontera"))
    assert toy_plots.escenario_del_gif(par, "frontera") == (
        "frontera", pytest.approx(0.30))
    assert toy_plots.escenario_del_gif(par, "reubicado") == (
        "reubicado", pytest.approx(0.30))


def test_two_perturbations_at_one_rho_refuse_to_pick_one_by_default():
    """No default, because a default would choose the conclusion.

    `reubicado` is the adversarial regime -- the perturbed points land decided
    inside the wrong class, so `w` is HIGH and the weighting provably breaks.
    `frontera` places them BETWEEN classes, where `g` really flattens, so `w`
    DROPS and discounting them is right. One scenario drawn under the other's
    caption states the opposite of what it measured, and nothing raises.
    """
    par = _contam_sintetico(rhos=(0.10, 0.30), largos=(3, 4),
                            perturbaciones=("reubicado", "frontera"))
    with pytest.raises(ValueError) as caida:
        toy_plots.escenario_del_gif(par)
    assert "reubicado" in str(caida.value) and "frontera" in str(caida.value)
    # An unknown name is a refusal too, never the nearest match.
    with pytest.raises(ValueError):
        toy_plots.escenario_del_gif(par, "no-existe")
    # And a roster that disagrees with the keys is a refusal: checking one half
    # verifies the half nobody breaks.
    torcido = _contam_sintetico(rhos=(0.10, 0.30), largos=(3, 4),
                                perturbaciones=("reubicado",))
    assert torcido["perturbaciones"] == ["reubicado"]
    torcido["perturbaciones"] = ["frontera"]
    with pytest.raises(ValueError):
        toy_plots.perturbaciones_del_artefacto(torcido)
    # A perturbation with no written prediction does not borrow the other's.
    with pytest.raises(ValueError):
        toy_plots.rotulo_perturbacion("inventada")
    assert set(toy_plots.PREDICCION_PERTURBACION) == {"reubicado", "frontera"}
    assert "ARRIBA" in toy_plots.PREDICCION_PERTURBACION["reubicado"]
    assert "DEBAJO" in toy_plots.PREDICCION_PERTURBACION["frontera"]
    # And a prediction quotes NO measured number. It used to carry the one
    # cell that confirmed it (`lunas`, rho=0.20, 0.8226 against 0.7746) while
    # 7 of the 9 measured cells end the OTHER way -- a derived statement that
    # breaks the rules of the measurement it came from, printed beside the
    # measurement that contradicts it. The number lives in the panel and in
    # the measured half of the caption, where it is recomputed every frame.
    import re
    for nombre, texto in toy_plots.PREDICCION_PERTURBACION.items():
        cifras = re.findall(r"\d+\.\d{3,}", texto)
        assert cifras == [], (nombre, cifras,
            "a prediction written before the run is quoting a measurement, so "
            "it can go stale and still read as evidence")


def _cu_techo():
    fila = [0, np.zeros((2, 2), dtype=np.float32), 0.5, 0.5,
            np.zeros(2, dtype=np.int64), 0.5, 0.5, 0.5]
    assert len(fila) == T.CAMPOS_CU, (len(fila), T.CAMPOS_CU)
    return [tuple(fila)]


def _nf_techo():
    # `tr_e`/`tr_d` DISTINTOS entre si y de todo lo demas: con dos campos
    # iguales un cruce entre ellos pasaria el test sin que nada falle.
    return dict(acc_te=0.5, hp2=dict(T.HP_2), acc_s_te=0.5, paso_fin=1,
                mejor_paso=1, vigia_fin=0.5, tr_e=11.0, tr_d=13.0)


def test_a_ceiling_arm_is_marked_as_one_or_the_table_says_it_is_not():
    """An arm that reads `y_t` is a CEILING, never a competing method.

    The third CREDA arm (`w oráculo`) reads the true target labels, so putting
    it in the ladder unmarked publishes a method that beats the others by
    cheating. The marking is read from `techo: True` on the arm's own rows --
    the same way `sopesado` is -- and NEVER guessed from the name: an oracle
    named otherwise stays unmarked, and an honest arm whose name contains the
    word gets marked for nothing.

    A record that carries no flag is the shape on disk today, and the table
    SAYS so rather than printing an unmarked ladder that looks complete.
    """
    # The arm name comes from the writer too, so a rename cannot leave this
    # test marking a name nobody writes.
    padron = list(T.BRAZOS_CONTAMINACION)
    assert T.BRAZO_TECHO in padron, (T.BRAZO_TECHO, padron)
    marcado = _contam_sintetico(padron=padron, techo=T.BRAZO_TECHO)
    assert toy_plots.brazos_techo(marcado) == {T.BRAZO_TECHO}
    # The join, crossed: the reader finds the field the writer actually wrote,
    # whose name is `techo_no_es_metodo` and not `techo`. A reader looking for
    # a bare `techo` key passes every hand-built fixture and misses this.
    escrita = T._fila_brazo(_nf_techo(), _cu_techo(), T.PESO_ORACULO)
    assert "techo" not in escrita, (
        list(escrita), "the field is literally named `techo`, so this test no "
        "longer distinguishes reading the word from reading the key")
    assert toy_plots._dice_techo(escrita) is True, escrita
    assert toy_plots._dice_techo(
        T._fila_brazo(_nf_techo(), _cu_techo(), T.PESO_DECLARADO)) is False
    md = toy_plots.tabla_aciertos_contaminacion(
        "lunas", (None, 0.30), marcado, _banderas_contaminacion()).data
    assert toy_plots.MARCA_TECHO in md, md
    assert "TECHO" in md
    assert T.BRAZO_TECHO in md, md
    # Unflagged: no marks, and the footnote reports zero rather than staying
    # silent about a ladder that may contain an oracle.
    sin = _contam_sintetico(padron=padron)
    assert toy_plots.brazos_techo(sin) == set()
    md2 = toy_plots.tabla_aciertos_contaminacion(
        "lunas", (None, 0.30), sin, _banderas_contaminacion()).data
    assert toy_plots.MARCA_TECHO not in md2
    assert "no marca ningún brazo" in md2, md2
    # And the roster comes from the artifact, so a third CREDA arm appears
    # without anybody editing a list of four.
    for brazo in padron:
        assert brazo in md2, (brazo, "the arm roster was hardcoded")


def test_the_contamination_gif_is_written_with_more_than_one_frame(tmp_path):
    """Written, and an animation rather than a single still.

    A mere existence check passes on a one-frame GIF, which is a PNG with a
    different suffix and exactly what a broken truncation produces.
    """
    CONTAM = _contam_sintetico(rhos=(0.10, 0.30), largos=(3, 4))
    ruta = toy_plots.animar_contaminacion(
        "lunas", CONTAM, _banderas_contaminacion(), tmp_path,
        al_paso=lambda *_: None)
    assert ruta.exists(), ruta
    assert ruta.suffix == ".gif", ruta
    # The perturbation is in the NAME, even when the record declares none: two
    # perturbations run at one rho, and a shared filename means the second
    # overwrites the first and leaves one file fewer that nobody looks at.
    assert toy_plots.RANURA_SIN_PERTURBACION in ruta.name, ruta.name
    esperado = len(CONTAM["celdas"][("lunas", 0.30)]["figura"]["pasos"])
    assert toy_plots.escenario_del_gif(CONTAM) == (None, pytest.approx(0.30))
    assert esperado == 4, esperado
    from PIL import Image as _Im
    with _Im.open(ruta) as g:
        assert g.n_frames > 1, g.n_frames
        assert g.n_frames == esperado, (g.n_frames, esperado)


def _banderas_contaminacion():
    """The flags the drawing half reads, with the seed count the fixture has.

    `_semillas` REFUSES a flag that disagrees with the record rather than
    preferring either, so a fixture whose flag says a different count would
    fail for the wrong reason.
    """
    banderas = dict(BALANCE_DE_MASA="trace", POS_FILAS=list(T.POS_FILAS),
                    SEMILLAS_REPORTE=[7], TECHO_RAMPA=1.0)
    assert len(banderas["SEMILLAS_REPORTE"]) == 1, banderas
    return banderas


def test_the_weight_panel_reports_the_ordering_it_measured(tmp_path):
    """The panel says which mean is on top, measured, not predicted.

    The caption states the prediction -- `w` is uncertainty, so a confident
    mistake enters at full weight and the contaminated mean ends ABOVE the
    clean one. A caption is a sentence and ages without failing; this is the
    drawing code measuring the same claim, so the figure adjusts itself and
    says what it found instead of asserting what was expected.
    """
    pasos = [0, 20, 40]
    n = 10
    idx = np.array([0, 1])
    # Hand-built so the ordering is known: the contaminated mean ends ABOVE.
    W = [np.array([0.2, 0.2] + [0.9] * 8),
         np.array([0.5, 0.5] + [0.5] * 8),
         np.array([0.9, 0.9] + [0.2] * 8)]
    assert toy_plots.medias_w(W[0], idx)[0] < toy_plots.medias_w(W[0], idx)[1]
    assert toy_plots.medias_w(W[2], idx)[0] > toy_plots.medias_w(W[2], idx)[1]
    fig, a = plt.subplots()
    try:
        toy_plots.panel_medias_w(a, pasos, W, idx, 2)
        titulo = a.get_title()
        assert "ARRIBA" in titulo, titulo
        assert toy_plots.SPLIT_DIBUJADO in titulo or "w" in titulo, titulo
        assert a.get_yscale() == "linear", a.get_yscale()
        assert a.get_ylim() == (0.0, 1.0), a.get_ylim()
        # Progressive: only the frames up to `j`, never the whole trajectory.
        xs = [list(l.get_xdata()) for l in a.get_lines() if len(l.get_xdata()) > 1]
        assert xs, "no curve was drawn"
        assert all(max(x) <= pasos[2] for x in xs), xs
    finally:
        plt.close(fig)
    # And the other ordering is reported as the other ordering.
    fig, a = plt.subplots()
    try:
        toy_plots.panel_medias_w(a, pasos, W, idx, 0)
        assert "DEBAJO" in a.get_title(), a.get_title()
    finally:
        plt.close(fig)


def test_the_gif_caption_states_the_measured_ending_and_the_split():
    """Both halves of the sentence the section rests on, on the figure itself.

    The contamination is injected in the target TRAIN split and the metric is
    read on TEST. The GIF shows the first half and the tables show the second,
    so each has to be legible alone -- an unlabelled panel of train points
    beside a table of test accuracy reads as test.
    """
    pie = toy_plots.PIE_CONTAMINACION_GIF
    assert toy_plots.PIE_CONTAMINACION in pie, (
        "the GIF caption dropped the static figure's caption instead of "
        "extending it, so the two can now disagree about the same claim")
    assert toy_plots.SPLIT_DIBUJADO in pie, pie
    assert "INCERTIDUMBRE" in pie, pie
    # WHY the split is the one it draws, and not only which one it is. A
    # probe that deleted this clause left every other assertion here green,
    # which is the measurement that this sentence was unheld: `w` exists on
    # train because that is where the term is evaluated, and a figure that
    # names its split without saying that invites the next reader to ask for
    # the same panel on test.
    assert "sobre test" in pie, pie
    assert "término" in pie, pie
    # The DIRECTION is not in the constant, and must not be: the two declared
    # perturbations predict the opposite, so a fixed caption would be wrong
    # over half the section. It comes from the measurement.
    assert "ARRIBA" not in pie and "DEBAJO" not in pie, pie
    arriba = toy_plots.pie_del_gif("reubicado", 0.82, 0.77)
    abajo = toy_plots.pie_del_gif("frontera", 0.60, 0.80)
    assert "**ARRIBA de**" in arriba, arriba
    assert "**DEBAJO de**" in abajo, abajo
    assert "0.8200" in arriba and "0.7700" in arriba, arriba
    # And with no declared perturbation, no prediction is borrowed.
    muda = toy_plots.pie_del_gif(None, 0.82, 0.77)
    assert "no declara qué perturbación" in muda, muda
    assert toy_plots.PREDICCION_PERTURBACION["reubicado"] not in muda
    assert "**ARRIBA de**" in muda, muda


def test_the_contamination_accuracy_table_names_what_the_record_cannot_fill():
    """Test accuracy for the four arms, and the absent columns NAMED.

    `tabla_dataset` is the shape this follows, and the contamination record is
    narrower than the main sweep's: it carries no per-seed source epoch, no
    `.predict` source accuracy and no per-checkpoint test series. Filling
    those columns from something else would be a table that reads like the
    first section's and answers a different question, so they are measured as
    absent and named in the footnote.
    """
    CONTAM = _contam_sintetico()
    md = toy_plots.tabla_aciertos_contaminacion(
        "lunas", (None, 0.30), CONTAM, _banderas_contaminacion()).data
    assert "blanco · test" in md, md
    assert "fuente · test" in md, md
    for brazo in T.BRAZOS_CONTAMINACION:
        assert brazo in md, (brazo, md)
    # The absent fields, measured against what `tabla_dataset` reads.
    assert toy_plots.CAMPOS_TABLA_DATASET, "the enumeration is empty"
    fuente = (PAQUETE / "toy_plots.py").read_text(encoding="utf-8")
    cuerpo = fuente.split("def tabla_dataset", 1)[1].split("\ndef ", 1)[0]
    for campo in toy_plots.CAMPOS_TABLA_DATASET:
        assert f"'{campo}'" in cuerpo or f'"{campo}"' in cuerpo, (
            campo, "declared as a tabla_dataset field and not read by it")
    for campo in ("ep_s", "acc_predict", "acc_te_ser"):
        assert campo in md, (campo, "absent from the record and unnamed")
    # And the split it reports is named, so it cannot be confused with the GIF.
    assert "test" in md


# ===================== 9. the record's own shape, written and read back
# The producer and the consumer are two halves of one rule: the sweep writes a
# cell key and a roster, and the drawing side resolves a scenario out of them.
# Every test above this point holds one half against a fixture written by the
# same hand, which verifies both halves and never the join. These run the
# writer, read the bytes it left, and ask the reader what it sees.


def _relabel(CONTAM, desde, hacia):
    """The same sweep, relabelled to another perturbation name.

    **This is a RECORD fixture and not a second measurement.** The numbers are
    the ones the first sweep produced; what is being tested is the shape of the
    record -- two names in one artifact, keyed apart -- and running a second
    real sweep to assert a key costs a minute of compute to check a string.
    Nothing here reads these numbers as a result of the boundary perturbation.
    """
    assert CONTAM["perturbaciones"] == [desde], CONTAM["perturbaciones"]
    otro = dict(CONTAM)
    otro["perturbaciones"] = [hacia]
    otro["celdas"] = {(f, hacia, r): v
                      for (f, p, r), v in CONTAM["celdas"].items()}
    assert len(otro["celdas"]) == len(CONTAM["celdas"])
    return otro


def test_the_sweep_writes_triple_keys_and_a_perturbation_roster(barrido_minimo):
    """The producer's own shape: a LIST roster and the perturbation in the key.

    A scalar `perturbacion` field cannot grow to two without every reader
    changing shape, and a `(familia, rho)` key cannot tell two perturbations
    apart at all -- at one rho they run both and predict the OPPOSITE direction
    for the weight pair, so the second sweep would overwrite the first in
    silence.
    """
    CONTAM = barrido_minimo
    assert "perturbacion" not in CONTAM, (
        "the scalar field is back; a scalar cannot carry two perturbations")
    assert CONTAM["perturbaciones"] == [T.MODO_REUBICADO], (
        CONTAM["perturbaciones"])
    assert T.MODO_REUBICADO == "reubicado", T.MODO_REUBICADO
    for clave in CONTAM["celdas"]:
        assert len(clave) == 3, (clave, "not a (familia, perturbacion, rho)")
        assert clave[1] == T.MODO_REUBICADO, clave
    # The reader resolves a perturbation for EVERY key, and never `None`.
    for clave in CONTAM["celdas"]:
        resuelta = toy_plots._perturbacion_de_clave(clave)
        assert resuelta == T.MODO_REUBICADO, (clave, resuelta)
        assert resuelta is not None, clave
    # And its two halves agree, which is the cross-check the reader does.
    assert toy_plots.perturbaciones_del_artefacto(CONTAM) == [
        T.MODO_REUBICADO], toy_plots.perturbaciones_del_artefacto(CONTAM)


def test_both_perturbation_names_round_trip_through_one_artifact(
        barrido_minimo, tmp_path):
    """One file, two perturbations, keyed apart. And an unknown name refused.

    The suffix-per-perturbation scheme this replaces meant the drawing side had
    to ask for the one it wanted, and a default there draws one scenario under
    the other's caption -- which states the opposite of what it measured.
    """
    fundido = T.fusionar_contaminacion(
        barrido_minimo, _relabel(barrido_minimo, T.MODO_REUBICADO,
                                 T.MODO_FRONTERA))
    assert fundido["perturbaciones"] == ["reubicado", "frontera"], (
        fundido["perturbaciones"])

    escritos = T.guardar_contaminacion(tmp_path, fundido)
    nombres = sorted(Path(p).name for p in escritos)
    assert nombres == ["contaminacion.json",
                       "contaminacion_lunas.npz"], nombres

    vuelto = T.cargar_contaminacion(tmp_path)
    assert vuelto["perturbaciones"] == ["reubicado", "frontera"], (
        vuelto["perturbaciones"])
    assert set(vuelto["celdas"]) == set(fundido["celdas"]), (
        sorted(str(k) for k in vuelto["celdas"]))
    # Both names come back, both figures survive the shared .npz, and the
    # reader resolves each key rather than defaulting.
    for nombre in ("reubicado", "frontera"):
        claves = [k for k in vuelto["celdas"]
                  if toy_plots._perturbacion_de_clave(k) == nombre]
        assert claves, (nombre, sorted(str(k) for k in vuelto["celdas"]))
        for k in claves:
            assert vuelto["celdas"][k]["figura"] is not None, (
                k, "the figure of one perturbation was overwritten by the "
                "other inside the shared .npz")
    assert toy_plots.perturbaciones_del_artefacto(vuelto) == [
        "frontera", "reubicado"]

    # An unknown name is a refusal and never the nearest match or a default.
    with pytest.raises(ValueError):
        toy_plots.escenario_del_gif(vuelto, "no-existe")
    # With two declared there is no default at all.
    with pytest.raises(ValueError, match="2 perturbaciones"):
        toy_plots.escenario_del_gif(vuelto)
    # And the writer refuses a name the module does not declare, before the
    # bytes are written: discovering it on the drawing side costs the sweep.
    torcido = dict(fundido, perturbaciones=["inventada"])
    with pytest.raises(AssertionError):
        T.guardar_contaminacion(tmp_path / "no", torcido)
    # A roster that disagrees with the keys is refused on the way OUT too, not
    # only on the way in: checking one direction verifies the half nobody
    # breaks.
    desparejo = dict(fundido, perturbaciones=["reubicado"])
    with pytest.raises(AssertionError, match="no coinciden"):
        T.guardar_contaminacion(tmp_path / "no", desparejo)


def test_every_rho_selector_on_a_fused_artifact_refuses_to_pick_a_default(
        barrido_minimo, tmp_path):
    """A fused artifact is read correctly, and NOTHING resolves it by rho alone.

    `escenario_del_gif` already refuses. The other three selectors reach a cell
    through `_celda_contaminacion`, and a function with no `perturbacion`
    parameter cannot be told which scenario to draw at all -- so on a fused
    artifact it either refuses or, worse, would draw one scenario under the
    other's caption. `reubicado` is the adversarial case where the weighting
    provably breaks and `frontera` is the case where it can work: the two
    predict the OPPOSITE direction at the same rho, so a default picks the
    conclusion.

    This crosses the join rather than testing each half against a fixture of
    its own: the artifact is the real fused one and the selectors are asked
    what they see.
    """
    fundido = T.fusionar_contaminacion(
        barrido_minimo, _relabel(barrido_minimo, T.MODO_REUBICADO,
                                 T.MODO_FRONTERA))
    assert sorted(toy_plots.perturbaciones_del_artefacto(fundido)) == [
        "frontera", "reubicado"], toy_plots.perturbaciones_del_artefacto(
            fundido)
    nombre = T.FAMILIAS[0]["nombre"]
    rho = max(fundido["rhos"])
    banderas = {"POS_FILAS": list(T.POS_FILAS), "SEMILLAS_REPORTE": [0, 1],
                "BALANCE_DE_MASA": "trace"}

    # The table: refuses without a name, and names both so the reader can pick.
    with pytest.raises(ValueError, match="perturbacion") as caida:
        toy_plots.tabla_cuatro_brazos(nombre, rho, fundido, banderas)
    assert "frontera" in str(caida.value), str(caida.value)
    assert "reubicado" in str(caida.value), str(caida.value)
    # And with a name it resolves, to that perturbation's own cell.
    for perturbacion in ("reubicado", "frontera"):
        res = toy_plots.comparar_cuatro_brazos(
            nombre, rho, fundido, banderas, perturbacion=perturbacion)
        assert res["escenario"] == pytest.approx(float(rho)), res["escenario"]

    # The figure: the same parameter and the same refusal. Without a parameter
    # at all it cannot be named, which is why this asserts the SIGNATURE and
    # not only the behaviour -- a function that refuses and offers no way
    # through corners whoever reads the refusal.
    import inspect
    firma = inspect.signature(toy_plots.figura_contaminacion)
    assert "perturbacion" in firma.parameters, (
        sorted(firma.parameters),
        "the figure cannot say which scenario it draws")
    assert firma.parameters["perturbacion"].default is None, firma
    with pytest.raises(ValueError, match="perturbacion"):
        toy_plots.figura_contaminacion(nombre, fundido, banderas)

    # And the parameter is PLUMBED THROUGH, which the refusal above cannot
    # show: the ambiguity is raised inside the cell lookup whether or not the
    # figure forwards what it was given, so a figure that accepted the argument
    # and dropped it passed the refusal test unchanged -- measured, by probe.
    # Each perturbation is drawn, lands in a file of ITS OWN name, and says so
    # in its title; one file for both would lose one drawing in silence.
    rutas = {}
    for perturbacion in ("reubicado", "frontera"):
        ruta = toy_plots.figura_contaminacion(
            nombre, fundido, banderas, tmp_path, perturbacion=perturbacion)
        assert Path(ruta).exists(), ruta
        assert perturbacion in Path(ruta).name, (
            Path(ruta).name, perturbacion,
            "two perturbations write one figure, so one is overwritten")
        rutas[perturbacion] = Path(ruta)
    assert rutas["reubicado"] != rutas["frontera"], rutas
    assert rutas["reubicado"].read_bytes() != rutas["frontera"].read_bytes(), (
        "both figures are byte-identical, so the perturbation reached the "
        "filename and not the drawing")

    # The animation already takes one; it is asserted here beside the others so
    # the three are read as one rule rather than three habits.
    assert "perturbacion" in inspect.signature(
        toy_plots.animar_contaminacion).parameters
    with pytest.raises(ValueError, match="2 perturbaciones"):
        toy_plots.escenario_del_gif(fundido)


def test_the_ceiling_mark_reaches_the_saved_json_and_comes_back(
        barrido_minimo, tmp_path):
    """The ceiling marking's round trip, through the real writer and real bytes.

    `test_the_oracle_arm_is_labelled_a_ceiling_in_the_artifact_it_writes`
    pins `_fila_brazo`'s in-memory dict. This is the other half: that the field
    survives `guardar_contaminacion`, lands in the JSON text, loads back, and
    is read as a ceiling by the side that marks the table. No artifact had ever
    been written with five arms, so nobody had seen this join hold.
    """
    CONTAM = barrido_minimo
    assert T.BRAZO_TECHO in CONTAM["brazos"], CONTAM["brazos"]

    T.guardar_contaminacion(tmp_path, CONTAM)
    crudo = (tmp_path / "contaminacion.json").read_text(encoding="utf-8")
    # The word is in the JSON TEXT, which is what a person printing the record
    # reads. A boolean under some other key would not be.
    assert T.PALABRA_TECHO in crudo.lower(), (
        "no field name in the written JSON carries the ceiling word")
    plano = json.loads(crudo)
    celda = next(iter(plano["celdas"].values()))
    fila = celda["brazos"][T.BRAZO_TECHO][0]
    claves = [k for k in fila if T.PALABRA_TECHO in k.lower().split("_")]
    assert claves, (sorted(fila), "the ceiling field did not reach the JSON")
    assert all(fila[k] is True for k in claves), (fila, claves)
    # And the method arms are marked FALSE rather than merely lacking it, where
    # the writer records them at all.
    for b in (T.BRAZO_SIN_SOP, T.BRAZO_CON_SOP):
        otra = celda["brazos"][b][0]
        assert all(otra[k] is False for k in claves), (b, otra)

    vuelto = T.cargar_contaminacion(tmp_path)
    assert toy_plots.brazos_techo(vuelto) == {T.BRAZO_TECHO}, (
        toy_plots.brazos_techo(vuelto),
        "the ceiling marking did not survive the round trip")
    # The four-arm table reads it as a ceiling: out of the treatments, in its
    # own row. That is the consequence the round trip exists for.
    nombre = T.FAMILIAS[0]["nombre"]
    res = toy_plots.comparar_cuatro_brazos(
        nombre, max(vuelto["rhos"]), vuelto,
        {"POS_FILAS": list(T.POS_FILAS), "SEMILLAS_REPORTE": [0, 1],
         "BALANCE_DE_MASA": "trace"},
        perturbacion=T.MODO_REUBICADO)
    assert T.BRAZO_TECHO not in res["brazos"], res["brazos"]
    assert T.BRAZO_TECHO not in res["rangos"], res["rangos"]
    assert res["no_tratamientos"] == [T.BRAZO_TECHO], res["no_tratamientos"]
    assert [t["brazo"] for t in res["filas_techo"]] == [T.BRAZO_TECHO]


# --------------------------------------------------------------------------
# The four-arm table's statistics: the sign, the Holm family and the arity.
#
# These three are the half of this table that breaks WITHOUT MOVING A NUMBER.
# An inverted sign prints the same magnitude and says the other arm won; a Holm
# family of six where four were declared prints a larger adjusted p and a
# tamer conclusion; an arity inherited from the sweep's roster ranks the
# ceiling among the methods and moves every other arm's rank. None of the
# three raises, and none of them is visible in a figure.
# --------------------------------------------------------------------------

#: Ten seeds, which is not a round number chosen for looks. Wilcoxon's exact
#: two-sided p is quantized, and the window where Holm's denominator CHANGES A
#: VERDICT at 0.05 is `[0.05/6, 0.05/3) = [0.00833, 0.01667)`. Enumerated over
#: every sign pattern: n = 6 admits no p in that window at all, n = 7 and n = 8
#: admit only 0.01562, n = 9 admits 0.01172, and n = 10 admits 0.00977 and
#: 0.01367. At 0.00977 the three declared pairs give 0.0293 -- significant --
#: and the six pairs of four arms give 0.0586 -- not. A fixture at n = 8 could
#: not show that, which is why the count is measured and not picked.
_SEMILLAS_ESTADISTICAS = 10

#: The axis pair's paired differences: eight seeds favour the weighted arm and
#: the two that do not carry the two SMALLEST magnitudes, which is the sign
#: pattern that lands Wilcoxon on p = 0.00977 exactly.
_DIF_EJE = (-0.001, -0.002, 0.004, 0.006, 0.009,
            0.012, 0.016, 0.021, 0.027, 0.034)

_SIN_SOP = (0.700, 0.705, 0.710, 0.695, 0.715,
            0.690, 0.708, 0.702, 0.712, 0.698)

#: AdaBN and the floor sit CLOSE to the weighted arm rather than below it, and
#: that is deliberate: Holm's denominator only decides a verdict where a p sits
#: near the threshold, so a fixture with the four arms cleanly ordered would
#: leave every pair far from 0.05 and both families would conclude the same
#: thing. The ordering here is not the experiment's expected one and is not
#: meant to be; what this fixture exercises is the correction, not the science.
_ADABN = (0.699037, 0.711962, 0.705776, 0.674282, 0.710360,
          0.672251, 0.725804, 0.763206, 0.724234, 0.713386)
_PISO = (0.713732, 0.722669, 0.708938, 0.646368, 0.709482,
         0.693110, 0.685478, 0.749478, 0.667197, 0.674700)


def _brazos_estadisticos():
    """The four methods plus the ceiling, over `_SEMILLAS_ESTADISTICAS` blocks.

    Asserts its own properties before anything under test reads it, because a
    fixture that already equals the mutant is a passing nothing: the axis pair
    has to land on the p that makes Holm's denominator decide, and the ceiling
    has to be above every method or "the ceiling is not a treatment" would be
    testable by accident.
    """
    sin_sop = np.array(_SIN_SOP, dtype=float)
    con_sop = np.round(sin_sop + np.array(_DIF_EJE, dtype=float), 6)
    brazos = {
        toy_plots.BRAZO_PISO: np.array(_PISO, dtype=float),
        toy_plots.BRAZO_ADABN: np.array(_ADABN, dtype=float),
        T.BRAZO_SIN_SOP: sin_sop,
        T.BRAZO_CON_SOP: con_sop,
    }
    assert list(brazos) == list(toy_plots.BRAZOS_CUATRO), (
        list(brazos), "the fixture's arm order is not the declared one")
    techo = np.round(con_sop + 0.08, 6)
    assert all(len(v) == _SEMILLAS_ESTADISTICAS for v in brazos.values())

    # The property the window was computed for, asserted rather than trusted.
    cruda = float(_st.wilcoxon(con_sop - sin_sop).pvalue)
    assert 0.05 / 6 <= cruda < 0.05 / 3, (
        cruda, "the axis pair's raw p left the window where Holm's "
               "denominator changes the verdict, so this fixture can no "
               "longer tell a family of three from a family of six")
    # And the ceiling is above every method, so a rank that included it would
    # take the top rather than land somewhere harmless in the middle.
    assert all(float(techo.mean()) > float(v.mean()) for v in brazos.values())
    return brazos, techo


def test_the_four_arm_sign_is_the_first_named_arm_minus_the_second():
    """`delta = (first named) - (second named)`, so `delta > 0` means the FIRST
    wins.

    It came out inverted once in this repository -- the subtracted arm sat in
    the first column, so `+0.0819` beside `.predict` read as `.predict` winning
    when it means the opposite -- and a table whose sign is read backwards is
    worse than no table, because it is read with confidence.

    Asserted against the arrays themselves and against the ORDER the declared
    pair names, not against the row's own label: a mutation that swapped both
    the subtraction and the label would keep every number consistent with its
    own heading and still invert the reading of `PARES_CUATRO_BRAZOS`.
    """
    brazos, techo = _brazos_estadisticos()
    res = toy_plots.comparar_n_brazos(
        brazos, toy_plots.PARES_CUATRO_BRAZOS, _SEMILLAS_ESTADISTICAS,
        familia="el signo", extra={toy_plots.BRAZO_TECHO_CUATRO: techo})
    lectura = {**brazos, toy_plots.BRAZO_TECHO_CUATRO: techo}

    assert len(res["filas"]) == len(toy_plots.PARES_CUATRO_BRAZOS)
    for fila, (primero, segundo) in zip(res["filas"],
                                        toy_plots.PARES_CUATRO_BRAZOS):
        # The row reports the pair in the order the DECLARATION names, so the
        # sign cannot be rescued by relabelling.
        assert (fila["primero"], fila["segundo"]) == (primero, segundo)
        esperado = float(np.mean(lectura[primero]) - np.mean(lectura[segundo]))
        assert fila["delta"] == pytest.approx(esperado, abs=1e-12), fila
        # The win count follows the same direction, which is the other half an
        # inversion breaks: `gana` counts the seeds where the FIRST is ahead.
        gana = int((np.asarray(lectura[primero])
                    > np.asarray(lectura[segundo])).sum())
        assert fila["gana"] == gana, fila

    # The axis pair, by name: the weighted arm is ahead on average in this
    # fixture, so its delta must be POSITIVE. Under the inversion it is the
    # same magnitude with the other sign, which is why the magnitude alone
    # cannot be what this asserts.
    eje = res["filas"][0]
    assert (eje["primero"], eje["segundo"]) == (T.BRAZO_CON_SOP,
                                                T.BRAZO_SIN_SOP)
    assert eje["delta"] > 0, eje
    assert eje["delta"] == pytest.approx(float(np.mean(_DIF_EJE)), abs=1e-12)


def test_the_four_arm_holm_corrects_within_its_own_declared_pairs():
    """Holm's family is `PARES_CUATRO_BRAZOS` -- three pairs -- and never the
    six pairs four arms admit.

    **The name used to assert a number and the number went stale.** It read
    `..._within_its_own_six_pairs`; six was every pair of four arms, the
    roster then grew to five, and a declaration naming a count that no longer
    holds is worse than none, because its test stays green on the old fixture
    and nobody reads the name again.

    Measured, which is why this fixture is at ten seeds: the axis pair's raw p
    is 0.00977, so Holm over THREE gives 0.0293 and Holm over SIX gives
    0.0586. At 0.05 that is significant and not significant -- the family is
    not a cosmetic denominator, it decides the conclusion of the one pair the
    section exists to report.
    """
    brazos, techo = _brazos_estadisticos()
    res = toy_plots.comparar_n_brazos(
        brazos, toy_plots.PARES_CUATRO_BRAZOS, _SEMILLAS_ESTADISTICAS,
        familia="la familia", extra={toy_plots.BRAZO_TECHO_CUATRO: techo})

    # The declared family is three, and the table reports the REAL denominator
    # rather than `len(pares)`.
    assert len(toy_plots.PARES_CUATRO_BRAZOS) == 3, toy_plots.PARES_CUATRO_BRAZOS
    assert res["familia_de"] == 3, res["familia_de"]
    assert res["pares_sin_p"] == [], res["pares_sin_p"]

    # Recomputed from the raw p's of exactly those pairs, so a family that
    # silently included another pair disagrees here.
    crudas = [fila["p"] for fila in res["filas"]]
    assert [fila["holm"] for fila in res["filas"]] == pytest.approx(
        toy_plots._holm(crudas))

    # The six pairs four arms admit, which is what the stale name corrected
    # over. Same data, same Wilcoxon, different denominator.
    seis = tuple((a, b) for i, a in enumerate(brazos)
                 for b in list(brazos)[i + 1:])
    assert len(seis) == 6, seis
    ancha = toy_plots.comparar_n_brazos(
        brazos, seis, _SEMILLAS_ESTADISTICAS, familia="los seis")
    assert ancha["familia_de"] == 6, ancha["familia_de"]

    def _por_par(resultado, primero, segundo):
        for fila in resultado["filas"]:
            if {fila["primero"], fila["segundo"]} == {primero, segundo}:
                return fila
        raise AssertionError((primero, segundo, resultado["familia"]))

    eje = toy_plots.PARES_CUATRO_BRAZOS[0]
    estrecha = _por_par(res, *eje)
    amplia = _por_par(ancha, *eje)
    # The same raw p, so nothing but the denominator differs.
    assert estrecha["p"] == pytest.approx(amplia["p"])
    assert estrecha["p"] == pytest.approx(0.009766, abs=1e-6), estrecha["p"]
    assert estrecha["holm"] == pytest.approx(0.029297, abs=1e-6)
    assert amplia["holm"] == pytest.approx(0.058594, abs=1e-6)
    # And the verdict, which is the consequence and not a number.
    assert estrecha["holm"] < 0.05 <= amplia["holm"], (
        estrecha["holm"], amplia["holm"],
        "the two families agree on this fixture, so it no longer shows that "
        "the declared family is what decides")


def test_the_oracle_ceiling_is_outside_the_friedman_and_outside_the_ranks():
    """The ceiling is not a treatment: out of the omnibus and out of the ranks.

    **Part of this is already covered under another name** --
    `test_the_ceiling_mark_reaches_the_saved_json_and_comes_back` asserts the
    ceiling is absent from `brazos` and `rangos` and present in
    `no_tratamientos`. What that test cannot say is whether the exclusion is
    doing any WORK: an absence is asserted by looking where the thing would
    live, and the place it would live is the Friedman statistic itself.

    So this runs the mutation: the same ceiling handed in as a treatment rather
    than as `extra`. A ceiling reads the target's labels, so asking a rank
    which method beats it is asking which method beats something no method can
    be -- and it takes the top rank, pushing every method's mean rank down.
    """
    brazos, techo = _brazos_estadisticos()
    fuera = toy_plots.comparar_n_brazos(
        brazos, toy_plots.PARES_CUATRO_BRAZOS, _SEMILLAS_ESTADISTICAS,
        familia="el techo afuera",
        extra={toy_plots.BRAZO_TECHO_CUATRO: techo})

    assert toy_plots.BRAZO_TECHO_CUATRO not in fuera["brazos"]
    assert toy_plots.BRAZO_TECHO_CUATRO not in fuera["rangos"]
    assert fuera["no_tratamientos"] == [toy_plots.BRAZO_TECHO_CUATRO]
    assert fuera["gl_friedman"] == 3, "four treatments give three degrees"
    assert fuera["mejor_rango"] == 4, fuera["mejor_rango"]
    assert sorted(fuera["rangos"]) == sorted(toy_plots.BRAZOS_CUATRO)

    # The mutation: the ceiling as a fifth treatment. Nothing raises, and that
    # is the point -- this is what "silently" means for this defect.
    adentro = dict(brazos)
    adentro[toy_plots.BRAZO_TECHO_CUATRO] = techo
    mutado = toy_plots.comparar_n_brazos(
        adentro, toy_plots.PARES_CUATRO_BRAZOS, _SEMILLAS_ESTADISTICAS,
        familia="el techo adentro")

    assert mutado["gl_friedman"] == 4, "the mutation did not add a treatment"
    assert mutado["mejor_rango"] == 5
    assert mutado["chi2"] != pytest.approx(fuera["chi2"]), (
        fuera["chi2"], mutado["chi2"],
        "the omnibus read the same with and without the ceiling, so this "
        "fixture cannot tell whether the exclusion does anything")
    # And the ceiling takes the top rank, which is the reading it corrupts.
    assert mutado["rangos"][toy_plots.BRAZO_TECHO_CUATRO] == pytest.approx(
        float(mutado["mejor_rango"]))

    # **The four methods keep their mean ranks to the digit, and that is the
    # finding rather than a weakness here.** A first version of this test
    # asserted that they MOVE; they do not, because a ceiling reads the
    # target's labels and therefore sits above every method in every single
    # block, so it takes rank 5 everywhere and displaces nobody. What changes
    # is the SCALE those unchanged numbers are read against: 3.1 out of 4 and
    # 3.1 out of 5 are the same printed rank and two different statements, and
    # the second one is the table silently reporting that its best method
    # reaches 62% of the way up a ladder whose top rung no method can stand on.
    for b in toy_plots.BRAZOS_CUATRO:
        assert mutado["rangos"][b] == pytest.approx(fuera["rangos"][b]), (
            b, "a method's mean rank moved, so the ceiling did not dominate "
               "every block and this fixture is not the one described")
    assert mutado["mejor_rango"] != fuera["mejor_rango"], (
        "the best rank is identical with and without the ceiling, so nothing "
        "here distinguishes the two readings")


def test_the_four_arm_arity_does_not_follow_the_contamination_roster():
    """Four treatments, declared, and never `len(BRAZOS_CONTAMINACION)`.

    This is the chain that was CUT, and the cut is the invariant. The sweep's
    roster grew to five by adding the oracle ceiling; a table deriving its
    treatments from that roster would have ranked the ceiling among the methods
    without a line of code changing. `BRAZOS_CUATRO` is written out instead,
    and the module asserts at import that none of its members carries the word
    that marks a ceiling.

    The mutation is the roster growing again: a sixth arm is added to
    `BRAZOS_CONTAMINACION` and the four-arm declaration must not notice.
    """
    assert len(toy_plots.BRAZOS_CUATRO) == 4, toy_plots.BRAZOS_CUATRO
    # The roster is STRICTLY larger, which is what makes the two distinguishable
    # at all. If they were ever the same length this test proves nothing.
    assert len(T.BRAZOS_CONTAMINACION) > len(toy_plots.BRAZOS_CUATRO), (
        T.BRAZOS_CONTAMINACION, toy_plots.BRAZOS_CUATRO,
        "the roster is no longer bigger than the declared treatments, so a "
        "table deriving one from the other would look identical here")
    assert toy_plots.BRAZO_TECHO_CUATRO in T.BRAZOS_CONTAMINACION, (
        "the ceiling left the roster, so inheriting it would no longer be the "
        "defect this guards")
    assert toy_plots.BRAZO_TECHO_CUATRO not in toy_plots.BRAZOS_CUATRO

    # The word that marks a ceiling, read from the module that writes it.
    for brazo in toy_plots.BRAZOS_CUATRO:
        assert T.PALABRA_TECHO not in brazo.lower(), brazo
    assert T.PALABRA_TECHO in toy_plots.BRAZO_TECHO_CUATRO.lower()

    # And the declared pairs never name a treatment that is not one of the four.
    for primero, segundo in toy_plots.PARES_CUATRO_BRAZOS:
        for brazo in (primero, segundo):
            assert brazo in toy_plots.BRAZOS_CUATRO, (brazo, "a tested pair "
                                                      "names a non-treatment")

    # The mutation: the roster grows and the arity must not follow. Asserted on
    # the ARITY the table actually runs, not only on the tuple's length.
    brazos, techo = _brazos_estadisticos()
    padron_crecido = tuple(T.BRAZOS_CONTAMINACION) + ("brazo nuevo del barrido",)
    assert len(padron_crecido) == len(T.BRAZOS_CONTAMINACION) + 1
    res = toy_plots.comparar_n_brazos(
        brazos, toy_plots.PARES_CUATRO_BRAZOS, _SEMILLAS_ESTADISTICAS,
        familia="el padron crecido",
        extra={toy_plots.BRAZO_TECHO_CUATRO: techo})
    assert len(res["brazos"]) == 4, res["brazos"]
    assert res["gl_friedman"] == 3
    assert "brazo nuevo del barrido" not in res["brazos"]
