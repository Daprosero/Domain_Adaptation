"""The intrusion experiment: damage by latent distance, and who can remove it.

Two earlier constructions of the weighting section came back null, and both
nulls are now explained by measurement rather than guessed at:

* `reubicado` moves a point by `x <- x - mu_own + mu_donor`, which lands it at
  the donor centroid PLUS its own within-class deviation -- about one rms radius
  from the class it routes into, and in four of six measured cells CLOSER to
  that class's centre than the clean points are. A point geometrically
  indistinguishable from a native member of the block it joins distorts no
  second-order structure of that block, so there is nothing for any weighting to
  remove -- not the entropy estimator, and not a perfect oracle. That, and not
  "`w` measures uncertainty rather than error", is why the oracle ceiling came
  out null there.
* pushing a point far along a generic ray makes the posterior MORE confident
  rather than flat: with `UNA_CAPA` the head is affine per region, so along a
  ray the ReLU pattern freezes, the logits grow linearly and the softmax
  saturates.

So this file guards a third construction, and the reason it can produce a
non-null where the other two could not is the first finding above: what creates
damage is GEOMETRIC distance from the block the point joins. `intrusar`
REPLACES a declared fraction of the samples of each class of the target train
split with intruders whose latent lands far from every class centroid, so that
class's second-order structure is distorted BY CONSTRUCTION and there is
genuinely something a weighting could remove.

Whether `w = 1 - H2(g)/ln C` catches it is then a separate, measurable
question, and the three arms separate the two answers: `oraculo - apagado`
bounds what removing the intruders can buy, `declarado - apagado` says how much
of that the entropy estimator captures. A high `w` on the intruders with an
oracle that recovers the damage is the most informative outcome available --
the weighting IDEA contributes while OUR estimator does not capture it -- and
neither previous construction could reach it.

The oracle here is NOT `pesos_oraculo`. That one keys on pseudo-label
correctness over the whole target train, so clean-but-misrouted points are
zeroed too and at zero intrusion it is not a no-op. This experiment's ceiling is
"zero on the intruders, one elsewhere", and the intruders are known exactly
because we injected them.

Every fixture asserts its own property before the thing under test is called. A
fixture that already equals the mutant turns a passing test into a passing
nothing, and that has happened in this repository.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from CREDA import toy_experiment as T

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE = RAIZ / "src" / "CREDA"

#: The grid the premise is measured over: every family, two seeds. One family
#: would let a premise that holds on one geometry pass for all three, and one
#: seed cannot tell a property from a draw.
FAMILIAS_PREMISA = tuple(f["nombre"] for f in T.FAMILIAS)
SEMILLAS_PREMISA = (0, 1)


def _fase1(nombre="lunas", s=0):
    """The clean splits and the frozen phase-1 pair the run itself uses.

    The same object in both halves on purpose: a premise measured under a
    different head than the one that runs is a premise about nothing.
    """
    fam = next(f for f in T.FAMILIAS if f["nombre"] == nombre)
    limpio = T.datos_semilla(fam, s)
    f1 = T.fase1_congelada(fam, limpio, T.POS_CONTAMINACION, s, T.HP_1)
    return fam, limpio, f1


# ---------------------------------------------------------------- the premise


@pytest.fixture(scope="module")
def premisa():
    """The step-0 latent geometry of the intruders against the clean points.

    Measured HERE and not read off what `intrusar` reports. `intrusar` chooses
    its radius by clearing `FACTOR_INTRUSO`, so believing its own number would
    make this test assert the calibration against itself: the licence has to be
    re-measured by a reader that did not choose the radius.
    """
    out = {}
    for nombre in FAMILIAS_PREMISA:
        for s in SEMILLAS_PREMISA:
            fam, limpio, f1 = _fase1(nombre, s)
            latente = T.latente_del_codificador(f1[0])
            D, idx, prem = T.intrusar(limpio, T.RHO_UNICO, s, latente=latente)
            assert len(idx) > 0, (nombre, s, "nothing was replaced")
            y = D["yt_tr"].numpy()
            mascara = np.zeros(len(y), dtype=bool)
            mascara[np.asarray(idx, dtype=int)] = True
            # The reader's own measurement, over the latent of the perturbed
            # train split as the frozen encoder sees it.
            Z = latente(D["Xt_tr"])
            medido = T.distancias_al_centroide(Z, y, mascara)
            out[(nombre, s)] = dict(medido=medido, declarado=prem,
                                    n=len(y), k=int(len(idx)))
    return out


def test_the_intruders_land_materially_farther_from_their_class_centroid(premisa):
    """The licence of the whole experiment. If this fails nothing else means anything.

    Per family and per seed, never on average: an average lets one geometry
    carry the claim for three. Both readings, because they are two different
    claims -- far from the class the point JOINS (which is what distorts that
    block's second-order structure) and far from EVERY class centroid (which is
    what makes it an intruder rather than a member of another class, the thing
    `reubicado` turned out to be).
    """
    assert T.FACTOR_INTRUSO > 1.0, T.FACTOR_INTRUSO
    for nombre in FAMILIAS_PREMISA:
        for s in SEMILLAS_PREMISA:
            c = premisa[(nombre, s)]["medido"]
            assert c["limpio"] > 0.0, (nombre, s, c,
                "the clean points sit on their own centroid, so the ratio "
                "below divides by nothing")
            assert c["intruso"] >= T.FACTOR_INTRUSO * c["limpio"], (
                nombre, s, c, T.FACTOR_INTRUSO,
                "the intruders are not materially farther from their class "
                "centroid than the clean points, so the intervention distorts "
                "nothing and the experiment is void")
            assert (c["cercano_intruso"]
                    >= T.FACTOR_INTRUSO * c["limpio"]), (
                nombre, s, c, T.FACTOR_INTRUSO,
                "the intruders are far from their own class but close to "
                "another one, so they are relocations and not intruders")
            # And what `intrusar` recorded agrees with what the reader measured:
            # the record is what a later session re-measures the premise from.
            d = premisa[(nombre, s)]["declarado"]
            assert abs(d["intruso"] - c["intruso"]) < 1e-4, (nombre, s, d, c)
            assert abs(d["limpio"] - c["limpio"]) < 1e-4, (nombre, s, d, c)
            assert d["radio"] in [float(r) for r in T.RADIOS_INTRUSO], (
                nombre, s, d, "the radius is off the declared grid")


# ------------------------------------------------- what the intrusion leaves alone


def test_the_intrusion_leaves_the_target_test_split_byte_identical():
    """The poison goes where the term trains; the metric stays on untouched data.

    The fixture asserts it moved the train split first, so "the test split did
    not change" cannot pass on an intrusion that changed nothing at all. And
    `yt_tr` is in the list: these are REPLACEMENTS, so the class composition of
    the train split is exactly what it was.

    And the half the clean splits cannot cover: the CLEAN POINTS' OWN
    REPRESENTATION inside the split that was touched. A crude injection along a
    FIXED direction inflates the variance along one axis, which is a feature
    rescaling wearing an intrusion's clothes -- measured, it moved AdaBN by
    +0.2139 in lunas and by -0.1657 and -0.1436 in the other two, the sign
    depending on whether that rescaling happens to suit the geometry. A positive
    result under that construction would be an artifact, so this asserts the
    clean rows are untouched bit for bit, their per-feature mean and variance
    are exactly what they were, and their step-0 latent is identical.
    """
    for nombre in FAMILIAS_PREMISA:
        _fam, limpio, f1 = _fase1(nombre, 0)
        latente = T.latente_del_codificador(f1[0])
        D, idx, _ = T.intrusar(limpio, T.RHO_UNICO, 0, latente=latente)
        assert len(idx) > 0, nombre
        assert not torch.equal(D["Xt_tr"], limpio["Xt_tr"]), (
            nombre, "the intrusion moved nothing, so the test below reads "
            "nothing")
        for clave in ("Xt_te", "yt_te", "Xt_va", "yt_va", "Xs_tr", "ys_tr",
                      "Xs_va", "ys_va", "Xs_te", "ys_te", "yt_tr"):
            assert torch.equal(D[clave], limpio[clave]), (nombre, clave)
            assert D[clave] is limpio[clave], (
                nombre, clave, "the split was copied rather than shared, so "
                "'nobody moved it' is a comparison and not a structure")

        mascara = np.zeros(len(D["yt_tr"]), dtype=bool)
        mascara[np.asarray(idx, dtype=int)] = True
        antes = limpio["Xt_tr"].numpy()[~mascara]
        ahora = D["Xt_tr"].numpy()[~mascara]
        assert np.array_equal(antes, ahora), (
            nombre, "a clean row moved, so this is not a replacement")
        assert np.array_equal(antes.mean(0), ahora.mean(0)), nombre
        assert np.array_equal(antes.var(0), ahora.var(0)), nombre
        z_antes = latente(limpio["Xt_tr"])[~mascara]
        z_ahora = latente(D["Xt_tr"])[~mascara]
        assert np.array_equal(z_antes, z_ahora), (
            nombre, float(np.abs(z_antes - z_ahora).max()),
            "the clean points' step-0 representation moved, so what gets "
            "measured afterwards is a feature rescaling and not an intrusion")


def test_a_zero_intrusion_is_a_no_op_and_so_is_its_oracle_arm():
    """Two halves, and the second is why this ceiling is a new weight mode.

    The declared ladder keeps `RHOS[0] == 0.0`, and that rung is the control the
    damage number is read against: the harness has to move NOTHING by itself,
    bit for bit and by identity.

    Then the half `pesos_oraculo` cannot satisfy. At zero intrusion the
    intruder oracle weights by ones, so its arm is the `apagado` arm exactly --
    the ceiling and the floor coincide where there is nothing to remove. The
    pseudo-label oracle does not: it zeroes every clean-but-misrouted point, so
    at zero intrusion it is already a different arm, which is why reusing it
    here would answer a different question.
    """
    assert float(T.RHOS[0]) == 0.0, (T.RHOS, "the zero rung is not declared "
                                     "first, so the control is not expressible")
    fam, limpio, f1 = _fase1("lunas", 0)
    latente = T.latente_del_codificador(f1[0])
    D, idx, prem = T.intrusar(limpio, 0.0, 0, latente=latente)
    assert len(idx) == 0, idx
    assert prem is None, prem
    assert D["Xt_tr"] is limpio["Xt_tr"], (
        "the zero rung returned a copy, so it cannot be asserted bit for bit")

    def _corre(peso):
        f1c = (copy.deepcopy(f1[0]), copy.deepcopy(f1[1]), f1[2], f1[3])
        *_, nf = T.corrida(fam, D, T.POS_CONTAMINACION, semilla=0,
                           mover_fuente=False, cuadros=T.CTRL, hp1=T.HP_1,
                           fase1=f1c, peso=peso, intrusos=idx)
        return nf

    apagado = _corre(T.PESO_APAGADO)
    techo = _corre(T.PESO_ORACULO_INTRUSOS)
    assert float(techo["acc_te"]) == float(apagado["acc_te"]), (
        techo["acc_te"], apagado["acc_te"],
        "at zero intrusion the intruder ceiling is not the floor, so it "
        "removes something nobody injected")
    assert np.array_equal(np.asarray(techo["w_usado_cu"][0]),
                          np.asarray(apagado["w_usado_cu"][0])), (
        "the intruder oracle did not weight by ones where there are no "
        "intruders")

    # And the contrast that makes this a NEW weight mode rather than a reuse.
    viejo = _corre(T.PESO_ORACULO)
    w_viejo = np.asarray(viejo["w_usado_cu"][0])
    assert 0.0 in set(w_viejo.tolist()), (
        sorted(set(w_viejo.tolist())),
        "the pseudo-label oracle zeroed nothing at zero intrusion, so the "
        "whole reason for a second oracle mode has disappeared")


def test_the_intruder_count_per_class_is_the_declared_rounding():
    """`round(rho * n_c)` IN EACH CLASS, for every declared rung.

    Per class and not over the split: the intervention is declared per class, so
    a global count would hide a draw that emptied one class and doubled another
    -- which is a different intervention with the same total.

    Plus determinism from the seed, both halves: same seed same bytes, different
    seed different bytes. An implementation that ignored the seed entirely would
    pass the first half alone.
    """
    _fam, limpio, f1 = _fase1("rollo", 0)
    latente = T.latente_del_codificador(f1[0])
    y = limpio["yt_tr"].numpy()
    clases = np.unique(y)
    cuentas = {int(c): int((y == c).sum()) for c in clases}
    assert len(set(cuentas.values())) > 1, (
        cuentas, "the classes are equally sized, so a per-class rounding "
        "cannot be told apart from a global one")
    for rho in T.RHOS:
        _, idx, _ = T.intrusar(limpio, rho, 0, latente=latente)
        idx = np.asarray(idx, dtype=int)
        assert list(idx) == sorted(idx), (rho, idx)
        assert len(set(idx.tolist())) == len(idx), rho
        for c in clases:
            esperado = int(round(rho * cuentas[int(c)]))
            visto = int((y[idx] == c).sum()) if len(idx) else 0
            assert visto == esperado, (rho, int(c), visto, esperado,
                                       cuentas[int(c)])

    a, ia, _ = T.intrusar(limpio, T.RHO_UNICO, 3, latente=latente)
    b, ib, _ = T.intrusar(limpio, T.RHO_UNICO, 3, latente=latente)
    c, ic, _ = T.intrusar(limpio, T.RHO_UNICO, 4, latente=latente)
    assert torch.equal(a["Xt_tr"], b["Xt_tr"])
    assert np.array_equal(ia, ib)
    assert not torch.equal(a["Xt_tr"], c["Xt_tr"])
    assert not np.array_equal(ia, ic)


def test_the_intruder_oracle_weight_is_exactly_the_intruder_indicator():
    """0 on the intruders, 1 elsewhere, on a fixture whose answer is known.

    Both poles present, and then the reading that says it is an INDICATOR and
    not a constant: moving the index set has to move the answer.
    """
    esperado = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    assert set(esperado.tolist()) == {0.0, 1.0}, (
        esperado, "the fixture does not exercise both poles")
    w = T.pesos_oraculo_intrusos(5, [1, 4])
    assert np.array_equal(np.asarray(w, dtype=np.float32), esperado), (w,
        esperado)
    w2 = T.pesos_oraculo_intrusos(5, [0, 2])
    assert not np.array_equal(np.asarray(w2, dtype=np.float32), esperado), w2
    assert np.array_equal(np.asarray(T.pesos_oraculo_intrusos(5, []),
                                     dtype=np.float32),
                          np.ones(5, dtype=np.float32))
    # It never reads a label, and it fails closed on an index that is not one.
    with pytest.raises(AssertionError):
        T.pesos_oraculo_intrusos(5, [5])

    # And the join into the run: the arm's recorded weight IS this indicator
    # over the points it was given, which is the only thing that says the
    # function is what the arm used.
    fam, limpio, f1 = _fase1("lunas", 0)
    latente = T.latente_del_codificador(f1[0])
    D, idx, _ = T.intrusar(limpio, T.RHO_UNICO, 0, latente=latente)
    f1c = (copy.deepcopy(f1[0]), copy.deepcopy(f1[1]), f1[2], f1[3])
    *_, nf = T.corrida(fam, D, T.POS_CONTAMINACION, semilla=0,
                       mover_fuente=False, cuadros=T.CTRL, hp1=T.HP_1,
                       fase1=f1c, peso=T.PESO_ORACULO_INTRUSOS, intrusos=idx)
    usado = np.asarray(nf["w_usado_cu"][0], dtype=np.float32)
    assert np.array_equal(
        usado, T.pesos_oraculo_intrusos(len(D["Xt_tr"]), idx)), (
        sorted(set(usado.tolist())),
        "the arm did not weight by the indicator it declares")
    # Fail closed: the ceiling cannot run without being told who the intruders
    # are, because a silent fall back to ones would make it the floor under a
    # ceiling's label.
    f1d = (copy.deepcopy(f1[0]), copy.deepcopy(f1[1]), f1[2], f1[3])
    with pytest.raises(AssertionError):
        T.corrida(fam, D, T.POS_CONTAMINACION, semilla=0, mover_fuente=False,
                  cuadros=T.CTRL, hp1=T.HP_1, fase1=f1d,
                  peso=T.PESO_ORACULO_INTRUSOS, intrusos=None)


# ------------------------------------------------- the three arms, one search


@pytest.fixture(scope="module")
def barrido(tmp_path_factory):
    """The real sweep, at the smallest honest scale, serialized as it goes.

    One family, two seeds, the zero rung and one intruded rung, one Optuna
    trial. Two seeds and three arms because a one-arm one-seed run cannot see
    the ragged-length class of defect that has bitten this notebook three
    times, and both rungs because the damage number is a difference between
    them.

    `destino` is read back from DISK rather than from the returned dict: the
    sweep serializes each cell as it completes, and a previous sweep lost ~198
    completed scenarios because it only wrote at the finish.
    """
    destino = tmp_path_factory.mktemp("intrusos") / "intrusos.jsonl"
    CONTAM = T.barrer_intrusos(
        familias=[T.FAMILIAS[0]], semillas=[0, 1], rhos=(0.0, T.RHO_UNICO),
        HP1={T.FAMILIAS[0]["nombre"]: dict(T.HP_1)}, trials2=1,
        semillas_busqueda=[100, 101], al_paso=lambda *_: None,
        destino=destino)
    return CONTAM, destino


def test_all_three_intrusion_arms_of_a_scenario_receive_the_same_hp2(barrido):
    """One search per scenario, shared by all three weights.

    Searching per arm would make it "best of `w` on" against "best of `w` off"
    -- a different claim, with the search-versus-term confound back in. Read off
    what each `corrida` actually computed with, never off what the sweep meant
    to pass, and read it from the file the sweep wrote rather than from the
    object it returned: the file is what survives an interruption.
    """
    CONTAM, destino = barrido
    nombre = T.FAMILIAS[0]["nombre"]
    registros = [json.loads(l) for l in
                 destino.read_text(encoding="utf-8").splitlines() if l.strip()]
    cabeceras = [r for r in registros if r["tipo"] == "cabecera"]
    celdas = [r for r in registros if r["tipo"] == "celda"]
    assert len(cabeceras) == 1, registros
    assert cabeceras[0]["perturbacion"] == T.MODO_INTRUSOS, cabeceras[0]
    assert len(celdas) == len(CONTAM["celdas"]) == 2, (len(celdas),
                                                       len(CONTAM["celdas"]))
    tres = tuple(b for b, _ in T.BRAZOS_POR_PESO_INTRUSOS)
    for celda in celdas:
        buscado = celda["hp2"]
        assert buscado, (celda["familia"], celda["rho"],
                         "the scenario recorded no phase-2 search")
        assert celda["busquedas"] == 1, celda["busquedas"]
        assert celda["familia"] == nombre, celda
        filas = [celda["brazos"][b] for b in tres]
        assert {len(f) for f in filas} == {2}, [len(f) for f in filas]
        for trio in zip(*filas):
            for fila in trio:
                assert fila["hp2"] == buscado, (celda["rho"], fila["hp2"],
                                                buscado)
        pesos = {b: {f["peso"] for f in celda["brazos"][b]} for b in tres}
        assert pesos == {b: {p} for b, p in T.BRAZOS_POR_PESO_INTRUSOS}, pesos
        largos = {b: len(v) for b, v in celda["brazos"].items()}
        assert set(largos.values()) == {2}, largos
        # The premise travels in the record, not only in a test: a premise
        # nobody can re-measure from disk is a premise nobody can re-measure.
        if float(celda["rho"]) > 0.0:
            prem = celda["premisa"]
            assert set(prem) == {str(s) for s in CONTAM["semillas"]}, list(prem)
            for s, c in prem.items():
                assert c["intruso"] >= T.FACTOR_INTRUSO * c["limpio"], (s, c)
        else:
            assert celda["premisa"] == {}, celda["premisa"]
            assert all(v == [] for v in celda["indices"].values()), celda


def test_no_search_can_select_the_intruder_ceiling_and_it_says_it_is_one(barrido):
    """Unreachable from any search, and labelled a ceiling in what it writes.

    Four readings, because each one alone is a different claim: the declared set
    of searchable weights excludes it, the search refuses it by name, the
    Optuna objective suggests no weight at all, and the row the arm writes
    carries the ceiling word in a field NAME -- a boolean among fifteen fields
    is read by whoever remembered to look.
    """
    assert T.PESO_ORACULO_INTRUSOS in T.PESOS, T.PESOS
    assert T.PESO_ORACULO_INTRUSOS in T.PESOS_TECHO, T.PESOS_TECHO
    assert T.PESO_ORACULO_INTRUSOS not in T.PESOS_BUSCABLES, T.PESOS_BUSCABLES
    assert not (set(T.PESOS_BUSCABLES) & set(T.PESOS_TECHO)), (
        T.PESOS_BUSCABLES, T.PESOS_TECHO)

    with pytest.raises(AssertionError):
        T.buscar_hp2(T.FAMILIAS[0], dict(T.HP_1), trials=1, semillas=[100],
                     al_paso=lambda *_: None, rho=0.0,
                     pos_filas=[T.POS_CONTAMINACION],
                     peso=T.PESO_ORACULO_INTRUSOS)

    fuente = (PAQUETE / "toy_experiment.py").read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    buscar = next(n for n in ast.walk(arbol)
                  if isinstance(n, ast.FunctionDef) and n.name == "buscar_hp2")
    sugeridos = []
    for sub in ast.walk(buscar):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr.startswith("suggest_")):
            sugeridos += [a.value for a in sub.args
                          if isinstance(a, ast.Constant)]
    assert sugeridos, "no suggestions found, so this guard reads nothing"
    for nombre in sugeridos:
        assert "peso" not in str(nombre).lower(), (nombre, sugeridos)
        assert T.PESO_ORACULO_INTRUSOS not in str(nombre).lower(), nombre

    assert T.PALABRA_TECHO in T.BRAZO_TECHO_INTRUSOS.lower(), \
        T.BRAZO_TECHO_INTRUSOS
    assert T.BRAZO_TECHO_INTRUSOS in T.BRAZOS_INTRUSOS, T.BRAZOS_INTRUSOS
    for otro in (T.BRAZO_SIN_SOP, T.BRAZO_CON_SOP):
        assert T.PALABRA_TECHO not in otro.lower(), (
            otro, "a method arm carries the ceiling word, so the word no "
            "longer marks anything")
    # Read off the row the REAL sweep wrote, not a hand-built one: the same
    # hand writing the fixture and the reader verifies both halves and never
    # the join.
    CONTAM, _destino = barrido
    for celda in CONTAM["celdas"].values():
        for fila in celda["brazos"][T.BRAZO_TECHO_INTRUSOS]:
            claves = [k for k in fila if T.PALABRA_TECHO in k.lower()]
            assert claves, (list(fila), "no field name carries the ceiling "
                            "word")
            assert all(fila[k] for k in claves), (fila, claves)
        for fila in celda["brazos"][T.BRAZO_CON_SOP]:
            claves = [k for k in fila if T.PALABRA_TECHO in k.lower()]
            assert claves and not any(fila[k] for k in claves), (fila, claves)
