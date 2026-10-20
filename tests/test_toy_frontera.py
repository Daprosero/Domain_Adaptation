"""The boundary perturbation, the oracle ceiling, and the premise of both.

The section this file guards was rebuilt after its first design asked the
weighting to do the one thing it structurally cannot. The weight is
`w_j = 1 - H2(g_j)/ln C`, which measures UNCERTAINTY and not error, so a point
relocated into another class's region is *confidently wrong* and gets a HIGH
weight. Measured on this disk at `rho=0.30`, over the three families and two
seeds, that perturbation moved the mean weight of the points it touched by
`-0.0297, +0.0753, +0.0087, +0.0150, +0.0502, +0.0213` against the clean ones:
five of six UP. Asking the weighting to discount those points is asking it to
rank by error using a signal that is blind to error, and the twelve-cell result
that came back was null for that reason and not because weighting does nothing.

So there are two perturbations now, and each one is a question the weighting can
answer on its merits:

1. **`frontera`** puts the chosen points where the frozen source head is
   UNDECIDED -- it interpolates each one toward a point the head routes to a
   different class and keeps the position along that segment whose `g` is
   flattest. There `w` genuinely collapses, so discounting is the right move and
   failing to discount should cost accuracy. Measured, same grid:
   `-0.4830, -0.4737, -0.5813, -0.6208, -0.5766, -0.6078`.
2. **`adversaria`** is the original relocation, kept under its own name because
   it is now a reportable finding -- the regime where `w` breaks -- rather than
   dead code. Its premise is the NEGATIVE one, and it is held here so the
   finding stays visible instead of being quietly deleted.

The third arm, `oraculo`, reads `y_t` and is therefore NOT a method. It is a
ceiling, and three of the tests below exist only to make it impossible to
mistake for one: no search may select it, its label carries the word that marks
it a ceiling, and its weights are exactly the correctness indicator.

Every fixture asserts its own property before the thing under test is called. A
fixture that already equals the mutant turns a passing test into a passing
nothing, and that has happened in this repository.
"""

from __future__ import annotations

import ast
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


# ---------------------------------------------------------------- the premise


@pytest.fixture(scope="module")
def premisa():
    """The step-0 mean weight of the moved points against the clean ones.

    One dict per `(family, seed, mode)`, measured on the SAME frozen phase-1
    head the run uses -- built once per `(family, seed)` and handed both to the
    perturbation and to the reading, because a premise measured under a
    different head than the one that runs is a premise about nothing.
    """
    out = {}
    for nombre in FAMILIAS_PREMISA:
        fam = next(f for f in T.FAMILIAS if f["nombre"] == nombre)
        for s in SEMILLAS_PREMISA:
            limpio = T.datos_semilla(fam, s)
            f1 = T.fase1_congelada(fam, limpio, T.POS_CONTAMINACION, s, T.HP_1)
            pesar = T.pesador_del_cabezal(f1[0], f1[1])
            for modo in T.MODOS_CONTAMINACION:
                D, idx = T.contaminar(limpio, T.RHO_UNICO, s, modo=modo,
                                      pesar=pesar)
                assert len(idx) > 0, (nombre, s, modo, "nothing was moved")
                w, _ = pesar(D["Xt_tr"])
                mask = np.zeros(len(w), dtype=bool)
                mask[np.asarray(idx, dtype=int)] = True
                out[(nombre, s, modo)] = dict(
                    perturbado=float(w[mask].mean()),
                    limpio=float(w[~mask].mean()))
    return out


def test_the_boundary_perturbation_drops_the_weight_of_the_points_it_moved(premisa):
    """The premise of the whole experiment. If this fails the experiment is void.

    `frontera` exists so that the points the term should discount are points the
    weight can SEE. Below the declared margin on every cell, not on average: an
    average would let one family carry the claim for three.
    """
    assert T.MARGEN_FRONTERA > 0.0, T.MARGEN_FRONTERA
    for nombre in FAMILIAS_PREMISA:
        for s in SEMILLAS_PREMISA:
            c = premisa[(nombre, s, T.MODO_FRONTERA)]
            delta = c["perturbado"] - c["limpio"]
            assert delta <= -T.MARGEN_FRONTERA, (
                nombre, s, c, delta, T.MARGEN_FRONTERA,
                "the boundary perturbation did not flatten g, so the weighting "
                "has nothing to discount and the experiment is void")


def test_the_adversarial_perturbation_does_not_drop_the_weight(premisa):
    """The measured finding, held so it stays visible.

    A point relocated INSIDE another class's region is confidently wrong, so
    `w` does not fall for it. That is the regime where the weight breaks, it is
    what the first design got wrong, and a test is the only thing that keeps it
    from being deleted as a null result.
    """
    for nombre in FAMILIAS_PREMISA:
        for s in SEMILLAS_PREMISA:
            c = premisa[(nombre, s, T.MODO_REUBICADO)]
            delta = c["perturbado"] - c["limpio"]
            assert delta > -T.MARGEN_FRONTERA, (
                nombre, s, c, delta, T.MARGEN_FRONTERA,
                "the relocation flattened g, which contradicts the finding "
                "this arm exists to report")


def test_the_two_perturbations_are_separated_by_the_declared_margin(premisa):
    """Neither premise is a tolerance somebody picked after seeing the numbers.

    The margin has to sit between the two measured populations with room on
    both sides, or it is fitted to the result. Measured worst cases:
    `frontera` -0.4737 and `adversaria` -0.0297.
    """
    fr = [premisa[(n, s, T.MODO_FRONTERA)] for n in FAMILIAS_PREMISA
          for s in SEMILLAS_PREMISA]
    ad = [premisa[(n, s, T.MODO_REUBICADO)] for n in FAMILIAS_PREMISA
          for s in SEMILLAS_PREMISA]
    peor_fr = max(c["perturbado"] - c["limpio"] for c in fr)
    peor_ad = min(c["perturbado"] - c["limpio"] for c in ad)
    assert peor_fr < -T.MARGEN_FRONTERA, (peor_fr, T.MARGEN_FRONTERA)
    assert peor_ad > -T.MARGEN_FRONTERA, (peor_ad, T.MARGEN_FRONTERA)
    assert peor_ad - peor_fr > 2 * T.MARGEN_FRONTERA, (
        peor_fr, peor_ad, "the two populations are not separated by twice the "
        "margin, so the margin is fitted rather than declared")


# ------------------------------------------------- what both must leave alone


def _fixture_pesada(nombre="lunas", s=0):
    fam = next(f for f in T.FAMILIAS if f["nombre"] == nombre)
    limpio = T.datos_semilla(fam, s)
    f1 = T.fase1_congelada(fam, limpio, T.POS_CONTAMINACION, s, T.HP_1)
    return limpio, T.pesador_del_cabezal(f1[0], f1[1])


def test_both_perturbations_leave_the_target_test_split_byte_identical():
    """The poison goes where the term trains; the metric stays on untouched data.

    The fixture asserts it moved the train split first, so "the test split did
    not change" cannot pass on a perturbation that changed nothing at all.
    """
    limpio, pesar = _fixture_pesada()
    for modo in T.MODOS_CONTAMINACION:
        D, idx = T.contaminar(limpio, T.RHO_UNICO, modo=modo, semilla=0,
                              pesar=pesar)
        assert not torch.equal(D["Xt_tr"], limpio["Xt_tr"]), (
            modo, "the perturbation moved nothing, so the test below reads "
            "nothing")
        for clave in ("Xt_te", "yt_te", "Xt_va", "yt_va", "Xs_tr", "ys_tr",
                      "Xs_va", "ys_va", "Xs_te", "ys_te", "yt_tr"):
            assert torch.equal(D[clave], limpio[clave]), (modo, clave)
            assert D[clave] is limpio[clave], (
                modo, clave, "the split was copied rather than shared, so "
                "'nobody moved it' is a comparison and not a structure")


def test_both_perturbations_are_deterministic_from_the_seed():
    """Same seed, same bytes; different seed, different bytes.

    Both halves, because an implementation that ignored the seed entirely would
    pass the first one alone.
    """
    limpio, pesar = _fixture_pesada()
    for modo in T.MODOS_CONTAMINACION:
        a, ia = T.contaminar(limpio, T.RHO_UNICO, 3, modo=modo, pesar=pesar)
        b, ib = T.contaminar(limpio, T.RHO_UNICO, 3, modo=modo, pesar=pesar)
        c, ic = T.contaminar(limpio, T.RHO_UNICO, 4, modo=modo, pesar=pesar)
        assert torch.equal(a["Xt_tr"], b["Xt_tr"]), modo
        assert np.array_equal(ia, ib), modo
        assert not torch.equal(a["Xt_tr"], c["Xt_tr"]), modo
        assert not np.array_equal(ia, ic), modo


def test_the_perturbed_index_count_is_the_declared_rounding():
    """`round(rho * n_train)`, for every declared rung and both modes."""
    limpio, pesar = _fixture_pesada()
    n = len(limpio["Xt_tr"])
    for modo in T.MODOS_CONTAMINACION:
        for rho in T.RHOS:
            _, idx = T.contaminar(limpio, rho, 0, modo=modo, pesar=pesar)
            assert len(idx) == int(round(rho * n)), (modo, rho, len(idx), n)
            assert len(set(int(j) for j in idx)) == len(idx), (modo, rho)
            assert list(idx) == sorted(idx), (modo, rho)


def test_a_zero_fraction_is_a_no_op_for_both_modes():
    """The rung the operator dropped, kept as the guard it was.

    The ladder was reduced to one level, so the zero rung no longer runs. What
    it proved -- that the harness moves nothing by itself -- is proved here
    instead, bit for bit and by identity.
    """
    limpio, pesar = _fixture_pesada()
    for modo in T.MODOS_CONTAMINACION:
        D, idx = T.contaminar(limpio, 0.0, 0, modo=modo, pesar=pesar)
        assert len(idx) == 0, (modo, idx)
        assert D["Xt_tr"] is limpio["Xt_tr"], (
            modo, "the zero rung returned a copy, so it cannot be asserted "
            "bit for bit")


def test_the_boundary_mode_refuses_to_run_without_the_head():
    """Fail closed. A silent fall back to the midpoint would pass every other
    test here while measuring a perturbation nobody declared."""
    limpio = T.datos_semilla(T.FAMILIAS[0], 0)
    with pytest.raises(AssertionError):
        T.contaminar(limpio, T.RHO_UNICO, 0, modo=T.MODO_FRONTERA, pesar=None)


# ----------------------------------------------------------- the oracle arm


def test_the_oracle_weight_is_exactly_the_correctness_indicator():
    """1 where the pseudo-label matches `yt_tr`, 0 where it does not.

    On a hand-built fixture whose answer is known by construction, and whose
    rows cover both outcomes AND a tie-free argmax, so neither pole is absent.
    """
    logits = torch.tensor([
        [3.0, 0.0, 0.0],    # argmax 0, label 0 -> correct   -> 1
        [0.0, 3.0, 0.0],    # argmax 1, label 2 -> wrong     -> 0
        [0.1, 0.0, 0.2],    # argmax 2, label 2 -> correct   -> 1
        [0.0, 0.5, 0.4],    # argmax 1, label 0 -> wrong     -> 0
    ])
    y = torch.tensor([0, 2, 2, 0])
    esperado = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert set(esperado.tolist()) == {0.0, 1.0}, (
        esperado, "the fixture does not exercise both poles")
    w = T.pesos_oraculo(logits, y)
    assert np.array_equal(np.asarray(w, dtype=np.float32), esperado), (w,
        esperado)
    # And it really reads the labels: permuting them has to move the answer.
    w2 = T.pesos_oraculo(logits, torch.tensor([1, 1, 1, 1]))
    assert not np.array_equal(np.asarray(w2, dtype=np.float32), esperado), w2


def test_no_hyperparameter_search_can_select_the_oracle_arm():
    """The ceiling is unreachable from the search, structurally and by refusal.

    Three readings, because each one alone is a different claim: the declared
    set of searchable weights excludes it, the search refuses it when asked for
    it by name, and the search's Optuna objective never suggests a weight at
    all.
    """
    assert T.PESO_ORACULO in T.PESOS, T.PESOS
    assert T.PESO_ORACULO not in T.PESOS_BUSCABLES, T.PESOS_BUSCABLES
    assert set(T.PESOS_BUSCABLES) < set(T.PESOS), (T.PESOS_BUSCABLES, T.PESOS)

    fam = T.FAMILIAS[0]
    with pytest.raises(AssertionError):
        T.buscar_hp2(fam, dict(T.HP_1), trials=1, semillas=[100],
                     al_paso=lambda *_: None, rho=0.0,
                     pos_filas=[T.POS_CONTAMINACION], peso=T.PESO_ORACULO)

    # Nothing in the Optuna objective names a weight: a `suggest_*` over the
    # weight axis would make the ceiling reachable by a trial.
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
        assert T.PESO_ORACULO not in str(nombre).lower(), (nombre, sugeridos)


def test_the_oracle_arm_is_labelled_a_ceiling_in_the_artifact_it_writes():
    """The word that marks it a ceiling is in the arm's name AND in a field name.

    A row that only carried a boolean would be read by whoever remembered to
    look; the word in the key is read by anybody who prints the record.
    """
    assert T.PALABRA_TECHO, T.PALABRA_TECHO
    assert T.PALABRA_TECHO in T.BRAZO_TECHO.lower(), T.BRAZO_TECHO
    # El padrón dejó de traerlo el 2026-10-19: el operador retiró el brazo de
    # la sección de contaminación. Lo que este test fija sigue vivo --- la
    # palabra está en el nombre Y en una clave, y `_fila_brazo` la escribe ---
    # y lo sigue ejercitando la sección de INTRUSOS, cuyo padrón declara dos
    # techos. Lo único que dejó de ser cierto es que ESTE padrón traiga uno.
    assert T.BRAZO_TECHO not in T.BRAZOS_CONTAMINACION, (
        T.BRAZOS_CONTAMINACION,
        "el techo volvió al padrón de contaminación: esta aserción vuelve a "
        "ser `in`")
    assert any(T.PALABRA_TECHO in b.lower() for b in T.BRAZOS_INTRUSOS), (
        T.BRAZOS_INTRUSOS,
        "ningún padrón declarado lleva un techo, así que la palabra que esto "
        "fija no marca nada que corra")
    for otro in (T.BRAZO_SIN_SOP, T.BRAZO_CON_SOP, ".predict", "AdaBN"):
        assert T.PALABRA_TECHO not in otro.lower(), (
            otro, "a method arm carries the ceiling word, so the word no "
            "longer marks anything")
    fila = T._fila_brazo(_nf_sintetico(), _cu_sintetico(), T.PESO_ORACULO)
    claves = [k for k in fila if T.PALABRA_TECHO in k.lower()]
    assert claves, (list(fila), "no field name carries the ceiling word")
    assert all(fila[k] for k in claves), (fila, claves)
    limpia = T._fila_brazo(_nf_sintetico(), _cu_sintetico(), T.PESO_DECLARADO)
    assert all(not limpia[k] for k in claves), (limpia, claves)


def _cu_sintetico():
    fila = [0, np.zeros((2, 2), dtype=np.float32), 0.5, 0.5,
            np.zeros(2, dtype=np.int64), 0.5, 0.5, 0.5]
    assert len(fila) == T.CAMPOS_CU, (len(fila), T.CAMPOS_CU)
    return [tuple(fila)]


def _nf_sintetico():
    # `tr_e`/`tr_d` DISTINTOS entre si y de todo lo demas: con dos campos
    # iguales un cruce entre ellos pasaria el test sin que nada falle.
    return dict(acc_te=0.5, hp2=dict(T.HP_2), acc_s_te=0.5, paso_fin=1,
                mejor_paso=1, vigia_fin=0.5, peso=T.PESO_DECLARADO,
                sopesado=True, tr_e=11.0, tr_d=13.0)


# ------------------------------------------------- the three arms, one search


def _celda_del_barrido(CONTAM, nombre, rho):
    """A cell of a real sweep, by the TRIPLE key, resolved through the roster.

    Not a `(nombre, rho)` lookup with the perturbation guessed: the sweep
    declares its perturbations in `perturbaciones` and names one inside every
    cell key, and reading the key THROUGH the roster is what crosses that join.
    A sweep whose roster and keys disagreed would pass a test that only indexed
    by the key it already knew.
    """
    declaradas = CONTAM["perturbaciones"]
    assert isinstance(declaradas, list), (
        declaradas, "`perturbaciones` is a LIST; a scalar cannot grow to two")
    assert len(declaradas) == 1, (
        declaradas, "one sweep runs one perturbation")
    clave = (nombre, declaradas[0], float(rho))
    assert clave in CONTAM["celdas"], (
        clave, sorted(str(k) for k in CONTAM["celdas"]),
        "the sweep's cell key does not carry the perturbation its roster "
        "declares")
    return CONTAM["celdas"][clave]


@pytest.fixture(scope="module")
def barrido_frontera():
    """The real sweep on the boundary perturbation, at the smallest honest scale.

    One family, two seeds, one rung, one Optuna trial. Two seeds and three arms
    because a one-arm one-seed run cannot see the ragged-length class of defect
    that has bitten this notebook three times.
    """
    return T.barrer_contaminacion(
        familias=[T.FAMILIAS[0]], semillas=[0, 1], rhos=T.ESCALERA_UNICA,
        modo=T.MODO_FRONTERA,
        HP1={T.FAMILIAS[0]["nombre"]: dict(T.HP_1)}, trials2=1,
        al_paso=lambda *_: None)


def test_both_arms_of_a_scenario_receive_the_same_hp2(barrido_frontera):
    """One search per scenario, shared by both weights.

    It read THREE until 2026-10-19, when the operator retired the oracle
    ceiling from this section. The claim did not change -- one search per
    scenario, shared -- only how many arms there are to share it. The
    three-and-more version still runs over a roster that has them, in
    `tests/test_toy_intrusos.py::test_all_three_intrusion_arms_of_a_scenario_receive_the_same_hp2`.

    Searching per arm would make it "best of `w` on" against "best of `w` off"
    -- a different claim, with the search-versus-term confound back in. Read off
    what each `corrida` actually computed with, never off what the sweep meant
    to pass.
    """
    CONTAM = barrido_frontera
    nombre = T.FAMILIAS[0]["nombre"]
    assert CONTAM["perturbaciones"] == [T.MODO_FRONTERA], CONTAM["perturbaciones"]
    assert CONTAM["rhos"] == [float(r) for r in T.ESCALERA_UNICA], CONTAM["rhos"]
    dos = (T.BRAZO_SIN_SOP, T.BRAZO_CON_SOP)
    assert set(dos) == {b for b, _ in T.BRAZOS_POR_PESO}, (
        dos, T.BRAZOS_POR_PESO,
        "los brazos declarados cambiaron y este test sigue nombrando dos a "
        "mano")
    for rho in CONTAM["rhos"]:
        celda = _celda_del_barrido(CONTAM, nombre, rho)
        buscado = celda["hp2"]
        assert buscado, (nombre, rho, "the scenario recorded no phase-2 search")
        assert celda["busquedas"] == 1, celda["busquedas"]
        filas = [celda["brazos"][b] for b in dos]
        assert {len(f) for f in filas} == {2}, [len(f) for f in filas]
        for par in zip(*filas):
            for fila in par:
                assert fila["hp2"] == buscado, (rho, fila["hp2"], buscado)
        # And the two really are two different arms.
        pesos = {b: {f["peso"] for f in celda["brazos"][b]} for b in dos}
        assert pesos == {T.BRAZO_SIN_SOP: {T.PESO_APAGADO},
                         T.BRAZO_CON_SOP: {T.PESO_DECLARADO}}, pesos
        largos = {b: len(v) for b, v in celda["brazos"].items()}
        assert set(largos.values()) == {2}, largos


def test_the_sweep_records_the_premise_it_rests_on(barrido_frontera):
    """The step-0 weight pair, in the artifact and not only in a test.

    A premise checked by a test nobody runs again is a premise nobody can
    re-measure from the record, so the record carries it.
    """
    CONTAM = barrido_frontera
    nombre = T.FAMILIAS[0]["nombre"]
    for rho in CONTAM["rhos"]:
        celda = _celda_del_barrido(CONTAM, nombre, rho)
        prem = celda["premisa_w"]
        assert set(prem) == set(CONTAM["semillas"]), (list(prem),
                                                      CONTAM["semillas"])
        for s, c in prem.items():
            assert c["perturbado"] is not None, (s, c)
            delta = c["perturbado"] - c["limpio"]
            assert abs(delta - c["delta"]) < 1e-6, (s, c)
            assert delta <= -T.MARGEN_FRONTERA, (s, c, delta)


def test_each_arm_weights_what_it_says_it_weights(barrido_frontera):
    """Each arm's recorded weights are what its weight claims, in the real sweep.

    The unit tests pin the functions; this crosses the join into the run, which
    is the only thing that says the function is what the arm used.

    **The ceiling's half of this left on 2026-10-19**, when the operator
    retired the oracle arm from this section: a run that does not run the arm
    records no `w_usado` for it, so the crossing cannot be made here any more.
    The crossing itself is NOT lost --- `tests/test_toy_intrusos.py` makes it
    for the intruder ceiling, reading `w_usado_cu` off a real sweep at
    `test_a_zero_intrusion_is_a_no_op_and_so_is_its_oracle_arm` and
    `test_the_intruder_count_per_class_is_the_declared_rounding` --- and what
    was dropped here is the SECTION, not the property.

    What stays is the half the retirement did not touch, and it is the half
    this section's own question rests on: the arm with the weighting off
    weights by one, and the declared weight is not binary.
    """
    CONTAM = barrido_frontera
    nombre = T.FAMILIAS[0]["nombre"]
    for rho in CONTAM["rhos"]:
        celda = _celda_del_barrido(CONTAM, nombre, rho)
        usados = celda["w_usado"]
        assert set(usados) == {T.BRAZO_SIN_SOP, T.BRAZO_CON_SOP}, list(usados)
        assert T.BRAZO_TECHO not in usados, (
            list(usados),
            "el barrido volvió a registrar el techo: hay que reponer la mitad "
            "que cruzaba el indicador contra la corrida")
        for fila in usados[T.BRAZO_SIN_SOP]:
            assert set(fila["valores"]) == {1.0}, (
                fila, "the arm with the weighting off did not weight by one")
        for fila in usados[T.BRAZO_CON_SOP]:
            assert not set(fila["valores"]) <= {0.0, 1.0}, (
                fila, "the declared weight is binary, so it is not an entropy")
