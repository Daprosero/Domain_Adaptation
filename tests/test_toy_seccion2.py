"""Section 2's ladder figures and ladder tables: the plotting half, measured.

Section 2 draws and tabulates ONE sweep over a declared ladder of noise levels.
Six artifacts live here, and each one has a way of going wrong that leaves every
number looking plausible:

1. **The input ladder is ONE figure of levels by toysets.** Nine clouds in one
   picture, and the axes have to be the axes they claim: rows the noise levels,
   columns the toysets. Transposed, the figure is still nine clouds and still
   renders.

2. **Extracting a panel out of `figura_entradas` must not change it.** The
   two-column figure is already published; a helper that drew the same points in
   a different order, or dropped the outlier ring, would still produce a figure.
   The pin here is the rendered PIXELS against a replica of the original body.

3. **The ladder frame keeps the FOUR columns of `cuadro_config`** --- embedding
   by real class, embedding by prediction, phase 1, phase 2 --- and adds none.
   A fifth column is a different figure wearing the same name, and the operator
   decided the columns do not change.

4. **Every ladder table carries the rho = 0 rung.** Without it there is nothing
   to say the noise did damage AGAINST, and a table whose smallest level is 0.10
   reads exactly like one whose smallest level is zero.

5. **The clustering table reports `tr_e` and `tr_d` as two separate columns and
   NEVER their ratio.** `tr_e` falling is classes FUSING and `tr_d` rising is
   clouds SPREADING: opposite phenomena. Measured under weighting, `tr_e` falls
   4.6% (p<0.001) while `tr_d` does not move (-0.7%, p=0.052) --- one ratio
   would have reported a single number and the reading would have been another.

6. **A table with a statistical test shows the MEDIAN and the WIN COUNT beside
   the mean.** A metric in this project once read mean -0.0254, which looks like
   an improvement, while the median was +0.0119 and only 28 of 90 pairs
   improved. It was a tail, not a trend, and the mean alone could not say so.

Every fixture asserts its own property before the thing under test is called,
and every field of a synthetic row gets a DISTINCT value: three fixtures in this
repository broke because they used 0.5 everywhere, so a crossed field was
invisible.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats as _st

from CREDA import toy_experiment as T
from CREDA import toy_plots

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE = RAIZ / "src" / "CREDA"

#: The toysets the ladder figure's columns are, in the order the module declares
#: them. Read from `FAMILIAS` rather than written out: a hand-written trio would
#: keep passing after the families changed.
TOYSETS = tuple(f["nombre"] for f in T.FAMILIAS)

#: Three of the four DECLARED rungs, the zero among them. Three because the
#: figure the operator asked for is 3x3; the zero because every table below has
#: to be able to show it.
NIVELES = (0.00, 0.10, 0.30)

#: The seeds the fixtures carry. Four, not two: a win count of "2/2" cannot tell
#: a unanimous result from a coin flip, and the median of two values is a mean.
SEMILLAS = (0, 1, 2, 3)

PERT = T.MODO_REUBICADO


def _banderas():
    return {
        "POS_FILAS": list(T.POS_FILAS),
        "SEMILLAS_REPORTE": list(SEMILLAS),
        "BALANCE_DE_MASA": "trace",
    }


# ------------------------------------------------------------------ fixtures


def _hist_fase1():
    """Phase 1's history, with the chosen epoch LAST and its accuracy highest.

    `panel_fase1` asserts both of those about what it is handed, so a fixture
    that got them wrong would fail inside the panel and say nothing about the
    ladder.
    """
    filas = [(e, 0.9 - 0.05 * e, 0.92 - 0.04 * e, 0.3 + 0.05 * e)
             for e in range(6)]
    assert all(len(f) == T.CAMPOS_HS for f in filas), T.CAMPOS_HS
    assert int(np.argmax([f[3] for f in filas])) == len(filas) - 1, filas
    return filas, filas[-1][0]


def _hist_fase2():
    """Phase 2's history, with the recorded total equal to its two raw terms.

    `panel_fase2` recomputes that sum and refuses when it does not hold, so the
    fixture has to be arithmetically consistent and not merely shaped right.
    """
    # Seis campos desde el 2026-10-11: la columna del factor del warmup se fue
    # con el warmup, y las dos de despues se corrieron una posicion.
    filas = [(p, 0.10 + 0.001 * p, 0.40 - 0.001 * p,
              (0.10 + 0.001 * p) + (0.40 - 0.001 * p),
              0.42 - 0.001 * p, 0.8 - 0.002 * p) for p in range(0, 60, 10)]
    assert all(len(f) == T.CAMPOS_HT for f in filas), T.CAMPOS_HT
    return filas


def _figura(*, F=4, n=12, rho=0.30, mejor_paso=40, semilla=7):
    """One scenario's `figura` block, hand-built, with every claim asserted.

    The class labels CYCLE rather than being drawn, so all three classes are
    present in both domains: `panel_embedding` takes a per-class centroid, and a
    missing class would put a `nan` diamond on the panel instead of raising.
    """
    rng = np.random.default_rng(semilla + int(round(100 * rho)))
    sucios = max(1, int(round(rho * n))) if rho > 0 else 0
    # NOT a prefix: `np.arange(k)` would make "paint the first k points"
    # byte-identical to selecting `idx`, so that mutation would survive.
    idx = (np.sort(rng.choice(n, size=sucios, replace=False)).astype(np.int64)
           if sucios else np.zeros(0, dtype=np.int64))
    hist_s, ep_s = _hist_fase1()
    pasos = [20 * j for j in range(F)]
    ciclo = np.array([c % T.CLASES for c in range(n)])
    f = dict(
        semilla=semilla, k=min(1, F - 1), paso=pasos[min(1, F - 1)],
        pasos=pasos, mejor_paso=mejor_paso, ep_s=ep_s,
        Pt=rng.normal(size=(F, n, 2)),
        PS_t=[rng.normal(loc=1.5, size=(n, 2)) for _ in range(F)],
        w=[rng.uniform(size=n) for _ in range(F)],
        # The routing is DELIBERATELY wrong on a known subset, so the
        # mis-routed count the figure prints is a number the test can predict.
        rut=np.array([[(c + (1 if j in idx else 0)) % T.CLASES
                       for j, c in enumerate(ciclo)] for _ in range(F)]),
        idx=idx,
        pred_s=(ciclo + 1) % T.CLASES,
        ys_tr=ciclo, yt_tr=ciclo,
        acc_va=[0.30 + 0.05 * j for j in range(F)],
        acc_te=[0.28 + 0.04 * j for j in range(F)],
        acc_s=[0.95] * F,
        pok=[0.40] * F,
        u_t=[0.20] * F,
        hist_s=hist_s, hist_t=_hist_fase2())
    assert len(f["pasos"]) == F == len(f["w"]) == len(f["Pt"])
    assert len(f["idx"]) == sucios, (len(f["idx"]), sucios)
    assert set(np.unique(f["yt_tr"])) == set(range(T.CLASES)), f["yt_tr"]
    if sucios:
        assert not np.array_equal(f["idx"], np.arange(sucios)), (
            f["idx"], "the contaminated indices are the first k, so selecting "
            "them and painting a prefix are the same mutation")
    return f


def _entradas(nombres=TOYSETS, semilla=7, n=12):
    """The persisted input clouds, keyed exactly as `guardar_entradas` writes.

    The target cloud is LONGER than nothing and exactly `n`: `figura_entradas`
    refuses a cloud whose length does not match the routing, because pairing a
    1080-point cloud with a 1079-point routing marks the wrong points and
    nothing raises.
    """
    ENTR = {}
    ciclo = np.array([c % T.CLASES for c in range(n)])
    for i, nombre in enumerate(nombres):
        rng = np.random.default_rng(1000 + i)
        ENTR[f"{nombre}|{semilla}|Xs_tr"] = rng.normal(size=(n, 3))
        ENTR[f"{nombre}|{semilla}|Xt_tr"] = rng.normal(loc=2.0, size=(n, 3))
        ENTR[f"{nombre}|{semilla}|ys_tr"] = ciclo
    return ENTR


#: The per-seed jitter every arm shares, so each paired difference is EXACTLY
#: the declared gap in every seed. Distinct per seed: a constant would make the
#: paired differences degenerate and the Wilcoxon undefined.
_JITTER = np.array([0.000, 0.003, 0.007, 0.012])
assert len(_JITTER) == len(SEMILLAS) and len(set(_JITTER)) == len(_JITTER)


#: El pase de los registros escritos a mano: CUATRO pasos por época, nunca uno.
#: Con uno, el paso y la época serían el mismo número y un test que lee la
#: columna de la época no podría distinguir cuál de los dos imprime.
_PASE_A_MANO = 4


def _filas_creda(*, acc, tr_e, tr_d, peso, pico):
    """One CREDA arm's per-seed rows, every field a DISTINCT value.

    `tr_e` and `tr_d` are deliberately far apart and move in OPPOSITE
    directions between the two weights, which is the phenomenon the ratio
    conflates: a fixture where both moved the same way could not tell a table
    that reports two columns from one that reports their quotient.
    """
    return [dict(acc_te=float(acc + j), split="test", peso=peso,
                 sopesado=bool(peso == T.PESO_DECLARADO),
                 techo_no_es_metodo=bool(peso in T.PESOS_TECHO),
                 hp2={"LR_2": 1e-3}, acc_s_te=float(0.95 - j),
                 # El techo entero y un `mejor_paso` que cae en frontera de
                 # época: desde el 2026-10-19 la fase 2 no para temprano, y la
                 # tabla deriva la época dividiendo por el pase que este techo
                 # declara. Un techo corto describe una corrida que ya no
                 # existe, y la derivación lo dice en vez de inventar un pase.
                 paso_fin=_PASE_A_MANO * T.EPOCAS_2,
                 mejor_paso=_PASE_A_MANO * (40 + int(1000 * j)),
                 vigia_fin=float(0.80 - j),
                 tr_e=float(tr_e + j), tr_d=float(tr_d - j),
                 pico_te=float(pico + j))
            for j in _JITTER]


def _celda(*, rho, con, sin, techo=None, n=12):
    """One `(familia, perturbacion, rho)` cell, with the two weighting arms.

    `con` and `sin` are `(acc, tr_e, tr_d)` triples. The ARM LABELS are the
    module's own; what identifies them to the tables is the `peso` each row
    records, which is the only field the writer promises.
    """
    brazos = {
        T.BRAZO_SIN_SOP: _filas_creda(acc=sin[0], tr_e=sin[1], tr_d=sin[2],
                                      peso=T.PESO_APAGADO, pico=sin[0] + 0.02),
        T.BRAZO_CON_SOP: _filas_creda(acc=con[0], tr_e=con[1], tr_d=con[2],
                                      peso=T.PESO_DECLARADO,
                                      pico=con[0] + 0.02),
    }
    # The two arms that carry no weight at all: one accuracy and the split, the
    # shape `barrer_contaminacion` writes for them.
    brazos[T.BRAZOS_CONTAMINACION[0]] = [
        dict(acc_te=float(0.31 + j), split="test") for j in _JITTER]
    brazos[T.BRAZOS_CONTAMINACION[1]] = [
        dict(acc_te=float(0.34 + j), split="test") for j in _JITTER]
    if techo is not None:
        brazos[T.BRAZO_TECHO] = _filas_creda(
            acc=techo, tr_e=con[1] - 1.0, tr_d=con[2] + 1.0,
            peso=T.PESO_ORACULO, pico=techo + 0.02)
        assert any(T.PALABRA_TECHO in k.lower()
                   for k in brazos[T.BRAZO_TECHO][0]), (
            "the ceiling rows lost the word that marks them")
    sucios = max(1, int(round(rho * n))) if rho > 0 else 0
    return dict(hp2={"LR_2": 1e-3}, hp2_elegido={"LR_2": 1e-3}, busquedas=1,
                brazos=brazos,
                indices={s: list(range(sucios)) for s in SEMILLAS},
                w_resumen={}, w_usado={}, premisa_w={},
                figura=_figura(rho=rho))


def _contam(*, nombres=TOYSETS, rhos=NIVELES, techo=None, perturbacion=PERT,
            padron=None, deltas=None):
    """A whole `CONTAM` over `nombres` x `rhos`, keyed by the TRIPLE.

    `deltas` maps rho -> (gap of the weighted arm on accuracy, gap on `tr_e`,
    gap on `tr_d`). The default makes the weighted arm WIN on accuracy, LOWER
    `tr_e` and RAISE `tr_d` --- the measured directions, and the only shape in
    which a table that printed the ratio could be caught.
    """
    deltas = deltas or {r: (+0.05, -0.50, +0.30) for r in rhos}
    celdas = {}
    for i, nombre in enumerate(nombres):
        for r in rhos:
            d_acc, d_e, d_d = deltas[r]
            base_acc = 0.40 + 0.01 * i - 0.10 * r
            base_e, base_d = 11.0 + i, 23.0 + i
            celdas[(nombre, perturbacion, float(r))] = _celda(
                rho=float(r),
                sin=(base_acc, base_e, base_d),
                con=(base_acc + d_acc, base_e + d_e, base_d + d_d),
                techo=None if techo is None else base_acc + techo)
    padron = list(padron if padron is not None else
                  # El padrón MÁS el techo cuando se pide, y el padrón pelado
                  # cuando no. Decía `BRAZOS_CONTAMINACION[:-1]` para quitarlo,
                  # que daba por sentado que el techo era el último del padrón;
                  # el 2026-10-19 el operador lo retiró y ese `[:-1]` empezó a
                  # quitar `CREDA con sopesado`, que es el brazo del eje de la
                  # sección. El nombre se escribe, no se recorta por posición.
                  (tuple(T.BRAZOS_CONTAMINACION) + (T.BRAZO_TECHO,)
                   if techo is not None else tuple(T.BRAZOS_CONTAMINACION)))
    return dict(rhos=[float(r) for r in rhos], semillas=list(SEMILLAS),
                pos=T.POS_CONTAMINACION, brazos=padron,
                perturbaciones=[perturbacion], hp1={}, hp2={}, celdas=celdas)


# ================================================== 1. the input-cloud ladder


def test_the_input_ladder_is_one_figure_of_levels_by_toysets():
    """Nine panels, rows the noise levels and columns the toysets, IN ORDER.

    The transposed figure draws the same nine clouds, so the only thing that
    can tell them apart is which label lands on which panel. Each panel's own
    title names its toyset and its rho, and the grid is read in row-major
    order, which is the order `add_subplot` fills.
    """
    CONTAM = _contam()
    fig = toy_plots.figura_entradas_escalera(
        CONTAM, _entradas(), _banderas(), perturbacion=PERT,
        rhos=NIVELES, nombres=TOYSETS)
    try:
        # UNA columna por toyset: la fuente y el blanco van en el MISMO
        # panel, `x` y `o`, con `COL_DURA` --- el estilo de la figura
        # `familias_duras` de la seccion 1, que es la que el operador pidio.
        # Antes eran dos paneles porque yo copiaba `figura_entradas`, que es
        # OTRA figura.
        assert len(fig.axes) == len(NIVELES) * len(TOYSETS), len(fig.axes)
        for r, rho in enumerate(NIVELES):
            for c, nombre in enumerate(TOYSETS):
                a = fig.axes[r * len(TOYSETS) + c]
                titulo = a.get_title()
                assert nombre in titulo, (r, c, titulo, "wrong column")
                assert f"{rho:.2f}" in titulo, (r, c, titulo, "wrong row")
                for otro in TOYSETS:
                    if otro != nombre and otro not in nombre:
                        assert otro not in titulo, (titulo, otro)
                # La FUENTE y el BLANCO en el mismo panel: `CLASES` colecciones
                # de cada uno, mas la de perturbados cuando los hay.
                from CREDA import toy_experiment as _T
                esperadas = 2 * _T.CLASES + (1 if rho > 0 else 0)
                assert len(a.collections) == esperadas, (
                    r, c, len(a.collections), esperadas,
                    "falta la fuente o el blanco en el panel")
    finally:
        plt.close(fig)


def test_the_extracted_input_panel_leaves_the_two_column_figure_pixel_identical(
        tmp_path):
    """The published two-column figure, rendered through the helper, unchanged.

    The replica below is the ORIGINAL body of `figura_entradas`'s panel loop,
    kept here on purpose: the claim is that extracting it changed nothing, and
    the only thing that can hold that claim is the original drawing sitting
    beside the new one. Pixels, not artist counts --- a reordered scatter draws
    the same number of collections and a different picture.

    The pixel leg alone passes over a `figura_entradas` that kept its own
    inline copy, which is the state before the extraction and exactly the
    outcome this exists to prevent: two drawing paths for one published panel,
    the second one drifting with nobody watching. So the second leg reads the
    source --- the panel loop draws through the shared helper and scatters
    nothing itself.
    """
    arbol = ast.parse((PAQUETE / "toy_plots.py").read_text(encoding="utf-8"))
    cuerpo = next((n for n in ast.walk(arbol)
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "figura_entradas"), None)
    assert cuerpo is not None, "toy_plots has no figura_entradas"
    llamadas = [sub.func.id for sub in ast.walk(cuerpo)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)]
    atributos = [sub.func.attr for sub in ast.walk(cuerpo)
                 if isinstance(sub, ast.Call)
                 and isinstance(sub.func, ast.Attribute)]
    # Once in the source, twice at run time: the two columns are one loop over
    # a pair. The count that matters is on the OTHER line --- `scatter` must be
    # gone, or the helper is an unused second path rather than the only one.
    assert llamadas.count("_panel_entradas") == 1, (
        llamadas, "figura_entradas does not draw its panels through the "
        "shared helper, so the ladder figure would need a second copy")
    assert "scatter" not in atributos, (
        atributos, "figura_entradas still scatters on its own: that is the "
        "second drawing path the extraction exists to remove")

    CONTAM = _contam(nombres=("lunas",), rhos=(0.30,))
    ENTR = _entradas(nombres=("lunas",))
    ruta = toy_plots.figura_entradas(
        "lunas", CONTAM, ENTR, _banderas(), destino=tmp_path,
        perturbacion=PERT, rho=0.30)
    producido = plt.imread(str(ruta))

    f = CONTAM["celdas"][("lunas", PERT, 0.30)]["figura"]
    k = int(f["k"])
    rut = np.asarray(f["rut"][k]).ravel()
    yt = np.asarray(f["yt_tr"]).ravel()
    mal = rut != yt
    sem = int(np.asarray(f["semilla"]))
    Xs = np.asarray(ENTR[f"lunas|{sem}|Xs_tr"], dtype=float)
    Xt = np.asarray(ENTR[f"lunas|{sem}|Xt_tr"], dtype=float)
    ys = np.asarray(ENTR[f"lunas|{sem}|ys_tr"]).ravel()

    replica = plt.figure(figsize=(13.5, 6.2))
    try:
        for col, (X, y, rot) in enumerate((
                (Xs, ys, "FUENTE" + chr(183) + " color = clase real"),
                (Xt, yt, "BLANCO" + chr(183)
                 + " círculo hueco = mal ruteado"))):
            a = replica.add_subplot(1, 2, col + 1, projection="3d")
            for c in range(T.CLASES):
                m = y == c
                a.scatter(X[m, 0], X[m, 1], X[m, 2], s=7, alpha=0.45,
                          label=f"clase {c}")
            if col == 1 and mal.any():
                a.scatter(Xt[mal, 0], Xt[mal, 1], Xt[mal, 2], s=70,
                          facecolors="none", edgecolors="k", linewidths=0.9)
            a.set_title(rot + chr(10) + f"{int(mal.sum())} de {len(mal)} "
                        f"mal ruteados ({100*mal.mean():.0f}%)"
                        if col == 1 else rot, fontsize=10, linespacing=1.4)
            a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
        replica.suptitle(
            f"lunas: el dato de entrada, semilla {sem}, "
            f"en la época que el criterio eligió (época "
            f"{int(f['pasos'][k]) // toy_plots.pasos_por_epoca_del_cuadro(f['pasos'])})",
            fontsize=12)
        replica.tight_layout()
        esperada = tmp_path / "replica.png"
        replica.savefig(esperada, dpi=110)
    finally:
        plt.close(replica)

    assert producido.shape == plt.imread(str(esperada)).shape
    assert np.array_equal(producido, plt.imread(str(esperada))), (
        "the extraction changed what `figura_entradas` draws")


# ===================================================== 2. the ladder GIF frame


def _llamadas_de_paneles(nombre_funcion):
    """The panel functions a frame builder calls, IN ORDER, read from the AST.

    The rendered figure can say how many columns there are; only the source can
    say that no second drawing path exists for one of them.
    """
    arbol = ast.parse((PAQUETE / "toy_plots.py").read_text(encoding="utf-8"))
    for node in ast.walk(arbol):
        if isinstance(node, ast.FunctionDef) and node.name == nombre_funcion:
            llamadas = [sub for sub in ast.walk(node)
                        if isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id.startswith("panel_")]
            # Sorted by POSITION, not by `ast.walk`'s order: walk is
            # breadth-first, so a call nested one level deeper than its
            # neighbours comes out of order and the comparison below would
            # read a column swap as a column change.
            return [sub.func.id for sub in
                    sorted(llamadas, key=lambda s: (s.lineno, s.col_offset))]
    raise AssertionError(f"toy_plots has no function {nombre_funcion!r}")


def test_the_ladder_frame_keeps_the_four_columns_of_the_trajectory_frame():
    """The SAME four panels, in the same order, and nothing else.

    The operator fixed the columns: embedding by real class, embedding by
    prediction, phase 1, phase 2. A fifth column is a different figure under
    the same name, and the three panels this section's other GIF added --- the
    weight pair, the accuracy curves, the contaminated scatter --- are exactly
    what would creep in.
    """
    assert (_llamadas_de_paneles("cuadro_escalera_panel")
            == _llamadas_de_paneles("cuadro_config")), (
        _llamadas_de_paneles("cuadro_escalera_panel"),
        _llamadas_de_paneles("cuadro_config"))
    # And the enumeration is not empty, which would make the equality above
    # pass while reading nothing.
    assert _llamadas_de_paneles("cuadro_config") == [
        "panel_embedding", "panel_embedding", "panel_fase1", "panel_fase2"], (
        _llamadas_de_paneles("cuadro_config"))

    CONTAM = _contam(nombres=("lunas",))
    figs = [CONTAM["celdas"][("lunas", PERT, float(r))]["figura"]
            for r in NIVELES]
    lim, cuadrada = toy_plots.caja_contaminacion(figs)
    fig = toy_plots.figura_cuadro_escalera(
        "lunas", [(PERT, r) for r in NIVELES], figs, 1, lim, cuadrada,
        T.POS_CONTAMINACION, _banderas())
    try:
        # `plt.subplots` creates the whole grid BEFORE the loop draws, and the
        # two twin axes `panel_fase1` and `panel_fase2` add land after it. So
        # the grid is the first `rows * 4`, in row-major order, and the total
        # is exactly that plus two twins per row --- stated, so a fifth column
        # cannot hide inside a ">=".
        propios = fig.axes[:len(NIVELES) * 4]
        assert len(fig.axes) == len(NIVELES) * 4 + 2 * len(NIVELES), (
            len(fig.axes), "the ladder frame does not have exactly four "
            "columns and two twins per row")
        titulos = [a.get_title() for a in propios]
        assert sum(toy_plots.COL_EMB_REAL in t for t in titulos) == len(NIVELES)
        assert sum(toy_plots.COL_EMB_PRED in t for t in titulos) == len(NIVELES)
        assert sum(t.startswith("fase 1") for t in titulos) == len(NIVELES)
        assert sum(t.startswith("fase 2") for t in titulos) == len(NIVELES)
    finally:
        plt.close(fig)


def test_the_ladder_gif_is_written_with_more_than_one_frame(tmp_path):
    """A GIF of one frame is a PNG with another suffix, and it is refused.

    The frame count is the MINIMUM over every series every row indexes, never
    the first row's: early stopping cuts each row at a different step, and
    indexing the short row with the long row's count dies with an `IndexError`
    inside it --- which has shipped in this notebook three times, and
    `nbconvert` exited 0 anyway.
    """
    CONTAM = _contam(nombres=("lunas",))
    # Ragged on purpose: the rows measure 4, 4 and 2 frames.
    corta = CONTAM["celdas"][("lunas", PERT, 0.30)]["figura"]
    for clave in ("pasos", "Pt", "PS_t", "w", "acc_va", "acc_te", "acc_s",
                  "pok", "u_t", "rut"):
        corta[clave] = corta[clave][:2]
    ruta = toy_plots.animar_escalera("lunas", CONTAM, _banderas(), tmp_path,
                                     perturbacion=PERT, al_paso=lambda *_: None)
    assert ruta.exists() and ruta.stat().st_size > 0, ruta
    from PIL import Image
    with Image.open(ruta) as gif:
        assert gif.n_frames == 2, gif.n_frames

    # One frame left in the shortest row is a refusal, not a one-frame GIF.
    for clave in ("pasos", "Pt", "PS_t", "w", "acc_va", "acc_te", "acc_s",
                  "pok", "u_t", "rut"):
        corta[clave] = corta[clave][:1]
    with pytest.raises(AssertionError):
        toy_plots.animar_escalera("lunas", CONTAM, _banderas(), tmp_path,
                                  perturbacion=PERT,
                                  al_paso=lambda *_: None)


# ========================================================= 3. the zero rung


def _todas_las_tablas(CONTAM):
    """The four Section 2 ladder tables, as the markdown each one emits."""
    b = _banderas()
    return {
        "acc": toy_plots.tabla_acc_escalera(
            "lunas", CONTAM, b, perturbacion=PERT).data,
        "sopesado": toy_plots.tabla_sopesado_escalera(
            "lunas", CONTAM, b, perturbacion=PERT).data,
        "agrupamiento": toy_plots.tabla_agrupamiento(
            "lunas", CONTAM, b, perturbacion=PERT).data,
        # UNA TABLA POR CANTIDAD desde el 2026-10-12: `tr_e` y `tr_d` ganan
        # en direcciones OPUESTAS, asi que alternarlas fila por fila obliga a
        # leer la columna `cantidad` antes de cada signo. Se concatenan para
        # que los tests que recorren "todas las tablas" las cubran a las dos.
        "sopesado_agrupamiento": (chr(10)*2).join(
            t.data for t in toy_plots.tabla_sopesado_agrupamiento(
                "lunas", CONTAM, b, perturbacion=PERT)),
    }


def test_a_declared_rung_the_sweep_skipped_is_printed_as_absent():
    """A level the DECLARED ladder names and the artifact lacks prints as absent.

    **This was `..._carries_the_zero_rung` until 2026-10-19.** It read: "`rho =
    0.00` has a row in all four, EVEN when the sweep skipped it. The zero is
    the level that says the perturbation harness moved nothing on its own.
    Without it a table has nothing to compare the damage against." The operator
    took the zero out of this section that day --- "sin la base limpia, eso ya
    es la sección 1" --- so the zero half of this claim is retired, and said
    rather than deleted.

    The MECHANISM it pinned is untouched and is what this measures now: the
    rows come from the declared ladder, not from what the artifact happens to
    carry, so a declared level nobody ran is printed as measured-absent instead
    of vanishing. That is what keeps a table of one rung from looking like a
    table of four.
    """
    # La decisión, en sus dos mitades: el cero sigue declarado en `RHOS` --- lo
    # lee el resto del árbol --- y no está en la escalera de esta sección.
    assert 0.0 in {float(r) for r in T.RHOS}, T.RHOS
    assert 0.0 not in {float(r) for r in T.ESCALERA_CONTAMINACION}, (
        T.ESCALERA_CONTAMINACION,
        "el cero volvió a esta sección: la mitad retirada de este test vuelve "
        "a ser medible y hay que reponerla")

    # **Ver la ausencia necesita una escalera declarada de DOS escalones, con
    # uno corrido y uno no.** Con la de la sección --- un escalón --- un
    # artefacto que no lo trae deja la tabla sin NINGUNA fila medida, y el
    # renderizador se niega ahí con «no hay celdas de las que leer las
    # semillas», que es otra guarda y es correcta. Así que se declara una de
    # dos: la propiedad es del renderizador y no del número de escalones que la
    # sección corra hoy.
    #
    # Se parchea `toy_plots` y no `T`: el renderizador la importó por nombre,
    # así que mover la del módulo no lo tocaría --- y ése es justo el error que
    # haría pasar este test sin medir nada.
    declarada = tuple(T.ESCALERA_CONTAMINACION) + (
        sorted({float(r) for r in T.RHOS}
               - {float(r) for r in T.ESCALERA_CONTAMINACION})[-1],)
    assert len(declarada) == 2, declarada
    previa = toy_plots.ESCALERA_CONTAMINACION
    toy_plots.ESCALERA_CONTAMINACION = declarada
    try:
        # El barrido corrió UNO de los dos declarados.
        CONTAM = _contam(nombres=("lunas",),
                         rhos=(float(T.ESCALERA_CONTAMINACION[0]),))
        corridos = {float(k[-1]) for k in CONTAM["celdas"]}
        faltan = [r for r in declarada if float(r) not in corridos]
        assert len(faltan) == 1, (declarada, corridos,
                                 "el fixture corrió los dos o ninguno, así "
                                 "que la ausencia no se puede ver")
        for cual, md in _todas_las_tablas(CONTAM).items():
            for r in declarada:
                assert f"{float(r):.2f}" in md, (
                    cual, r, "falta la fila de un escalón declarado")
            assert "no corrió" in md or "no corrio" in md, (
                cual, "the missing rung is not declared missing")

        # Y cuando el barrido corrió los dos, ninguna fila dice que falte.
        completo = _contam(nombres=("lunas",), rhos=declarada)
        for cual, md in _todas_las_tablas(completo).items():
            for r in declarada:
                assert f"{float(r):.2f}" in md, (cual, r)
            assert "no corrió" not in md and "no corrio" not in md, (
                cual, "la tabla declara ausente un nivel que sí corrió")
    finally:
        toy_plots.ESCALERA_CONTAMINACION = previa


# ================================================ 4. the clustering table


def test_the_clustering_table_reports_only_the_within_class_trace():
    """`tr_d` y SOLO `tr_d`. `tr_e` salió del todo el 2026-10-12.

    La razón es del método, no de la tabla: **CREDA no tiene ningún término
    que separe clases.** Tiene uno que alinea y compacta, así que `tr_d` --- la
    dispersión DENTRO de clase --- es su criterio y `tr_e` no lo es. Que la
    separación entre clases caiga es una consecuencia, no un incumplimiento, y
    tenerla al lado invitaba a leerla como si el método hubiera prometido algo
    ahí.

    El registro SIGUE guardando las dos (`_fila_brazo`): lo que se retiró es el
    reporte, no el dato.
    """
    b = _banderas()
    CONTAM = _contam()
    md = toy_plots.tabla_agrupamiento("lunas", CONTAM, b, perturbacion=PERT).data
    encabezado = [l for l in md.split(chr(10)) if l.startswith("| ") and "`" in l][0]
    assert "tr_d" in encabezado, encabezado
    assert "tr_e" not in encabezado, (
        "`tr_e` volvió a la tabla; salió del todo por decisión del operador",
        encabezado)
    # Y la razón va DICHA, no sólo aplicada: una cantidad que desaparece sin
    # que la tabla diga por qué se lee como un olvido.
    assert toy_plots.CAMPOS_AGRUPAMIENTO == ("tr_d",), \
        toy_plots.CAMPOS_AGRUPAMIENTO
    # El dato sigue registrado aunque no se reporte.
    celda = next(iter(CONTAM["celdas"].values()))
    from CREDA import toy_experiment as _T
    fila = celda["brazos"][_T.BRAZO_CON_SOP][0]
    assert "tr_e" in fila and "tr_d" in fila, sorted(fila)


def test_no_section_two_table_prints_the_ratio_of_the_two_traces():
    """The quotient appears nowhere: not a column, not a derived value.

    The fixture's `tr_e` and `tr_d` are chosen so their quotient has a
    distinctive four-decimal rendering that nothing else in the table produces.
    Searching for that string is what makes the absence measurable instead of
    asserted.

    The footer is allowed --- required, in fact --- to EXPLAIN the absence. What
    is banned is the quotient as a value and as a column: a reason a reader can
    check is the opposite of a number they would misread.
    """
    CONTAM = _contam(nombres=("lunas",))
    celda = CONTAM["celdas"][("lunas", PERT, 0.30)]
    con = celda["brazos"][T.BRAZO_CON_SOP]
    cociente = float(np.mean([r["tr_e"] for r in con])) / float(
        np.mean([r["tr_d"] for r in con]))
    assert 0.1 < cociente < 0.9, cociente
    tablas = _todas_las_tablas(CONTAM)
    for cual, md in tablas.items():
        assert f"{cociente:.4f}" not in md, (cual, cociente, "the ratio is printed")
        encabezados = [c for c in md.split(chr(10)) if c.strip().startswith("|")]
        for linea in encabezados:
            for palabra in ("cociente", "tr_e/tr_d", "tr_e / tr_d"):
                assert palabra not in linea, (cual, palabra, linea)
    # And the two clustering tables say WHY, with the measured evidence, so the
    # absence reads as a decision instead of an omission.
    for cual in ("agrupamiento", "sopesado_agrupamiento"):
        assert "4.6%" in tablas[cual] and "0.7%" in tablas[cual], (
            cual, "the table does not carry the measurement that decided this")
        assert "FUSIONAN" in tablas[cual] and "DESPARRAMAN" in tablas[cual], cual
    # And no new function divides one trace by the other. A value the table
    # computes and does not print today is one a later column reaches for.
    fuente = (PAQUETE / "toy_plots.py").read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    for node in ast.walk(arbol):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        texto = ast.get_source_segment(fuente, node) or ""
        assert not ("tr_e" in texto and "tr_d" in texto), texto


# ============================================ 5. median and win count


def test_a_section_two_table_with_a_test_reports_the_median_and_the_win_count():
    """Both, beside the mean, in every table that runs a statistical test.

    The mean alone already lied once here: a metric read mean -0.0254, which
    looks like an improvement, while the median was +0.0119 and only 28 of 90
    pairs improved. It was a tail, not a trend. The fixture below is built with
    that same disagreement --- the mean says one direction and the median and
    the win count say the other --- so a table that printed only the mean would
    publish the wrong reading.
    """
    # Three seeds barely lose, one wins enormously: mean positive, median
    # negative, win count 1 of 4.
    CONTAM = _contam(nombres=("lunas",), rhos=(0.30,),
                     deltas={0.30: (+0.0, -0.50, +0.30)})
    celda = CONTAM["celdas"][("lunas", PERT, 0.30)]
    sin = celda["brazos"][T.BRAZO_SIN_SOP]
    con = celda["brazos"][T.BRAZO_CON_SOP]
    cola = [+0.60, -0.02, -0.03, -0.04]
    for fila, base, extra in zip(con, sin, cola):
        fila["acc_te"] = float(base["acc_te"] + extra)
    dif = np.array([c["acc_te"] - s["acc_te"] for c, s in zip(con, sin)])
    assert dif.mean() > 0 > np.median(dif), (dif.mean(), np.median(dif))
    assert int((dif > 0).sum()) == 1, dif

    res = toy_plots.comparar_sopesado_escalera(
        "lunas", CONTAM, _banderas(), "acc_te", perturbacion=PERT)
    fila = res["general"]
    assert fila["delta"] == pytest.approx(float(dif.mean()))
    assert fila["mediana"] == pytest.approx(float(np.median(dif)))
    assert fila["gana"] == 1 and fila["n"] == len(SEMILLAS), fila

    for cual in ("sopesado", "sopesado_agrupamiento"):
        md = _todas_las_tablas(CONTAM)[cual]
        assert "mediana" in md, (cual, "no median column")
        assert f"1/{len(SEMILLAS)}" in md or f"/{len(SEMILLAS)}" in md, cual
    md = _todas_las_tablas(CONTAM)["sopesado"]
    assert f"{float(np.median(dif)):+.4f}" in md, "the median is not printed"
    assert f"1/{len(SEMILLAS)}" in md, "the win count is not printed"


# ======================================== 6. the correction scope, and the arms


def test_the_weighting_family_is_corrected_within_one_level_and_never_across_the_ladder():
    """Holm's family is a level's own pairs. The ladder is not a family.

    The existing four-arm table says this at length and this table has to say
    the same thing: correcting across the rungs corrects over a family nobody
    declared. With two arms there is exactly ONE pair per level, so Holm within
    a level is the raw p --- which is a statement the table makes, not a
    coincidence it hides.

    And the pooled table pools WITHIN the seed, giving `n` pairs and never
    `n * levels`: a seed appears at every rung, so stacking them is
    pseudo-replication. The point estimate comes out the same either way --- it
    is the denominator that changes --- which is the quietest way a number can
    be wrong.
    """
    # **La escalera de la sección pasó a UN escalón el 2026-10-19, y con uno
    # «dentro de un nivel» y «a lo largo de la escalera» son lo mismo: no hay
    # a lo largo.** Así que este test DECLARA una escalera de varios para
    # medir la propiedad, que es del renderizador y no del número de escalones
    # que la sección corra hoy. Se parchea `toy_plots` y no `T`: el
    # renderizador la importó por nombre, así que mover la del módulo no lo
    # tocaría --- y ese es justo el error que haría pasar este test sin medir.
    assert len(NIVELES) > 1, NIVELES
    previa = toy_plots.ESCALERA_CONTAMINACION
    toy_plots.ESCALERA_CONTAMINACION = tuple(NIVELES)
    try:
        CONTAM = _contam(nombres=("lunas",), rhos=NIVELES)
        res = toy_plots.comparar_sopesado_escalera(
            "lunas", CONTAM, _banderas(), "acc_te", perturbacion=PERT)
        medidos = [n for n in res["niveles"] if n["fila"] is not None]
        assert len(medidos) == len(NIVELES), [n["rho"] for n in res["niveles"]]
        for n in medidos:
            assert n["fila"]["familia_de"] == 1, n
            assert n["fila"]["holm"] == pytest.approx(n["fila"]["wilcoxon"]), n
        # The pooled row has `n` pairs, not `n * levels`.
        assert res["general"]["n"] == len(SEMILLAS), res["general"]["n"]
        assert res["n"] == len(SEMILLAS), res["n"]
        md = toy_plots.tabla_sopesado_escalera(
            "lunas", CONTAM, _banderas(), perturbacion=PERT).data
        assert "escalera" in md, (
            "the table does not say what it does NOT correct over")
        assert str(len(SEMILLAS) * len(NIVELES)) in md, (
            "the table does not name the pseudo-replicated count it refuses")
    finally:
        toy_plots.ESCALERA_CONTAMINACION = previa


def test_the_weighted_and_unweighted_arms_are_read_from_the_recorded_weight():
    """From `peso`, never from the arm's NAME, and a tie is a refusal.

    An arm that reads `y_t` is a ceiling however it is labelled, and an arm
    whose label says "con sopesado" is not weighted unless its rows say so. The
    record's `peso` field is the only thing the writing half promises, so
    renaming both arms must change nothing and crossing their weights must be
    refused.
    """
    CONTAM = _contam(nombres=("lunas",), rhos=(0.30,))
    clave = ("lunas", PERT, 0.30)
    esperado = toy_plots.comparar_sopesado_escalera(
        "lunas", CONTAM, _banderas(), "acc_te", perturbacion=PERT)

    # Rename both arms to labels that say nothing, keeping `peso` intact.
    brazos = CONTAM["celdas"][clave]["brazos"]
    brazos["brazo A"] = brazos.pop(T.BRAZO_SIN_SOP)
    brazos["brazo B"] = brazos.pop(T.BRAZO_CON_SOP)
    renombrado = toy_plots.comparar_sopesado_escalera(
        "lunas", CONTAM, _banderas(), "acc_te", perturbacion=PERT)
    assert renombrado["general"]["delta"] == pytest.approx(
        esperado["general"]["delta"]), (
        "the comparison followed the arm names instead of the recorded weight")
    assert renombrado["brazo_con"] == "brazo B", renombrado["brazo_con"]

    # Two arms recording the same weight is a refusal: nothing in the record
    # says which one the table should read.
    for fila in brazos["brazo A"]:
        fila["peso"] = T.PESO_DECLARADO
    with pytest.raises(ValueError, match="peso"):
        toy_plots.comparar_sopesado_escalera(
            "lunas", CONTAM, _banderas(), "acc_te", perturbacion=PERT)


def test_a_ceiling_arm_stays_marked_in_the_ladder_accuracy_table():
    """The oracle column carries the mark, and an unmarked record is DECLARED.

    A ceiling reads `y_t`, so a ceiling column printed like a method is a table
    that publishes an oracle as a competitor. The mark is read from the
    record's own flag --- never guessed from the arm's name --- and a record
    that marks nothing makes the table say so rather than go quiet.
    """
    CONTAM = _contam(nombres=("lunas",), rhos=(0.30,), techo=+0.20)
    md = toy_plots.tabla_acc_escalera("lunas", CONTAM, _banderas(),
                                      perturbacion=PERT).data
    assert T.BRAZO_TECHO in md, md[:400]
    encabezado = [c for c in md.split(chr(10)) if c.strip().startswith("|")][0]
    columna = [c for c in encabezado.strip("|").split("|")
               if T.BRAZO_TECHO in c]
    assert columna, encabezado
    assert toy_plots.MARCA_TECHO_PAR in columna[0], columna
    assert T.PALABRA_TECHO in md.lower()

    # A record with no ceiling at all says so instead of printing nothing.
    sin_techo = _contam(nombres=("lunas",), rhos=(0.30,))
    md2 = toy_plots.tabla_acc_escalera("lunas", sin_techo, _banderas(),
                                       perturbacion=PERT).data
    assert "ningún brazo" in md2 or "ningun brazo" in md2, md2[-600:]


def test_the_ladder_tables_count_their_seeds_from_the_data():
    """The label is the artifact's seed count, cross-checked with the flag.

    A table in this repository once printed "5 semillas" over numbers computed
    from 10, because it read the flag. The two have to agree or there is no
    table to label: preferring the data would stay silent about a short run and
    preferring the flag is the defect that already shipped.
    """
    CONTAM = _contam(nombres=("lunas",), rhos=(0.30,))
    md = toy_plots.tabla_acc_escalera("lunas", CONTAM, _banderas(),
                                       perturbacion=PERT).data
    assert f"{len(SEMILLAS)} semillas" in md, md[:300]
    mal = dict(_banderas(), SEMILLAS_REPORTE=list(range(len(SEMILLAS) + 3)))
    for llamada in (toy_plots.tabla_acc_escalera,
                    toy_plots.tabla_sopesado_escalera,
                    toy_plots.tabla_agrupamiento,
                    toy_plots.tabla_sopesado_agrupamiento):
        with pytest.raises(ValueError):
            llamada("lunas", CONTAM, mal, perturbacion=PERT)


# --------------------------------------------------------------------------
# The two-arm table's statistics: the sign, and the row that averages.
#
# `comparar_sin_contra_con` is the entry point for `sin CREDA` against
# `con CREDA` in one toyset, and nothing in this repository called it from a
# test until now. Both properties below break while every printed number stays
# plausible: an inverted sign reports the same magnitude for the other arm, and
# a pooled average row reports the same delta with a p inflated by orders of
# magnitude.
# --------------------------------------------------------------------------

#: Ten report seeds and the three declared normalizations, so the pooled
#: alternative has thirty cell-seed pairs against ten independent ones. The
#: counts have to differ by more than one for the degrees of freedom to be
#: distinguishable at all.
_SEMILLAS_PAREADAS = 10

#: `(sin CREDA, con CREDA)` per declared position. Written out rather than
#: drawn, so the degrees of freedom and the p values this test pins do not
#: depend on a generator's version.
_SERIES_DOS_BRAZOS = (
    ((0.588141, 0.607917, 0.618214, 0.570835, 0.623030,
      0.607652, 0.623489, 0.608173, 0.634860, 0.571867),
     (0.663663, 0.671965, 0.646215, 0.624038, 0.671925,
      0.612737, 0.675393, 0.636465, 0.669860, 0.599819)),
    ((0.607157, 0.622152, 0.622902, 0.573225, 0.611939,
      0.579656, 0.581877, 0.609591, 0.645659, 0.638926),
     (0.635064, 0.647944, 0.646354, 0.616080, 0.670672,
      0.620017, 0.635732, 0.655899, 0.715743, 0.638792)),
    ((0.576069, 0.634047, 0.659096, 0.627684, 0.652773,
      0.604794, 0.602219, 0.618503, 0.665240, 0.660891),
     (0.601804, 0.677669, 0.686145, 0.668085, 0.670487,
      0.643863, 0.627504, 0.657616, 0.692199, 0.684456)),
)


def _celdas_dos_brazos():
    """`CELDAS` for one toyset over the declared positions, and its flags.

    Asserts its own properties first: the weighted arm has to be ahead on
    average, or an inverted sign would be undetectable, and the three
    positions have to carry DIFFERENT series, or averaging within the seed and
    pooling the cells would give the same degrees of freedom by accident.
    """
    pos_filas = list(T.POS_FILAS)
    assert len(pos_filas) == len(_SERIES_DOS_BRAZOS), (
        pos_filas, "the declared positions and the fixture's series disagree")
    CELDAS, sin_pos, con_pos = {}, [], []
    for pos, (sin, con) in zip(pos_filas, _SERIES_DOS_BRAZOS):
        assert len(sin) == len(con) == _SEMILLAS_PAREADAS
        CELDAS[("lunas", pos, T.ETQ_UNICA)] = [
            dict(acc_predict=float(s), acc_te=float(c))
            for s, c in zip(sin, con)]
        sin_pos.append(np.array(sin, dtype=float))
        con_pos.append(np.array(con, dtype=float))

    assert float(np.mean(con_pos)) > float(np.mean(sin_pos)), (
        "`con CREDA` is not ahead on this fixture, so an inverted sign would "
        "not change which arm the table reports as winning")
    assert len({tuple(s) for s in sin_pos}) == len(sin_pos), (
        "two positions carry the identical series, so the within-seed mean "
        "and the pooled stack cannot be told apart here")

    banderas = dict(POS_FILAS=pos_filas,
                    SEMILLAS_REPORTE=list(range(_SEMILLAS_PAREADAS)),
                    BALANCE_DE_MASA="trace")
    return CELDAS, banderas, sin_pos, con_pos


def test_the_sign_of_a_toy_table_delta_is_the_first_named_arm_minus_the_second():
    """`delta = con CREDA - sin CREDA`, so `delta > 0` means CREDA wins.

    It came out inverted once in this repository: the subtracted arm sat in
    the first column, so `+0.0819` beside `.predict` read as `.predict`
    winning when it means the opposite. A table whose sign is read backwards
    is worse than no table, because it is read with confidence.

    Every row is checked, not only the one that averages, and the check is
    against the arrays -- `mean(con) - mean(sin)` -- rather than against the
    row's own `sin`/`con` columns. A mutation that swapped the subtraction AND
    the two columns would stay internally consistent and still invert what the
    header `con CREDA vs sin CREDA` claims.
    """
    CELDAS, banderas, sin_pos, con_pos = _celdas_dos_brazos()
    res = toy_plots.comparar_sin_contra_con("lunas", CELDAS, banderas)

    # The header names the first arm first, which is what the sign refers to.
    assert (res["primero"], res["segundo"]) == ("con CREDA", "sin CREDA")

    por_clave = {fila["clave"]: fila for fila in res["filas"]}
    for pos, sin, con in zip(banderas["POS_FILAS"], sin_pos, con_pos):
        fila = por_clave[pos]
        assert (fila["primero"], fila["segundo"]) == ("con CREDA", "sin CREDA")
        esperado = float(np.mean(con) - np.mean(sin))
        assert fila["delta"] == pytest.approx(esperado, abs=1e-12), pos
        assert fila["delta"] > 0, (pos, fila["delta"])
        # The win count follows the same direction: seeds where CREDA is ahead.
        assert fila["gana"] == int((con > sin).sum()), pos
        # And the two level columns are not swapped either, which is the other
        # half of the inversion that shipped.
        assert fila["con"] == pytest.approx(float(np.mean(con)), abs=1e-12)
        assert fila["sin"] == pytest.approx(float(np.mean(sin)), abs=1e-12)
        assert fila["con"] > fila["sin"], pos

    # The averaging row carries no position key, so nothing can confuse it for
    # one, and its sign points the same way.
    promedio = por_clave[None]
    assert promedio["rotulo"] == toy_plots.ROTULO_PROMEDIO
    assert promedio["delta"] > 0
    assert promedio["delta"] == pytest.approx(
        float(np.mean(con_pos) - np.mean(sin_pos)), abs=1e-12)


def test_the_averaged_row_of_a_toy_table_is_a_within_seed_mean():
    """`PROMEDIO` averages the three normalizations WITHIN each seed, giving
    ten independent pairs, and never pools the thirty cell-seed pairs.

    Seed 0 appears in all three normalizations, so the thirty pairs are not
    thirty independent observations and pooling them is pseudo-replication.

    **The delta is identical either way, and that is what makes this silent.**
    Measured on this fixture: the within-seed row and the pooled stack both
    report a delta of 0.03718757 to every digit. What changes is the
    denominator -- 9 degrees of freedom against 29 -- and with it the p, from
    8.0e-07 to 2.0e-12. Five orders of magnitude, with the table keeping its
    exact shape, its exact columns and its correct-looking effect size.

    A sibling covers the same property on the OTHER module:
    `tests/test_toy_tables.py::test_the_average_row_pairs_inside_the_seed_and_
    never_pools_the_cells` holds it for `MIL_CREDA_Benchmark.toy_tables`. That
    is a different table in a different package, and a property held on one of
    two implementations is held on one of two.
    """
    CELDAS, banderas, sin_pos, con_pos = _celdas_dos_brazos()
    res = toy_plots.comparar_sin_contra_con("lunas", CELDAS, banderas)
    promedio = [f for f in res["filas"]
                if f["rotulo"] == toy_plots.ROTULO_PROMEDIO][0]

    # The pair count is the seed count, not the cell-seed count.
    assert promedio["n"] == _SEMILLAS_PAREADAS
    assert promedio["gl"] == _SEMILLAS_PAREADAS - 1, promedio["gl"]
    assert promedio["gana"] <= _SEMILLAS_PAREADAS, (
        promedio["gana"], "more wins than there are independent pairs")

    # The within-seed vectors, recomputed: ten numbers, not a stack of thirty.
    dentro_sin = np.mean(sin_pos, axis=0)
    dentro_con = np.mean(con_pos, axis=0)
    assert dentro_sin.shape == (_SEMILLAS_PAREADAS,)
    assert promedio["delta"] == pytest.approx(
        float((dentro_con - dentro_sin).mean()), abs=1e-12)
    assert promedio["t"] == pytest.approx(
        float(_st.ttest_rel(dentro_con, dentro_sin).statistic), abs=1e-9)
    assert promedio["p"] == pytest.approx(
        float(_st.ttest_rel(dentro_con, dentro_sin).pvalue), rel=1e-9)

    # The mutation: the pooled stack. Same delta, different denominator.
    pila_sin = np.concatenate(sin_pos)
    pila_con = np.concatenate(con_pos)
    esperado_pares = _SEMILLAS_PAREADAS * len(banderas["POS_FILAS"])
    assert pila_sin.shape == (esperado_pares,)
    apilada = _st.ttest_rel(pila_con, pila_sin)

    assert promedio["delta"] == pytest.approx(
        float((pila_con - pila_sin).mean()), abs=1e-12), (
        "the pooled delta differs from the within-seed one on this fixture, "
        "so the silent half of this defect is not what it exercises")
    assert promedio["gl"] != esperado_pares - 1, (
        promedio["gl"], "the row reports the pooled degrees of freedom")
    assert promedio["p"] != pytest.approx(float(apilada.pvalue), rel=1e-6), (
        promedio["p"], float(apilada.pvalue),
        "the two readings give the same p here, so this fixture cannot show "
        "that pooling changes the conclusion")
    # The direction and the size of that inflation, pinned: the pooled p is
    # SMALLER -- pooling buys significance it did not measure.
    assert float(apilada.pvalue) < promedio["p"]
    assert promedio["p"] / float(apilada.pvalue) > 1e3, (
        promedio["p"], float(apilada.pvalue))
