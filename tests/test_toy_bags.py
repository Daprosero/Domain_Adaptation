"""The toy bags, and the claims the figure's composition rests on.

Every assertion here has a reachable red, and the ones that probe a decision
carry both poles: the decision satisfies the criterion AND the alternative
fails it. One pole alone cannot distinguish a design decision from a
measurement that would have passed anything.

The figure is drawn into `tmp_path` and never into the product tree, so a run
of this suite cannot leave a figure behind that the notebook did not draw.
"""

from __future__ import annotations

import inspect

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from CREDA.toy_experiment import DURAS  # noqa: E402
from MIL_CREDA_Benchmark import toy_bags  # noqa: E402

FAMILIAS = dict(DURAS)
#: The two families whose class cut IS one formula over the parameter, so the
#: reproduction can be replayed and compared. `lunas` is checked separately.
CON_FORMULA = ("toroide", "rollo")


@pytest.fixture(scope="module")
def construidas():
    return {nombre: toy_bags.construir(nombre, fn) for nombre, fn in DURAS}


# --------------------------------------------------------------- el parámetro


@pytest.mark.parametrize("nombre", CON_FORMULA)
def test_el_parametro_reproduce_las_clases_del_generador(nombre, construidas):
    """The scalar the bands are cut by is the one the generator cut classes by.

    100% and not «high»: the parameter is the generator's own coordinate read
    back by an exact inverse, so anything short of every point would mean the
    inverse is wrong rather than noisy.

    Red reachable: the companion test below cuts by a WRONG parameter and
    requires it to fail, so this assertion is not passing on something that
    would accept any column.
    """
    fam = construidas[nombre]
    for X, y, donde in ((fam.Xs, fam.ys, "fuente"),
                        (fam.Xt_crudo, fam.yt, "blanco sin perturbar")):
        tasa = (toy_bags.clase_reproducida(nombre, X) == y).mean()
        assert tasa == 1.0, f"{nombre} en {donde}: {tasa:.4f}"


@pytest.mark.parametrize("nombre", CON_FORMULA)
def test_un_parametro_equivocado_no_reproduce_las_clases(nombre, construidas):
    """The other pole: the same formula over the wrong column misses.

    Without this the test above could be asserting something true of every
    column of `X`, and a parameter nobody checked would be cutting bands of
    something else while the figure still showed bands.

    The mutation is asserted to have CHANGED the input before the result is
    believed: a negative test whose fixture silently stayed the same reports
    success while testing nothing.
    """
    fam = construidas[nombre]
    X_mal = fam.Xs[:, [1, 0, 2]]          # dos ejes permutados
    assert not np.allclose(X_mal, fam.Xs), "la mutación no cambió nada"

    tasa = (toy_bags.clase_reproducida(nombre, X_mal) == fam.ys).mean()
    assert tasa < 1.0, (
        f"{nombre}: permutar dos ejes dejó la reproducción en {tasa:.4f}, así "
        "que la afirmación de arriba no distingue el parámetro correcto")


def test_lunas_no_finge_una_formula_que_no_tiene(construidas):
    """`lunas` returns `None`, and what is checked instead is stated.

    Red reachable: returning any array for `lunas` fails the first assertion,
    and a parameter that did not order each sheet's arm fails the rest.
    """
    fam = construidas["lunas"]
    assert toy_bags.clase_reproducida("lunas", fam.Xs) is None

    s = toy_bags.separacion_de_lunas(fam.Xs, fam.ys)
    assert s["clase0NoPositiva"] == 1.0
    assert s["clase1NoNegativa"] == 1.0
    # La clase 2 es la lámina opuesta, corrida +1.5: su `t` medio queda por
    # encima del de las otras dos. Sin ese corrimiento las tres se mezclarían
    # sobre el mismo rango y la banda de la 2 no sería una banda de nada.
    assert s["desplazamientoDeClase2"] > 1.0


def test_una_familia_que_no_existe_se_niega_en_vez_de_devolver_algo():
    with pytest.raises(KeyError):
        toy_bags.parametro("no_existe", np.zeros((3, 3)))
    with pytest.raises(KeyError):
        toy_bags.clase_reproducida("no_existe", np.zeros((3, 3)))


# ------------------------------------------------------------- los cuantiles


def _bandas_equiancho(p, y, n):
    """The alternative the quantile decision is measured against.

    Equal-WIDTH cuts over the same parameter: the partition somebody writes
    when the point of the quantiles has not been noticed. It lives here and
    not in `src/` because nothing in the implementation may call it -- it is a
    control, not a second way of cutting bags.
    """
    b = np.zeros(len(p), dtype=int)
    for c in np.unique(y):
        m = y == c
        q = p[m]
        bordes = np.linspace(q.min(), q.max(), n + 1)[1:-1]
        b[m] = np.clip(np.searchsorted(bordes, q, side="right"), 0, n - 1)
    return b


def _dispersion_dentro_de_clase(y, b, n):
    return max(max(f) - min(f) for f in
               ([int(((y == c) & (b == k)).sum()) for k in range(n)]
                for c in np.unique(y)))


@pytest.mark.parametrize("nombre", list(FAMILIAS))
def test_los_cuantiles_dan_bolsas_del_mismo_tamano(nombre, construidas):
    """Within a class, the `n` bags carry the same count to within one.

    To within one and not exactly: a class whose size does not divide by `n`
    cannot be cut into equal integers, and that leftover is integer division
    rather than a drift. Asserting exact equality would be asserting something
    false of two of the three families.
    """
    fam = construidas[nombre]
    n = toy_bags.BAGS_PER_CLASS_TOY
    assert _dispersion_dentro_de_clase(fam.ys, fam.bs, n) <= 1
    # Y cada bolsa existe: una banda vacía no es una bolsa chica.
    for c in np.unique(fam.ys):
        for k in range(n):
            assert ((fam.ys == c) & (fam.bs == k)).sum() > 0


@pytest.mark.parametrize("nombre", list(FAMILIAS))
def test_cortes_de_ancho_igual_darian_bolsas_desparejas(nombre, construidas):
    """The other pole: the alternative partition is measurably uneven.

    This is what makes the test above a measurement instead of a restatement.
    Measured on this toyset the equal-width spread is 8 to 29 points against a
    quantile spread of 0 or 1, so the margin is not marginal.
    """
    fam = construidas[nombre]
    n = toy_bags.BAGS_PER_CLASS_TOY
    p = toy_bags.parametro(nombre, fam.Xs)
    b_mal = _bandas_equiancho(p, fam.ys, n)
    assert not np.array_equal(b_mal, fam.bs), "el control no cortó distinto"

    assert (_dispersion_dentro_de_clase(fam.ys, b_mal, n)
            > _dispersion_dentro_de_clase(fam.ys, fam.bs, n))


@pytest.mark.parametrize("nombre", list(FAMILIAS))
def test_cada_bolsa_sigue_siendo_una_rebanada_del_parametro(nombre,
                                                            construidas):
    """A bag is a BAND: within a class it occupies one interval of the parameter
    and shares that interval with no other bag.

    This is what each bag's hull claims to be enclosing. A partition that
    interleaved would still produce `n` hulls per class, and the `n` would be
    the same sack around the whole class -- it would read as a traverse of the
    manifold while being nothing of the kind.

    **This test used to make a STRONGER claim, and the stronger half was
    deliberately given up.** It read `bag k lies wholly below bag k+1`: the
    bands were not merely intervals, their NUMBERS rose with the parameter. That
    second property is what made the role split a split by geometry -- the role
    is chosen by bag index in `bags.build`, so an index that tracked position on
    the manifold handed `train` the first stretch of every class and `test` the
    last. `toy_bags.permutacion_de_bandas` renumbers the bands precisely to
    break it, and `tests/test_toy_run.py::test_every_role_receives_bands_from_
    across_the_parameter_range` is where the new numbering is asserted.

    What must NOT be given up is the interval itself, and that is what this
    test now measures: the bands are ordered by where they sit rather than by
    what they are called, and in that order each one ends before the next
    begins. A permutation applied one level too deep -- to the POINTS instead of
    to the band numbers -- would scatter every bag across its class and destroy
    exactly this, which is the pole below.
    """
    fam = construidas[nombre]
    n = toy_bags.BAGS_PER_CLASS_TOY
    p = toy_bags.parametro(nombre, fam.Xs)

    def son_rebanadas(indices):
        """Every band an interval, disjoint from the others, ordered by place."""
        for c in np.unique(fam.ys):
            tramos = sorted(
                (p[(fam.ys == c) & (indices == k)].min(),
                 p[(fam.ys == c) & (indices == k)].max())
                for k in range(n))
            for (_, hasta), (desde, _) in zip(tramos, tramos[1:]):
                if hasta > desde:
                    return False
        return True

    assert son_rebanadas(fam.bs), (
        f"{nombre}: some bag's parameter range overlaps another's, so its hull "
        "is not a slice of the manifold")

    # The pole, and it is the mutation that matters here: permuting the POINTS
    # is the "one level too deep" mistake, and it has to be caught.
    revuelto = fam.bs.copy()
    rng = np.random.default_rng(0)
    for c in np.unique(fam.ys):
        m = fam.ys == c
        revuelto[m] = rng.permutation(revuelto[m])
    assert not np.array_equal(revuelto, fam.bs), "la mutación no cambió nada"
    assert not son_rebanadas(revuelto), (
        "un orden revuelto punto por punto pasó por rebanadas, así que esta "
        "prueba no distingue una banda de un saco")


@pytest.mark.parametrize("nombre", list(FAMILIAS))
def test_los_dos_dominios_comparten_una_sola_permutacion_de_bandas(
        nombre, construidas):
    """The decision `construir` records: ONE permutation, both domains.

    Why it is the right one is argued in `construir`'s own docstring from two
    measurements -- CREDA passes the same `semilla` to `partir` for `ys` and for
    `yt`, and independent permutations would collapse the bag-to-bag
    correspondence `correspondencia_de_bolsas` exists to report. This test is
    the other half: that the code does it.

    Measured without reading any stored permutation, so this crosses the join
    rather than comparing a field with itself. The band a bag holds is recovered
    from WHERE it sits: the bands of a class are ordered by their parameter, and
    the `j`-th of them must carry the same number in both domains. The target is
    read on `Xt_crudo`, the draw its bands were cut off.
    """
    fam = construidas[nombre]
    n = toy_bags.BAGS_PER_CLASS_TOY

    def numeros_por_lugar(X, y, b):
        """Band numbers, in order of where the bands sit on the parameter."""
        p = toy_bags.parametro(nombre, X)
        por_clase = {}
        for c in np.unique(y):
            orden = sorted(range(n),
                           key=lambda k: p[(y == c) & (b == k)].mean())
            por_clase[int(c)] = orden
        return por_clase

    fuente = numeros_por_lugar(fam.Xs, fam.ys, fam.bs)
    blanco = numeros_por_lugar(fam.Xt_crudo, fam.yt, fam.bt)
    assert fuente == blanco, (
        f"{nombre}: the two domains were numbered differently, so bag k is a "
        "different stretch of the manifold in each and the bag-to-bag "
        "correspondence is gone")

    # The control: the numbering is a real permutation and not the identity, so
    # the agreement above is two permutations matching and not two untouched
    # columns.
    assert any(orden != list(range(n)) for orden in fuente.values()), (
        f"{nombre}: every class came back in its natural order, so this test "
        "would also pass with no permutation at all")


def test_la_permutacion_de_bandas_se_sortea_por_clase_como_partir():
    """Per class, independently -- `CREDA.toy_experiment.partir`'s own shape.

    `partir` runs ONE generator and calls `rng.shuffle(idx)` INSIDE its
    per-class loop, so each class draws off a later state of the same stream and
    no two classes receive the same order. Replaying one permutation across the
    classes would leave every role at the same relative positions of all three
    parameters -- scattered, but still systematic, which is a residue of the
    defect the permutation removes.

    Both poles: the classes differ from each other, and the whole draw is
    reproducible from the seed.
    """
    y = np.repeat([0, 1, 2], 300)
    n = toy_bags.BAGS_PER_CLASS_TOY
    sorteo = toy_bags.permutacion_de_bandas(y, n, semilla=0)

    assert sorted(sorteo) == [0, 1, 2], sorted(sorteo)
    for c, orden in sorteo.items():
        assert sorted(int(v) for v in orden) == list(range(n)), (c, orden)

    pares = [(a, b) for a in sorteo for b in sorteo if a < b]
    for a, b in pares:
        assert not np.array_equal(sorteo[a], sorteo[b]), (
            f"classes {a} and {b} received the SAME order, so the roles sit at "
            "the same relative positions of both parameters")

    # Reproducible from the seed, and actually a function OF the seed.
    otra_vez = toy_bags.permutacion_de_bandas(y, n, semilla=0)
    assert all(np.array_equal(sorteo[c], otra_vez[c]) for c in sorteo)
    distinta = toy_bags.permutacion_de_bandas(y, n, semilla=1)
    assert any(not np.array_equal(sorteo[c], distinta[c]) for c in sorteo), (
        "two different seeds drew the same permutation, so the draw does not "
        "depend on the seed at all")


# ------------------------------------------- el blanco, sin perturbar


@pytest.mark.parametrize("nombre", list(FAMILIAS))
def test_las_bandas_del_blanco_salen_del_sorteo_sin_perturbar(nombre,
                                                              construidas):
    """The target's bands come from the raw draw, and it matters a lot.

    The mutation is the real alternative: cut the same bands off the PERTURBED
    cloud, which is what happens when `perturbar` is applied before the
    parameter is read. `arctan2` on an already rotated and translated torus
    gives a different angle, so the bands would come from the perturbation
    instead of from the manifold.

    Both poles: what the module produced agrees with the raw draw exactly, and
    the perturbed alternative disagrees with it -- so this cannot pass by the
    two happening to coincide.
    """
    fam = construidas[nombre]
    n = toy_bags.BAGS_PER_CLASS_TOY

    # `bolsas` is still the pure band cutter; the band NUMBERS a `Familia`
    # carries are those bands renumbered by the role permutation
    # (`construir`). So the expected column is both steps, in that order --
    # comparing against `bolsas` alone would now be comparing against the
    # pre-permutation numbering and would fail on a correct draw.
    # The seed is read off `construir`'s own default, which is what the
    # `construidas` fixture draws at, so this cannot drift into asserting one
    # seed's permutation against another seed's draw.
    semilla = inspect.signature(toy_bags.construir).parameters["semilla"].default
    permutacion = toy_bags.permutacion_de_bandas(fam.ys, n, semilla=semilla)
    esperado = toy_bags.bandas_permutadas(
        toy_bags.bolsas(toy_bags.parametro(nombre, fam.Xt_crudo), fam.yt, n),
        fam.yt, permutacion)
    assert np.array_equal(fam.bt, esperado)

    assert not np.allclose(fam.Xt, fam.Xt_crudo), "la perturbación no movió nada"
    mutado = toy_bags.bandas_permutadas(
        toy_bags.bolsas(toy_bags.parametro(nombre, fam.Xt), fam.yt, n),
        fam.yt, permutacion)
    assert not np.array_equal(mutado, fam.bt), (
        f"{nombre}: cortar después de perturbar dio las MISMAS bandas, así que "
        "este test no distingue de dónde salieron")


# ------------------------------------------------- el nombre de cada bolsa


def _aclarado(color, k, n):
    """The RETIRED lightness channel, kept here as the control it now is.

    This is what `toy_bags.tinte` did before the operator looked at the drawn
    figure and rejected it: the class hue lightened toward white by the band
    index, `ACLARADO_MAXIMO = 0.72` of the way at `k = 0`. It lives in the test
    file and not in `src/` for the same reason `_bandas_equiancho` does -- it
    is the alternative the current decision is measured against, and nothing in
    the implementation may call it. Leaving it in `src/` with its own passing
    tests would have been worse than deleting it: a green test over a function
    nothing calls reads exactly like a green test over a live one.
    """
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    f = 0.72 * (1.0 - k / max(n - 1, 1))
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def test_el_rotulo_dice_dominio_clase_y_bolsa_contando_desde_uno():
    """`B` + domain + class + bag, with BOTH indices shown 1-based.

    The arrays index classes `0, 1, 2` and bags `0 .. n - 1`; the figure writes
    `c + 1` and `k + 1`. The two numbering schemes disagree on purpose and this
    is the only place they meet, so the exact names are spelled out here rather
    than recomputed from the same expression the function uses -- a test that
    rebuilt `f"B{d}{c+1}{k+1}"` would pass against any off-by-one the function
    happened to have.
    """
    assert toy_bags.etiqueta_de_bolsa("s", 0, 0) == "Bs11"
    assert toy_bags.etiqueta_de_bolsa("t", 0, 0) == "Bt11"
    assert toy_bags.etiqueta_de_bolsa("t", 2, 3) == "Bt34"
    assert toy_bags.etiqueta_de_bolsa("s", 2, 3) == "Bs34"
    assert toy_bags.etiqueta_de_bolsa("s", 1, 2) == "Bs23"
    # At twelve bags per class the bag index reaches two digits, and the
    # concatenation stops having exactly one reading. Pinned rather than left
    # implicit: `Bs112` is bag 12 of class 1 here, and a reader can only know
    # that because this toyset's class index is a single digit. The format is
    # not changed by this test; the ambiguity is recorded by it.
    assert toy_bags.etiqueta_de_bolsa("s", 0, 11) == "Bs112"
    assert toy_bags.etiqueta_de_bolsa("s", 0, 9) == "Bs110"
    # The reading this string would ALSO carry if a class index could reach two
    # digits -- class 11, bag 2 -- is the same string, which is the ambiguity.
    assert toy_bags.etiqueta_de_bolsa("s", 10, 1) == "Bs112"


def test_un_rotulo_contado_desde_cero_no_pasaria_por_estos_nombres():
    """The other pole: for the same bag, the 0-based name is a different name.

    Per bag, and NOT as two sets of names -- which is what this test asserted
    first, and it was false. The two numbering schemes produce overlapping
    strings: a 0-based implementation names class 1 bag 1 `Bs11`, which is
    exactly what this one names class 0 bag 0. Twelve of the twenty-four names
    collide that way. So «no 0-based name appears among ours» cannot be the
    check: an off-by-one would hand back names that all look valid and point at
    the wrong bag, and only comparing bag by bag catches it.

    Red reachable: dropping either `+ 1` makes some bag's name equal its
    0-based form, and the digit check fails on every class-0 or bag-0 name.
    """
    n = toy_bags.BAGS_PER_CLASS_TOY
    producidos = set()
    for dominio in ("s", "t"):
        for c in range(3):
            for k in range(n):
                nombre = toy_bags.etiqueta_de_bolsa(dominio, c, k)
                assert nombre != f"B{dominio}{c}{k}", (
                    f"la bolsa ({dominio}, {c}, {k}) se llamó igual contando "
                    "desde cero que contando desde uno")
                producidos.add(nombre)

    assert len(producidos) == 2 * 3 * n, "dos bolsas comparten nombre"
    # Ningún ÍNDICE del nombre empieza en cero. No «ningún nombre lleva un
    # cero», que es lo que esta línea pedía con cuatro bolsas por clase y es
    # falso con doce: la bolsa 10 se llama `Bs110` y ese cero es legítimo. Lo
    # que sigue valiendo es que ningún índice de los dos arranca en 0, que es
    # el off-by-one que esto cuida.
    #
    # El corte en la posición 3 sólo es correcto porque el índice de clase de
    # este toyset es de un dígito --- la misma ambigüedad que
    # `etiqueta_de_bolsa` declara en su docstring.
    #
    # Lo que se compara son los RANGOS 1-based, no la expresión que la función
    # usa: rearmar `f"B{d}{c+1}{k+1}"` acá pasaría contra cualquier off-by-one
    # que la función tuviera. Sacar un `+ 1` mueve uno de estos dos conjuntos a
    # `{0, ...}` y esto se pone rojo.
    clases_vistas = {nombre[2] for nombre in producidos}
    bolsas_vistas = {nombre[3:] for nombre in producidos}
    assert clases_vistas == {str(i) for i in range(1, 4)}, clases_vistas
    assert bolsas_vistas == {str(i) for i in range(1, n + 1)}, bolsas_vistas
    # La colisión entre las dos numeraciones, medida en vez de supuesta: es la
    # razón por la que la comparación de arriba es por bolsa.
    desde_cero = {f"B{d}{c}{k}"
                  for d in ("s", "t") for c in range(3) for k in range(n)}
    assert producidos & desde_cero, (
        "si los dos espacios de nombres no se tocaran, un conjunto alcanzaría")


def test_un_dominio_que_no_existe_no_compone_un_nombre():
    """A wrong letter is a name that reads perfectly and points elsewhere."""
    with pytest.raises(KeyError):
        toy_bags.etiqueta_de_bolsa("fuente", 0, 0)
    with pytest.raises(KeyError):
        toy_bags.etiqueta_de_bolsa("S", 0, 0)


# ------------------------------------------------------------------ el casco


def test_el_casco_encierra_los_puntos_de_su_bolsa_y_tiene_volumen():
    """Every point of the bag is inside the hull, and the hull is a solid.

    Both halves matter. A hull that left points outside would not be enclosing
    the bag it names, and a flat one would enclose nothing: `volume` is what
    separates «a body around these points» from «a sheet through them».

    Red reachable: the companion assertion feeds the hull a bag and then tests
    a point from ANOTHER bag against it, which has to come out outside.
    """
    fam = toy_bags.construir("toroide", FAMILIAS["toroide"])
    n = toy_bags.BAGS_PER_CLASS_TOY
    for c in np.unique(fam.ys):
        for k in range(n):
            P = fam.Xs[(fam.ys == c) & (fam.bs == k)]
            casco = toy_bags.casco_de_bolsa(P)
            assert casco is not None, (c, k)
            assert casco.volume > 0.0, (c, k, casco.volume)

            dentro = casco.equations[:, :3] @ P.T + casco.equations[:, 3:]
            assert dentro.max() <= 1e-9, (c, k, float(dentro.max()))

    # El otro polo: un punto de otra bolsa de la misma clase no está adentro,
    # así que «todo cae adentro» no es algo que esta prueba acepte de cualquier
    # nube.
    P0 = fam.Xs[(fam.ys == 0) & (fam.bs == 0)]
    P3 = fam.Xs[(fam.ys == 0) & (fam.bs == 3)]
    casco0 = toy_bags.casco_de_bolsa(P0)
    fuera = casco0.equations[:, :3] @ P3.T + casco0.equations[:, 3:]
    assert fuera.max() > 1e-9, (
        "la bolsa 3 entera cayó dentro del casco de la bolsa 0: el casco no "
        "está encerrando una bolsa, está encerrando la clase")


def test_una_bolsa_sin_volumen_no_dibuja_casco_en_vez_de_reventar():
    """Too few points, or points on a plane, answer `None` and draw nothing.

    Two different ways a bag encloses no volume, and the figure survives both:
    losing twenty-three readable hulls because the twenty-fourth came out flat
    would be a worse answer than a missing outline.

    Red reachable: returning a hull for any of these, or raising instead of
    returning, fails here -- and the third case proves the function is not just
    returning `None` for everything.
    """
    import matplotlib.pyplot as plt

    solido = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert len(solido) == toy_bags.MINIMO_PARA_CASCO
    assert toy_bags.casco_de_bolsa(solido) is not None, (
        "con el mínimo de puntos y volumen, tiene que haber casco")

    assert toy_bags.casco_de_bolsa(solido[:3]) is None, "tres puntos, un plano"
    assert toy_bags.casco_de_bolsa(np.zeros((0, 3))) is None

    plano = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [.5, .5, 0]])
    assert len(plano) > toy_bags.MINIMO_PARA_CASCO
    assert toy_bags.casco_de_bolsa(plano) is None, (
        "cinco puntos coplanares no encierran volumen")

    # Y dibujar tampoco revienta: no agrega nada y sigue.
    fig = plt.figure()
    try:
        eje = fig.add_subplot(projection="3d")
        assert toy_bags._dibujar_casco(eje, plano, "#1f77b4") is None
        assert not eje.collections
        assert toy_bags._dibujar_casco(eje, solido, "#1f77b4") is not None
        assert len(eje.collections) == 1
    finally:
        plt.close(fig)


# -------------------------------------------------- el techo y su nivel de azar


@pytest.mark.parametrize("nombre", list(FAMILIAS))
def test_la_correspondencia_existe_en_la_variedad(nombre, construidas):
    """Bag `k` of class `c` is the `k`-th quantile in BOTH domains, so it is
    recoverable on the draw -- and that is the ceiling an alignment is read
    against.

    Chance is reported beside it because the rate means nothing alone: read
    against 1.0 instead of against what the construction offers, a method at
    most of what is attainable reports as a failure.

    **The ceiling is no longer 1.0, and that is what twelve bands cost.** At
    four bags per class it was exactly 1.0 on all three families: the bands were
    wide, and a target band's centroid was always nearest its own source band's.
    At twelve the bands are three times narrower and neighbouring ones start
    being confusable on independent draws. Measured over the thirteen seeds
    either scale uses: `lunas` runs 0.6667 to 1.0, `toroide` 0.8333 to 1.0,
    `rollo` 0.8611 to 1.0. So the local correspondence is now read against
    something imperfect and seed-dependent, which is a property of the cut and
    not of any method.

    So what is asserted is that the construction still puts MUCH MORE of the
    correspondence there than either chance or proximity in the observed domain
    recovers, and NOT that it puts all of it there. A pinned 1.0 would have been
    a claim the cut makes false; a pinned 0.8611 would be a snapshot of one seed
    of prior work's generator.

    **The absolute floor of 0.5 was dropped when the cut went from 12 to 27
    bands per class, because the cut makes it false.** Measured over the
    thirteen seeds either scale draws, at 27 bands: `lunas` runs 0.3210 to
    0.7778, `toroide` 0.3704 to 0.6049, `rollo` 0.4198 to 0.6296, against a
    chance of 0.0370. At 12 bands the same measurement ran 0.6667 to 1.0, so
    the ceiling the local correspondence of Eqs. (30)-(31) is read against
    FELL with the finer cut -- that is a cost of the resolution the cut bought
    and a property of the construction, not of any method. What survives as a
    claim is the two RELATIVE poles below, which hold on every one of those
    thirteen seeds; the seed this fixture draws is only one of them, so an
    absolute threshold here would be a snapshot that passes while the claim is
    false elsewhere.
    """
    c = toy_bags.correspondencia_de_bolsas(construidas[nombre])
    # El azar sale de las bandas DIBUJADAS, no de la constante: a veintisiete
    # bolsas por clase es 1/27 y no 1/12, y un número escrito acá envejecería
    # con el corte sin que nada fallara.
    bandas = len(set(construidas[nombre].bs.tolist()))
    assert bandas == 27, bandas
    assert c["azar"] == pytest.approx(1 / bandas)
    assert c["bolsasComparadas"] == 3 * bandas
    assert c["enLaVariedad"] > 4 * c["azar"], (
        f"{nombre}: {c['enLaVariedad']:.4f} contra un azar de {c['azar']:.4f}")
    # Y el polo que hace de esto una medición y no un piso cómodo: la cercanía
    # de centroides en el dominio OBSERVADO sí queda cerca del azar, así que el
    # umbral de arriba distingue las dos lecturas en vez de pasar con cualquiera.
    assert c["enElDominioObservado"] <= 4 * c["azar"], (
        f"{nombre}: el piso observado dio {c['enElDominioObservado']:.4f}, que "
        f"ya pasaría el umbral del techo; el umbral no distingue nada")


@pytest.mark.parametrize("nombre", list(FAMILIAS))
def test_la_proximidad_en_el_dominio_observado_no_la_recupera(nombre,
                                                              construidas):
    """And the floor: on the domains a method SEES, proximity alone barely beats
    guessing.

    This is the half that makes the ceiling worth reporting. If proximity in
    the observed space already recovered the correspondence there would be
    nothing for an alignment to do, and a result at 100% would be measuring the
    construction rather than the method.

    Asserted as a gap rather than as a constant: the exact rate is a property
    of `perturbar`'s rotation and would turn this into a snapshot test of prior
    work. What is claimed is that the observed-space rate is far below the
    ceiling, which is the thing a reader needs.
    """
    c = toy_bags.correspondencia_de_bolsas(construidas[nombre])
    assert c["enElDominioObservado"] < c["enLaVariedad"] - 0.4, (
        f"{nombre}: la proximidad observada dio {c['enElDominioObservado']:.4f} "
        f"contra un techo de {c['enLaVariedad']:.4f}")
    assert c["enElDominioObservado"] >= c["azar"] - 1e-9


def test_el_azar_sigue_al_corte_que_la_llamada_hizo(construidas):
    """`azar` is `1 / n` of THIS call, never the module default's.

    It used to read `AZAR_DE_CORRESPONDENCIA`, which is `1 /
    BAGS_PER_CLASS_TOY`. At the default the two agree, so the defect was
    invisible: a caller passing a different `n` was handed the chance level of a
    cut it did not make, beside a rate measured on the cut it did. Both poles --
    the default agrees, and a non-default does not.
    """
    fam = construidas["toroide"]
    por_defecto = toy_bags.correspondencia_de_bolsas(fam)
    assert por_defecto["azar"] == pytest.approx(
        toy_bags.AZAR_DE_CORRESPONDENCIA)
    assert por_defecto["bolsasComparadas"] == 3 * toy_bags.BAGS_PER_CLASS_TOY

    # Un corte distinto del declarado: las bandas que el `Familia` trae son 27,
    # así que pedir 4 compara las cuatro primeras y el azar tiene que ser 1/4.
    otro = toy_bags.correspondencia_de_bolsas(fam, 4)
    assert otro["azar"] == pytest.approx(1 / 4)
    assert otro["azar"] != por_defecto["azar"], (
        "los dos cortes dieron el mismo azar; este test no distingue nada")
    assert otro["bolsasComparadas"] == 3 * 4


# ------------------------------------------------------------- los ejes


def test_los_ejes_se_miden_y_salen_no_compartidos(construidas):
    """The collapse of shared decoration is a claim, so it gets measured.

    The three families live on three scales, so the measurement must say «not
    shared» -- and a panel grid drawn as if they shared one would be a figure
    that lies in the direction of tidiness.

    The other pole: fed three copies of ONE family the same measurement has to
    say «shared», or it is a function that always refuses and the figure's
    tidiness is decided by nothing.
    """
    tres = [construidas[n] for n in ("lunas", "toroide", "rollo")]
    medido = toy_bags._ejes_compartidos(tres)
    assert medido["compartidos"] is False
    assert medido["discrepanciaRelativa"] > medido["tolerancia"]

    una = [construidas["toroide"]] * 3
    igual = toy_bags._ejes_compartidos(una)
    assert igual["compartidos"] is True
    assert igual["discrepanciaRelativa"] <= igual["tolerancia"]


# ---------------------------------------------------------------- la figura


def test_la_figura_escribe_el_archivo_y_sus_hechos_son_los_que_dibujo(tmp_path):
    """The file is written AND the returned facts match what was drawn.

    A cell that only printed a path would have reported a filename, and facts
    recomputed by the caller could disagree with the picture above them. So the
    figure hands back both, and this crosses the join: the readings that come
    back are compared against readings measured independently from the same
    construction, not against themselves.
    """
    import matplotlib.pyplot as plt

    r = toy_bags.figura_familias(tmp_path)
    try:
        assert r.ruta.is_file()
        assert r.ruta.stat().st_size > 0
        assert r.ruta.parent == tmp_path

        assert [l["familia"] for l in r.lecturas] == [n for n, _ in DURAS]
        assert r.ejes["compartidos"] is False

        # El cruce: las lecturas se vuelven a medir desde la construcción, no
        # se comparan con ellas mismas.
        for devuelta in r.lecturas:
            fam = toy_bags.construir(devuelta["familia"],
                                     FAMILIAS[devuelta["familia"]])
            otra = toy_bags.lectura(fam)
            assert devuelta["direccionesEfectivas"] == pytest.approx(
                otra["direccionesEfectivas"])
            assert devuelta["separacionEntreClases"] == pytest.approx(
                otra["separacionEntreClases"])
            assert devuelta["correspondencia"] == otra["correspondencia"]
            assert devuelta["puntosPorBolsa"] == otra["puntosPorBolsa"]

        # Y el panel dibujó las tres familias con sus doce bolsas cada una.
        assert len(r.fig.axes) == len(DURAS)
    finally:
        plt.close(r.fig)


# ----------------------------------------------- los cascos sobre la figura


def _cascos(eje):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    return [c for c in eje.collections if isinstance(c, Poly3DCollection)]


@pytest.fixture(scope="module")
def dibujada(tmp_path_factory):
    """The real figure, drawn once for every assertion about what it contains.

    Module-scoped because drawing it is the expensive part of this file and
    nothing below mutates it; each test reads a different property of the same
    picture.
    """
    import matplotlib.pyplot as plt

    r = toy_bags.figura_familias(tmp_path_factory.mktemp("figura"))
    yield r
    plt.close(r.fig)


def test_cada_bolsa_de_cada_dominio_recibe_su_casco_y_su_rotulo(dibujada):
    """`3 * n * 2` hulls and the same number of labels, on every panel.

    Three classes, `n` bags each, two domains -- and the source bag and the
    target bag of the same `(c, k)` are two different bags of points, so each
    gets its own enclosure. A figure drawing one hull per `(c, k)` would be
    drawing `3 * n` sacks each holding two clouds that the whole experiment is
    about telling apart.

    **`3 * n * 2` is 162 at the declared twenty-seven bags per class, and not
    81.**
    The doubling is the two domains, and it is the factor a reader counting
    hulls per class forgets: a panel draws the SOURCE and the TARGET of every
    bag. Both counts are written out here because the number is what the
    crowding measurement is read against.

    Red reachable: dropping either domain from the loop halves the count, and
    enclosing the class instead of the bag divides it by `n`.
    """
    n = toy_bags.BAGS_PER_CLASS_TOY
    esperados = 3 * n * 2
    assert esperados == 162
    assert esperados == 2 * (3 * n), "la figura dibuja los dos dominios"

    for eje, medida in zip(dibujada.fig.axes, dibujada.rotulos):
        assert len(_cascos(eje)) == esperados
        assert len(eje.texts) == esperados
        assert medida["rotulos"] == esperados
        # Y son los nombres de las setenta y dos bolsas, sin repetir ninguno.
        assert {t.get_text() for t in eje.texts} == {
            toy_bags.etiqueta_de_bolsa(d, c, k)
            for d in ("s", "t") for c in range(3) for k in range(n)}


def test_el_casco_va_relleno_y_sin_aristas(dibujada):
    """Filled at `ALFA_DEL_CASCO`, and with no edges at all.

    This is the property that was MEASURED to matter: the hull of a bag of
    ~50 points in three dimensions -- ~150 when the cut was four bags per
    class -- comes back as hundreds of triangles, and
    drawing the edges of that triangulation buries both the points inside it
    and the label on top of it. So the body is filled and the edges are not
    drawn -- and «not drawn» means no edge colours at all, not edges painted
    the same colour as the face, which would still ink every simplex boundary.

    Red reachable: an `edgecolors=` that paints anything gives a non-empty edge
    colour array, and a hull drawn opaque fails the alpha.
    """
    for eje in dibujada.fig.axes:
        for cuerpo in _cascos(eje):
            caras = cuerpo.get_facecolor()
            assert len(caras) > 0
            assert np.allclose(caras[:, 3], toy_bags.ALFA_DEL_CASCO)
            assert 0.0 < toy_bags.ALFA_DEL_CASCO < 1.0, (
                "un casco opaco deja de ser un envoltorio y tapa su bolsa")
            assert len(cuerpo.get_edgecolor()) == 0, (
                f"el casco trajo {len(cuerpo.get_edgecolor())} colores de "
                "arista: la triangulación se va a dibujar encima de los puntos")


def test_el_tono_del_casco_es_el_de_la_clase_a_fuerza_plena(dibujada):
    """Each hull carries its class hue untouched -- not a tint of it.

    Both poles, and the second one is the retired channel itself. The drawn
    colours are compared against `COL_DURA` exactly, and then against what the
    lightness ramp would have produced for every band but the last: not one
    drawn hull may be any of those, or the hue would still be carrying the bag
    and the class at once, which is what the operator rejected.
    """
    from CREDA.toy_experiment import COL_DURA
    from matplotlib.colors import to_rgb

    n = toy_bags.BAGS_PER_CLASS_TOY
    puros = {c: np.array(to_rgb(color)) for c, color in COL_DURA.items()}
    aclarados = [np.array(_aclarado(color, k, n))
                 for color in COL_DURA.values() for k in range(n - 1)]
    assert all(not np.allclose(a, puros[c]) for a in aclarados
               for c in puros), "el control no aclara nada"

    for eje in dibujada.fig.axes:
        cuenta = {c: 0 for c in puros}
        for cuerpo in _cascos(eje):
            rgb = cuerpo.get_facecolor()[0, :3]
            cual = [c for c, puro in puros.items() if np.allclose(rgb, puro)]
            assert len(cual) == 1, (
                f"el casco salió en {rgb}, que no es el tono de ninguna clase")
            cuenta[cual[0]] += 1
            assert not any(np.allclose(rgb, a) for a in aclarados), (
                f"el casco salió en {rgb}, que es un tono aclarado")
        assert set(cuenta.values()) == {2 * n}, cuenta


def test_los_puntos_tambien_van_en_el_tono_pleno_de_su_clase(dibujada):
    """And the points, which is where the lightness channel used to live.

    Retiring the channel in the hull and leaving it in the scatter would be the
    worst of the two: the enclosure would say one thing about a bag and its own
    points another.
    """
    from CREDA.toy_experiment import COL_DURA
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Path3DCollection

    n = toy_bags.BAGS_PER_CLASS_TOY
    puros = [np.array(to_rgb(color)) for color in COL_DURA.values()]
    aclarados = [np.array(_aclarado(color, k, n))
                 for color in COL_DURA.values() for k in range(n - 1)]

    for eje in dibujada.fig.axes:
        nubes = [c for c in eje.collections
                 if isinstance(c, Path3DCollection)]
        assert len(nubes) == 3 * n * 2
        for nube in nubes:
            rgb = nube.get_facecolor()[0, :3]
            assert any(np.allclose(rgb, p) for p in puros), rgb
            assert not any(np.allclose(rgb, a) for a in aclarados), rgb


# --------------------------------------------------- el encimado de rótulos


def test_la_medicion_de_rotulos_distingue_encimados_de_separados():
    """The crowding measurement sees an overlap, and sees its absence.

    A measurement that reported «none» whatever it was handed would be the
    quietest possible failure here: the figure would be declared readable by a
    function that cannot say otherwise. So it is driven with two labels on the
    same point and with two at opposite corners, and it has to answer
    differently.

    This is not hypothetical. Asking the artists for their own
    `get_window_extent` after the draw reported no overlap anywhere, because a
    `Text3D` puts its data coordinates back when its draw ends -- the first
    version of this measurement said `lunas` was clean while the drawn panel
    has four pairs sitting on top of each other.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    try:
        eje = fig.add_subplot(projection="3d")
        eje.set_xlim(0, 1)
        eje.set_ylim(0, 1)
        eje.set_zlim(0, 1)

        juntos = [(nombre, (0.5, 0.5, 0.5),
                   eje.text(0.5, 0.5, 0.5, nombre, fontsize=8.5,
                            ha="center", va="center"))
                  for nombre in ("Bs11", "Bt11")]
        medido = toy_bags._separacion_de_rotulos(fig, eje, juntos)
        assert medido["pares"] == 1
        assert medido["encimados"] == [["Bs11", "Bt11"]]
        assert medido["separacionMinima"] is None

        separados = [("Bs11", (0.02, 0.02, 0.02), juntos[0][2]),
                     ("Bt11", (0.98, 0.98, 0.98), juntos[1][2])]
        otro = toy_bags._separacion_de_rotulos(fig, eje, separados)
        assert otro["encimados"] == []
        assert otro["separacionMinima"] > 0.0
        assert otro["parMasCercano"] == ["Bs11", "Bt11"]
    finally:
        plt.close(fig)


def test_la_figura_informa_donde_sus_rotulos_se_enciman(dibujada):
    """The crowding the figure has is reported, family by family.

    The claim the figure rests on is that every bag can be read by name, and
    where two names land on top of each other that claim is weaker there. It is
    measured by the drawing code and handed back rather than left to the eye:
    a limit stated in pairs is a limit; a limit somebody notices later is a
    defect.

    What is asserted is the shape and the arithmetic, never the exact pairs --
    those are a property of three generators and three view angles, and
    freezing them here would turn this into a snapshot test of prior work that
    reddens on any change of view. The measured pairs travel in the notebook,
    where a reader can see them.
    """
    assert [m["familia"] for m in dibujada.rotulos] == [n for n, _ in DURAS]
    for m in dibujada.rotulos:
        assert m["pares"] == m["rotulos"] * (m["rotulos"] - 1) // 2
        assert len(m["encimados"]) < m["pares"], (
            f"{m['familia']}: todos los pares se encimaron, que es lo que "
            "informaría una medición rota")
        for a, b in m["encimados"]:
            assert a != b
        if len(m["encimados"]) < m["pares"]:
            assert m["separacionMinima"] is not None
            assert m["separacionMinima"] >= 0.0
            assert m["parMasCercano"] is not None


def test_a_veintisiete_bolsas_por_clase_el_canal_del_rotulo_dejo_de_leerse(
        dibujada):
    """The measurement that says the names no longer read, asserted as such.

    `BAGS_PER_CLASS_TOY` went 4 -> 12 -> 27 and the label channel went with it.
    At four bags per class the measurement read 24 names per panel with 4
    overlapping pairs in `lunas` and none in `toroide` or `rollo`. At twelve it
    read 72 names with most of them sitting on top of something. At the declared
    27, measured on the drawn panels: 162 names per panel, 512 overlapping pairs
    in `lunas` with 148 of 162 names involved, 487 in `toroide` with all 162
    involved, 319 in `rollo` with 160 of 162, and the smallest gap between two
    names that do not touch down to 0.0034 pt. That is the same failure the
    lightness channel was retired for, in another channel, and it is asserted
    here rather than only described -- a figure whose stated limit is not
    measured is a figure whose limit nobody re-checks.

    What is asserted is the DIRECTION and the scale of the failure, never exact
    pairs: those are a property of three generators and three view angles. The
    thresholds below are loose for that reason and still discriminate -- at four
    bags per class two of the three panels had zero overlaps, so any of these
    assertions would have been red.
    """
    assert toy_bags.BAGS_PER_CLASS_TOY == 27
    for m in dibujada.rotulos:
        assert m["rotulos"] == 162, (m["familia"], m["rotulos"])
        involucrados = {nombre for par in m["encimados"] for nombre in par}
        assert len(m["encimados"]) >= 100, (m["familia"], len(m["encimados"]))
        assert len(involucrados) >= m["rotulos"] * 3 // 4, (
            f"{m['familia']}: {len(involucrados)} de {m['rotulos']} nombres "
            f"encimados")
        assert m["separacionMinima"] is not None
        assert m["separacionMinima"] < 0.5, (m["familia"],
                                             m["separacionMinima"])


def test_la_lectura_trae_cada_numero_que_la_tabla_del_cuaderno_imprime(
        construidas):
    """Every column the notebook's table shows comes from here.

    The notebook types no number, so a key missing from this dictionary is a
    column the notebook would have to compute itself -- which is the way a
    report starts disagreeing with its run.
    """
    l = toy_bags.lectura(construidas["toroide"])
    for clave in ("familia", "nota", "direccionesEfectivas",
                  "separacionEntreClases", "puntosPorBolsa",
                  "claseReproducida", "correspondencia", "bolsasPorClase"):
        assert clave in l, clave
    assert l["bolsasPorClase"] == len(set(construidas["toroide"].bs.tolist()))
    assert l["claseReproducida"] == 1.0
    assert toy_bags.lectura(construidas["lunas"])["claseReproducida"] is None
    assert toy_bags.lectura(construidas["lunas"])["lunas"] is not None


def test_el_numero_de_bolsas_llega_a_las_bolsas_que_se_dibujan(construidas):
    """Twenty-seven, and it reaches the bands the figure draws.

    Not a search and not a tunable with a different default: a figure drawing
    `3 * n * 2` hulls and as many names, with `n` chosen by a sweep, would be
    showing the sweep.

    **It used to be read off `config.BAGS_PER_CLASS` and no longer is.** The
    toyset borrowed the digit benchmark's rule while the two agreed at 12; the
    27-bag cut is this toyset's own, so the two declarations now DIFFER on
    purpose and that difference is asserted here rather than left to be met as
    a failed equality. `toy_run.material_bound` is what bends the digit global
    to this cut for the duration of a run, and the module docstring of
    `toy_run` records that the bend stopped being a no-op.

    **What is asserted is the MATERIAL and not the constant.** A test comparing
    `BAGS_PER_CLASS_TOY` with a literal would pass while `bolsas` cut a
    different number of bands -- the constant is only that function's default,
    and every caller may pass its own `n`. So the bands of a real draw are
    counted per class, in both domains, and the chance level is derived from the
    count that came back rather than from the constant.
    """
    from MIL_CREDA_Benchmark import config

    assert toy_bags.BAGS_PER_CLASS_TOY == 27
    assert config.BAGS_PER_CLASS == 12, (
        "the digit benchmark's own cut moved; this toyset bends it per run and "
        "does not own it")
    assert toy_bags.BAGS_PER_CLASS_TOY != config.BAGS_PER_CLASS, (
        "the two declarations agree again; if that is intended the docstring "
        "above is stale, and if it is not, one of them moved by accident")
    for nombre, fam in construidas.items():
        for labels, indices in ((fam.ys, fam.bs), (fam.yt, fam.bt)):
            assert sorted(set(indices.tolist())) == list(range(27)), nombre
            for c in np.unique(labels):
                bandas = sorted(set(indices[labels == c].tolist()))
                assert bandas == list(range(27)), (nombre, int(c), bandas)
            # Every band is non-empty, which is what makes the count above a
            # reading of the cut and not of the index range.
            for k in range(27):
                assert int(((labels == c) & (indices == k)).sum()) > 0, (nombre, k)
    bandas_dibujadas = len(set(construidas["toroide"].bs.tolist()))
    assert toy_bags.AZAR_DE_CORRESPONDENCIA == 1 / bandas_dibujadas
    # The reachable red for deriving the chance level: at a different cut it is
    # a different number, so the equality above is not a tautology. Both the
    # cut this replaces and the one before it are named, so a silent return to
    # either is red here.
    assert toy_bags.AZAR_DE_CORRESPONDENCIA not in (1 / 4, 1 / 12)


def test_las_dos_leyendas_explican_los_dos_canales_que_quedaron(dibujada):
    """One legend for the hue, one for the enclosure and its name -- and the
    retired channel is named in neither.

    The second legend used to be titled «la LUMINOSIDAD: la bolsa», and a
    legend describing a channel the figure no longer has is worse than no
    legend: it is read, and it is wrong. So this asserts what the two legends
    now say AND that nothing on the panel still promises a lightness ramp.

    The class swatches are checked against the untinted hue for the reason that
    was a real defect once: hung off a `scatter`, they came out in the colour
    of whichever bag the loop reached first, and one swatch meant two things.

    Red reachable: restoring either the old title or the tinted swatches fails
    here.
    """
    from CREDA.toy_experiment import COL_DURA
    from matplotlib.colors import to_rgb

    todas = [art for art in dibujada.fig.axes[0].get_children()
             if art.__class__.__name__ == "Legend"]
    por_titulo = {l.get_title().get_text(): l for l in todas}
    assert len(por_titulo) == 2, por_titulo

    titulos = " · ".join(por_titulo)
    assert "TONO" in titulos
    assert "CASCO" in titulos and "RÓTULO" in titulos
    assert "LUMINOSIDAD" not in titulos, (
        "una leyenda sigue prometiendo el canal que se retiró")

    clases = next(l for t, l in por_titulo.items() if "TONO" in t)
    muestras = [h.get_color() for h in clases.legend_handles]
    esperado = [to_rgb(COL_DURA[c]) for c in sorted(COL_DURA)]
    assert len(muestras) == len(esperado)
    for visto, quiero in zip(muestras, esperado):
        assert np.allclose(to_rgb(visto), quiero), (visto, quiero)

    # Y la leyenda de la bolsa muestra el casco con la MISMA opacidad con que
    # está dibujado en el panel: una muestra más fuerte que el casco real sería
    # una leyenda que promete algo que el panel no tiene.
    bolsa = next(l for t, l in por_titulo.items() if "CASCO" in t)
    casco_muestra = bolsa.legend_handles[0]
    assert casco_muestra.get_alpha() == toy_bags.ALFA_DEL_CASCO
    assert np.allclose(to_rgb(casco_muestra.get_facecolor()[:3]),
                       to_rgb(COL_DURA[0]))

    # Y el ejemplo que da la leyenda es un nombre que la figura dibuja de
    # verdad, no uno escrito a mano que podría quedar desfasado.
    nombres = {t.get_text() for t in dibujada.fig.axes[0].texts}
    ejemplos = bolsa.legend_handles[1].get_label()
    assert toy_bags.etiqueta_de_bolsa("s", 0, 0) in ejemplos
    assert toy_bags.etiqueta_de_bolsa("s", 0, 0) in nombres
    assert toy_bags.etiqueta_de_bolsa("t", 2, 3) in ejemplos
    assert toy_bags.etiqueta_de_bolsa("t", 2, 3) in nombres


def test_la_leyenda_de_clases_dice_las_dos_numeraciones(dibujada):
    """The class legend carries the array's index AND the one in the labels.

    `Bs11` is class index 0, and a reader handed a legend that says only
    «clase 0» has to work the `+ 1` out from nothing. The figure is the one
    place both numberings appear, so it is the place that has to say so.

    Red reachable: dropping the second number, or printing `c` twice, fails --
    and the second assertion is what stops this passing on a legend that
    repeated the same index in both halves.
    """
    todas = [art for art in dibujada.fig.axes[0].get_children()
             if art.__class__.__name__ == "Legend"]
    clases = next(l for l in todas if "TONO" in l.get_title().get_text())
    textos = [t.get_text() for t in clases.get_texts()]
    assert len(textos) == 3

    for c, texto in enumerate(sorted(textos)):
        assert f"clase {c}" in texto, texto
        assert str(c + 1) in texto.split("rótulo")[-1], texto
        assert texto.split("rótulo")[-1].strip(" ,") != str(c), texto
