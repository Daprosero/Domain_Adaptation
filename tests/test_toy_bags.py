"""The toy bags: how `toy_bags.construir` cuts each class into bags.

Every assertion here has a reachable red, and the ones that probe a decision
carry both poles: the decision satisfies the criterion AND the alternative
fails it. One pole alone cannot distinguish a design decision from a
measurement that would have passed anything.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from CREDA.toy_experiment import DURAS
from MIL_CREDA_Benchmark import toy_bags

FAMILIAS = dict(DURAS)


@pytest.fixture(scope="module")
def construidas():
    return {nombre: toy_bags.construir(nombre, fn) for nombre, fn in DURAS}


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

    This is what makes a bag a slice of the manifold. A partition that
    interleaved would still produce `n` bags per class, and each would be the
    same sack around the whole class -- it would read as a traverse of the
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
        f"{nombre}: some bag's parameter range overlaps another's, so it is "
        "not a slice of the manifold")

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


# ---------------------------------------------------- el numero de bolsas

def test_el_numero_de_bolsas_llega_al_material(construidas):
    """Twenty-seven, and it reaches the bands a real draw carries.

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
    counted per class, in both domains.
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
