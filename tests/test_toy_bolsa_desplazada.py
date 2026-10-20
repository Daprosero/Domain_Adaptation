"""Section 2's perturbation: three bags chosen at the frontier, and moved.

The section reads how `w` moves, so what has to be true BEFORE any of that is
measured is that the right bags were chosen and that they moved whole. Both are
checked on material where the answer is known in advance rather than on a run,
because a perturbation verified against its own output verifies nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from MIL_CREDA_Benchmark import toy_run


def _material(bolsas_por_clase=4, instancias=5, clases=3, sep=10.0):
    """Bands of three classes, far enough apart that the destination of a move
    is unambiguous, with the bag index and the label known by construction."""
    X, y, b = [], [], []
    for c in range(clases):
        for k in range(bolsas_por_clase):
            posicion = c * bolsas_por_clase + k
            centro = np.array([c * sep, k * 1.0])
            X.append(centro + np.linspace(-0.1, 0.1, instancias)[:, None])
            y += [c] * instancias
            b += [posicion] * instancias
    return np.vstack(X), np.array(y), np.array(b)


def test_la_frontera_elige_los_margenes_mas_chicos_y_en_orden():
    """A bag is at the frontier when its two highest scores are closest.

    The scores here are built so the margin ORDER is known: bag 0 is a tie,
    bag 1 is nearly one, and the rest sit between. The expected side is derived
    from that construction, never read back from the function.
    """
    scores = np.array([
        [0.50, 0.50, 0.00],   # margen 0.00
        [0.99, 0.01, 0.00],   # margen 0.98
        [0.45, 0.40, 0.15],   # margen 0.05
        [0.60, 0.30, 0.10],   # margen 0.30
    ])
    elegidas = toy_run.bolsas_en_la_frontera(scores, cuantas=3)
    assert list(elegidas) == [0, 2, 3], elegidas
    assert 1 not in elegidas, "la bolsa que el cabezal ya decidió entró igual"


def test_un_empate_de_margenes_se_rompe_por_posicion_y_no_por_el_azar():
    """Two bags with the same margin choose the same way on two runs.

    A tie broken by iteration order would let one material pick different bags
    on two runs, and then a difference between two runs could be the choice.
    """
    scores = np.array([[0.5, 0.5, 0.0]] * 4)
    primera = toy_run.bolsas_en_la_frontera(scores, cuantas=2)
    segunda = toy_run.bolsas_en_la_frontera(scores, cuantas=2)
    assert list(primera) == list(segunda) == [0, 1], (primera, segunda)


@pytest.mark.parametrize("cuantas", [0, -1, 99])
def test_un_pedido_que_no_cabe_en_el_material_se_niega(cuantas):
    with pytest.raises(ValueError):
        toy_run.bolsas_en_la_frontera(np.zeros((4, 3)), cuantas=cuantas)


def test_la_bolsa_se_mueve_ENTERA_y_su_geometria_interna_no_cambia():
    """Every instance moves by the same vector, so the bag arrives whole.

    This is the property that keeps the reading attributable: a per-instance
    displacement would move the WITHIN-bag kernel too, and then a change in `w`
    could be the position or the shape and the section could not say which.
    """
    X, y, b = _material()
    scores = np.zeros((12, 3)); scores[:, 0] = 1.0
    scores[0] = [0.5, 0.5, 0.0]
    movido, _ = toy_run.desplazar_bolsas(X, b, y, np.array([0]), scores, 0.5)
    filas = np.flatnonzero(b == 0)
    pasos = movido[filas] - X[filas]
    assert np.allclose(pasos, pasos[0]), pasos
    antes = X[filas] - X[filas].mean(axis=0)
    despues = movido[filas] - movido[filas].mean(axis=0)
    assert np.allclose(antes, despues), "la forma de la bolsa cambió"


def test_solo_las_bolsas_elegidas_se_mueven_y_el_resto_queda_byte_a_byte():
    """A change in `w` on an untouched bag is the method propagating it.

    If the perturbation leaked into other bags, that reading would be damage
    applied there instead, and the section's whole claim would collapse.
    """
    X, y, b = _material()
    scores = np.zeros((12, 3)); scores[:, 0] = 1.0
    scores[0] = [0.5, 0.5, 0.0]; scores[5] = [0.5, 0.0, 0.5]
    elegidas = np.array([0, 5])
    movido, _ = toy_run.desplazar_bolsas(X, b, y, elegidas, scores, 1.0)
    quietas = ~np.isin(b, elegidas)
    assert np.array_equal(movido[quietas], X[quietas])
    assert not np.array_equal(movido[~quietas], X[~quietas])


def test_el_cero_de_la_escalera_corre_el_camino_entero_y_no_mueve_nada():
    """The control runs the whole perturbation path and displaces nothing.

    So a difference between it and the unperturbed run is the HARNESS, which
    is the one thing a perturbation section cannot measure about itself.
    """
    X, y, b = _material()
    scores = np.zeros((12, 3)); scores[:, 0] = 1.0
    scores[0] = [0.5, 0.5, 0.0]
    movido, registros = toy_run.desplazar_bolsas(
        X, b, y, np.array([0]), scores, 0.0)
    assert np.array_equal(movido, X)
    assert registros and registros[0]["distanciaRecorrida"] == 0.0
    # Y el registro SÍ se llena: el destino y la distancia total se nombran
    # aunque no se recorran, o el control no diría contra qué no se movió.
    assert registros[0]["distanciaTotal"] > 0.0
    assert registros[0]["claseDestino"] != registros[0]["claseOriginal"]


@pytest.mark.parametrize("fraccion", toy_run.ESCALERA_DESPLAZAMIENTO)
def test_la_distancia_recorrida_es_la_fraccion_de_la_total(fraccion):
    """The ladder is a fraction of the distance, measured and not promised."""
    X, y, b = _material()
    scores = np.zeros((12, 3)); scores[:, 0] = 1.0
    scores[0] = [0.5, 0.5, 0.0]
    _, registros = toy_run.desplazar_bolsas(
        X, b, y, np.array([0]), scores, fraccion)
    r = registros[0]
    assert r["distanciaRecorrida"] == pytest.approx(
        fraccion * r["distanciaTotal"], rel=1e-9, abs=1e-12)


def test_el_destino_es_la_clase_que_salio_SEGUNDA_y_no_otra():
    """The bag moves toward the class that came second for IT.

    Checked by flipping which class is the runner-up on the same material: if
    the destination were fixed, or read off the label, the moved bag would land
    in the same place both times.
    """
    X, y, b = _material()
    scores = np.zeros((12, 3)); scores[:, 0] = 1.0
    hacia_uno = scores.copy(); hacia_uno[0] = [0.5, 0.4, 0.1]
    hacia_dos = scores.copy(); hacia_dos[0] = [0.5, 0.1, 0.4]
    a, ra = toy_run.desplazar_bolsas(X, b, y, np.array([0]), hacia_uno, 1.0)
    c, rc = toy_run.desplazar_bolsas(X, b, y, np.array([0]), hacia_dos, 1.0)
    assert ra[0]["claseDestino"] == 1 and rc[0]["claseDestino"] == 2
    filas = np.flatnonzero(b == 0)
    assert not np.allclose(a[filas], c[filas]), (
        "la bolsa aterrizó en el mismo lugar con dos segundas distintas")


def test_una_segunda_clase_que_el_blanco_no_tiene_se_niega():
    """No destination is invented when the runner-up class has no bags."""
    X, y, b = _material(bolsas_por_clase=2, clases=2)
    scores = np.zeros((4, 3)); scores[:, 0] = 1.0
    scores[0] = [0.5, 0.0, 0.5]        # la segunda es la clase 2, que no existe
    with pytest.raises(ValueError, match="no hay destino"):
        toy_run.desplazar_bolsas(X, b, y, np.array([0]), scores, 1.0)
