"""La figura de la seccion 4: que la relevancia se vea SUBIR contra el azar.

La figura anterior dibujaba `pi` como 48 barras, el socio con borde negro, y
`d^2` bajando --- y la relevancia del socio, que es lo que la seccion afirma,
no estaba graficada como curva en ningun lado: era una barra entre 48 y un
numero en el titulo. Un lector no podia decir si lo que veia era bueno o malo
porque no habia ninguna referencia dibujada.

**Nada se retira.** El panel de las 48 barras dice a DONDE se fue la masa en
este barrido, `d^2` y la brecha al centro dicen que mueven los dos terminos, y
la curva del socio se AGREGA. Tres paneles donde habia dos.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from MIL_CREDA_Benchmark import toy_run


def _caminata(barridos: int = 6, bolsas: int = 6) -> dict:
    """Una caminata con la forma que el artefacto trae, y nada mas.

    Construida y no leida de disco: esto prueba el DIBUJO, y un artefacto real
    ataria la prueba a una corrida. Los numeros son crecientes a proposito ---
    lo que se asierta es que la curva que sale es la columna del socio, no que
    suba, porque si subir fuera la asercion la prueba mediria la fisica del
    paseo en lugar de la figura.
    """
    rng = np.random.default_rng(0)
    pi = rng.uniform(0.0, 0.05, size=(barridos, bolsas))
    pi[:, 0] = np.linspace(0.02, 0.30, barridos)
    return {
        "barridos": barridos,
        "bolsaDelBlanco": 0,
        "bolsaFuenteDestino": 0,
        "bolsasFuenteDeLaClase": bolsas,
        "clase": 0,
        "clasesFuente": [0] * (bolsas // 2) + [1] * (bolsas - bolsas // 2),
        "columnaDestino": 0,
        "g": [[0.8, 0.1, 0.1]] * barridos,
        "pi": pi.tolist(),
        "piMaximoDentroDeLaClase": pi.max(axis=1).tolist(),
        "pisoDeAzar": 0.0625,
        "distanciaCuadrada": np.linspace(0.48, 0.26, barridos).tolist(),
        "brechaAlCentro": np.linspace(0.90, 0.40, barridos).tolist(),
        "brechaCuadradaMinimaCruda": 0.07,
        "nube": rng.normal(size=(bolsas, 5)).tolist(),
        "camino": rng.normal(size=(barridos, 5)).tolist(),
    }


@pytest.fixture
def barrido() -> dict:
    return {"seccion": 4, "caminatas": {"lunas": _caminata()}}


def test_la_bolsa_se_ve_moverse_y_el_socio_esta_marcado(barrido):
    """El panel izquierdo trae la nube, la cruz del socio y la trayectoria.

    Una curva sola no muestra que algo se mueve --- muestra un numero
    cambiando --- y la figura de la seccion 2 funciona justamente porque el
    punto se ve caminar. La trayectoria es PARCIAL por cuadro, asi que lo que
    se asierta es que llega hasta `k` y no hasta el final.
    """
    import matplotlib.pyplot as plt

    caminata = barrido["caminatas"]["lunas"]
    fig = toy_run.figura_de_correspondencia(barrido, "lunas", 3)
    try:
        a = fig.axes[0]
        # La nube: un grupo de puntos por clase presente.
        assert len(a.collections) >= len(set(caminata["clasesFuente"])), \
            len(a.collections)
        # La cruz del socio, dicha por su nombre.
        assert any("socio" in t.get_text() for t in a.texts), \
            [t.get_text() for t in a.texts]
        # La trayectoria llega hasta `k`, no hasta el final.
        largos = {len(l.get_xdata()) for l in a.get_lines()}
        assert 4 in largos, largos
        assert len(caminata["camino"]) not in largos, largos
    finally:
        plt.close(fig)


def test_sin_nube_la_figura_lo_DICE_en_vez_de_dibujar_un_panel_vacio(barrido):
    """Un artefacto viejo no trae el plano, y un panel vacio miente.

    Vacio se leeria como "la bolsa no se movio", que es un resultado. Decirlo
    es la caida honesta.
    """
    import matplotlib.pyplot as plt

    del barrido["caminatas"]["lunas"]["nube"]
    fig = toy_run.figura_de_correspondencia(barrido, "lunas", 3)
    try:
        assert any("re-corr" in t.get_text() for t in fig.axes[0].texts), \
            [t.get_text() for t in fig.axes[0].texts]
    finally:
        plt.close(fig)


def test_la_curva_roja_es_el_loss_de_la_Ec_38_y_no_d2_crudo(barrido):
    """`l_loc,j = d_j^2 / 2`, que es lo que `normalized_local_distance` hace.

    La cantidad por su nombre y no `d^2` crudo: la Ec. (38) vive en [0, 1], asi
    que la curva se lee contra sus dos extremos en vez de contra una escala que
    cambia con la familia. Se asierta contra la funcion de `MIL_CREDA` y nunca
    contra un `/ 2` escrito aca, que pasaria igual el dia que la ecuacion
    cambie de constante.
    """
    import matplotlib.pyplot as plt
    from MIL_CREDA import local_term

    d2 = np.asarray(barrido["caminatas"]["lunas"]["distanciaCuadrada"], float)
    esperado = local_term.normalized_local_distance(d2[:4]).numpy()
    fig = toy_run.figura_de_correspondencia(barrido, "lunas", 3)
    try:
        series = [np.asarray(l.get_ydata(), dtype=float)
                  for a in fig.axes for l in a.get_lines()]
        assert any(s.shape == esperado.shape and np.allclose(s, esperado)
                   for s in series), [s.tolist() for s in series]
        # Y `d^2` crudo NO esta dibujado: si lo estuviera, el lector leeria la
        # escala equivocada en la unica curva que dice "lo que el local
        # minimiza".
        assert not any(s.shape == d2[:4].shape and np.allclose(s, d2[:4])
                       for s in series)
    finally:
        plt.close(fig)


def test_la_curva_del_socio_es_su_columna_y_lleva_la_linea_del_azar(barrido):
    """La curva es `pi[:k+1, columnaDestino]`, y el azar es una LINEA.

    La linea y no un numero en el titulo: sin una referencia dibujada el lector
    no tiene con que comparar la altura, que es exactamente por lo que la
    figura anterior no decia nada. El valor sale de `pisoDeAzar`, del artefacto,
    y nunca de una constante escrita en la figura.
    """
    caminata = barrido["caminatas"]["lunas"]
    esperada = np.asarray(caminata["pi"], dtype=float)[:4, 0]
    azar = float(caminata["pisoDeAzar"])
    fig = toy_run.figura_de_correspondencia(barrido, "lunas", 3)
    try:
        series = [np.asarray(l.get_ydata(), dtype=float)
                  for a in fig.axes for l in a.get_lines()]
        assert any(s.shape == esperada.shape and np.allclose(s, esperada)
                   for s in series), [s.tolist() for s in series]
        # La linea del azar: constante, en `pisoDeAzar`, y de punta a punta.
        planas = [s for s in series
                  if s.size >= 2 and np.allclose(s, azar)]
        assert planas, f"ninguna serie constante en {azar}"
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_el_azar_sale_del_artefacto_y_no_de_una_constante(barrido):
    """Cambiale `pisoDeAzar` y la linea se mueve con el.

    Un piso escrito en la figura pasaria los dos tests de arriba y mentiria el
    dia que la seccion corra con otro numero de bolsas por clase.
    """
    import matplotlib.pyplot as plt

    otro = 0.0250
    barrido["caminatas"]["lunas"]["pisoDeAzar"] = otro
    fig = toy_run.figura_de_correspondencia(barrido, "lunas", 3)
    try:
        series = [np.asarray(l.get_ydata(), dtype=float)
                  for a in fig.axes for l in a.get_lines()]
        assert any(s.size >= 2 and np.allclose(s, otro) for s in series)
        assert not any(s.size >= 2 and np.allclose(s, 0.0625) for s in series)
    finally:
        plt.close(fig)
