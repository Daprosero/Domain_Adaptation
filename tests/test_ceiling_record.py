"""El esquema del registro de techos y la regla de meseta.

La forma la escribe la búsqueda y la leen seis consumidores. Estos tests
sostienen que las dos formas conviven —la rejilla que gobierna la campaña
vigente y la búsqueda por trials— y que la regla que reemplaza al desempate hace
lo que dice.
"""

from __future__ import annotations

import pytest

from MIL_CREDA_Benchmark import ceiling_record as cr


def _trials(*pares):
    return [{"ceiling": c, "value": v} for c, v in pares]


def test_las_dos_formas_se_distinguen_por_lo_que_traen():
    assert cr.kind_of({"grid": [], "ceiling": 0.01}) == cr.KIND_GRID
    assert cr.kind_of({"search": {"trials": 30}, "ceiling": 0.01}) == cr.KIND_OPTUNA


def test_una_entrada_con_las_dos_es_ambigua_y_se_declara():
    """Dos motores no escriben la misma entrada: verlo significa una fusión."""
    with pytest.raises(ValueError, match="ambigua"):
        cr.kind_of({"grid": [], "search": {"trials": 30}})


def test_una_entrada_sin_forma_se_niega_en_vez_de_adivinar():
    with pytest.raises(ValueError, match="sin forma"):
        cr.kind_of({"ceiling": 0.01})


def test_dentro_de_la_meseta_gana_el_mas_chico():
    """La regla, y es la misma que la del desempate viejo."""
    elegido = cr.choose(_trials((0.1, 0.80), (0.01, 0.79), (0.001, 0.795)), noise=0.02)
    assert elegido["ceiling"] == 0.001
    assert elegido["decidedByFlatRule"] is True
    assert elegido["plateau"] == [0.001, 0.01, 0.1]


def test_fuera_de_la_meseta_gana_el_criterio_y_no_la_regla():
    """Cuando el GP sí distingue, el más chico pierde — que es lo correcto."""
    elegido = cr.choose(_trials((0.1, 0.90), (0.001, 0.50)), noise=0.02)
    assert elegido["ceiling"] == 0.1
    assert elegido["decidedByFlatRule"] is False
    assert elegido["plateau"] == [0.1]


def test_la_meseta_la_define_el_ruido_estimado_y_no_una_igualdad_exacta():
    """Sobre un rango continuo dos evaluaciones no empatan nunca.

    Con igualdad exacta la regla no se activaría jamás y el ganador lo pondría
    el último decimal, que es ruido con autoridad de criterio.
    """
    juego = _trials((0.1, 0.8000), (0.001, 0.7999))
    assert cr.choose(juego, noise=0.0)["ceiling"] == 0.1
    assert cr.choose(juego, noise=0.01)["ceiling"] == 0.001


def test_el_detalle_dice_cuanto_sostiene_el_numero_elegido():
    """«Cuál ganó» y «la búsqueda pudo distinguir» son hechos distintos."""
    d = cr.choose(_trials((0.1, 0.80), (0.01, 0.799)), noise=0.05)
    assert d["best"] == 0.80 and d["value"] == 0.799
    assert d["trials"] == 2 and d["noise"] == 0.05


def test_la_escala_se_reporta_en_los_ejes_que_la_forma_tiene():
    """Devolver `seeds` sobre una busqueda por trials lo haria indistinguible
    de una que corrio con cero semillas."""
    rejilla = {"grid": [], "epochs": 20, "seeds": [0, 1, 2]}
    trials = {"search": {"trials": 30}, "epochs": 20}
    assert cr.scale_of(rejilla) == {"epochs": 20, "seeds": 3}
    assert cr.scale_of(trials) == {"epochs": 20, "trials": 30}


def test_sin_trials_no_hay_techo_que_elegir():
    with pytest.raises(ValueError):
        cr.choose([], noise=0.01)
