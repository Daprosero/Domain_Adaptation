"""El motor de búsqueda por trials, y qué lo separa de la rejilla.

Los dos miden lo mismo — el mismo `run_one`, el mismo rol, el mismo criterio — y
difieren solo en quién elige dónde mirar. Estos tests sostienen esa frase, que es
la que hace comparable un registro con el otro.
"""

from __future__ import annotations

import json

import pytest

from MIL_CREDA_Benchmark import ceiling_record as cr
from MIL_CREDA_Benchmark import config, harness


@pytest.fixture
def motor(tmp_path, monkeypatch):
    """Un motor completo con `run_one` falso: mide el buscador, no el modelo."""
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")
    monkeypatch.setattr(config, "SEARCH_ENGINE", "optuna")
    monkeypatch.setattr(config, "PILOT_SEARCH_TRIALS", 5)
    vistos = []

    class _Bolsa:
        pass

    def _build(code, cache, seed):
        return _Bolsa()

    def _run_one(arm, transfer, seed, reduction, device, material, *,
                 ceiling, role):
        vistos.append({"arm": arm, "transfer": transfer, "seed": seed,
                       "ceiling": ceiling, "role": role})
        # Un óptimo real adentro del rango, para que el GP tenga qué encontrar.
        return {config.SEARCH_CRITERION: 1.0 - abs(ceiling - 0.01)}

    monkeypatch.setattr(harness.bags, "build", _build)
    monkeypatch.setattr(harness, "run_one", _run_one)
    return vistos


def _correr(pilot=True):
    return harness.search_ceilings_trials(
        harness.Reduction(seeds=[config.SEARCH_SEED],
                          epochs=config.PILOT_SEARCH_EPOCHS),
        "cpu", progress=lambda *_: None, pilot=pilot)


def test_busca_en_todas_las_transferencias(motor):
    """La rejilla medía dos y cuatro heredaban. Ya no hereda nada."""
    found = _correr()
    for entrada in found.values():
        assert len(entrada["byTransfer"]) == len(config.SEARCH_TRANSFERS) == 6
        assert entrada["inheritanceRule"].startswith("ninguna")


def test_solo_busca_sobre_los_metodos_completos(motor):
    """Un techo por brazo haría indistinguible el término del coeficiente."""
    vistos = motor
    _correr()
    assert {v["arm"] for v in vistos} == set(config.SEARCH_ARMS.values()) == {"D", "G"}


def test_una_sola_semilla_declarada_en_todos_los_trials(motor):
    """Dos trials sobre semillas distintas medirían el techo y el sorteo a la vez."""
    vistos = motor
    _correr()
    assert {v["seed"] for v in vistos} == {config.SEARCH_SEED}


def test_mide_sobre_el_rol_de_busqueda_y_nunca_sobre_el_del_veredicto(motor):
    """Lo que mantiene disjunto el material, ahora que las transferencias se
    comparten enteras con el veredicto."""
    vistos = motor
    _correr()
    assert {v["role"] for v in vistos} == {config.SEARCH_ROLE} == {"valid"}


def test_cada_trial_es_una_evaluacion_y_no_se_repite(motor):
    vistos = motor
    _correr()
    esperado = (len(config.SEARCH_ARMS) * len(config.SEARCH_TRANSFERS)
                * config.PILOT_SEARCH_TRIALS)
    assert len(vistos) == esperado


def test_los_trials_de_una_transferencia_visitan_techos_distintos(motor):
    """Un rango continuo: si dos trials cayeran en el mismo punto, el buscador
    estaría gastando evaluaciones en repetir en vez de en explorar."""
    vistos = motor
    _correr()
    de_una = [v["ceiling"] for v in vistos
              if v["arm"] == "D" and v["transfer"] == config.SEARCH_TRANSFERS[0]]
    assert len(set(de_una)) == len(de_una) > 1


def test_el_registro_sale_con_la_forma_que_los_lectores_esperan(motor):
    found = _correr()
    for entrada in found.values():
        assert cr.kind_of(entrada) == cr.KIND_OPTUNA
        assert cr.scale_of(entrada)["trials"] == config.PILOT_SEARCH_TRIALS
        # `byTransfer` sigue siendo etiqueta -> float: es lo que leen
        # `config.ceilings_by_transfer_on_record` y `harness.ceiling_for`.
        assert all(isinstance(v, float) for v in entrada["byTransfer"].values())
        assert set(entrada["perTransfer"]) == set(entrada["byTransfer"])


def test_el_ensayo_escribe_su_propio_registro(motor):
    _correr(pilot=True)
    assert config.CEILINGS_PILOT_RECORD.exists()
    assert not config.CEILINGS_RECORD.exists()


def test_un_ensayo_nunca_dice_estar_a_la_escala_requerida(motor):
    """Su respuesta no es citable, y el registro tiene que decirlo solo."""
    found = _correr(pilot=True)
    for entrada in found.values():
        assert entrada["atRequiredScale"] is False
        assert entrada["requiredScale"] == {"epochs": config.FULL_SEARCH_EPOCHS,
                                            "trials": config.SEARCH_TRIALS}


def test_el_techo_agrupado_sale_de_la_misma_regla_y_no_de_la_nada(motor):
    """Con las seis medidas nadie lo alcanza, pero un campo que nadie sabe de
    dónde salió es peor que un campo ausente."""
    found = _correr()
    for entrada in found.values():
        assert entrada["ceiling"] in set(entrada["byTransfer"].values())


def test_cada_estudio_explora_puntos_propios_y_no_los_del_vecino(motor):
    """Doce búsquedas independientes que arrancan del mismo lugar no lo son.

    Con una semilla compartida las propuestas de arranque son idénticas y los
    doce estudios recorren los mismos techos — medido en el ensayo local, las
    seis transferencias de `creda` visitaron los mismos cuatro. Cuando la meseta
    es ancha el ganador es el más chico *visitado*, así que el registro muestra
    un acuerdo entre transferencias que es artefacto de la semilla y se lee como
    hallazgo.
    """
    vistos = motor
    _correr()
    por_estudio = {}
    for v in vistos:
        por_estudio.setdefault((v["arm"], v["transfer"]), []).append(v["ceiling"])
    conjuntos = [tuple(sorted(c)) for c in por_estudio.values()]
    assert len(por_estudio) == 12
    assert len(set(conjuntos)) == len(conjuntos), (
        "dos estudios visitaron exactamente los mismos techos")


def test_la_semilla_de_cada_estudio_es_reproducible(motor):
    """Independientes, no aleatorias: dos corridas del mismo ensayo coinciden."""
    primera = _correr()
    segunda = _correr()
    assert (primera["creda"]["byTransfer"] == segunda["creda"]["byTransfer"])
