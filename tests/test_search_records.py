"""Dos registros de techos, y cuál rige.

La búsqueda real cuesta horas y corre en Kaggle. El ensayo local existe para
saber si el programa **corre**, que es lo único que un ensayo puede contestar: a
escala piloto la rampa se satura en la segunda época y todo techo se alcanza casi
enseguida, así que su respuesta no es citable.

Antes había un solo archivo, y por eso `run_search` se negaba a tener un dial de
escala — un techo de ensayo se habría escrito donde va la respuesta completa y
una campaña lo habría consumido sin una palabra. Estos tests sostienen la
separación que hace admisible ese dial.
"""

from __future__ import annotations

import json

import pytest

from MIL_CREDA_Benchmark import config, harness


@pytest.fixture
def registros(tmp_path, monkeypatch):
    lleno = tmp_path / "ceilings.json"
    ensayo = tmp_path / "ceilings.pilot.json"
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", lleno)
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", ensayo)
    return lleno, ensayo


def _escribir(path, epochs, seeds):
    path.write_text(json.dumps({"creda": {
        "ceiling": 0.01, "epochs": epochs, "seeds": list(range(seeds)),
        "atRequiredScale": epochs >= config.FULL_SEARCH_EPOCHS,
        "requiredScale": {"epochs": config.FULL_SEARCH_EPOCHS,
                          "seeds": config.FULL_SEARCH_SEEDS},
        "byTransfer": {"M->U": 0.01}}}), encoding="utf-8")


def test_sin_ningun_registro_no_rige_nada(registros):
    assert config.ceilings_record_in_force() == (None, "none")
    assert config.ceilings_on_record() == {}


def test_solo_el_ensayo_rige_para_que_un_piloto_local_pueda_correr(registros):
    """Sin esto un piloto local no podría correr hasta gastar Kaggle."""
    lleno, ensayo = registros
    _escribir(ensayo, config.PILOT_SEARCH_EPOCHS, 1)
    assert config.ceilings_record_in_force() == (ensayo, "pilot")
    assert config.ceilings_on_record() == {"creda": 0.01}


def test_el_registro_completo_le_gana_siempre_al_ensayo(registros):
    """Un ensayo no desplaza una medición, exista o no primero."""
    lleno, ensayo = registros
    _escribir(ensayo, config.PILOT_SEARCH_EPOCHS, 1)
    _escribir(lleno, config.SEARCH_EPOCHS, 3)
    path, kind = config.ceilings_record_in_force()
    assert (path, kind) == (lleno, "full")


def test_la_procedencia_dice_cual_y_a_que_escala(registros):
    """«De dónde salió este escalar» es la pregunta que el registro debe poder
    contestar. Un resolutor que devolviera el mapping a secas haría
    indistinguibles los dos casos justo donde importa."""
    lleno, ensayo = registros
    _escribir(ensayo, config.PILOT_SEARCH_EPOCHS, 1)
    p = config.ceilings_provenance()
    assert p["source"] == "pilot"
    assert p["epochs"] == config.PILOT_SEARCH_EPOCHS
    assert p["seeds"] == 1
    assert p["atRequiredScale"] is False

    _escribir(lleno, config.SEARCH_EPOCHS, 3)
    p = config.ceilings_provenance()
    assert p["source"] == "full" and p["atRequiredScale"] is True


def test_el_ensayo_corre_a_su_propia_escala_declarada(registros):
    """Ni la del piloto que lo llama ni la de la búsqueda real."""
    assert config.PILOT_SEARCH_EPOCHS < config.SEARCH_EPOCHS
    assert len(config.PILOT_SEARCH_SEEDS) < len(config.SEARCH_SEEDS)


def test_el_parcial_del_ensayo_es_su_propio_archivo(registros):
    """Compartirlo dejaría que un ensayo reanude una búsqueda real a medias, o al
    revés: media rejilla de veinte épocas continuada con celdas de tres."""
    a = harness.shard_paths(None, pilot=True)["partial"]
    b = harness.shard_paths(None)["partial"]
    assert a != b
    assert "pilot" in a.name and "pilot" not in b.name


def test_search_record_lee_el_registro_que_se_le_pide(registros):
    lleno, ensayo = registros
    _escribir(ensayo, config.PILOT_SEARCH_EPOCHS, 1)
    assert harness.search_record(pilot=True) is not None
    assert harness.search_record() is None
    _escribir(lleno, config.SEARCH_EPOCHS, 3)
    assert harness.search_record() is not None


def test_el_destino_sale_de_un_solo_lugar_y_lo_usan_los_tres(registros):
    """Escritor, parcial y lector resuelven `pilot` por la misma funcion.

    Estaba repetido en los tres, y un test que mockeaba el del medio dejo pasar
    una mutacion en el escritor: cada mitad verificada contra su propio fixture
    y la union sin verificar. Esto ata las tres al mismo mapeo, asi que una
    mutacion en cualquiera de ellas cae aca.
    """
    lleno, ensayo = registros
    assert config.ceilings_record_for(True) == ensayo
    assert config.ceilings_record_for(False) == lleno
    # el parcial, que es el escritor de la busqueda a medias
    assert harness.shard_paths(None, pilot=True)["partial"].parent == ensayo.parent
    assert ensayo.stem in harness.shard_paths(None, pilot=True)["partial"].name
    # el lector
    _escribir(ensayo, config.PILOT_SEARCH_EPOCHS, 1)
    assert harness.search_record(pilot=True) is not None
    assert harness.search_record(pilot=False) is None


def test_un_ensayo_nunca_escribe_en_el_registro_completo(registros, monkeypatch):
    """La propiedad que hace admisible el dial de escala.

    Sin ella `pilot=True` sería exactamente el knob que la versión anterior de
    `run_search` prohibía con razón: aceptado, corrido barato, y su respuesta
    consumida por una campaña completa sin una palabra.
    """
    lleno, ensayo = registros
    visto = {}

    def _falso(reduction, device, progress=print, shard=None, pilot=False):
        visto["pilot"] = pilot
        destino = config.CEILINGS_PILOT_RECORD if pilot else config.CEILINGS_RECORD
        _escribir(destino, reduction.epochs, len(reduction.seeds))
        return {}

    monkeypatch.setattr(harness, "search_ceilings", _falso)
    monkeypatch.setattr(harness, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(harness, "environment", lambda: {})

    harness.ceilings_in_force(
        harness.Reduction(seeds=list(config.PILOT_SEARCH_SEEDS),
                          epochs=config.PILOT_SEARCH_EPOCHS),
        "cpu", progress=lambda *_: None, pilot=True)

    assert visto["pilot"] is True
    assert ensayo.exists()
    assert not lleno.exists(), "el ensayo escribio donde va la respuesta completa"


def test_la_busqueda_y_el_veredicto_se_separan_por_rol_no_por_transferencia():
    """La garantía que el docstring de `search_ceilings` describía mal.

    Decía que la búsqueda corre sobre transferencias que el veredicto nunca ve.
    Es falso y siempre lo fue: `SEARCH_TRANSFERS` está contenido en
    `VERDICT_TRANSFERS`. Lo que mantiene disjunto el material es el **rol** —
    `valid` para elegir, `eval` para juzgar— y eso vale para las seis, que es lo
    que hace admisible buscar en todas.
    """
    buscadas = {f"{a}->{b}" for a, b in config.SEARCH_TRANSFERS}
    juzgadas = {f"{a}->{b}" for a, b in config.VERDICT_TRANSFERS}
    assert buscadas & juzgadas, "si esto queda vacio, el docstring viejo era cierto"
    assert config.SEARCH_ROLE == "valid"
    assert config.VALID_BAGS and config.EVAL_BAGS
    assert config.VALID_BAGS + config.EVAL_BAGS <= config.BAGS_PER_DOMAIN
