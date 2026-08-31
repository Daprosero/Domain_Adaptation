"""Lo que la búsqueda promete y lo que su registro efectivamente escribe.

Los dos motores están cubiertos por separado — `test_search_engine.py` mide los
trials y `test_benchmark_declarations.py` la rejilla — y entre los dos quedaban
sin sostener las promesas que no son *quién gana*: contra qué neutro se lee el
techo elegido, si el registro dice que cada semilla habría elegido igual, qué
ancho de meseta recibe la regla, y qué dimensiones NO se buscan.

Todo lo de acá conduce a la función real con un `run_one` falso: se mide el
buscador y su registro, nunca el modelo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from MIL_CREDA_Benchmark import ceiling_record, config, harness


# --------------------------------------------------------------- la rejilla

def _grid_search(scores: dict[float, list[float]], monkeypatch, tmp_path,
                 transfers: int = 1) -> dict:
    """Una entrada de familia, dada una puntuación por (celda, techo).

    Conduce `harness.search_ceilings` con la rejilla declarada como motor: la
    regla bajo prueba es la que corrió la campaña vigente, no una copia.
    """
    grid = sorted(scores)
    seeds = list(range(len(scores[grid[0]])))
    monkeypatch.setattr(config, "SEARCH_ENGINE", "grid")
    monkeypatch.setattr(config, "CEILING_GRID", grid)
    monkeypatch.setattr(config, "SEARCH_SEEDS", seeds)
    monkeypatch.setattr(config, "SEARCH_ARMS", {"milcreda": "G"})
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")

    class _Bags:
        pass

    def _build(code, cache, seed, noise=0.0):
        return _Bags()

    def _run_one(arm, transfer, seed, reduction, device, material, *, ceiling, role):
        label = harness.transfer_label(transfer)
        offset = config.SEARCH_TRANSFERS.index(transfer) / 100
        return {config.SEARCH_CRITERION: scores[ceiling][seeds.index(seed)] + offset}

    monkeypatch.setattr(harness.bags, "build", _build)
    monkeypatch.setattr(harness, "run_one", _run_one)
    found = harness.search_ceilings(
        harness.Reduction(seeds=seeds, epochs=config.SEARCH_EPOCHS),
        "cpu", progress=lambda *_: None,
        transfers=list(config.SEARCH_TRANSFERS[:transfers]))
    return found["milcreda"]


def test_el_registro_escribe_el_neutro_al_lado_del_techo_que_encontro(
        monkeypatch, tmp_path) -> None:
    """El 1.0 dejó de ser el valor y pasó a ser contra qué se lee el elegido.

    Sin el neutro en el registro, un techo de 0.98 y uno de 0.02 son dos números
    sueltos: lo que dice si la normalización quedó confirmada por medición es la
    distancia al neutro, y eso exige que el neutro viaje al lado del valor y en
    la misma corrida que lo eligió.

    Rojo alcanzable: sacar `neutral` del registro, o escribirlo desde otra cosa
    que no sea `RAMP_CEILING`.
    """
    entrada = _grid_search({0.01: [0.50, 0.52], 1.0: [0.90, 0.92]},
                           monkeypatch, tmp_path)
    assert entrada["neutral"] == config.RAMP_CEILING == 1.0
    assert entrada["ceiling"] == 1.0

    escrito = json.loads(config.CEILINGS_RECORD.read_text(encoding="utf-8"))
    assert escrito["milcreda"]["neutral"] == config.RAMP_CEILING
    assert "ceiling" in escrito["milcreda"], "el neutro sin el valor no se lee contra nada"


def test_el_registro_dice_si_cada_semilla_habria_elegido_lo_mismo(
        monkeypatch, tmp_path) -> None:
    """Tres semillas sobre tres techos distintos y tres sobre el mismo producen
    el mismo ganador y no son la misma evidencia.

    Las dos direcciones, porque un campo que siempre dice `True` pasa la mitad de
    esta prueba sin medir nada.

    Rojo alcanzable: fijar `seedsAgree` en `True`, o calcularlo sobre el ganador
    agrupado en vez de sobre la elección de cada semilla.
    """
    de_acuerdo = _grid_search({0.01: [0.50, 0.50], 1.0: [0.90, 0.90]},
                              monkeypatch, tmp_path)
    assert de_acuerdo["seedsAgree"] is True
    assert set(de_acuerdo["perSeedPick"].values()) == {1.0}

    # la primera semilla prefiere el techo chico y la segunda el grande: el
    # agrupado elige uno igual, y el registro tiene que decir que no hubo acuerdo
    en_desacuerdo = _grid_search({0.01: [0.90, 0.10], 1.0: [0.10, 0.95]},
                                 monkeypatch, tmp_path)
    assert en_desacuerdo["seedsAgree"] is False
    assert en_desacuerdo["perSeedPick"] == {"0": 0.01, "1": 1.0}


def test_la_rejilla_no_busca_la_velocidad_de_crecimiento(monkeypatch, tmp_path) -> None:
    """`RAMP_DELTA` es de CREDA, vale 20 y las dos familias corren con ella.

    Buscarla también habría convertido la corrida en una rejilla 2D de tres a
    cinco horas, y techo y velocidad están confundidos: un techo alto alcanzado
    despacio y uno bajo alcanzado rápido dan trayectorias parecidas.

    Que no se busque no es una intención: `run_one` no tiene por dónde recibir
    una velocidad, así que no hay forma de expresar un trial que la varíe, y el
    registro no lleva ningún campo que la nombre.

    Rojo alcanzable: agregarle un parámetro `delta` a `run_one`, o dejar que la
    búsqueda escriba una velocidad en su registro.
    """
    import inspect

    assert config.RAMP_DELTA == 20
    # la de CREDA, leída de su propia configuración y no repetida acá
    propia = json.loads((config.REPOSITORY / "CREDA" / "Notebooks"
                         / "Results_Generator.ipynb").read_text(encoding="utf-8"))
    fuente = "".join("".join(cell["source"]) for cell in propia["cells"]
                     if cell["cell_type"] == "code")
    declaradas = {int(v) for v in re.findall(r'"delta":\s*(\d+)', fuente)}
    assert declaradas == {config.RAMP_DELTA}, \
        f"la velocidad del trabajo previo es {declaradas} y acá corre {config.RAMP_DELTA}"

    assert "delta" not in inspect.signature(harness.run_one).parameters
    assert harness.ramp.__defaults__[0] == config.RAMP_DELTA

    entrada = _grid_search({0.01: [0.5], 1.0: [0.9]}, monkeypatch, tmp_path)
    escrito = json.dumps(entrada).lower()
    assert "delta" not in escrito, "la búsqueda escribió una velocidad que no buscó"
    # y la corrida sí la lleva en sus cotas, que es donde una constante compartida
    # tiene que quedar registrada
    assert harness.Reduction().rampDelta == config.RAMP_DELTA


# ------------------------------------------------------------------ los trials

@pytest.fixture
def estudios(tmp_path, monkeypatch):
    """El motor de trials, con `run_one` falso y los estudios de optuna a la vista."""
    import optuna

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")
    monkeypatch.setattr(config, "SEARCH_ENGINE", "optuna")
    monkeypatch.setattr(config, "PILOT_SEARCH_TRIALS", 5)
    creados = []
    anchos = []

    original_create = optuna.create_study

    def _create_study(*args, **kwargs):
        study = original_create(*args, **kwargs)
        creados.append(study)
        return study

    original_choose = ceiling_record.choose

    def _choose(visited, width):
        anchos.append(width)
        return original_choose(visited, width)

    class _Bags:
        pass

    def _build(code, cache, seed, noise=0.0):
        return _Bags()

    def _run_one(arm, transfer, seed, reduction, device, material, *, ceiling, role):
        return {config.SEARCH_CRITERION: 1.0 - abs(ceiling - 0.01)}

    monkeypatch.setattr(optuna, "create_study", _create_study)
    monkeypatch.setattr(harness.ceiling_record, "choose", _choose)
    monkeypatch.setattr(harness.bags, "build", _build)
    monkeypatch.setattr(harness, "run_one", _run_one)
    return {"studies": creados, "widths": anchos}


def _correr_trials(transfers: int = 1) -> dict:
    return harness.search_ceilings_trials(
        harness.Reduction(seeds=[config.SEARCH_SEED],
                          epochs=config.PILOT_SEARCH_EPOCHS),
        "cpu", progress=lambda *_: None, pilot=True,
        transfers=list(config.SEARCH_TRANSFERS[:transfers]))


def test_los_trials_buscan_una_sola_dimension_y_es_el_techo(estudios) -> None:
    """La velocidad de crecimiento sigue sin buscarse, y ahora la razón es otra:
    las dos familias ya están a un orden de magnitud en `adaptationShare`, así que
    una segunda dimensión libre amplifica ese desbalance en vez de resolverlo.

    Se mide sobre el espacio que optuna realmente exploró y no sobre la prosa:
    cada trial declara sus propias distribuciones, y ahí una segunda dimensión
    aparecería sin que nadie tenga que acordarse de mirarla.

    Rojo alcanzable: agregar un `trial.suggest_*` más al objetivo.
    """
    _correr_trials()
    assert estudios["studies"], "ningún estudio corrió"
    for study in estudios["studies"]:
        for trial in study.trials:
            assert set(trial.distributions) == {"ceiling"}
            assert set(trial.params) == {"ceiling"}


def test_la_meseta_es_la_resolucion_del_instrumento_y_no_el_ruido_del_gp(
        estudios) -> None:
    """Dos techos que difieren en menos de una bolsa no son distinguibles por la
    medición, opine lo que opine el modelo.

    Lo que la regla recibe es el ancho, así que es el ancho lo que se mide acá y
    no la frase que lo describe: una cantidad ajustada por el GP haría que la
    meseta dependiera de qué tan bien ajustó, que es la propiedad equivocada —
    y con `run_one` falso el GP ajusta cualquier cosa sin que esto se mueva.

    Rojo alcanzable: pasarle a `choose` cualquier otro ancho, incluido uno
    derivado del propio estudio.
    """
    encontrado = _correr_trials()
    assert estudios["widths"], "la regla de meseta nunca fue invocada"
    assert set(estudios["widths"]) == {config.SEARCH_RESOLUTION}
    assert config.SEARCH_RESOLUTION == 1.0 / config.VALID_BAGS

    entrada = encontrado["milcreda"]
    assert entrada["noise"] == config.SEARCH_RESOLUTION
    assert entrada["search"]["resolution"] == config.SEARCH_RESOLUTION
    assert entrada["flatRule"] == ceiling_record.FLAT_RULE


def test_el_registro_de_trials_tambien_escribe_el_neutro(estudios) -> None:
    """El mismo neutro, en el motor que reemplazó a la rejilla: si sólo lo
    escribiera el motor retirado, el campo desaparecería con él."""
    encontrado = _correr_trials()
    for entrada in encontrado.values():
        assert entrada["neutral"] == config.RAMP_CEILING == 1.0


# ------------------------------------------------------- la campaña y su material

def test_la_campana_se_niega_sin_techos(tmp_path, monkeypatch) -> None:
    """Financiar el propio coeficiente con la corrida que se va a informar es
    elegir y juzgar en una sola pasada.

    La mitad *por debajo de la escala* ya estaba sostenida; ésta es la otra, y es
    la que decide si el rechazo existe del todo: sin ella, borrar la guarda entera
    deja la campaña corriendo a `RAMP_CEILING` en las dos familias sin una palabra.

    Y se niega **antes** de construir material: `bags.build` acá levanta, así que
    una campaña que llegue a dibujar una sola bolsa cae por ahí y no por el
    rechazo. Sin eso, sacar la guarda no deja un test rojo sino una campaña real
    corriendo dentro de la suite.

    Rojo alcanzable: sacar el `if not reduction.ceilings`.
    """
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")

    def _nunca(*args, **kwargs):
        raise AssertionError("la campaña empezó a construir material sin techos")

    monkeypatch.setattr(harness.bags, "build", _nunca)

    with pytest.raises(SystemExit) as levantado:
        harness.campaign(harness.Reduction(ceilings={}), torch.device("cpu"),
                         arms=["A"], progress=lambda *a: None)
    dicho = str(levantado.value)
    assert "refusing to run without the searched ceilings" in dicho
    assert str(config.SEARCH_EPOCHS) in dicho, "el rechazo no dice a qué escala se busca"


# ------------------------------------------------- el neutro y lo que se dijo de él

def _comentario_de(nombre: str) -> str:
    """El bloque `#:` pegado arriba de una constante de `config.py`.

    Leído del archivo y no de un docstring porque una constante no tiene uno: lo
    que documenta a `RAMP_CEILING` es ese bloque, y es donde vive la promesa que
    esta prueba sostiene.
    """
    lineas = Path(config.__file__).read_text(encoding="utf-8").splitlines()
    n = next(i for i, l in enumerate(lineas) if l.startswith(f"{nombre} ="))
    bloque = []
    while n and lineas[n - 1].startswith("#:"):
        n -= 1
        bloque.append(lineas[n])
    assert bloque, f"{nombre} no lleva comentario"
    return "\n".join(reversed(bloque))


def test_el_comentario_del_neutro_fecha_el_1e_4_y_no_promete_un_techo_comun(
) -> None:
    """Dos cosas que el comentario decía y ya no son ciertas del mismo modo.

    La medición vieja — a 1e-4, 1e-2 y 1e-1 todo brazo adaptado puntuó lo que su
    propio piso — se tomó contra el objetivo **sin normalizar**, antes de que la
    Ec. (18) se dividiera por `B_src`. Sin esa marca se lee como una lectura del
    objetivo que el mismo comentario describe arriba, que es otro objetivo.

    Y la cláusula de cierre decía que *ambos lados corren en este techo*. Dejó de
    ser cierta cuando cada familia empezó a buscar el suyo: acá no se lee que la
    frase no esté, se mide por qué era falsa — `ceiling_for` devuelve techos
    distintos para las dos familias en la misma transferencia.

    Rojo alcanzable: volver a poner la cláusula, o sacar la marca `un-normalized`.
    """
    comentario = _comentario_de("RAMP_CEILING")
    assert "1e-4" in comentario, "el comentario ya no fecha la medición vieja"
    assert "un-normalized" in comentario.lower(), \
        "la medición de 1e-4 no está marcada como tomada sin normalizar"
    assert "both sides run at this ceiling" not in comentario.lower(), \
        "el comentario sigue prometiendo un techo común para las dos familias"

    # Por qué esa promesa era falsa, ejecutado y no leído.
    reduccion = harness.Reduction(ceilings={"creda": 1e-4, "milcreda": 1.0},
                                  ceilingsByTransfer={})
    transferencia = config.SEARCH_TRANSFERS[0]
    assert (harness.ceiling_for(reduccion, "creda", transferencia)
            != harness.ceiling_for(reduccion, "milcreda", transferencia)), \
        "las dos familias corrieron en el mismo techo: la cláusula vieja era cierta"
    assert "ceiling_for" in comentario, \
        "el comentario no nombra de dónde sale el coeficiente de cada brazo"
