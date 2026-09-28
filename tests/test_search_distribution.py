"""Lo que hace falta para que la búsqueda se reparta entre máquinas.

La campaña y los mecanismos ya se reparten: su eje es la semilla, `run_campaign_
shard`/`run_mechanism_sweep_shard` aceptan `seeds=`, y `config.execution_seed_
units()` traduce las unidades de una submisión a ese eje. La búsqueda no, y la
razón que se creyó durante una sesión entera ---"no tiene eje"--- es falsa:
`search_ceilings` y `search_ceilings_trials` ya aceptan `transfers=`, medido en
sus firmas. Lo que falta está más arriba y más abajo de ellas.

Cuatro piezas, una por test, y ninguna es la misma:

1. La declaración nombra un segundo eje. Hoy `__benchmark__["distribution"]
   ["axis"]` dice `"seed"` en singular, y el docstring de `execution_seed_units`
   apoya su propia decisión en eso: "no hay un segundo eje que una unidad
   pudiera nombrar". Mientras la declaración diga eso, cualquier lector que
   devuelva transferencias está contradiciendo al repositorio.
2. `run_search` acepta y reenvía `transfers`. Es el único punto de entrada que
   un `run-config.json` puede nombrar, así que una capacidad que él no expone
   es una capacidad que ningún worker puede pedir.
3. Las unidades se leen en el eje que el paso declara, no siempre como semillas.
   Una sola variable de entorno con dos significados posibles no puede
   resolverse adivinando: la búsqueda no tiene semillas que repartir y la
   campaña no tiene transferencias.
4. **Los registros parciales se unen.** Ésta es la que no existe en ninguna
   forma. `run_search` escribe UN registro en `config.ceilings_record_for(pilot)`
   y lo vuelve a leer del disco; seis máquinas escribiendo una transferencia
   cada una se pisan en la misma ruta, o dejan seis archivos que nada ensambla.
   Y el modo en que falla es el peor: un `ceilings.json` que dice tener seis
   transferencias mientras sostiene una, con todo lo de arriba en verde.

Rojo a propósito, y antes de tocar `src/`: `report_digest.source_digest` cubre
`src/` entero y nada más, así que una prueba nueva no vuelve rancio ningún
cuaderno, y el primer byte que se toque de `src/` sí obliga a re-caminar el
piloto completo y a repetir los dieciocho ensayos.
"""

from __future__ import annotations

import inspect

import pytest

import MIL_CREDA_Benchmark
from MIL_CREDA_Benchmark import config, harness


def test_the_declaration_names_the_axis_the_search_can_split_on() -> None:
    """El eje declarado es plural, y la transferencia es uno de ellos.

    No se afirma que `"seed"` desaparezca: la campaña y los mecanismos lo
    siguen usando y el registro de shards lo nombra. Lo que se pide es que la
    declaración pueda sostener DOS, porque hoy `execution_seed_units` cita
    explícitamente la singularidad de este campo como la razón de leer una
    unidad como semilla, y esa cita deja de ser cierta en cuanto la búsqueda
    se reparta.
    """
    distribution = MIL_CREDA_Benchmark.__benchmark__["distribution"]
    axes = distribution.get("axes")
    assert axes is not None, (
        "`distribution` declara `axis` en singular y no `axes`; mientras sea "
        "un solo valor, ningún paso puede declarar que reparte por otra cosa"
    )
    assert "seed" in axes, "la campaña y los mecanismos siguen repartiendo por semilla"
    assert "transfer" in axes, (
        "la búsqueda reparte por transferencia -- `search_ceilings` ya acepta "
        "`transfers=`, y ninguna declaración lo nombra"
    )


def test_run_search_forwards_the_transfers_it_is_given() -> None:
    """El lanzador expone el eje que su motor ya acepta.

    `run_search` es el único `module.function` que un `run-config.json` puede
    nombrar para este paso. Que `search_ceilings_trials` acepte `transfers` no
    sirve de nada mientras el lanzador no lo pase: la capacidad existe y es
    inalcanzable desde un worker.
    """
    firma = inspect.signature(harness.run_search)
    assert "transfers" in firma.parameters, (
        "`run_search` no expone `transfers`, así que un worker no puede pedir "
        f"una porción de la búsqueda; hoy acepta {list(firma.parameters)}"
    )
    assert firma.parameters["transfers"].default is None, (
        "omitirlo tiene que seguir significando `la búsqueda entera`, que es "
        "el comportamiento de hoy y el que la caminata local usa"
    )


def test_a_unit_resolves_to_the_axis_the_step_declares() -> None:
    """Una unidad no es siempre una semilla.

    `FORGE_RUN_UNITS` es una sola variable y la forja la declara opaca a
    propósito: decidir qué es una unidad es de este repositorio. Con dos ejes
    la lectura ya no puede ser incondicional, y adivinar es exactamente lo que
    no se puede hacer -- una transferencia leída como semilla no falla, corre
    otra cosa y devuelve números con la forma correcta.
    """
    assert hasattr(config, "execution_units_for"), (
        "no hay lectura de unidades por eje; hoy sólo existe "
        "`execution_seed_units()`, que decide 'una unidad ES una semilla' sin "
        "preguntar de qué paso se trata"
    )
    leer = config.execution_units_for
    firma = inspect.signature(leer)
    assert "axis" in firma.parameters or "eje" in firma.parameters, (
        "la lectura tiene que recibir el eje y nunca deducirlo del contenido: "
        "'0' es una semilla válida y también podría ser un índice de "
        f"transferencia; hoy recibe {list(firma.parameters)}"
    )


def test_partial_search_records_merge_instead_of_overwriting() -> None:
    """Seis máquinas, seis transferencias, UN registro que las contiene.

    La pieza que no existe en ninguna forma, y la única cuyo modo de falla es
    silencioso. `run_search` escribe en `config.ceilings_record_for(pilot)` --
    una ruta, no una por shard -- y `search_record` la vuelve a leer de ahí. Sin
    una unión, repartir la búsqueda produce un registro que afirma seis
    transferencias sosteniendo una, y la campaña que lo lea corre con techos que
    nunca se buscaron para cinco de sus seis transferencias.

    Se pide la unión y también su negativa: un registro al que le falta una
    transferencia declarada no puede pasar por completo, porque un techo ausente
    leído como neutro es precisamente el reemplazo que el repositorio ya prohíbe
    en otros lados.
    """
    unir = getattr(harness, "merge_search_records", None)
    assert unir is not None, (
        "no existe unión de registros parciales de la búsqueda; con "
        "`transfers` expuesto y sin esto, seis workers se pisan en la misma "
        "ruta o dejan seis archivos que nada ensambla"
    )
    firma = inspect.signature(unir)
    assert "transfers" in firma.parameters or "expected" in firma.parameters, (
        "la unión tiene que saber qué transferencias ESPERABA para poder "
        "negarse cuando falta una; contar lo que llegó no distingue 'llegaron "
        f"todas' de 'llegaron las que se mandaron'; hoy recibe {list(firma.parameters)}"
    )


def test_the_search_still_has_no_seed_axis_to_split() -> None:
    """Lo que este cambio NO toca, fijado para que nadie lo mueva de paso.

    El motor vivo es `optuna` y cancela el sorteo por construcción: "los treinta
    trials de una transferencia corren sobre el material idéntico, dibujado una
    vez". Repartir por transferencia no es una puerta para repartir por semilla.

    Y se fija por la firma del lanzador, NO por `config.SEARCH_SEEDS`. Esa
    constante vale `[0, 1, 2]` y pertenece a la lectura apareada de la rejilla,
    el motor retirado: leerla acá mediría algo que no corre, que es exactamente
    el error que cinco tests de este repositorio ya cometieron con
    `CEILING_GRID`.
    """
    assert config.SEARCH_ENGINE == "optuna", (
        "este test describe el motor de trials; si el motor cambió, la decisión "
        f"de una sola semilla hay que volver a leerla (hoy {config.SEARCH_ENGINE})"
    )
    firma = inspect.signature(harness.run_search)
    assert "seeds" not in firma.parameters, (
        "`run_search` no reparte por semilla y no debe empezar a hacerlo: su "
        "eje es la transferencia"
    )


def test_a_partial_search_refuses_to_present_its_slice_as_the_record(monkeypatch) -> None:
    """Pedir una porción es alcanzable; presentarla como el registro, no.

    `run_search` reenvía `transfers` hasta el motor, así que la porción se puede
    pedir hoy. Lo que NO se puede es devolverla: no hay unión de registros
    parciales, y las dos salidas sin negativa son ambas incorrectas -- devolver
    la porción entrega a la campaña techos que nunca se buscaron para las otras
    transferencias, y un segundo worker escribiendo esa misma ruta pisa al
    primero. Las dos producen números con la forma exacta de los correctos.

    Se conduce la función real con la búsqueda falsa: se mide la negativa, nunca
    el modelo. Y se mide en las dos direcciones, que es lo único que prueba que
    la negativa distingue algo: con `transfers` se niega, sin `transfers`
    devuelve el registro igual que antes.
    """
    registro = {"milcreda": {"ceiling": 1e-4, "currentStamp": True}}
    monkeypatch.setattr(harness, "ceilings_in_force", lambda *a, **k: {})
    monkeypatch.setattr(harness, "search_record", lambda pilot=False: registro)
    monkeypatch.setattr(harness, "resolve_device", lambda: "cpu")

    with pytest.raises(SystemExit) as refusal:
        harness.run_search(transfers=[config.VERDICT_TRANSFERS[0]])
    mensaje = str(refusal.value)
    assert "PARTIAL" in mensaje, (
        "la negativa tiene que nombrar que lo pedido es PARCIAL. Ojo con el "
        "sujeto: parcial es el SUBCONJUNTO pedido, no el registro en disco -- "
        "esta misma prosa afirmaba lo segundo, que es lo que el mensaje decia "
        f"en falso y `..._por_el_camino_real` ahora no deja pasar: {mensaje!r}"
    )
    assert str(len(config.VERDICT_TRANSFERS)) in mensaje, (
        "tiene que nombrar cuántas transferencias son en total, para que quien "
        f"la lea sepa qué le falta: {mensaje!r}"
    )

    # La otra dirección: sin `transfers` el comportamiento de hoy no cambia.
    assert harness.run_search() is registro, (
        "la búsqueda entera tiene que seguir devolviendo su registro; un guard "
        "que se dispara siempre no es un guard, es una rotura"
    )


# --------------------------------------------- el guard, por el camino real

def _registro_en_disco(path) -> None:
    """Un registro de techos con la procedencia CURRENT, como el de
    `test_search_records.py` y por la misma razón: sin el sello,
    `ceilings_in_force` lo rechazaría por un motivo ajeno a lo que se mide acá.
    """
    import json
    path.write_text(json.dumps({"creda": {
        "ceiling": 0.01, "epochs": config.SEARCH_EPOCHS,
        "seeds": list(config.SEARCH_SEEDS),
        "atRequiredScale": True,
        "requiredScale": {"epochs": config.FULL_SEARCH_EPOCHS,
                          "seeds": config.FULL_SEARCH_SEEDS},
        "byTransfer": {"M->U": 0.01},
        "revision": config.REVISION, "kernelSigma": config.KERNEL_SIGMA,
        "attentionGamma": config.ATTENTION_GAMMA,
        "attentionTemperature": config.ATTENTION_TEMPERATURE}}),
        encoding="utf-8")


@pytest.fixture
def arbol(tmp_path, monkeypatch):
    """Las dos rutas de registro dentro de `tmp_path`, como el fixture que
    `test_search_records.py` ya usa. Se redirige `RESULTS` además de las dos
    constantes porque `ceilings_record_for` las resuelve desde ahí.
    """
    lleno = tmp_path / "ceilings.json"
    ensayo = tmp_path / "ceilings.pilot.json"
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", lleno)
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", ensayo)
    return lleno, ensayo


@pytest.mark.parametrize("registro_presente", [False, True],
                         ids=["sin-registro", "con-registro"])
def test_el_rechazo_del_subconjunto_llega_por_el_camino_real(
        arbol, monkeypatch, registro_presente) -> None:
    """La prueba que el test anterior NO era, y la diferencia es el método.

    `test_a_partial_search_refuses_to_present_its_slice_as_the_record` tapa
    `ceilings_in_force` Y `search_record`, o sea los dos predecesores del guard.
    Prueba que la línea se ejecuta; no puede ver lo que la línea AFIRMA, porque
    el orden real de las ramas nunca corre. Y el mensaje afirmaba dos cosas
    falsas por eso mismo.

    Acá se tapa SOLO `search_ceilings` ---lo único que gastaría una búsqueda
    optuna de verdad--- y corren `ceilings_in_force` y `search_record` reales,
    contra un árbol de verdad. Las dos ramas que existen se recorren las dos:

    * `sin-registro`: `ceilings_in_force` no ve registro y busca. La escritura
      de un subconjunto está bloqueada aguas arriba, así que no queda nada en
      disco. Antes de este arreglo, acá ganaba "left no record".
    * `con-registro`: `ceilings_in_force` corta y NO busca. Antes de este
      arreglo, el guard afirmaba que la búsqueda había corrido y había dejado un
      registro parcial en la ruta canónica ---falso dos veces, sobre un registro
      completo escrito por otra cosa---.

    En las dos gana el rechazo del subconjunto, y en ninguna el mensaje afirma
    qué pasó aguas arriba, porque desde ese frame no se puede saber.
    """
    lleno, _ = arbol
    if registro_presente:
        _registro_en_disco(lleno)

    busco = {"llamada": False}

    def _sin_buscar(*a, **k):
        busco["llamada"] = True
        return {}

    monkeypatch.setattr(harness, "search_ceilings", _sin_buscar)
    monkeypatch.setattr(harness, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(harness, "environment", lambda: {})

    with pytest.raises(SystemExit) as refusal:
        harness.run_search(transfers=[config.SEARCH_TRANSFERS[0]])
    mensaje = str(refusal.value)

    assert busco["llamada"] == (not registro_presente), (
        "la rama de arriba tiene que ser la real: busca cuando NO hay registro y "
        "corta cuando hay. Si esto falla, el test no est\u00e1 recorriendo el camino "
        f"que dice recorrer (registro={registro_presente}, busc\u00f3={busco['llamada']})"
    )
    assert "subset" in mensaje, (
        f"tiene que ganar el rechazo del subconjunto, no otro: {mensaje!r}"
    )
    assert "left no record" not in mensaje, (
        "'left no record' describe un registro ausente, que no es lo que pasó: "
        f"se pidió un subconjunto. {mensaje!r}"
    )
    assert "the search ran" not in mensaje, (
        "el mensaje no puede afirmar que la búsqueda corrió: con un registro en "
        f"disco `ceilings_in_force` corta y no busca. {mensaje!r}"
    )
    assert str(len(config.SEARCH_TRANSFERS)) in mensaje, (
        f"tiene que nombrar el total contra el que el subconjunto es parcial: {mensaje!r}"
    )
