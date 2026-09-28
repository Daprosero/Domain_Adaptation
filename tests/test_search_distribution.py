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
