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

import copy
import inspect
import json

import pytest

import MIL_CREDA_Benchmark
from MIL_CREDA_Benchmark import config, harness, shards


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

    # La inversión que motiva la capacidad entera: los tres campos que el eje
    # semilla exige idénticos son exactamente los que el eje transferencia
    # permite -- necesita -- que difieran, porque cada shard de búsqueda
    # sostiene una porción DISTINTA de exactamente esos campos.
    seed_identical = axes["seed"]["identicalAcrossShards"]
    transfer_identical = axes["transfer"]["identicalAcrossShards"]
    for field in ("ceilings", "ceilingsByTransfer", "hyperByTransfer"):
        assert field in seed_identical, (
            f"el eje semilla tiene que seguir exigiendo `{field}` idéntico "
            f"entre shards de campaña; hoy {seed_identical!r}"
        )
        assert field not in transfer_identical, (
            f"el eje transferencia no puede exigir `{field}` idéntico entre "
            "shards: cada shard de búsqueda sostiene una porción distinta de "
            "ese campo a propósito, y pedirles que coincidan es pedirles que "
            f"no hayan hecho su trabajo; hoy {transfer_identical!r}"
        )

    # Todo eje declarado tiene los cuatro grupos, nunca un hueco -- un grupo
    # ausente leído como `[]` haría que `partition()` reportara cada
    # dimensión `unpartitioned` y un merge la descartara en silencio.
    for axis_name, block in axes.items():
        for group in ("poolable", "perEnvironment", "perRun", "identicalAcrossShards"):
            assert group in block, (
                f"el eje `{axis_name}` no declara `{group}`; un lector "
                "genérico no puede resolver sus reglas sin ese hueco"
            )
            assert isinstance(block[group], list), (
                f"`{axis_name}`.`{group}` tiene que ser una lista, posiblemente "
                f"vacía, pero nunca `None` ni faltante; hoy es {block[group]!r}"
            )


def test_the_flat_keys_are_the_seed_axis_by_identity() -> None:
    """La proyección es por identidad, nunca por copia.

    Dos estructuras que sólo sostienen listas iguales son dos copias, y una
    igualdad que hoy se cumple es exactamente cómo dos copias empiezan a
    discrepar. Se afirma con `is`, explícitamente nunca con `==`: una prueba
    con `==` pasaría contra una copia construida aparte y no mediría nada de
    lo que esta decisión afirma.
    """
    distribution = MIL_CREDA_Benchmark.__benchmark__["distribution"]
    seed = distribution["axes"]["seed"]
    for group in ("identicalAcrossShards", "poolable", "perEnvironment", "perRun"):
        assert distribution[group] is seed[group], (
            f"`distribution[{group!r}]` no es el MISMO objeto que "
            f"`distribution['axes']['seed'][{group!r}]`; la proyección tiene "
            "que ser por identidad, no una copia que hoy resulta igual"
        )


def test_the_forges_arm_split_refusal_remains_reachable() -> None:
    """Nada en esta capacidad puede costarle a la forja su capacidad de negar
    el reparto por brazo.

    `implementation_engine.py:1603` lee `dist.get("axis") == "arm"`. Sobre la
    declaración real tiene que seguir evaluando `False`; sobre una asignada a
    mano con `axis: "arm"` tiene que seguir pudiendo evaluar `True` -- si la
    comparación degradara a comparar contra `None` bajo la forma anidada,
    quedaría permanentemente `False` y el rechazo nunca podría dispararse.
    """
    distribution = MIL_CREDA_Benchmark.__benchmark__["distribution"]
    assert (distribution.get("axis") == "arm") is False, (
        "sobre la declaración real, la comparación de la forja tiene que "
        f"evaluar False; hoy `axis` es {distribution.get('axis')!r}"
    )
    hand_built = {"axis": "arm"}
    assert (hand_built.get("axis") == "arm") is True, (
        "la comparación tiene que seguir siendo capaz de evaluar True sobre "
        "un mapeo que de verdad declara `axis: \"arm\"` -- de lo contrario "
        "está leyendo una constante, no un campo vivo"
    )


def test_shards_root_survives_the_axes_nesting_unchanged() -> None:
    """Una clave que nada en la suite de hoy notaría perder si se cayera en
    silencio, así que se fija explícitamente en vez de confiar en que algún
    otro test la proteja por accidente."""
    distribution = MIL_CREDA_Benchmark.__benchmark__["distribution"]
    assert distribution["shardsRoot"] == "MIL-CREDA/Results/Benchmark/shards"


def test_no_docstring_still_claims_the_axis_is_singular() -> None:
    """Ninguna prosa puede seguir jurando que no hay un segundo eje.

    Tres lugares apoyaban parte de su argumento en la singularidad de
    `axis`: el docstring de `config.execution_seed_units`, el de
    `harness.run_campaign_shard`, y el bloque de razones que acompaña a
    `distribution` en `__init__.py`. El campo sobrevive (`axis` sigue
    leyendo `"seed"`); la justificación no puede, porque deja de ser cierta
    en cuanto `axes` nombra dos valores.

    `config.execution_seed_units` se corrige recién en la Unidad de Trabajo
    A2, cuando la función misma cambia (Tasca 1.14's nota explícita); este
    test se escribe y se observa en rojo ahora, y se espera que la parte de
    `execution_seed_units` siga en rojo hasta que A2 aterrice -- no es una
    regresión, es la parte todavía no hecha.
    """
    import inspect as _inspect

    import MIL_CREDA_Benchmark as _package

    markers = (
        "no hay un segundo eje",
        "is what the declaration says may be split",
        "no second axis",
        "único eje",
        "unico eje",
    )

    module_source = _inspect.getsource(_package)
    start = module_source.index("# `shards.declaration()` reads this block")
    end = module_source.index('"distribution": {')
    rationale_block = module_source[start:end]

    sources = {
        "config.execution_seed_units.__doc__": config.execution_seed_units.__doc__ or "",
        "harness.run_campaign_shard.__doc__": harness.run_campaign_shard.__doc__ or "",
        "the __init__.py rationale block beside `distribution`": rationale_block,
    }
    for label, text in sources.items():
        lowered = text.lower()
        hits = [marker for marker in markers if marker.lower() in lowered]
        assert not hits, (
            f"{label} todavía afirma que el eje es singular ({hits!r}); esa "
            "cita es falsa en cuanto `axes` nombra `seed` y `transfer`"
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


def test_partial_search_records_merge_instead_of_overwriting(_seis_rebanadas) -> None:
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

    Crecido más allá de la firma (D9/D10/D11): seis rebanadas reales, escritas
    por el motor de verdad, tienen que ensamblar en UN registro que trae cada
    campo de un registro entero -- comparado contra un registro entero de
    verdad, construido por el MISMO motor falso, no contra un roster tipeado a
    mano -- más la nueva clave `assembly`, y el registro canónico tiene que
    quedar escrito porque las seis llegaron.
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

    rebanadas = _seis_rebanadas
    reduccion_entera = harness.Reduction(seeds=[config.SEARCH_SEED],
                                         epochs=config.SEARCH_EPOCHS)
    harness.search_ceilings_trials(reduccion_entera, "cpu",
                                   progress=lambda *_: None, shard=None,
                                   transfers=None)
    registro_entero = json.loads(
        config.ceilings_record_for(False).read_text())["milcreda"]
    # El registro que la búsqueda entera acaba de escribir se borra: lo que se
    # mide es que `merge_search_records` sea quien lo vuelva a escribir a
    # partir de las seis rebanadas, no que lo encuentre ya ahí.
    config.ceilings_record_for(False).unlink()

    ensamblado = harness.merge_search_records(rebanadas)
    assert set(ensamblado) == {"milcreda"}
    entrada = ensamblado["milcreda"]

    faltan = set(registro_entero) - set(entrada)
    assert not faltan, (
        f"al ensamblado le faltan campos que un registro entero SÍ trae: {faltan}")
    sobran = set(entrada) - set(registro_entero) - {"assembly"}
    assert not sobran, (
        "el ensamblado trae campos que ni un registro entero ni `assembly` "
        f"explican: {sobran}")

    etiquetas = [config.transfer_label(t) for t in config.SEARCH_TRANSFERS]
    assert entrada["transfers"] == etiquetas
    assert set(entrada["perTransfer"]) == set(etiquetas)
    assert entrada["assembly"]["expected"] == etiquetas, (
        "assembly.expected tiene que nombrar el mismo conjunto esperado que "
        "decidió la completitud"
    )
    assert {s["id"] for s in entrada["assembly"]["shards"]} == {
        r["milcreda"]["shard"] for r in rebanadas}, (
        "assembly.shards tiene que nombrar cada rebanada que contribuyó, por "
        "su propio id -- nunca uno solo hablando por las seis"
    )

    # El re-agrupado (D9) produce el MISMO número que una búsqueda entera sin
    # partir: las seis semillas de estudio son deterministas por (familia,
    # etiqueta), así que dividir y reensamblar no puede mover el resultado.
    assert entrada["ceiling"] == registro_entero["ceiling"], (
        "el techo ensamblado tiene que coincidir con el de una búsqueda "
        f"entera sobre el mismo material: {entrada['ceiling']} vs "
        f"{registro_entero['ceiling']}"
    )
    assert entrada["byTransfer"] == registro_entero["byTransfer"]

    assert config.ceilings_record_for(False).exists(), (
        "un ensamblado completo y limpio (D7/D9) tiene que escribir el "
        "registro canónico"
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


# --------------------------------------------- A3: el artefacto por shard

def test_the_slice_artifact_lives_under_shard_paths() -> None:
    """`shard_paths(...)["searchSlice"]` es la cuarta clave, no un lugar nuevo.

    Un shard con id va bajo su propia carpeta, exactamente donde ya viven
    `runs.jsonl` y `shard.json`; `shard=None` cae al lado del registro
    canónico, nunca adentro de él. `pilot` y `noise` la mueven porque mueven
    `results_for(...)` entero -- la ruta de la rebanada no puede quedarse
    atrás de la del registro que rebana.
    """
    con_id = harness.shard_paths("s03")
    esperado = (config.results_for(0.0, "campaign", False) / "shards" / "s03"
                / "ceilings.slice.json")
    assert con_id["searchSlice"] == esperado, (
        "la rebanada de un shard con id tiene que vivir bajo su carpeta, "
        f"exactamente donde `runs`/`stamp` ya viven; hoy "
        f"{con_id['searchSlice']} y se esperaba {esperado}"
    )

    sin_id = harness.shard_paths(None)
    registro = config.ceilings_record_for(False)
    assert sin_id["searchSlice"] == registro.with_name(
        registro.stem + ".slice.json"
    ), (
        "sin id la rebanada tiene que caer al lado del registro canónico, "
        f"nunca adentro de él; hoy {sin_id['searchSlice']}"
    )

    pilot_id = harness.shard_paths("s03", pilot=True)["searchSlice"]
    assert pilot_id != con_id["searchSlice"], (
        "un shard de ensayo no puede compartir ruta con el shard de la "
        f"campaña completa; hoy los dos son {pilot_id}"
    )

    ruidoso = harness.shard_paths("s03", noise=0.4)["searchSlice"]
    assert ruidoso != con_id["searchSlice"], (
        "un shard contaminado no puede compartir ruta con el limpio; hoy "
        f"los dos son {ruidoso}"
    )


@pytest.fixture
def _rebanada_engine(tmp_path, monkeypatch):
    """El motor por trials con `run_one` falso, la misma forma que
    `tests/test_search_declarations.py:estudios` -- se mide el escritor de
    la rebanada, nunca el modelo."""
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")
    monkeypatch.setattr(config, "SEARCH_ENGINE", "optuna")
    monkeypatch.setattr(config, "PILOT_SEARCH_TRIALS", 3)
    monkeypatch.setattr(config, "SEARCH_TRIALS", 3)

    class _Bags:
        pass

    def _build(code, cache, seed, noise=0.0):
        return _Bags()

    def _run_one(arm, transfer, seed, reduction, device, material, *, ceiling,
                role, hyper=None):
        return {config.SEARCH_CRITERION: 1.0 - abs(ceiling - 0.01)}

    monkeypatch.setattr(harness.bags, "build", _build)
    monkeypatch.setattr(harness, "run_one", _run_one)


def test_a_subset_search_leaves_a_readable_slice_and_the_canonical_path_untouched(
        _rebanada_engine) -> None:
    """La integración que D7 promete: un subconjunto deja una rebanada legible
    y el registro canónico queda intacto.

    Conducida con `run_one` falso, la forma que `tests/test_search_declarations
    .py`'s `estudios` fixture ya usa: se mide el escritor, nunca el modelo.
    """
    subconjunto = list(config.SEARCH_TRANSFERS[:1])
    encontrado = harness.search_ceilings_trials(
        harness.Reduction(seeds=[config.SEARCH_SEED], epochs=config.SEARCH_EPOCHS),
        "cpu", progress=lambda *_: None, shard="search-s00",
        transfers=subconjunto)

    slice_path = harness.shard_paths(
        "search-s00", noise=0.0, kind="campaign", pilot=False)["searchSlice"]
    assert slice_path.exists(), (
        "un subconjunto tiene que dejar una rebanada legible donde antes no "
        f"dejaba nada; {slice_path} no existe"
    )
    sellado = json.loads(slice_path.read_text())
    assert set(sellado) == set(encontrado), (
        "la rebanada en disco tiene que sostener las mismas familias que "
        f"`found` devolvió: {set(sellado)!r} vs {set(encontrado)!r}"
    )
    assert not config.ceilings_record_for(False).exists(), (
        "un subconjunto no gobierna el registro canónico (D7) y no puede "
        "dejar nada ahí"
    )


def test_the_slice_sealer_never_touches_the_governed_roster() -> None:
    """`sellar_rebanada` no escribe nada ni nombra el path canónico.

    La derivación del roster (`tests/test_label_noise.py:1253-1274`) marca
    cualquier función de `harness` cuyo código nombre `ceilings_record_for`
    Y `write_text` como escritora del registro gobernado. Afirmado acá de
    forma directa -- no observando que el roster no creció, que un tamaño
    constante sería igual de consistente con `sellar_rebanada` quedando mal
    excluida.
    """
    fuente = inspect.getsource(harness.sellar_rebanada)
    assert "ceilings_record_for" not in fuente, (
        "sellar_rebanada solo sella una entrada; si nombrara el path "
        f"canónico, la derivación del roster lo confundiría con un tercer "
        f"escritor: {fuente!r}"
    )
    assert "write_text" not in fuente, (
        "sellar_rebanada no escribe nada -- el llamador escribe con el path "
        "que devuelve -- así que tampoco puede entrar al roster por "
        f"`write_text`: {fuente!r}"
    )


# --------------------------------------------- A4: la unión de rebanadas

def _buscar_una_rebanada(transfer, shard, epochs=None) -> dict:
    """Corre el motor real sobre UNA sola transferencia y devuelve la
    rebanada que `sellar_rebanada` selló, leída de vuelta del disco -- nunca
    armada a mano, para que cada campo que un lector real espera esté
    presente con su forma real.
    """
    reduccion = harness.Reduction(
        seeds=[config.SEARCH_SEED],
        epochs=epochs if epochs is not None else config.SEARCH_EPOCHS)
    harness.search_ceilings_trials(
        reduccion, "cpu", progress=lambda *_: None, shard=shard,
        transfers=[transfer])
    slice_path = harness.shard_paths(shard)["searchSlice"]
    return json.loads(slice_path.read_text())


@pytest.fixture
def _par_rebanadas(_rebanada_engine) -> list[dict]:
    """Dos rebanadas reales, una por cada una de las dos primeras
    transferencias declaradas -- la base compartida para cada prueba de
    negativa de D10 que necesita un único defecto aislado."""
    return [
        _buscar_una_rebanada(config.SEARCH_TRANSFERS[0], "search-s00"),
        _buscar_una_rebanada(config.SEARCH_TRANSFERS[1], "search-s01"),
    ]


@pytest.fixture
def _seis_rebanadas(_rebanada_engine) -> list[dict]:
    """Las seis rebanadas reales, una por transferencia de
    `config.SEARCH_TRANSFERS` -- la base compartida para las pruebas de
    cobertura completa, incompleta y de orden."""
    return [
        _buscar_una_rebanada(t, f"search-s{i:02d}")
        for i, t in enumerate(config.SEARCH_TRANSFERS)
    ]


def test_the_assembled_ceiling_is_repooled_over_the_union_not_copied_forward(
        _seis_rebanadas) -> None:
    """D9's re-pool mutation test: six single-transfer slices, each with its
    OWN distinct ceiling, and none of them equal to the union's winner --
    which is exactly what distinguishes a re-pool from a dict-merge that
    copies one slice's pooled field forward.
    """
    rebanadas = copy.deepcopy(_seis_rebanadas)
    techos = [0.05, 0.03, 0.01, 0.07, 0.02, 0.09]
    for rebanada, techo in zip(rebanadas, techos):
        entrada = rebanada["milcreda"]
        etiqueta = entrada["transfers"][0]
        pick = entrada["perTransfer"][etiqueta]
        pick["ceiling"] = techo
        # El mismo `best` en las seis: la meseta a nivel de familia las cubre
        # a todas, y el desempate lo decide únicamente el techo más chico.
        pick["best"] = 1.0
        entrada["ceiling"] = techo
        entrada["byTransfer"][etiqueta] = techo

    ensamblado = harness.merge_search_records(rebanadas)
    ganador = min(techos)
    assert ensamblado["milcreda"]["ceiling"] == ganador, (
        "el techo ensamblado tiene que ser el que gana la meseta sobre la "
        f"UNIÓN de las seis rebanadas ({ganador}), no el de ninguna rebanada "
        f"copiado hacia adelante: {ensamblado['milcreda']['ceiling']}"
    )
    assert ensamblado["milcreda"]["ceiling"] not in (
        rebanadas[0]["milcreda"]["ceiling"], rebanadas[-1]["milcreda"]["ceiling"]
    ), (
        "un dict-merge que copie la primera o la última rebanada hacia "
        "adelante tendría que fallar esta prueba: el ganador real no es ni "
        "el primero ni el último de la lista tal como se entregó"
    )


def test_a_slice_with_no_declared_coverage_refuses(_par_rebanadas) -> None:
    par = copy.deepcopy(_par_rebanadas)
    par[0]["milcreda"]["transfers"] = []
    with pytest.raises(shards.ShardIncomplete) as rechazo:
        harness.merge_search_records(par)
    mensaje = str(rechazo.value)
    assert par[0]["milcreda"]["shard"] in mensaje, (
        f"la negativa tiene que nombrar el id de la rebanada sin cobertura: {mensaje!r}")


def test_a_slice_stamped_under_a_different_revision_refuses(_par_rebanadas) -> None:
    par = copy.deepcopy(_par_rebanadas)
    par[1]["milcreda"]["revision"] = "una-revision-vieja"
    with pytest.raises(shards.ShardsDisagree) as rechazo:
        harness.merge_search_records(par)
    mensaje = str(rechazo.value)
    assert par[1]["milcreda"]["shard"] in mensaje, (
        f"la negativa tiene que nombrar el id de la rebanada corrida: {mensaje!r}")
    assert "revision" in mensaje


def test_slices_of_different_search_kinds_refuse(_par_rebanadas) -> None:
    par = copy.deepcopy(_par_rebanadas)
    del par[1]["milcreda"]["search"]
    par[1]["milcreda"]["grid"] = {}
    with pytest.raises(shards.ShardsDisagree) as rechazo:
        harness.merge_search_records(par)
    mensaje = str(rechazo.value)
    assert "optuna" in mensaje and "grid" in mensaje, (
        f"la negativa tiene que nombrar las dos formas en conflicto: {mensaje!r}")


def test_two_slices_claiming_the_same_transfer_refuse(_par_rebanadas) -> None:
    par = copy.deepcopy(_par_rebanadas)
    etiqueta = par[0]["milcreda"]["transfers"][0]
    entrada = par[1]["milcreda"]
    otra = entrada["transfers"][0]
    entrada["transfers"] = [etiqueta]
    entrada["byTransfer"] = {etiqueta: entrada["byTransfer"].pop(otra)}
    entrada["perTransfer"] = {etiqueta: entrada["perTransfer"].pop(otra)}
    with pytest.raises(shards.ShardsDisagree) as rechazo:
        harness.merge_search_records(par)
    mensaje = str(rechazo.value)
    assert etiqueta in mensaje
    assert par[0]["milcreda"]["shard"] in mensaje
    assert par[1]["milcreda"]["shard"] in mensaje


def test_slices_disagreeing_on_labelnoise_refuse_and_name_both_values(
        _par_rebanadas) -> None:
    par = copy.deepcopy(_par_rebanadas)
    par[0]["milcreda"]["labelNoise"] = 0.0
    par[1]["milcreda"]["labelNoise"] = 0.4
    with pytest.raises(shards.ShardsDisagree) as rechazo:
        harness.merge_search_records(par)
    mensaje = str(rechazo.value)
    assert "labelNoise" in mensaje
    assert "0.4" in mensaje


def test_agreeing_slices_do_not_raise_the_disagreement_refusal(
        _par_rebanadas) -> None:
    """La otra dirección de D10#5: acordando en todo lo declarado idéntico,
    dos rebanadas no disparan esta negativa."""
    harness.merge_search_records(_par_rebanadas,
                                 expected=list(config.SEARCH_TRANSFERS[:2]))


def test_a_slice_covering_a_label_outside_expected_refuses(_par_rebanadas) -> None:
    par = copy.deepcopy(_par_rebanadas)
    solo_la_primera = [config.SEARCH_TRANSFERS[0]]
    segunda_etiqueta = par[1]["milcreda"]["transfers"][0]
    with pytest.raises(shards.ShardsDisagree) as rechazo:
        harness.merge_search_records(par, expected=solo_la_primera)
    mensaje = str(rechazo.value)
    assert segunda_etiqueta in mensaje, (
        f"la negativa tiene que nombrar la etiqueta inesperada: {mensaje!r}")


def test_slices_covering_different_family_sets_refuse(_par_rebanadas) -> None:
    par = copy.deepcopy(_par_rebanadas)
    par[0]["phantom"] = copy.deepcopy(par[0]["milcreda"])
    with pytest.raises(shards.ShardsDisagree) as rechazo:
        harness.merge_search_records(par)
    mensaje = str(rechazo.value)
    assert "milcreda" in mensaje and "phantom" in mensaje, (
        f"la negativa tiene que nombrar los dos conjuntos de familias: {mensaje!r}")


def test_five_of_six_refuses_and_names_the_sixth(_seis_rebanadas) -> None:
    faltante = _seis_rebanadas[-1]["milcreda"]["transfers"][0]
    cinco = _seis_rebanadas[:-1]
    with pytest.raises(shards.ShardIncomplete) as rechazo:
        harness.merge_search_records(cinco)
    mensaje = str(rechazo.value)
    assert faltante in mensaje, (
        "la negativa tiene que nombrar la transferencia ausente ESPECÍFICA, "
        f"no solo un conteo: {mensaje!r}")


def test_a_complete_set_never_raises_the_missing_transfer_refusal(
        _seis_rebanadas) -> None:
    """La otra dirección de D10#6: seis de seis no dispara esta negativa."""
    harness.merge_search_records(_seis_rebanadas)


def test_individually_complete_slices_still_refuse_on_expected_coverage(
        _seis_rebanadas) -> None:
    """Coverage completeness is judged against the expected set, not against
    what was handed in: tres rebanadas, cada una individualmente sellada y
    completa, igual se niegan citando las tres que faltan."""
    tres = _seis_rebanadas[:3]
    faltantes = {r["milcreda"]["transfers"][0] for r in _seis_rebanadas[3:]}
    with pytest.raises(shards.ShardIncomplete) as rechazo:
        harness.merge_search_records(tres)
    mensaje = str(rechazo.value)
    for etiqueta in faltantes:
        assert etiqueta in mensaje, (
            f"la negativa tiene que nombrar las tres transferencias ausentes: {mensaje!r}")


def test_an_explicit_smaller_expected_set_assembles_without_writing_canonical(
        _seis_rebanadas) -> None:
    una = _seis_rebanadas[:1]
    solicitada = [config.SEARCH_TRANSFERS[0]]
    ensamblado = harness.merge_search_records(una, expected=solicitada)
    assert set(ensamblado) == {"milcreda"}
    assert ensamblado["milcreda"]["transfers"] == [
        config.transfer_label(config.SEARCH_TRANSFERS[0])]
    assert not config.ceilings_record_for(False).exists(), (
        "un `expected` explícito más chico tiene que ensamblar y NO tocar el "
        "registro canónico -- eso es lo que lo distingue de omitir el "
        "argumento, que jamás se alcanza así"
    )


def test_expected_order_does_not_decide_the_guard(_seis_rebanadas) -> None:
    al_reves = list(reversed(config.SEARCH_TRANSFERS))
    harness.merge_search_records(_seis_rebanadas, expected=al_reves)
    assert config.ceilings_record_for(False).exists(), (
        "el orden de `expected` no puede decidir el guardián: las seis "
        "llegan igual, en cualquier orden que se pidan"
    )


def test_the_assembly_never_reads_verdict_transfers_as_its_default() -> None:
    """La lectura a nivel de FUENTE, no solo de conteo: las dos listas
    comparten longitud hoy, así que una prueba que solo contara no
    distinguiría una fuente divergente."""
    fuente = inspect.getsource(harness.merge_search_records)
    assert "VERDICT_TRANSFERS" not in fuente, (
        "la unión tiene que derivar su expectativa por defecto de "
        f"`SEARCH_TRANSFERS` únicamente, nunca de `VERDICT_TRANSFERS`: {fuente!r}")


def test_the_canonical_write_consults_the_guard(_seis_rebanadas, monkeypatch) -> None:
    llamadas = []
    original = harness.governs_the_ceilings_record

    def _observado(*a, **k):
        llamadas.append((a, k))
        return original(*a, **k)

    monkeypatch.setattr(harness, "governs_the_ceilings_record", _observado)
    harness.merge_search_records(_seis_rebanadas)
    assert llamadas, (
        "un ensamblado completo y limpio tiene que CONSULTAR el guardián, no "
        "solo pasar por casualidad"
    )


def test_an_incomplete_assembly_never_attempts_the_canonical_write(
        _seis_rebanadas) -> None:
    cinco = _seis_rebanadas[:-1]
    with pytest.raises(shards.ShardIncomplete):
        harness.merge_search_records(cinco)
    assert not config.ceilings_record_for(False).exists(), (
        "un rechazo por cobertura incompleta no puede dejar nada escrito en "
        "el registro canónico"
    )


def test_merge_search_records_enters_the_guarded_writer_roster() -> None:
    """El compañero explícito de `test_the_slice_sealer_never_touches_the_
    governed_roster`: `merge_search_records` SÍ tiene que entrar al roster
    que deriva `tests/test_label_noise.py:1253-1274`, y consultar el
    guardián -- no porque el roster no haya crecido, sino porque su propia
    fuente nombra las tres piezas."""
    fuente = inspect.getsource(harness.merge_search_records)
    assert "ceilings_record_for" in fuente
    assert "write_text" in fuente
    assert "governs_the_ceilings_record" in fuente
