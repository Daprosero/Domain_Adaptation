"""Los pasos locales que la forja puede ejecutar por su cuenta.

Cada función de acá ejecuta un cuaderno de este repositorio y nada más. La
forja las nombra estáticamente en `__steps__` (módulo + función), las resuelve
sin importarlas, y recién las importa y llama dentro del intérprete de este
repositorio.

**Por qué no se escribe acá el prefijo de `PATH`.** El kernelspec que graban
los cuadernos arranca un `python` pelado, sin ruta absoluta, así que resuelve
contra el `PATH` del proceso que lo lanza. Componer ese `PATH` es trabajo de la
forja, que ya antepone `<repo>/.venv/bin` antes de llamar a estas funciones. Si
cada autor lo escribiera de nuevo acá, la corrección dependería de que ninguno
se olvide -- y ya se olvidó una vez, con quince fallas fantasma contra una
suite que pasa 297/297.

`sys.executable` y nunca `python3`: adentro de este proceso ya es el intérprete
de este repositorio, y nombrarlo directamente no depende de que el `PATH` que
lo rodea sea el correcto.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

CUADERNOS = Path(__file__).resolve().parents[2] / "MIL-CREDA" / "Notebooks"

# Los cuadernos que este módulo NO corre, cada uno con su razón al lado.
#
# Un nombre pelado en una lista y un olvido se leen igual, y ese fue el defecto:
# `Benchmark_Noise_Diagnostic_v1.ipynb` estuvo cinco celdas sin ejecutar ni una
# sola, computado y sin dibujar, sin que nada lo notara. La prueba deriva los
# cuadernos que sí se corren del código y le resta el disco; lo que sobra tiene
# que estar acá, y estar acá cuesta escribir por qué. El próximo que se excluya
# tiene que decir lo suyo.
CUADERNOS_SIN_PASO: dict[str, str] = {
    "Benchmark_Campaign_v1.ipynb": (
        "lanza al servicio remoto, y un envío necesita una aprobación humana "
        "por lanzamiento que ningún paso local puede darle"),
}


def _ejecutar(nombre: str) -> str:
    """Ejecuta un cuaderno en el lugar y devuelve su ruta.

    `--inplace` a propósito: la salida ejecutada ES el informe, y un cuaderno
    que se ejecuta a un archivo nuevo deja el original en el árbol diciendo
    algo distinto. Sin `--allow-errors`, así que una celda que falla corta acá
    en lugar de dejar un informe que parece completo.
    """
    cuaderno = CUADERNOS / nombre
    if not cuaderno.is_file():
        raise FileNotFoundError(f"no existe el cuaderno {cuaderno}")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert",
         "--to", "notebook", "--execute", "--inplace", str(cuaderno)],
        check=True)
    return str(cuaderno)


def verificacion() -> str:
    """Nivel 1: la suite corre adentro del cuaderno y el sello queda actual."""
    return _ejecutar("verification.ipynb")


def ensayo_de_busqueda() -> dict:
    """Corre la búsqueda de techos a escala de ensayo y devuelve su registro.

    `pilot=True` y sin manera de pedir otra cosa. La búsqueda a escala completa
    es una decisión con su propia autorización, y un paso local que aceptara un
    argumento para elegirla dejaría esa decisión en manos de quien escribe la
    invocación. Escribe `ceilings.pilot.json`, que el registro completo siempre
    le gana cuando existe.

    Existe porque no existía: la única forma de correr esta búsqueda era un
    `python -c` a mano contra `harness`, por fuera de la skill y sin dejar
    rastro en el ledger. Un paso que falta no es disciplina que falta.
    """
    from MIL_CREDA_Benchmark import harness

    return harness.run_search(pilot=True)


def campana() -> dict:
    """Corre la campaña en esta máquina, a la escala que declara `config`.

    Arma la reducción igual que el cuaderno de campaña -- dispositivo,
    ambiente, y los techos vigentes -- y llama a `harness.campaign()`. Lo que
    la vuelve un ensayo no es un parámetro de acá sino `config`: las épocas y
    las semillas que el repositorio declara. Este paso no las elige, y por eso
    no toma argumentos.

    No hay que ordenar nada acá: `campaign()` se niega sola sin un registro de
    techos buscado, así que la búsqueda va antes por construcción y no por
    convención.

    Esto NO envía nada al servicio remoto. `Benchmark_Campaign_v1.ipynb` sigue
    fuera de `__steps__` por eso mismo: un envío necesita una aprobación humana
    por lanzamiento, y ningún paso local puede dársela.
    """
    from MIL_CREDA_Benchmark import harness

    from MIL_CREDA_Benchmark import config

    dispositivo = harness.resolve_device()
    # `with_ceilings_in_force` era lo que estaba acá y es una trampa: sin
    # `ceilings.json` lanza la búsqueda COMPLETA --- 2 familias x 6
    # transferencias x 30 trials a 20 épocas, unas nueve horas y media --- sin
    # anunciarlo y sin que nadie la haya autorizado. Un paso local no puede ser
    # la puerta por la que entra la corrida larga.
    if harness.search_record() is None and harness.search_record(pilot=True) is None:
        raise SystemExit(
            "no hay registro de techos. Corré primero la búsqueda: una campaña "
            "sin techos mide el método y la falta de coeficiente a la vez.")
    reduccion = replace(
        harness.Reduction(device=str(dispositivo),
                          environment=harness.environment(), pilot=True),
        ceilings=config.ceilings_on_record(),
        ceilingsByTransfer=config.ceilings_by_transfer_on_record())
    return harness.campaign(reduccion, dispositivo)


def informe_de_busqueda() -> str:
    """El INFORME de la búsqueda de techos, no la búsqueda.

    El cuaderno lee un registro que ya existe (`harness.search_record()`) y lo
    presenta; la llamada que corre la búsqueda está comentada adentro, a
    propósito -- un cuaderno de informe que además ejecutara horas de cómputo
    haría que abrirlo cueste lo que cuesta correrlo.

    El nombre dice cuál de las dos cosas es. La primera versión de esta función
    se llamaba `busqueda` y prometía en su docstring "la búsqueda de techos",
    que es exactamente lo que NO hace: un lector la habría llamado esperando un
    registro nuevo y habría recibido el viejo, presentado.
    """
    return _ejecutar("Benchmark_Search_v1.ipynb")


def informe() -> str:
    """Las tablas y conclusiones sobre el registro fusionado."""
    return _ejecutar("Benchmark_Report_v1.ipynb")


def latente() -> str:
    """El análisis latente sobre los checkpoints promovidos."""
    return _ejecutar("Benchmark_Latent_v1.ipynb")


def diagnostico_de_ruido() -> dict:
    """Re-busca el techo al nivel de diagnóstico y corre ahí los dos completos.

    El experimento que separa *falló el término* de *le faltó coeficiente*, y
    que existe porque los techos de la campaña se buscaron en limpio y se
    mantienen fijos en los cinco niveles. Esa decisión hace que una caída no se
    pueda atribuir, y este paso es lo barato que la atribuye.

    Necesita tres puntos y paga uno solo. Los dos brazos a este nivel bajo el
    techo limpio ya salen de la campaña contaminada; lo que se corre acá es la
    búsqueda al nivel de diagnóstico y los dos brazos debajo de ella. Si el
    techo re-buscado recupera lo perdido, fue el coeficiente; si no lo
    recupera, fue el término.

    `D` y `G`, y nadie más: son los dos métodos completos, uno por familia, y
    los únicos que llevan el coeficiente. `A` y `B` no tienen término de
    adaptación al que re-buscarle un techo, y `C`, `E` y `F` son ablaciones que
    multiplicarían la búsqueda sin agregar diagnóstico.

    Sus números son de diagnóstico y no entran en las tablas del veredicto: lo
    único que deciden es si vale reestructurar para techos por nivel.
    """
    import json

    from MIL_CREDA_Benchmark import config, contamination, harness

    tasa = config.NOISE_DIAGNOSTIC_LEVEL
    dispositivo = harness.resolve_device()
    reduccion = harness.Reduction(device=str(dispositivo),
                                  environment=harness.environment(),
                                  labelNoise=tasa, pilot=True)

    # El techo buscado SOBRE material contaminado: es el punto que la campaña no
    # tiene y la única razón por la que este paso cuesta algo.
    # Sobre la transferencia de la curva y ninguna otra. Buscar sobre las seis
    # costaría seis veces lo que el diagnóstico vale y mediría cinco
    # transferencias que la curva nunca recorrió, así que no habría contra qué
    # leerlas.
    buscado = harness.search_ceilings(reduccion, dispositivo, noise=tasa,
                                      transfers=[config.NOISE_TRANSFER],
                                      pilot=True)

    registro = {
        "level": tasa,
        "transfer": "{}->{}".format(*config.NOISE_TRANSFER),
        "arms": list(config.NOISE_DIAGNOSTIC_ARMS),
        "searchedUnderNoise": buscado,
        # El otro extremo, leído y no vuelto a correr: si esta celda y el
        # barrido dijeran cosas distintas habría dos versiones del mismo número.
        # Del BARRIDO y no de una campaña: a este nivel no hay campaña completa
        # --- sólo 0.0 y NOISE_REPORTED la tienen --- y la comparación es sobre
        # la transferencia del barrido de todos modos.
        "cleanCeilingRun": (contamination.load(tasa, kind="curve") or {}).get("summary"),
        "revision": config.REVISION,
        "diagnosticOnly": ("estos números no entran en las tablas del veredicto; "
                           "deciden si vale reestructurar para techos por nivel"),
    }
    # Bajo la raíz de ESTA corrida y no bajo `Results/` a secas: un diagnóstico
    # de ensayo escrito donde va el de la campaña completa la pisa, y lo pisa con
    # números que no se pueden citar.
    destino = config.results_for(0.0, "curve", True).parents[1]
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "diagnostic.json").write_text(
        json.dumps(registro, indent=2, default=str), encoding="utf-8")
    return registro


def barrido_de_ruido() -> dict:
    """Corre la campaña sobre UNA transferencia, en cada nivel declarado.

    Este es el ejercicio del ruido, y es una forma distinta de una campaña, no
    una campaña más chica: una transferencia recorriendo los cinco niveles
    contra seis transferencias en un nivel. Por eso escribe bajo `kind="curve"`
    -- las dos pueden pararse en la misma tasa, y `runs.jsonl` se abre en `"w"`.

    Los techos salen de la búsqueda en limpio y se mantienen fijos en los cinco
    niveles: la curva es el coeficiente elegido sin contaminación aplicado con
    ella. Lo que eso cuesta lo separa después `diagnostico_de_ruido`.

    No guarda pesos en ningún nivel. La curva se lee de los `runs.jsonl`, y un
    nivel que escribiera 8 GB que nadie abre dejaría un directorio que parece
    evidencia.
    """
    from MIL_CREDA_Benchmark import config, harness

    dispositivo = harness.resolve_device()
    # Los techos que YA estén en el registro, sin buscar nada acá. Llamar a
    # `with_ceilings_in_force` era el reflejo obvio y es una trampa: cuando no
    # existe `ceilings.json` esa función lanza la búsqueda COMPLETA --- 2
    # familias x 6 transferencias x 30 trials a 20 épocas, unas nueve horas y
    # media --- sin decir que lo está haciendo y sin que nadie la haya
    # autorizado. Un barrido a escala de ensayo no puede ser la puerta por la
    # que entra la corrida larga.
    #
    # Sin registro no corre: un barrido con techos vacíos mediría el ruido y la
    # falta de coeficiente a la vez.
    if harness.search_record() is None and harness.search_record(pilot=True) is None:
        raise SystemExit(
            "no hay registro de techos. Corré primero la búsqueda "
            "(`search-pilot` a escala de ensayo, o la completa con su "
            "autorización): un barrido sin techos mide dos cosas a la vez.")
    base = replace(
        harness.Reduction(device=str(dispositivo),
                          environment=harness.environment(), pilot=True),
        ceilings=config.ceilings_on_record(),
        ceilingsByTransfer=config.ceilings_by_transfer_on_record())

    corridos = {}
    for tasa in config.NOISE_LEVELS:
        reduccion = replace(base, labelNoise=tasa)
        corridos[f"{tasa:g}"] = harness.campaign(
            replace(reduccion, kind="curve"), dispositivo,
            transfers=[config.NOISE_TRANSFER])
    return corridos


def informe_de_ruido() -> str:
    """La curva de degradación sobre los niveles que dejaron registro."""
    return _ejecutar("Benchmark_Noise_v1.ipynb")


def informe_del_diagnostico() -> str:
    """La tabla y la conclusión que separan el término del coeficiente.

    Sólo lee: `diagnostico_de_ruido` ya escribió `diagnostic.json` y este
    cuaderno lo presenta. No estaba, y la ausencia no era una decisión sino un
    hueco: el eje era el único con un paso que computa y ninguno que dibuje, y
    lo que decide si vale reestructurar para techos por nivel quedaba computado
    y sin que nadie pudiera leerlo.
    """
    return _ejecutar("Benchmark_Noise_Diagnostic_v1.ipynb")
