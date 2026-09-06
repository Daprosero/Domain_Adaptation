"""Los pasos locales que la forja puede ejecutar por su cuenta.

Casi todas ejecutan un cuaderno de este repositorio y nada más, que es la forma
que corresponde: el ensayo existe para correr los cuadernos que se van a enviar,
así que un paso que compute en lugar del suyo prueba la biblioteca y deja sin
ejercitar el artefacto. Las que todavía computan directo dicen en su propia
docstring por qué. La forja las nombra estáticamente en `__steps__` (módulo +
función), las resuelve sin importarlas, y recién las importa y llama dentro del
intérprete de este repositorio.

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
# `Benchmark_Noise_Diagnostic_Report_v1.ipynb` estuvo cinco celdas sin ejecutar ni una
# sola, computado y sin dibujar, sin que nada lo notara. La prueba deriva los
# cuadernos que sí se corren del código y le resta el disco; lo que sobra tiene
# que estar acá, y estar acá cuesta escribir por qué. El próximo que se excluya
# tiene que decir lo suyo.
#
# Está vacío, y vacío es una afirmación: hoy todo cuaderno del árbol lo corre un
# paso. `Benchmark_Campaign_v1.ipynb` estuvo acá con la razón «lanza al servicio
# remoto», que era falsa contra el archivo --- el cuaderno no importa
# `remote_cli`, no submite, y sólo menciona `/kaggle/working` para encontrar el
# repositorio si alguien lo corre allá --- y por esa razón inventada el único
# cuaderno que de verdad se envía era el único que el ensayo nunca ejercitaba.
CUADERNOS_SIN_PASO: dict[str, str] = {}


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

    **El único paso que computa sin correr un cuaderno, y es a propósito.** La
    búsqueda no tiene cuaderno propio y no le corresponde tener uno:

    * `Benchmark_Search_Report_v1.ipynb` es el INFORME de la búsqueda y ya lo corre
      `informe_de_busqueda`. Que este paso lo corriera también dejaría un
      cuaderno con dos dueños, y `Notebooks/Benchmark_Search_Report_v1.ipynb` ya es
      raíz declarada de `search-report`: la declaración se pone en rojo sola.
      Descomentar la llamada de su celda 7 es la misma cosa por dentro, y
      además haría que abrir el informe cueste lo que cuesta correrlo, que es
      exactamente por lo que esa línea está comentada.
    * Un cuaderno nuevo cuyo único contenido fuera esta llamada sería una
      segunda forma, más débil, de pedir lo mismo: `run_search` existe
      justamente para que un trabajo remoto pueda nombrar `module.function` y
      pasarle JSON, y un cuaderno no puede ser eso.

    Lo que sí se ejercita en el recorrido es su informe, un peldaño después.
    """
    from MIL_CREDA_Benchmark import harness

    return harness.run_search(pilot=True)


def campana() -> str:
    """Corre la campaña ejecutando SU cuaderno, a la escala que declara `config`.

    Dos pasadas y una sola ejecución. La campaña es cada transferencia a UNA
    tasa, así que correr los dos niveles que el informe muestra --- `config.NOISE`
    y `config.NOISE_REPORTED` --- son dos pasadas de esa misma forma y no un paso
    nuevo: el bucle vive en la celda 7 del cuaderno, que es donde vive la campaña.
    Ejecutar el cuaderno dos veces sería la otra forma de pedirlo y es peor:
    `_ejecutar` corre `--inplace` y la salida ejecutada ES el informe, así que la
    segunda ejecución borraría lo único que la primera deja. Además el cuaderno
    resuelve sus techos y paga su pronóstico una vez por ejecución, y las dos
    cosas se pagarían de nuevo.

    Este paso llamaba a `harness.campaign()` en lugar de correr
    `Benchmark_Campaign_v1.ipynb`, y ese era el defecto: el cuaderno que se
    envía era el único que el ensayo nunca ejercitaba. Su primera celda dice de
    sí mismo que corre «el pronóstico de costo, la búsqueda del techo de cada
    familia y la campaña completa», y tenía cero celdas ejecutadas. Un ensayo
    que computa por su cuenta prueba la biblioteca y no prueba el artefacto.

    La biblioteca no se toca: el cuaderno llama a `harness.campaign()` en su
    celda 7, así que lo que cambió es el cuerpo de este paso y nada más.

    Las dos guardas son precondiciones, no cómputo, y las dos existen para que
    el cuaderno pueda correr sin escribir fuera de las raíces que este paso
    declara:

    * **La escala.** `produces` nombra los dos árboles de ENSAYO --- el limpio y
      el contaminado ---, y el cuaderno elige su escala con
      `config.is_pilot_scale()`. A escala completa escribiría en
      `Results/Benchmark/` y `Results/Noise/rho0p2/`, que son de otro, así que
      acá se niega en vez de mudarse en silencio.
    * **El registro de techos.** Sin ninguno de los dos archivos, `campaign()`
      se negaría adentro del cuaderno con un mensaje sobre techos vacíos;
      negarse acá dice qué correr antes. Los techos se resuelven una vez, arriba
      del bucle, y las dos pasadas corren bajo los mismos: los de la búsqueda EN
      LIMPIO. La pasada contaminada no re-busca nada --- lo que la contaminación
      le cuesta al techo lo mide `diagnostico_de_ruido`, que busca a su nivel y
      no gobierna este registro. Las dos escalas preguntadas por
      separado y las dos dichas: pregunta si existe ALGUNO de los dos archivos,
      no cuál rige. Desde que la omisión significa «el que rige», una llamada
      pelada contestaría por los dos y las dos mitades de la pregunta se
      volverían una sola.

    Esto NO envía nada al servicio remoto, y el cuaderno tampoco: no importa
    `remote_cli`, no submite, y lo único que lo menciona es su bootstrap, que
    resuelve el repositorio bajo `/kaggle/working` para poder correr ALLÁ si
    alguien lo manda. La exclusión anterior decía que «lanza al servicio
    remoto», y eso era falso contra el archivo.
    """
    from MIL_CREDA_Benchmark import config, harness

    if not config.is_pilot_scale():
        raise SystemExit(
            f"la escala configurada es la completa ({config.EPOCHS} épocas, "
            f"{len(config.SEEDS)} semillas). Este paso declara el árbol de "
            "ensayo y el cuaderno escribiría en el de la corrida completa: "
            "una campaña a escala completa se lanza con su propia "
            "autorización, no por acá.")
    if (harness.search_record(pilot=False) is None
            and harness.search_record(pilot=True) is None):
        raise SystemExit(
            "no hay registro de techos. Corré primero la búsqueda: una campaña "
            "sin techos mide el método y la falta de coeficiente a la vez.")
    return _ejecutar("Benchmark_Campaign_v1.ipynb")


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
    return _ejecutar("Benchmark_Search_Report_v1.ipynb")


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
    # números que no se pueden citar. `pilot=True` fijo y no heredado, porque
    # esta función arma su propia reducción de ensayo, con `pilot=True`, arriba.
    destino = config.noise_axis_for(True)
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

    Guarda pesos en dos de los cinco niveles y en ninguno de los otros tres, y
    no lo decide este paso: `campaign()` pregunta por `keeps_checkpoints`, que es
    verdadero exactamente en los niveles que el cuaderno latente dibuja --- 0.0 y
    `NOISE_REPORTED`. Los otros tres corren y escriben sus `runs.jsonl`, que es
    de donde sale la curva, y ni un peso: 8 GB por nivel que nadie abre dejarían
    un directorio que parece evidencia. Esta frase decía «no guarda pesos en
    ningún nivel», que era falso desde que existe `CHECKPOINT_LEVELS`.
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
    # Las dos escalas preguntadas por separado y las dos dichas: esta guarda
    # pregunta si existe ALGUNO de los dos archivos, no cuál rige. Desde que la
    # omisión significa «el que rige», una llamada pelada contestaría por los dos
    # y las dos mitades de la pregunta se volverían una sola.
    if (harness.search_record(pilot=False) is None
            and harness.search_record(pilot=True) is None):
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
    return _ejecutar("Benchmark_Noise_Report_v1.ipynb")


def informe_del_diagnostico() -> str:
    """La tabla y la conclusión que separan el término del coeficiente.

    Sólo lee: `diagnostico_de_ruido` ya escribió `diagnostic.json` y este
    cuaderno lo presenta. No estaba, y la ausencia no era una decisión sino un
    hueco: el eje era el único con un paso que computa y ninguno que dibuje, y
    lo que decide si vale reestructurar para techos por nivel quedaba computado
    y sin que nadie pudiera leerlo.
    """
    return _ejecutar("Benchmark_Noise_Diagnostic_Report_v1.ipynb")
