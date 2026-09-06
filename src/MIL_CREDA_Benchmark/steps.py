"""Los pasos locales que la forja puede ejecutar por su cuenta.

Todas ejecutan un cuaderno de este repositorio y nada más, que es la forma que
corresponde: el ensayo existe para correr los cuadernos que se van a leer y a
enviar, así que un paso que compute en lugar del suyo prueba la biblioteca y deja
sin ejercitar el artefacto. Ya no hay ninguna que compute directo, y esa frase
--- «las que todavía computan directo dicen en su propia docstring por qué» ---
llegó a cubrir tres pasos a la vez: la búsqueda, el barrido y el diagnóstico
tenían cada uno su cuaderno de informe y ninguno que corriera. La forja las
nombra estáticamente en `__steps__` (módulo + función), las resuelve sin
importarlas, y recién las importa y llama dentro del intérprete de este
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


def ensayo_de_busqueda() -> str:
    """Corre el ENSAYO de la búsqueda de techos ejecutando SU cuaderno.

    `Benchmark_Search_Pilot_v1.ipynb` fija `pilot=True` y no ofrece manera de
    pedir otra cosa. La búsqueda a escala completa es una decisión con su propia
    autorización, y un cuaderno que derivara su escala de
    `config.is_pilot_scale()` --- como sí hace el de la campaña, y con razón ---
    dejaría esa decisión en manos del tamaño configurado de la CAMPAÑA, que es
    otra cosa: la búsqueda tiene su propia escala declarada aparte. Escribe
    `ceilings.pilot.json`, que el registro completo siempre le gana cuando existe.

    Existe porque no existía: la única forma de correr esta búsqueda era un
    `python -c` a mano contra `harness`, por fuera de la skill y sin dejar rastro
    en el ledger. Un paso que falta no es disciplina que falta.

    **Este paso llamaba a `harness.run_search(pilot=True)` en vez de correr un
    cuaderno.** La razón escrita acá era que la búsqueda no tenía cuaderno propio
    y no le correspondía tener uno, y se apoyaba en dos argumentos que caían los
    dos sobre el mismo hecho --- que el único cuaderno de la búsqueda era su
    INFORME. Ese hecho dejó de ser cierto, y los dos argumentos quedan escritos
    acá en vez de reemplazados en silencio:

    * «`Benchmark_Search_Report_v1.ipynb` es el INFORME de la búsqueda y ya lo
      corre `informe_de_busqueda`. Que este paso lo corriera también dejaría un
      cuaderno con dos dueños.» Sigue siendo verdad y ya no aplica: la búsqueda
      corre el suyo, el informe corre el suyo, y cada raíz declarada sigue
      teniendo un solo dueño --- que es lo que
      `test_ninguna_raiz_es_de_dos_pasos` mide.
    * «Un cuaderno nuevo cuyo único contenido fuera esta llamada sería una
      segunda forma, más débil, de pedir lo mismo: `run_search` existe justamente
      para que un trabajo remoto pueda nombrar `module.function` y pasarle JSON, y
      un cuaderno no puede ser eso.» La premisa se midió y es correcta --- un
      `run-config.json` nombra `{module, function, kwargs}` y no puede nombrar un
      `.ipynb`, y por eso `run_search` queda intacta y sigue siendo ese camino ---
      pero la conclusión no se seguía: las dos formas no compiten. La remota
      manda una función a otra máquina; ésta ejercita acá el artefacto que un
      lector abre.

    Lo que ese razonamiento costaba está medido: tres de los pasos declarados
    --- éste, el barrido y el diagnóstico --- no nombraban ningún cuaderno entre
    sus raíces, así que el ensayo los recorría probando la biblioteca y dejando
    sin ejercitar el artefacto. Es el mismo defecto que la campaña ya había
    tenido, tres veces más.
    """
    return _ejecutar("Benchmark_Search_Pilot_v1.ipynb")


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
    registro nuevo y habría recibido el viejo, presentado. El cuaderno arrastraba
    la misma mentira en su nombre --- se llamaba `Benchmark_Search_v1` --- y por
    eso ahora lleva `_Report_`.

    La llamada comentada de su celda 7 tampoco está más. Existía porque era la
    única forma de pedir el ensayo de la búsqueda desde un cuaderno, y esa razón
    murió con `Benchmark_Search_Pilot_v1.ipynb`: descomentarla ahora le daría dos
    dueños al ensayo y haría que abrir este informe cueste lo que cuesta correrlo.
    """
    return _ejecutar("Benchmark_Search_Report_v1.ipynb")


def informe() -> str:
    """Las tablas y conclusiones sobre el registro fusionado."""
    return _ejecutar("Benchmark_Report_v1.ipynb")


def latente() -> str:
    """El análisis latente sobre los checkpoints promovidos."""
    return _ejecutar("Benchmark_Latent_v1.ipynb")


def diagnostico_de_ruido() -> str:
    """Corre el diagnóstico ejecutando SU cuaderno, a la escala que declara `config`.

    El experimento que separa *falló el término* de *le faltó coeficiente*, y que
    existe porque los techos de la campaña se buscaron en limpio y se mantienen
    fijos en los cinco niveles. Esa decisión hace que una caída no se pueda
    atribuir, y este paso es lo barato que la atribuye.

    Necesita tres puntos y paga uno solo: la búsqueda del techo al nivel de
    diagnóstico, sobre material contaminado. Los otros dos ya salen del barrido.
    Si el techo re-buscado recupera lo perdido, fue el coeficiente; si no lo
    recupera, fue el término.

    Este paso computaba en lugar de correr un cuaderno, y el eje quedaba con dos
    mitades desparejas: `informe_del_diagnostico` dibujaba un `diagnostic.json`
    que ningún cuaderno había escrito. La biblioteca no se toca --- la celda que
    mide llama a `harness.search_ceilings()` con las mismas tres restricciones
    que estaban acá ---, así que lo que cambió es el cuerpo de este paso.

    La guarda es una precondición, no cómputo, y existe para que el cuaderno pueda
    correr sin escribir fuera de las raíces que este paso declara: `produces`
    nombra el árbol de ENSAYO, y el cuaderno elige su escala con
    `config.is_pilot_scale()`. A escala completa escribiría en `Results/Noise/`,
    que es de otro, así que acá se niega en vez de mudarse en silencio --- la
    misma forma que `campana`. Antes el cuaderno no existía y la escala estaba
    fijada en `True` adentro de esta función, así que a escala completa un
    diagnóstico de veinte épocas se archivaba bajo `Pilot/`: una medición
    completa etiquetada como ensayo, que es la falla inversa y del mismo tamaño.
    """
    from MIL_CREDA_Benchmark import config

    if not config.is_pilot_scale():
        raise SystemExit(
            f"la escala configurada es la completa ({config.EPOCHS} épocas, "
            f"{len(config.SEEDS)} semillas). Este paso declara el árbol de "
            "ensayo y el cuaderno escribiría en el de la corrida completa: "
            "un diagnóstico a escala completa se lanza con su propia "
            "autorización, no por acá.")
    return _ejecutar("Benchmark_Noise_Diagnostic_Search_v1.ipynb")


def barrido_de_ruido() -> str:
    """Corre el barrido ejecutando SU cuaderno, a la escala que declara `config`.

    Este es el ejercicio del ruido, y es una forma distinta de una campaña, no una
    campaña más chica: una transferencia recorriendo los cinco niveles contra seis
    transferencias en un nivel. Por eso el cuaderno escribe bajo `kind="curve"` ---
    las dos pueden pararse en la misma tasa, y `runs.jsonl` se abre en `"w"`.

    Los techos salen de la búsqueda en limpio y se mantienen fijos en los cinco
    niveles: la curva es el coeficiente elegido sin contaminación aplicado con
    ella. Lo que eso cuesta lo separa después `diagnostico_de_ruido`.

    Guarda pesos en dos de los cinco niveles y en ninguno de los otros tres, y no
    lo decide este paso: `campaign()` pregunta por `keeps_checkpoints`, que es
    verdadero exactamente en los niveles que el cuaderno latente dibuja --- 0.0 y
    `NOISE_REPORTED`. Los otros tres corren y escriben sus `runs.jsonl`, que es de
    donde sale la curva, y ni un peso: 8 GB por nivel que nadie abre dejarían un
    directorio que parece evidencia.

    Este paso computaba en lugar de correr un cuaderno, y el eje quedaba con la
    mitad que dibuja y sin la que corre. La biblioteca no se toca: el bucle de la
    celda de la corrida llama a `harness.campaign()` nivel por nivel, igual que la
    campaña llama al suyo, con los mismos techos leídos UNA vez arriba del bucle.

    Las dos guardas son precondiciones, no cómputo:

    * **La escala.** `produces` nombra el árbol de ENSAYO, y el cuaderno elige su
      escala con `config.is_pilot_scale()`. A escala completa escribiría en
      `Results/Noise/curve/`, que es de otro, así que acá se niega en vez de
      mudarse en silencio --- la misma forma que `campana`. Antes la escala estaba
      fijada en `True` adentro de esta función, así que a escala completa un
      barrido de veinte épocas y treinta semillas se archivaba bajo `Pilot/`: una
      medición completa etiquetada como ensayo.
    * **El registro de techos.** Sin ninguno de los dos archivos el cuaderno se
      negaría adentro; negarse acá dice qué correr antes, sin pagar el arranque
      del intérprete del cuaderno. Las dos escalas preguntadas por separado y las
      dos dichas: pregunta si existe ALGUNO de los dos archivos, no cuál rige.
      Desde que la omisión significa «el que rige», una llamada pelada
      contestaría por los dos y las dos mitades de la pregunta se volverían una
      sola.

    Lo que el cuaderno NO hace, y es la razón por la que su celda de techos los
    lee en vez de resolverlos: llamar a `with_ceilings_in_force` era el reflejo
    obvio y es una trampa --- cuando no existe `ceilings.json` esa función lanza
    la búsqueda COMPLETA, unas nueve horas y media, sin decir que lo está
    haciendo. Un barrido a escala de ensayo no puede ser la puerta por la que
    entra la corrida larga.
    """
    from MIL_CREDA_Benchmark import config, harness

    if not config.is_pilot_scale():
        raise SystemExit(
            f"la escala configurada es la completa ({config.EPOCHS} épocas, "
            f"{len(config.SEEDS)} semillas). Este paso declara el árbol de "
            "ensayo y el cuaderno escribiría en el de la corrida completa: "
            "un barrido a escala completa se lanza con su propia "
            "autorización, no por acá.")
    if (harness.search_record(pilot=False) is None
            and harness.search_record(pilot=True) is None):
        raise SystemExit(
            "no hay registro de techos. Corré primero la búsqueda "
            "(`search-pilot` a escala de ensayo, o la completa con su "
            "autorización): un barrido sin techos mide dos cosas a la vez.")
    return _ejecutar("Benchmark_Noise_Sweep_v1.ipynb")


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

