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

import json
import os
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

    `Benchmark_Ceiling_Search_v1.ipynb` recibe su escala de
    `config.is_pilot_scale()`, igual que la campaña, el barrido y el
    diagnóstico, y la guarda de abajo es lo que hace que por ACÁ corra siempre
    el ensayo. Se llamaba `Benchmark_Search_Pilot_v1.ipynb` y fijaba `pilot=True`
    adentro, y eso costaba lo que el nombre prometía: **ningún cuaderno podía
    correr la búsqueda a escala completa**, así que a escala completa los techos
    salían de `harness.run_search` mientras todo el resto del recorrido salía de
    un cuaderno --- la misma divergencia entre lo que el ensayo ejercita y lo que
    la corrida real hace que estos pasos existen para cerrar. La autorización no
    se perdió: se mudó al único lugar donde ya vivía para los otros tres, que es
    la guarda de este paso.

    En ensayo escribe `ceilings.pilot.json`, que es la raíz que este paso
    declara; a escala completa escribiría `ceilings.json`, que es de otro, y por
    eso la guarda se niega en vez de mudarse en silencio.

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

    La guarda es una precondición, no cómputo, y es la misma forma que `campana`,
    `barrido_de_ruido` y `diagnostico_de_ruido`: `produces` nombra
    `ceilings.pilot.json` y ninguna otra raíz, y a escala completa el cuaderno
    escribiría `ceilings.json` --- el registro que gobierna toda campaña, unas
    nueve horas y media. Que la escala sea un modo del recorrido no significa que
    la forja pueda elegirla: significa que hay UN lugar donde se declara, y que
    este paso corre el modo de ensayo.
    """
    from MIL_CREDA_Benchmark import config

    if not config.is_pilot_scale():
        raise SystemExit(
            f"la escala configurada es la completa ({config.EPOCHS} épocas, "
            f"{len(config.SEEDS)} semillas). Este paso declara "
            "`ceilings.pilot.json` y el cuaderno escribiría `ceilings.json`, "
            "que es el registro que gobierna toda campaña: la búsqueda a escala "
            "completa se lanza con su propia autorización, no por acá.")
    return _ejecutar("Benchmark_Ceiling_Search_v1.ipynb")


def campana() -> str:
    """Corre la campaña ejecutando SU cuaderno, a la escala que declara `config`.

    Dos pasadas y una sola ejecución. La campaña es cada transferencia a UNA
    tasa, así que correr los dos niveles que el informe muestra --- `config.NOISE`
    y `config.NOISE_REPORTED` --- son dos pasadas de esa misma forma y no un paso
    nuevo: el bucle vive en la celda 8 del cuaderno, que es donde vive la campaña.
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
    celda 8, así que lo que cambió es el cuerpo de este paso y nada más.

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
      no gobierna este registro. La escala preguntada es la de ENTRADA,
      que es la misma que el cuaderno lee y no la de ESTE paso: preguntaba si
      existía ALGUNO de los dos archivos mientras el cuaderno leía uno solo, así
      que con sólo un `ceilings.json` en disco esta precondición decía
      «adelante» y el cuaderno se negaba adentro por techos vacíos --- pagando
      el arranque del intérprete para decir lo que acá se sabía. Decía `True`
      fijo, y la razón escrita era que este paso «corre siempre en ensayo porque
      la guarda de arriba lo obliga». Sigue corriendo siempre en ensayo y eso ya
      no contesta la pregunta: en el ENSAYO REMOTO el paso corre en ensayo y el
      cuaderno lee el registro COMPLETO a propósito, así que una guarda que
      preguntara por el del ensayo negaría cada ensayo remoto por la ausencia de
      un archivo que ese ensayo no va a abrir.

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
    if harness.search_record(pilot=config.upstream_pilot_scale()) is None:
        raise SystemExit(
            "no hay registro de techos a la escala que el cuaderno lee. Corré "
            "primero la búsqueda (`search-pilot`): una campaña sin techos mide "
            "el método y la falta de coeficiente a la vez, y una campaña de "
            "ensayo bajo los techos de la búsqueda completa mide un coeficiente "
            "que no midió acá.")
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
    murió con `Benchmark_Ceiling_Search_v1.ipynb`: descomentarla ahora le daría dos
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
    * **El registro de techos.** Sin el archivo de esta escala el cuaderno se
      negaría adentro; negarse acá dice qué correr antes, sin pagar el arranque
      del intérprete del cuaderno. La escala preguntada es la de ENTRADA, la
      misma que el cuaderno lee: una precondición que preguntara por ALGUNO de
      los dos archivos dejaría pasar un barrido que después corre bajo un
      registro que no es el que se le pidió. Decía `True` fijo, con la razón de
      que este paso «corre siempre en ensayo porque la guarda de arriba lo
      obliga»; sigue siendo cierto y ya no contesta la pregunta, porque en el
      ENSAYO REMOTO el paso corre en ensayo y el cuaderno lee el registro
      COMPLETO a propósito.

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
    if harness.search_record(pilot=config.upstream_pilot_scale()) is None:
        raise SystemExit(
            "no hay registro de techos a la escala que el cuaderno lee. Corré "
            "primero la búsqueda (`search-pilot`): un barrido sin techos mide "
            "dos cosas a la vez, y uno de ensayo bajo los techos de la búsqueda "
            "completa mide un coeficiente que no midió acá.")
    return _ejecutar("Benchmark_Noise_Sweep_v1.ipynb")


def informe_de_ruido() -> str:
    """La curva de degradación sobre los niveles que dejaron registro."""
    return _ejecutar("Benchmark_Noise_Report_v1.ipynb")


# --------------------------------------------------------------- el ensayo remoto
#
# La regla que gobierna esta sección, en las palabras del dueño: el ensayo que
# ocurre EN EL WORKER tiene que usar únicamente los resultados previos de los
# cuadernos que corrieron en modo COMPLETO, para saber que todo está bien antes
# del envío --- salvo el primero, que no depende de ningún otro cuaderno.
#
# Es el propósito del ensayo local, un nivel afuera. El ensayo local prueba que
# el paso corre; este prueba que el paso corre CONTRA LAS ENTRADAS REALES que
# sus predecesores dejaron. La diferencia no es de grado: el archivo que la
# corrida real va a abrir es el completo, y es el único que puede tener la forma
# vieja, el brazo que falta o la clave que se renombró. Un ensayo contra las
# salidas de otro ensayo, o contra el neutral declarado de un módulo, pasa sin
# haber tocado nada de lo que dice probar.


def raiz_a_escala_completa(raiz: str) -> str:
    """La misma raíz, escrita como la escribe la corrida COMPLETA.

    Existe porque las dos listas hablan escalas distintas y tienen que poder
    compararse: `produces` de los cuatro pasos que computan nombra el árbol de
    ENSAYO ---y tiene que nombrarlo, porque es ahí donde escriben--- mientras
    `reads` nombra siempre la ortografía de la corrida completa, que es la única
    que el ensayo remoto puede consumir.

    Dos reglas, y las dos salen de `config` en vez de estar escritas acá:

    * el segmento que `results_for`/`models_for` insertan cuando `pilot`
      (`base = RESULTS.parent`, y recién ahí `/ "Pilot"`), leído del propio
      `results_for` y no deletreado;
    * el nombre del registro de techos, que separa su escala por NOMBRE DE
      ARCHIVO y no por directorio (`ceilings_record_for`), así que la regla de
      arriba no lo toca y hace falta la segunda.

    Las dos se preguntan a las PUERTAS y no a las constantes que hay detrás:
    `config.RESULTS` y `config.CEILINGS_PILOT_RECORD` dirian hoy lo mismo, y el
    día que una puerta cambie como compone su camino esto lo seguiría sin que
    nadie lo toque --- que es la única diferencia entre derivar y volver a
    deletrear.

    Cualquiera de las dos que se mueva en `config` mueve esto con ella;
    `tests/test_steps.py` recompone las dos desde `config` y se pone en rojo si
    alguna deja de valer.
    """
    from MIL_CREDA_Benchmark import config

    partes = list(Path(raiz).parts)
    if not partes:
        return raiz
    marca = config.results_for(0.0, "campaign", True).relative_to(
        config.results_for(0.0, "campaign", False).parent).parts[0]
    if len(partes) > 1 and partes[1] == marca:
        del partes[1]
    if partes[-1] == config.ceilings_record_for(True).name:
        partes[-1] = config.ceilings_record_for(False).name
    return Path(*partes).as_posix()


def productores() -> dict[str, str]:
    """`{raíz a escala completa: paso que la escribe}`, para todo `produces`.

    Se arma de la declaración y de nada más. Dos pasos que declararan la misma
    raíz serían el defecto que `produces` existe para hacer visible, y
    `test_ninguna_raiz_es_de_dos_pasos` ya lo vigila del otro lado; acá el último
    ganaría, así que se prefiere el primero y la ambigüedad no se inventa un
    dueño nuevo.
    """
    import MIL_CREDA_Benchmark as paquete

    mapa: dict[str, str] = {}
    for nombre, entrada in paquete.__steps__.items():
        for raiz in entrada.get("produces", []):
            mapa.setdefault(raiz_a_escala_completa(raiz), nombre)
    return mapa


def _cubre(raiz: str, otra: str) -> bool:
    """Si `otra` cae bajo `raiz`, por segmentos y nunca por prefijo de cadena.

    `Results/one` no se come a `Results/one-more`, y esa diferencia es la que
    separa un guarda de una comparación de cadenas con forma de guarda. Es la
    misma lectura que hace `_owns` del lado de la forja.
    """
    partes, otras = Path(raiz).parts, Path(otra).parts
    return otras[: len(partes)] == partes


def predecesores(paso: str) -> tuple[str, ...]:
    """Los pasos que producen algo que este paso consume, en orden declarado.

    DERIVADO de las dos listas y nunca escrito: un paso es predecesor de otro
    cuando alguna raíz de su `produces` cubre alguna raíz del `reads` del otro.
    El paso exento de la regla ---el que no depende de ningún otro cuaderno---
    es el que devuelve la tupla vacía, y eso sale de que su `reads` esté vacío,
    no de una lista de excepciones que alguien tenga que mantener.
    """
    import MIL_CREDA_Benchmark as paquete

    if paso not in paquete.__steps__:
        raise SystemExit(
            f"no hay un paso declarado {paso!r}; declarados: "
            f"{', '.join(sorted(paquete.__steps__))}")
    duenios = productores()
    encontrados: list[str] = []
    for raiz in paquete.__steps__[paso].get("reads", []):
        for producida, duenio in duenios.items():
            if duenio != paso and _cubre(producida, raiz) and duenio not in encontrados:
                encontrados.append(duenio)
    return tuple(encontrados)


def entradas_faltantes(paso: str) -> list[dict]:
    """Las entradas de escala COMPLETA que este paso necesita y no están.

    Devuelve, y no se niega: quien decide qué hacer con la falta es
    `ensayo_remoto`. Que falten es el estado ORDINARIO al principio de un
    recorrido completo ---todavía no corrió el paso de arriba--- y no un defecto:
    la regla dice que el ensayo puede consumir SOLO lo completo, no que todo
    tenga que existir ya.
    """
    import MIL_CREDA_Benchmark as paquete

    duenios = productores()
    faltan = []
    for raiz in paquete.__steps__[paso].get("reads", []):
        if (config_producto() / raiz).exists():
            continue
        duenio = next((d for producida, d in duenios.items()
                       if _cubre(producida, raiz)), None)
        faltan.append({"root": raiz, "producedBy": duenio})
    return faltan


def config_producto() -> Path:
    """La carpeta de producto, preguntada a `config` y nunca compuesta acá.

    Una función y no una constante de módulo: la suite redirige
    `config.PRODUCT` con `monkeypatch`, y una constante congelada al importar
    dejaría estas lecturas mirando el árbol de la corrida real desde adentro de
    un test.
    """
    from MIL_CREDA_Benchmark import config

    return config.PRODUCT


def cuaderno_de(paso: str) -> str | None:
    """El cuaderno que ejecuta un paso, leído de su propio `produces`.

    Cada paso que corre un cuaderno lo declara entre sus raíces ---se ejecuta
    `--inplace`, así que la salida ejecutada ES una de sus salidas---, y por eso
    no hace falta una segunda lista ni leer el árbol de sintaxis de este módulo
    para saber cuál es.
    """
    import MIL_CREDA_Benchmark as paquete

    nombres = [Path(raiz).name
               for raiz in paquete.__steps__[paso].get("produces", [])
               if Path(raiz).suffix == ".ipynb"]
    return nombres[0] if len(nombres) == 1 else None


def honra_la_escala_de_entrada(paso: str) -> bool:
    """Si el cuaderno de este paso sabe leer sus entradas a escala completa.

    Un cuaderno que compone TODAS sus lecturas con `config.is_pilot_scale()`
    ---o con el `pilot` del registro que ya encontró--- no puede ser ensayado
    contra la corrida completa: en el worker abriría el árbol de ensayo, que
    está vacío, o el completo y escribiría encima de él. Los dos casos son
    peores que negarse.

    Se lee del cuaderno y no de una lista de pasos habilitados: convertir un
    cuaderno lo habilita, sin que nadie tenga que acordarse de anotarlo en un
    segundo lugar. `upstream_pilot_scale` y no un `import` cualquiera porque es
    la función que decide de qué árbol se lee, y nombrarla es exactamente lo que
    hay que probar.
    """
    nombre = cuaderno_de(paso)
    if nombre is None:
        return False
    documento = json.loads((CUADERNOS / nombre).read_text(encoding="utf-8"))
    return any("upstream_pilot_scale" in "".join(celda["source"])
               for celda in documento["cells"] if celda["cell_type"] == "code")


def notas_de_fuente() -> tuple[str, ...]:
    """Los nombres con los que un cuaderno declara DE QUÉ ÁRBOL leyó.

    Se derivan del paquete y no se escriben acá: cualquier `def` cuyo nombre
    termine en `source_note` entra sola. Agregar una nota nueva la habilita sin
    que nadie tenga que acordarse de anotarla en un segundo lugar, y borrar la
    última deja el control en ROJO en vez de en verde por una lista que quedó
    vieja --- que es la única forma en que una lista escrita a mano falla.
    """
    import ast

    nombres = set()
    for archivo in sorted((Path(__file__).parent).glob("*.py")):
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        nombres.update(
            nodo.name for nodo in ast.walk(arbol)
            if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef))
            and nodo.name.endswith("source_note"))
    return tuple(sorted(nombres))


def declara_su_fuente(paso: str) -> bool:
    """Si el cuaderno de este paso dice de cuál de los dos árboles salieron sus números.

    La otra mitad de `honra_la_escala_de_entrada`, y la que hacía falta.

    Un cuaderno que compone sus lecturas con `config.upstream_pilot_scale()` lee
    un árbol EXACTO: el de la escala del recorrido, sin caídas, y no hay nada que
    declarar porque no eligió nada. Los que no la nombran son los que resuelven su
    propia fuente ---`contamination.in_force`, `harness.search_record`,
    `tables._diagnostic_record`: los tres prefieren la corrida completa y caen al
    ensayo cuando no hay--- y ésos SÍ eligieron, en silencio y hacia abajo.

    Una caída silenciosa es peor que no tener caída. La figura sale con la forma
    correcta y los números del ensayo, y nada arriba de ella lo dice: ya pasó al
    revés en este repositorio, con «Piloto de 1 repetición» impreso encima de
    tablas de 3 semillas. Por eso la caída se permite ---es lo que hace que estos
    cinco cuadernos sirvan para mirar mientras la corrida completa todavía no
    volvió--- y el estampado no.

    Se lee del cuaderno, igual que su hermana, así que convertir uno lo habilita
    solo y este control lo sigue sin una segunda lista de pasos.
    """
    import ast

    nombre = cuaderno_de(paso)
    if nombre is None:
        return False
    documento = json.loads((CUADERNOS / nombre).read_text(encoding="utf-8"))
    notas = set(notas_de_fuente())
    mostradores = {"show", "display", "print"}

    # Nombrarla no alcanza: la nota tiene que LLEGAR a quien mira. Una nota
    # calculada y nunca mostrada deja este control en verde sobre un cuaderno
    # que no dice nada --- que es la misma falla, con un paso más.
    llamadas, asignadas, mostrado = set(), {}, set()
    for celda in documento["cells"]:
        if celda["cell_type"] != "code":
            continue
        try:
            arbol = ast.parse("".join(celda["source"]))
        except SyntaxError:
            continue  # una celda con magias no parsea; ninguna nota vive ahí
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Call):
                if isinstance(nodo.func, ast.Attribute) and nodo.func.attr in notas:
                    llamadas.add(nodo.func.attr)
                if isinstance(nodo.func, ast.Name) and nodo.func.id in notas:
                    llamadas.add(nodo.func.id)
                nombre_llamado = (nodo.func.id if isinstance(nodo.func, ast.Name)
                                  else getattr(nodo.func, "attr", ""))
                if nombre_llamado in mostradores:
                    mostrado.update(
                        hijo.id for hijo in ast.walk(nodo)
                        if isinstance(hijo, ast.Name))
                    mostrado.update(
                        getattr(hijo.func, "attr", "")
                        for hijo in ast.walk(nodo) if isinstance(hijo, ast.Call))
            if isinstance(nodo, ast.Assign):
                dentro = {getattr(h.func, "attr", "") or getattr(h.func, "id", "")
                          for h in ast.walk(nodo.value) if isinstance(h, ast.Call)}
                if dentro & notas:
                    for destino in nodo.targets:
                        if isinstance(destino, ast.Name):
                            asignadas[destino.id] = True

    if not llamadas:
        return False
    return bool(llamadas & mostrado) or bool(set(asignadas) & mostrado)


def ensayo_remoto(paso: str) -> dict:
    """El ensayo de UN paso, corriendo en el worker antes de gastar la cuota.

    UN mecanismo y no dos, con la rama derivada de la cadena y nunca de un
    nombre:

    * **Sin predecesores** ---`reads` vacío--- no hay nada arriba que consumir,
      así que lo que queda por probar es que el cable lleva corriente en ESTA
      máquina, y eso es `harness.run_smoke()`, que ya existe y ya es lo que la
      ranura `smoke` de la forja apunta. Su independencia de todo registro
      dejó de ser una propiedad del ensayo en general y pasó a ser la propiedad
      correcta EXACTAMENTE para estos pasos; su docstring lo dice así.
    * **Con predecesores** corre el cuaderno del propio paso, a escala reducida,
      contra lo que esos predecesores dejaron a escala COMPLETA.

    La escala reducida no se decide acá: la decide el commit que el worker
    clonó, y el paso mismo se niega si esa escala es la completa ---la misma
    guarda que ya tenían `campana`, `barrido_de_ruido`, `ensayo_de_busqueda` y
    `diagnostico_de_ruido`---. Acá no hay una segunda ortografía de esa
    decisión, que es como una de las dos queda vieja.

    **Cuando falta una salida completa se NIEGA y dice cuál.** Es el estado
    ordinario al principio de un recorrido ---el paso de arriba todavía no
    corrió--- y por eso el mensaje nombra la raíz y el paso que la escribe en
    vez de hablar de un error. Negarse y no caer al árbol de ensayo ni al
    neutral declarado de un módulo: las dos caídas dejan el ensayo en verde sin
    haber tocado nada de lo que dice probar, que es exactamente la falla que
    este ensayo existe para no tener.

    El modo viaja por el entorno y no por un argumento, porque quien tiene que
    verlo es el KERNEL del cuaderno, que es otro proceso: `_ejecutar` lanza
    `nbconvert` heredando este entorno, y `config.upstream_pilot_scale()` lo lee
    allí adentro. Se restaura lo que hubiera antes en vez de borrarse, así que
    dos llamadas anidadas no se pisan.
    """
    import MIL_CREDA_Benchmark as paquete
    from MIL_CREDA_Benchmark import config, harness

    previos = predecesores(paso)
    if not previos:
        return {"step": paso, "shape": "wire", "consumed": [],
                "result": harness.run_smoke()}

    if not honra_la_escala_de_entrada(paso):
        raise SystemExit(
            f"el cuaderno de {paso!r} ({cuaderno_de(paso)}) compone sus "
            "lecturas con su propia escala, así que un ensayo suyo leería el "
            "árbol de ensayo ---vacío en el worker--- o escribiría encima de la "
            "corrida completa.\n"
            "  Hacele leer sus entradas con `config.upstream_pilot_scale()` y "
            "dejale la escritura en `config.is_pilot_scale()`.")

    faltan = entradas_faltantes(paso)
    if faltan:
        detalle = "\n".join(
            f"  {item['root']} -- lo escribe "
            f"{item['producedBy'] or 'ningún paso declarado'}"
            for item in faltan)
        raise SystemExit(
            f"todavía no están las salidas de escala COMPLETA que {paso!r} "
            f"consume:\n{detalle}\n"
            "  No es un error: es el estado ordinario mientras el recorrido "
            "completo no llegó hasta acá. Corré esos pasos a escala completa y "
            "volvé. Un ensayo que cayera al árbol de ensayo, o a un neutral "
            "declarado, pasaría sin haber tocado la entrada que la corrida real "
            "va a abrir.")

    entrada = paquete.__steps__[paso]
    anterior = os.environ.get(config.REHEARSAL_ENV)
    os.environ[config.REHEARSAL_ENV] = "1"
    try:
        salida = globals()[entrada["function"]]()
    finally:
        if anterior is None:
            os.environ.pop(config.REHEARSAL_ENV, None)
        else:
            os.environ[config.REHEARSAL_ENV] = anterior
    return {"step": paso, "shape": "notebook",
            "consumed": list(entrada.get("reads", [])),
            "predecessors": list(previos), "result": salida}


def informe_del_diagnostico() -> str:
    """La tabla y la conclusión que separan el término del coeficiente.

    Sólo lee: `diagnostico_de_ruido` ya escribió `diagnostic.json` y este
    cuaderno lo presenta. No estaba, y la ausencia no era una decisión sino un
    hueco: el eje era el único con un paso que computa y ninguno que dibuje, y
    lo que decide si vale reestructurar para techos por nivel quedaba computado
    y sin que nadie pudiera leerlo.
    """
    return _ejecutar("Benchmark_Noise_Diagnostic_Report_v1.ipynb")

