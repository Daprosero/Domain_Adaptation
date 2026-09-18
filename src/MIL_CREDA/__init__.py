"""MIL-CREDA: reference implementation of the managed proposal.

One module per mathematical object. Every module declares `__provenance__`
binding it to the revision it was written against.

The numeric backend is PyTorch. That is a change of backend, not of method:
every equation is the same equation, but each term of Eq. (39) now carries a
gradient and can be placed on a device, which is the difference between an
objective that is evaluated and one that is trained.

This file holds the only thing here that is not a mathematical object — how a
caller's input becomes a tensor — precisely so that no module of mathematics has
to spend lines on plumbing.

Double precision is deliberate. The invariants of Section 2 are checked at
tolerances near 1e-12, and single precision cannot resolve them: a bound that
holds only to 1e-6 is not the bound the proposal states. A caller who trains in
single precision passes float32 tensors and gets them back respected.
"""

from __future__ import annotations

import torch

#: The dtype used for anything this package has to materialize itself.
DTYPE = torch.float64


def as_tensor(value: object, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Accept whatever the caller has; return a tensor without disturbing one.

    A tensor is returned untouched. Re-casting it would silently detach it from
    the graph that produced it, or pull a device-resident term back to the host,
    and either one turns a trainable implementation back into a calculator.
    Anything else — a list, an array, a Python float — becomes a tensor.
    """
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=DTYPE if dtype is None else dtype)


def as_matrix(value: object) -> torch.Tensor:
    """`as_tensor`, then at least two dimensions: one row per sample."""
    tensor = as_tensor(value)
    return tensor.reshape(1, -1) if tensor.dim() < 2 else tensor


def as_index(value: object, *, device: torch.device | None = None) -> torch.Tensor:
    """An index set or a label vector as a flat `long` tensor.

    `device` moves it next to the matrix it will index: advanced indexing
    requires both to live in the same place, and the caller's index set is
    usually built on the host even when the kernel matrices are not.
    """
    if isinstance(value, torch.Tensor):
        index = value.reshape(-1).long()
    else:
        index = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    return index if device is None else index.to(device)

__implementation__ = {
    "revision": "research-concept-r21.md",
    "premises": {
        "prediction": "a single class per bag, chosen from CLASSES alternatives",
        "unit": "the bag, for every arm",
        "metric": "accuracy over the evaluation bags of the target domain",
        "direction": "higher is better",
    },
}

# La escalera de escalones que un paso de la sección de posición puede
# alcanzar, en las palabras de este repositorio. Tres, y el primero no es un
# adorno: `_record_scale_level` usa el escalón más bajo para decir «todavía no
# corrió nada», y coloca «corrió, pero corto de la escala declarada» un escalón
# debajo del tope solo si la escalera tiene tres o más. Con dos, «no hay
# búsqueda» y «la búsqueda corrió en ensayo» caerían en el mismo escalón.
#
# `none`   -- no corrió nada. Es un escalón, no la ausencia de uno: «no miramos»
#             es otra cosa y la forja la reporta aparte, sin medir.
# `pilot`  -- corrió acá, a escala de ensayo. Prueba que el cable lleva
#             corriente y no habilita a citar ningún número.
# `remote` -- corrió a la escala que el protocolo declara. Es el único escalón
#             cuyos resultados se reportan.
#
# El orden es el de la lista y lo compara la forja por posición, sin conocer
# ninguno de los tres nombres. Una primera pasada apunta a `pilot` y ahí
# termina; pasar de `pilot` a `remote` es una decisión aparte, con su
# autorización y su porqué.
__levels__: list = ["none", "pilot", "remote"]

# Los registros que un testigo con nivel puede direccionar por nombre. Literal
# aparte de `__benchmark__` por la misma razón que `__levels__` y `__steps__`.
#
# Hay una sola entrada y su escala es la COMPLETA, a propósito: la forja gradúa
# una entrada contra su propia escala declarada, así que el registro del ensayo
# --que existe pero queda corto-- alcanza el peldaño intermedio, y sólo el de
# escala completa llega al tope. Declarar el ensayo con su propia escala chica
# lo haría cumplirla y marcaría el tope, que es exactamente lo contrario.
__records__: dict = {
    "ceilings": {"path": "MIL-CREDA/Results/Benchmark/ceilings.json",
                 "requiredScale": {"epochs": 20, "trials": 30}},
}

# Lo que hay que instalar en el worker antes de que corra nada, en la forma
# EXACTA del bloque `environment` de `run-config.json`. La forja lo pasa entrada
# por entrada (`generate-job --environment-requirement`) y el arranque lo corre
# como `python -m pip install <entradas>`, sin shell: cada entrada es un
# especificador posicional y uno que empiece con `-` se rechaza, así que no hay
# forma de meter una bandera acá.
#
# **Por qué recién ahora.** Mientras un envío llamaba a una FUNCIÓN este paso
# del arranque era un no-op y no se notaba: el worker importaba
# `MIL_CREDA_Benchmark.harness` y lo que esa importación necesita ya venía en la
# imagen. Ejecutar un CUADERNO agrega requisitos que ninguna de esas corridas
# ejerció, y la celda 0 se niega cuando no los encuentra --- nombrando
# justamente esta declaración como el remedio. Sin ella el primer envío se
# rechaza antes de ejecutar una celda.
#
# **Las cuatro, y qué hace cada una.** Las tres primeras son las que la celda 0
# IMPORTA por su nombre para decidir si el runtime puede ejecutar un cuaderno:
# `nbformat` lo lee y lo vuelve a escribir, `nbclient` lo ejecuta, y
# `jupyter-client` es lo que resuelve un NOMBRE de kernel contra un kernelspec
# instalado. La tercera no sobra por venir de arrastre con la segunda: un
# runtime puede tener los dos lectores y ningún kernelspec, y ésa es la falla
# que ya costó un envío entero. `ipykernel` es la cuarta y no la nombra ningún
# rechazo: es lo que instala el kernelspec `python3`, que es el que los diez
# cuadernos de este repositorio declaran en su `metadata.kernelspec.name`. Sin
# él las tres primeras importan y el kernel no arranca igual.
#
# **Y nada más, que es la otra mitad de la decisión.** No va `torch`, ni
# `optuna`, ni nada del stack científico: eso lo satisface la imagen ---medido y
# no supuesto, por los envíos que corrieron `run_search` importando ese mismo
# árbol--- y un `pip install torch` encima podría reemplazar la compilación CUDA
# de la imagen por una rueda de CPU. Declarar de más no es prudencia acá: es
# cambiarle el entorno al worker sin haber medido que hacía falta.
#
# **Sin versiones y sin `indexUrl`, a propósito.** Un instalador ya satisfecho
# es un no-op y no cuesta nada; uno con pin fuerza una subida o una bajada sobre
# el stack de Jupyter que la imagen ya trae. Y un espejo sería una decisión
# sobre el servicio, que no es conocimiento de este repositorio.
__environment__: dict = {
    "install": {
        "requirements": ["nbformat", "nbclient", "jupyter-client", "ipykernel"],
    },
}

# Los pasos locales que la forja puede ejecutar sola, en el venv de este
# repositorio. Literal aparte de `__benchmark__` por la misma razón que
# `__levels__`: declararlos no vuelve "declarado" un repositorio que todavía no
# lo está.
#
# `produces`: las raíces que cada paso escribe, relativas a la carpeta de
# producto. La forja fotografía el producto antes de lanzar el paso y después
# de que reporta, y contrasta lo que cambió contra estas raíces. Sin ellas las
# dos lecturas quedan apagadas: una corrida que volvió sin haber escrito nada
# se lee igual que una que produjo toda su salida, y una que escribió en el
# árbol de OTRO paso se lee igual que una que se quedó en el suyo. Las dos
# fallas se midieron acá el mismo día --- el barrido escribió su nivel limpio
# en el directorio de checkpoints de la campaña y le reetiquetó diez
# manifiestos, y la re-búsqueda del diagnóstico pisó el registro de techos
# vigente con una búsqueda contaminada de una sola transferencia --- y ninguna
# de las dos levantó nada: las dos reportaron `outcome: "returned"`.
#
# Cada raíz sale de leer a dónde escribe el paso --- las puertas que
# `config.DESTINOS` declara (`results_for`, `models_for`, `ceilings_record_for`,
# `noise_axis_for`, `harness.shard_paths`, `harness.seal_shard_stamp`) y las
# celdas del cuaderno que ejecuta --- y nunca de copiar la del vecino.
#
# **La escala va adentro de la raíz, y no hay una sola raíz que cubra las
# dos.** `results_for` mete el segmento `Pilot/` ARRIBA de la forma compartida
# (`base = RESULTS.parent`, y recién ahí `/ "Pilot"`), así que el único
# ancestro común de `Results/Benchmark` y `Results/Pilot/Benchmark` es
# `Results/`, que además contiene el árbol de todos los demás pasos:
# declararlo volvería `own` a cualquier escritura en cualquier lado y apagaría
# el guarda sin decir que lo apagó. Entonces la declaración es específica de
# escala. Los dos pasos de ensayo remoto (`search-pilot`, `noise-sweep`)
# declaran la raíz de ENSAYO y ninguna otra --- el día que uno escriba a
# escala completa eso es `foreign`, que es exactamente lo que hay que ver, y
# los DOS derivan su escala de `config.is_pilot_scale()` y se niegan antes de
# llegar ahí. `results` es local y sólo lee lo que cualquiera de las dos
# escalas ya dejó, así que sus `produces` no llevan segmento de escala en
# absoluto -- la misma forma que `report`/`latent` tenían antes de este
# stretch.
#
# Las raíces son literales porque la forja lee este archivo con `ast` y sin
# importarlo. Las que llevan un `rho` adentro salen de `config.NOISE_REPORTED`
# (0.2) y del formato de `results_for` (`f"rho{rate:g}".replace(".", "p")`);
# `tests/test_steps.py` las vuelve a componer desde `config` y se pone en rojo
# si alguna de las dos cosas cambia.
# `reads`: las raíces que cada paso CONSUME y que produjo otro paso, escritas
# siempre en la ortografía de la corrida COMPLETA. Es la otra mitad de
# `produces` y existe por el ensayo remoto: el paso corre a escala reducida en
# el worker, contra lo que los pasos anteriores dejaron a escala completa, y
# `steps.ensayo_remoto` se niega ---antes de abrir el cuaderno y antes de gastar
# nada--- cuando alguna de estas raíces todavía no está en disco.
#
# **La cadena se DERIVA de las dos listas y no se escribe.** Un paso es
# predecesor de otro cuando alguna raíz de su `produces` cubre alguna raíz del
# `reads` del otro, comparado por segmentos y con la escala normalizada
# (`steps.raiz_a_escala_completa`). Nadie nombra a nadie: el paso sin predecesor
# ---el único exento de la regla, porque no hay nada arriba suyo que consumir---
# sale de que su `reads` quede vacío, no de una lista de excepciones. Agregar un
# paso que lea lo que otro escribe lo encadena solo.
#
# Cada raíz sale de leer qué abre el cuaderno del paso ---`harness.search_record`,
# `config.ceilings_on_record`, `contamination.load`/`in_force`,
# `latent.available`--- y de resolver a qué archivo apunta esa lectura, nunca de
# copiar el `produces` del vecino al revés.
#
# Lo que NO va acá: lo que el paso escribe (eso es `produces`), lo que lee de su
# propia corrida ---la campaña relee su `runs.jsonl` recién escrito--- y el
# código fuente, que viaja con el clon y no lo produce ningún paso. Por eso
# `verification` lee cero: corre la suite sobre `src/MIL_CREDA` y nada más.
#
# Lo que un cuaderno lee bajo un `if ... .exists()` SÍ va acá, y no es un
# descuido. `results` dibuja su mitad contaminada sólo si hay corridas y sigue
# adelante si no; el ensayo, en cambio, existe para ejercitar el paso contra sus
# entradas reales, y uno que corra con la mitad contaminada ausente prueba la
# mitad del cuaderno y reporta lo mismo que uno completo. No cuesta un rechazo
# de más: las dos pasadas de la campaña viven adentro de UNA ejecución, así que
# si `results` corrió completo están las dos o no está ninguna.
__steps__: dict = {
    # Corre la suite adentro del cuaderno y dibuja la cota local. Dos clases de
    # raíz: el dato que produce y el cuaderno que ejecuta en el lugar.
    # `ROOT / "MIL-CREDA" / "Results" / "local_distance_bound"` en la celda de
    # la cota, con el `.pdf` que le pone `figures.emit`.
    "verification": {"module": "MIL_CREDA_Benchmark.steps", "function": "verificacion",
                     "advances": 1,
                     "reads": [],
                     "produces": ["Results/local_distance_bound.pdf",
                                  "Notebooks/verification.ipynb"],
                     "placement": "local"},
    # Corre `Benchmark_Ceiling_Search.ipynb`, que es el cuaderno que CORRE la
    # búsqueda, y no computa en su lugar. Su celda de la corrida llama a
    # `harness.run_search(pilot=ES_ENSAYO)` con `ES_ENSAYO` derivado de
    # `config.is_pilot_scale()`, igual que los otros tres, y el paso se niega si
    # esa escala no es la del ensayo --- así estas raíces son las de ensayo y no
    # pueden ser otras.
    #
    # El cuaderno se llamaba `Benchmark_Search_Pilot_v1.ipynb` y fijaba `True`
    # adentro. Ese nombre afirmaba una escala que no le toca elegir y, mientras
    # la afirmaba, ningún cuaderno podía correr la búsqueda completa: a escala
    # completa los techos venían de la biblioteca y todo lo demás del recorrido
    # de un cuaderno. `Benchmark_Search_v1` --- el nombre que el informe dejó
    # libre --- no se recicla: una referencia vieja seguiría resolviendo contra
    # otro artefacto.
    #
    # `run_search` -> `ceilings_in_force` -> el motor `optuna`, que escribe un solo
    # archivo: `config.ceilings_record_for(True)`, o sea `CEILINGS_PILOT_RECORD`.
    # Es OTRO árbol que el de `results_for`: el registro de ensayo se separa por
    # nombre de archivo y no por directorio (ver el comentario de `shard_paths`),
    # así que cae al lado del registro completo y no bajo `Results/Pilot/`. El
    # motor por grilla dejaría además un `.partial.json` acá; no está declarado
    # porque no es el motor que `config.SEARCH_ENGINE` nombra.
    #
    # Y el cuaderno mismo, que se ejecuta `--inplace`: la salida ejecutada ES lo
    # que queda de esta corrida.
    "search-pilot": {"module": "MIL_CREDA_Benchmark.steps",
                     "function": "ensayo_de_busqueda",
                     "advances": 2,
                     "reads": [],
                     "produces": ["Results/Benchmark/ceilings.pilot.json",
                                  "Notebooks/Benchmark_Ceiling_Search.ipynb"],
                     "placement": "remote",
                     "job": "ceiling-search",
                     "service": "kaggle"},
    # `search-report` (`informe_de_busqueda`, que corría
    # `Benchmark_Search_Report_v1.ipynb`) fue retirado entero, y no por
    # duplicación: el informe de la búsqueda no es un resultado de este paper.
    # Lo que la búsqueda deja es el registro de techos, y ese registro ya
    # atraviesa el recorrido entero --- lo consume el barrido, lo consume la
    # campaña, y lo que un lector tiene que poder juzgar es la corrida que
    # corrió bajo esos techos, no la elección de los techos presentada como
    # experimento propio. Un paso que sólo presenta un insumo le pone al
    # recorrido un ordinal, una notebook y un ítem de posición que ninguna
    # afirmación del paper necesita.
    #
    # El eje de ruido, reducido a un solo paso que lo genera y lo corre. La
    # separación previa entre el barrido y su informe, más el diagnóstico y el
    # suyo, era el eje del noise-diagnostic que este stretch retira (ver el
    # comentario grande sobre `noise-diagnostic` en la revisión anterior de
    # este archivo, y `config.py`'s own retirement note: la re-búsqueda del
    # diagnóstico re-buscaba el techo bajo contaminación, lo que contradice la
    # decisión de que la búsqueda corre siempre sobre material limpio).
    #
    # Corre `Benchmark_Noise_Sweep.ipynb` y no computa en su lugar. Ya no
    # una campaña por nivel sobre cada arm/transferencia: la celda de la
    # corrida llama a `harness.campaign(reduction, device, arms=["B"],
    # transfers=[config.NOISE_TRANSFER], ...)` por nivel, con
    # `kind="curve"` y el `pilot` que la celda deriva de
    # `config.is_pilot_scale()` -- el paso se niega si esa escala no es la del
    # ensayo, así que estas raíces son las de ensayo y no pueden ser otras.
    # Sólo el brazo piso (`B`, `FLOOR_OF`'s own floor) y una sola
    # transferencia (`config.NOISE_TRANSFER`): la curva de degradación es
    # sobre el material, no sobre el método, así que no necesita repetir cada
    # brazo.
    #
    # El directorio `curve/` de `results_for` cubre los cinco `rho*`, y el
    # `Probe_results.json` que `campaign()` deja en su padre. Los pesos
    # también son un directorio, aunque con un solo brazo su volumen es una
    # fracción de lo que costaba antes.
    #
    # Y el cuaderno mismo, ejecutado `--inplace` una sola vez: los cinco
    # niveles viven adentro de una ejecución, así que la salida ejecutada los
    # muestra a los cinco.
    "noise-sweep": {"module": "MIL_CREDA_Benchmark.steps",
                    "function": "barrido_de_ruido",
                    "advances": 3,
                    "reads": ["Results/Benchmark/ceilings.json"],
                    "produces": ["Results/Pilot/Noise/curve",
                                 "Models/Pilot/Noise/curve",
                                 "Notebooks/Benchmark_Noise_Sweep.ipynb"],
                     "placement": "remote",
                     "job": "noise-sweep",
                     "service": "kaggle"},
    # La campaña completa: cada arm declarado, sobre cada transferencia que
    # `config.VERDICT_TRANSFERS` nombra, a `config.FULL_EPOCHS`/`FULL_SEEDS`.
    # `campaign-local` -- el paso que antes ejercitaba `harness.campaign()`
    # corriendo `Benchmark_Campaign_v1.ipynb` -- fue retirado junto con esa
    # notebook (commit `2f9bf32`) y este paso es su sucesor. Corrió un tiempo
    # sin cuaderno, llamando a `harness.run_campaign_shard()` desde la
    # biblioteca, y eso es lo que `verify` reportaba bajo
    # `undeclaredStepNotebooks`: el ensayo lo recorría probando la biblioteca
    # mientras el artefacto que llevaría el mismo trabajo a otra máquina no lo
    # ejecutaba nadie. Hoy corre `Benchmark_Campaign.ipynb`, y ese cuaderno
    # llama a `run_campaign_shard()` en su celda de la corrida -- la función
    # queda intacta y sigue siendo lo que un `run-config.json` remoto nombra,
    # así que el ensayo y el envío real ejercitan el mismo código por dos
    # puertas y no dos códigos distintos.
    #
    # **Con `Pilot/` en `produces`, y antes sin él.** Acá decía: «sin `Pilot/`
    # en `produces`: `run_campaign_shard` entrena siempre a escala completa
    # (la misma razón que la búsqueda), así que no hay árbol de ensayo propio
    # que declarar -- escribe siempre en `Results/Benchmark`/
    # `Models/Benchmark`». Era una descripción del hueco y no una propiedad
    # del paso: la función no tomaba dial de escala, así que el recorrido de
    # ensayo la corría a escala completa --- cinco brazos, seis
    # transferencias, treinta semillas, veinte épocas, 56 minutos antes de
    # que alguien la matara a mano --- sin que nada se negara.
    #
    # Hoy la función RECIBE la escala (`run_campaign_shard(pilot=...)`), el
    # cuaderno la deriva de `config.is_pilot_scale()` y el paso se niega
    # cuando esa lectura no es la del ensayo, así que estas raíces son las de
    # ensayo y no pueden ser otras. Las dos salen de las puertas ---
    # `config.results_for(0.0, "campaign", True)` y
    # `config.models_for(0.0, "campaign", True)` --- y `steps.
    # raiz_a_escala_completa` las traduce de vuelta a la ortografía completa,
    # que es lo que deja a `results` encadenado a este paso por su `reads`.
    #
    # `reads` vacío y no `["Results/Benchmark/ceilings.json"]`, a propósito:
    # ver la docstring de `steps.campana` para el porqué -- ningún paso
    # declarado produce esa raíz (la búsqueda completa es la entrada única de
    # `__records__`, corrida por fuera del recorrido de pasos), así que
    # declararla no encadenaría nada que `predecesores` pudiera derivar. Sin
    # predecesores, el ensayo remoto de este paso prueba el cable
    # (`harness.run_smoke()`) y no una campaña de ensayo a nueve horas y
    # media.
    "campaign": {"module": "MIL_CREDA_Benchmark.steps", "function": "campana",
                "advances": 4,
                "reads": [],
                # Los dos archivos, no el directorio: `Results/Benchmark` es
                # compartido -- `search-pilot` escribe `ceilings.pilot.json`
                # ahí mismo, la búsqueda completa (fuera de `__steps__`)
                # escribiría `ceilings.json`, y el paso de la Sección 4
                # (`attention_mechanisms.json`, ver ese `__steps__`) también
                # -- y declarar el directorio entero se tragaría las raíces de
                # los otros por segmentos, exactamente la colisión
                # `test_ninguna_raiz_es_de_dos_pasos` existe para atrapar.
                #
                # Y el cuaderno mismo, que se ejecuta `--inplace`: la salida
                # ejecutada ES lo que queda de esta corrida, la misma forma
                # que `search-pilot` y `noise-sweep` ya tenían.
                "produces": ["Results/Pilot/Benchmark/runs.jsonl",
                            "Results/Pilot/Benchmark/summary.json",
                            # `write_shard_stamp` lo escribe al lado de las
                            # corridas que citan su manija. Nombrado archivo
                            # por archivo y no como directorio: en la
                            # ortografía completa `Results/Benchmark` contiene
                            # el `ceilings.pilot.json` de `search-pilot`, y una
                            # raíz no puede ser de dos pasos.
                            "Results/Pilot/Benchmark/shard.json",
                            # El registro que la verificación busca, escrito
                            # por `campaign()` en el PADRE de su propio árbol
                            # --- `results_for(...).parent` --- y ya declarado
                            # en el contrato del banco. Reportado `foreign` en
                            # la primera corrida de piloto que llegó al final.
                            "Results/Pilot/Probe_results.json",
                            # La condición contaminada, que este paso corre en
                            # la misma llamada: cada tabla declara un bloque
                            # limpio y uno con ruido, y una corrida que dejara
                            # sólo el primero deja la mitad de cada tabla sin
                            # existir y toda la familia de lecturas que compara
                            # las dos sin ejercitar.
                            "Results/Pilot/Noise/rho0p2",
                            "Models/Pilot/Noise/rho0p2",
                            "Models/Pilot/Benchmark",
                            "Notebooks/Benchmark_Campaign.ipynb"],
                "placement": "remote",
                "job": "campaign",
                "service": "kaggle"},
    # Sección 4: los cinco mecanismos de atención (`wiring.MECHANISMS`), sobre
    # el método completo (`G`, nunca un id declarado -- `wiring.MechanismArm`
    # hereda todo lo demás de `G` por herencia, nunca por copia). Mismo
    # patrón que `campaign`, por la misma razón (ver la docstring de
    # `steps.mecanismos_de_atencion`): corre
    # `Benchmark_Attention_Mechanisms.ipynb`, y ese cuaderno llama a
    # `harness.run_mechanism_sweep_shard()` en su celda de la corrida --- la
    # función queda intacta y sigue siendo lo que un `run-config.json` remoto
    # nombra.
    #
    # **Dos cuadernos y no uno, decidido y no heredado.** Este barrido son
    # cinco mecanismos sobre UNA configuración; la campaña es la rejilla
    # entera. Meterlos en un cuaderno ataría para siempre el barato al caro:
    # re-correr la Sección 4 pediría volver a pagar la campaña.
    #
    # `reads` vacío: consume el registro de techos a escala completa, y por
    # la misma razón que `campaign` ningún paso declarado lo produce (ver esa
    # entrada). Sin predecesores, su ensayo remoto prueba el cable
    # (`harness.run_smoke()`).
    #
    # `produces` nombra un solo archivo de datos, no un directorio:
    # `Results/Pilot/Benchmark` sigue siendo compartido con `campaign`, la
    # misma colisión que la entrada de `campaign` ya explica.
    # Y el cuaderno mismo, ejecutado `--inplace`.
    #
    # **Con `Pilot/`, y antes sin él, y esta mitad costó más que la de
    # `campaign`.** El registro de la Sección 4 no viajaba con la escala:
    # `run_mechanism_sweep` lo componía como `config.PRODUCT /
    # tables.MECHANISM_RECORD`, un camino fijo, excusado en
    # `config.DESTINOS_SIN_COORDENADA` con el argumento de que «la sección 4
    # no declara un `Reduction` de ensayo propio». Ese argumento describía la
    # ausencia del dial. Darle el dial sin mover el destino habría escrito los
    # números de tres épocas encima del registro completo que la Sección 4
    # presenta --- el mismo defecto que este cambio cierra, del otro lado ---
    # así que el destino se pide hoy por `config.results_for(rate=0.0,
    # kind="campaign", pilot=reduction.pilot)`, y el NOMBRE del archivo sale
    # de `tables.MECHANISM_RECORD`, que es lo que los lectores ya nombran: a
    # escala completa el camino es byte por byte el de antes.
    "mechanisms": {"module": "MIL_CREDA_Benchmark.steps",
                   "function": "mecanismos_de_atencion",
                   "advances": 5,
                   "reads": [],
                   "produces": ["Results/Pilot/Benchmark/attention_mechanisms.json",
                                "Notebooks/Benchmark_Attention_Mechanisms.ipynb"],
                   "placement": "remote",
                   "job": "attention-mechanisms",
                   "service": "kaggle"},
    # El único paso que PRESENTA: reemplaza lo que `report`, `latent` y
    # `noise-report` dibujaban por separado -- las tres notebooks de esos
    # pasos fueron borradas junto con la reestructuración de este stretch.
    # `Benchmark_Results.ipynb` sólo lee y dibuja -- medido contra el archivo: no
    # llama a `harness.campaign()` ni a `config.is_pilot_scale()` en ninguna
    # celda, sólo a `cargar_corridas()`, que resuelve viendo cuál de los dos
    # árboles (completo o ensayo) tiene `runs.jsonl` y `summary.json` --
    # así que es LOCAL, como `report`/`latent` lo eran, y no lleva `job`/
    # `service`: no envía nada.
    #
    # `FIGURES = config.PRODUCT / "Results" / "figures"`, un solo árbol sin
    # segmento de escala ni de tasa -- medido en la notebook, no supuesto --
    # así que la raíz es la que el archivo compone de verdad y no la que
    # `results_for` habría dado. `config.DESTINOS_SIN_COORDENADA` ya declara
    # por qué esa raíz no lleva coordenada.
    "results": {"module": "MIL_CREDA_Benchmark.steps", "function": "resultados",
                     "advances": 6,
                     "reads": ["Results/Benchmark/runs.jsonl",
                               "Results/Benchmark/summary.json",
                               "Results/Benchmark/ceilings.json",
                               "Results/Benchmark/attention_mechanisms.json",
                               "Results/Noise/curve",
                               "Models/Benchmark",
                               "Models/Noise/rho0p2"],
                     "produces": ["Results/figures",
                                  "Notebooks/Benchmark_Results.ipynb"],
                     "placement": "local"},
}
