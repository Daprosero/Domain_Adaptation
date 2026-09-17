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
# escala. Los cuatro pasos de ensayo (`search-pilot`, `campaign-local`,
# `noise-sweep`, `noise-diagnostic`) declaran la raíz de ENSAYO y ninguna otra
# --- el día que uno escriba a escala completa eso es `foreign`, que es
# exactamente lo que hay que ver, y los CUATRO derivan su escala de
# `config.is_pilot_scale()` y se niegan antes de llegar ahí. Eran tres: la
# búsqueda fijaba `pilot=True` adentro de su cuaderno, así que no tenía de qué
# negarse --- y tampoco había forma de correr la búsqueda completa desde un
# cuaderno ---
# y los cuadernos, que dibujan sobre la corrida que esté vigente y por lo
# tanto pueden caer de cualquiera de los dos lados, declaran las dos, una por
# escala.
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
# descuido. `report` dibuja su mitad contaminada sólo si hay corridas y sigue
# adelante si no; el ensayo, en cambio, existe para ejercitar el paso contra sus
# entradas reales, y uno que corra con la mitad contaminada ausente prueba la
# mitad del cuaderno y reporta lo mismo que uno completo. No cuesta un rechazo
# de más: las dos pasadas de la campaña viven adentro de UNA ejecución, así que
# si `campaign-local` corrió completo están las dos o no está ninguna.
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
    # Corre `Benchmark_Ceiling_Search_v1.ipynb`, que es el cuaderno que CORRE la
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
                                  "Notebooks/Benchmark_Ceiling_Search_v1.ipynb"],
                     "placement": "remote",
                     "job": "ceiling-search",
                     "service": "kaggle"},
    # Corre `Benchmark_Campaign_v1.ipynb`, que es el cuaderno que se envía, y no
    # computa en su lugar. Su celda 8 llama a `harness.campaign()` con
    # `kind="campaign"` y con el `pilot` que la celda 3 deriva de
    # `config.is_pilot_scale()`; el paso se niega si esa escala no es la del
    # ensayo, así que estas raíces son las de ensayo y no pueden ser otras.
    #
    # **Dos pasadas y por eso dos árboles.** La celda 8 recorre `NIVELES`, que es
    # `(config.NOISE, config.NOISE_REPORTED)`: la campaña es cada transferencia a
    # UNA tasa, así que el nivel contaminado es una segunda pasada de la misma
    # forma y no un paso nuevo. La limpia cae en `results_for(0.0, "campaign",
    # True)` y la contaminada en `results_for(NOISE_REPORTED, "campaign", True)`,
    # que es exactamente la raíz que el informe y el latente leen y que hasta acá
    # no escribía nadie --- por eso sus celdas contaminadas decían «no hay
    # corridas». El `rho0p2` de estas rutas sale de `NOISE_REPORTED` y del
    # formato de `results_for`, y `tests/test_steps.py` lo vuelve a componer.
    #
    # `shard_paths(None, pilot=True)` da `runs.jsonl` y `shard.json` bajo la raíz
    # de cada pasada, al lado va `summary.json`, y `Probe_results.json` sale un
    # directorio más arriba de la raíz LIMPIA --- `campaign()` lo ancla en
    # `results_for(0.0, ...)` para las dos, así que es un solo archivo y lo
    # escribe la última pasada, igual que el barrido con sus cinco niveles. Los
    # archivos y no el directorio: el informe escribe sus figuras y su
    # `report.md` adentro de estas mismas raíces, y declarar el directorio entero
    # haría que un `report.md` escrito por la campaña --- que no lo escribe ---
    # se leyera como suyo.
    # Los pesos sí son un directorio, uno por pasada: `keep_median` los nombra
    # por brazo, transferencia y semilla, y `keeps_checkpoints` es verdadero en
    # las dos tasas --- son justamente las dos que el latente dibuja.
    # Y el cuaderno mismo, que se ejecuta `--inplace` una sola vez: las dos
    # pasadas viven adentro de una ejecución, así que la salida ejecutada muestra
    # las dos. Correrlo dos veces le pisaría a la primera lo único que deja.
    "campaign-local": {"module": "MIL_CREDA_Benchmark.steps",
                       "function": "campana",
                     "advances": 6,
                     "reads": ["Results/Benchmark/ceilings.json"],
                     "produces": ["Results/Pilot/Benchmark/runs.jsonl",
                                  "Results/Pilot/Benchmark/summary.json",
                                  "Results/Pilot/Benchmark/shard.json",
                                  "Results/Pilot/Noise/rho0p2/runs.jsonl",
                                  "Results/Pilot/Noise/rho0p2/summary.json",
                                  "Results/Pilot/Noise/rho0p2/shard.json",
                                  "Results/Pilot/Probe_results.json",
                                  "Models/Pilot/Benchmark",
                                  "Models/Pilot/Noise/rho0p2",
                                  "Notebooks/Benchmark_Campaign_v1.ipynb"],
                     "placement": "remote",
                     "job": "campaign",
                     "service": "kaggle"},
    # Sólo lee y presenta --- la llamada que corre la búsqueda está comentada
    # adentro del cuaderno --- y aun así escribe: se ejecuta `--inplace`, así
    # que su propio cuaderno es su raíz y la única.
    "search-report": {"module": "MIL_CREDA_Benchmark.steps",
                      "function": "informe_de_busqueda",
                     "advances": 3,
                     "reads": ["Results/Benchmark/ceilings.json"],
                     "produces": ["Notebooks/Benchmark_Search_Report_v1.ipynb"],
                     "placement": "local"},
    # Dibuja sobre la corrida vigente (`contamination.in_force(0.0,
    # "campaign")["root"]`), que es la completa si existe y el ensayo si no:
    # por eso las dos escalas. Escribe `curves/*.pdf`, `report.txt` y
    # `report.md` en esa raíz, y una segunda tanda de curvas bajo
    # `results_for(RHO, "campaign", ES_ENSAYO)` con `RHO = NOISE_REPORTED`.
    "report": {"module": "MIL_CREDA_Benchmark.steps", "function": "informe",
                     "advances": 7,
                     "reads": ["Results/Benchmark/runs.jsonl",
                               "Results/Benchmark/summary.json",
                               "Results/Noise/rho0p2/runs.jsonl",
                               "Results/Benchmark/ceilings.json"],
                     "produces": ["Results/Benchmark/curves",
                                  "Results/Benchmark/report.txt",
                                  "Results/Benchmark/report.md",
                                  "Results/Pilot/Benchmark/curves",
                                  "Results/Pilot/Benchmark/report.txt",
                                  "Results/Pilot/Benchmark/report.md",
                                  "Results/Noise/rho0p2/curves",
                                  "Results/Pilot/Noise/rho0p2/curves",
                                  "Notebooks/Benchmark_Report_v1.ipynb"],
                     "placement": "local"},
    # Las dos mitades van por escala. La limpia --- `latent/grid.pdf`,
    # `latent/correspondence.pdf`, `latent.json`, `latent.md` --- sale de
    # `results_for(0.0, "campaign", ES_ENSAYO)`, y hasta hace poco salía de
    # `config.RESULTS` a secas: el cuaderno LEÍA los pesos por escala y ESCRIBÍA
    # siempre en el árbol completo, así que un análisis de ensayo se dibujaba
    # encima del de la corrida completa. Por eso están las dos escalas acá.
    # La mitad contaminada ya era por escala: `results_for(RHO, "campaign",
    # ES_ENSAYO) / "latent"`, con `RHO = NOISE_REPORTED`.
    "latent": {"module": "MIL_CREDA_Benchmark.steps", "function": "latente",
                     "advances": 8,
                     "reads": ["Results/Benchmark/runs.jsonl",
                               "Results/Benchmark/summary.json",
                               "Models/Benchmark",
                               "Models/Noise/rho0p2"],
                     "produces": ["Results/Benchmark/latent",
                                  "Results/Benchmark/latent.json",
                                  "Results/Benchmark/latent.md",
                                  "Results/Pilot/Benchmark/latent",
                                  "Results/Pilot/Benchmark/latent.json",
                                  "Results/Pilot/Benchmark/latent.md",
                                  "Results/Noise/rho0p2/latent",
                                  "Results/Pilot/Noise/rho0p2/latent",
                                  "Notebooks/Benchmark_Latent_v1.ipynb"],
                     "placement": "local"},
    # El eje de ruido. `noise-report` y `noise-diagnostic-report` sólo leen y
    # dibujan; `noise-diagnostic` sí corre -- una búsqueda sobre una
    # transferencia y dos brazos -- y está acá porque es local y barato, a
    # diferencia de la campaña contaminada, que es un envío y necesita su propia
    # autorización por lanzamiento.
    #
    # Las cuatro llevan `advances`, y el lugar es el que el dueño del
    # repositorio le da al eje: EN EL MEDIO del recorrido y no al costado. El
    # barrido y su informe van entre la búsqueda y la campaña ---4 y 5---, y el
    # diagnóstico y el suyo cierran ---9 y 10---, después del latente. Acá
    # estuvo escrito lo contrario ---que el eje era «un ejercicio al costado»---
    # y esa razón no era de nadie: la inventó quien escribió el comentario.
    #
    # Los diez ordinales se comprobaron contra la cadena que `steps.predecesores`
    # DERIVA de `reads` y `produces`, y ninguno deja a un paso adelante de algo
    # que lee. `tests/test_steps.py` rehace esa cuenta y se pone en rojo si una
    # de las dos cosas se mueve sin la otra: un ordinal que adelantara a su
    # predecesor es peor que ningún ordinal, porque el gate lo dejaría correr.
    # Corre `Benchmark_Noise_Sweep_v1.ipynb` y no computa en su lugar. Una campaña
    # por nivel, todas con `kind="curve"` y con el `pilot` que el cuaderno deriva
    # de `config.is_pilot_scale()`; el paso se niega si esa escala no es la del
    # ensayo, así que estas raíces son las de ensayo y no pueden ser otras.
    #
    # El directorio `curve/` de `results_for` cubre los cinco `rho*` y el
    # `Probe_results.json` que `campaign()` deja en su padre. Los pesos también
    # son un directorio y no un vacío: `keeps_checkpoints` es verdadero en 0.0 y
    # en 0.2, así que dos de los cinco niveles sí escriben checkpoints.
    #
    # Y el cuaderno mismo, ejecutado `--inplace` una sola vez: los cinco niveles
    # viven adentro de una ejecución, así que la salida ejecutada los muestra a
    # los cinco.
    "noise-sweep": {"module": "MIL_CREDA_Benchmark.steps",
                    "function": "barrido_de_ruido",
                    "advances": 4,
                    "reads": ["Results/Benchmark/ceilings.json"],
                    "produces": ["Results/Pilot/Noise/curve",
                                 "Models/Pilot/Noise/curve",
                                 "Notebooks/Benchmark_Noise_Sweep_v1.ipynb"],
                     "placement": "remote",
                     "job": "noise-sweep",
                     "service": "kaggle"},
    # Sólo lee y dibuja, y aun así deja tres cosas: su cuaderno ejecutado y las
    # dos que escriben sus celdas, `degradation.pdf` (por `figures.noise_curves`
    # sobre `config.noise_axis_for(ES_ENSAYO) / "degradation"`, con el `.pdf` de
    # `emit`) y `degradation.json`. Las dos siguen al barrido que resumen, y por
    # eso están las dos escalas: el cuaderno componía esa ruta a mano desde
    # `PRODUCT` y el resumen de un barrido de ensayo caía en el árbol completo.
    "noise-report": {"module": "MIL_CREDA_Benchmark.steps",
                     "function": "informe_de_ruido",
                     "advances": 5,
                     "reads": ["Results/Noise/curve"],
                     "produces": ["Results/Noise/degradation.pdf",
                                  "Results/Noise/degradation.json",
                                  "Results/Pilot/Noise/degradation.pdf",
                                  "Results/Pilot/Noise/degradation.json",
                                  "Notebooks/Benchmark_Noise_Report_v1.ipynb"],
                     "placement": "local"},
    # Corre `Benchmark_Noise_Diagnostic_Search_v1.ipynb` y no computa en su lugar.
    # Un solo archivo de datos, y es todo lo que escribe además de su cuaderno: la
    # re-búsqueda que paga NO gobierna ningún registro
    # (`governs_the_ceilings_record` es falso bajo contaminación y sobre una
    # transferencia sola) y el motor `optuna` no deja parcial. El destino es
    # `config.noise_axis_for(ES_ENSAYO)`, con el `pilot` que el cuaderno deriva de
    # `config.is_pilot_scale()`; el paso se niega si esa escala no es la del
    # ensayo, así que la raíz es la de ENSAYO y no puede ser otra --- escrito bajo
    # `Results/Noise/` a secas pisaría el diagnóstico de la corrida completa con
    # números de ensayo.
    "noise-diagnostic": {"module": "MIL_CREDA_Benchmark.steps",
                         "function": "diagnostico_de_ruido",
                         "advances": 9,
                         "reads": ["Results/Noise/curve"],
                         "produces": [
                             "Results/Pilot/Noise/diagnostic.json",
                             "Notebooks/Benchmark_Noise_Diagnostic_Search_v1.ipynb"],
                     "placement": "remote",
                     "job": "noise-diagnostic",
                     "service": "kaggle"},
    # Presenta el `diagnostic.json` que ya existe y no computa nada, así que su
    # cuaderno ejecutado es su única raíz.
    "noise-diagnostic-report": {"module": "MIL_CREDA_Benchmark.steps",
                                "function": "informe_del_diagnostico",
                                "advances": 10,
                                "reads": ["Results/Noise/diagnostic.json"],
                                "produces": [
                                    "Notebooks/Benchmark_Noise_Diagnostic_Report_v1.ipynb"],
                     "placement": "local"},
}
