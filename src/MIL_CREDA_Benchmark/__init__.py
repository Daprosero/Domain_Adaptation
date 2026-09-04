"""Everything the bounded comparison needs, and nothing the method needs.

This package is deliberately NOT `MIL_CREDA`. Every module of that package
declares `__provenance__` binding it to the revision it implements, and the
verification reads all of them: a benchmark harness placed inside it would be
read as a module of the method missing its provenance, and stamping a fake one on
plumbing that implements no equation would hollow out the only check that keeps
the code tied to the mathematics.

So the separation the name promises is real. Nothing here is part of the
formulation; deleting this package leaves MIL-CREDA intact.

What it does declare is which revision it was built against and which sections of it
each arm exercises. Without that, a new revision raises a question nobody can answer:
does this change oblige the bench to change? With it, the verification names the arms
a changed section reaches.
"""

__benchmark__ = {
    "revision": "research-concept-r17.md",
    "arms": {
        # No section: this arm trains on instances with no term of the formulation.
        "A": {"sections": []},
        "B": {"sections": ["3"]},
        # The baseline is prior work; it exercises none of the proposal's sections.
        "C": {"sections": []},
        "D": {"sections": []},
        "E": {"sections": ["1", "2", "3", "5"]},
        "F": {"sections": ["1", "2", "3", "5"]},
        "G": {"sections": ["1", "2", "3", "4", "5"]},
        # The three selecting arms compute exactly what G computes, over a subset
        # of each bag's instances. Same sections, different budget.
        "SU": {"sections": ["1", "2", "3", "4", "5"]},
        "SA": {"sections": ["1", "2", "3", "4", "5"]},
        "SK": {"sections": ["1", "2", "3", "4", "5"]},
    },
    # The ceiling search, declared as the experiment it is. A value chosen by
    # looking at outcomes needs everything a run needs, and the three below are
    # the ones that go invisible: without them a ceiling found at pilot scale, or
    # on the material the verdict rests on, or by a tie nobody wrote a rule for,
    # is indistinguishable from one that was measured.
    "search": {
        "what": "the ceiling of the adaptation coefficient, measured on every "
                "transfer and shared by all arms of that family within a "
                "transfer -- never per arm, or the term and the coefficient "
                "could not be told apart. Nothing is inherited: the six are each "
                "measured, so the pooled fallback is unreachable. The growth "
                "rate is not searched: it stays at RAMP_DELTA, which is CREDA's "
                "own, because a second free dimension amplifies the imbalance "
                "between the two families rather than resolving it, and ceiling "
                "and growth rate are confounded -- a high ceiling reached slowly "
                "and a low one reached fast give similar trajectories",
        "requiredScale": {"epochs": 20, "trials": 30},
        "role": "valid",
        # Reemplaza al desempate, que sobre un rango continuo no se activaria
        # nunca: dos evaluaciones no dan el mismo numero, asi que el ganador lo
        # pondria el ultimo decimal. La meseta la define la resolucion del
        # instrumento -- una bolsa de las veinte del rol de busqueda.
        "tieRule": "within the plateau the smallest ceiling wins: the same "
                   "outcome for less adaptation is the weaker claim, and a "
                   "search should not hand a term more weight than the "
                   "measurement asked for. The plateau is what the criterion "
                   "cannot tell apart -- one bag out of the search role's "
                   "twenty -- and not an exact tie, which on a continuous range "
                   "would never occur",
        "record": "Results/Benchmark/ceilings.json",
    },
    # What the protocol assumes about the prediction being measured. These are what
    # a change of reach destroys while leaving every arm intact: a formulation that
    # moved from deciding a class to estimating a quantity would leave all of the
    # above standing and every dimension below meaningless.
    # What the report is made of, so the verification can check the document a
    # human reads without knowing one word of this field. It names which calls
    # render a measurement, which produce a conclusion, and which way each
    # dimension wins — and nothing else needs to be guessed from that.
    "report": {
        "renderers": [
            "tables.render",
            "tables.render_rungs",
            "tables.render_readings",
            "tables.render_correspondence",
            # The ceiling search's whole grid, not only its winner. The scalar that
            # governs every table below is chosen here, so the report has to show
            # what it was chosen over: a ceiling that wins among four identical
            # scores and one that wins by a real difference are the same number and
            # not the same evidence.
            "tables.render_ceilings",
            "tables.render_ceilings_by_transfer",
            # El eje de ruido. `render_at` es el MISMO `render` sobre el
            # registro de una tasa contaminada y no un renderer paralelo: dos
            # funciones que dibujan la misma tabla son dos cosas que se pueden
            # desalinear, y el lector no tendría cómo saber cuál se movió.
            # La sección de tiempo, sus dos formas. Ninguna de las dos estaba
            # declarada, así que la mitad del informe que MEJOR se porta --- la
            # que se niega a promediar `seconds` y saca fila por corrida --- era
            # justo la que ningún control miraba. Declarar sólo la contaminada
            # la dejaba sin conclusión a los ojos del contrato, que es como
            # apareció.
            "tables.render_per_run",
            # Y su forma inline: mediana con rango min-max, colapsando el eje de
            # semillas dentro de un entorno. Dos renderers y no un parámetro
            # porque son dos afirmaciones distintas, y el contrato nombra cuál
            # se usó.
            "tables.render_per_run_summary",
            "tables.render_at",
            # Y su gemela por corrida, que NO es la misma tabla: `render_at`
            # promedia y `cells` se niega ante una dimensión `perRun`, así que
            # la sección contaminada de `seconds` no tenía tabla que imprimir
            # mientras su conclusión sí promediaba.
            "tables.render_per_run_summary_at",
            "tables.render_noise",
            "tables.render_diagnostic",
            "tables.render_readings_contaminated",
            "tables.render_gains_at",
            "tables.render_rungs_at",
        ],
        "conclusions": [
            "tables.conclusion",
            "tables.conclusion_rungs",
            "tables.conclusion_geometry",
            "tables.conclusion_distances",
            "tables.conclusion_separability",
            "tables.conclusion_mass",
            "tables.conclusion_attention",
            "tables.conclusion_correspondence",
            "tables.conclusion_ceilings",
            "tables.conclusion_ceilings_by_transfer",
            # `conclusion_versus_clean` informa la diferencia entre las dos
            # tasas, que es lo único que ninguna de las dos tablas contiene por
            # separado. Nunca enumera la tabla que tiene al lado: una conclusión
            # que repite su propia tabla dejó de concluir.
            "tables.conclusion_noise",
            # No computa nada y esa es su afirmación: `render_per_run` ya se
            # negó a promediar, y una conclusión que después imprimiera
            # «mejor/peor» sobre las mismas lecturas devolvería en prosa lo que
            # la tabla acaba de declinar en números.
            "tables.conclusion_per_run",
            "tables.conclusion_versus_clean",
            "tables.conclusion_diagnostic",
            "tables.conclusion_readings_versus_clean",
            "tables.conclusion_weighting_under_noise",
            "tables.conclusion_rungs_versus_clean",
        ],
        # One call that takes a record and returns {label: text}. It exists so the
        # verification can run every conclusion over permuted numbers without
        # knowing a single signature: a conclusion whose text survives that is tied
        # to nothing, exactly as an assertion that cannot fail proves nothing.
        "conclusionEntry": "tables.conclusions",
        # Qué valor busca cada lectura, calculado de la configuración. Un lector
        # que no conoce la métrica necesita el hito contra el que se compara —el
        # azar, una cota, el acuerdo entre transferencias— y no solo la dirección.
        "objectiveEntry": "tables.objective",
        # The calls that produce or show a picture. Naming them here is what lets
        # the verification ask whether a figure actually rendered without knowing
        # one word about who draws — a check that recognised matplotlib would go
        # blind the day a notebook used anything else.
        #
        # Every entry earns its place by drawing, not by being in this module:
        # `inline` is what puts a figure in front of a reader, the four builders
        # return one, and `plt.show` is here because the verification notebook
        # draws its bound with bare matplotlib rather than through `figures`.
        # `emit` files the vector copy and is declared for the same reason the
        # others are — a cell that archives a figure and never shows it has
        # reported a filename, and with `emit` named that comes out as a finding
        # instead of passing quietly.
        "figures": [
            "figures.inline",
            "figures.emit",
            "figures.adaptation_curves",
            "figures.supervised_curves",
            "figures.contribution_curves",
            "figures.noise_curves",
            "latent.latent_grid",
            "latent.correspondence_grid",
            "latent.projection",
            "plt.show",
        ],
        # Constants that name a subset of another constant. Each one is a selection
        # somebody wrote out, which is legitimate only when the rule that fixed it
        # looks at no outcome — so the rule is stated here and can be argued with,
        # instead of being inferred from the shape of a list.
        "selections": {
            "SEEDS": "el piloto: un prefijo de FULL_SEEDS, y las dos escalas se "
                     "informan juntas en cada tabla",
            "PILOT_SEARCH_SEEDS": "la escala del ensayo de la búsqueda: una "
                                  "semilla, fijada antes de medir nada y sin "
                                  "mirar ningún resultado. No elige un techo — "
                                  "escribe a `ceilings.pilot.json`, al que el "
                                  "registro completo le gana siempre — así que "
                                  "no hay outcome que pudiera haberla fijado",
            "NOISE_REPORTED": "el nivel contaminado que report y latent muestran "
                              "al lado de 0.0: el punto medio de NOISE_LEVELS, "
                              "aritmética y no resultado. Elegido después de ver "
                              "la curva sería el nivel que más favorece al método",
            "NOISE_TRANSFER": "donde se mide la curva de degradación: una sola "
                              "transferencia, la de menor brecha de dominio. La "
                              "regla es del instrumento — una transferencia ya "
                              "cerca de su piso en 0.0 no tiene de dónde caer — y "
                              "la brecha es propiedad del material, no de ninguna "
                              "medición",
            "NOISE_DIAGNOSTIC_ARMS": "los dos métodos completos, uno por familia, "
                                     "y los únicos que llevan el coeficiente: A y "
                                     "B no tienen término de adaptación al que "
                                     "re-buscarle un techo, y C, E y F son "
                                     "ablaciones que multiplicarían la búsqueda "
                                     "sin agregar diagnóstico",
            "NOISE_DIAGNOSTIC_LEVEL": "el tope del rango, fijado antes de que la "
                                      "curva exista. En el extremo el coeficiente "
                                      "está bajo la máxima presión, así que un "
                                      "techo re-buscado que no recupera nada ahí "
                                      "no recupera nada en ningún lado",
            "LATENT_PANELS": "los métodos que alinean, elegidos por lo que computan "
                             "y no por lo que puntúan; los pisos entran por medición",
            "BAG_PANELS": "el peldaño donde vive el término local: piso, sin el "
                          "término y con él, elegidos por el mecanismo",
            "SEARCH_SEEDS": "tres repeticiones, elegidas por cuenta y no por "
                            "resultado: con 20 bolsas de validación por "
                            "transferencia, una sola semilla deja la granularidad "
                            "en cinco puntos y el argmax entre cinco celdas lo "
                            "decide el ruido. Tres es el piso para que la "
                            "elección signifique algo",
            "SEARCH_TRANSFERS": "una transferencia fácil y una difícil, elegidas "
                                "por dificultad y no por resultado, para que el "
                                "techo no quede ajustado a una sola. El veredicto "
                                "se lee igual sobre las seis: los roles ya son "
                                "disjuntos por bolsa",
        },
        # Where the record a conclusion is exercised against lives.
        "record": "latent.json",
        # What a run leaves under `Results/`. Named so a later artefact — a second
        # experiment arriving as a file, with its own scale and its own material
        # role — has to be written down instead of appearing unremarked.
        #
        # `Benchmark/` is declared as a directory rather than file by file: the
        # campaign writes its record, its readable summary and its archived
        # figures there together, and they are one output. `ceilings.json` will
        # land beside them, which is precisely why it gets its own line — the
        # ceiling search is a separate experiment and reads as one here.
        "records": [
            "Results/Probe_results.json",
            "Results/Benchmark",
            "Results/Benchmark/ceilings.json",
            "Results/local_distance_bound.pdf",
            # The noise axis leaves its own artefacts, and naming them here is
            # what stops `undeclaredRecords` from reporting them as a second
            # experiment nobody accounted for -- which, until they are named, is
            # exactly what they are.
            "Results/Noise",
            "Results/Noise/degradation.json",
            "Results/Noise/diagnostic.json",
        ],
        # The terms Eq. (39) combines, and the dimension carrying their share.
        #
        # `contribution` on its own is the numerator: it cannot separate a term
        # that commanded nothing from a term that was scaled to nothing, and both
        # print small. Eq. (18) is divided by B_src precisely so the three terms
        # can be read against each other, so the ratio is the quantity the
        # normalization exists to make meaningful.
        #
        # Two terms and not three: `supervised` and `contribution` are what an
        # arm's objective is made of here, because the harness applies one shared
        # coefficient to the global and local terms together rather than the two
        # of Eq. (39) separately. Splitting them would need two coefficients, and
        # that is a change to what the experiment is, not a declaration.
        "components": {
            "terms": ["supervised", "contribution"],
            "share": "adaptationShare",
        },
        "dimensions": {
            "targetAccuracy": "higher",
            "sourceAccuracy": "higher",
            "seconds": "lower",
            "contribution": "descriptive",
            "supervised": "descriptive",
            "adaptationShare": "descriptive",
            "peakMiB": "descriptive",
            "parameters": "descriptive",
            "geometry.ratio": "lower",
            "geometry.crossDomainSameClass": "descriptive",
            "geometry.betweenClasses": "descriptive",
            "domainSeparability": "toward-chance",
            "correspondence.massOnTrueClass": "higher",
            "attentionSpread": "descriptive",
        },
    },
    "premises": {
        "prediction": "a single class per bag, chosen from CLASSES alternatives",
        "unit": "the bag, for every arm, including the instance-unit ones",
        "metric": "accuracy over the evaluation bags of the target domain",
        "direction": "higher is better",
    },
    # `shards.declaration()` reads this block; `shards.merge()` refuses every
    # merge until it exists. The axis a shard is split on, and which of a
    # run's dimensions may be pooled across machines, read per environment,
    # read per run, or must match exactly before shards are trusted to be one
    # campaign split up.
    #
    # `axis`: the seed. `tools/distribute.py`'s own `shard_seeds()` splits the
    # seed list and nothing else — every arm of every transfer within one seed
    # runs on one machine, so no rung's subtraction ever crosses a hardware
    # boundary.
    #
    # The four groups below are not a guess: a replication (`A` on one
    # machine, `B` on a second, `C` back on the first as a same-machine
    # control — one arm, `G`, one transfer, `M->U`, seed 7) measured all eight
    # of `config.DIMENSIONS` three times and compared the parsed JSON floats
    # with exact `==`.
    #
    # Which machines those were is deliberately not written here. What the
    # replication established is a relation — two runs on one machine, one on
    # another — and that relation is the whole argument; the identities added
    # nothing to it. Naming them by hand would also be this comment claiming a
    # distribution nobody has decided for the current experiment, when the
    # record of what ran where belongs to whatever actually distributes the
    # work and not to a paragraph beside the declaration.
    #
    # `poolable`: `sourceAccuracy`, `targetAccuracy`, `contribution`,
    # `supervised`, `adaptationShare` and `parameters` came back bit-identical
    # on all three runs, including across the two different machines. A
    # quantity measured under the same seed and the same code does not change
    # because a different machine computed it, so all six pool freely across
    # every shard that arrived.
    #
    # `perEnvironment`: empty. Nothing the replication measured turned out to
    # be stable on one machine and different from another's — the two
    # candidates for this group, `seconds` and `peakMiB`, turned out not even
    # to be stable on the *same* machine (see `perRun` below), so this group
    # is declared and left empty rather than removed: a dimension that is
    # genuinely a property of the environment, and not of one run inside it,
    # has a place to go without inventing a fourth category. Whether any
    # dimension actually belongs here is a claim only another measurement can
    # settle, the same way this one settled `poolable` and `perRun`.
    #
    # `perRun`: `seconds` and `peakMiB` differed on every one of the three
    # runs — including `A` vs `C`, the same machine measured twice. That is
    # what rules out `perEnvironment` for them: that label claims the value
    # belongs to the machine, which implies re-running there reproduces it,
    # and the control run shows it does not. `shards.merge()` reports each
    # such dimension as every run's own reading rather than averaging it —
    # see `merge()`'s own docstring for why a mean is not offered here.
    #
    # `identicalAcrossShards`: `epochs`, `ceilings` and `ceilingsByTransfer`.
    # The ceilings are here because the search is what most recently changed
    # them: two shards straddling that search would merge into one table with
    # adaptation inert on one half and not on the other, and nothing would
    # object. Both are real flat top-level fields of every stamp, which is the
    # property the next paragraph is about. `commit` and `codeDigest` were
    # approved alongside it, but neither is a name `shards.disagreements()`
    # can actually check: both live nested at `evidence.commit` /
    # `evidence.codeDigest` on a shard's stamp, and `disagreements()` (the
    # forge's own `shard_io.py`, re-exported unchanged) compares
    # `stamp.get(field)` — a flat top-level lookup, never a dotted path.
    # Declaring `"commit"` or `"codeDigest"` here would resolve to `None` on
    # every shard and silently pass regardless of whether the shards actually
    # agree, which is worse than not checking at all: it would read as a
    # guarantee that was never enforced. `epochs` is a real top-level field of
    # every stamp, so it is the only name from that approval that belongs
    # here; the other two need either a dotted-path-aware `disagreements()` in
    # the forge or a different field name, and that decision was not this
    # task's to make.
    #
    # `parameters` was weighed against this same group and kept out of it for
    # the identical reason: `harness.run_one` writes `parameters` once per
    # *run*, into the dict that becomes a line of `runs.jsonl` — it is never
    # part of the dict `harness.write_shard_stamp` writes to a shard's stamp
    # (`shard`, `env`, `environment`, `seeds`, `epochs`, `revision`,
    # `ceilings`, `evidence`). `disagreements()`'s flat `stamp.get("parameters")`
    # would therefore resolve to `None` on every shard, same as `commit` and
    # `codeDigest` above, and "agree" regardless of whether two shards'
    # models actually share an architecture — a check that always passes is
    # not a check. Two shards reporting different parameter counts would be a
    # real reason to refuse a merge (different architectures, not different
    # hardware), but `identicalAcrossShards` cannot be that reason today
    # without the same forge change `commit`/`codeDigest` would need.
    # `poolable` is the only place left that can actually hold it, and it
    # costs nothing to be there: the replication measured `parameters`
    # bit-identical across every run, so averaging it returns exactly the
    # constant it already was.
    "distribution": {
        "axis": "seed",
        "poolable": ["sourceAccuracy", "targetAccuracy", "contribution",
                     "supervised", "adaptationShare", "parameters"],
        "perEnvironment": [],
        "perRun": ["seconds", "peakMiB"],
        # `labelNoise` refuses the merge that no averaging could repair: two shards
    # contaminated at different rates are two experiments, and one table drawn
    # over both is a table nobody can attribute. It is a flat top-level field of
    # every stamp, which is what `disagreements()` needs to be able to see it.
    "identicalAcrossShards": ["epochs", "ceilings", "ceilingsByTransfer",
                              "labelNoise"],
    },
    # Which module carries the runtime, so a reading about this repository's
    # environment is about the module that actually imports it. The same two
    # values `tools/kaggle/ceiling-search/run-config.json` already fixes.
    "entry": {
        "module": "MIL_CREDA_Benchmark.harness",
        "function": "run_pilot",
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

# Los pasos locales que la forja puede ejecutar sola, en el venv de este
# repositorio. Literal aparte de `__benchmark__` por la misma razón que
# `__levels__`: declararlos no vuelve "declarado" un repositorio que todavía no
# lo está.
#
# `Benchmark_Campaign_v1.ipynb` NO está acá, y la ausencia es la decisión: ese
# cuaderno lanza al servicio remoto, y un lanzamiento remoto no se ejecuta sin
# una aprobación humana por envío. Un paso local que lo corriera sería
# exactamente la vuelta que esa aprobación existe para impedir.
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
# Cada raíz sale de leer a dónde escribe el paso --- `config.results_for`,
# `config.models_for`, `config.ceilings_record_for`, `harness.shard_paths`, y
# las celdas del cuaderno que ejecuta --- y nunca de copiar la del vecino.
#
# **La escala va adentro de la raíz, y no hay una sola raíz que cubra las
# dos.** `results_for` mete el segmento `Pilot/` ARRIBA de la forma compartida
# (`base = RESULTS.parent`, y recién ahí `/ "Pilot"`), así que el único
# ancestro común de `Results/Benchmark` y `Results/Pilot/Benchmark` es
# `Results/`, que además contiene el árbol de todos los demás pasos:
# declararlo volvería `own` a cualquier escritura en cualquier lado y apagaría
# el guarda sin decir que lo apagó. Entonces la declaración es específica de
# escala. Los pasos que fijan `pilot=True` en su propio código
# (`search-pilot`, `campaign-local`, `noise-sweep`, `noise-diagnostic`)
# declaran la raíz de ENSAYO y ninguna otra --- el día que uno escriba a
# escala completa eso es `foreign`, que es exactamente lo que hay que ver ---
# y los cuadernos, que dibujan sobre la corrida que esté vigente y por lo
# tanto pueden caer de cualquiera de los dos lados, declaran las dos, una por
# escala.
#
# Las raíces son literales porque la forja lee este archivo con `ast` y sin
# importarlo. Las que llevan un `rho` adentro salen de `config.NOISE_REPORTED`
# (0.2) y del formato de `results_for` (`f"rho{rate:g}".replace(".", "p")`);
# `tests/test_steps.py` las vuelve a componer desde `config` y se pone en rojo
# si alguna de las dos cosas cambia.
__steps__: dict = {
    # Corre la suite adentro del cuaderno y dibuja la cota local. Dos clases de
    # raíz: el dato que produce y el cuaderno que ejecuta en el lugar.
    # `ROOT / "MIL-CREDA" / "Results" / "local_distance_bound"` en la celda de
    # la cota, con el `.pdf` que le pone `figures.emit`.
    "verification": {"module": "MIL_CREDA_Benchmark.steps", "function": "verificacion",
                     "advances": 1,
                     "produces": ["Results/local_distance_bound.pdf",
                                  "Notebooks/verification.ipynb"]},
    # `harness.run_search(pilot=True)` -> `ceilings_in_force` -> el motor
    # `optuna`, que escribe un solo archivo: `config.ceilings_record_for(True)`,
    # o sea `CEILINGS_PILOT_RECORD`. Es OTRO árbol que el de `results_for`: el
    # registro de ensayo se separa por nombre de archivo y no por directorio
    # (ver el comentario de `shard_paths`), así que cae al lado del registro
    # completo y no bajo `Results/Pilot/`. El motor por grilla dejaría además un
    # `.partial.json` acá; no está declarado porque no es el motor que
    # `config.SEARCH_ENGINE` nombra.
    "search-pilot": {"module": "MIL_CREDA_Benchmark.steps",
                     "function": "ensayo_de_busqueda",
                     "advances": 2,
                     "produces": ["Results/Benchmark/ceilings.pilot.json"]},
    # `harness.campaign()` con `pilot=True` y `kind="campaign"`:
    # `shard_paths(None, pilot=True)` da `runs.jsonl` y `shard.json` bajo
    # `results_for(0.0, "campaign", True)`, al lado va `summary.json`, y
    # `Probe_results.json` sale un directorio más arriba
    # (`results_for(...).parent`). Los tres archivos y no el directorio: el
    # informe escribe sus figuras y su `report.md` adentro de esta misma raíz, y
    # declarar el directorio entero haría que un `report.md` escrito por la
    # campaña --- que no lo escribe --- se leyera como suyo.
    # Los pesos sí son un directorio: `keep_median` los nombra por brazo,
    # transferencia y semilla, y `keeps_checkpoints(0.0)` es verdadero.
    "campaign-local": {"module": "MIL_CREDA_Benchmark.steps",
                       "function": "campana",
                     "advances": 4,
                     "produces": ["Results/Pilot/Benchmark/runs.jsonl",
                                  "Results/Pilot/Benchmark/summary.json",
                                  "Results/Pilot/Benchmark/shard.json",
                                  "Results/Pilot/Probe_results.json",
                                  "Models/Pilot/Benchmark"]},
    # Sólo lee y presenta --- la llamada que corre la búsqueda está comentada
    # adentro del cuaderno --- y aun así escribe: se ejecuta `--inplace`, así
    # que su propio cuaderno es su raíz y la única.
    "search-report": {"module": "MIL_CREDA_Benchmark.steps",
                      "function": "informe_de_busqueda",
                     "advances": 3,
                     "produces": ["Notebooks/Benchmark_Search_v1.ipynb"]},
    # Dibuja sobre la corrida vigente (`contamination.in_force(0.0,
    # "campaign")["root"]`), que es la completa si existe y el ensayo si no:
    # por eso las dos escalas. Escribe `curves/*.pdf`, `report.txt` y
    # `report.md` en esa raíz, y una segunda tanda de curvas bajo
    # `results_for(RHO, "campaign", ES_ENSAYO)` con `RHO = NOISE_REPORTED`.
    "report": {"module": "MIL_CREDA_Benchmark.steps", "function": "informe",
                     "advances": 5,
                     "produces": ["Results/Benchmark/curves",
                                  "Results/Benchmark/report.txt",
                                  "Results/Benchmark/report.md",
                                  "Results/Pilot/Benchmark/curves",
                                  "Results/Pilot/Benchmark/report.txt",
                                  "Results/Pilot/Benchmark/report.md",
                                  "Results/Noise/rho0p2/curves",
                                  "Results/Pilot/Noise/rho0p2/curves",
                                  "Notebooks/Benchmark_Report_v1.ipynb"]},
    # La mitad limpia va a `config.RESULTS` a secas --- `latent/grid.pdf`,
    # `latent/correspondence.pdf`, `latent.json`, `latent.md` --- y no a
    # `results_for(0.0, "campaign", ES_ENSAYO)`: el cuaderno LEE los pesos por
    # escala y ESCRIBE siempre en el árbol completo. Queda declarado como está
    # escrito y no como uno querría que estuviera; el día que el cuaderno pase
    # a escribir por escala, esta raíz es la que hay que mover.
    # La mitad contaminada sí es por escala: `results_for(RHO, "campaign",
    # ES_ENSAYO) / "latent"`, con `RHO = NOISE_REPORTED`.
    "latent": {"module": "MIL_CREDA_Benchmark.steps", "function": "latente",
                     "advances": 6,
                     "produces": ["Results/Benchmark/latent",
                                  "Results/Benchmark/latent.json",
                                  "Results/Benchmark/latent.md",
                                  "Results/Noise/rho0p2/latent",
                                  "Results/Pilot/Noise/rho0p2/latent",
                                  "Notebooks/Benchmark_Latent_v1.ipynb"]},
    # El eje de ruido. `noise-report` y `noise-diagnostic-report` sólo leen y
    # dibujan; `noise-diagnostic` sí corre -- una búsqueda sobre una
    # transferencia y dos brazos -- y está acá porque es local y barato, a
    # diferencia de la campaña contaminada, que es un envío y necesita su propia
    # autorización por lanzamiento.
    #
    # Ninguna de las cuatro lleva `advances`, y la ausencia es la misma decisión
    # en las cuatro: el eje del ruido no es un peldaño de la secuencia del
    # ensayo sino un ejercicio al costado, y numerarlo lo metería en un orden que
    # el veredicto no recorre. `noise-diagnostic-report` hereda esa decisión de
    # su eje, no de su forma: `report` y `latent` también sólo dibujan y sí
    # avanzan, porque los suyos son los cuadernos del veredicto.
    # Una campaña por nivel, todas con `kind="curve"` y `pilot=True`: el
    # directorio `curve/` de `results_for` cubre los cinco `rho*` y el
    # `Probe_results.json` que `campaign()` deja en su padre. Los pesos también
    # son un directorio y no un vacío: `keeps_checkpoints` es verdadero en 0.0 y
    # en 0.2, así que dos de los cinco niveles sí escriben checkpoints --- lo
    # contrario de lo que dice la docstring del paso, que quedó vieja.
    "noise-sweep": {"module": "MIL_CREDA_Benchmark.steps",
                    "function": "barrido_de_ruido",
                    "produces": ["Results/Pilot/Noise/curve",
                                 "Models/Pilot/Noise/curve"]},
    # Sólo lee y dibuja, y aun así deja tres cosas: su cuaderno ejecutado y las
    # dos que escriben sus celdas, `degradation.pdf` (por `figures.noise_curves`
    # sobre `PRODUCT / "Results" / "Noise" / "degradation"`, con el `.pdf` de
    # `emit`) y `degradation.json`. Las dos van al árbol COMPLETO aunque el
    # barrido que resumen haya corrido en ensayo: el cuaderno compone esa ruta a
    # mano y no por `results_for`.
    "noise-report": {"module": "MIL_CREDA_Benchmark.steps",
                     "function": "informe_de_ruido",
                     "produces": ["Results/Noise/degradation.pdf",
                                  "Results/Noise/degradation.json",
                                  "Notebooks/Benchmark_Noise_v1.ipynb"]},
    # Un solo archivo, y es todo lo que escribe: la re-búsqueda que paga NO
    # gobierna ningún registro (`governs_the_ceilings_record` es falso bajo
    # contaminación y sobre una transferencia sola) y el motor `optuna` no deja
    # parcial. El destino es `results_for(0.0, "curve", True).parents[1]`, la
    # raíz del ENSAYO: escrito bajo `Results/Noise/` a secas pisaría el
    # diagnóstico de la corrida completa con números de ensayo.
    "noise-diagnostic": {"module": "MIL_CREDA_Benchmark.steps",
                         "function": "diagnostico_de_ruido",
                         "produces": ["Results/Pilot/Noise/diagnostic.json"]},
    # Presenta el `diagnostic.json` que ya existe y no computa nada, así que su
    # cuaderno ejecutado es su única raíz.
    "noise-diagnostic-report": {"module": "MIL_CREDA_Benchmark.steps",
                                "function": "informe_del_diagnostico",
                                "produces": [
                                    "Notebooks/Benchmark_Noise_Diagnostic_v1.ipynb"]},
}
