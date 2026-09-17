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
    "arms": {
        "B": {"sections": ["3"]},
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
            "tables.render_correspondence_contaminated",
            # The ceiling search's whole grid, not only its winner. The scalar that
            # governs every table below is chosen here, so the report has to show
            # what it was chosen over: a ceiling that wins among four identical
            # scores and one that wins by a real difference are the same number and
            # not the same evidence.
            "tables.render_ceilings",
            "tables.render_ceilings_by_transfer",
            # La sección de tiempo, sus dos formas. Ninguna de las dos estaba
            # declarada, así que la mitad del informe que MEJOR se porta --- la
            # que se niega a promediar `seconds` y saca fila por corrida --- era
            # justo la que ningún control miraba.
            "tables.render_per_run",
            # Y su forma inline: mediana con rango min-max, colapsando el eje de
            # semillas dentro de un entorno. Dos renderers y no un parámetro
            # porque son dos afirmaciones distintas, y el contrato nombra cuál
            # se usó.
            "tables.render_per_run_summary",
            "tables.render_noise",
            "tables.render_diagnostic",
            # El eje de ruido ya no agrega renderers. Las cinco gemelas que
            # tenía --- `render_readings_contaminated` primero, y después
            # `render_at`, `render_per_run_summary_at`, `render_gains_at` y
            # `render_rungs_at` --- dibujaban la misma tabla sobre el registro
            # de la campaña contaminada, y estaban declaradas aparte sólo para
            # que el chequeo de duplicación no leyera dos renderizaciones del
            # mismo número donde hay dos números distintos. Cada par es ahora
            # una tabla con la columna `Ruido` adelante y dos bloques, `sin` y
            # después `con`: una cantidad, una llamada, una renderización, y
            # nada que el chequeo pueda confundir.
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
            # `conclusion_versus_clean` informa la diferencia entre los dos
            # bloques, que es lo único que la tabla no contiene: pone el
            # material limpio y el contaminado uno sobre otro y no los resta.
            # Nunca enumera la tabla que tiene al lado: una conclusión que
            # repite su propia tabla dejó de concluir.
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
            # Las tres composiciones: una tabla, una conclusión. No calculan
            # nada propio --- juntan la conclusión limpia con la cruzada, con
            # las frases que las dos ya emitían --- y están declaradas igual
            # que las demás porque son las que el cuaderno llama, y lo que la
            # verificación mira es el documento que alguien lee.
            "tables.conclusion_with_noise",
            "tables.conclusion_rungs_with_noise",
            "tables.conclusion_readings_with_noise",
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
            "NOISE_DIAGNOSTIC_ARMS": "el método completo, y el único que lleva el "
                                     "coeficiente: B no tiene término de "
                                     "adaptación al que re-buscarle un techo, y E "
                                     "y F son ablaciones que multiplicarían la "
                                     "búsqueda sin agregar diagnóstico",
            "NOISE_DIAGNOSTIC_LEVEL": "el tope del rango, fijado antes de que la "
                                      "curva exista. En el extremo el coeficiente "
                                      "está bajo la máxima presión, así que un "
                                      "techo re-buscado que no recupera nada ahí "
                                      "no recupera nada en ningún lado",
            "LATENT_PANELS": "los métodos que alinean, elegidos por lo que computan "
                             "y no por lo que puntúan, y el único piso que queda "
                             "declarado, que es lo que los hace legibles: "
                             "«alineado» no se ve sin un «no alineado» al lado",
            "BAG_PANELS": "el peldaño donde vive el término local: piso, sin el "
                          "término y con él, elegidos por el mecanismo",
            "SEARCH_SEEDS": "tres repeticiones, elegidas por cuenta y no por "
                            "resultado: con 20 bolsas de validación por "
                            "transferencia, una sola semilla deja la granularidad "
                            "en cinco puntos y el argmax entre cinco celdas lo "
                            "decide el ruido. Tres es el piso para que la "
                            "elección signifique algo",
            "CEILING_RANGE": "los extremos de CEILING_GRID, que son los dos "
                             "valores por defecto ya declarados: el "
                             "`creda_lambda_special` publicado de CREDA abajo y "
                             "el neutro de la Ec. (39) normalizada arriba. Lo "
                             "que cambia al pasar de rejilla a rango es que "
                             "adentro hay un continuo en vez de cinco puntos; "
                             "los bordes no los eligió ninguna medición",
            "CHECKPOINT_LEVELS": "los niveles cuyos pesos se conservan, y son "
                                 "exactamente los que el cuaderno latente "
                                 "dibuja: 0.0 y NOISE_REPORTED, que es el punto "
                                 "medio del rango. La lista se deriva de lo que "
                                 "el informe ya declara, no de qué nivel salió "
                                 "mejor. Los demás niveles corren y registran "
                                 "sus corridas — la curva de degradación las "
                                 "necesita — pero no escriben checkpoints",
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
            # El piloto escribe un árbol paralelo: las mismas rutas de arriba,
            # colgadas de `Pilot/`. Se declara el directorio una sola vez y no
            # archivo por archivo, porque no es un segundo experimento sino la
            # misma corrida a escala reducida, y la escala ya se lee del propio
            # registro. Sin esta línea las cuarenta salidas del piloto se
            # reportan como material que nadie declaró, que es justo lo que
            # `undeclaredRecords` existe para forzar a escribir.
            "Results/Pilot",
        ],
        # The terms Eq. (39) combines, and the dimension carrying their share.
        #
        # `contribution` on its own is the numerator: it cannot separate a term
        # that commanded nothing from a term that was scaled to nothing, and both
        # print small. Eq. (21) is divided by B_src precisely so the three terms
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
        # Donde aterriza un shard que volvio. No es una eleccion nueva: es el
        # mismo lugar que `shards.read_shards()` ya usa cuando quien la llama no
        # dice otro --- la campana completa, limpia y de forma campania. Lo que
        # cambia es quien puede leerlo: sin este campo, `gate`, `close` y
        # `probe` no tienen bandera con que decir donde mirar, y un testigo
        # `@shard` les queda SIN MEDIR para siempre, que no es lo mismo que
        # decir que el shard no llego. Declarado una vez, todos comparan contra
        # la misma respuesta.
        "shardsRoot": "MIL-CREDA/Results/Benchmark/shards",
    },
    # Which module carries the runtime, so a reading about this repository's
    # environment is about the module that actually imports it. The same two
    # values a generated job folder's `run-config.json` fixes under `run`;
    # `__environment__` below carries the other block that file needs.
    "entry": {
        "module": "MIL_CREDA_Benchmark.harness",
        "function": "run_pilot",
    },
}
