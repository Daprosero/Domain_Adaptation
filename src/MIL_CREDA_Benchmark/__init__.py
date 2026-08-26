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
        "what": "the ceiling of the adaptation coefficient, one per family, "
                "inherited by that family's derivations. The growth rate is not "
                "searched: it stays at RAMP_DELTA, which is CREDA's own",
        "requiredScale": {"epochs": 20, "seeds": 3},
        "role": "valid",
        "tieRule": "the smallest ceiling among the tied: the same outcome for less "
                   "adaptation is the weaker claim, and a search should not hand a "
                   "term more weight than the measurement asked for",
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
    # The four groups below are not a guess: a replication (`A` on
    # Trayectoria51, `B` on Daprosero, `C` back on Trayectoria51 as a
    # same-machine control — one arm, `G`, one transfer, `M->U`, seed 7)
    # measured all eight of `config.DIMENSIONS` three times and compared the
    # parsed JSON floats with exact `==`.
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
        "identicalAcrossShards": ["epochs", "ceilings", "ceilingsByTransfer"],
    },
    # Which module carries the runtime, so a reading about this repository's
    # environment is about the module that actually imports it. The same two
    # values `tools/kaggle/ceiling-search/run-config.json` already fixes.
    "entry": {
        "module": "MIL_CREDA_Benchmark.harness",
        "function": "run_pilot",
    },
}
