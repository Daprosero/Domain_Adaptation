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
}
