"""Everything the bounded comparison needs, and nothing the method needs.

This package is deliberately NOT `MIL_CREDA`. Every module of that package
declares `__provenance__` binding it to the revision it implements, and the
verification reads all of them: a benchmark harness placed inside it would be
read as a module of the method missing its provenance, and stamping a fake one on
plumbing that implements no equation would hollow out the only check that keeps
the code tied to the mathematics.

So the separation the name promises is real. Nothing here is part of the
formulation; deleting this package leaves MIL-CREDA intact.

What it does declare is which sections of the revision each arm exercises. Without
that, a new revision raises a question nobody can answer: does this change oblige the
bench to change? With it, the verification names the arms a changed section reaches.

It does NOT declare the revision itself. The revision lives in two hand-written
constants outside this file -- `MIL_CREDA.__implementation__["revision"]`, which is
what the forge's verification reads for this package, and `config.REVISION`.
Nothing under this checkout resolves either against the published document,
because the proposals live in the forge and not here; binding them to the newest
published revision is the verification's own question (`fidelity`), never this
package's.

The digit benchmark this package was built for is retired with the image
datasets, and with it every block that described it: its ceiling search, its
renderers and conclusions, its distribution over seeds and transfers, and its
entry point. Those blocks stay present at their empty values, the shape the
forge's own scaffold writes, so a reader finds them unanswered rather than
missing. What remains declared is what the toyset still exercises.
"""

__benchmark__ = {
    "arms": {
        "B": {"sections": ["3"]},
        "E": {"sections": ["1", "2", "3", "5"]},
        "F": {"sections": ["1", "2", "3", "5"]},
        "G": {"sections": ["1", "2", "3", "4", "5"]},
        # `GN` is retired. It declared exactly G's sections -- it exercised the
        # same equations, and the single axis that separated it (the target
        # forward normalized with the current source-batch statistics instead
        # of learning from the target) was a wiring decision and never a
        # section of its own the revision names -- so retiring it removes an
        # arm and no equation from what this comparison reaches.
    },
    # Retired with the digit benchmark's ceiling search. The toyset's own
    # search was retired with its sweeps: the tutorial runs declared values.
    "search": {},
    "report": {
        # The calls that produce or show a picture. Naming them here is what lets
        # the verification ask whether a figure actually rendered without knowing
        # one word about who draws. `inline` is what puts a figure in front of a
        # reader, and `plt.show` is here for any cell that draws with bare
        # matplotlib rather than through `figures`.
        "figures": [
            "figures.inline",
            "plt.show",
            # Los dos GIF del cuaderno tutorial
            # `MIL-CREDA/Notebooks/Toyset.ipynb`, dibujados en memoria: la
            # trayectoria real de la adaptacion y la caminata de confianza por
            # bolsa. Reemplazan al retirado `toy_run.animar_familia`, y se
            # declaran aca por la misma razon que las otras: sin la
            # declaracion el motor no puede distinguir una celda que dibuja de
            # una que muestra una imagen sin haberla dibujado.
            "tutorial_visuals.gif_de_adaptacion",
            "tutorial_visuals.gif_de_caminata",
        ],
        # The tutorial keeps new results in memory and saved notebook outputs.
        # Declare the retained historical archive too: retiring its writers
        # must not make those existing files undeclared records in audits.
        "records": ["Results/Toyset"],
    },
    # Retired with the digit campaign's shards and its seed/transfer axes.
    "distribution": {},
    # Retired with `harness.run_pilot`; the toyset runs from its notebooks.
    "entry": {"module": "", "function": ""},
}
