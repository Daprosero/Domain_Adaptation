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
    "revision": "research-concept-r16.md",
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
        "H1": {"sections": ["1", "2", "3", "4", "5"]},
        "H2": {"sections": ["1", "2", "3", "4", "5"]},
    },
}
