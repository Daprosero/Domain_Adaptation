"""Audit findings on the proposal, each with the remedy it proposes.

A finding is a defect in the mathematics itself: a term that is ill-formed, a
claim stated more strongly than the construction supports, a missing complement
the rest of the development assumes, or a constant that does not hold up. It is
not a bug in this code.

The contract, checked statically by `implementation_cli.py verify`:

- every `id` has a `test_finding_<id>` proving the defect is real, measured over
  the sweep, and a `test_remedy_<id>` proving the fix resolves it AND preserves
  what was already established;
- `status` is `theorem` (holds across the whole sweep) or `tendency` (declares
  its measured rate and is never asserted as a law);
- `uses` is notation the remedy leans on and must appear verbatim in the
  revision; `introduces` is notation it would add, which keeps the audit at
  `needs-deliberation` because adding notation is the deliberation's call;
- `adoption.absent` is text present in the revision today whose disappearance
  signals the remedy was taken in; `expect` are forms that confirm it;
- `becomes_invariant` names what the remedy turns into once published: at that
  point the remedy test retires and the claim moves to the invariant suite;
- `remedy_block` is the corrected equation written out, exactly as it should
  read in the document, carrying the same `\\tag{n}` it replaces. A finding of
  local reach that omits it cannot be settled inline: the handoff defers it to
  a session of its own, because describing a correction in prose is not the
  same as writing it, and nothing here will paraphrase mathematics into a
  document. Findings that add notation are structural anyway and need none.

Remedies live in the tests, never in src/. Establishing that a correction is
sound is not the same as adopting it.
"""

FINDINGS = [
    {
        "id": "local_normalizer_loose_by_two",
        "kind": "loose-constant",
        "equations": ["38"],
        "remedy_equations": ["38"],
        "uses": [
            r"\ell_{\mathrm{loc},j}",
            r"d_j^{2}",
            r"w_j^{t}",
            r"\varepsilon_{\mathrm{loc}}",
            r"\mathcal L_{\mathrm{loc}}",
        ],
        "introduces": [],
        "adoption": {
            "absent": r"\frac{d_j^{2}}{4},",
            "expect": [r"\frac{d_j^{2}}{2},"],
        },
        "becomes_invariant": "local_distance_bounded_by_two",
        "status": "theorem",
        "rate": "200/200 configurations keep d_j^2 < 2; max observed 1.999267 over 905 target bags",
        "statement": (
            "Eq. (38) normalizes the local discrepancy by 4, a constant Section 5 derives "
            "from the triangle inequality alone: ||Psi|| <= 1 and ||mu|| <= 1 give "
            "||Psi - mu|| <= 2. That derivation discards a property the same section "
            "already established. The Gaussian instance kernel is STRICTLY positive, so "
            "every inner product <Phi(h), Phi(h')> is strictly positive; Psi(B_j^t) and "
            "mu_j^s are convex combinations of those images with non-negative weights, "
            "hence <Psi(B_j^t), mu_j^s> > 0 and "
            "d_j^2 = ||Psi||^2 - 2<Psi, mu> + ||mu||^2 < 2. The value 4 is therefore not "
            "reachable and not even approachable: l_loc,j can never leave [0, 1/2), so "
            "the upper half of the interval the equation declares is unusable and the "
            "local term enters Eq. (39) at systematically half the intended scale, "
            "silently halving lambda_loc."
        ),
        "remedy": (
            "Normalize by 2 instead of 4. The bound d_j^2 < 2 is tight: it is approached "
            "when both representations concentrate on single instances placed far enough "
            "apart that the kernel value vanishes, which the sweep reaches at 1.999267. "
            "The correction is expressible with the notation already in the revision and "
            "adds none: it changes one constant in Eq. (38). Section 5's derivation "
            "sentence, which states 0 <= d_j^2 <= 4 from the triangle inequality, must be "
            "rewritten in the same edit to derive the tighter bound from the strict "
            "positivity of kappa_sigma^I; leaving that prose untouched would contradict "
            "the corrected equation."
        ),
        "remedy_block": (
            "$$\n"
            "\\ell_{\\mathrm{loc},j}\n"
            "=\n"
            "\\frac{d_j^{2}}{2},\n"
            "\\qquad\n"
            "0 \\leq \\ell_{\\mathrm{loc},j} \\leq 1,\n"
            "\\qquad\n"
            "\\mathcal L_{\\mathrm{loc}}\n"
            "=\n"
            "\\frac{\n"
            "\\displaystyle \\sum_{j=1}^{N_t} w_j^{t} \\ell_{\\mathrm{loc},j}\n"
            "}{\n"
            "\\displaystyle \\sum_{j=1}^{N_t} w_j^{t} + \\varepsilon_{\\mathrm{loc}}\n"
            "} .\n"
            "\\tag{38}\n"
            "$$"
        ),
    },
    {
        "id": "source_stabilizer_breaks_non_negativity",
        "kind": "ill-formed",
        "equations": ["18"],
        "remedy_equations": ["18"],
        "uses": [
            r"\mathcal L_{\mathrm{src}}",
            r"y_{i,c}^{s}",
            r"g_{i,c}^{s}",
            r"\varepsilon_{\mathrm{src}}",
            r"N_s",
        ],
        "introduces": [],
        "adoption": {
            "absent": r"\ln \left( g_{i,c}^{s} + \varepsilon_{\mathrm{src}} \right),",
            "expect": [
                r"\frac{g_{i,c}^{s} + \varepsilon_{\mathrm{src}}}{1 + \varepsilon_{\mathrm{src}}}",
            ],
        },
        "becomes_invariant": "source_loss_non_negative",
        "status": "tendency",
        "rate": "1/200 configurations drive L_src below zero (min -9.999e-09)",
        "statement": (
            "Eq. (18) places the stabilizer INSIDE the logarithm, as "
            "ln(g_{i,c}^s + epsilon_src). Since g_{i,c}^s is a probability, the argument "
            "reaches 1 + epsilon_src > 1 whenever a source bag is predicted with "
            "certainty, so the logarithm turns positive and the term becomes NEGATIVE. A "
            "negative log-likelihood is non-negative by definition and vanishes only at a "
            "perfect prediction; here the perfect prediction is rewarded with "
            "-ln(1 + epsilon_src) instead of zero, and Eq. (39) inherits an objective "
            "whose supervised reference no longer has a floor at zero. The magnitude is "
            "small, but the sign is wrong and the anchoring role Eq. (39) assigns to "
            "L_src as the term fixed at coefficient one depends on it."
        ),
        "remedy": (
            "Divide the stabilized argument by its own maximum: "
            "ln((g + epsilon_src) / (1 + epsilon_src)). Because g <= 1, the ratio never "
            "exceeds one, so the logarithm is never positive and L_src >= 0, with equality "
            "exactly at g = 1. The stabilizer keeps doing its only job, since the ratio "
            "stays strictly positive when g = 0. The correction uses only notation the "
            "revision already defines and adds none."
        ),
        "remedy_block": (
            "$$\n"
            "\\mathcal L_{\\mathrm{src}}\n"
            "=\n"
            "- \\frac{1}{N_s}\n"
            "\\sum_{i=1}^{N_s}\n"
            "\\sum_{c=1}^{C}\n"
            "y_{i,c}^{s}\n"
            "\\ln \\left( \\frac{g_{i,c}^{s} + \\varepsilon_{\\mathrm{src}}}"
            "{1 + \\varepsilon_{\\mathrm{src}}} \\right),\n"
            "\\qquad\n"
            "\\varepsilon_{\\mathrm{src}} > 0,\n"
            "\\tag{18}\n"
            "$$"
        ),
    },
]
