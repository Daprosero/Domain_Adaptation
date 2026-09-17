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

---

No finding is open against research-concept-r21.md.

r21 rewrote Section 3's attention block (Eqs. 15-16, plus the un-numbered
R_phi/v_R_tilde reparametrization and the m_eff/self-similarity diagnostic
pair) and renumbered the equations around it (old 16/17/18/19/20/21 ->
new 19/20/21/14/17/18; see `MIL_CREDA.attention.__provenance__` and
`references/usage.md` for the full map). Every new claim it states was swept
over 200 randomized configurations, varying bag size, embedding dimension,
the instance-kernel bandwidth, gamma, tau_att, the raw relevance parameter's
scale, and (for the separation condition) group size and compactness --
`tests/test_invariants.py`'s attention section carries the sweep beside each
claim's own invariant test, one `test_<id>` per id in
`MIL_CREDA.attention.__provenance__["invariants"]`. None failed: every claim
held over its full sweep, so none of them is a finding here. This includes
the l1 mechanism itself (`_l1_ball_reparametrization`), the permutation
equivariance and singleton-bag claims, the consensus term's range and its two
bandwidth limits, the gamma=0 and neutral-hyperparameter reductions, the
logit-spread/weight-ratio bound, the two diagnostics' ranges, and the
separation condition's sufficient-condition implication.

Two findings were raised against r14, measured over 200 configurations,
validated with both poles, and adopted by the deliberation:

- `local_normalizer_loose_by_two` (loose-constant, theorem, Eq. 38) -> published
  in r15, now held by `test_local_distance_bounded_by_two`.
- `source_stabilizer_breaks_non_negativity` (ill-formed, tendency 1/200, Eq. 18
  under r14's numbering, now Eq. 21) -> published in r16, now held by
  `test_source_loss_non_negative`.

An adopted remedy is no longer a correction under consideration: it is what the
proposal says. Its claim moved to the invariant suite, its remedy test retired,
and the module implementing it declares the invariant. What remains here of the
audit is its evidence, in `test_audit.py`, which reproduces the indicted forms
so the record of what was wrong keeps measuring what was wrong.
"""

FINDINGS: list[dict] = []
