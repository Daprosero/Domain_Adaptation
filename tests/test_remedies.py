"""Level 5 - remedies. Each proposed correction, validated at the same rigour.

A remedy is accepted only when the sweep shows BOTH poles: the correction
satisfies the criterion, and the declared formulation fails it. With one pole
only, nothing distinguishes a real improvement from a measurement that would
have passed whatever it was handed — `verify` reports such a test as a remedy
without control.

Every test starts by refusing to measure what was not ruled admissible:
soundness and expressibility are decided first, efficacy second.

The proposed replacements live here, never in src/.

---

No remedy is open against research-concept-r16.md.

Both findings the audit raised against r14 were validated here, adopted by the
deliberation, and published: the local normalizer in Eq. (38) and the placement
of the stabilizer in Eq. (18). An adopted remedy stops being a proposal and
becomes the formulation, so it moved out of this file: `src/` implements it and
the invariant suite holds it to the same contract as every other claim —
`test_local_distance_bounded_by_two` and `test_source_loss_non_negative`.

Leaving those remedy tests here would keep reporting a defect the revision no
longer has, and would keep measuring `src/` against a formulation it no longer
implements.
"""

from __future__ import annotations
