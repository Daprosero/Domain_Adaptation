"""CREDA's own hard families, with each class cut into `BAGS_PER_CLASS_TOY` bags.

Nothing about the toyset moves: the same three families from
`CREDA.toy_experiment.DURAS`, the same generators, the same perturbation of
the target. What is added is the GROUPING: `construir` draws one family and
returns its two domains with a bag index per point, which is what
`toy_run.build_family` hands to `bags.build`.

A bag is a contiguous band of the SAME scalar the generator cut the classes
out of -- `t` along the parabola's arm, the torus's major angle, the roll's own
winding -- so cutting bags is the generator's own rule applied one level finer,
never a partition invented here. Being contiguous is what makes each bag a
SLICE of the manifold rather than a sack around scattered points.

**Why CREDA's toyset and not nine synthetic domains.** The alternative --
`MIL_CREDA_Benchmark.toy`'s nine domains -- was measured and rejected: its
noise is declared in absolute units while the three families live on scales
0.013, 0.496 and 5.137, so one nominal sigma inflates one subject 29x, the next
1.34x and the last 1.01x. That is three different experiments wearing one
label. Cutting CREDA's own families into bags leaves the global structure of
the data exactly where it was and changes only the grouping.

**No `__provenance__`, deliberately.** This is a benchmark-side module and it
implements no equation of the formulation: it regroups prior work's
generators. Stamping a provenance here would claim a link to the mathematics
that does not exist. The import direction is the legal one --
`MIL_CREDA_Benchmark` may read `CREDA`, as `wiring.py` and `schedules.py`
already do; what must never happen is `CREDA` importing anything of
MIL-CREDA's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from CREDA.toy_experiment import perturbar

#: How many bags each class is cut into. The operator's number, not a search,
#: and it is THIS toyset's own number rather than the digit benchmark's rule:
#: with CREDA's three classes that is 81 bags per domain where the digit
#: campaign has 120 over ten classes.
#:
#: **It was 12, which was `config.BAGS_PER_CLASS`, and what the move buys is
#: the instrument.** The published accuracy of every cell is read over the
#: `test` role of every class, so the finest step the column can take is one
#: bag out of that many. Measured over `Scale.realized_split` at several
#: counts: 12 bags per class realize 7/2/3 and so 9 test bags, a step of
#: 11.11 pp; 24 realize 14/5/5 and so 15 test bags, 6.67 pp; 27 realize 16/5/6
#: and so 18 test bags, 5.56 pp; 30 realize 18/6/6 and so 18 test bags again,
#: the SAME 5.56 pp for more material. 27 is therefore the efficient point and
#: 30 buys nothing, which is why this is 27 and not simply the largest count
#: that would fit. CREDA measures 360 points at a step of 0.28 pp, so this
#: narrows the gap without closing it.
BAGS_PER_CLASS_TOY = 27

#: `construir`'s default points per domain: the same `n` CREDA's own
#: three-families figure draws.
#:
#: **It is not the run's number.** `toy_run.build_family` passes
#: `Scale.points_per_domain` explicitly: 81 bags of 30 instances need a draw
#: whose SMALLEST quantile band still funds 30, and 1800 leaves a smallest bag
#: of 20, so `toy_run._instances_per_bag` would refuse it.
PUNTOS_POR_DOMINIO = 1800

#: The offset `CREDA.toy_experiment.datos_semilla` puts between the source
#: seed and the target seed. Source and target are INDEPENDENT draws there, and
#: that is why the two domains' bands are read separately below.
DESPLAZAMIENTO_DE_SEMILLA = 1000

#: The offset between the draw's seed and the seed the ROLE PERMUTATION below is
#: drawn from. Derived from the draw's own seed, like every other stream in this
#: file, and never from a generator of its own: a permutation seeded
#: independently of the draw would make a family's roles unreproducible from its
#: seed, which is the one thing every number here is read against.
#:
#: **It is a distinct offset and not a bare `semilla`, and that is the point.**
#: `construir`'s own `rng` IS `default_rng(semilla)`, and it has already handed
#: the source cloud to `fn` by the time the bands exist. A second generator on
#: the same seed would replay the very stream the cloud came out of, so which
#: band became which bag would be a deterministic function of the same bits that
#: placed the points -- a correlation between the material and the role
#: assignment, which is exactly what a split exists not to have. One step past
#: `DESPLAZAMIENTO_DE_SEMILLA` so it is not the TARGET draw's stream either.
DESPLAZAMIENTO_DE_PERMUTACION = 2000


def parametro(nombre: str, X: np.ndarray) -> np.ndarray:
    """The scalar the generator cut the classes out of, read back off `X`.

    One per family, and each one is the generator's own coordinate rather than
    a projection chosen here:

    - `lunas` -- `t`, the parabola's arm. The generator writes it straight into
      the first column (`x = t`), so reading it back is the identity on that
      column.
    - `toroide` -- `phi`, the major angle. `R + r cos(theta)` is `1.9` at its
      smallest and never changes sign, so `arctan2(y, x)` recovers `phi`
      exactly rather than up to a quadrant.
    - `rollo` -- `t`, the winding. The generator writes `(t cos t, h, t sin t)`
      and `t` is positive throughout `[1.5 pi, 4.5 pi]`, so `hypot(x, z)` is
      `|t|` and therefore `t`.
    """
    if nombre == "lunas":
        return X[:, 0]
    if nombre == "toroide":
        return np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi)
    if nombre == "rollo":
        return np.hypot(X[:, 0], X[:, 2])
    raise KeyError(f"no hay parámetro declarado para la familia {nombre!r}")


def bolsas(p: np.ndarray, y: np.ndarray, n: int = BAGS_PER_CLASS_TOY) -> np.ndarray:
    """Each point's bag: its band of the parameter WITHIN its own class.

    By quantiles and not by equal cuts, so the `n` bags of a class carry the
    same number of points to within one -- a bag size that varied with the
    density of the parameter would be a difference nobody declared, and the eye
    reads a denser band as a bigger one. Measured on this toyset, equal-width
    cuts spread a class's bag sizes by 8 to 29 points while quantiles spread
    them by 0 or 1; the leftover 1 is integer division of a class that does not
    divide by `n`, not a drift.

    Within a class the bands are contiguous in the parameter by construction,
    which is what makes each bag's hull a slice of the manifold: a partition
    that interleaved would still produce `n` hulls per class, and they would
    all be the same sack around the whole class.
    """
    b = np.zeros(len(p), dtype=int)
    for c in np.unique(y):
        m = y == c
        bordes = np.quantile(p[m], np.linspace(0, 1, n + 1)[1:-1])
        b[m] = np.clip(np.searchsorted(bordes, p[m], side="right"), 0, n - 1)
    return b


def permutacion_de_bandas(y: np.ndarray, n: int,
                          semilla: int) -> dict[int, np.ndarray]:
    """Which BAG INDEX each band position becomes, per class.

    `bolsas` numbers a class's bands 0..n-1 along the parameter, and
    `bags.build` hands bag `k` to a role by comparing `k` against the role
    counts. So the band number IS the position on the manifold and a split by
    bag index is a split by geometry: measured on `toroide` at 27 bands, `train`
    held bands 0..15 of every class, `valid` 16..20 and `test` 21..26, and the
    encoder read 1.0000 on train and below chance on test IN ITS OWN DOMAIN.
    What the roles differed in was the region, not the sample.

    This is the renumbering that fixes it, and it is the ONLY thing that moves:
    the bands themselves are untouched, so each bag is still one contiguous
    slice of the parameter and every hull the figure draws is still a slice.
    What changes is which slice a role receives.

    **The shape is CREDA's `partir`, not a scheme of this file's.** That
    function is what this toyset is being brought to parity with, and it splits
    by shuffling: one generator, `rng.shuffle(idx)` INSIDE its per-class loop,
    so every class draws its own shuffle off successive states of the same
    stream. Two properties follow from that and both are deliberate here:

    - **Per class, independently.** Each class gets its own draw, so the roles
      are not the same stretch of every class's parameter. One permutation
      replayed across classes would leave the test role at the same relative
      positions in all three -- scattered, but still systematic, which is a
      residue of the very defect this removes.
    - **Drawn from the seed, so a family's roles are reproducible from it.**

    Returns `{class: array}` where `array[j]` is the bag index band position `j`
    becomes. Taking `n` rather than reading the module default matters: a
    caller that cut a different number of bands must not be handed the
    default's permutation.
    """
    rng = np.random.default_rng(semilla + DESPLAZAMIENTO_DE_PERMUTACION)
    return {int(c): rng.permutation(n) for c in np.unique(y)}


def bandas_permutadas(b: np.ndarray, y: np.ndarray,
                      permutacion: dict[int, np.ndarray]) -> np.ndarray:
    """`b` renumbered by `permutacion`: the band column `bags.build` reads.

    One level up from the points on purpose. Applying a permutation to the
    POINTS would scatter each bag across the whole class, so every bag would
    become the same sack around it. This renames bands; it never moves a point
    between them, so each bag is still one slice of the parameter afterwards.
    """
    salida = np.empty_like(b)
    for c, orden in permutacion.items():
        m = y == c
        salida[m] = np.asarray(orden)[b[m]]
    return salida


@dataclass(frozen=True)
class Familia:
    """One family's two domains, their labels and their bag indices."""

    nombre: str
    Xs: np.ndarray
    ys: np.ndarray
    bs: np.ndarray
    Xt: np.ndarray
    yt: np.ndarray
    bt: np.ndarray
    #: The target BEFORE `perturbar`. Kept because it is what the target's own
    #: bands were read off, so a test can check that rather than take it on
    #: trust.
    Xt_crudo: np.ndarray


def construir(nombre: str, fn, semilla: int = 0,
              n: int = PUNTOS_POR_DOMINIO,
              bolsas_por_clase: int = BAGS_PER_CLASS_TOY) -> Familia:
    """Source, target, labels and bag indices for one family.

    The draw is CREDA's, down to the seed offset: `datos_semilla` takes the
    source from `semilla` and the target from `semilla + 1000`, independently,
    and then perturbs the target toward the source. Nothing here changes that.

    **The target's bands are read off the UNPERTURBED draw**, and this is the
    one subtlety in the file. `perturbar` rotates, translates and jitters the
    cloud; `arctan2` on an already rotated and translated torus gives a
    different angle, so bands cut after the perturbation would be bands of the
    perturbation rather than of the manifold. Measured on this toyset, cutting
    after the perturbation agrees with cutting before it on only 36.67% of the
    points of `toroide` -- so this is not a fine point, it is most of the
    labelling.

    **The two domains SHARE one band permutation, and that is a decision.**
    `permutacion_de_bandas` is drawn once, off `semilla`, and applied to `bs`
    and `bt` alike, so band position `j` of class `c` becomes bag `j` in both
    domains. Independent permutations were the alternative and are wrong here,
    for two measured reasons:

    - **It is CREDA's own choice.** `CREDA.toy_experiment._empacar` calls
      `partir(ys, semilla=semilla)` and `partir(yt, semilla=semilla)` -- the
      SAME seed for both domains, so the two splits are paired by construction
      rather than by accident. Sharing one draw here is that choice made
      structural: it holds whatever each class's size is, where two calls on one
      seed hold only while the class counts match.
    - **Independent permutations would reintroduce the defect between the
      domains.** Bag `k` of class `c` is the `k`-th band of the same parameter
      in BOTH domains, and the target accuracy is read on the TARGET's
      evaluation role. Permuting the two independently makes source bag `k` and
      target bag `k` different stretches of the manifold, which makes the
      target's evaluation role a region the source never trained on -- the same
      gap this permutation exists to remove, moved from between the roles to
      between the domains.
    """
    rng = np.random.default_rng(semilla)
    Xs, ys = fn(n, rng)
    Xt_crudo, yt = fn(n, np.random.default_rng(semilla + DESPLAZAMIENTO_DE_SEMILLA))
    bs = bolsas(parametro(nombre, Xs), ys, bolsas_por_clase)
    bt = bolsas(parametro(nombre, Xt_crudo), yt, bolsas_por_clase)
    # The role permutation, drawn ONCE and applied to BOTH domains -- see the
    # docstring's third paragraph for why they share it rather than each
    # drawing its own.
    permutacion = permutacion_de_bandas(ys, bolsas_por_clase, semilla)
    bs = bandas_permutadas(bs, ys, permutacion)
    bt = bandas_permutadas(bt, yt, permutacion)
    # `perturbar` consumes `rng`, so it runs after both parameter reads: moving
    # it earlier would change the source draw and silently reseed the figure.
    Xt = perturbar(Xs, Xt_crudo, rng)
    return Familia(nombre=nombre, Xs=Xs, ys=ys, bs=bs,
                   Xt=Xt, yt=yt, bt=bt, Xt_crudo=Xt_crudo)
