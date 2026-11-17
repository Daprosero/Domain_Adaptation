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

import dataclasses
from dataclasses import dataclass

import numpy as np

from CREDA.toy_experiment import ANGULO_SIGMA, _rotacion_generica, perturbar

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

#: The offset of the per-pair shift's own stream (see `desplazar_por_par`).
#: Its own stream, so switching the option on changes no other draw.
DESPLAZAMIENTO_DE_PAR = 3000

#: The per-pair (subject-like) shift's declared scale, relative to the source's
#: mean coordinate spread `esc` (the unit `perturbar` measures its translation
#: in). OFF in `construir` by default; this is the value the subject-shift
#: proof of concept declares. **Why 0.25.** Measured on `rollo` at the full
#: scale (seeds 10 and 11): `esc` = 5.66, the global translation is
#: `3 esc` = 17.0, and the global Sigma-preserving rotation moves a point by
#: 1.31 on average. The translation is shared by every bag and BatchNorm closes
#: it; the rotation is the part of the global shift no second-order statistic
#: closes. A per-pair translation of `0.25 esc` = 1.42 (RMS norm) plus a
#: rotation about the pair's own centroid with angle sd `ANGULO_SIGMA` (10
#: degrees) is that same size, per pair: comparable to the non-trivial global
#: shift, far below the 17.0 translation, and of the order of the spacing of
#: neighbouring bag centroids (1.1), so each target bag stays in its partner's
#: region of the roll.
ESCALA_POR_PAR_DECLARADA = 0.25

#: The subject field's own stream offset (its phase), so switching it on
#: changes no other draw.
DESPLAZAMIENTO_DE_CAMPO = 4000

#: How many full periods the subject field makes over `rollo`'s whole
#: parameter range: one per class segment, so the bands of one class see
#: shifts of both signs and every size up to the amplitude.
PERIODOS_DEL_CAMPO = 3

#: `CREDA.toy_experiment.dura_rollo` draws `t` uniformly on this interval.
T_ROLLO = (1.5 * np.pi, 4.5 * np.pi)

#: The subject signature's own stream offset, so switching it on changes no
#: other draw.
DESPLAZAMIENTO_DE_FIRMA = 6000

#: The coordinate the subject signature moves: `rollo`'s height `h`. The
#: generator draws it uniformly on [0, 12] independently of `t`, and the class
#: is cut out of `t` alone, so an offset along `h` carries no class information
#: by construction.
EJE_DE_FIRMA = 1


def t_bajo_campo(t: np.ndarray, amplitud_t: float, periodos: int,
                 fase: float) -> np.ndarray:
    """`t + amplitud_t * sin(2 pi periodos (t - t0) / L + fase)` on `rollo`'s range.

    A smooth, low-frequency displacement of the roll's own parameter. It is
    invertible (strictly increasing in `t`) whenever
    `amplitud_t * 2 pi periodos / L < 1`; that is checked, never assumed.
    """
    bajo, alto = T_ROLLO
    escala = 2 * np.pi * periodos / (alto - bajo)
    if amplitud_t * escala >= 1.0:
        raise ValueError(
            f"amplitud_t = {amplitud_t} with {periodos} periods is not invertible "
            f"along t: it needs amplitud_t < {1.0 / escala}")
    return t + amplitud_t * np.sin(escala * (t - bajo) + fase)


def espaciado_minimo_t(Xs, ys, bs, Xt, yt, bt) -> float:
    """The smallest gap, in `t`, between the median `t` of two neighbouring
    bands of one class, over both raw draws. A slide below half of it keeps
    every band nearest its own place (condition 4 of the run-3 analysis) for
    the tightest band, not only the typical one."""
    gaps = []
    for X, y, b in ((Xs, ys, bs), (Xt, yt, bt)):
        t = parametro("rollo", X)
        for c in np.unique(y):
            centres = np.sort([np.median(t[(y == c) & (b == k)])
                               for k in np.unique(b[y == c])])
            gaps.append(np.diff(centres).min())
    return float(min(gaps))


def campo_de_sujeto(X: np.ndarray, amplitud_t: float, periodos: int,
                    fase: float) -> np.ndarray:
    """Every `rollo` point slid ALONG the roll by `t_bajo_campo`, height kept.

    The point stays on the manifold `(t cos t, h, t sin t)`; only where along
    it changes, smoothly with `t`, so neighbouring bands move alike and an
    instance-level encoder can invert the map.
    """
    t = parametro("rollo", X)
    nuevo = t_bajo_campo(t, amplitud_t, periodos, fase)
    return np.column_stack([nuevo * np.cos(nuevo), X[:, 1], nuevo * np.sin(nuevo)])


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
    #: The per-pair shift applied to the target, `{(class, band): {"angulo",
    #: "eje", "traslacion"}}`; `None` when the option is off.
    desplazamientos: dict | None = None
    #: The subject field applied to the target (`fraccionDeBanda`,
    #: `espaciadoMinimoT`, `amplitudT`, `periodos`, `fase`); `None` when off.
    campo: dict | None = None
    #: The subject signature applied to both domains (see `firmar_sujetos`);
    #: `None` when off.
    firma: dict | None = None
    #: The per-subject target shift (`desplazar_sujetos_en_target`); `None` off.
    desplazamientoDeSujeto: dict | None = None
    #: The instance noise (`ruido_de_instancia`); `None` when off.
    ruido: dict | None = None


def geometria_de_firma(X: np.ndarray, y: np.ndarray, b: np.ndarray) -> dict:
    """The two scales the subject signature is read against, on one domain:

    - `withinBagSpread`: the median over bags of the RMS distance of a bag's
      instances to its own centroid;
    - `betweenClassSeparation`: the smallest distance between two class
      centroids.
    """
    spreads = []
    for c in np.unique(y):
        for k in np.unique(b[y == c]):
            P = X[(y == c) & (b == k)]
            spreads.append(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
    centres = [X[y == c].mean(0) for c in np.unique(y)]
    gaps = [np.linalg.norm(a - c) for i, a in enumerate(centres) for c in centres[i + 1:]]
    return {"withinBagSpread": float(np.median(spreads)),
            "betweenClassSeparation": float(min(gaps))}


def firmar_sujetos(familia: Familia, fraccion: float, semilla: int) -> Familia:
    """`familia` with one fixed offset per subject added to BOTH domains.

    A subject is bag index `k` within its class: source bag `k` and its
    partner, target bag `k`, of the same class. Every instance of subject `k`
    moves by `s_k` along `EJE_DE_FIRMA`, in the source and in the (already
    perturbed) target alike, so the shared encoder sees the same mark on both
    sides of the pair and the partner can become the kernel-nearest bag.

    The `s_k` are drawn once per index from `semilla + DESPLAZAMIENTO_DE_FIRMA`,
    centred over the indices and rescaled to an RMS of exactly `amplitude`,
    and the same `s_k` is used in every class: each class receives the same
    multiset of offsets, zero-mean over its subjects, along a coordinate the
    class is not cut from, so the class information is unchanged.

    `amplitude` is `fraccion` times the source's between-class separation
    (`geometria_de_firma`), the unit the EEG-grounded calibration is stated in:
    on BALLADEER the own-partner percentile across conditions is read off band
    power, and the toy matches it at a multiple of that separation. The
    within-bag spread is recorded beside it as the other reference scale.
    `fraccion` 0 is off and returns `familia` itself.
    """
    if fraccion == 0:
        return familia
    if not (np.isfinite(fraccion) and fraccion > 0):
        raise ValueError(f"fraccion = {fraccion}: a multiple of the class separation "
                         "is finite and non-negative")
    if familia.nombre != "rollo":
        raise ValueError(f"the subject signature is declared on `rollo` only, "
                         f"not {familia.nombre!r}")
    geometria = geometria_de_firma(familia.Xs, familia.ys, familia.bs)
    amplitud = fraccion * geometria["betweenClassSeparation"]
    indices = np.unique(np.concatenate([familia.bs, familia.bt]))
    firmas = np.random.default_rng(semilla + DESPLAZAMIENTO_DE_FIRMA).normal(size=len(indices))
    firmas -= firmas.mean()
    firmas *= amplitud / np.sqrt((firmas ** 2).mean())
    por_indice = dict(zip(indices.tolist(), firmas.tolist()))

    def marcar(X, b):
        salida = X.copy()
        salida[:, EJE_DE_FIRMA] += firmas[np.searchsorted(indices, b)]
        return salida

    registro = {"multipleOfClassSeparation": float(fraccion), "amplitude": float(amplitud),
                **geometria, "axis": EJE_DE_FIRMA, "signatures": por_indice}
    return dataclasses.replace(familia, Xs=marcar(familia.Xs, familia.bs),
                               Xt=marcar(familia.Xt, familia.bt), firma=registro)


#: The per-subject target shift's own stream offset (its phase), and the
#: instance noise's: their own, so switching either on changes no other draw.
DESPLAZAMIENTO_DE_SUJETO_EN_TARGET = 7000
DESPLAZAMIENTO_DE_RUIDO = 8000


def desplazar_sujetos_en_target(familia: Familia, fraccion: float,
                                semilla: int) -> Familia:
    """`familia` with a smooth per-subject shift added to the TARGET only.

    Every target point moves along `EJE_DE_FIRMA` (the class-free height) by
    `f(t) = a sin(2 pi P (t - t0) / L + phase)`, `t` its own place on the roll
    (read off `Xt_crudo`), `P = PERIODOS_DEL_CAMPO` (one period per class
    segment, so every class sees both signs and the field is zero-mean over
    its subjects) and the phase from its own stream. A subject is one band of
    `t`, so the shift is nearly constant within a subject and distinct across
    subjects; it is a smooth function of position, so a shared instance
    encoder can learn to undo it, and the global term cannot, because no shared
    translation removes it. `a` is set so the RMS over the target points is
    exactly `fraccion` times the source's between-class separation
    (`geometria_de_firma`, the subject signature's unit). 0 is off.
    """
    if fraccion == 0:
        return familia
    if not (np.isfinite(fraccion) and fraccion > 0):
        raise ValueError(f"fraccion = {fraccion}: a multiple of the class separation "
                         "is finite and non-negative")
    if familia.nombre != "rollo":
        raise ValueError(f"the target subject shift is declared on `rollo` only, "
                         f"not {familia.nombre!r}")
    geometria = geometria_de_firma(familia.Xs, familia.ys, familia.bs)
    amplitud = fraccion * geometria["betweenClassSeparation"]
    bajo, alto = T_ROLLO
    fase = float(np.random.default_rng(semilla + DESPLAZAMIENTO_DE_SUJETO_EN_TARGET)
                 .uniform(0.0, 2 * np.pi))
    t = parametro("rollo", familia.Xt_crudo)
    campo = np.sin(2 * np.pi * PERIODOS_DEL_CAMPO * (t - bajo) / (alto - bajo) + fase)
    campo *= amplitud / np.sqrt((campo ** 2).mean())
    Xt = familia.Xt.copy()
    Xt[:, EJE_DE_FIRMA] += campo
    registro = {"multipleOfClassSeparation": float(fraccion), "amplitude": float(amplitud),
                "periods": PERIODOS_DEL_CAMPO, "phase": fase, "axis": EJE_DE_FIRMA}
    return dataclasses.replace(familia, Xt=Xt, desplazamientoDeSujeto=registro)


def ruido_de_instancia(familia: Familia, fraccion: float, semilla: int) -> Familia:
    """`familia` with i.i.d. isotropic Gaussian noise on every instance of BOTH
    domains, standard deviation `fraccion` times the source's between-class
    separation. Labels, bags and partners unchanged; its own stream. 0 is off.
    """
    if fraccion == 0:
        return familia
    if not (np.isfinite(fraccion) and fraccion > 0):
        raise ValueError(f"fraccion = {fraccion}: a multiple of the class separation "
                         "is finite and non-negative")
    sd = fraccion * geometria_de_firma(familia.Xs, familia.ys, familia.bs)[
        "betweenClassSeparation"]
    rng = np.random.default_rng(semilla + DESPLAZAMIENTO_DE_RUIDO)
    Xs = familia.Xs + rng.normal(0.0, sd, familia.Xs.shape)
    Xt = familia.Xt + rng.normal(0.0, sd, familia.Xt.shape)
    return dataclasses.replace(familia, Xs=Xs, Xt=Xt,
                               ruido={"multipleOfClassSeparation": float(fraccion),
                                      "sd": float(sd)})


def desplazar_por_par(Xt: np.ndarray, yt: np.ndarray, bt: np.ndarray,
                      semilla: int, escala: float, esc: float
                      ) -> tuple[np.ndarray, dict]:
    """A shift of its own for every (class, band) pair of the target.

    Applied on top of `perturbar`'s global shift. For pair `(c, k)` -- the
    points of target bag `k` of class `c`, whose partner is source bag `k` of
    class `c` -- a rigid rotation about the pair's own centroid (random axis,
    angle ~ N(0, ANGULO_SIGMA degrees)) and a translation ~ N(0, (escala *
    esc)^2 / 3) per coordinate, so its RMS norm is `escala * esc`. The
    translations are centered across pairs, so the shared part of the shift
    stays the global one and what this adds is only subject-specific.

    The band column is untouched: a point stays in its bag, so the partners
    stay partners; what changes is that different pairs need different
    corrections. Drawn from its own stream (`semilla + DESPLAZAMIENTO_DE_PAR`).
    """
    rng = np.random.default_rng(semilla + DESPLAZAMIENTO_DE_PAR)
    pares = [(int(c), int(k)) for c in np.unique(yt) for k in np.unique(bt[yt == c])]
    angulos = rng.normal(0.0, np.radians(ANGULO_SIGMA), len(pares))
    ejes = rng.normal(size=(len(pares), 3))
    traslaciones = rng.normal(0.0, escala * esc / np.sqrt(3), (len(pares), 3))
    traslaciones -= traslaciones.mean(0)
    salida = Xt.copy()
    registro = {}
    for (c, k), angulo, eje, traslacion in zip(pares, angulos, ejes, traslaciones):
        m = (yt == c) & (bt == k)
        centro = Xt[m].mean(0)
        R = _rotacion_generica(angulo, eje)
        salida[m] = (Xt[m] - centro) @ R.T + centro + traslacion
        registro[(c, k)] = {"angulo": float(angulo), "eje": eje.tolist(),
                            "traslacion": traslacion.tolist()}
    return salida, registro


def construir(nombre: str, fn, semilla: int = 0,
              n: int = PUNTOS_POR_DOMINIO,
              bolsas_por_clase: int = BAGS_PER_CLASS_TOY,
              escala_por_par: float = 0.0, grados: float | None = None,
              fraccion_de_campo: float = 0.0) -> Familia:
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
    # The subject field, OFF by default: when on, the raw target is slid along
    # the roll by at most `fraccion_de_campo` of a band's width in `t` BEFORE
    # the global shift. The bands were already read off `Xt_crudo`, so every
    # point keeps its bag and every target bag keeps its partner.
    if fraccion_de_campo < 0:
        raise ValueError(f"fraccion_de_campo = {fraccion_de_campo}: it is non-negative")
    campo = None
    base = Xt_crudo
    if fraccion_de_campo > 0:
        if nombre != "rollo":
            raise ValueError(f"the subject field is declared on `rollo` only, not {nombre!r}")
        ancho = espaciado_minimo_t(Xs, ys, bs, Xt_crudo, yt, bt)
        fase = float(np.random.default_rng(semilla + DESPLAZAMIENTO_DE_CAMPO)
                     .uniform(0.0, 2 * np.pi))
        amplitud = fraccion_de_campo * ancho
        base = campo_de_sujeto(Xt_crudo, amplitud, PERIODOS_DEL_CAMPO, fase)
        campo = {"fraccionDeBanda": fraccion_de_campo, "espaciadoMinimoT": ancho,
                 "amplitudT": amplitud, "periodos": PERIODOS_DEL_CAMPO, "fase": fase}
    # `perturbar` consumes `rng`, so it runs after both parameter reads: moving
    # it earlier would change the source draw and silently reseed the figure.
    Xt = perturbar(Xs, base, rng, grados=grados)
    # Off by default, and off this is the draw exactly as it was.
    if escala_por_par < 0:
        raise ValueError(f"escala_por_par = {escala_por_par}: a scale is non-negative")
    desplazamientos = None
    if escala_por_par > 0:
        Xt, desplazamientos = desplazar_por_par(
            Xt, yt, bt, semilla, escala_por_par, float(Xs.std(0).mean()))
    return Familia(nombre=nombre, Xs=Xs, ys=ys, bs=bs,
                   Xt=Xt, yt=yt, bt=bt, Xt_crudo=Xt_crudo,
                   desplazamientos=desplazamientos, campo=campo)
