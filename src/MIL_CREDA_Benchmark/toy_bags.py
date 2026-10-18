"""CREDA's own hard families, with each class cut into `BAGS_PER_CLASS_TOY` bags.

Nothing about the toyset moves: the same three families from
`CREDA.toy_experiment.DURAS`, the same generators, the same views, the same
`COL_DURA` hue per class, the same `x`/`o` for source/target, the same 3-D
positions. What is added is the GROUPING, and it is drawn by enclosing each
bag in its own convex hull and writing that bag's name over it.

Five channels, five facts, none of them overwritten::

    hue        -> the CLASS           (COL_DURA at full strength, as CREDA draws it)
    hull       -> the BAG             (one translucent convex hull per bag)
    label      -> WHICH bag           (`Bs11`, `Bt34` -- see `etiqueta_de_bolsa`)
    marker     -> the DOMAIN          (`x` source, `o` target, as CREDA draws it)
    position   -> the MANIFOLD        (untouched, so nothing of the toyset is lost)

**Lightness used to carry the bag and no longer does.** The class hue was
lightened band by band toward white, and the operator looked at the drawn
figure and rejected it: the bands did not read apart. An enclosure is a
stronger channel than a shade of one -- it says exactly which points belong
together instead of asking the eye to rank four tints -- and it leaves the hue
free to carry the class at full strength, which the tint was eating into. The
function that produced the tint is gone rather than left unused; what it did
survives in `tests/test_toy_bags.py` as the control the drawn colours are
measured against.

**And from `BAGS_PER_CLASS_TOY = 12` onward the LABEL channel stopped reading
too.** That is measured, not feared: at 12 bags per class, 70 of a panel's 72
names overlapped at least one other in `toroide`, 50 of 72 in `lunas`, 53 of 72
in `rollo`. The cut is now 27, so a panel carries 162 names and the channel is
further past that limit rather than back inside it. The hull still says which
points group; the name no longer says which group. It is reported by
`_separacion_de_rotulos` on every draw and printed by the notebook, and the
names are deliberately left on the panel rather than removed -- which channels
a figure carries is the operator's decision and this module's job is to make
the failure visible in numbers. See `BAGS_PER_CLASS_TOY`.

A bag is a contiguous band of the SAME scalar the generator cut the classes
out of -- `t` along the parabola's arm, the torus's major angle, the roll's own
winding -- so cutting bags is the generator's own rule applied one level finer,
never a partition invented here. Being contiguous is what makes each bag's
hull a SLICE of the manifold rather than a sack around scattered points.

**Why CREDA's toyset and not nine synthetic domains.** The alternative --
`MIL_CREDA_Benchmark.toy`'s nine domains -- was measured and rejected: its
noise is declared in absolute units while the three families live on scales
0.013, 0.496 and 5.137, so one nominal sigma inflates one subject 29x, the next
1.34x and the last 1.01x. That is three different experiments wearing one
label. Cutting CREDA's own families into bags leaves the global structure of
the data exactly where it was and changes only the grouping.

**No `__provenance__`, deliberately.** This is a benchmark-side module and it
implements no equation of the formulation: it regroups prior work's generators
and draws them. Stamping a provenance here would claim a link to the
mathematics that does not exist, and would hollow out the one check that ties
code to equations for the modules that really do implement them. The import
direction is the legal one -- `MIL_CREDA_Benchmark` may read `CREDA`, as
`wiring.py` and `schedules.py` already do; what must never happen is `CREDA`
importing anything of MIL-CREDA's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError

from CREDA.toy_experiment import (COL_DURA, DURAS, NOTA_DURA, VISTA_DURA,
                                  dim_ef_dura, entre_dura, perturbar)

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
#:
#: **What it costs the figure, measured rather than argued.** A panel draws
#: `3 * n * 2` hulls and as many names -- both domains, not one -- so this is
#: 162 labels per panel. The label channel stopped reading at 12 and this is
#: further past that limit, not back inside it: `_separacion_de_rotulos`
#: measures exactly this and the notebook prints it, so the limit travels with
#: the picture instead of being left to the eye -- but a reader cannot name a
#: bag off this panel any more, and the hull is the only grouping channel still
#: carrying. That is a cost of this number and not a defect of the drawing:
#: the geometry did not change, the cut got finer again.
#:
#: **And the name itself stops being unambiguous here.** `etiqueta_de_bolsa`
#: concatenates the two 1-based indices, so bag 27 of class 1 is `Bs127`, which
#: reads equally as bag 7 of class 12. At 12 bags the ambiguity needed a reader
#: who knew the class index was a single digit; at 27 the collision is between
#: two bags that both exist in this cut. See that function's own docstring; the
#: format is not changed here, and this is a reported cost rather than a fix.
BAGS_PER_CLASS_TOY = 27

#: Points per domain the FIGURE draws, the same `n` CREDA's own three-families
#: figure draws, so the two figures show the same clouds and only the grouping
#: differs.
#:
#: **It is the figure's number and not the run's, and the two have never been
#: the same.** `toy_run.build_family` passes `Scale.points_per_domain`
#: explicitly, and the pilot has drawn 1260 against this 1800 since the bag
#: size was declared. What the 27-bag cut changes is that the run's number can
#: no longer fall anywhere near this one: 81 bags of 30 instances need a draw
#: whose SMALLEST quantile band still funds 30, and `_instances_per_bag`
#: refuses below it. Measured over the thirteen seeds either scale uses, that
#: floor is 2586 -- 1800 leaves a smallest bag of 20 and refuses on all 39
#: (family, seed) pairs. So the run's two numbers are declared where the
#: programme is, on `toy_run.FULL` and `toy_run.PILOT`, and this one stays at
#: CREDA's so the drawn clouds stay comparable to the figure it mirrors.
PUNTOS_POR_DOMINIO = 1800

#: The offset `CREDA.toy_experiment.construir_dura` puts between the source
#: seed and the target seed. Source and target are INDEPENDENT draws there, and
#: that is the fact the bag correspondence below exists to size.
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

#: How opaque a bag's hull is. Measured on the drawn figure: a hull this faint
#: still reads as one body and still lets its own points, and the points of
#: whatever sits behind it, through. The enclosure has to be visible without
#: becoming the thing one looks at.
ALFA_DEL_CASCO = 0.13

#: How many points a bag needs before a hull can be drawn around it at all.
#: Three points in space span a plane and have no volume, so `ConvexHull`
#: refuses them -- correctly, and the figure answers by drawing no hull for
#: that bag rather than by failing. Four is the minimum for a solid.
MINIMO_PARA_CASCO = 4

#: Point sizes and opacity, per domain. Smaller and fainter than the pre-hull
#: figure drew them: with the hull doing the enclosing, the points are what is
#: INSIDE it, and at the old `13`/`8` they swallowed both the hull and the
#: label sitting on top of them.
TAMANO_DE_PUNTO = {"s": 6, "t": 3}
ALFA_DE_PUNTO = 0.85

#: The white box behind a label is drawn with `boxstyle="round,pad=0.12"`, and
#: that `pad` is a fraction of the font size. Declared here because the
#: crowding measurement has to use the same number the drawing does: measuring
#: the bare glyphs would call two labels separated while their boxes already
#: touch, which is what a reader actually sees.
RELLENO_DEL_ROTULO = 0.12

#: What a bag-to-bag correspondence would score by guessing, at the DECLARED
#: cut. Reported beside every correspondence rate, because a rate read against
#: 1.0 reports a failure where the construction is in fact at most of what is
#: attainable.
#:
#: `correspondencia_de_bolsas` computes its own `1.0 / n` rather than reading
#: this, and that is a correction the twelve-band cut exposed: the function
#: takes `n` and this constant is only the default's value, so a caller that
#: passed any other `n` was handed the chance level of a cut it did not make --
#: silently, because at the shipped default the two agree. This stays as the
#: declared cut's number, which is what the notebook prints before any draw.
AZAR_DE_CORRESPONDENCIA = 1.0 / BAGS_PER_CLASS_TOY


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

    `clase_reproducida` is what checks this is the right scalar, and it is a
    separate function rather than an assertion in here because the check has to
    be runnable on data this module did not draw.
    """
    if nombre == "lunas":
        return X[:, 0]
    if nombre == "toroide":
        return np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi)
    if nombre == "rollo":
        return np.hypot(X[:, 0], X[:, 2])
    raise KeyError(f"no hay parámetro declarado para la familia {nombre!r}")


def clase_reproducida(nombre: str, X: np.ndarray) -> np.ndarray | None:
    """The class `parametro` implies, by the generator's own cut, or `None`.

    This exists so the claim can be MEASURED. A module that cut bands by a
    parameter nobody checked would be cutting bands of something else, and the
    figure would read as the manifold's traverse while showing an arbitrary
    partition -- a failure with no symptom, because the bands would still look
    like bands.

    Returns `None` for `lunas` and that is not a gap. The other two families
    cut their classes by one formula over the parameter, so the formula can be
    replayed. `lunas` cuts by SIGN of `t` for classes 0 and 1 -- two halves of
    one sheet -- while class 2 is the opposite sheet, shifted `+1.5`, which
    puts its `t` range back on top of both. No single formula over `X[:, 0]`
    recovers all three, so returning a formula here would mean inventing one
    and then measuring it against itself. `lunas` is checked per class instead,
    which is what `separacion_de_lunas` reports.
    """
    p = parametro(nombre, X)
    if nombre == "toroide":
        return np.floor(p / (2 * np.pi / 3)).astype(np.int64) % 3
    if nombre == "rollo":
        cortes = np.linspace(1.5 * np.pi, 4.5 * np.pi, 4)
        return np.digitize(p, cortes[1:-1]).astype(np.int64)
    if nombre == "lunas":
        return None
    raise KeyError(f"no hay familia {nombre!r}")


def separacion_de_lunas(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """What `clase_reproducida` can still check on `lunas`, stated as rates.

    Two claims, both about the parameter and neither needing a formula that
    covers three classes at once: class 0 sits at or below zero and class 1 at
    or above it, so the SIGN of `t` separates the two halves of the shared
    sheet; and class 2, the opposite sheet, is displaced, so its mean `t` is
    above both. Together they say the parameter orders the arm of each sheet,
    which is all the bands need of it.
    """
    p = parametro("lunas", X)
    return {
        "clase0NoPositiva": float((p[y == 0] <= 0).mean()),
        "clase1NoNegativa": float((p[y == 1] >= 0).mean()),
        "desplazamientoDeClase2": float(p[y == 2].mean() - p[y != 2].mean()),
    }


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
    becomes. Taking `n` rather than reading the module default matters for the
    same reason `correspondencia_de_bolsas` computes its own `1.0 / n`: a caller
    that cut a different number of bands must not be handed the default's
    permutation.
    """
    rng = np.random.default_rng(semilla + DESPLAZAMIENTO_DE_PERMUTACION)
    return {int(c): rng.permutation(n) for c in np.unique(y)}


def bandas_permutadas(b: np.ndarray, y: np.ndarray,
                      permutacion: dict[int, np.ndarray]) -> np.ndarray:
    """`b` renumbered by `permutacion`: the band column `bags.build` reads.

    One level up from the points on purpose. Applying a permutation to the
    POINTS would scatter each bag across the whole class and every hull would
    become the same sack around it -- the figure would still draw `n` bodies per
    class and they would enclose nothing. This renames bands; it never moves a
    point between them, which is why
    `tests/test_toy_bags.py::test_cada_bolsa_sigue_siendo_una_rebanada_del_parametro`
    still holds afterwards.
    """
    salida = np.empty_like(b)
    for c, orden in permutacion.items():
        m = y == c
        salida[m] = np.asarray(orden)[b[m]]
    return salida


#: The letter each domain contributes to a bag's name.
LETRA_DE_DOMINIO = {"s": "s", "t": "t"}


def etiqueta_de_bolsa(dominio: str, c: int, k: int) -> str:
    """A bag's name: `B` + domain + class + bag, and BOTH indices are 1-based.

    `Bs11` is bag 1 of class 1 in the source; `Bt34` is bag 4 of class 3 in the
    target. The letter is `s` for the source domain and `t` for the target.

    **The `+ 1` is deliberate and is not an off-by-one.** Everywhere else in
    this module classes are `0, 1, 2` and bags are `0 .. n - 1`, because that
    is what indexes an array; the figure shows `c + 1` and `k + 1` because that
    is how the operator names the bags, and a reader counting bags in a picture
    starts at one. Do not "fix" it: the arrays and the labels disagree on
    purpose, this function is the single place the two meet, and
    `tests/test_toy_bags.py` fails on a 0-based label.

    **The concatenation is ambiguous once a bag index reaches two digits, and
    that is stated rather than silently shipped.** At `BAGS_PER_CLASS_TOY = 4`
    both indices were single digits, so `Bs34` had exactly one reading. At 12
    the bag index runs to 12 and bag 12 of class 1 is `Bs112`, which a reader
    can only parse by already knowing the class index is a single digit -- true
    for CREDA's three classes and not a property of this name. At the declared
    27 it is worse than unparseable-without-context: bag 27 of class 1 reads
    `Bs127`, and so would bag 7 of class 12 if there were twelve classes, so
    the reading now depends on the CLASS count as well as on the digit count.
    The format is
    deliberately NOT changed here: separating the two indices is a decision
    about how the figure names a bag, and this module reports the ambiguity
    instead of resolving it on somebody's behalf.
    `test_el_rotulo_dice_dominio_clase_y_bolsa_contando_desde_uno` pins the
    two-digit case so the ambiguity is a recorded fact rather than a silence.

    Refuses a domain it does not know rather than composing a name out of it. A
    wrong letter here is a name that reads perfectly and points at the other
    domain, which is the one error nothing downstream could catch.
    """
    if dominio not in LETRA_DE_DOMINIO:
        raise KeyError(
            f"no hay dominio {dominio!r}: las letras declaradas son "
            f"{sorted(LETRA_DE_DOMINIO)}")
    return f"B{LETRA_DE_DOMINIO[dominio]}{c + 1}{k + 1}"


def casco_de_bolsa(P: np.ndarray) -> ConvexHull | None:
    """The convex hull of one bag's points, or `None` when there is no solid.

    `None` rather than an exception, and in two cases: fewer than
    `MINIMO_PARA_CASCO` points, and points that do span enough of them but lie
    on a plane or a line, which `ConvexHull` refuses with `QhullError`. Both
    are the same fact -- this bag encloses no volume -- and both are answered
    by drawing no hull around it.

    A raise would be the wrong answer because the hull is DECORATION over data
    the figure is already showing. A degenerate bag still has its points, its
    colour and its label on the panel; losing the whole figure because one of
    the panel's `3 * BAGS_PER_CLASS_TOY * 2` bags came out flat would hide
    every readable one to report a missing outline.
    """
    P = np.asarray(P)
    if len(P) < MINIMO_PARA_CASCO:
        return None
    try:
        return ConvexHull(P)
    except QhullError:
        return None


def _dibujar_casco(ax, P: np.ndarray, color) -> Poly3DCollection | None:
    """Add one bag's hull to `ax`, filled and with no edges, or add nothing.

    **Filled and edge-less, and that is a measurement rather than a taste.**
    A hull over a bag's points in three dimensions -- ~22 of them at the
    declared `BAGS_PER_CLASS_TOY = 27` and `PUNTOS_POR_DOMINIO` points per
    domain, ~50 when the cut was twelve bags per class and ~150 when it was
    four -- comes back as tens to hundreds of
    triangles, and drawing the edges of that triangulation puts hundreds of
    segments on top of the points it encloses and on top of the label that
    names it -- measured on the drawn figure, it buried both. The translucent
    body encloses the bag and leaves what is inside it visible, which is the
    whole reason the enclosure replaced the lightness channel.
    """
    casco = casco_de_bolsa(P)
    if casco is None:
        return None
    caras = [P[simplex] for simplex in casco.simplices]
    cuerpo = Poly3DCollection(caras, facecolors=color, edgecolors="none",
                              alpha=ALFA_DEL_CASCO)
    ax.add_collection3d(cuerpo)
    return cuerpo


def _caja_de_rotulo(ax, renderer, nombre, punto, texto):
    """Where one label's white box lands on the canvas, in display pixels.

    Computed rather than asked of the artist, and that is not a preference.
    `Text3D` holds its position in DATA coordinates and only turns it into a
    canvas position inside its own `draw`, inside a context manager that puts
    the data position back on the way out -- so `get_window_extent` afterwards
    reports a box built from the three-dimensional coordinates run through the
    two-dimensional transform. Measured on this figure, that answer said no two
    labels in `lunas` touched while the drawn panel has several sitting on top
    of each other.

    So the projection is redone here, the same way `Text3D.draw` does it, and
    the box is the text's own measured size centred on the result, grown by the
    padding the white box is drawn with.
    """
    xp, yp, _ = proj3d.proj_transform(punto[0], punto[1], punto[2],
                                      ax.get_proj())
    px, py = ax.transData.transform((xp, yp))
    ancho, alto, _ = renderer.get_text_width_height_descent(
        nombre, texto.get_fontproperties(), False)
    relleno = RELLENO_DEL_ROTULO * texto.get_fontsize() * ax.figure.dpi / 72.0
    return (px - ancho / 2 - relleno, px + ancho / 2 + relleno,
            py - alto / 2 - relleno, py + alto / 2 + relleno)


def _separacion_de_rotulos(fig, ax, rotulos) -> dict:
    """Which of the panel's labels actually land on top of each other.

    The labels are placed in three dimensions and read in two, so whether two
    of them collide is a fact about the projection and not about the data: two
    bags a long way apart in space fall on the same spot under one view and
    not under another. That makes "every bag can be read by name" a claim, and
    a claim the figure rests on is measured by the code that draws it rather
    than believed.

    Measured on the boxes as DRAWN and not on the projected centres, because
    what decides whether a name can be read is whether another name is printed
    over it, and that depends on the text's own size. `encimados` are the pairs
    whose boxes intersect, and `separacionMinima` is the smallest gap, in
    points, between two that do not. Touching counts as overlapping, so a pair
    never appears in both.

    Points and not pixels, so the number means the same thing at any dpi and
    can be read against the font size the labels are set in.

    Nothing is moved and nothing is hidden on the strength of this. It travels
    beside the figure so that where the geometry crowds, a reader is handed the
    pairs instead of being left to find them.

    `rotulos` is a sequence of `(nombre, punto, texto)` and must be read after
    the panel's limits, view and placement are all settled: the three of them
    enter the projection, so a measurement taken earlier describes a figure
    nobody is going to look at.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    por_punto = 72.0 / fig.dpi
    cajas = [(nombre, _caja_de_rotulo(ax, renderer, nombre, punto, texto))
             for nombre, punto, texto in rotulos]
    encimados, separaciones = [], []
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            (na, ca), (nb, cb) = cajas[i], cajas[j]
            dx = max(0.0, max(ca[0], cb[0]) - min(ca[1], cb[1]))
            dy = max(0.0, max(ca[2], cb[2]) - min(ca[3], cb[3]))
            if dx == 0.0 and dy == 0.0:
                encimados.append([na, nb])
            else:
                separaciones.append((float(np.hypot(dx, dy)) * por_punto,
                                     na, nb))
    separaciones.sort()
    return {
        "rotulos": len(cajas),
        "pares": len(cajas) * (len(cajas) - 1) // 2,
        "encimados": encimados,
        "separacionMinima": separaciones[0][0] if separaciones else None,
        "parMasCercano": ([separaciones[0][1], separaciones[0][2]]
                          if separaciones else None),
    }


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
    #: trust, and because the correspondence below is measured on both.
    Xt_crudo: np.ndarray


def construir(nombre: str, fn, semilla: int = 0,
              n: int = PUNTOS_POR_DOMINIO,
              bolsas_por_clase: int = BAGS_PER_CLASS_TOY) -> Familia:
    """Source, target, labels and bag indices for one family.

    The draw is CREDA's, down to the seed offset: `construir_dura` takes the
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

    - **It is CREDA's own choice.** `CREDA.toy_experiment.corrida` calls
      `partir(ys, semilla=semilla)` and `partir(yt, semilla=semilla)` -- the
      SAME seed for both domains, so the two splits are paired by construction
      rather than by accident. Sharing one draw here is that choice made
      structural: it holds whatever each class's size is, where two calls on one
      seed hold only while the class counts match.
    - **Independent permutations would reintroduce the defect between the
      domains.** `correspondencia_de_bolsas` exists because bag `k` of class `c`
      is the `k`-th band of the same parameter in BOTH domains; that is the
      bag-to-bag correspondence ceiling the local term is read against, and the
      published accuracy is read on the TARGET's evaluation role. Permuting the
      two independently makes source bag `k` and target bag `k` different
      stretches of the manifold, which collapses that ceiling to chance and
      makes the target's evaluation role a region the source never trained on --
      the same gap this permutation exists to remove, moved from between the
      roles to between the domains.
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


def _centroides(X, y, b, c, n):
    return np.array([X[(y == c) & (b == k)].mean(0) for k in range(n)])


def _tasa_de_correspondencia(Xs, ys, bs, Xt, yt, bt, n) -> tuple[int, int]:
    """How often a target bag's centroid is nearest its OWN source bag's."""
    aciertos = total = 0
    for c in np.unique(ys):
        fuente = _centroides(Xs, ys, bs, c, n)
        for k in range(n):
            Q = Xt[(yt == c) & (bt == k)]
            if not len(Q):
                continue
            d = ((fuente - Q.mean(0)) ** 2).sum(1)
            aciertos += int(d.argmin() == k)
            total += 1
    return aciertos, total


def correspondencia_de_bolsas(fam: Familia,
                              n: int = BAGS_PER_CLASS_TOY) -> dict[str, float]:
    """The bag-to-bag correspondence, and the two very different readings of it.

    CREDA's source and target are INDEPENDENT draws, so there is no
    point-to-point correspondence to recover -- but there IS a bag-to-bag one
    by construction, because bag `k` of class `c` is the `k`-th quantile of the
    same parameter in both domains. This number has to exist before anyone
    reads an alignment result on this toyset: a rate read against 1.0 reports a
    failure where the construction never offered 1.0 to begin with.

    It is measured twice because the two measurements answer different
    questions and only reporting both is honest:

    - `enLaVariedad` -- centroids taken on the target as DRAWN. This is the
      correspondence the construction puts there, so it is the ceiling: it is
      what is available to be recovered at all.
    - `enElDominioObservado` -- centroids taken on the target after
      `perturbar`. This is not a ceiling, it is the FLOOR: what nearest-centroid
      proximity alone scores on the domains a method actually sees, before any
      alignment. Measured on this toyset it sits at or just above chance, so
      proximity in the observed space carries almost none of the
      correspondence, and every point of it an alignment recovers is the
      alignment's own work.

    Reporting only the second as "the ceiling" would be the error this
    docstring exists to prevent: it would declare a construction broken when
    what it measures is the shift the experiment is about.
    """
    a_variedad, total = _tasa_de_correspondencia(
        fam.Xs, fam.ys, fam.bs, fam.Xt_crudo, fam.yt, fam.bt, n)
    a_observado, _ = _tasa_de_correspondencia(
        fam.Xs, fam.ys, fam.bs, fam.Xt, fam.yt, fam.bt, n)
    return {
        "enLaVariedad": a_variedad / total,
        "enElDominioObservado": a_observado / total,
        "bolsasComparadas": total,
        # `1.0 / n` and NOT `AZAR_DE_CORRESPONDENCIA`: the chance level of the
        # cut this call made, not of the cut the module defaults to. They agree
        # at the default, which is exactly why reading the constant here was a
        # defect nothing could see -- see that constant's own note.
        "azar": 1.0 / n,
    }


def lectura(fam: Familia, n: int = BAGS_PER_CLASS_TOY) -> dict:
    """Everything measurable about one family, in one dictionary.

    The table in the notebook is rendered straight out of this, so no number in
    the report is typed by a person. `claseReproducida` is `None` exactly where
    `clase_reproducida` has no single formula, and the notebook prints what is
    there rather than a dash it chose.
    """
    rep = clase_reproducida(fam.nombre, fam.Xs)
    tamanos = [int(((fam.ys == c) & (fam.bs == k)).sum())
               for c in np.unique(fam.ys) for k in range(n)]
    por_clase = [[int(((fam.ys == c) & (fam.bs == k)).sum()) for k in range(n)]
                 for c in np.unique(fam.ys)]
    return {
        "familia": fam.nombre,
        "nota": NOTA_DURA[fam.nombre],
        "direccionesEfectivas": float(dim_ef_dura(fam.Xs)),
        "separacionEntreClases": float(entre_dura(fam.Xs, fam.ys)),
        "puntosPorBolsa": {"minimo": min(tamanos), "maximo": max(tamanos),
                           "dispersionDentroDeClase":
                               max(max(f) - min(f) for f in por_clase)},
        "claseReproducida": None if rep is None else float((rep == fam.ys).mean()),
        "lunas": separacion_de_lunas(fam.Xs, fam.ys) if rep is None else None,
        "correspondencia": correspondencia_de_bolsas(fam, n),
        "bolsasPorClase": n,
    }


def lecturas(familias=DURAS, semilla: int = 0,
             n: int = PUNTOS_POR_DOMINIO,
             bolsas_por_clase: int = BAGS_PER_CLASS_TOY) -> list[dict]:
    """`lectura` for every family, in the order `DURAS` declares them."""
    return [lectura(construir(nombre, fn, semilla, n, bolsas_por_clase),
                    bolsas_por_clase)
            for nombre, fn in familias]


def _ejes_compartidos(construidas, tolerancia: float = 0.05) -> dict:
    """Whether the panels really share a range -- measured, never assumed.

    The tidy version of a three-panel grid collapses the axis labels into one
    for the whole figure and keeps tick labels at the border. That collapse is
    a CLAIM about the data: it says the panels live on one scale. Here they do
    not -- three families on three different scales -- so the code measures the
    ranges and refuses the collapse, and the figure keeps each panel's own
    limits instead of drawing three incomparable scales as though they were
    one. A grid whose panels quietly differ in range, drawn as if they shared
    one, is a figure that lies in the direction of tidiness.
    """
    rangos = []
    for fam in construidas:
        todo = np.vstack([fam.Xs, fam.Xt])
        rangos.append((todo.max(0) - todo.min(0)))
    rangos = np.array(rangos)
    mayor, menor = rangos.max(0), rangos.min(0)
    discrepancia = float((mayor / np.clip(menor, 1e-12, None)).max() - 1.0)
    return {
        "compartidos": bool(discrepancia <= tolerancia),
        "discrepanciaRelativa": discrepancia,
        "tolerancia": tolerancia,
        "rangoPorPanel": [[float(v) for v in r] for r in rangos],
    }


@dataclass(frozen=True)
class FiguraDeFamilias:
    """What `figura_familias` hands back: the picture, the file, the facts."""

    fig: object
    ruta: Path
    lecturas: list[dict]
    ejes: dict
    #: Per panel, what `_separacion_de_rotulos` measured on the labels as they
    #: were actually projected. Travels with the figure because the crowding it
    #: reports is a limit of this figure that a reader is owed in numbers, not
    #: left to notice.
    rotulos: list[dict]


def figura_familias(destino: Path, semilla: int = 0,
                    n: int = PUNTOS_POR_DOMINIO,
                    bolsas_por_clase: int = BAGS_PER_CLASS_TOY,
                    familias=DURAS,
                    nombre_archivo: str = "toy_familias_bolsas.pdf",
                    ) -> FiguraDeFamilias:
    """The three families with their bags, written to `destino` and returned.

    Returns the live figure so the caller can show it inline AND the path of
    the archived vector copy, because a report owes both: a cell that only
    prints a filename has reported a filename, and a notebook that only shows
    the picture leaves nothing a paper can be built from.

    The facts come back with it rather than being recomputed by the caller, so
    the table under the figure and the figure itself cannot disagree.

    `rotulos` comes back for the same reason, and from `BAGS_PER_CLASS_TOY = 12`
    onward it is the reason: the figure used to claim that each bag can be told
    apart AND named, and the naming half of that claim no longer holds. Measured
    on the drawn panels at 12 bags per class, 50 of 72 names in `lunas`, 70 of
    72 in `toroide` and 53 of 72 in `rollo` sat on top of at least one other,
    and the smallest gap between two that did not touch was 0.01 pt. At four
    bags per class the same measurement read 7 of 24, 0 of 24 and 0 of 24. The
    declared cut is now 27, so a panel carries 162 names; what the field reports
    is the measurement on the cut that was actually drawn, and
    `tests/test_toy_bags.py` asserts the direction at the declared cut rather
    than pinning the counts of a cut the module no longer makes.

    So what the figure still carries is the HULL -- which points group together
    -- and what it has lost is the NAME of each group. That is said here, and
    printed by the notebook from this same field, rather than left for a reader
    to discover by trying to read a panel. Nothing is moved and nothing is
    hidden on the strength of it: the geometry is the toyset's and the cut is
    the operator's, and a figure that quietly dropped the names would be
    answering a decision that is not this module's.
    """
    import matplotlib.pyplot as plt

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / nombre_archivo

    construidas = [construir(nombre, fn, semilla, n, bolsas_por_clase)
                   for nombre, fn in familias]
    ejes = _ejes_compartidos(construidas)
    #: Every point of every panel, used for the limits ONLY when the ranges
    #: were measured to agree. Computed unconditionally so the two branches
    #: below differ in which data decides the limits and nothing else.
    comun = np.vstack([np.vstack([fam.Xs, fam.Xt]) for fam in construidas])

    fig = plt.figure(figsize=(19.5, 9.2))
    textos_por_panel = []
    for i, fam in enumerate(construidas):
        a = fig.add_subplot(1, len(construidas), i + 1, projection="3d")
        a.set_box_aspect(None, zoom=1.18)
        rotulos = []
        for c in sorted(np.unique(fam.ys)):
            # El tono va a FUERZA PLENA: con el casco encerrando la bolsa, el
            # color ya no tiene que codificar dos cosas a la vez, y aclararlo
            # era justo lo que le sacaba fuerza a la clase.
            col = COL_DURA[c]
            for k in range(bolsas_por_clase):
                for X, y, b, dominio, marca in (
                        (fam.Xs, fam.ys, fam.bs, "s", "x"),
                        (fam.Xt, fam.yt, fam.bt, "t", "o")):
                    P = X[(y == c) & (b == k)]
                    if not len(P):
                        continue
                    # Sin `edgecolors`: la `x` se dibuja con el trazo del
                    # borde, así que apagarlo --- como hacía la `o` --- la
                    # borraría de la figura.
                    a.scatter(P[:, 0], P[:, 1], P[:, 2], color=col,
                              marker=marca, s=TAMANO_DE_PUNTO[dominio],
                              linewidths=.5, alpha=ALFA_DE_PUNTO)
                    _dibujar_casco(a, P, col)
                    rotulos.append(
                        (etiqueta_de_bolsa(dominio, int(c), k), P.mean(0), col))
        # Los rótulos se agregan después de los puntos y los cascos para que
        # queden encima de ambos, y con una caja blanca detrás porque el
        # nombre de una bolsa sobre su propia nube de puntos del mismo color
        # es ilegible --- medido mirando la figura, no supuesto.
        textos_por_panel.append((fam.nombre, a, [
            (nombre, punto,
             a.text(punto[0], punto[1], punto[2], nombre, color=col,
                    fontsize=8.5, fontweight="bold", ha="center",
                    va="center",
                    bbox=dict(boxstyle=f"round,pad={RELLENO_DEL_ROTULO}",
                              fc="white", ec=col, lw=.7, alpha=.93)))
            for nombre, punto, col in rotulos]))
        # Per-panel limits because `_ejes_compartidos` MEASURED that the panels
        # do not share a range. Had it measured that they do, the decoration
        # would collapse into one scale for the whole figure -- which is the
        # tidy version, and is only honest when the measurement allows it.
        todo = (comun if ejes["compartidos"] else np.vstack([fam.Xs, fam.Xt]))
        margen = 0.05 * (todo.max(0) - todo.min(0))
        bajo, alto = todo.min(0) - margen, todo.max(0) + margen
        a.set_xlim(bajo[0], alto[0])
        a.set_ylim(bajo[1], alto[1])
        a.set_zlim(bajo[2], alto[2])
        a.view_init(*VISTA_DURA[fam.nombre])
        # Decoration appears once, and WHERE once is depends on the same
        # measurement. On one shared scale the tick labels belong on the border
        # panel alone; on three different scales they belong nowhere, because a
        # number on an axis would invite exactly the cross-panel comparison
        # `_ejes_compartidos` just measured as invalid.
        if not (ejes["compartidos"] and i == 0):
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_zticklabels([])
        a.set_title(
            f"{i + 1}. {fam.nombre}\n{NOTA_DURA[fam.nombre]}\n"
            f"direcciones efectivas {dim_ef_dura(fam.Xs):.2f} de 3 · "
            f"separación entre clases {entre_dura(fam.Xs, fam.ys):.0f}%",
            fontsize=10.5, linespacing=1.5)
        if i == 0:
            # Two legends, because there are two channels to explain: the first
            # says which hue is which class, the second says what the enclosure
            # and the name on it mean. One legend with 3x4x2 entries does not
            # read, and naming every bag in a legend would only repeat what is
            # already written on the panel.
            # Las muestras de ESTA leyenda son explícitas en vez de salir de un
            # `label` colgado de uno de los `scatter`: colgarlas de un `scatter`
            # las ataría a la bolsa por la que el bucle pasa primero, y una
            # leyenda que dice «la clase» mostrando el color de una bolsa es la
            # confusión que esta figura ya tuvo una vez.
            # Cada entrada dice las DOS numeraciones de su clase: el arreglo la
            # llama `c` y el rótulo la llama `c + 1`. Un lector que ve `Bs11`
            # sobre un punto azul y una leyenda que dice «clase 0» tiene que
            # resolver el desfase solo, y la figura es el único lugar donde las
            # dos numeraciones se ven juntas.
            clases = [Line2D([], [], marker="x", linestyle="", markersize=7,
                             markeredgewidth=1.6, color=COL_DURA[c],
                             label=f"clase {c}  ·  en el rótulo, {c + 1}")
                      for c in sorted(COL_DURA)]
            prim = a.legend(handles=clases, loc="upper left", fontsize=9,
                            title="el TONO: la clase")
            prim.get_title().set_fontsize(8.5)
            a.add_artist(prim)
            # Las muestras de la segunda leyenda son las dos cosas que
            # reemplazaron a la luminosidad, dibujadas con los mismos valores
            # con que se dibujan en el panel: el casco con su propia opacidad
            # --- se ve tan tenue acá como allá, que es el punto --- y la caja
            # blanca con el borde de la clase que lleva el nombre. El esquema
            # completo del nombre está en el pie, así que acá va un ejemplo.
            bolsa = [
                Patch(facecolor=COL_DURA[0], edgecolor="none",
                      alpha=ALFA_DEL_CASCO, label="la bolsa, encerrada entera"),
                Patch(facecolor="white", edgecolor=COL_DURA[0], linewidth=.7,
                      label=f"su nombre: `{etiqueta_de_bolsa('s', 0, 0)}`, "
                            f"`{etiqueta_de_bolsa('t', 2, 3)}` …"),
            ]
            seg = a.legend(handles=bolsa, loc="lower left", fontsize=8,
                           handlelength=1.6,
                           title="el CASCO y su RÓTULO: la bolsa")
            seg.get_title().set_fontsize(8.5)

    menor_bolsa = min(min(int(((fam.ys == c) & (fam.bs == k)).sum())
                          for c in np.unique(fam.ys)
                          for k in range(bolsas_por_clase))
                      for fam in construidas)
    fig.suptitle(
        f"Las tres familias, con cada clase partida en {bolsas_por_clase} "
        f"bolsas de al menos {menor_bolsa} puntos   ·   "
        "`x` fuente, `o` blanco\n"
        "el TONO dice la clase, el CASCO encierra la bolsa y el RÓTULO dice "
        "cuál es (`Bs11` = bolsa 1, clase 1, fuente · `Bt34` = bolsa 4, clase "
        "3, blanco), el MARCADOR dice el dominio, y la POSICIÓN no se toca: "
        "nada del toyset se pierde\n"
        "una bolsa es una banda contigua del MISMO parámetro del que el "
        "generador cortó las clases — el brazo de la parábola, el ángulo mayor "
        "del toro, el enrollado del rollo — así que es su propia regla un "
        "nivel más fina",
        fontsize=11.0, y=0.115, linespacing=1.7)
    # `bottom` at 0.255 because the footer is three lines and the third was
    # clipped by the border at 0.145 --- measured by looking at the figure, not
    # supposed.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.255,
                        wspace=0.03)
    # Los rótulos se miden ACÁ: después de los límites, de la vista y del
    # acomodo de los paneles, porque los tres entran en dónde termina cayendo
    # cada caja, y antes de guardar, para que lo que se informa sea lo que el
    # archivo tiene adentro.
    rotulos_por_panel = [dict(_separacion_de_rotulos(fig, eje, textos),
                              familia=nombre)
                         for nombre, eje, textos in textos_por_panel]
    fig.savefig(ruta)
    return FiguraDeFamilias(
        fig=fig, ruta=ruta,
        lecturas=[lectura(fam, bolsas_por_clase) for fam in construidas],
        ejes=ejes, rotulos=rotulos_por_panel)
