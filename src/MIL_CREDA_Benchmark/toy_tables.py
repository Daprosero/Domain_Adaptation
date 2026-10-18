"""Markdown tables over the artefact `toy_run.guardar` leaves on disk.

**This module reads and cannot run anything.** It imports no `torch`, no
`optuna` and not `toy_run` itself, so a notebook that imports it has no path to
a cell of the sweep, a search or a training loop -- the plot notebook's claim
that it does not train is a property of its import closure and not of a comment
somebody can delete. The loader stays with the writer: `toy_run.cargar` is what
opens the files, because `tests/test_toy_run.py` crosses that join by running
`guardar` to disk and asking `cargar` what it sees, and splitting the pair would
leave the round trip tested on one side only.

`scipy.stats` is the one numeric import, and it is here for the declared tests
and nowhere else: a Wilcoxon, a Friedman, a paired t, a Shapiro-Wilk, a t
quantile and the rank transform Friedman's mean ranks need are what
`### <familia> — los tests declarados` is made of, and reimplementing any of
them would be a second implementation of a test whose method was fixed before
anything ran. It is not a path to a training loop -- it holds no model, no optimizer and
no generator -- so the property above is unchanged.

It lives beside `tables.py` for the same reason `toy_run.py` lives beside
`harness.py`: this is not a campaign over `config.DOMAINS`, it reads none of a
campaign's coordinates, and the two renderers share no record.

**A fourth rule was retired, and it is named here so nobody re-derives it.**
It said both of phase 1's readings travel together: phase 1 CHOOSES its best
epoch on the source `valid` role, the same frozen encoder is then read on the
published role, and a table showing one reading without the other misleads in
both directions. `_refuse_half_the_pair` enforced that over a whole block, and
`render_familia_escalera` satisfied it with a second small grid under the same
heading. **Settled by the operator: the per-family block emits exactly one
table, because CREDA's own emits exactly one, and same structure as CREDA means
same structure.** The operator read the rule's reasoning before ruling. So the
grid, its paragraph and the refusal are gone rather than worked around, and the
selection reading is not lost -- `phase1ValidAccuracy` is still measured and
still in the record for every cell; it simply stops being rendered here. Do not
re-raise it as an open gap, and do not reintroduce the grid under another
heading.

**And the stamp rule was retired, the mechanism going with it rather than
moving somewhere quieter.** It said the stamp travels inside the output: `escala_sello`
was appended by the one function that builds a table, so the line saying these
numbers are a trial could not be dropped without editing this module, and a
stamp written into a notebook's prose is one cell away from being deleted with
the paragraph. **Settled by the operator: no table carries it, because CREDA's
plot notebook emits no footer under any of its ten blocks and same structure as
CREDA means same structure.** The operator read that reasoning before ruling,
twice. So the append is gone, `_bloque` no longer takes the record it fed, and
the tests that asserted a stamp under every rendering are deleted rather than
weakened. What a reader loses is the per-table line: that these numbers are a
trial is still stated by the record itself and printed by the notebook's own
preamble cell.

**And on 2026-10-15 the operator extended that ruling to the one place the stamp
had survived.** It was still reaching the latent-space figure's suptitle through
`toy_run._sello_plano`, which read `escala_sello` -- the figure had been outside
the surface of the pass that retired it. That figure is now held to CREDA's
frame, and CREDA's suptitle is ONE line, so the stamp went from the figure and
`escala_sello` went with it, this module's last reader of its own stamp. The
note where it stood records the ruling. Do not re-raise it as an open gap, and
do not re-introduce the line as prose in a cell or as a second figure text.

Two rules shape every renderer here, and each of them is a way a table reads
as evidence while being none:

* **A number appears in exactly one table.** Two renderings of one measurement
  are two things that can drift apart and the reader cannot tell which moved.
  That is why a quantity that belongs to ONE rung of the ladder
  is dashed on the rungs it does not belong to instead of being repeated down
  the block. The declared exceptions are three, and each of them is CREDA's own
  shape rather than an economy taken here:

  - the chance level, which the framing states and each family's header states
    again because that header is CREDA's own; both call `azar`;
  - the two LEVEL columns of `render_familia_sin_contra_con`, which reduce the
    same field as the family table's `blanco · <rol>` for two of its three
    rungs. CREDA's table A carries them beside its `Δ`, and a reader cannot
    check the sign of a difference against the two numbers it was taken between
    if the two numbers are in another table. The family table prints the
    dispersion beside the mean and this one does not -- CREDA's shape, because
    the dispersion this table reports is the DIFFERENCE's, in the interval;
  - the `MIL-CREDA` against `` `.predict` `` row, which is one row of table A
    and one row of table B wherever AdaBN has a rung. CREDA's two tables share
    exactly that row -- in table A it sits in the band of architectures, each
    with its own `t` and its own Shapiro; in table B it sits inside the Friedman
    family of three rungs, under Holm.

  None of the three can drift: the level columns reduce the same field with the
  same `statistics.fmean` the family table's reduction uses, and the shared row
  is one call of `_par_medido` over the same two series. What a duplication rule protects
  against is two implementations of one number, and there is one in each case.
  The cost -- a reader meets the same value twice -- is the one the operator
  asked for when they asked for CREDA's two tables.
* **A table is grouped BY FAMILY, never by kind of quantity.** Every renderer
  whose rows are cells takes a `familia` and shows that family alone, so one
  block of the document holds one toyset's figure and one toyset's numbers
  together -- the shape `CREDA/Notebooks/Toyset_Plot.ipynb` already has. The
  shape this replaced crossed all three families inside each table, one table
  per kind of quantity, and a reader who wanted to know what happened to
  `toroide` had to assemble it out of five tables. The grouping is enforced
  rather than remembered: `claves_de_celda` takes the filter, so a renderer
  that forgot it would have to build its own key list to get the other
  families back.

**And a family block carries CREDA's table, not five of our own.** It carried
five -- a mapped CREDA table, the ladder, the objective's terms, the search and
the bandwidth -- accumulated one per stretch, each added and none removed. They
were folded into one, which came out at seventeen columns and nine rows. What a
family block carries now is `tabla_dataset` itself: `render_familia_escalera`
renders CREDA's eight columns and seven rows, and nothing else: the one grid
CREDA has no column for -- phase 1's pair -- was retired with the rule above,
so the block is one grid as well as one table. Every reading the seventeen
columns carried and these eight do not is accounted for in the banner above the
renderers. Most of it had moved into a computed conclusion under each table; the
conclusions are GONE as of this revision -- CREDA's own plot notebook prints
none, and that is the document this one is held to -- so those readings left the
document with them. The banner names every one, because a reading removed in
silence is a measurement that stops existing without anybody deciding it should.

`render_*` names starting with `render_familia` are the per-family half and
every one of them carries `familia` in its signature; `renderizados` reads that
signature rather than a list of names, so a renderer added later is expanded
over the declared families without being told.
"""

from __future__ import annotations

import inspect
import statistics
from dataclasses import dataclass, field

import scipy.stats as _st

SEP = "|"

#: The cell with no number in it. One string, so a dash that means «this rung
#: has no such reading» and a dash that means «this test could not run» render
#: identically on purpose: both say the table is not hiding a number.
NADA = "—"

@dataclass(frozen=True)
class Columna:
    """One column of a table, and which record fields it reads.

    `lee` is the declaration a reader of this module checks a column against:
    a derived column -- a difference, a ratio -- names the fields behind it, so
    what a column computes with is readable without reading its lambda.
    """

    titulo: str
    valor: object
    lee: tuple[str, ...] = field(default=())


# ------------------------------------------------------------------ the record

def _escala(registro: dict) -> dict:
    return registro["scale"]


def _banderas(registro: dict) -> dict:
    return registro["flags"]


def _bolsas_por_rol(registro: dict) -> dict:
    """How many bags each role holds IN TOTAL, from the split the run realized.

    Per class in the record, times the classes the scale declares. The requested
    fractions are not it: `realizedSplit` exists because integer bags cannot
    honour a fraction and the realized counts are what was measured on.
    """
    escala = _escala(registro)
    por_clase = escala["realizedSplit"]["bagsPerClass"]
    return {rol: cuenta * escala["classes"] for rol, cuenta in por_clase.items()}


def resolucion(registro: dict) -> dict:
    """What the finest difference between two cells is, and why.

    An accuracy over a finite number of bags is a fraction with that number as
    its denominator, so it takes one of `bolsas + 1` values and two cells cannot
    differ by less than one bag. A reader comparing two cells needs this before
    the comparison, not after it.
    """
    bolsas = _bolsas_por_rol(registro)
    evaluacion = bolsas["test"]
    return {
        "bolsasEvaluacion": evaluacion,
        "bolsasValidacion": bolsas["valid"],
        "bolsasEntrenamiento": bolsas["train"],
        "valoresPosibles": evaluacion + 1,
        "puntosPorcentualesPorBolsa": 100.0 / evaluacion,
    }


def azar(registro: dict) -> float:
    """The landmark every accuracy here is read against, from the declared classes.

    Computed and never written: the day the class count moves, a number typed
    into a sentence goes on naming the old one.
    """
    return 1.0 / _escala(registro)["classes"]


#: The role every published accuracy of the per-family table is measured on.
#: `test`, which is both CREDA's own word for it and the name THIS record's
#: split carries -- the two agree because the role was renamed in the record
#: (`toy_run.Scale.realized_split`), not because the header prints a word the
#: record does not use. That direction is the whole point: the header states
#: the role the number in front of the reader was measured on, so the record is
#: what has to say `test`. `rol_publicado` refuses it where the record's own
#: split does not carry that role, so a renamed role fails loud instead of
#: leaving a header naming a role nobody measured -- and that refusal is what
#: an artefact written before the rename trips, correctly.
ROL_PUBLICADO = "test"


def rol_publicado(registro: dict) -> str:
    """The role name the header states, checked against the record's own split."""
    roles = sorted(_escala(registro)["realizedSplit"]["bagsPerClass"])
    if ROL_PUBLICADO not in roles:
        raise KeyError(
            f"el registro no parte en un rol {ROL_PUBLICADO!r}; parte en "
            f"{roles}. El encabezado de cada familia nombra el rol sobre el que "
            "se midió lo que publica, así que un rol renombrado tiene que "
            "reventar acá y no quedar nombrado en una cabecera que nadie midió")
    return ROL_PUBLICADO


#: The role phase 1 CHOSE its best epoch on, which is not the role this table
#: publishes. Declared beside `ROL_PUBLICADO` because the whole of phase 1's pair
#: is that the two are DIFFERENT roles, and checked the same way.
ROL_SELECCION = "valid"


def rol_seleccion(registro: dict) -> str:
    """The role phase 1 chose on, checked against the record's own split.

    Refuses it where it IS the published role: the pair exists because selecting
    and judging happen on disjoint material, so a record that named one role
    for both would make this grid a reading of itself.
    """
    roles = sorted(_escala(registro)["realizedSplit"]["bagsPerClass"])
    if ROL_SELECCION not in roles:
        raise KeyError(
            f"el registro no parte en un rol {ROL_SELECCION!r}; parte en "
            f"{roles}: la fase 1 elige sobre algún rol, y si no es ése la "
            "pareja de la fase 1 nombra uno que nadie midió")
    if ROL_SELECCION == ROL_PUBLICADO:
        raise ValueError(
            "el rol que elige y el rol que publica son el mismo, así que la "
            "pareja de la fase 1 sería una lectura de sí misma")
    return ROL_SELECCION


#: **`escala_sello` is RETIRED, and this note stands where it stood.** It built
#: the line that says what a reading was measured under -- «escala «pilot» ·
#: ENSAYO · revisión `...` · dispositivo `...` — ningún número de esta lectura se
#: cita como resultado» -- and the operator retired it from every table on a
#: previous pass. What kept the function alive was its one remaining caller,
#: `toy_run._sello_plano`, which stripped the markdown off it and hung it as the
#: SECOND line of the latent-space figure's suptitle.
#:
#: **Settled by the operator on 2026-10-15: that figure carries CREDA's frame,
#: and CREDA's suptitle is one line.** So the stamp goes from the figure, this
#: was its last reader, and the function goes rather than standing for nobody.
#: Do not re-raise it as an open gap, and do not reintroduce the line as prose
#: in a cell: what a reader loses is still said by the record itself and by the
#: preamble cell of `Toyset_Plot.ipynb`, which prints the scale and the revision
#: before anything is drawn.


def familias(registro: dict) -> list[str]:
    """The families the record declares, in its own order."""
    return list(_banderas(registro)["families"])


def normalizaciones(registro: dict) -> list[str]:
    """The normalizations the record declares, in its own order.

    **Read and never written, and the sentence that used to count them here is
    why.** It said «two here and three in CREDA», which was true of the record
    that existed when it was written and is not true of the one on disk: that
    one declares the three CREDA draws and leaves `ausentes` empty. Every
    heading, every column count and every «the N architectures» in the document
    comes from this call for exactly that reason -- a count written down would
    have gone on saying two over a grid of three.
    """
    return list(_banderas(registro)["normalizations"])


def ausentes(registro: dict) -> list[str]:
    """The normalizations CREDA draws that this benchmark has no backbone for.

    Read off the record and never written: the run that produced it is what
    knows which rows it could not fill, and an absence a reader has to notice
    is an absence nobody declared.
    """
    return list(_banderas(registro).get("normalizationsAbsent") or ())


def _exigir_familia(registro: dict, familia: str) -> str:
    if familia not in familias(registro):
        raise KeyError(
            f"el registro no declara la familia {familia!r}; declara "
            f"{familias(registro)}. Una tabla pedida para una familia que no "
            "está saldría vacía, y una tabla vacía se lee igual que una tabla "
            "cuyos números son malos")
    return familia


def claves_de_celda(registro: dict,
                    familia: str | None = None) -> list[tuple[str, str, str, int]]:
    """Every cell of the sweep, in the order the record declares rather than the
    order the JSON happens to hold.

    `familia` narrows it to one toyset, and it is a parameter HERE rather than a
    filter each renderer writes for itself: the per-family grouping is the whole
    shape of the document, and a renderer that had to drop rows after the fact
    could get the other families back by building its own key list.

    Fails closed on a missing cell, and on a family the record never declared.
    A loop that skipped one would render a shorter table than the grid it claims
    to cover, and nothing in the output would say so.
    """
    banderas = _banderas(registro)
    escala = _escala(registro)
    pedidas = (familias(registro) if familia is None
               else [_exigir_familia(registro, familia)])
    claves = []
    for nombre in pedidas:
        for normalizacion in banderas["normalizations"]:
            for brazo in banderas["arms"]:
                for semilla in escala["report_seeds"]:
                    clave = SEP.join((nombre, normalizacion, brazo, str(semilla)))
                    if clave not in registro["cells"]:
                        raise KeyError(
                            f"el registro declara la celda {clave!r} en sus "
                            "banderas y no la trae: una tabla más corta que la "
                            "grilla que dice cubrir no lo dice en ninguna parte")
                    claves.append((nombre, normalizacion, brazo, int(semilla)))
    return claves


def celdas(registro: dict, familia: str | None = None) -> list[dict]:
    """The cell records, in declared order."""
    return [registro["cells"][SEP.join((f, n, b, str(s)))]
            for f, n, b, s in claves_de_celda(registro, familia)]


# ------------------------------------------------------------------ the builder

def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _pp(x: float) -> str:
    return f"{x:+.2f} pp"


def _num(x: float, cifras: int = 4) -> str:
    return f"{x:.{cifras}f}"


def _media_desvio(valores: list[float], cifras: int = 4) -> str:
    """CREDA's `media ± desvío` over the report seeds, and the MEAN ALONE at one.

    `pstdev` and not `stdev`: CREDA's own reduction is `np.std`, which is the
    population deviation, and two implementations of one reduction that differ
    in their denominator produce two tables nobody can compare.

    **At one report seed there is no dispersion, so none is printed.** `± 0.0000`
    there is not a small spread, it is the claim that a spread was measured and
    came out zero -- and what happened is that nothing was measured. The header
    says how many seeds the reduction ran over, so the reader is told which of
    the two shapes they are looking at rather than left to infer it from an
    absent sign.
    """
    if not valores:
        raise ValueError(
            "no hay valores que reducir: una celda vacía se lee igual que una "
            "celda cuyo número es malo, así que la tabla no se arma")
    media = statistics.fmean(valores)
    if len(valores) == 1:
        return _num(media, cifras)
    return f"{_num(media, cifras)} ± {_num(statistics.pstdev(valores), cifras)}"


def _media_desvio_pp(valores: list[float]) -> str:
    """The same reduction over a quantity read in percentage points."""
    if not valores:
        raise ValueError("no hay valores que reducir")
    media = statistics.fmean(valores)
    if len(valores) == 1:
        return _pp(media)
    return f"{_pp(media)} ± {_num(statistics.pstdev(valores), 2)} pp"


def _mediana_entera(valores: list[int]) -> str:
    """CREDA's `_med`: the MEDIAN of an epoch or a step count, as an integer.

    The median and not the mean, which is CREDA's choice and its reason holds
    here: an epoch index is a count of whole epochs, and a mean of them names a
    point no run ever stood at.
    """
    if not valores:
        raise ValueError("no hay valores que reducir")
    return f"{int(statistics.median(valores))}"


def _solo_la_grilla(columnas: list[Columna], filas: list) -> str:
    """The markdown of one grid and nothing else: no title, no footnote.

    Not a second builder. `_bloque` below is the one builder and this is the
    loop it runs per table it was asked for, lifted out so the block can hold
    more than one grid without the title or the footnote moving with them.

    **The spacing is CREDA's, and the two rows are spaced differently because
    CREDA spaces them differently.** `src/CREDA/toy_plots.py::_md` writes
    `| a | b |` for the header and for every body row, and `|---|---|` with no
    padding for the rule between them. Markdown renders padded and unpadded
    pipes identically, so nothing about the rendered table says which one this
    emits -- and the two documents are compared as TEXT, which is where the
    difference is. Written here, in the one builder, because a renderer that
    spaced its own rows would be a second answer to «what does a cell of this
    document look like».

    `SEP` is not used for it: that constant is the RECORD key's separator
    (`familia|norma|brazo|semilla`), which happens to be the same character and
    is a different thing -- a table whose padding followed a record key's
    spelling would move the day the key did.
    """
    lineas = ["| " + " | ".join(c.titulo for c in columnas) + " |",
              "|" + "|".join(["---"] * len(columnas)) + "|"]
    for fila in filas:
        lineas.append("| " + " | ".join(c.valor(fila) for c in columnas) + " |")
    return "\n".join(lineas)


def _bloque(grupos: list[tuple[str | None, list[Columna], list]]) -> str:
    """The one place a table is built, so what a cell looks like is one answer.

    Every renderer goes through here, so CREDA's cell spacing is a property of
    this function and not a habit of the callers. A renderer added later gets it
    without being told, and `tests/test_toy_tables.py` watches the call rather
    than reading the source.

    **It takes no record, and it used to.** The argument existed to feed the
    scale stamp this function appended under every block; the operator retired
    the stamp (the module docstring records the ruling), so the parameter went
    with it rather than staying as a record nothing reads.

    Each group is `(title, columns, rows)` or `(title, columns, rows, footnote)`.

    The title is prose above its grid -- a block of two grids needs to say which
    is which. It may carry a measurement only when that measurement belongs to
    its OWN grid and to no other: the Friedman statistic of one architecture's
    three rungs is CREDA's own line above CREDA's own table, and it is rendered
    exactly once because there is one of it per grid. What the rule forbids is a
    number OF THE BLOCK OR OF THE RUN in a per-grid title -- the scale's epochs,
    the chance level -- which would be one number rendered as many times as the
    block has grids.

    The footnote is the line UNDER a grid, and it exists because a reason that
    belongs to a whole table was being rendered once per row. `render_familia_tests`
    carried a `por qué no hay test` column whose every row held the same
    paragraph; CREDA's shape is one line under the table, and a column of
    repeated prose is what this argument replaced. A footnote may carry a count
    that bounds its own grid (the size of the family Holm corrected inside) and
    never a reading the grid already prints.

    **An empty list of groups is legitimate and builds no grid**, which is how a
    renderer says it has nothing to tabulate -- `render_familia_tests` when no
    architecture carries re-estimable buffers -- while still going through the
    one builder. It returns the empty string: with the stamp retired there is
    nothing a grid-less block carries, and the sentence that replaces the table
    is the renderer's own.
    """
    normalizados = [(grupo[0], grupo[1], grupo[2],
                     grupo[3] if len(grupo) > 3 else None)
                    for grupo in grupos]
    partes = []
    for titulo, columnas, filas, pie in normalizados:
        grilla = _solo_la_grilla(columnas, filas)
        if titulo:
            grilla = f"{titulo}\n\n{grilla}"
        if pie:
            grilla = f"{grilla}\n\n{pie}"
        partes.append(grilla)
    return "\n\n".join(partes)


def _tabla(columnas: list[Columna], filas: list) -> str:
    """One grid, through the one builder. The single-grid case of `_bloque`."""
    return _bloque([(None, columnas, filas)])


def _etiqueta(clave: tuple[str, str, str, int]) -> str:
    familia, normalizacion, brazo, semilla = clave
    return f"{familia} / {normalizacion} / {brazo} / s{semilla}"


def _normalizacion(clave: tuple[str, str, str, int]) -> str:
    """The row key of a per-family table.

    Inside one family's block the family is the heading, the arm is declared
    once for the whole run and the report seed is in the scale table, so the one
    coordinate that still varies from row to row is the normalization. Printing
    the other three per row would render three constants as columns.
    """
    return clave[1]


def _por_celda(registro: dict, familia: str | None = None) -> list[dict]:
    """Rows keyed by cell: the coordinates and the record, together."""
    return [{"clave": clave, "celda": celda}
            for clave, celda in zip(claves_de_celda(registro, familia),
                                    celdas(registro, familia))]


# ----------------------------------------------------------------- the renderers
#
# CREDA's `tabla_dataset` IS the per-family table, column for column and row for
# row, and the map below is what each of its eight columns reads out of this
# artefact. Its header is
#
#     método · normalización · época fuente · paso que eligió · fuente · test ·
#     blanco · test · vigía: fuente conservada · blanco pico · test (oráculo)
#
# * `método` -- the rung, by CREDA's own row label (`ROTULO_ESCALON`). Three
#   labels and not a column of one arm repeated: `.predict` is this same run read
#   at its step-0 checkpoint and AdaBN is that checkpoint with the target's
#   normalization statistics re-estimated, so two of CREDA's three rungs cost no
#   second sweep and are measured by `toy_run.run_cell` itself.
# * `normalización` -- the architecture, in CREDA's words (`NOMBRE_FILA`, copied
#   from `CREDA.toy_experiment` rather than imported: importing it would put
#   `torch` on this module's path and this module's whole claim is that it
#   cannot train).
# * `época fuente` -- the median `phase1BestEpoch`, dashed on AdaBN exactly as
#   CREDA dashes it.
# * `paso que eligió` -- the median `phase2BestStep`, on the one rung that ran a
#   phase 2 and dashed on the two that did not.
# * `fuente · test` -- `sourceAccuracy`, the frozen phase-1 encoder read on the
#   source role nothing selected on. It prints on all three rungs because it IS
#   the same reading on all three: AdaBN re-estimates the TARGET encoder's
#   buffers on a deep copy and phase 2 moves `theta_t` alone, both of which the
#   record measures (`adabn_reading` touches nothing, `sourceMoved` is read per
#   cell). CREDA's own table repeats it the same way and for the same reason.
# * `blanco · test` -- the rung's own target reading on the test role:
#   `predictTargetAccuracy`, `adabnTargetAccuracy`, `targetAccuracy`.
# * `vigía: fuente conservada` -- `watchmanRestored`, which is CREDA's
#   `vigia_fin`: the watchman read AFTER the best step's weights are restored,
#   «el vigía en el punto donde paró, para la columna de reporte». Dashed on the
#   rungs that ran no phase 2.
# * `blanco pico · test (oráculo)` -- the maximum of the per-step oracle series
#   `toy_run.run_cell` records, over the SAME `test` role the column beside it
#   is read on, printed on the rung that ran a phase 2 and dashed on the two
#   that did not. See `SERIE_BLANCO` for which role and why, and
#   `toy_run.TARGET_SERIES` for the boundary that keeps a target label out of
#   every criterion. The gap between this column and `blanco · test` is what
#   separates «the term is the problem, and no stopping rule saves it» from
#   «the stopping rule is»: CREDA's own numbers show that split, where
#   `lunas/batchnorm` publishes 0.3808 against a 0.5400 ceiling while AdaBN
#   gets 0.4997.
#
#   It stays DERIVED from presence: a record whose cells carry no series dashes
#   here, which is what every artefact produced before the run recorded one
#   does. CREDA dashes the same cell for every row that has no trajectory and
#   its reason is the same: repetir ahí su valor final lo haría comparable con
#   un máximo que nunca midieron.
#
# ---------------------------------------------------------------------------
# WHAT THE SEVENTEEN-COLUMN TABLE CARRIED THAT THESE EIGHT DO NOT
# ---------------------------------------------------------------------------
#
# The table this replaced had seventeen columns and nine rows: three rungs per
# architecture, with every reading of four folded-in tables printed on the rung
# that produced it and dashed on the other two. CREDA's table has eight columns
# and seven rows, and the operator asked for CREDA's. Every reading that left is
# accounted for here, because a column removed in silence is a measurement that
# stops existing without anybody deciding it should.
#
# STOPPED BEING RENDERED, and stays in the record:
#
# * `phase1ValidAccuracy` and the valid -> test drop -- these were a second
#   small grid below CREDA's own, held there by `_refuse_half_the_pair`. The
#   operator retired both: CREDA's per-family cell emits exactly one table, and
#   same structure as CREDA means same structure. The reading was NOT deleted
#   -- `phase1ValidAccuracy` is measured per cell and is in the artefact --
#   it stopped being rendered. The module docstring carries the ruling and the
#   reasoning it overrode.
#
# LEFT THE DOCUMENT WITH THE CONCLUSIONS, and this is the only place that says so.
#
# These columns were folded out of the eight into a computed conclusion under
# each table, and then the conclusions themselves were removed: CREDA's plot
# notebook prints no conclusion in any of its three family blocks or anywhere
# else in the file, and the operator pointed at that document and asked for its
# form. The general report rule that asks for «a conclusion that comes from a
# function» does not outrank the specific form it was pointed at.
#
# So these are NOT rendered anywhere now. They were not moved into the framing
# prose either: a number typed into a paragraph cannot be recomputed, which is
# the defect that rule exists to stop, so moving them there would have traded a
# removal for a worse one. Every one of them is still in the record
# `Toyset_Generator.ipynb` wrote, under the field named here, and is one
# `toy_run.cargar` away from whoever wants it.
#
# * `phase1Epochs` / `phase2Steps` -- the count of cells whose stop chose an
#   INTERIOR epoch or step. A stop that chose the last point never fired, and
#   nothing in the document now says whether either one did.
# * `watchmanBest` -- how far the restored watchman sits from the best one phase
#   2 saw. The one thing the pair said that neither number says alone.
# * `adabnBatches` -- how many target batches AdaBN re-estimated over, where it
#   has a rung.
# * `adabnNote` / `phase2Note` -- the per-cell notes. WHY AdaBN has no rung
#   where it has none survives, because it does not need the record to be said:
#   it is the plot notebook's framing above the three blocks, and the two
#   MEASURED rows of `render_arquitecturas` (`adabnNormLayers`, `adabnLayers`)
#   are what make it a reading rather than a claim -- a backbone that normalizes
#   and holds nothing is the grid's control, and those two rows are where a
#   reader checks which row that is.
# * `supervised` / `contributionGlobal` / `contributionLocal` /
#   `adaptationShare` -- which term commanded the gradient, and whether the
#   supervised term dominated. The magnitudes had already left the table; the
#   count has now left too.
# * `weightingCostRatio` / `weightingCostOmegaSum` -- the smallest ratio and the
#   range of the sum of Eq. (37)'s weights.
# * `sourceMoved` / `sourceBuffersMoved` -- the MEASURED freeze of phase 1,
#   weights and buffers, over every cell of the family. `objetivo` still carries
#   its landmark (`fuente movida`, zero and nothing else), so what a reader
#   meets is the target and no longer the measurement.
# * `kernelSigma` and its two medians -- the bandwidth the search chose against
#   the two scales one band has to serve.
# * `¿la fuente supera el azar?` -- the count of cells clearing chance in the
#   source. The per-cell source accuracy is still a COLUMN of CREDA's eight, so
#   what left is the count and not the reading.
#
# GOES, and is prose rather than a measurement:
#
# * `qué se mueve` and `cómo se midió` -- the two columns were the same three
#   declared sentences in all three family blocks, which is one explanation
#   rendered nine times. They are stated once, in the plot notebook's framing
#   above the three blocks, where an explanation belongs. What was MEASURED in
#   `cómo se midió` did not go with them: `adabnLayers` and `adabnNormLayers` are
#   two measured rows of `render_arquitecturas` already.
#
# ---------------------------------------------------------------------------
# THE FIVE TABLES THAT BECAME ONE, and what happened to each
# ---------------------------------------------------------------------------
#
# A family block used to carry five tables. They accumulated one per stretch:
# each added, none removed. CREDA's block is one figure and one table plus the
# declared tests, and the operator asked for that shape.
#
# * `render_familia_escalera` -- SURVIVES, and is now CREDA's eight columns.
# * `render_familia`, `render_familia_objetivo`, `render_familia_ancho_banda` --
#   folded in, then folded out again by the accounting above: their readings
#   were in the conclusion under each table, and left the document when the
#   conclusions did. The accounting above names each one and its record field.
# * `render_familia_busqueda` -- DROPPED, and not folded. Every fact it
#   rendered -- the trial counts, the best valid accuracy, the ties, the
#   criteria, the searched dimensions and the six chosen values -- is already
#   rendered by `MIL-CREDA/Notebooks/Toyset_Generator.ipynb`'s own search cell,
#   out of the same four record fields (`records1`, `records2`, `hp1`, `hp2`).
#   It was therefore one measurement in two documents, which is the duplication
#   rule across the two notebooks of one run, and the document that OWNS a
#   search is the one that ran it. Nothing is left computing it: the renderer,
#   its conclusion and their tests are gone. Of the two readings its conclusion
#   carried that the generator does not state, one survives and one does not:
#   the valid-role and test-role bag counts are a landmark in `objetivo`, and
#   the bandwidth against its two medians left the document with the
#   conclusions, as the accounting above records.

#: What each normalization keeps between batches, NAMED once. The names are
#: what the layer IS and the record carries no field for them; HOW MANY of each
#: a backbone holds is measured instead, in the two rows of
#: `render_arquitecturas` that read `adabnNormLayers` and `adabnLayers`. The
#: pair is the division that matters: a sentence can say what a layer type
#: keeps, and only the record can say what this run's encoder actually held.
ESTADO_DE_NORMA = {
    "ninguna": "ninguno",
    "batchnorm": "`running_mean`, `running_var`",
    "layernorm": "ninguno — y eso es lo que la hace el CONTROL de esta grilla",
}

SOBRE_QUE_NORMALIZA = {
    "ninguna": "nada",
    "batchnorm": "el **lote**, por canal",
    "layernorm": "los **features**, dentro de cada muestra",
}


def render_arquitecturas(registro: dict) -> str:
    """The architectures this run declares, and the one CREDA has that it has not.

    Transposed the way CREDA's own architecture table is -- one column per
    normalization, one row per fact -- because the facts are of different kinds
    and a row of mixed kinds does not read.

    Two things here are measured rather than asserted. The COUNT of columns
    comes from the record, so a heading or a sentence claiming three over a grid
    of two is contradicted by the table beside it. And the parameter count is
    checked for being the same in every family before it is printed once: the
    input is three-dimensional in all three families, so it should be, and
    «should be» is exactly the kind of belief that quietly stops being true. If
    it ever differs the cell prints the range it found instead of a number that
    describes one family.
    """
    declaradas = normalizaciones(registro)
    faltantes = ausentes(registro)
    todas = _por_celda(registro)
    por_norma = {norma: [fila for fila in todas
                         if _normalizacion(fila["clave"]) == norma]
                 for norma in declaradas}

    def _constante(filas: list[dict], campo: str) -> str:
        """One integer the whole column agrees on, or the range it found.

        Same discipline as `_parametros` below and for the same reason: «it
        should be the same in every family» is a belief, and a belief printed
        as a number is the kind that quietly stops being true. If the cells
        disagree the cell says so instead of describing one of them.
        """
        medidos = {fila["celda"][campo] for fila in filas}
        if len(medidos) == 1:
            return str(next(iter(medidos)))
        return f"{min(medidos)} a {max(medidos)} según la celda"

    def _parametros(norma: str) -> tuple[str, bool]:
        medidos = {fila["celda"]["parameters"] for fila in por_norma[norma]}
        if len(medidos) == 1:
            return f"{next(iter(medidos)):,}", True
        return (f"{min(medidos):,} a {max(medidos):,} según la familia"), False

    base = declaradas[0]
    base_cuenta = min(fila["celda"]["parameters"] for fila in por_norma[base])
    filas = [
        {"que": "backbone",
         **{norma: f"`{_banderas(registro)['normalizations'][norma]}`"
            for norma in declaradas},
         **{norma: "sin backbone" for norma in faltantes}},
        {"que": "celdas medidas",
         **{norma: str(len(por_norma[norma])) for norma in declaradas},
         **{norma: "ninguna" for norma in faltantes}},
        {"que": "parámetros del brazo",
         **{norma: _parametros(norma)[0] for norma in declaradas},
         **{norma: "sin medición" for norma in faltantes}},
        {"que": f"parámetros, relativo a «{base}»",
         **{norma: _num(min(fila["celda"]["parameters"]
                            for fila in por_norma[norma]) / base_cuenta)
            for norma in declaradas},
         **{norma: "sin medición" for norma in faltantes}},
        # Two MEASURED rows beside the named one, and they are what make the
        # third column a control rather than a claim: a backbone can normalize
        # and still hold nothing AdaBN can act on, and only the record can say
        # which of the two a row is. `adabnNormLayers` is what the encoder
        # normalizes with; `adabnLayers` is what AdaBN found to re-estimate.
        {"que": "capas que normalizan (medido en la corrida)",
         **{norma: _constante(por_norma[norma], "adabnNormLayers")
            for norma in declaradas},
         **{norma: "sin medición" for norma in faltantes}},
        {"que": "capas que AdaBN puede reestimar (medido en la corrida)",
         **{norma: _constante(por_norma[norma], "adabnLayers")
            for norma in declaradas},
         **{norma: "sin medición" for norma in faltantes}},
        {"que": "estado que la fase 2 reestima",
         **{norma: ESTADO_DE_NORMA.get(norma, "sin declarar")
            for norma in declaradas + faltantes}},
        {"que": "normaliza sobre",
         **{norma: SOBRE_QUE_NORMALIZA.get(norma, "sin declarar")
            for norma in declaradas + faltantes}},
        {"que": "por qué está o no está",
         **{norma: "declarada y corrida" for norma in declaradas},
         # Derived from the record and not from a count written here. The
         # sentence this replaced said «`mlp.MLP_BACKBONES` declara dos
         # backbones y ninguno de los dos es LayerNorm», which was a literal
         # two and a named layer type in a cell that only fires when the record
         # says something is missing -- so the day the record said otherwise,
         # the cell would have gone on naming the old grid.
         **{norma: (f"AUSENTE: el registro declara "
                    f"{len(declaradas)} backbone(s) y ninguno es `{norma}`, "
                    "así que esta fila no se puede llenar con nada de este "
                    "paquete")
            for norma in faltantes}},
    ]
    # La primera columna lleva título --- `qué`, el mismo que las otras dos
    # tablas tecleadas hacia abajo de este módulo --- y no la celda vacía con
    # que CREDA encabeza la suya: un encabezado vacío al principio de una fila
    # de markdown es una celda que cualquier lector de tablas, incluido el de
    # los tests de este repositorio, confunde con el borde.
    columnas = ([Columna("qué", lambda r: r["que"])]
                + [Columna(norma, (lambda n: lambda r: r[n])(norma))
                   for norma in declaradas]
                + [Columna(f"{norma} (CREDA la tiene)",
                           (lambda n: lambda r: r[n])(norma))
                   for norma in faltantes])
    return _tabla(columnas, filas)


#: CREDA's intervention ladder, in the order it declares: least intervention
#: first. The order is the reading -- a ladder read from the top says nothing,
#: because the question each rung answers is «what did the one below it leave
#: on the table» -- so it is a declared tuple and never the order a dict
#: happened to iterate in.
#:
#: Each entry is `(rung, CREDA's own row label, which record field carries its
#: target reading)`. The field names are `toy_run.run_cell`'s own, and the ladder
#: is READ off the record rather than recomputed here: this module imports
#: nothing that can run.
#:
#: **The label is CREDA's and the rung's short name is this repository's.** The
#: `método` column prints the label, because the table is read beside CREDA's
#: own; everything that compares rungs -- `pares_de_la_escalera`,
#: `comparar_familia`, `render_familia_tests` -- prints the short name, because a
#: pair of markdown-bold labels in a cell of its own table reads as emphasis
#: rather than as two names.
#:
#: The two sentences that used to be here -- what each rung moves and how it was
#: read -- are no longer declared, and that is the accounting in the banner
#: above: they were the same three sentences printed in all three family blocks,
#: so they are stated once in the plot notebook's framing instead of nine times
#: in its tables.
ESCALERA = (
    ("`.predict`", "`.predict` (sin adaptar)", "predictTargetAccuracy"),
    ("AdaBN", "**AdaBN** (sólo buffers, sin optimizador)", "adabnTargetAccuracy"),
    ("MIL-CREDA", "**MIL-CREDA** · HP sintonizados", "targetAccuracy"),
)

#: Each rung's label, by its short name. Derived from `ESCALERA` so a rung added
#: there arrives here without anybody remembering to.
ROTULO_ESCALON = {escalon: rotulo for escalon, rotulo, _ in ESCALERA}

ESCALON_FASE_1 = ESCALERA[0][0]
ESCALON_ADABN = ESCALERA[1][0]
ESCALON_METODO = ESCALERA[2][0]

#: CREDA's own row label for each architecture, from
#: `CREDA.toy_experiment.NOMBRE_FILA`, copied rather than imported: that module
#: imports `torch`, and this one's whole claim -- pinned by
#: `tests/test_toy_tables.py` against its import lines -- is that it has no path
#: to anything that trains.
#:
#: `_rotulo_norma` refuses an architecture it has no label for, so a fourth
#: backbone reddens here instead of printing the record's own key in a column a
#: reader is comparing with CREDA's.
NOMBRE_FILA = {"ninguna": "sin normalización",
               "batchnorm": "BatchNorm",
               "layernorm": "LayerNorm"}

#: The record field the per-step target reading travels in, and the name the
#: `blanco pico · test (oráculo)` column reads. `toy_run.run_cell` now records
#: it -- one reading per phase-2 step, over the target's TEST role -- and
#: `toy_run.TARGET_SERIES_FIELD` is THIS constant rather than a second literal,
#: so the writer and the reader cannot drift apart.
#:
#: **Which role, and why it had to be that one.** The published accuracy is read
#: over `test`, so the peak is read over `test`: the two sit in one row and the
#: gap between them is the reading. The trap avoided is the per-frame target
#: reading that already existed -- `trajectoryPrediction`, over every bag of the
#: target domain at the frame stride -- whose maximum printed beside a final
#: value over the test role would put two statistical units in one line.
#: `toy_run.TARGET_SERIES` declares the rest: that it reads `y_t`, that reading
#: it to MEASURE is not the leakage that reading it to CHOOSE would be, and that
#: it enters no criterion and neither search.
#:
#: The dash this column used to carry in every row was a reading of the
#: artefact's own silence, and it still is: a record whose cells carry no series
#: dashes here, which is every artefact produced before the run recorded one.
SERIE_BLANCO = "targetAccuracySeries"


def _rotulo_norma(norma: str) -> str:
    """CREDA's word for one architecture, or a refusal."""
    if norma not in NOMBRE_FILA:
        raise KeyError(
            f"no hay rótulo de CREDA para la arquitectura {norma!r}; conocidas: "
            f"{sorted(NOMBRE_FILA)}. Una fila que imprimiera la clave del "
            "registro donde va la palabra de CREDA haría esta tabla ilegible al "
            "lado de la suya, y el rótulo no se puede derivar de la clave")
    return NOMBRE_FILA[norma]


def arquitecturas_con_escalon_adabn(registro: dict,
                                    familia: str | None = None) -> list[str]:
    """The architectures AdaBN has a rung on, DERIVED from buffer presence.

    **By what the run FOUND to re-estimate and never by the layer's name.**
    `adabnLayers` is `adabn_reading`'s own count of buffer-carrying layers in
    that cell's backbone, so the rung exists where it is positive and nowhere
    else. Deciding by name would give a rung to the architecture that normalizes
    and keeps nothing -- the control of this grid -- and a number there would be
    `.predict` under another name.
    **An architecture without a rung contributes no row at all**, which is
    CREDA's own shape: its table carries ONE AdaBN row, for BatchNorm, and the
    other two architectures simply do not appear in it. A row with a dash across
    it would say AdaBN was measured and came out empty.

    Refuses an architecture whose cells disagree: whether a backbone carries
    re-estimable buffers is a property of the backbone, so cells answering
    differently mean the record is describing two encoders under one name.
    """
    salida = []
    for norma in normalizaciones(registro):
        suyas = [fila["celda"] for fila in _por_celda(registro, familia)
                 if _normalizacion(fila["clave"]) == norma]
        medidos = {celda["adabnLayers"] > 0 for celda in suyas}
        if len(medidos) > 1:
            raise ValueError(
                f"las celdas de `{norma}` no se ponen de acuerdo en si su "
                "backbone lleva buffers reestimables: llevar buffers es una "
                "propiedad del backbone, así que celdas que contestan distinto "
                "dicen que el registro describe dos codificadores con un nombre")
        if medidos != {True}:
            continue
        # Los dos campos tienen que decir lo mismo. Una arquitectura con capas
        # reestimables y sin acierto de AdaBN daría una fila con raya, y una
        # fila con raya dice «AdaBN se midió y salió vacío», que es distinto de
        # «AdaBN no tiene escalón acá» --- la lectura por la que la fila no
        # está.
        sin_numero = [celda for celda in suyas
                      if celda["adabnTargetAccuracy"] is None]
        if sin_numero:
            raise ValueError(
                f"`{norma}` declara capas reestimables y "
                f"{len(sin_numero)} de sus {len(suyas)} celdas no traen "
                "acierto de AdaBN: una fila con raya diría que AdaBN se midió "
                "y salió vacía, que es otra cosa que no tener escalón")
        salida.append(norma)
    return salida


def encabezado_familia(registro: dict, familia: str) -> str:
    """CREDA's own header line, every part of it derived from the record.

    `**<familia> · <n> semillas · media ± desvío · azar <azar> · <rol>, sólo
    reporte**`, and at ONE report seed the reduction clause says what is honest
    for that case instead of `media ± desvío` over a dispersion nothing
    measured. The seed count is in the line either way, so the reader is told
    which of the two they are reading.

    **The chance level is here as well as in `objetivo`, and that is a
    decision.** `objetivo` is the framing's computed half and says WHY that
    value is the landmark; this says what the table under it is read against,
    which is what CREDA's own header does. Both call `azar`, so they are one
    measurement rendered twice and not two that can drift -- the cost is that a
    reader meets it twice, and the operator asked for CREDA's header.
    """
    _exigir_familia(registro, familia)
    semillas = _escala(registro)["report_seeds"]
    cuantas = ("1 semilla" if len(semillas) == 1
               else f"{len(semillas)} semillas")
    reduccion = ("media de una sola semilla, sin dispersión"
                 if len(semillas) == 1 else "media ± desvío")
    return (f"**{familia} · {cuantas} · {reduccion} · "
            f"azar {_num(azar(registro))} · {rol_publicado(registro)}, "
            "sólo reporte**")


def _filas_escalera(registro: dict, familia: str) -> list[dict]:
    """CREDA's seven rows: the floor once per architecture, AdaBN only where it
    has a rung, the method once per architecture.

    Grouped BY RUNG and not by architecture, which is CREDA's order and the
    reading: the three floor rows beside each other are the three architectures'
    starting points, and reading them in one band is how the AdaBN row below
    them and the method rows below that are read as what each intervention
    added.

    Each row carries the cells of one (architecture, rung) -- one per report
    seed -- because every column of this table is a reduction over the seeds and
    not a single cell's value.
    """
    con_adabn = arquitecturas_con_escalon_adabn(registro, familia)
    filas = []
    for escalon, _, campo in ESCALERA:
        for norma in normalizaciones(registro):
            if escalon == ESCALON_ADABN and norma not in con_adabn:
                continue
            celdas_ = [fila["celda"] for fila in _por_celda(registro, familia)
                       if _normalizacion(fila["clave"]) == norma]
            if not celdas_:
                raise KeyError(
                    f"el registro declara la arquitectura {norma!r} y la tabla "
                    f"de {familia} no tiene ninguna celda suya: una tabla a la "
                    "que le falta una arquitectura declarada no lo dice en "
                    "ninguna parte, y sus filas restantes salen correctas")
            filas.append({"escalon": escalon, "norma": norma, "campo": campo,
                          "celdas": celdas_})
    return filas


def render_familia_escalera(registro: dict, familia: str) -> str:
    """ONE toyset's table, which is CREDA's `tabla_dataset`: eight columns and
    seven rows, and one grid -- phase 1's pair was retired from this block.

    The module docstring records that ruling and that `phase1ValidAccuracy` is
    still measured and still in the record for every cell.

    **Three rungs out of one run, and that is what makes the table honest.**
    `.predict` and AdaBN are not arms and cost no second sweep: the first is the
    step-0 checkpoint of this very run -- so it shares phase 1 with the rung
    below it bit for bit, which is CREDA's own reason for reading it there
    instead of training a separate floor -- and the second is that same
    checkpoint with the target's normalization statistics re-estimated.

    **AdaBN is ONE row and not a rung repeated per architecture.** It exists
    where the backbone holds buffers to re-estimate, which is measured
    (`arquitecturas_con_escalon_adabn`) and never read off the layer's name; an
    architecture without them contributes no AdaBN row at all, which is CREDA's
    own shape. The architecture that normalizes and keeps nothing is still the
    control of this grid, and what says so is its ABSENCE from the AdaBN band
    beside `render_arquitecturas`'s two measured layer counts -- not a dash with
    a paragraph in it.

    **ONE grid, because CREDA's own per-family cell emits one.** There used to
    be a second one here carrying phase 1's selection-role reading and the drop
    to the published role; the operator retired it and the refusal that held it,
    and the module docstring records both the ruling and the reasoning it
    overrode. `phase1ValidAccuracy` is still in the record for every cell.

    Paired by construction: every rung of one architecture comes from the same
    cells, the same seeds and the same phase 1.
    """
    _exigir_familia(registro, familia)
    filas = _filas_escalera(registro, familia)

    def _de(campo: str):
        return lambda r: [celda[campo] for celda in r["celdas"]]

    def _acierto_blanco(r) -> str:
        valores = [celda[r["campo"]] for celda in r["celdas"]]
        if any(valor is None for valor in valores):
            return NADA
        return _media_desvio(valores)

    def _epoca(r) -> str:
        # Dashed on AdaBN exactly as CREDA dashes it: that rung is read at the
        # step-0 checkpoint and chose no epoch of its own. The floor and the
        # method print the same number because it IS the same phase 1.
        if r["escalon"] == ESCALON_ADABN:
            return NADA
        return _mediana_entera(_de("phase1BestEpoch")(r))

    def _del_metodo(lector):
        """A column that only the rung which ran a phase 2 can fill."""
        return lambda r: (lector(r) if r["escalon"] == ESCALON_METODO else NADA)

    def _pico(r) -> str:
        # The maximum of the per-step oracle series, over the SAME role the
        # published value is read on -- which is what makes the gap between the
        # two columns readable as «the stopping rule did not hit the best
        # pass» rather than as «two different splits».
        #
        # Still DERIVED from presence and never written as a dash: a record
        # whose cells carry no series still dashes here, which is the state
        # every artefact produced before `toy_run.run_cell` recorded one was
        # in, and `tests/test_toy_tables.py` removes the field from a copy to
        # prove the dash is a reading and not a literal.
        #
        # Wrapped in `_del_metodo`, so it prints on the rung that ran a phase 2
        # and dashes on the two that did not. CREDA dashes the same two cells
        # (`toy_plots.tabla_dataset`), and for a reason that is not a
        # convention: `.predict` and AdaBN are read at the step-0 checkpoint
        # and have no trajectory, so the method's own peak printed on their row
        # would attribute this run's best pass to an intervention that took no
        # step.
        series = [celda[SERIE_BLANCO] for celda in r["celdas"]
                  if SERIE_BLANCO in celda]
        if len(series) != len(r["celdas"]) or not all(series):
            return NADA
        return _media_desvio([max(serie) for serie in series])

    columnas = [
        Columna("método", lambda r: ROTULO_ESCALON[r["escalon"]]),
        Columna("normalización", lambda r: _rotulo_norma(r["norma"])),
        Columna("época fuente", _epoca, ("phase1BestEpoch",)),
        Columna("paso que eligió",
                _del_metodo(lambda r: _mediana_entera(_de("phase2BestStep")(r))),
                ("phase2BestStep",)),
        Columna(f"fuente · {rol_publicado(registro)}",
                lambda r: _media_desvio(_de("sourceAccuracy")(r)),
                ("sourceAccuracy",)),
        Columna(f"blanco · {rol_publicado(registro)}", _acierto_blanco,
                tuple(campo for _, _, campo in ESCALERA)),
        Columna("vigía: fuente conservada",
                _del_metodo(
                    lambda r: _media_desvio(_de("watchmanRestored")(r))),
                ("watchmanRestored",)),
        Columna(f"blanco pico · {rol_publicado(registro)} (oráculo)",
                _del_metodo(_pico), (SERIE_BLANCO,)),
    ]

    # ONE grid, which is what CREDA's own per-family cell emits. Phase 1's
    # selection-role grid used to sit under this same heading; the module
    # docstring records why it is gone and that `phase1ValidAccuracy` is still
    # in the record.
    return (encabezado_familia(registro, familia) + "\n\n"
            + _bloque([(None, columnas, filas)]))


# --------------------------------------------------------- los tests declarados
#
# CREDA fills `### <familia> — los tests declarados` with `tabla_sin_contra_con`
# (two arms, paired t) and `tabla_tres_brazos` (three arms, Friedman plus
# pairwise Wilcoxon corrected with Holm inside the family of pairs of ONE
# toyset). The subsection was ABSENT here while this toyset had one arm and
# every column of either table was a two-or-more-arm quantity.
#
# **It is TWO tables here as well, and the three mappings each were a decision.**
# It was one table of nine rows -- three pairs by three architectures -- with an
# architecture column CREDA has in one table and not the other, a median column
# CREDA prints in neither, and a `por qué no hay test` column whose every row
# held the same paragraph. One explanation rendered nine times is the defect this
# shape replaced, and the three decisions behind the two tables are:
#
# 1. **What `con CREDA` / `sin CREDA` is here.** `PAREJA_SIN_CON`: the method's
#    rung against the floor rung of the SAME run. CREDA's `sin CREDA` is
#    `acc_predict`, the step-0 checkpoint, so the pair is the regularizer on
#    against nothing on -- not against AdaBN, which is an intervention of its own
#    and exists only where a backbone keeps buffers, so a column of it would be
#    empty in most rows of a table that has one row per architecture. The columns
#    are named for the two rungs they hold rather than with CREDA's two words
#    over different quantities.
# 2. **Which architecture table B is.** DERIVED, by
#    `arquitecturas_con_escalon_adabn` -- buffer presence measured in the run,
#    never the layer's name. CREDA renders BatchNorm because that is where all
#    three rungs exist; here the same condition is asked of the record, one table
#    per architecture that answers yes, and one sentence instead of a table when
#    none does.
# 3. **What `PROMEDIO` averages.** The normalization, WITHIN each seed, which
#    gives one pair per report seed and never the pooled cell-seed pairs: seed 0
#    appears in every architecture, so the pooled count is not a count of
#    independent observations. The point estimate comes out the same either way
#    and the DENOMINATOR does not, which is the quietest way a statistic goes
#    wrong -- the table keeps its shape and multiplies its `t`.
#
# **Three rungs exist now, so the subsection is buildable, and what it can
# report at this scale was MEASURED before it was written.** The pilot declares
# ONE report seed, and at one block of a paired design neither declared test
# produces a reading:
#
# * `scipy.stats.friedmanchisquare` over three treatments and ONE block returns
#   chi2 = 2.000000 and p = 0.367879 for EVERY input -- measured over
#   ([1],[2],[3]), ([3],[2],[1]), ([0],[100],[50]) and a tied triple. With one
#   block the within-block ranks are always a permutation of 1..k, so the
#   statistic is a function of the TREATMENT COUNT and of nothing in the data.
# * `scipy.stats.wilcoxon` over ONE pair returns statistic 0.0 and p = 1.000000
#   for every difference, measured at +0.1, -0.9 and +0.0001.
# * `scipy.stats.ttest_rel` over one pair returns `nan` for the statistic and
#   `nan` for the p, with `df = 0`.
# * `scipy.stats.shapiro` returns `nan` below THREE values, measured at one and
#   at two and real from three up -- which is why Shapiro-Wilk has a floor of its
#   own, `BLOQUES_MINIMOS_SHAPIRO`, above the floor the other two share.
#
# A number that cannot come out different is not a measurement, for exactly the
# reason an assertion that cannot fail proves nothing -- and printing 0.367879
# under a header that says `p` would lend the whole apparatus's authority to a
# constant. So below `BLOQUES_MINIMOS` the test columns are dashed, the reason
# is stated ONCE under the table, and the paired DIFFERENCE -- which is exact at
# one seed and is a real reading -- is printed. An honest table saying what it
# cannot yet compute beats an absent subsection, and both beat a table of nulls.
#
# The reason is a footnote and no longer a column, and that is the whole of the
# third change: every row of one architecture shares one reason, so a column of
# it is one sentence printed once per row. CREDA's footnotes are one line under
# each table; these are too, and `_bloque` is what renders them, so a renderer
# cannot put the reason back into a row without leaving the one builder.

#: The method, fixed before this table existed and taken from CREDA's own
#: declaration rather than chosen here. `CREDA.toy_plots` records that it was
#: fixed once on purpose: four different methods were run over one dataset in
#: response to successive objections, and the accumulated effect of that is
#: method-shopping even when each step answered something fair.
METODO_DECLARADO = {
    "omnibus": ("Friedman, un tratamiento por escalón de `ESCALERA` y un "
                "bloque por semilla de reporte"),
    "postHoc": ("Wilcoxon por pares, con Holm DENTRO de la familia de pares de "
                "ESTA familia y de esta arquitectura, nunca sobre los pares de "
                "las tres familias juntas"),
    "referencia": ("Demšar (2006), *Statistical Comparisons of Classifiers "
                   "over Multiple Data Sets*"),
}

#: The floor under both tests, in BLOCKS of report seed. Two and not a
#: tolerance chosen by eye: at one block each declared test returns a value
#: that does not depend on the data at all (see the banner above, where the
#: three constants are quoted with the inputs they were measured over). Two is
#: the first count at which the statistic moves when the numbers move.
BLOQUES_MINIMOS = 2

#: Below this range a set of paired differences has no dispersion to estimate
#: and what would be measured is floating-point cancellation. CREDA's own
#: `RANGO_MINIMO`, with its reason: an accuracy over this evaluation role is a
#: fraction with a handful of bags as its denominator, so the granularity of a
#: real difference is orders of magnitude above this, and a set of differences
#: identical to 1e-17 yields a `t` of 1e15 that prints like a statistic.
RANGO_MINIMO = 1e-12

#: Shapiro-Wilk's own floor, in the same blocks, and it is HIGHER than the one
#: above. Measured on this venv's scipy rather than read off a doctrine:
#: `shapiro` returns `nan` over one value and over two, and a real statistic
#: from three up. `nan` printed under a header that says `Shapiro p` reads as a
#: number, so the cell is a dash with the reason under the table. The normality
#: is what LICENSES the t rather than something assumed, so its absence is
#: reported and never filled in.
BLOQUES_MINIMOS_SHAPIRO = 3

#: The two rungs CREDA's `con CREDA` / `sin CREDA` pair is, HERE. `(con, sin)`,
#: so the first is the arm carrying the regularizer and the second is the same
#: arm with nothing on it: CREDA subtracts `acc_predict` -- its step-0 reading --
#: from `acc_te`, and the rung that reads this run at its step-0 checkpoint is
#: the floor of `ESCALERA`.
#:
#: **AdaBN is neither side of it, and that is a decision rather than an
#: omission.** It re-estimates the target's running statistics, so it is an
#: intervention of its own and not «the same arm without the term»; and it
#: exists only where the backbone keeps buffers, so it could not fill a column
#: of a table that carries one row per declared architecture. Its pairs are
#: table B's, under the Friedman family of three rungs.
#:
#: Derived from `ESCALERA` so a renamed rung moves both column titles and the
#: footnote with it, and never spelled out: two lists of the same two rungs are
#: two things that drift.
PAREJA_SIN_CON = (ESCALON_METODO, ESCALON_FASE_1)

#: The label of the row that averages, CREDA's own word for it. Its `clave` is
#: `None`, so nothing can confuse it with a declared architecture.
ROTULO_PROMEDIO = "PROMEDIO"

#: The axis that row collapses, said where the aggregation happens and in
#: CREDA's own words. An aggregation that does not name its axis produces
#: something with the right shape and another meaning: the normalization
#: averaged WITHIN each seed is one pair per seed, and the same values pooled
#: across cells and seeds are as many pairs as there are cells -- the same point
#: estimate over a denominator nobody measured.
EJE_PROMEDIO = "la normalización, DENTRO de cada semilla"


def pares_de_la_escalera() -> tuple[tuple[str, str], ...]:
    """Every ordered pair of rungs, the MORE intervened one named first.

    Derived from `ESCALERA` and never listed: a rung added to the ladder adds
    its pairs here without anybody remembering to. The order of a pair is FIXED
    and does not follow the winner -- CREDA's reason, kept: if it followed the
    winner, an always-positive sign would say nothing.

    Ordered least-intervened-pair first, which is CREDA's own
    `PARES_TRES_BRAZOS` order, so the two tables are read the same way.
    """
    nombres = [escalon for escalon, _, _ in ESCALERA]
    return tuple((nombres[i], nombres[j])
                 for i in range(1, len(nombres))
                 for j in range(i))


def _campo_de(escalon: str) -> str:
    return next(campo for nombre, _, campo in ESCALERA if nombre == escalon)


def _serie(registro: dict, familia: str, norma: str, escalon: str):
    """One rung's reading, one entry per report seed, in the record's order.

    Returns `None` where the rung has no reading at all in this architecture --
    which is AdaBN wherever the backbone holds nothing re-estimable -- rather
    than an empty list, because «this rung does not exist here» and «this rung
    came out empty» are different facts and a list cannot hold the difference.
    """
    campo = _campo_de(escalon)
    valores = [fila["celda"][campo] for fila in _por_celda(registro, familia)
               if _normalizacion(fila["clave"]) == norma]
    if any(valor is None for valor in valores):
        return None
    return [float(valor) for valor in valores]


def _holm(crudas: list) -> list:
    """Holm step-down over ONE family: `m*p` on the smallest, `m-1` on the next.

    Monotone by construction -- each adjusted p cannot fall below that of a
    smaller p -- and capped at 1. `m` is the number of pairs that PRODUCED a p,
    so a pair that could not be tested leaves the denominator instead of
    loosening the correction of the others. CREDA's own, with its reason: both
    counting it in `m` and inventing `p=1.0` for it assert something nobody
    measured.
    """
    vivas = [(i, float(p)) for i, p in enumerate(crudas) if p is not None]
    salida: list = [None] * len(crudas)
    if not vivas:
        return salida
    orden = sorted(range(len(vivas)), key=lambda k: vivas[k][1])
    corriendo = 0.0
    for i, k in enumerate(orden):
        j, p = vivas[k]
        corriendo = max(corriendo, (len(vivas) - i) * p)
        salida[j] = min(corriendo, 1.0)
    return salida


def _par_medido(primero: str, segundo: str, a: list, b: list) -> dict:
    """One pair already subtracted, with everything the table reports of it.

    `delta = a - b` and `a` is the rung named FIRST, so `delta > 0` means the
    first wins. CREDA's table had that inverted once -- the subtracted arm sat
    in the first column, so `+0.0819` beside `.predict` read as `.predict`
    winning when it means the opposite -- and a table whose sign reads backwards
    is worse than no table, because it is read with confidence.

    Below `BLOQUES_MINIMOS` blocks, or with no dispersion in the differences,
    the interval and the p are `None` with the reason in `razon`. They are not
    computed and then hidden: at one block they are constants of the block count
    (see the banner above), so computing them would produce a number that cannot
    come out different.

    **Three p-like fields and each one names its test.** `p` is Wilcoxon's --
    the one Holm corrects and the one table B prints -- and `pt` is the paired
    t's, which table A prints beside it so a reader can see whether the two
    agree. `shapiro` is what LICENSES the t rather than something assumed, and
    it has a floor of its own because scipy returns `nan` for it below three
    values. Which test produced a p is never left to the column title alone:
    this table had its sign read backwards off a printed column once, and a p
    whose test nobody can name is the same class of defect.
    """
    dif = [x - y for x, y in zip(a, b)]
    n = len(dif)
    media = statistics.fmean(dif)
    fila = {
        "primero": primero, "segundo": segundo, "n": n, "delta": media,
        "gana": sum(1 for d in dif if d > 0),
        "mediana": None, "ic": None, "p": None, "razon": None,
        "t": None, "pt": None, "shapiro": None, "gl": n - 1,
    }
    if n < BLOQUES_MINIMOS:
        fila["razon"] = (
            f"{n} bloque de semilla: con menos de {BLOQUES_MINIMOS} el "
            f"estadístico de Friedman y la p de Wilcoxon no dependen de los "
            f"números —son constantes del número de tratamientos y de bloques— "
            f"así que no se calculan. La diferencia pareada sí es exacta y está "
            f"en su columna. Hace falta "
            + (f"{BLOQUES_MINIMOS - n} bloque más como mínimo"
               if BLOQUES_MINIMOS - n == 1
               else f"{BLOQUES_MINIMOS - n} bloques más como mínimo")
            + ", y la escala completa declara los suyos")
        return fila
    fila["mediana"] = statistics.median(dif)
    if (max(dif) - min(dif)) <= RANGO_MINIMO:
        fila["razon"] = (
            "las diferencias pareadas no tienen dispersión que estimar: los "
            "dos escalones salieron iguales en cada bloque, así que no hay "
            "error estándar ni hay un test que hacer")
        return fila
    error = statistics.stdev(dif) / (n ** 0.5)
    critico = float(_st.t.ppf(0.975, n - 1))
    fila["ic"] = (media - critico * error, media + critico * error)
    fila["p"] = (None if all(d == 0.0 for d in dif)
                 else float(_st.wilcoxon(dif).pvalue))
    prueba = _st.ttest_rel(a, b)
    fila["t"] = float(prueba.statistic)
    fila["pt"] = float(prueba.pvalue)
    if n >= BLOQUES_MINIMOS_SHAPIRO:
        fila["shapiro"] = float(_st.shapiro(dif).pvalue)
    else:
        fila["razon"] = (
            f"Shapiro-Wilk no se calcula con {n} bloques de semilla: scipy "
            f"devuelve `nan` por debajo de {BLOQUES_MINIMOS_SHAPIRO} valores, y "
            f"un `nan` impreso donde va una p se lee como un número. El t y la "
            f"p de Wilcoxon sí están, y lo que falta es la normalidad que "
            f"LICENCIA al t, así que su columna va con raya en vez de llenarse")
    return fila


def _rangos_medios(tratamientos: dict, bloques: int) -> dict:
    """Friedman's mean ranks: the rungs of one seed ranked against each other.

    `scipy.stats.rankdata` per block and the mean over the blocks, which is
    CREDA's `rankdata(pila, axis=0).mean(axis=1)` without the array: this module
    may not import numpy -- the plot notebook's claim that it cannot train is a
    property of this module's import closure -- so the reduction over the blocks
    is `statistics.fmean` and the ranks themselves come back from scipy as
    floats.

    Ties are averaged, which is `rankdata`'s default and the right one here: two
    rungs that read the same accuracy in one seed did not beat each other, and
    breaking the tie by position would hand the win to whichever rung the ladder
    happens to declare first.
    """
    etiquetas = list(tratamientos)
    por_bloque = [[float(rango) for rango
                   in _st.rankdata([tratamientos[nombre][bloque]
                                    for nombre in etiquetas])]
                  for bloque in range(bloques)]
    return {nombre: statistics.fmean([fila[indice] for fila in por_bloque])
            for indice, nombre in enumerate(etiquetas)}


def _llano(escalon: str) -> str:
    """One rung's name with the backticks `ESCALERA` declares it with removed.

    The floor's short name IS `` `.predict` `` -- the markers are part of the
    declaration, so every cell that prints a rung prints it as code. Inside a
    code span they would nest, and a nested pair of backticks renders as a
    literal backtick beside a broken span: measured on the first rendering of
    this footnote, which came out as ``` `Δ = MIL-CREDA − `.predict`` ```.
    """
    return escalon.strip("`")


def _pares(cuantos: int) -> str:
    """`n par` or `n pares`, in one place.

    The footnote states two of these counts in one sentence -- what the average
    ran over and what pooling would have claimed -- and a hard-coded plural
    reads as «1 pares» at exactly the scale this document is rendered at.
    """
    return f"{cuantos} par" if cuantos == 1 else f"{cuantos} pares"


def _cuantos_bloques(registro: dict) -> str:
    """How many blocks of report seed this record carries, said in one place.

    One implementation because two places state it -- table A's own header
    (`encabezado_tests`) and the Friedman line above each grid of table B --
    and two spellings of one integer are two things that can disagree about it.
    The count is the scale's `report_seeds`, so it is read and never written.
    """
    bloques = len(_escala(registro)["report_seeds"])
    return (f"{bloques} bloque de semilla" if bloques == 1
            else f"{bloques} bloques de semilla")


def comparar_familia(registro: dict, familia: str) -> dict:
    """The declared tests of ONE toyset, per architecture, numbers only.

    Split from the renderer the way CREDA splits `comparar_*` from `tabla_*`:
    the numbers are asked for by this module's tests as well as by the table,
    and two implementations of one test diverge.

    **The pairing is REAL and not an assumption.** The three rungs of one row
    come out of ONE cell, so they share the seed, the material and phase 1 bit
    for bit -- `toy_run.run_cell` reads `.predict` off the step-0 checkpoint of
    the same run whose phase 2 produces `targetAccuracy` -- which is why a pair
    can be subtracted at all. That phase 1 did not move between them is MEASURED
    (`sourceMoved`, `sourceBuffersMoved`) and is no longer STATED anywhere in
    the document: the per-row columns that carried those two left the table when
    it took CREDA's eight columns, and the count that replaced them left with
    the conclusions. The pairing still rests on the measurement; what a reader
    no longer meets is the assertion that it held in this run.

    Friedman is per (family, architecture) and never pooled across
    architectures: the three rows train different encoders, so one omnibus over
    all of them would be counting three experiments as one.
    """
    _exigir_familia(registro, familia)
    bloques = len(_escala(registro)["report_seeds"])
    salida = []
    for norma in normalizaciones(registro):
        series = {escalon: _serie(registro, familia, norma, escalon)
                  for escalon, _, _ in ESCALERA}
        ausentes_aqui = [nombre for nombre, serie in series.items()
                         if serie is None]
        filas = []
        for primero, segundo in pares_de_la_escalera():
            if series[primero] is None or series[segundo] is None:
                sin = [n for n in (primero, segundo) if series[n] is None]
                filas.append({
                    "primero": primero, "segundo": segundo, "n": bloques,
                    "delta": None, "gana": None, "mediana": None, "ic": None,
                    "p": None,
                    "razon": (f"{', '.join(sin)} no tiene escalón en "
                              f"`{norma}`: no hay segunda serie que restar, "
                              "así que el par está declarado y no se testea"),
                })
                continue
            filas.append(_par_medido(primero, segundo,
                                     series[primero], series[segundo]))
        for fila, ajustada in zip(filas, _holm([f["p"] for f in filas])):
            fila["holm"] = ajustada
        tratamientos = {nombre: serie for nombre, serie in series.items()
                        if serie is not None}
        omnibus = None
        if len(tratamientos) >= 3 and bloques >= BLOQUES_MINIMOS:
            chi, p = _st.friedmanchisquare(*tratamientos.values())
            omnibus = {"chi2": float(chi), "gl": len(tratamientos) - 1,
                       "p": float(p), "tratamientos": list(tratamientos),
                       # CREDA's mean ranks, and they travel with the omnibus
                       # because they are what it is READ with: a chi-square
                       # says the rungs are not interchangeable and says nothing
                       # about which one is ahead. Ranked WITHIN each block --
                       # the rungs of one seed against each other -- and averaged
                       # over the blocks; ranking the other axis would order the
                       # seeds, which is not the question. The high rank is the
                       # best rung, so `mejorRango` is the treatment count.
                       "rangos": _rangos_medios(tratamientos, bloques),
                       "mejorRango": len(tratamientos)}
        salida.append({
            "normalizacion": norma,
            "bloques": bloques,
            "escalonesAusentes": ausentes_aqui,
            "tratamientos": list(tratamientos),
            "omnibus": omnibus,
            "familiaDeHolm": sum(1 for f in filas if f["p"] is not None),
            "filas": filas,
        })
    return {"familia": familia, "bloques": bloques, "porArquitectura": salida}


def _dentro_de_la_semilla(series: dict, normas: list[str], cual: int,
                          bloques: int) -> list[float]:
    """One value per SEED: the architectures averaged inside each block.

    `cual` picks the rung -- 0 is the one named first in `PAREJA_SIN_CON` -- and
    what comes back is as long as there are report seeds, never as long as there
    are cells. That is the whole of `PROMEDIO`: seed 0 appears in every
    architecture, so the cell-seed pairs are not independent observations and
    pooling them is pseudo-replication.

    The point estimate is the SAME either way, which is what makes this quiet:
    the mean of the averaged differences equals the mean of all of them, so
    pooling moves no number in the delta column and multiplies the statistic
    under it. The table keeps its shape and its `t`, its interval and its win
    count start describing a design nobody ran.
    """
    return [statistics.fmean([series[norma][cual][bloque] for norma in normas])
            for bloque in range(bloques)]


def comparar_familia_sin_contra_con(registro: dict, familia: str) -> dict:
    """CREDA's table A for ONE toyset, numbers only: the two rungs of
    `PAREJA_SIN_CON` per architecture, plus the row that averages.

    Two arms, so the paired t -- with Shapiro-Wilk over the paired differences
    beside it, because normality is what licenses the t rather than something
    assumed -- and Wilcoxon reported as well, so a reader sees whether the two
    agree. No Friedman: Friedman is for three or more, and the three rungs are
    table B.

    **The pairing is real and not an assumption.** Both rungs of one row come
    out of the same cells, in the record's own seed order, so the positional
    index IS the seed and a pair can be subtracted at all. That the two series
    are as long as there are report seeds is CHECKED rather than trusted: with a
    second declared arm the cell walk would return two values per seed and the
    averaging would silently fold the arms into the seeds, which is an
    aggregation collapsing two axes while naming one.

    Refuses an architecture missing either rung. Both exist in every
    architecture by construction -- the floor is this run read at its step-0
    checkpoint and the method is its phase 2 -- so a missing one means the
    record is not what it says it is, and a table one row short says so nowhere.
    """
    _exigir_familia(registro, familia)
    con_nombre, sin_nombre = PAREJA_SIN_CON
    normas = normalizaciones(registro)
    bloques = len(_escala(registro)["report_seeds"])
    series = {}
    for norma in normas:
        lecturas = (_serie(registro, familia, norma, con_nombre),
                    _serie(registro, familia, norma, sin_nombre))
        for nombre, serie in zip((con_nombre, sin_nombre), lecturas):
            if serie is None:
                raise ValueError(
                    f"{familia}: `{norma}` no trae el escalón {nombre}, y los "
                    f"dos escalones de este par existen en toda arquitectura "
                    f"—el piso es esta misma corrida leída en el paso 0 y el "
                    f"método es su fase 2—, así que una fila menos no es una "
                    f"ausencia del diseño sino del registro")
            if len(serie) != bloques:
                raise ValueError(
                    f"{familia}: `{norma}` trae {len(serie)} lecturas de "
                    f"{nombre} y la escala declara {bloques} semillas de "
                    f"reporte. Con menos, el registro declara una arquitectura "
                    f"de la que no trae celdas y una tabla a la que le falta "
                    f"una fila no lo dice en ninguna parte; con más, el "
                    f"recorrido trae más de una lectura por semilla —los brazos "
                    f"declarados son {_banderas(registro)['arms']}— y "
                    f"promediarlas DENTRO de la semilla colapsaría dos ejes "
                    f"nombrando uno")
        series[norma] = lecturas

    def _fila(rotulo: str, clave: str | None, con: list, sin: list) -> dict:
        fila = _par_medido(con_nombre, sin_nombre, con, sin)
        fila["rotulo"] = rotulo
        fila["clave"] = clave
        fila["con"] = statistics.fmean(con)
        fila["sin"] = statistics.fmean(sin)
        return fila

    filas = [_fila(_rotulo_norma(norma), norma, *series[norma])
             for norma in normas]
    filas.append(_fila(
        ROTULO_PROMEDIO, None,
        _dentro_de_la_semilla(series, normas, 0, bloques),
        _dentro_de_la_semilla(series, normas, 1, bloques)))
    return {
        "familia": familia,
        "con": con_nombre,
        "sin": sin_nombre,
        "bloques": bloques,
        "ejeColapsado": EJE_PROMEDIO,
        # The two denominators, both computed: what `PROMEDIO` ran over and what
        # pooling would have claimed. The footnote prints the pair because the
        # difference between them is the whole decision, and a reader cannot see
        # it in a table that looks right either way.
        "paresDelPromedio": bloques,
        "paresDeCeldaSemilla": bloques * len(normas),
        "filas": filas,
    }


def encabezado_tests(registro: dict, familia: str) -> str:
    """The header above table A of the declared tests: CREDA's own line, derived.

    `**<familia> · <n> bloque(s) de semilla · <rol>, sólo reporte**`. The role
    is read off the record's own split and the count off the declared report
    seeds, so a renamed role or a shorter run moves the header rather than
    leaving it naming something nobody measured.

    **Table A only, and it used to head table B as well.** CREDA's own second
    tests block carries ONE title and it is that grid's, naming the
    architecture; a family header above it was a second bold title the
    reference document does not have, and the operator retired it. The module
    docstring records the ruling, and `_linea_friedman` carries the block count
    that header was stating for table B.

    It carries no chance level, unlike `encabezado_familia`: these tables report
    DIFFERENCES, and the landmark of a difference is zero rather than one over
    the classes -- which the framing's computed half states for the readings it
    belongs to.
    """
    _exigir_familia(registro, familia)
    return (f"**{familia} · {_cuantos_bloques(registro)} · "
            f"{rol_publicado(registro)}, sólo reporte**")


def _motivos(filas: list[dict]) -> list[str]:
    """Every DISTINCT reason among a table's rows, in the order they appear.

    The reason a column is dashed belongs to the table and not to the row: the
    rows of one architecture share one of them, so a column of it prints the
    same paragraph as many times as the table has rows. Distinct and ordered
    rather than sorted, so a table whose rows were dashed for two different
    reasons states both, once each, in the order a reader meets them.
    """
    vistos = []
    for fila in filas:
        if fila.get("razon") and fila["razon"] not in vistos:
            vistos.append(fila["razon"])
    return vistos


def render_familia_sin_contra_con(registro: dict, familia: str) -> str:
    """CREDA's table A for ONE toyset: the method's rung against the floor's,
    one row per declared architecture plus the row that averages.

    CREDA's ten columns, in CREDA's order: the architecture, the two levels, the
    difference, its interval, the t with its degrees of freedom, the t's p,
    Shapiro's p over the paired differences, Wilcoxon's p beside it, and how
    many blocks the first rung won.

    **The two level columns are named for the rungs they hold.** CREDA's words
    are `con CREDA` and `sin CREDA`; printing them over these two quantities
    would relabel somebody else's table onto a different pair, so what the
    titles say is which rung of `ESCALERA` each column is -- and the footnote
    says why that pair is the one CREDA's pair maps to here.

    The `con` column goes FIRST, as CREDA has it: with the subtracted rung in
    front, a reader walking the row left to right reads the difference backwards
    from the way it is defined.
    """
    res = comparar_familia_sin_contra_con(registro, familia)
    filas = res["filas"]
    con, sin = res["con"], res["sin"]

    def _opcional(clave: str, cifras: int = 4, signo: bool = False):
        def valor(r):
            if r[clave] is None:
                return NADA
            return (f"{r[clave]:+.{cifras}f}" if signo
                    else f"{r[clave]:.{cifras}f}")
        return valor

    columnas = [
        Columna("normalización", lambda r: r["rotulo"]),
        # The LEVEL of each rung, reduced over the report seeds and printed
        # without a dispersion, which is CREDA's own shape for this table: the
        # dispersion of the DIFFERENCE is what the interval beside it carries,
        # and the two levels are there so the sign of that difference can be
        # checked against the two numbers it was taken between.
        Columna(con, lambda r: _num(r["con"]), (_campo_de(con),)),
        Columna(sin, lambda r: _num(r["sin"]), (_campo_de(sin),)),
        Columna(f"Δ = {_llano(con)} − {_llano(sin)}",
                _opcional("delta", signo=True),
                (_campo_de(con), _campo_de(sin))),
        Columna("IC95 de Δ",
                lambda r: (NADA if r["ic"] is None
                           else f"[{r['ic'][0]:+.4f}, {r['ic'][1]:+.4f}]")),
        Columna(f"t({filas[0]['gl']})", _opcional("t", cifras=2)),
        Columna("p", _opcional("pt", cifras=5)),
        Columna("Shapiro p", _opcional("shapiro", cifras=3)),
        Columna("Wilcoxon p", _opcional("p")),
        Columna("gana", lambda r: f"{r['gana']}/{r['n']}"),
    ]
    pie = [
        f"`Δ = {_llano(con)} − {_llano(sin)}`, así que **Δ>0 quiere decir que "
        f"gana `{_llano(con)}`**",
        (f"es el par que CREDA llama `con CREDA` contra `sin CREDA`: "
         f"`{_llano(sin)}` es esta misma corrida leída en su punto de control "
         f"del paso 0, así que las dos columnas comparten semilla, material y "
         f"fase 1 bit a bit. AdaBN no es ninguno de los dos lados —reestima las "
         f"estadísticas corridas del blanco, que es otra intervención, y sólo "
         f"existe donde el backbone guarda algo— y sus pares están en la tabla "
         f"de abajo"),
        (f"`{ROTULO_PROMEDIO}` promedia {res['ejeColapsado']}, o sea "
         f"{_pares(res['paresDelPromedio'])}, y no agrupa los "
         f"{_pares(res['paresDeCeldaSemilla'])} de celda-semilla: la primera "
         f"semilla de reporte está en las {len(normalizaciones(registro))} "
         f"normalizaciones, así que esos pares no son observaciones "
         f"independientes"),
        "t pareado, Shapiro-Wilk sobre las diferencias pareadas, Wilcoxon al lado",
    ]
    motivos = _motivos(filas)
    if motivos:
        pie.append("las columnas con raya, y por qué: " + "; ".join(motivos))
    return (encabezado_tests(registro, familia) + "\n\n"
            + _bloque([(None, columnas, filas, " · ".join(pie))]))


def render_familia_tests(registro: dict, familia: str) -> str:
    """CREDA's table B for ONE toyset: the three rungs of the architecture where
    all three exist, one row per declared pair.

    CREDA's seven columns, in CREDA's order: who wins if the difference is
    positive, against whom, the difference, its interval, Wilcoxon's p, the p
    Holm adjusted inside this family of pairs, and how many blocks the first
    rung won. No architecture column -- the architecture is in the title, as
    CREDA has it, because one table is one architecture -- and no median column,
    which CREDA prints in neither table.

    **Which architecture it is comes from the measurement.** AdaBN has a rung
    where the run FOUND buffers to re-estimate, so this renders one table per
    architecture `arquitecturas_con_escalon_adabn` returns and a single sentence
    instead of a table when it returns none. CREDA renders BatchNorm for exactly
    that reason and never because of the layer's name: deciding by name would
    give a rung to the architecture that normalizes and keeps nothing, and the
    number there would be the floor under another name.

    **The Friedman line sits above its own table and not in a column.** It is
    one reading per architecture, so a column of it would print one number once
    per row; CREDA puts it above the table, with the mean ranks that say which
    rung is ahead, and that is where it is.

    The sign is stated in the FIRST column's own title and not only in a
    docstring: where it was read backwards in this lineage, it was read off the
    printed table. The pair order is fixed and does not follow the winner -- if
    it followed the winner, an always-positive sign would say nothing.
    """
    comparacion = comparar_familia(registro, familia)
    con_adabn = arquitecturas_con_escalon_adabn(registro, familia)
    declarados = len(pares_de_la_escalera())
    # What Holm is NOT corrected over, computed: every declared pair of every
    # architecture that has the three rungs, in every family. CREDA writes «no
    # sobre los 9»; nine is three pairs by three toysets, and a literal nine
    # would go on naming the old grid the day a family or a rung arrived.
    de_todas = declarados * sum(
        len(arquitecturas_con_escalon_adabn(registro, nombre))
        for nombre in familias(registro))

    def _opcional(clave: str, cifras: int = 4, signo: bool = False):
        def valor(r):
            if r[clave] is None:
                return NADA
            return (f"{r[clave]:+.{cifras}f}" if signo
                    else f"{r[clave]:.{cifras}f}")
        return valor

    grupos = []
    for bloque in comparacion["porArquitectura"]:
        if bloque["normalizacion"] not in con_adabn:
            continue
        columnas = [
            Columna("gana si Δ>0", lambda r: r["primero"]),
            Columna("contra", lambda r: r["segundo"]),
            Columna("Δ = 1ª − 2ª", _opcional("delta", signo=True),
                    tuple(campo for _, _, campo in ESCALERA)),
            Columna("IC95 de Δ",
                    lambda r: (NADA if r["ic"] is None
                               else f"[{r['ic'][0]:+.4f}, {r['ic'][1]:+.4f}]")),
            Columna("p (Wilcoxon)", _opcional("p")),
            Columna(f"p Holm (familia de {bloque['familiaDeHolm']})",
                    _opcional("holm")),
            Columna("gana",
                    lambda r: (NADA if r["gana"] is None
                               else f"{r['gana']}/{r['n']}")),
        ]
        pie = [
            "`Δ = (columna «gana si Δ>0») − (columna «contra»)`, así que "
            "**Δ>0 quiere decir que gana el PRIMERO de los dos nombrados**",
            "el orden del par es fijo y no sigue al ganador",
            (f"Holm dentro de los {bloque['familiaDeHolm']} pares de esta "
             f"familia y esta arquitectura que produjeron una p, de los "
             f"{declarados} declarados, y nunca sobre los {de_todas} de las "
             f"{len(familias(registro))} familias juntas, que es una familia "
             f"que nadie declaró y corregiría de más"),
        ]
        motivos = _motivos(bloque["filas"])
        if motivos:
            pie.append("las columnas con raya, y por qué: " + "; ".join(motivos))
        grupos.append((
            # The architecture, and the ONE bold title of this block: CREDA's
            # own second tests table carries a single title and it is the
            # grid's. The family header that used to sit above the whole block
            # was the second one, and the operator retired it (the module
            # docstring records the ruling), so what the title says is which
            # architecture this grid is -- and the block count it took with it
            # is back in the Friedman line below, which is where CREDA prints
            # it. A block with two qualifying architectures therefore carries
            # two titles, one per grid, exactly as it carries two omnibuses.
            f"**{_rotulo_norma(bloque['normalizacion'])} · los "
            f"{len(bloque['tratamientos'])} escalones**\n\n"
            + _linea_friedman(bloque, registro),
            columnas, bloque["filas"], " · ".join(pie)))

    if not grupos:
        # One sentence and no table, which is CREDA's shape for an architecture
        # that has no AdaBN rung carried up one level: with none of them
        # carrying buffers there are no three rungs to order anywhere in this
        # family, and a table of dashes would say the three rungs were compared
        # and came out empty. Through `_bloque` all the same -- it is the one
        # builder and a renderer that skipped it would be a second one -- and
        # with no groups it adds nothing to the sentence.
        normas = normalizaciones(registro)
        return (
            f"**{familia}:** ninguna de las {len(normas)} arquitecturas "
            f"declaradas guarda estadísticas corridas que reestimar —medido "
            f"celda por celda, nunca por el nombre de la capa—, así que AdaBN "
            f"no tiene escalón en ninguna y no hay tres escalones que ordenar: "
            f"esta subsección no trae la tabla de los pares, y lo que sí se "
            f"puede restar es el par de la tabla de arriba."
            + _bloque([]))
    return _bloque(grupos)


def _linea_friedman(bloque: dict, registro: dict) -> str:
    """The line above table B: CREDA's omnibus, or the fact that there is none.

    **The block count is here, because CREDA prints it here.** Its own line
    reads `Friedman χ²(2) = 7.80 · p = 0.02024 · 10 bloques de semilla ·
    rangos medios ...`, and over how many blocks the statistic was taken is
    what licenses reading it at all. It used to be left out of this line: the
    family header above the whole block carried it, so printing it per
    architecture would have rendered one fact of the run once per grid. The
    operator retired that header -- CREDA's second tests block carries one
    title and it is the grid's -- and the count came back to the line CREDA
    keeps it in, rather than leaving the block with no statement of how many
    blocks its numbers reduce over.

    What else is here is of this grid alone: the statistic over its own three
    rungs, and the mean ranks that say which of them is ahead, which a
    chi-square on its own does not.

    The no-omnibus branch carries NO count, and that is not an omission: below
    the floor the footnote states the count beside the floor it names (`1 bloque
    de semilla: con menos de 2 ...`), and stating it here as well would print
    one number twice inside one block.
    """
    omnibus = bloque["omnibus"]
    if omnibus is None:
        return ("Sin lectura de Friedman: el pie de la tabla dice con cuántos "
                "bloques de semilla cada columna se puede calcular.")
    rangos = " | ".join(f"`{_llano(brazo)}` {valor:.2f}"
                        for brazo, valor in omnibus["rangos"].items())
    return (f"Friedman χ²({omnibus['gl']}) = {omnibus['chi2']:.2f} · p = "
            f"{omnibus['p']:.5f} · {_cuantos_bloques(registro)} · rangos "
            f"medios ({omnibus['mejorRango']} = mejor): {rangos}")



#: Qué valor busca cada lectura, calculado del registro. Una dirección no es un
#: hito: «más alto es mejor» dice para dónde mirar y nada sobre dónde está lo
#: bueno, y el hito es un número, así que se calcula acá y no se escribe en la
#: prosa de ninguna celda.
def objetivo(registro: dict) -> dict:
    """For each reading, which way it wins and the landmark it is read against."""
    res = resolucion(registro)
    return {
        "acierto": {
            "direccion": "más alto es mejor",
            "hito": _pct(azar(registro)),
            "porque": (f"con {_escala(registro)['classes']} clases el azar es "
                       f"una entre ellas, y por debajo de ese valor no hay "
                       f"representación que transferir"),
        },
        "caída valid → test": {
            "direccion": "más cerca de cero es mejor",
            "hito": "cero",
            "porque": (f"la fase 1 se entrena sobre "
                       f"{res['bolsasEntrenamiento']} bolsas, elige su mejor "
                       f"época sobre {res['bolsasValidacion']} y se lee después "
                       f"sobre {res['bolsasEvaluacion']}: una caída es la "
                       f"búsqueda sobreajustando el rol que elige, no el método "
                       f"fallando"),
        },
        "razón del sopesado": {
            "direccion": "más cerca de uno es mejor",
            "hito": _num(1.0),
            "porque": ("es `global(con sopesado) / global(sin sopesado)`: uno "
                       "significa que encender el sopesado no le costó nada al "
                       "término global, y con el coeficiente fijo en uno no hay "
                       "ningún lambda que absorba la diferencia"),
        },
        "fuente movida": {
            "direccion": "cero y nada más",
            "hito": _num(0.0, 6),
            "porque": ("la fase 1 está congelada o no lo está; cualquier valor "
                       "distinto de cero dice que la fuente se movió en la fase 2"),
        },
        "resolución": {
            "direccion": "más fina es mejor",
            "hito": f"{res['puntosPorcentualesPorBolsa']:.1f} pp",
            "porque": (f"el rol de evaluación tiene {res['bolsasEvaluacion']} "
                       f"bolsas, así que el acierto de una celda toma uno de "
                       f"{res['valoresPosibles']} valores y dos celdas no pueden "
                       f"diferir en menos de una bolsa"),
        },
    }





def _declaradas(prefijo: str) -> dict:
    """Every module-level callable of this module whose name starts with `prefijo`.

    Derived and never listed. A roster written by hand goes quiet on the one
    added after it was written, which is the one most likely to build its grid
    outside `_bloque` and so to spell a cell some other way than CREDA does.
    """
    return {nombre: valor for nombre, valor in sorted(globals().items())
            if nombre.startswith(prefijo) and callable(valor)}


def _expandido(declaradas: dict, registro: dict) -> dict:
    """Each declared call run over the record, once per family where it takes one.

    The family argument is read off the SIGNATURE and not off the name: a
    renderer that works per family says so by taking `familia`, so a caller
    walking this roster does not have to know which half it is in.
    Keyed `nombre[familia]` for the per-family half and `nombre` for the rest,
    so a failure names the family it happened in.
    """
    salida = {}
    for nombre, llamada in declaradas.items():
        if "familia" in inspect.signature(llamada).parameters:
            for familia in familias(registro):
                salida[f"{nombre}[{familia}]"] = llamada(registro, familia)
        else:
            salida[nombre] = llamada(registro)
    return salida


def renderizadores() -> dict:
    """Every table-building call in this module, derived and never listed."""
    return _declaradas("render_")


def renderizados(registro: dict) -> dict:
    """Every table this module can build over one record: label -> markdown."""
    return _expandido(renderizadores(), registro)
