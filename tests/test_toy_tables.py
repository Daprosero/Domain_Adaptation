"""What the plot notebook's tables have to be true of, pinned.

Every test here is about the DOCUMENT rather than the mathematics: that the
tables come from the artefact on disk instead of being recomputed, that each
one is about ONE toyset, that what each heading SHOWS is what CREDA's own plot
notebook shows under the same heading, and that each block is SPELLED the way
CREDA spells one -- no footer, one bold title per grid, CREDA's cell padding.

Two claims this docstring used to make are gone with the mechanisms they were
about, and both were retired by the operator rather than weakened here: that
phase 1's two readings cannot be separated (the grid that carried the second
one is retired) and that nothing renders without the stamp saying these
numbers are a trial (no table carries that stamp any more).  The module's own
docstring records both rulings. A table is evidence or it is decoration, and the difference is not
visible in the output.

The record every test reads is the REAL one on disk, written by
`Toyset_Generator.ipynb` through `toy_run.guardar`. A fixture written here would
verify these renderers against this file's idea of the artefact and never
against the artefact, which is the one join that matters.

**Nothing else in this repository can catch what these tests catch.**
`report.renderers` in `src/MIL_CREDA_Benchmark/__init__.py` names only the
`tables`, `figures` and `latent` modules of the digit benchmark, plus
matplotlib's own show, and the verification engine skips a cell it finds no
declared rendering in -- so `unframed`, `unaimed`,
`unconcluded`, `duplicated`, `undeclared`, `statedNotShown`, `unrendered` and
`restated` cannot fire on a `toy_tables.*` cell at all. A clean `report` says
nothing about this notebook.
"""

from __future__ import annotations

import copy
import inspect
import json
import re
import statistics
from pathlib import Path

import pytest

from CREDA.toy_experiment import ASPECTO_MAX
from MIL_CREDA_Benchmark import toy_run, toy_tables

#: El repositorio, que es donde las rutas que el cuaderno imprime se resuelven.
RAIZ = Path(toy_run.__file__).resolve().parents[2]

CUADERNO = RAIZ / "MIL-CREDA" / "Notebooks" / "Toyset_Plot.ipynb"

#: Todo lo de `toy_run` que entrena, busca, barre o escribe. El cuaderno de
#: gráficas no llama ninguna: lee. Derivado de una condición y no de una lista
#: elegida sería mejor, pero «qué función entrena» no es una propiedad que el
#: módulo declare, así que se nombran y el test de abajo dice por qué.
NADA_DE_ESTO_CORRE = ("sweep", "run_cell", "search_phase_one", "search_phase_two",
                      "train_phase_one", "adapt_phase_two", "guardar",
                      "build_family", "build_arm", "weighting_cost")

#: Una medición en un texto: lo que un lector lee como un número medido.
MEDICION = re.compile(r"-?\d+\.\d+")

#: Los números que los dos títulos contados del cuaderno tienen que decir, para
#: que la cuenta salga del registro y no de lo que alguien escribió una vez. El
#: defecto que esto cierra es exacto y ya pasó: un título que decía «las tres
#: arquitecturas» sobre una grilla de dos.
NUMERO_EN_PALABRAS = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco",
                      6: "seis", 7: "siete", 8: "ocho"}


@pytest.fixture(scope="module")
def artefacto() -> Path:
    destino = toy_run.destination(toy_run.PILOT)
    if not (destino / "barrido.json").exists():
        pytest.skip(f"no hay artefacto en {destino}: "
                    "corré MIL-CREDA/Notebooks/Toyset_Generator.ipynb")
    return destino


@pytest.fixture(scope="module")
def registro(artefacto) -> dict:
    return toy_run.cargar(artefacto)


@pytest.fixture(scope="module")
def cuaderno() -> dict:
    if not CUADERNO.exists():
        pytest.skip(f"no está {CUADERNO}")
    return json.loads(CUADERNO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cuaderno_de_creda() -> dict:
    """CREDA's own executed plot notebook: the shape this one has to have.

    Read rather than described. The operator pointed at a document and asked
    for ITS form, so the form is taken from the document; a list of counts
    typed into this file would be a second copy of it, and the day the two
    disagreed nothing would say which one is the reference.
    """
    if not CUADERNO_DE_CREDA.exists():
        pytest.skip(f"no está {CUADERNO_DE_CREDA}")
    return json.loads(CUADERNO_DE_CREDA.read_text(encoding="utf-8"))


def _codigo(cuaderno: dict) -> str:
    return "\n".join("".join(celda["source"]) for celda in cuaderno["cells"]
                     if celda["cell_type"] == "code")


def _prosa(cuaderno: dict) -> str:
    return "\n".join("".join(celda["source"]) for celda in cuaderno["cells"]
                     if celda["cell_type"] == "markdown")


#: CREDA's own executed plot notebook. The shape this one has to have is READ
#: off it rather than written here: a literal list of per-heading output counts
#: in this file would be one more sentence that ages, and the drift it has to
#: catch is exactly the drift a sentence does not notice.
CUADERNO_DE_CREDA = RAIZ / "CREDA" / "Notebooks" / "Toyset_Plot.ipynb"

#: Un encabezado, en una celda de markdown o EMITIDO por una celda de codigo.
#: Las dos cosas, porque el documento las renderiza igual: el cuaderno de CREDA
#: abre `## El paso 4, leido por dentro` desde un `display(Markdown(...))`, y un
#: lector que solo mirara las celdas de markdown se lo atribuiria a la
#: subseccion anterior --- que es como se lee que sus tres bloques de familia no
#: son iguales cuando si lo son.
ENCABEZADO = re.compile(r"^(#{1,6})[ \t]+(\S.*)$")


def _mime(salida: dict) -> str:
    """What a reader MEETS in one output, in one word.

    `text/plain` is not counted when a richer bundle carries it: IPython ships
    the repr of a `Markdown` or an `Image` object inside the same output as the
    rendered thing, so counting it would count every table and every figure
    twice and the count would stop meaning «how many things does this cell
    show».
    """
    if salida["output_type"] == "stream":
        return "stream"
    if salida["output_type"] == "error":
        return "error"
    datos = salida.get("data") or {}
    for clave in datos:
        if clave != "text/plain":
            return clave
    return "text/plain"


def _firma_por_encabezado(cuaderno: dict) -> list[dict]:
    """One entry per heading, in reading order: level, title, mimes, cells, texts.

    The reading this whole section rests on, and it is a reading of what the
    document SHOWS rather than of what its cells call: a cell that stopped
    emitting, or grew a second output, is invisible to every test that reads
    source. `mimes` is every output under that heading in order; `celdas` is
    the index of every code cell under it, so a count can be asked per heading
    and per CELL, which are different questions. `textos` is the markdown each
    of those outputs emitted, in the same order, for the three counts that are
    about the TEXT of a block rather than about how many of them there are.
    """
    firma: list[dict] = [{"nivel": 0, "titulo": "(preamble)",
                          "mimes": [], "celdas": [], "textos": []}]

    def abrir(nivel: int, titulo: str) -> None:
        firma.append({"nivel": nivel, "titulo": titulo,
                      "mimes": [], "celdas": [], "textos": []})

    for indice, celda in enumerate(cuaderno["cells"]):
        if celda["cell_type"] == "markdown":
            lineas = "".join(celda["source"]).strip().splitlines()
            encontrado = ENCABEZADO.match(lineas[0]) if lineas else None
            if encontrado:
                abrir(len(encontrado.group(1)), encontrado.group(2).strip())
            continue
        if celda["cell_type"] != "code":
            continue
        firma[-1]["celdas"].append(indice)
        for salida in celda.get("outputs", []):
            mime = _mime(salida)
            if mime == "text/markdown":
                texto = "".join(salida["data"]["text/markdown"]).lstrip()
                lineas = texto.splitlines()
                encontrado = ENCABEZADO.match(lineas[0]) if lineas else None
                if encontrado:
                    abrir(len(encontrado.group(1)), encontrado.group(2).strip())
                    continue
                firma[-1]["textos"].append(texto)
            firma[-1]["mimes"].append(mime)
    return firma


def _bloques_de_familia(firma: list[dict],
                        familias: list[str]) -> dict[str, dict]:
    """familia -> {suffix: mimes}, where `''` is the family heading itself.

    Positional and not by title prefix: CREDA's weighting section carries a
    `### <familia> — qué hace el peso, por construcción` subsection that a
    prefix match would pull into the block, and it is not part of it. A block
    runs from its own `## <familia>` to the next heading that is not inside it.
    """
    bloques: dict[str, dict] = {}
    actual = None
    for entrada in firma:
        titulo, nivel = entrada["titulo"], entrada["nivel"]
        if nivel <= 2:
            actual = titulo if titulo in familias else None
            if actual is not None:
                bloques[actual] = {"": entrada["mimes"]}
        elif actual is not None:
            sufijo = (titulo[len(actual):].strip()
                      if titulo.startswith(actual) else titulo)
            bloques[actual][sufijo] = entrada["mimes"]
    return bloques


def _plantilla_de_bloque(cuaderno: dict, familias: list[str]) -> dict:
    """The one block shape a notebook's family blocks agree on.

    The agreement is MEASURED and not assumed: a template taken from the first
    block would be one block's accident, and the document would be held to it.
    """
    bloques = _bloques_de_familia(_firma_por_encabezado(cuaderno), familias)
    assert sorted(bloques) == sorted(familias), (
        f"los bloques de familia del cuaderno son {sorted(bloques)} y las "
        f"familias declaradas son {sorted(familias)}")
    formas = {familia: tuple(sorted((sufijo, tuple(mimes))
                                    for sufijo, mimes in bloque.items()))
              for familia, bloque in bloques.items()}
    assert len(set(formas.values())) == 1, (
        "los bloques de familia de este cuaderno no tienen todos la misma "
        f"forma: {formas}")
    plantilla = bloques[familias[0]]
    assert len(plantilla) > 1 and sum(len(m) for m in plantilla.values()) > 0, (
        f"la plantilla salió vacía ({plantilla}): compararse contra ella no "
        "mediría nada")
    return plantilla


def _sufijo_con(plantilla: dict, mimes: tuple, que: str) -> str:
    """The one subsection of the template whose signature is exactly `mimes`."""
    hallados = [sufijo for sufijo, salidas in plantilla.items()
                if tuple(salidas) == mimes]
    assert len(hallados) == 1, (
        f"{len(hallados)} subsecciones del bloque de CREDA emiten {mimes}, así "
        f"que «{que}» no se puede localizar sin escribir su título acá: "
        f"{ {s: tuple(m) for s, m in plantilla.items()} }")
    return hallados[0]


def _celda_unica_bajo(cuaderno: dict, titulo: str) -> dict:
    """The one code cell under a heading, refusing where there is not exactly one."""
    bajo = [entrada for entrada in _firma_por_encabezado(cuaderno)
            if entrada["titulo"] == titulo]
    assert len(bajo) == 1, (
        f"«{titulo}» aparece {len(bajo)} veces en el cuaderno")
    celdas = bajo[0]["celdas"]
    assert len(celdas) == 1, (
        f"«{titulo}» tiene {len(celdas)} celdas de código y tiene que tener "
        "una: el conteo por celda no dice nada repartido entre dos")
    return cuaderno["cells"][celdas[0]]


def _mimes_de(celda: dict) -> list[str]:
    return [_mime(salida) for salida in celda.get("outputs", [])]


def _titulos(cuaderno: dict) -> list[str]:
    """Every markdown heading of the notebook, in the order a reader meets them."""
    return [linea.rstrip()
            for celda in cuaderno["cells"] if celda["cell_type"] == "markdown"
            for linea in "".join(celda["source"]).splitlines()
            if linea.startswith("#")]


# --------------------------------- la forma del documento, leída de la de CREDA
#
# What a reader MEETS, counted per heading and per cell, against the notebook
# the operator pointed at. Every other test in this file reads either a
# renderer's return value or a cell's source, and ten passes of this document
# drifted past all of them: each pass added one more output under a heading
# that already had the one CREDA has, and no renderer was wrong, no cell was
# unexecuted and no number was typed. What was wrong was the COUNT, which
# nothing read.


def test_the_family_blocks_show_what_credas_family_blocks_show(
        cuaderno, cuaderno_de_creda, registro):
    """Per heading, per block, against CREDA's own measured block shape.

    This is the test the drift needed. Add an output under any of the nine
    subsections and it fires; remove one and it fires; grow a fourth
    subsection and it fires. The template is derived from CREDA's three blocks
    and only after measuring that the three agree, so it is the shape of that
    document rather than one of its blocks.
    """
    familias = toy_tables.familias(registro)
    plantilla = _plantilla_de_bloque(cuaderno_de_creda, familias)
    mios = _bloques_de_familia(_firma_por_encabezado(cuaderno), familias)
    for familia in familias:
        assert mios.get(familia) == plantilla, (
            f"el bloque de {familia} muestra "
            f"{ {s: tuple(m) for s, m in (mios.get(familia) or {}).items()} } "
            f"y el bloque de CREDA muestra "
            f"{ {s: tuple(m) for s, m in plantilla.items()} }")


def test_the_figure_cell_shows_one_image_and_prints_nothing(
        cuaderno, cuaderno_de_creda, registro):
    """Per CELL, which the per-heading test above cannot answer.

    A heading whose two cells summed to CREDA's counts would pass that one, and
    the figure cell is exactly where that happened: it emitted the GIF and a
    stream saying where the file was archived. CREDA's own figure cell emits the
    image alone -- its `animar(...)` call, and the line that call prints, live
    in the `## <familia>` cell above it.
    """
    familias = toy_tables.familias(registro)
    plantilla = _plantilla_de_bloque(cuaderno_de_creda, familias)
    sufijo = _sufijo_con(plantilla, ("image/gif",), "la figura")
    for familia in familias:
        mimes = _mimes_de(_celda_unica_bajo(cuaderno, f"{familia} {sufijo}"))
        assert mimes == ["image/gif"], (
            f"la celda de la figura de {familia} emitió {mimes} y la de CREDA "
            "emite la imagen y nada más")


def test_the_family_table_cell_shows_exactly_one_table(
        cuaderno, cuaderno_de_creda, registro):
    """One markdown under `— la tabla`, which is what CREDA's cell emits.

    The second one was the computed conclusion. Phase 1's `valid` -> `test` pair
    used to survive as a second GRID inside this same markdown; the operator
    retired it, so the markdown now carries one grid as well as being one
    markdown -- `test_what_the_notebook_shows_is_what_the_artefact_holds` counts
    the grids.
    """
    familias = toy_tables.familias(registro)
    plantilla = _plantilla_de_bloque(cuaderno_de_creda, familias)
    sufijo = _sufijo_con(plantilla, ("text/markdown",), "la tabla")
    for familia in familias:
        mimes = _mimes_de(_celda_unica_bajo(cuaderno, f"{familia} {sufijo}"))
        assert mimes == ["text/markdown"], (
            f"la celda de la tabla de {familia} emitió {mimes} y la de CREDA "
            "emite una tabla sola")


def test_the_declared_tests_cell_shows_exactly_two_tables(
        cuaderno, cuaderno_de_creda, registro):
    """Two markdowns under `— los tests declarados`, and CREDA's count says two.

    They are its `tabla_sin_contra_con` and its `tabla_tres_brazos`. The third
    was the computed conclusion.
    """
    familias = toy_tables.familias(registro)
    plantilla = _plantilla_de_bloque(cuaderno_de_creda, familias)
    sufijo = _sufijo_con(plantilla, ("text/markdown", "text/markdown"),
                         "los tests declarados")
    for familia in familias:
        mimes = _mimes_de(_celda_unica_bajo(cuaderno, f"{familia} {sufijo}"))
        assert mimes == ["text/markdown", "text/markdown"], (
            f"la celda de los tests de {familia} emitió {mimes} y la de CREDA "
            "emite las dos tablas y nada más")


def test_the_family_heading_prints_where_the_figure_was_archived(
        cuaderno, cuaderno_de_creda, registro):
    """The stream the figure cell shed did not disappear: it moved up one
    heading, which is where CREDA's own `animar(...)` call prints it.

    Dropping it instead would have lost the one thing a figure cell cannot
    carry -- the path the GIF was written to -- so the file would be on disk and
    the document would not say where.
    """
    familias = toy_tables.familias(registro)
    plantilla = _plantilla_de_bloque(cuaderno_de_creda, familias)
    assert plantilla[""] == ["stream"], (
        f"la celda de `## <familia>` de CREDA emite {plantilla['']}")
    for familia in familias:
        celda = _celda_unica_bajo(cuaderno, familia)
        assert _mimes_de(celda) == ["stream"], familia
        texto = "\n".join("".join(salida.get("text") or [])
                           for salida in celda["outputs"])
        # El join con el disco, y no sólo que la línea esté: la celda dice
        # dónde archivó el GIF, así que el archivo que nombra tiene que estar
        # ahí. Nada se vuelve a animar acá --- se lee lo que la celda emitió.
        nombradas = [ruta for ruta in re.findall(r"\S+\.gif", texto)
                     if (RAIZ / ruta).exists()]
        assert nombradas, (
            f"la celda de `## {familia}` no nombra ningún GIF que esté en "
            f"disco; lo que imprimió fue: {texto!r}")
        assert any(familia in ruta for ruta in nombradas), (
            f"la celda de `## {familia}` nombra {nombradas}, y ninguno es de "
            "esta familia: el bloque imprimiría la ruta del GIF de otra")


def test_nothing_in_this_document_renders_a_conclusion(cuaderno):
    """CREDA's document prints no conclusion, in any of its three blocks or
    anywhere else, and this one is held to that document's form.

    The general report rule asks for «a conclusion that comes from a function»,
    and that rule is why one kept being added under every table. The reference
    document the operator pointed at has none, so here there is none -- and the
    absence is pinned in the two places it could come back: the module, where a
    `conclusion_*` callable would be declared, and the notebook, where it would
    be called.
    """
    declaradas = [nombre for nombre in dir(toy_tables)
                  if nombre.startswith("conclusion")]
    assert declaradas == [], (
        f"`toy_tables` vuelve a declarar {declaradas}: este documento no lleva "
        "conclusiones, así que una función que compone una no tiene quién la "
        "llame")
    assert "conclusion" not in _codigo(cuaderno), (
        "el cuaderno vuelve a llamar una conclusión")
    # El control de las dos aserciones de arriba: los renderizadores SÍ están y
    # el cuaderno SÍ los llama, así que ninguna de las dos pasó por leer un
    # módulo equivocado ni un cuaderno vacío.
    assert [nombre for nombre in dir(toy_tables)
            if nombre.startswith("render_")]
    assert "toy_tables.render_familia_escalera" in _codigo(cuaderno)


def test_the_architecture_cell_shows_as_many_tables_as_credas(
        cuaderno, cuaderno_de_creda):
    """The architecture section, located in CREDA rather than named here.

    It is the one `##` heading of CREDA's document that emits exactly one
    markdown, and its title is one this notebook shares -- so the heading is
    found by measuring both documents instead of by writing a title into this
    file.

    Only the TABLE count is asserted, and the stream beside it is now asserted
    by `test_the_architecture_cell_closes_with_a_printed_line_from_the_record`.

    **This docstring used to record the opposite decision, and the operator
    overruled it.** It said CREDA's stream was one measurement rendered twice
    -- CREDA prints the hidden-layer count and the module count per network,
    and `render_arquitecturas` already carries `adabnNormLayers` and
    `adabnLayers` as two MEASURED rows -- so the difference was reported and
    left open. The ruling is that this document has CREDA's shape, and the
    duplication is the price of that shape rather than an argument against it.
    """
    de_creda = _firma_por_encabezado(cuaderno_de_creda)
    mios = {entrada["titulo"]: entrada for entrada in _firma_por_encabezado(cuaderno)}
    candidatos = [entrada for entrada in de_creda
                  if entrada["nivel"] == 2
                  and [m for m in entrada["mimes"] if m == "text/markdown"]
                  and entrada["titulo"] in mios]
    assert len(candidatos) == 1, (
        f"{len(candidatos)} secciones `##` de CREDA emiten markdown y tienen "
        f"un título que este cuaderno comparte: {[c['titulo'] for c in candidatos]}")
    titulo = candidatos[0]["titulo"]
    esperadas = candidatos[0]["mimes"].count("text/markdown")
    mias = mios[titulo]["mimes"].count("text/markdown")
    assert mias == esperadas, (
        f"«{titulo}» muestra {mias} tablas y la de CREDA muestra {esperadas}")


def _corte_de_las_arquitecturas(firma: list[dict], titulos_mios: set) -> int:
    """The index of the architectures heading, located in CREDA and not named.

    The same measurement `test_the_architecture_cell_shows_as_many_tables_as_credas`
    makes, returned as a position instead of a title: it is the one `##` heading
    of CREDA's document that emits markdown and whose title this notebook
    shares.
    """
    candidatos = [n for n, entrada in enumerate(firma)
                  if entrada["nivel"] == 2
                  and "text/markdown" in entrada["mimes"]
                  and entrada["titulo"] in titulos_mios]
    assert len(candidatos) == 1, (
        f"{len(candidatos)} secciones `##` de CREDA emiten markdown y tienen "
        f"un título que este cuaderno comparte: "
        f"{[firma[n]['titulo'] for n in candidatos]}")
    return candidatos[0]


def test_the_architecture_cell_closes_with_a_printed_line_from_the_record(
        cuaderno, cuaderno_de_creda, registro):
    """CREDA's architecture cell shows the table and then prints one line, and
    this one does too -- computed from the record, never written.

    The print is asserted twice over, because the two halves fail differently.
    That the cell EMITTED a stream is read off the document: a line computed
    into a variable and never printed leaves every reading of the source green.
    That the numbers in it come from the record is read by moving the record and
    requiring the line to move with it -- a line typed by hand survives the run
    that contradicts it, which is the whole reason the table above it is
    rendered rather than written.
    """
    de_creda = _firma_por_encabezado(cuaderno_de_creda)
    mios = {entrada["titulo"]: entrada for entrada in _firma_por_encabezado(cuaderno)}
    corte = _corte_de_las_arquitecturas(de_creda, set(mios))
    titulo = de_creda[corte]["titulo"]
    assert mios[titulo]["mimes"] == de_creda[corte]["mimes"], (
        f"«{titulo}» emitió {mios[titulo]['mimes']} y la de CREDA emite "
        f"{de_creda[corte]['mimes']}")
    assert "stream" in de_creda[corte]["mimes"], (
        "la celda de arquitecturas de CREDA ya no imprime nada, así que esta "
        "sección dejó de tener la forma que este test copia")

    indices = mios[titulo]["celdas"]
    assert len(indices) == 1, indices
    celda = cuaderno["cells"][indices[0]]
    impreso = "\n".join("".join(salida.get("text") or [])
                        for salida in celda["outputs"]
                        if salida["output_type"] == "stream")
    # El join con el registro, y la mutación que lo prueba: el conteo de
    # columnas de la tabla sale de `normalizaciones`, así que el renglón que
    # dice lo mismo tiene que salir de ahí también. Se le quita una
    # arquitectura al registro y se exige que el número cambie -- sin esta
    # mitad, un renglón escrito a mano con el número correcto de hoy pasaría.
    declaradas = toy_tables.normalizaciones(registro)
    assert str(len(declaradas)) in impreso, (
        f"el renglón de la celda de arquitecturas no dice las "
        f"{len(declaradas)} arquitecturas que el registro declara: {impreso!r}")
    capas = {norma: sorted({celda_["adabnNormLayers"]
                            for celda_ in toy_tables.celdas(registro)
                            if celda_["normalization"] == norma})
             for norma in declaradas}
    assert any(len(valores) == 1 for valores in capas.values()), capas
    for valores in capas.values():
        if len(valores) == 1:
            assert str(valores[0]) in impreso, (
                f"el renglón no trae la cuenta de capas {valores[0]} que la "
                f"corrida midió: {impreso!r}")
    menos = copy.deepcopy(registro)
    menos["flags"]["normalizations"].pop(declaradas[-1])
    assert len(toy_tables.normalizaciones(menos)) != len(declaradas), (
        "la mutación no movió las arquitecturas declaradas, así que la "
        "exigencia de abajo no prueba nada")
    assert str(len(toy_tables.normalizaciones(menos))) != str(len(declaradas))


def test_the_preamble_shows_what_credas_preamble_shows(
        cuaderno, cuaderno_de_creda):
    """Per heading and in order, from the top through the architectures heading,
    against CREDA's own measured preamble.

    The family blocks have had this check since the drift that motivated the
    section above; the PREAMBLE had none, and four passes of it drifted: two
    framing tables CREDA does not render, a stream beside the data figure CREDA
    emits without, a per-family data table under a heading CREDA gives one png
    and nothing else, and an architectures cell that showed its table and
    dropped the line CREDA prints under it. Every one of those was a renderer
    that worked, a cell that ran and a number nobody typed. What was wrong was
    the COUNT, which nothing read.

    Compared by POSITION and never by title: one of these headings is worded for
    this toyset (`con cada clase en bolsas` against `de la más fácil a la más
    difícil`), so a comparison keyed on titles would silently skip the very
    section the data figure lives under.

    **The slice ends at the architectures heading, and that end is a ruling
    rather than a convenience.** Below it the two documents stop agreeing at
    CREDA's weighting section, which retrains phase 1 inside its own plot
    notebook; this notebook may not train, by the two-notebook rule, so that
    divergence is the operator's to rule on and is not this test's to pin.
    """
    de_creda = _firma_por_encabezado(cuaderno_de_creda)
    mios = _firma_por_encabezado(cuaderno)
    corte = _corte_de_las_arquitecturas(de_creda,
                                        {e["titulo"] for e in mios})
    suyas = [tuple(e["mimes"]) for e in de_creda[:corte + 1]]
    assert len(mios) > corte, (
        f"este cuaderno tiene {len(mios)} secciones y la de las "
        f"arquitecturas de CREDA está en la {corte}")
    mias = [tuple(e["mimes"]) for e in mios[:corte + 1]]
    assert mias == suyas, (
        "el preámbulo de este cuaderno muestra "
        f"{list(zip([e['titulo'] for e in mios[:corte + 1]], mias))} y el de "
        f"CREDA muestra "
        f"{list(zip([e['titulo'] for e in de_creda[:corte + 1]], suyas))}")


# ------------------------------------------- el cuaderno lee, y no recalcula

def test_the_plot_notebook_opens_the_artefact_instead_of_recomputing_it(cuaderno):
    """It calls the loader, and calls nothing that would produce the numbers again.

    A notebook that recomputed them would agree with the artefact on the day it
    ran and drift from it silently afterwards, and nothing in the document would
    say which of the two a reader is looking at.
    """
    codigo = _codigo(cuaderno)
    assert "toy_run.cargar" in codigo, (
        "el cuaderno de gráficas no abre el artefacto: sin `toy_run.cargar` sus "
        "tablas no salen de lo que el generador dejó en disco")
    corren = [nombre for nombre in NADA_DE_ESTO_CORRE
              if f"toy_run.{nombre}" in codigo]
    assert corren == [], (
        f"el cuaderno de gráficas llama {corren}: eso entrena o escribe, y este "
        "cuaderno lee. La fase que entrena es `Toyset_Generator.ipynb`, y el "
        "corte entre los dos es lo que hace barata cualquier pregunta posterior")


def test_the_renderers_cannot_train_because_they_cannot_import_what_trains():
    """`toy_tables` has no path to a training loop, and that is structural.

    Read off the module's own source rather than off a promise in its docstring:
    a comment saying it only reads is one edit away from being false, and the
    import line is what a reader can check.
    """
    fuente = Path(toy_tables.__file__).read_text(encoding="utf-8")
    importa = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", fuente, re.M)
    prohibidos = [modulo for modulo in importa
                  if modulo.split(".")[0] in ("torch", "optuna", "numpy")
                  or modulo.endswith("toy_run")]
    assert prohibidos == [], (
        f"`toy_tables` importa {prohibidos}: el cuaderno que sólo lee pierde esa "
        "propiedad por el camino de importación de este módulo, no por un "
        "comentario")


def test_a_table_follows_the_artefact_on_disk_and_not_a_value_in_the_code(
        tmp_path, artefacto):
    """Mutate the artefact, and the rendered table has to move with it.

    This is the only test here that proves the connection rather than either
    half of it: `cargar` is run over a COPY of the real files, one accuracy is
    changed on disk, and the table is asked again. A renderer with the number
    baked in passes every other test in this file.
    """
    for origen in artefacto.iterdir():
        (tmp_path / origen.name).write_bytes(origen.read_bytes())

    base = toy_run.cargar(tmp_path)
    barrido = json.loads((tmp_path / "barrido.json").read_text(encoding="utf-8"))
    clave = next(iter(barrido["celdas"]))
    familia = clave.split(toy_tables.SEP)[0]
    antes = toy_tables.render_familia_escalera(base, familia)

    original = barrido["celdas"][clave]["targetAccuracy"]
    # El valor movido se MIDE contra la tabla original en vez de elegirse: el
    # primer intento fue `1 - original`, que en esta grilla cae justo sobre el
    # acierto de otra celda, y entonces «apareció» no habría probado nada.
    #
    # El formato es el de la reducción que la tabla imprime --- `media ± desvío`
    # de CREDA, que a una semilla es la media sola --- y no el porcentaje que
    # esta tabla imprimía antes de tomar las ocho columnas de CREDA.
    movido = next(candidato for candidato in
                  (n / 1000.0 for n in range(1, 1000))
                  if candidato != original
                  and toy_tables._num(candidato) not in antes)
    barrido["celdas"][clave]["targetAccuracy"] = movido
    (tmp_path / "barrido.json").write_text(
        json.dumps(barrido, ensure_ascii=False), encoding="utf-8")

    despues = toy_tables.render_familia_escalera(toy_run.cargar(tmp_path), familia)

    assert antes != despues, (
        f"se cambió {clave}.targetAccuracy de {original} a {movido} en el "
        "artefacto y la tabla salió idéntica: no la está leyendo de ahí")
    assert toy_tables._num(movido) in despues
    assert toy_tables._num(movido) not in antes


def test_the_table_covers_every_cell_the_flags_declare_or_refuses(registro):
    """A loop that skipped a cell would print a shorter table than the grid it
    claims to cover, and nothing in the output would say so."""
    recortado = copy.deepcopy(registro)
    faltante = next(iter(recortado["cells"]))
    familia = faltante.split(toy_tables.SEP)[0]
    del recortado["cells"][faltante]
    with pytest.raises(KeyError, match="más corta que la"):
        toy_tables.render_familia_escalera(recortado, familia)


def test_a_table_asked_for_a_family_the_record_never_declared_is_refused(
        registro):
    """A table for an absent family would come out EMPTY, and an empty table
    reads exactly like a table whose numbers are bad."""
    with pytest.raises(KeyError, match="no declara la familia"):
        toy_tables.render_familia_escalera(registro, "una-familia-que-no-existe")


# ------------------------------------------------- un bloque por familia, solo

def _grillas(markdown: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every markdown grid of one block, back into its header and its rows.

    A BLOCK and not a table: a rendering may carry more than one grid under one
    heading -- table B carries one per architecture with re-estimable buffers.
    A parser that took every pipe line as one table would read the second
    grid's header as a row of the first -- which is exactly the failure mode the
    positional parsing below exists to prevent, one level up.

    Grids are split on the RULE line, which is the only line markdown requires
    between a header and its rows.

    **Cells are stripped, because CREDA's are padded.** Its `_md` writes
    `| método | normalización |`, so the cell between two pipes carries a
    leading and a trailing space; a parser that kept them would fail to find a
    column by title and would compare a padded value against a bare one. The
    padding is the SPELLING of a cell and never part of its value, so it is
    removed here, once, for both documents. The spelling itself is the claim of
    one test of its own, below, and of no other.
    """
    lineas = [linea for linea in markdown.splitlines()
              if linea.startswith("|") and linea.endswith("|")]
    partido = [[celda.strip() for celda in linea.strip("|").split("|")]
               for linea in lineas]

    def es_regla(celdas: list[str]) -> bool:
        return all(celda.strip() and set(celda.strip()) <= set("-:")
                   for celda in celdas)

    grillas: list[tuple[list[str], list[list[str]]]] = []
    indice = 0
    while indice < len(partido):
        if indice + 1 < len(partido) and es_regla(partido[indice + 1]):
            cabecera = partido[indice]
            indice += 2
            filas: list[list[str]] = []
            while indice < len(partido) and not (
                    indice + 1 < len(partido) and es_regla(partido[indice + 1])):
                filas.append(partido[indice])
                indice += 1
            grillas.append((cabecera, filas))
        else:
            indice += 1
    return grillas


def _parsear(markdown: str) -> tuple[list[str], list[list[str]]]:
    """The FIRST grid of a block, back into its header and its rows.

    Needed because «the value appears somewhere in the row» is not the claim. It
    was measured not being the claim: the first version of the test below
    asserted exactly that, and it passed with the test column DELETED, because
    on the cell it had chosen the target accuracy happened to print the same
    string. A column is a position, so the assertion has to be about positions.
    """
    grillas = _grillas(markdown)
    assert grillas, f"no hay ninguna grilla en:\n{markdown[:400]}"
    return grillas[0]


def _columna(cabecera: list[str], filas: list[list[str]], titulo: str) -> list[str]:
    assert titulo in cabecera, (
        f"la tabla no tiene la columna {titulo!r}; tiene {cabecera}")
    indice = cabecera.index(titulo)
    return [fila[indice] for fila in filas]


def _fila_llamada(cabecera: list[str], filas: list[list[str]],
                  etiqueta: str) -> dict[str, str]:
    """One row of a TRANSPOSED table, as column title -> cell.

    `render_arquitecturas` is keyed the way CREDA's own architecture table is --
    one column per normalization, one row per fact -- so a reading of it is a
    row and not a column, and asking for it by position would break the day a
    row is inserted above it.
    """
    halladas = [fila for fila in filas if fila[0].startswith(etiqueta)]
    assert len(halladas) == 1, (
        f"{len(halladas)} filas de la tabla empiezan con {etiqueta!r}; las "
        f"filas son {[fila[0] for fila in filas]}")
    return dict(zip(cabecera[1:], halladas[0][1:]))


def _por_familia() -> dict:
    """The renderers that take a family, read off their signatures.

    Derived rather than listed, exactly as `renderizadores` is: a per-family
    renderer added after this file was written is the one most likely to forget
    the filter, and a hand-written list would be silent about precisely that
    one.
    """
    return {nombre: llamada
            for nombre, llamada in toy_tables.renderizadores().items()
            if "familia" in inspect.signature(llamada).parameters}


#: Un valor que ninguna celda real imprime, usado para teñir las OTRAS familias.
#: Elegido con seis decimales distintos para que sus cuatro formatos --- los dos
#: de `_num`, el de `_pct` --- sean cadenas que nada más del registro produce.
CENTINELA = 0.123456789


def _formatos_del_centinela() -> list[str]:
    return [toy_tables._pct(CENTINELA),
            toy_tables._num(CENTINELA, 4),
            toy_tables._num(CENTINELA, 6)]


def _tenir(registro: dict, excepto: str) -> dict:
    """The record with every float of every OTHER family set to the sentinel.

    Cells and per-family search records both, because the five per-family
    renderers between them read both halves. A renderer that ignored its family
    argument would print one of these, which is what the test below reads.
    """
    tenido = copy.deepcopy(registro)

    def pintar(nodo):
        if isinstance(nodo, dict):
            for clave, valor in nodo.items():
                if isinstance(valor, float):
                    nodo[clave] = CENTINELA
                else:
                    pintar(valor)
        elif isinstance(nodo, list):
            for indice, valor in enumerate(nodo):
                if isinstance(valor, float):
                    nodo[indice] = CENTINELA
                else:
                    pintar(valor)

    for clave, celda in tenido["cells"].items():
        if clave.split(toy_tables.SEP)[0] != excepto:
            pintar(celda)
            # **A tint that makes every float EQUAL makes every DIFFERENCE
            # zero**, and `render_familia_tests` prints differences. Measured:
            # with the flat tint, that table's own rows came out `+0.0000` for
            # the tinted families, so the sentinel could never appear in them
            # and the control half of the test below asserted nothing -- a
            # renderer that ignored its family argument would have leaked
            # `+0.0000`, which is also what a real zero difference prints.
            # So the ladder's three fields get a SPREAD instead of a constant,
            # chosen so that every declared pair's difference is one of the
            # sentinel's own formats.
            if celda["predictTargetAccuracy"] is not None:
                celda["predictTargetAccuracy"] = 0.0
            if celda["adabnTargetAccuracy"] is not None:
                celda["adabnTargetAccuracy"] = CENTINELA
            celda["targetAccuracy"] = CENTINELA
    for libro in ("records1", "records2", "hp1", "hp2"):
        for familia, cuerpo in tenido[libro].items():
            if familia != excepto:
                pintar(cuerpo)
    return tenido


def test_each_family_table_shows_that_family_and_no_other(registro):
    """Put another family's value in, and this family's tables must not print it.

    The row key cannot carry this claim on its own: all three families declare
    the same two normalizations, so a renderer that ignored its family argument
    would still produce a `normalización` column that looks right. What
    distinguishes them is the VALUES, so the other families' values are replaced
    by a sentinel and each table is asked whether it leaked one.

    The control is the second half and is what makes the first half mean
    anything: the sentinel has to APPEAR in the other families' own tables. A
    value nothing could render would be absent from every table, and the
    assertion would pass while measuring nothing.
    """
    familias = toy_tables.familias(registro)
    assert len(familias) > 1, (
        "el registro declara una sola familia, así que este test no puede "
        "distinguir una tabla filtrada de una que no filtra: revisar el artefacto")
    for nombre, llamada in _por_familia().items():
        for familia in familias:
            tenido = _tenir(registro, excepto=familia)
            propia = llamada(tenido, familia)
            colados = [forma for forma in _formatos_del_centinela()
                       if forma in propia]
            assert colados == [], (
                f"`{nombre}` para {familia} imprime {colados}, que es un valor "
                f"de otra familia: la tabla no está filtrada por familia y el "
                f"bloque de {familia} muestra números que no son suyos")
            ajenas = [otra for otra in familias if otra != familia]
            visible = [otra for otra in ajenas
                       if any(forma in llamada(tenido, otra)
                              for forma in _formatos_del_centinela())]
            assert visible, (
                f"el centinela no aparece en la tabla de ninguna de {ajenas} "
                f"con `{nombre}`, así que su ausencia en la de {familia} no "
                f"prueba ningún filtrado: el valor no es imprimible por esta "
                f"tabla")


#: The per-family renderer whose shape is CREDA's and therefore NOT one equal
#: block per architecture. Named once, so the exclusion below is a declaration
#: rather than a condition repeated in two tests.
TABLA_DE_CREDA = "render_familia_escalera"

#: And CREDA's other two shapes, which are not one equal block per architecture
#: either: table A is one row per architecture PLUS the row that averages them,
#: and table B is one table per architecture that has the three rungs. Both are
#: held by stricter tests of their own under «los tests declarados» -- the
#: exact row sequence including `PROMEDIO` for the first, and the derivation
#: from measured buffer presence for the second -- so the exclusion here is a
#: declaration of which claim holds them, never a loosening.
TABLAS_DE_CREDA = (TABLA_DE_CREDA, "render_familia_sin_contra_con")


def test_each_family_table_has_one_row_per_declared_architecture(registro):
    """Every declared architecture, the same number of rows each, in the
    record's own order.

    Only the renderers whose rows are cells: `render_familia_busqueda` is keyed
    down the page because a search is chosen once per family, and demanding a
    `normalización` column of it would demand a shape that would print the same
    row twice.

    **Blocks of equal size and not one row each, which is a widening and not a
    loosening.** This used to compare the column against the declared list
    verbatim, so it held only while a table had exactly one row per
    architecture -- true of `render_familia` at one report seed and false of it
    at ten. What the claim always was is that every architecture gets the same
    number of rows and they arrive in the record's order: a dropped architecture
    makes the block count wrong, a reordered one makes the sequence wrong, and
    an architecture rendered with fewer rows than another makes the division
    fail outright.

    **CREDA's own shapes are excluded, and each exclusion is checked rather
    than assumed.** Its rows are CREDA's seven, and CREDA's shape is
    deliberately not one equal block per architecture: AdaBN has a row only
    where the backbone carries buffers to re-estimate, so BatchNorm gets three
    rows and the other two get two. What holds that shape is stricter than this
    claim and not looser -- the tests under «la tabla de CREDA, fila por fila»
    name the rung AND the architecture of every row.

    The column may print the record's own key or CREDA's word for it; both are
    accepted here because which of the two a table uses is that table's own
    decision and `_rotulo_norma` is what refuses an architecture with no word.
    """
    declaradas = toy_tables.normalizaciones(registro)
    otras = {nombre: llamada for nombre, llamada in _por_familia().items()
             if nombre not in TABLAS_DE_CREDA}
    faltan = [nombre for nombre in TABLAS_DE_CREDA
              if nombre not in _por_familia()]
    assert faltan == [], (
        f"{faltan} no está entre los renderizadores por familia, así que la "
        "exclusión de este test no excluye nada")
    assert otras, (
        "no queda ningún renderizador por familia aparte del de CREDA, así que "
        "este test no mide nada")
    for nombre, llamada in otras.items():
        for familia in toy_tables.familias(registro):
            cabecera, filas = _parsear(llamada(registro, familia))
            if "normalización" not in cabecera:
                continue
            columna = _columna(cabecera, filas, "normalización")
            assert len(columna) % len(declaradas) == 0, (
                f"`{nombre}` para {familia} trae {len(columna)} filas, que no "
                f"se reparte entre las {len(declaradas)} arquitecturas "
                f"declaradas")
            por_norma = len(columna) // len(declaradas)
            crudo = [norma for norma in declaradas for _ in range(por_norma)]
            rotulado = [toy_tables.NOMBRE_FILA[norma] for norma in declaradas
                        for _ in range(por_norma)]
            assert columna in (crudo, rotulado), (
                f"`{nombre}` para {familia} no trae el mismo número de filas "
                f"por arquitectura declarada, en el orden del registro: "
                f"{columna}")


# ----------------------------- la columna de la fuente, contra lo que el artefacto dice

def test_the_source_column_carries_what_the_artefact_says_per_architecture(registro):
    """La columna `fuente · <rol>` trae el `sourceAccuracy` del artefacto, por
    arquitectura y leído de la grilla EMITIDA.

    Asertado por COLUMNA y por posición, nunca como subcadena: «el valor aparece
    en algún lugar de la fila» pasaba con la columna borrada, porque en la celda
    elegida otro acierto imprimía la misma cadena.

    Esta afirmación venía adentro de
    `test_every_family_table_shows_both_of_phase_ones_readings`, que se borró con
    la grilla del rol de selección que medía. Es de la PRIMERA grilla —las ocho
    columnas de CREDA—, así que no se fue con la segunda, y nadie más la mide:
    `test_the_family_table_has_exactly_credas_eight_columns_in_credas_order`
    compara los nombres de las columnas y no sus valores.
    """
    for familia in toy_tables.familias(registro):
        salida = toy_tables.render_familia_escalera(registro, familia)
        grillas = _grillas(salida)
        assert len(grillas) == 1, (
            f"el bloque de {familia} trae {len(grillas)} grillas y la celda de "
            "CREDA emite una sola")
        cabecera, filas = grillas[0]
        rol = toy_tables.rol_publicado(registro)
        celdas = toy_tables.celdas(registro, familia)
        fuente = _columna(cabecera, filas, f"fuente · {rol}")
        arquitectura = _columna(cabecera, filas, "normalización")
        esperado = {toy_tables.NOMBRE_FILA[celda["normalization"]]:
                    toy_tables._num(celda["sourceAccuracy"])
                    for celda in celdas}
        assert [esperado[norma] for norma in arquitectura] == fuente, (
            f"en {familia}, la columna de la fuente no trae lo que el artefacto "
            f"dice por arquitectura: {fuente}")


# --------------------------------- un bloque de dos grillas, y las dos enteras

def test_a_block_of_two_grids_renders_both_with_their_own_titles(registro):
    """Dos grupos, dos grillas, y el título de cada una sobre la suya.

    La propiedad es del constructor y no de sus clientes: hoy la arma
    `render_familia_tests`, una grilla por arquitectura con buffers
    reestimables, y este artefacto trae una sola --- así que si esto se le
    preguntara a un renderizador, la segunda grilla no se mediría en ninguna
    parte.

    Esto era
    `test_a_block_of_several_grids_carries_exactly_one_scale_stamp`, cuya
    afirmación --- el sello va una vez por bloque --- se borró con el sello que
    medía. La composición del bloque no era del sello y nadie más la mide, así
    que queda, sin él.
    """
    una = [toy_tables.Columna("acierto", lambda r: "x", ("targetAccuracy",))]
    otra = [toy_tables.Columna("blanco", lambda r: "y", ("watchmanRestored",))]
    salida = toy_tables._bloque([(None, una, [{}]), ("la otra", otra, [{}])])
    assert len(_grillas(salida)) == 2, salida
    assert [cabecera for cabecera, _ in _grillas(salida)] == [["acierto"],
                                                              ["blanco"]]
    assert "la otra" in salida


def test_every_renderer_goes_through_the_one_builder(registro, monkeypatch):
    """What makes CREDA's cell spelling a property of the module, not a habit.

    The stamp this test was written for is retired; what routing through the
    one builder buys now is that every grid of this document is spelled the way
    `_solo_la_grilla` spells it, which is the way CREDA's `_md` does.

    Derived from the roster rather than from a list written here, so a renderer
    added after this test was written is covered by it -- and that is the one
    most likely to build its rows by hand and lose the spelling.

    Asked by WATCHING the call and not by reading the source. The first version
    read the source for the text `_tabla(`, and a mutation that made a renderer
    return a hand-built table from its first line survived it: the abandoned
    call was still in the function below the early return, and `getsource` reads
    text rather than what runs.
    """
    llamadas: list[int] = []
    real = toy_tables._bloque

    def espia(grupos):
        llamadas.append(len(grupos))
        return real(grupos)

    # `_bloque` y no `_tabla`: `_tabla` es el caso de una grilla y delega acá,
    # así que espiar el bloque ve las dos clases de renderizador.
    monkeypatch.setattr(toy_tables, "_bloque", espia)
    # Una sola pasada por `renderizados` llama a cada renderizador exactamente
    # una vez --- por familia donde corresponda --- así que el conteo de
    # llamadas a `_tabla` se compara contra el número de tablas que esa misma
    # pasada devolvió. Un renderizador que arma su markdown a mano devuelve su
    # tabla sin aumentar el conteo, y la igualdad se rompe.
    tablas = toy_tables.renderizados(registro)
    assert len(llamadas) == len(tablas), (
        f"`renderizados` devolvió {len(tablas)} tablas y `_bloque` se llamó "
        f"{len(llamadas)} veces: alguna arma su grilla sin pasar por el único "
        "constructor, así que no la escribe como la escribe CREDA")


# ----------------------------- el pie, los títulos y el espaciado, los de CREDA
#
# Las tres diferencias que quedaban entre este documento y el de CREDA, medidas
# sobre los dos. Ninguna se ve renderizada --- un pie en bastardilla se lee como
# una nota, dos títulos en negrita se leen como dos cosas distintas y los pipes
# con o sin espacios renderizan idénticos ---, y la comparación que el operador
# hace es entre los dos TEXTOS. Así que lo que se compara acá es el texto, y la
# referencia se LEE del cuaderno de CREDA en vez de escribirse como una cuenta
# en este archivo.

#: Una línea entera en bastardilla: la forma del pie que `escala_sello` ponía
#: debajo de cada bloque. `**negrita**` no entra --- empieza con dos asteriscos
#: --- y una bastardilla EN MEDIO de un párrafo tampoco, porque el pie era su
#: propia línea.
PIE_EN_BASTARDILLA = re.compile(r"^\*[^*].*\*$")

#: Una línea entera en negrita: un título de grilla o de bloque.
TITULO_EN_NEGRITA = re.compile(r"^\*\*.+\*\*$")


def _lineas_que_son(patron, markdown: str) -> list[str]:
    return [linea.strip() for linea in markdown.splitlines()
            if patron.match(linea.strip())]


def _espaciado_de_celda(markdown: str) -> frozenset:
    """Cómo escribe este markdown una celda, por clase de fila.

    Una entrada por fila con pipes: si es la regla, y qué padding tienen sus
    celdas. CREDA da dos --- `(regla, ((False, False),))` para `|---|---|` y
    `(fila, ((True, True),))` para `| a | b |` ---, y comparar conjuntos dice
    que las dos clases de fila se escriben igual en los dos documentos sin
    escribir acá ninguna de las dos formas.
    """
    formas = set()
    for linea in markdown.splitlines():
        if not (linea.startswith("|") and linea.endswith("|")):
            continue
        celdas = linea.strip("|").split("|")
        regla = all(celda.strip() and set(celda.strip()) <= set("-:")
                    for celda in celdas)
        formas.add((regla, tuple(sorted({(celda[:1] == " ", celda[-1:] == " ")
                                         for celda in celdas}))))
    return frozenset(formas)


def _bloques_con_grilla(cuaderno: dict) -> list[str]:
    """Cada markdown EMITIDO que trae una grilla, en el orden de lectura.

    Lo emitido y no la fuente: el pie, los títulos y el espaciado son del texto
    que el lector encuentra. Los markdown sin grilla quedan afuera --- el
    cuaderno de CREDA abre un `##` desde un `display(Markdown(...))` y eso no es
    una tabla.
    """
    bloques = []
    for celda in cuaderno["cells"]:
        for salida in celda.get("outputs", []):
            texto = "".join((salida.get("data") or {}).get("text/markdown") or [])
            if texto and _grillas(texto):
                bloques.append(texto)
    return bloques


def _bloques_por_subseccion(cuaderno: dict, familias: list[str]) -> dict:
    """{sufijo: (markdown, ...)} del bloque de familia, medido igual en las tres.

    `_plantilla_de_bloque` arriba hace esto con los MIME de cada salida; esto
    conserva el texto, que es lo que las tres cuentas de abajo leen. La
    plantilla sale de que las tres familias coincidan en cuántas salidas trae
    cada subsección, igual que allá: una plantilla tomada de la primera sería un
    accidente de un bloque.
    """
    firma = _firma_por_encabezado(cuaderno)
    textos: dict = {}
    actual = None
    for entrada in firma:
        titulo, nivel = entrada["titulo"], entrada["nivel"]
        if nivel <= 2:
            actual = titulo if titulo in familias else None
            if actual is not None:
                textos[actual] = {}
        elif actual is not None:
            sufijo = (titulo[len(actual):].strip()
                      if titulo.startswith(actual) else titulo)
            textos[actual][sufijo] = tuple(entrada["textos"])
    assert sorted(textos) == sorted(familias), sorted(textos)
    cuentas = {familia: {sufijo: len(mds) for sufijo, mds in bloque.items()}
               for familia, bloque in textos.items()}
    assert len({tuple(sorted(c.items())) for c in cuentas.values()}) == 1, (
        f"los bloques de familia no traen el mismo número de markdown por "
        f"subsección: {cuentas}")
    return textos


def test_no_rendering_carries_a_footer_line_because_credas_carries_none(
        cuaderno_de_creda, registro):
    """CREDA no pone pie bajo ninguno de sus bloques, así que acá tampoco.

    La cuenta de referencia se MIDE sobre su documento y no se escribe acá: si
    mañana su notebook trajera un pie, esto diría que falta en vez de seguir
    exigiendo cero. Y se le pregunta a `renderizados`, que expande cada
    renderizador por familia, porque un pie que se fuera en `lunas` y quedara en
    `rollo` sería un bloque de tres con pie.

    Lo que esto reemplaza es `test_every_table_carries_the_scale_stamp`, que
    afirmaba lo contrario --- que ninguna tabla se renderiza sin el sello ---.
    El operador retiró el sello; la afirmación no se debilitó, se dio vuelta.
    """
    suyos = {len(_lineas_que_son(PIE_EN_BASTARDILLA, md))
             for md in _bloques_con_grilla(cuaderno_de_creda)}
    assert suyos == {0}, (
        f"los bloques del cuaderno de CREDA traen {suyos} líneas de pie: la "
        "referencia de esta cuenta cambió y hay que volver a leerla")
    con_pie = {etiqueta: _lineas_que_son(PIE_EN_BASTARDILLA, salida)
               for etiqueta, salida in toy_tables.renderizados(registro).items()
               if _lineas_que_son(PIE_EN_BASTARDILLA, salida)}
    assert con_pie == {}, (
        f"{sorted(con_pie)} trae un pie en bastardilla y los bloques de CREDA "
        f"no traen ninguno: {con_pie}")


def test_a_renderer_that_grew_a_footer_is_caught(registro, monkeypatch):
    """El test de arriba es alcanzable: un renderizador con pie lo rompe.

    Sin esto, una cuenta de pies que pasara por no estar leyendo nada se vería
    igual que una que pasa porque ningún renderizador pone pie. Declarado con
    `familia` a propósito, así que también prueba que la expansión por familia
    de la que depende el test de arriba llega a un renderizador por familia ---
    que es la mitad que traía `test_a_renderer_that_forgot_the_stamp_is_caught`,
    borrado con el sello que medía.
    """
    # La línea en bastardilla es un LITERAL de este test desde que el operador
    # retiró `escala_sello` con el sello de la figura, que era su último lector.
    # Antes se tomaba prestada de ahí; lo que este test necesita es un pie, no
    # ese pie, así que el literal no pierde nada de lo que mide.
    def render_con_pie(registro_, familia):
        return (f"| familia |\n|---|\n| {familia} |\n\n"
                "*un pie en bastardilla, puesto por este test*")

    monkeypatch.setitem(toy_tables.__dict__, "render_con_pie", render_con_pie)
    expandidos = toy_tables.renderizados(registro)
    esperadas = {f"render_con_pie[{familia}]"
                 for familia in toy_tables.familias(registro)}
    assert esperadas <= set(expandidos), (
        "`renderizados` no expandió el renderizador por familia que este test "
        "declaró, así que la cobertura por familia del test de arriba no está "
        "probada")
    con_pie = sorted(etiqueta for etiqueta in esperadas
                     if _lineas_que_son(PIE_EN_BASTARDILLA,
                                        expandidos[etiqueta]))
    assert con_pie == sorted(esperadas), con_pie
    # Y el otro polo, sobre un renderizador de verdad: el que más arriba traía
    # el sello no trae ninguna línea en bastardilla hoy.
    assert _lineas_que_son(
        PIE_EN_BASTARDILLA,
        toy_tables.render_familia_escalera(
            registro, toy_tables.familias(registro)[0])) == []


def test_each_block_carries_the_bold_titles_credas_block_carries(
        cuaderno, cuaderno_de_creda, registro):
    """Por subsección del bloque de familia, contra el bloque de CREDA.

    Un título de más es exactamente lo que tenía la segunda tabla de los tests
    declarados: el encabezado de la familia arriba del bloque y el de la
    arquitectura arriba de la grilla, donde CREDA lleva uno solo. Se cuenta por
    SALIDA y no por subsección, porque la subsección de los tests emite dos
    markdown y el defecto estaba en el segundo.
    """
    familias = toy_tables.familias(registro)

    def por_subseccion(cuaderno_: dict) -> dict:
        return {familia: {sufijo: tuple(
                    len(_lineas_que_son(TITULO_EN_NEGRITA, md)) for md in mds)
                 for sufijo, mds in bloque.items()}
                for familia, bloque in _bloques_por_subseccion(
                    cuaderno_, familias).items()}

    # La plantilla sale de que los TRES bloques de CREDA coincidan, igual que
    # `_plantilla_de_bloque`: tomada del primero sería el accidente de un
    # bloque, y este documento quedaría atado a él.
    de_creda = por_subseccion(cuaderno_de_creda)
    suyos = de_creda[familias[0]]
    assert list(de_creda.values()) == [suyos] * len(familias), (
        f"los bloques de familia de CREDA no traen los mismos títulos en "
        f"negrita por subsección: {de_creda}")
    assert any(sum(cuentas) for cuentas in suyos.values()), (
        f"ningún bloque de CREDA trae un título en negrita, así que esta "
        f"cuenta pasaría leyendo ceros: {suyos}")
    mios = por_subseccion(cuaderno)
    for familia in familias:
        assert mios[familia] == suyos, (
            f"el bloque de {familia} trae {mios[familia]} títulos en negrita "
            f"por subsección y el de CREDA trae {suyos}")


def test_the_one_builder_spells_a_cell_the_way_credas_does(
        cuaderno, cuaderno_de_creda, registro):
    """`| a | b |` y `|---|---|`, que es como las escribe `toy_plots._md`.

    Leído de las dos tablas emitidas y no escrito acá: lo que se compara es el
    conjunto de formas de fila de cada documento, así que la referencia es el
    documento. Markdown renderiza las dos formas igual, y el operador compara
    los textos --- el espaciado no se ve y sí se diferencia.

    También sobre `renderizados`, porque el cuaderno sólo tiene lo que la
    última ejecución dejó: una grilla recién agregada que nadie ejecutó todavía
    no está en ninguna salida.
    """
    suyo = frozenset().union(*(_espaciado_de_celda(md)
                               for md in _bloques_con_grilla(cuaderno_de_creda)))
    assert len(suyo) == 2, (
        f"el cuaderno de CREDA escribe {sorted(suyo)} clases de fila y tiene "
        "que escribir dos --- la regla y las demás ---: la referencia cambió")
    mio = frozenset().union(*(_espaciado_de_celda(md)
                              for md in _bloques_con_grilla(cuaderno)))
    assert mio == suyo, (
        f"este cuaderno escribe sus celdas {sorted(mio)} y el de CREDA las "
        f"escribe {sorted(suyo)}")
    de_los_renderizadores = frozenset().union(
        *(_espaciado_de_celda(salida)
          for salida in toy_tables.renderizados(registro).values()
          if _grillas(salida)))
    assert de_los_renderizadores == suyo, (
        f"los renderizadores escriben sus celdas {sorted(de_los_renderizadores)}"
        f" y el cuaderno de CREDA las escribe {sorted(suyo)}")


# ---------------------------------------------------- las dos arquitecturas

def test_the_architecture_table_counts_what_the_record_declares(registro):
    """One column per declared normalization, and one more for each absent one.

    The count is the whole point. A heading that said three over a grid of two
    is the defect this section was rebuilt to close, and a table that read its
    own count from a constant would be the same defect one file down.

    **This test used to require the record to declare an ABSENT normalization,
    and that premise is gone**: the third backbone exists, so `ausentes` is
    empty and a test asserting it non-empty would be a test pinning a gap that
    was closed. The claim was never about the gap -- it is that the WIDTH comes
    from the record -- so the reachable red is now built instead of found: a
    fictional absent row is injected into a copy and the table has to grow a
    column for it. That direction is the stronger one anyway, because it is the
    one a future fourth row would arrive through.
    """
    declaradas = toy_tables.normalizaciones(registro)
    assert toy_tables.ausentes(registro) == [], (
        "el registro declara una normalización ausente; este test está escrito "
        "para la grilla completa y hay que releerlo antes de creerle")
    cabecera, filas = _parsear(toy_tables.render_arquitecturas(registro))
    assert cabecera[0] == "qué"
    assert cabecera[1:] == declaradas, (
        f"la tabla trae {cabecera[1:]} y el registro declara {declaradas}: el "
        "ancho no sale del registro")
    razon = _columna(cabecera, filas, "qué")
    assert "por qué está o no está" in razon

    # El rojo alcanzable: una ausencia inventada en una COPIA del registro
    # tiene que agregar su propia columna, nombrada, y la fila del motivo
    # tiene que decir cuántos backbones declara el registro.
    con_ausente = copy.deepcopy(registro)
    con_ausente["flags"]["normalizationsAbsent"] = ["instancenorm"]
    ancha, filas_anchas = _parsear(
        toy_tables.render_arquitecturas(con_ausente))
    assert len(ancha) == len(cabecera) + 1, (
        f"se declaró una normalización ausente y la tabla pasó de "
        f"{len(cabecera)} a {len(ancha)} columnas: las columnas no salen del "
        "registro")
    assert any("instancenorm" in titulo for titulo in ancha), ancha
    motivo = next(f for f in filas_anchas if f[0] == "por qué está o no está")
    assert f"{len(declaradas)} backbone" in motivo[-1], motivo[-1]


def test_the_parameter_count_is_measured_across_families_before_printing_one(
        registro):
    """It should be the same in all three, and «should be» is a claim.

    So the renderer checks it. Moved here from the bandwidth table, where it
    was rendered once per family and printed the same two numbers three times.
    """
    cabecera, filas = _parsear(toy_tables.render_arquitecturas(registro))
    fila = next(f for f in filas if f[0] == "parámetros del brazo")
    norma = toy_tables.normalizaciones(registro)[0]
    medidos = {celda["parameters"]
               for celda in toy_tables.celdas(registro)
               if celda["normalization"] == norma}
    assert len(medidos) == 1, (
        "las familias ya difieren en el conteo de parámetros de esta "
        "normalización, así que el test de abajo no mide el camino constante")
    assert fila[cabecera.index(norma)] == f"{next(iter(medidos)):,}"

    dispar = copy.deepcopy(registro)
    victima = next(clave for clave, celda in dispar["cells"].items()
                   if celda["normalization"] == norma)
    dispar["cells"][victima]["parameters"] += 1
    _, otras = _parsear(toy_tables.render_arquitecturas(dispar))
    cambiada = next(f for f in otras if f[0] == "parámetros del brazo")
    assert "según la familia" in cambiada[cabecera.index(norma)], (
        "una familia con otro conteo de parámetros no cambió la celda: el "
        "número se imprime sin medir que sea el mismo en las tres")



# ----------------------------------------------- la vuelta por los archivos reales

def test_cargar_round_trips_against_the_real_files_and_carries_every_field(
        registro, artefacto):
    """What the renderers read is what the loader actually brings back.

    Driven over the files `guardar` left, never over a fixture written here: a
    fixture verifies this file's idea of the artefact and the renderers against
    each other, and never either of them against the artefact.
    """
    assert set(registro) >= {"flags", "cells", "rows", "hp1", "hp2",
                             "records1", "records2", "trials1", "trials2",
                             "cost", "scale", "families"}
    # Dos fuentes, porque los renderizadores declaran lo que leen de dos
    # maneras y una sola dejaba la mitad afuera. El `lee` de cada columna es la
    # DECLARACIÓN del módulo y es lo único que cubre la tabla de CREDA,
    # cuyas columnas indexan la celda
    # por una variable y no por un literal. El regex sobre la fuente cubre a los
    # que sí indexan por literal. Medido: con el regex solo, este barrido salía
    # con un campo y creía estar midiendo nueve.
    leidos = {campo for nombre, llamada in toy_tables.renderizadores().items()
              for campo in re.findall(r'\["celda"\]\["(\w+)"\]',
                                      inspect.getsource(llamada))}
    leidos |= {campo for _, columnas, _ in _grupos_de_la_escalera(registro)
               for columna in columnas for campo in columna.lee}
    # El campo de la serie por paso es la UNA ausencia declarada: la columna del
    # oráculo lo nombra para poder poner raya cuando no está, así que exigirlo
    # acá sería exigir que el artefacto traiga lo que esa columna mide que no
    # trae. Su propio test cruza las dos mitades.
    assert toy_tables.SERIE_BLANCO in leidos
    leidos.discard(toy_tables.SERIE_BLANCO)
    assert len(leidos) > 1, (
        f"el barrido de abajo mira {sorted(leidos)}: con un campo o ninguno no "
        "está midiendo que el artefacto traiga lo que las tablas piden")
    for clave in toy_tables.claves_de_celda(registro):
        celda = registro["cells"][toy_tables.SEP.join(
            (clave[0], clave[1], clave[2], str(clave[3])))]
        faltan = sorted(campo for campo in leidos if campo not in celda)
        assert faltan == [], (
            f"la celda {clave} del artefacto en {artefacto} no trae {faltan}, "
            "y una tabla los pide")
    for familia, contenido in registro["families"].items():
        assert contenido["info"], f"{familia}.npz volvió sin registros"


def test_the_resolution_comes_from_the_split_the_run_realized(registro):
    """Not from the fractions it asked for: integer bags cannot honour a
    fraction, and what was measured on is the realized count."""
    res = toy_tables.resolucion(registro)
    partido = registro["scale"]["realizedSplit"]
    clases = registro["scale"]["classes"]
    assert res["bolsasEvaluacion"] == partido["bagsPerClass"]["test"] * clases
    assert res["valoresPosibles"] == res["bolsasEvaluacion"] + 1
    assert res["puntosPorcentualesPorBolsa"] == pytest.approx(
        100.0 / res["bolsasEvaluacion"])

    mitad = copy.deepcopy(registro)
    mitad["scale"]["realizedSplit"]["bagsPerClass"]["test"] = (
        partido["bagsPerClass"]["test"] * 2)
    movida = toy_tables.resolucion(mitad)
    assert movida["puntosPorcentualesPorBolsa"] < res["puntosPorcentualesPorBolsa"]


def test_the_landmarks_are_computed_from_the_record_and_not_written(registro):
    """A landmark typed into the prose goes on naming the old one the day a
    constant moves, which is how a framing outlives its own experiment."""
    clases = registro["scale"]["classes"]
    assert toy_tables.azar(registro) == pytest.approx(1.0 / clases)
    otras = copy.deepcopy(registro)
    otras["scale"]["classes"] = clases + 1
    assert toy_tables.azar(otras) != toy_tables.azar(registro)
    assert (toy_tables.objetivo(otras)["acierto"]["hito"]
            != toy_tables.objetivo(registro)["acierto"]["hito"])


# -------------------------------------------------- lo que el documento emitió

def test_the_notebooks_heading_tree_is_the_one_grouped_by_toyset(
        cuaderno, registro):
    """The whole shape of the document, asserted as a sequence.

    A sequence and not a set: a reordering, a missing family block, a family
    block whose figure and table subsections got separated, and a heading that
    counts wrong all fail here. The family blocks and the two counted words are
    DERIVED from the record, so the tree cannot be satisfied by a notebook that
    describes a different grid than the artefact it reads.
    """
    familias = toy_tables.familias(registro)
    normas = toy_tables.normalizaciones(registro)
    esperado = [
        "# El toyset de MIL-CREDA — figuras y tablas",
        "## Dónde está el repositorio",
        "## Lo que el generador dejó",
        f"## Las {NUMERO_EN_PALABRAS[len(familias)]} familias, con cada clase "
        "en bolsas",
        f"## Las {NUMERO_EN_PALABRAS[len(normas)]} arquitecturas",
        "## La figura, una por dataset",
    ]
    for familia in familias:
        esperado += [f"## {familia}", f"### {familia} — la figura",
                     f"### {familia} — la tabla",
                     f"### {familia} — los tests declarados"]
    esperado.append("## El sopesado, por construcción")
    assert _titulos(cuaderno) == esperado


def test_every_family_block_carries_its_figure_and_not_the_dataset_again(
        cuaderno, registro):
    """`### <familia> — la figura` is CREDA's embedded space, and it is HERE.

    **This test used to assert the opposite, and the reason it gave stopped
    being true.** It said the subsection was absent because the artefact could
    not give it: per cell the flat scalars, the phase-1 curve and the sigma
    sweep, nothing positional, the phase-2 trace dropped by `_without_curves`
    and no checkpoint to recompute from. `toy_run.LATENT_MEMBERS` is what
    changed that -- the generator now writes the bag positions of both domains,
    before and after phase 2 -- so the absence is over and a test still pinning
    it would be a test that lies.

    What survives unchanged is the guard on the ONE substitution that would
    fill the slot silently: drawing the input figure a second time. A reader
    meeting two different figures at the same place in the two notebooks' trees
    would read one as the other, and that risk did not go away when the slot
    got filled -- it is exactly what a figure cell could still quietly become.
    """
    titulos = _titulos(cuaderno)
    faltan = [f"### {familia} — la figura" for familia in toy_tables.familias(registro)
              if f"### {familia} — la figura" not in titulos]
    assert faltan == [], (
        f"faltan las subsecciones {faltan}: el artefacto trae las posiciones y "
        "el bloque de esa familia no las muestra")
    codigo = _codigo(cuaderno)
    assert codigo.count("figura_familias") == 1, (
        "el cuaderno dibuja la figura de los datos más de una vez: los datos "
        "crudos se muestran una sola vez y nunca más, y una segunda llamada es "
        "la sustitución que haría ilegible la comparación con el cuaderno de "
        "CREDA")
    assert "familias=" not in codigo, (
        "el cuaderno llama `figura_familias` con un subconjunto de familias: "
        "eso es el dataset de una familia dibujado de nuevo en el lugar donde "
        "va su espacio embebido")


def test_the_notebook_renders_every_family_through_the_per_family_renderers(
        cuaderno, registro):
    """The tree above is headings; this is what sits under them.

    A notebook could carry the right heading tree and render `lunas` three
    times, and the heading test would pass. So each per-family renderer is
    required to appear in the notebook once per declared family.
    """
    codigo = _codigo(cuaderno)
    faltan = [f"{nombre}(REGISTRO, \"{familia}\")"
              for nombre in _por_familia()
              for familia in toy_tables.familias(registro)
              if f"toy_tables.{nombre}(REGISTRO, \"{familia}\")" not in codigo]
    assert faltan == [], (
        f"el cuaderno no llama {faltan}: hay un bloque de familia cuyo título "
        "está y cuyas tablas son de otra familia o no están")


# ------------------- el espacio latente: lo que quedó, y lo que se retiró
#
# `toy_run.figura_latente` -- la lámina estática de un panel por arquitectura --
# ESTÁ RETIRADA, con `FiguraLatente`, `_panel_latente`, `_dibujar_panel` y
# `acercamiento_y_dispersion`. El cuadro animado dibuja las mismas posiciones y
# lleva además los pasos, así que las dos juntas eran una medición dibujada dos
# veces. Cuatro tests de este archivo medían esa lámina y se fueron con ella;
# lo que cada uno afirmaba, y a dónde fue:
#
# * `test_the_figure_has_one_panel_per_declared_architecture` -- que la figura
#   trae un panel por arquitectura DECLARADA. Sigue medido, sobre el cuadro
#   animado, en `test_the_grid_has_one_row_per_declared_normalization_and_four_
#   columns` más abajo, que es el mismo reclamo sobre la figura que quedó.
# * `test_the_limits_and_the_basis_are_per_architecture_and_never_shared` -- que
#   los límites y la base de PCA son por arquitectura. Sigue medido:
#   `animar_familia` devuelve los mismos `ejes` con el ángulo entre bases, y el
#   cuaderno los imprime; lo que se perdió es la lectura de las cajas DIBUJADAS
#   de la lámina, que ya no existe.
# * `test_the_notebook_shows_the_latent_figure_inline_and_archives_it` -- que la
#   lámina se mostraba Y se archivaba. Reemplazado por
#   `test_the_latent_sheet_is_called_nowhere` abajo, que mide lo contrario a
#   propósito: que no se llama en ninguna parte.
# * `test_the_latent_figure_cells_showed_an_image_and_not_a_description` -- que
#   las celdas de la lámina EMITIERON una imagen. Reemplazado por
#   `test_the_figure_cells_emitted_an_image_and_not_a_description`, que hace la
#   misma lectura sobre las celdas del GIF, que son las que quedaron.
#
# `acercamiento_y_dispersion` se fue y su fórmula no: vive una sola vez en
# `toy_run._lectura_de_cuadro`, y `trayectoria_de_acercamiento` es lo que la
# aplica, cuadro por cuadro, para la figura que quedó. El test de abajo es el
# de la lámina, apuntado a esa función.


# `test_the_trajectory_reports_the_approach_beside_the_collapse_that_fakes_it`
# SE BORRÓ acá el 2026-10-15, con la lectura que medía. Lo que afirmaba: que
# `trayectoria_de_acercamiento` trae una entrada por cuadro, que cada una lleva
# el acercamiento Y la dispersión, y --- los dos polos que lo hacían una
# medición y no una comprobación de forma --- que encoger la nube del blanco
# alrededor de su propio centroide baja la dispersión por ese factor y deja el
# acercamiento quieto, y que trasladarla entera hace lo contrario.
#
# El operador retiró esa lectura: la imprimía la segunda línea del título de la
# primera columna, CREDA no imprime ninguna de las dos en la suya, y las dos
# columnas de CREDA llevan UNA lectura repetida. Sin ese rótulo la fórmula no
# tenía lector, así que se fue con él. Un test que la mantuviera viva la haría
# indistinguible de una lectura que alguien usa.
#
# Lo que la afirmación cubría y hoy no cubre nada: que un acercamiento sin su
# complemento no es evidencia. Nada MEDIDO se perdió --- esto nunca estuvo en el
# artefacto, sólo en un rótulo --- y un colapso se sigue viendo en los cuadros,
# que es donde se dibuja.


def test_the_seed_drawn_is_the_median_by_target_accuracy_and_not_the_best(
        registro):
    """The median and never the best: the best of N repetitions is an extreme of
    N draws, so a panel chosen that way describes the luckiest run.

    Driven over a record with three seeds built from the real one, because the
    pilot runs a single seed and on one seed every selection rule agrees. Both
    poles: the reading is moved so a DIFFERENT seed holds the median, and the
    answer has to move with it.
    """
    familia = toy_tables.familias(registro)[0]
    norma = toy_tables.normalizaciones(registro)[0]
    brazo = registro["flags"]["arms"][0]
    base = registro["cells"][toy_tables.SEP.join((familia, norma, brazo, "0"))]

    tres = copy.deepcopy(registro)
    tres["scale"]["report_seeds"] = [0, 1, 2]
    for semilla, acierto in ((0, 0.10), (1, 0.90), (2, 0.50)):
        celda = copy.deepcopy(base)
        celda["targetAccuracy"] = acierto
        tres["cells"][toy_tables.SEP.join(
            (familia, norma, brazo, str(semilla)))] = celda
    assert toy_run.semilla_a_dibujar(tres, familia, norma) == 2, (
        "la semilla elegida no es la mediana por acierto en el blanco")

    movido = copy.deepcopy(tres)
    movido["cells"][toy_tables.SEP.join(
        (familia, norma, brazo, "1"))]["targetAccuracy"] = 0.20
    assert toy_run.semilla_a_dibujar(movido, familia, norma) == 1, (
        "se movió qué semilla está en la mediana y la elección no se movió: no "
        "la está leyendo del registro")


def test_an_artefact_without_positions_is_refused_rather_than_drawn_empty(
        registro):
    """A frame drawn from a missing handle comes out EMPTY, and an empty frame
    reads exactly like a measurement that came out flat.

    Asked of `_fila_animada`, which is what builds a row of the grid now that
    the static sheet is retired: the refusal used to live in `figura_latente`
    and the two had their own copies of it, so the one this test did not drive
    was the one that could stop refusing.
    """
    familia = toy_tables.familias(registro)[0]
    norma = toy_tables.normalizaciones(registro)[0]
    viejo = copy.deepcopy(registro)
    viejo["families"][familia]["latent"] = {}
    with pytest.raises(KeyError, match="Toyset_Generator"):
        toy_run._fila_animada(viejo, familia, norma)


def test_the_latent_sheet_is_called_nowhere(cuaderno):
    """`figura_latente` is gone, and «gone» is asserted rather than assumed.

    The operator asked for it twice and a previous stretch kept it, reasoning
    that it should not re-decide an agreed figure. The GIF replaced it: it draws
    the same positions and carries the steps a photograph cannot. So what is
    pinned here is the ABSENCE -- in the notebook's code, in the module, and in
    the declaration of what draws -- because a call re-added by a later session
    would restore a second figure in a block the operator asked to hold one.

    Read in three places on purpose. The module is where it would come back;
    the notebook is where it would be called; and `report.figures` is where it
    would have to be declared for the verification to see it at all.
    """
    codigo = _codigo(cuaderno)
    assert "figura_latente" not in codigo, (
        "el cuaderno vuelve a llamar `figura_latente`: el bloque de cada "
        "familia lleva UNA figura y el cuadro animado ya la ocupa")
    fuente = Path(toy_run.__file__).read_text(encoding="utf-8")
    # Las dos últimas entraron el 2026-10-15: el operador retiró la lectura del
    # acercamiento junto con la línea del título que era su único lector, así
    # que la fórmula ya no vive en ninguna parte y «no vive» se afirma acá en
    # vez de suponerse. Volver a declararlas sería volver a poner un número en
    # un título que la forma de CREDA no tiene.
    for nombre in ("def figura_latente", "class FiguraLatente",
                   "def _panel_latente", "def _dibujar_panel",
                   "def acercamiento_y_dispersion", "def _lectura_de_cuadro",
                   "def trayectoria_de_acercamiento"):
        assert nombre not in fuente, (
            f"`toy_run` vuelve a declarar `{nombre}`: la lámina estática está "
            "retirada y la lectura del acercamiento se fue con el rótulo que "
            "la imprimía")
    assert not hasattr(toy_run, "figura_latente")
    assert not hasattr(toy_run, "trayectoria_de_acercamiento")
    # El control: el nombre que SÍ tiene que estar, para que las aserciones de
    # arriba no pasen por leer un archivo vacío o un módulo equivocado.
    assert "def animar_familia" in fuente
    assert hasattr(toy_run, "animar_familia")

    import MIL_CREDA_Benchmark as declaracion
    dibujan = declaracion.__benchmark__["report"]["figures"]
    assert "toy_run.animar_familia" in dibujan, dibujan
    assert "toy_run.figura_latente" not in dibujan, dibujan


def test_each_family_block_holds_exactly_one_figure_and_one_table(cuaderno,
                                                                  registro):
    """One figure call and one table call per family block, counted.

    This is the shape the removal was for, and counting is the only way to pin
    it: every other test here asks whether something is PRESENT, and a block
    that grew a second figure or a sixth table would satisfy all of them. The
    block boundaries come from the heading tree, so a call that drifted into the
    wrong block fails too.

    The declared tests are counted apart from the family table, because they
    answer different questions and CREDA's own block holds both: what may not
    come back is a second FIGURE or a second table of the run's readings.

    **The declared tests are TWO calls and the count says so.** CREDA's
    subsection is `tabla_sin_contra_con` and `tabla_tres_brazos`, so each is
    required exactly once -- a block that called one of them twice, or dropped
    one, fails here rather than passing a test that counted «the tests» as one.
    """
    bloques = {}
    actual = None
    for celda in cuaderno["cells"]:
        texto = "".join(celda["source"])
        if celda["cell_type"] == "markdown":
            for linea in texto.splitlines():
                if linea.startswith("## ") and not linea.startswith("###"):
                    nombre = linea[3:].strip()
                    actual = nombre if nombre in toy_tables.familias(registro) else None
                    if actual:
                        bloques[actual] = []
        elif actual is not None:
            bloques[actual].append(texto)

    assert sorted(bloques) == sorted(toy_tables.familias(registro)), (
        f"los bloques del cuaderno son {sorted(bloques)} y el registro declara "
        f"{sorted(toy_tables.familias(registro))}")
    for familia, celdas_bloque in bloques.items():
        codigo = "\n".join(celdas_bloque)
        figuras = (codigo.count("toy_run.animar_familia")
                   + codigo.count("toy_run.figura_latente")
                   + codigo.count("toy_bags.figura_familias"))
        assert figuras == 1, (
            f"el bloque de {familia} hace {figuras} llamadas de figura y tiene "
            "que hacer una: la lámina estática se retiró y los datos crudos se "
            "dibujan una sola vez, arriba")
        tabla = codigo.count("toy_tables.render_familia_escalera")
        assert tabla == 1, (
            f"el bloque de {familia} llama la tabla de la familia {tabla} "
            "veces y tiene que llamarla una")
        for nombre in ("render_familia_sin_contra_con",
                       "render_familia_tests"):
            tests = codigo.count(f"toy_tables.{nombre}")
            assert tests == 1, (
                f"el bloque de {familia} llama `{nombre}` {tests} veces y "
                "tiene que llamarlo una: los tests declarados son las DOS "
                "tablas de CREDA, cada una una vez")
        otras = [nombre for nombre in _por_familia()
                 if nombre not in ("render_familia_escalera",
                                   "render_familia_sin_contra_con",
                                   "render_familia_tests")
                 and nombre in codigo]
        assert otras == [], (
            f"el bloque de {familia} llama {otras}: un bloque lleva la tabla "
            "de la familia y las dos tablas de los tests declarados, y nada más")


def test_the_figure_cells_emitted_an_image_and_not_a_description(cuaderno,
                                                                 registro):
    """Read off what the cells EMITTED, not off what they call.

    A cell that displays a figure object where no image formatter is registered
    emits a line of text like `<Figure size 640x480>`: it ran, raised nothing,
    produced an output, and every reading of its source calls it green.

    Measured over the figure cells that remain -- one GIF per family plus the
    one data figure -- and the mime is not assumed: the GIF cells emit
    `image/gif` and the data figure emits `image/png`, so what is required is
    an image of SOME kind beside no object repr.

    **The cells are found by what SHOWS and no longer by what builds.** The two
    used to be the same cell; CREDA writes its GIF in the `## <familia>` cell
    and shows it in the one below, and this notebook now does the same, so a
    selector on `animar_familia` would pick the cell that writes the file and
    emits a line of text instead of the one that draws.
    """
    mostradas = []
    for indice, celda in enumerate(cuaderno["cells"]):
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        if not ("display(Image(filename=" in fuente
                or "figures.inline(" in fuente):
            continue
        imagenes = [salida for salida in celda.get("outputs", [])
                    if any(clave.startswith("image/")
                           for clave in (salida.get("data") or {}))]
        textos = "\n".join(
            "".join((salida.get("data") or {}).get("text/plain") or [])
            for salida in celda.get("outputs", []))
        mostradas.append((indice, len(imagenes),
                          "<Figure size" in textos or "object at 0x" in textos))
    # Una por familia más la de los datos, derivado del registro y no escrito:
    # un cuaderno que perdiera un bloque entero pasaría una cuenta escrita a
    # mano sin que nada lo dijera.
    esperadas = len(toy_tables.familias(registro)) + 1
    assert len(mostradas) == esperadas, (
        f"el cuaderno trae {len(mostradas)} celdas que dibujan y el registro "
        f"declara {len(toy_tables.familias(registro))} familias, así que "
        f"tendrían que ser {esperadas} (una por familia más la de los datos)")
    malas = [indice for indice, imagenes, descrito in mostradas
             if imagenes != 1 or descrito]
    assert malas == [], (
        f"las celdas {malas} dicen dibujar y lo que emitieron no es una "
        "imagen: o el cuaderno no se re-ejecutó, o la celda imprimió una "
        "descripción de la figura donde va la figura")



def test_no_measurement_is_typed_into_the_notebooks_prose(cuaderno):
    """A number written by hand cannot be recomputed, so it survives the run
    that contradicts it. This is the check `verify` runs on every notebook, run
    here so the defect is caught before a campaign pays for it -- and here it is
    the ONLY place it is run, because the declared renderers `verify` reads do
    not name `toy_tables`."""
    escritos = re.findall(r"\d+[.,]\d{2,}", _prosa(cuaderno))
    assert escritos == [], (
        f"la prosa del cuaderno trae {escritos} escrito a mano: eso no se puede "
        "volver a medir y sobrevive a la corrida que lo contradiga")


def test_every_table_the_notebook_shows_came_out_as_a_table(cuaderno):
    """A rendering that came out as a sentence explaining why it has nothing to
    show passes every reading of the source: the cell ran, raised nothing and
    emitted output. It is judged on the structure of what it emitted."""
    mostradas = []
    for indice, celda in enumerate(cuaderno["cells"]):
        if celda["cell_type"] != "code":
            continue
        if "toy_tables.render_" not in "".join(celda["source"]):
            continue
        emitido = "\n".join(
            "".join((salida.get("data") or {}).get("text/markdown") or [])
            for salida in celda.get("outputs", []))
        regla = re.search(r"^(?=[^\n]*\|)(?=[^\n]*--)[ \t|:-]+$", emitido, re.M)
        mostradas.append((indice, bool(regla), bool(MEDICION.search(emitido))))
    assert mostradas, "ninguna celda del cuaderno llama una tabla de `toy_tables`"
    vacias = [indice for indice, regla, numero in mostradas
              if not regla or not numero]
    assert vacias == [], (
        f"las celdas {vacias} llaman una tabla y lo que emitieron no tiene ni "
        "la regla de una tabla ni una medición: lo que el lector encuentra donde "
        "va el resultado es una frase")


def test_what_the_notebook_shows_is_what_the_artefact_holds(cuaderno, registro):
    """The join, read off the emitted output: the numbers a reader meets in the
    document are the numbers the artefact carries.

    Per family and asserted per COLUMN of what that family's own cell emitted.
    Searching the whole notebook for each value was measured not working: a
    value removed from one table was still found in another, so the document
    could disagree with the artefact in the one table that matters and the test
    stayed quiet. With one table per family that failure mode is wider, not
    narrower -- three tables now carry the same column titles.

    ONE grid now: CREDA's eight columns. Phase 1's selection-role grid was
    retired with the refusal that held it, so the half of this test that crossed
    it is gone too -- what it asserted (the emitted `fase 1 ELIGE` column equals
    the artefact's `phase1ValidAccuracy` per architecture) is no longer a claim
    about this document, because the document no longer renders that column.
    """
    rol = toy_tables.rol_publicado(registro)
    for familia in toy_tables.familias(registro):
        llamada = f'toy_tables.render_familia_escalera(REGISTRO, "{familia}")'
        emitido = None
        for celda_nb in cuaderno["cells"]:
            if (celda_nb["cell_type"] == "code"
                    and llamada in "".join(celda_nb["source"])):
                emitido = "\n".join(
                    "".join((salida.get("data") or {}).get("text/markdown") or [])
                    for salida in celda_nb.get("outputs", []))
                break
        assert emitido, (
            f"ninguna celda ejecutada del cuaderno muestra la tabla de "
            f"{familia}: o no está, o el cuaderno no se re-ejecutó")

        grillas = _grillas(emitido)
        assert len(grillas) == 1, (
            f"el cuaderno muestra {len(grillas)} grillas para {familia} y la "
            "celda de CREDA emite una sola")
        cabecera, filas = grillas[0]
        celdas = toy_tables.celdas(registro, familia)
        normas = toy_tables.normalizaciones(registro)
        con_adabn = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
        esperadas = 2 * len(normas) + len(con_adabn)
        assert len(filas) == esperadas, (
            f"el cuaderno muestra {len(filas)} filas para {familia} y el "
            f"artefacto da {esperadas} —{len(normas)} arquitecturas en el piso, "
            f"{len(con_adabn)} con escalón de AdaBN, {len(normas)} en el "
            "método—: el documento no se re-ejecutó contra este artefacto")

        metodos = _columna(cabecera, filas, "método")
        arquitecturas = _columna(cabecera, filas, "normalización")
        blanco = _columna(cabecera, filas, f"blanco · {rol}")
        # El número que la corrida publica, por arquitectura, en las filas del
        # método y leído de lo EMITIDO.
        publicado = {toy_tables.NOMBRE_FILA[celda["normalization"]]:
                     toy_tables._num(celda["targetAccuracy"])
                     for celda in celdas}
        del_metodo = [(arq, v) for arq, met, v
                      in zip(arquitecturas, metodos, blanco)
                      if met == toy_tables.ROTULO_ESCALON[
                          toy_tables.ESCALON_METODO]]
        assert del_metodo == [(arq, publicado[arq]) for arq, _ in del_metodo], (
            f"la columna del blanco de {familia} dice {del_metodo} y el "
            "artefacto dice otra cosa en las filas del método")



# ------------------------------------------ el cuadro animado, y sus dos columnas

def _md5(ruta) -> str:
    import hashlib
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def _toy_run_mutado(tmp_path, antes: str, despues: str, nombre: str):
    """Un segundo `toy_run`, cargado de una mutación del MISMO LARGO de su fuente.

    La misma disciplina que `tests/test_toy_run.py` describe y por las mismas
    tres razones medidas acá: un reemplazo del mismo largo deja el conteo de
    bytes quieto, así que lo que mide que la mutación llegó es el md5; y un
    `.pyc` en caché se valida contra (tamaño, mtime), que una reescritura del
    mismo tamaño dentro del mismo segundo ya hizo coincidir en este repositorio,
    así que el `__pycache__` se borra antes de cargar.
    """
    import importlib.util
    import shutil
    import sys as _sys

    original = Path(toy_run.__file__)
    fuente = original.read_text(encoding="utf-8")
    assert len(antes) == len(despues), (len(antes), len(despues))
    assert fuente.count(antes) == 1, (antes, fuente.count(antes))
    destino = tmp_path / f"{nombre}.py"
    destino.write_text(fuente.replace(antes, despues), encoding="utf-8")
    assert destino.stat().st_size == original.stat().st_size
    assert _md5(destino) != _md5(original), (
        "la mutación no movió el md5: el reemplazo no llegó al archivo")
    shutil.rmtree(tmp_path / "__pycache__", ignore_errors=True)
    spec = importlib.util.spec_from_file_location(nombre, str(destino))
    modulo = importlib.util.module_from_spec(spec)
    _sys.modules[nombre] = modulo
    try:
        spec.loader.exec_module(modulo)
    finally:
        _sys.modules.pop(nombre, None)
    return modulo


@pytest.fixture(scope="module")
def filas_animadas(registro) -> dict:
    """One family's rows, read once: the trajectory is the expensive part."""
    familia = toy_tables.familias(registro)[0]
    normas = toy_tables.normalizaciones(registro)
    return {"familia": familia,
            "filas": [toy_run._fila_animada(registro, familia, norma)
                      for norma in normas]}


def _ejes_de_la_grilla(fig, filas: int):
    """The grid's own axes, row-major, without the twin axes the loss panels add."""
    columnas = len(toy_run.GRID_COLUMNS)
    assert len(fig.axes) >= filas * columnas, (len(fig.axes), filas, columnas)
    return [fig.axes[r * columnas:(r + 1) * columnas] for r in range(filas)]


def _posiciones_dibujadas(a):
    """Every point a panel put on screen, as one sorted array.

    Read off the artists and not off the data handed in, which is the whole
    point: what the test is about is what the two panels DREW.
    """
    import numpy as np

    puntos = np.vstack([np.asarray(c.get_offsets()) for c in a.collections])
    return puntos[np.lexsort((puntos[:, 1], puntos[:, 0]))]


def test_the_two_embedding_columns_draw_one_cloud_and_differ_only_in_colour(
        registro, filas_animadas):
    """Columns one and two carry the SAME positions and differ in the colouring.

    Both halves, and the second is what keeps the first from being satisfied by
    two identical panels. The positions are compared as drawn -- every artist's
    offsets, sorted -- so a panel that recomputed its own would fail here even
    if the recomputation agreed by construction today. And the colour channels
    have to actually differ somewhere, which they do exactly where the head got
    a bag wrong; a frame where it got everything right would make the claim
    unmeasurable, so the test looks for a frame where they differ and says so
    when there is none.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    familia, filas = filas_animadas["familia"], filas_animadas["filas"]
    cuadros = toy_run.cuadros_comunes(len(fila["pasos"]) for fila in filas)
    discriminantes = 0
    for k in range(cuadros):
        fig = toy_run.figura_del_cuadro(registro, familia, filas, k)
        try:
            for fila, ejes in zip(filas, _ejes_de_la_grilla(fig, len(filas))):
                real, prediccion = _posiciones_dibujadas(ejes[0]), _posiciones_dibujadas(ejes[1])
                assert real.shape == prediccion.shape, (k, real.shape,
                                                        prediccion.shape)
                assert np.allclose(real, prediccion), (
                    f"las columnas 1 y 2 del cuadro {k} de {fila['normalizacion']} "
                    "no dibujan las mismas posiciones: la diferencia entre los "
                    "dos paneles deja de leerse como lo que el cabezal erró")
                por_clase = toy_run.colores_del_cuadro(fila, k, False)
                por_pred = toy_run.colores_del_cuadro(fila, k, True)
                if any(not np.array_equal(np.asarray(a), np.asarray(b))
                       for a, b in zip(por_clase, por_pred)):
                    discriminantes += 1
        finally:
            plt.close(fig)
    assert discriminantes > 0, (
        "el cabezal acierta todas las bolsas en todos los cuadros de esta "
        "familia, así que las dos columnas son la misma figura dos veces y la "
        "aserción de arriba no distingue nada")


def test_a_second_column_drawn_at_another_frame_is_caught(registro,
                                                          filas_animadas,
                                                          tmp_path):
    """Draw column two at frame zero and the two columns stop agreeing.

    The red for the test above. Without it, «las mismas posiciones» would be a
    property of the two panels sharing one call, and nothing would say what
    happens when they stop sharing it -- which is exactly the edit a later
    session makes while adding a channel to one of them.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mutado = _toy_run_mutado(tmp_path,
                             "_panel_embebido(ax[r, 1], fila, k, True)",
                             "_panel_embebido(ax[r, 1], fila, 0, True)",
                             "toy_run_columna_dos_en_el_cuadro_cero")
    familia, filas = filas_animadas["familia"], filas_animadas["filas"]
    cuadros = toy_run.cuadros_comunes(len(fila["pasos"]) for fila in filas)
    assert cuadros > 1, "con un solo cuadro esta mutación no cambia nada"
    k = cuadros - 1
    fig = mutado.figura_del_cuadro(registro, familia, filas, k)
    try:
        desiguales = [i for i, ejes in enumerate(_ejes_de_la_grilla(fig, len(filas)))
                      if not np.allclose(_posiciones_dibujadas(ejes[0]),
                                         _posiciones_dibujadas(ejes[1]))]
    finally:
        plt.close(fig)
    assert desiguales, (
        "la mutación no separó las dos columnas: o el blanco no se movió entre "
        "el cuadro 0 y el último, o las dos columnas no salen de donde este "
        "test cree, y entonces el verde de arriba no prueba nada")


def test_the_grid_has_one_row_per_declared_normalization_and_four_columns(
        registro, filas_animadas):
    """The row count is READ off the record, and the column count is the declared
    tuple.

    CREDA draws three rows and this package has backbones for two, so a grid
    written as three would draw a row over nothing and a grid written as two
    would be right today and silent the day a third backbone appears. The count
    comes from `normalizations`, and the absent one is named in the footer out
    of `normalizationsAbsent` -- which the test below reads.
    """
    import matplotlib.pyplot as plt

    normas = toy_tables.normalizaciones(registro)
    familia, filas = filas_animadas["familia"], filas_animadas["filas"]
    assert [fila["normalizacion"] for fila in filas] == normas
    assert len(toy_run.GRID_COLUMNS) == 4, toy_run.GRID_COLUMNS
    fig = toy_run.figura_del_cuadro(registro, familia, filas, 0)
    try:
        ejes = _ejes_de_la_grilla(fig, len(normas))
        assert len(ejes) == len(normas)
        # The grid's own axes are exactly rows x columns; whatever else is in
        # `fig.axes` is a twin an individual loss panel created.
        assert len([a for a in fig.axes if a.get_subplotspec() is not None
                    and a.get_subplotspec().get_gridspec().get_geometry()
                    == (len(normas), 4)]) >= len(normas) * 4
    finally:
        plt.close(fig)


def test_a_grid_whose_row_count_is_written_down_is_caught(registro,
                                                          filas_animadas,
                                                          tmp_path):
    """Write CREDA's three in and the grid stops matching the record.

    The red for «derived and not written». A literal three is exactly what a
    reader porting CREDA's figure writes first, and every other assertion in
    this file would stay green: the two rows that exist would still be drawn,
    correctly, with a third empty one under them.
    """
    import matplotlib.pyplot as plt

    normas = toy_tables.normalizaciones(registro)
    # **The literal that is now wrong is no longer three, and WHICH wrong
    # literal matters.** CREDA draws three rows and this package has three
    # backbones, so a written three would be right today: mutating to it would
    # make this test pass by agreeing with the code instead of by catching it.
    # And the two directions fail differently, so both are driven: a literal
    # LARGER than the record draws an empty row and raises nothing, which is the
    # silent failure this test exists for, and a literal SMALLER raises outright.
    assert len(normas) not in (2, 4), (
        f"el registro declara {len(normas)} normalizaciones y una de las "
        "mutaciones de abajo coincide con ese número, así que no cambia nada")
    familia, filas = filas_animadas["familia"], filas_animadas["filas"]

    de_mas = _toy_run_mutado(tmp_path,
                             "plt.subplots(len(filas), len(GRID_COLUMNS)",
                             "plt.subplots(4         , len(GRID_COLUMNS)",
                             "toy_run_cuatro_filas_escritas")
    fig = de_mas.figura_del_cuadro(registro, familia, filas, 0)
    try:
        geometrias = {a.get_subplotspec().get_gridspec().get_geometry()
                      for a in fig.axes if a.get_subplotspec() is not None}
        assert (4, 4) in geometrias, geometrias
        assert (len(normas), 4) not in geometrias, geometrias
    finally:
        plt.close(fig)

    de_menos = _toy_run_mutado(tmp_path,
                               "plt.subplots(len(filas), len(GRID_COLUMNS)",
                               "plt.subplots(2         , len(GRID_COLUMNS)",
                               "toy_run_dos_filas_escritas")
    with pytest.raises(IndexError):
        de_menos.figura_del_cuadro(registro, familia, filas, 0)


# `test_the_animation_says_which_rows_creda_draws_whichever_way_it_comes_out`
# SE BORRÓ acá el 2026-10-15, con las frases del pie que medía. Lo que
# afirmaba: que el pie del cuadro dice, salga como salga, cuántas de las filas
# que CREDA dibuja están presentes y en cuáles no hay escalón de AdaBN porque el
# backbone no guarda nada reestimable --- leído de `normalizationsAbsent` y
# nunca escrito --- y lo medía por los dos lados, con el registro real y con una
# copia que declara una ausencia inventada.
#
# El operador retiró esas frases: el pie de CREDA es UNA línea y ninguna de las
# dos tiene contraparte en ella. Las dos lecturas siguen medidas y siguen
# publicadas --- `toy_tables.ausentes` las lee del registro y las tablas del
# mismo cuaderno las imprimen ---; lo que dejó de haber es un pie que las repite
# bajo la figura.


#: Lo que el marco de CREDA dice en su suptitle, leído de SU módulo y no escrito
#: acá: familia, régimen, el paso de este cuadro sobre el techo declarado, y el
#: azar. El día que su figura cambie de forma, esto dice que la referencia se
#: movió en vez de seguir exigiendo la de ayer.
CUADRO_DE_CREDA = Path(toy_run.__file__).parents[1] / "CREDA" / "toy_plots.py"


def _cuerpo_de_cuadro_config() -> str:
    """La fuente de `CREDA.toy_plots.cuadro_config`, que es la referencia."""
    fuente = CUADRO_DE_CREDA.read_text(encoding="utf-8")
    inicio = fuente.index("def cuadro_config(")
    return fuente[inicio:fuente.index("def cuadros_comunes(", inicio)]


def _titulos_de_columna_de_creda() -> dict:
    """Los títulos que CREDA le pasa a sus dos paneles de pérdida, de SU fuente.

    Son literales adentro de `cuadro_config` y no constantes, así que se
    recomponen de la fuente en vez de importarse. Medir la referencia es lo que
    hace que «la columna dice lo que dice la de CREDA» sea una comprobación y no
    una copia que puede quedarse atrás.
    """
    pares = re.findall(r"'(fase [12] )' \+ chr\(183\) \+ '([^']*)'",
                       _cuerpo_de_cuadro_config())
    titulos = {a.strip()[-1]: a + chr(183) + b for a, b in pares}
    assert set(titulos) == {"1", "2"}, (titulos, "cambiaron los títulos de CREDA")
    return titulos


def _suptitle_de_creda() -> str:
    """El texto del `suptitle` de `CREDA.toy_plots.cuadro_config`, como fuente."""
    fuente = CUADRO_DE_CREDA.read_text(encoding="utf-8")
    inicio = fuente.index("def cuadro_config(")
    cuerpo = fuente[inicio:fuente.index("def cuadros_comunes(", inicio)]
    desde = cuerpo.index("fig.suptitle(")
    return cuerpo[desde:cuerpo.index("pie = (", desde)]


def test_the_frame_carries_credas_shape_and_not_this_experiments_additions(
        registro, filas_animadas):
    """El cuadro tiene la forma de la figura de CREDA, elemento por elemento.

    **Lo que el operador resolvió el 2026-10-15**, sobre ocho divergencias
    medidas entre el primer cuadro de `CREDA/.../paso_lunas_quieta.gif` y el de
    `toy_paso_lunas.gif`: la figura de este paquete lleva el marco de CREDA
    --- «mismos límites, mismas cosas» --- y donde una lectura propia no tiene
    contraparte en ese marco, se va. Cada aserción de abajo es una de esas ocho,
    y cada una se mide sobre la figura viva: el título, los rótulos de columna,
    el rótulo de fila, el eje x de los paneles de embedding, la cuarta columna
    del cuadro 0 y el pie.

    **La referencia del suptitle se LEE del módulo de CREDA** en vez de
    escribirse acá: una línea sola, con el paso de este cuadro sobre el techo
    declarado. Lo demás se mide sobre la figura porque es ahí donde está.

    Los controles que hacen falta para que el verde valga están adentro: que las
    dos columnas de embedding tengan primeras líneas DISTINTAS (si no, «la
    segunda línea coincide» lo cumpliría una figura con dos paneles iguales), y
    que esa segunda línea CAMBIE entre cuadros en alguna fila (si no, lo
    cumpliría una constante). El rojo del segundo lo da la mutación de abajo.
    """
    import matplotlib.pyplot as plt

    familia, filas = filas_animadas["familia"], filas_animadas["filas"]
    columnas = len(toy_run.GRID_COLUMNS)

    # 1 y 2 --- el suptitle: UNA línea, y el paso dicho como lo dice CREDA.
    referencia = _suptitle_de_creda()
    assert "chr(10)" not in referencia and "\\n" not in referencia, (
        "el suptitle de CREDA trae un salto de línea: la referencia de esta "
        "aserción cambió y hay que volver a leerla")
    assert "paso {" in referencia and " de {" in referencia, referencia

    fig = toy_run.figura_del_cuadro(registro, familia, filas, 0)
    try:
        titulo = fig._suptitle.get_text()
        assert "\n" not in titulo, (
            f"el suptitle trae más de una línea: {titulo!r}")
        assert re.fullmatch(
            rf"{re.escape(familia)} · fuente y cabeza QUIETAS · "
            rf"paso \d+ de \d+ · azar \d\.\d{{3}}", titulo), titulo
        # El sello retirado, nombrado por sus palabras y no por su función: lo
        # que no puede volver es la LÍNEA, la escriba quien la escriba.
        for palabra in ("escala «", "ENSAYO", "revisión", "dispositivo",
                        "cuadro ", "(cada "):
            assert palabra not in titulo, (palabra, titulo)

        ejes = [fig.axes[r * columnas:(r + 1) * columnas]
                for r in range(len(filas))]

        # 3 --- una lectura, repetida bajo las dos columnas.
        lineas_compartidas = []
        for fila, (real, prediccion, _, _) in zip(filas, ejes):
            primera_r, segunda_r = real.get_title().split("\n")
            primera_p, segunda_p = prediccion.get_title().split("\n")
            assert primera_r != primera_p, (
                "las dos columnas se llaman igual, así que «la segunda línea "
                "coincide» no distingue dos paneles de uno repetido")
            assert segunda_r == segunda_p, (
                f"{fila['normalizacion']}: las dos columnas llevan lecturas "
                f"distintas, {segunda_r!r} contra {segunda_p!r}")
            assert re.fullmatch(r"blanco \d\.\d{3} · fuente \d\.\d{3}",
                                segunda_r), segunda_r
            lineas_compartidas.append(segunda_r)

        # 6 --- los paneles de embedding no llevan rótulo de eje x.
        for fila, (real, prediccion, _, _) in zip(filas, ejes):
            assert real.get_xlabel() == "" and prediccion.get_xlabel() == "", (
                fila["normalizacion"], real.get_xlabel(),
                prediccion.get_xlabel())

        # 4 y 5 --- el rótulo de fila: el NOMBRE de la arquitectura, la línea
        # condicional de CREDA cuando corresponde, y el paso en que paró.
        #
        # **El lado esperado se DERIVA de `ASPECTO_MAX` y no se escribe.** Este
        # test afirmaba la forma cuadrada para las tres filas y se puso rojo
        # cuando el registro pasó a traer cajas por eje --- que es la rama
        # diseñada disparando, bajo el umbral de CREDA, y no una regresión. Un
        # test que fija un lado mide el registro del día en que se escribió; lo
        # que hay que fijar es la CONDICIONAL, en los dos sentidos, sobre lo
        # que el registro traiga.
        for fila, (real, *_) in zip(filas, ejes):
            cuadrada = fila["razonDeAspecto"] <= ASPECTO_MAX
            assert fila["cajaCuadrada"] is cuadrada, (
                fila["normalizacion"], fila["razonDeAspecto"], ASPECTO_MAX,
                "la fila declara una forma que su propia razón de aspecto "
                "contradice bajo el umbral de CREDA")
            condicional = ("" if cuadrada else
                           f"\n(ejes {fila['razonDeAspecto']:.0f}:1, "
                           "no cuadrados)")
            esperado = (f"{toy_tables.NOMBRE_FILA[fila['normalizacion']]}"
                        f"{condicional}\n"
                        f"paró en el paso {fila['pasoElegido']}")
            assert real.get_ylabel() == esperado, (real.get_ylabel(), esperado)

        # 7 --- la cuarta columna del cuadro 0 son EJES, no una caja de texto.
        for fila, (*_, fase_dos) in zip(filas, ejes):
            assert not fase_dos.texts, (
                fila["normalizacion"],
                [t.get_text() for t in fase_dos.texts],
                "el panel de la fase 2 del cuadro 0 dibuja texto adentro de los "
                "ejes: CREDA dibuja sus ejes y los deja salir vacíos")
            assert len(fase_dos.get_xticks()) > 0, "el eje x quedó sin marcas"
            assert fase_dos.get_xlabel() == "paso de la fase 2", (
                fase_dos.get_xlabel())
            assert fase_dos.get_legend() is not None, (
                "el panel de la fase 2 del cuadro 0 no trae leyenda: la de "
                "CREDA nombra sus curvas también cuando salen vacías")

        # Las dos columnas de pérdida dicen lo que dicen las de CREDA, leído
        # de su fuente: la primera con su paréntesis de la CE realizada debajo,
        # la segunda de UNA línea.
        de_creda = _titulos_de_columna_de_creda()
        assert toy_run.COL_PHASE_ONE == de_creda["1"], (
            toy_run.COL_PHASE_ONE, de_creda["1"])
        assert toy_run.COL_PHASE_TWO == de_creda["2"], (
            toy_run.COL_PHASE_TWO, de_creda["2"])
        for fila, (_, _, fase_uno, fase_dos) in zip(filas, ejes):
            primera, segunda = fase_uno.get_title().split("\n")
            assert primera == de_creda["1"], (primera, de_creda["1"])
            assert re.fullmatch(
                r"\(CE realizada: \[-?\d\.\d{4}, -?\d\.\d{4}\] = "
                r"-?\d+\.\d% del eje\)", segunda), segunda
            assert fase_dos.get_title() == de_creda["2"], fase_dos.get_title()

        # La caja CUADRADA es una caja dibujada cuadrada, no sólo unos límites
        # iguales: CREDA la fija con `set_aspect('equal')` bajo la misma
        # condición, y sin esa línea el rótulo de la fila estaba anunciando una
        # caja que nadie cuadró. Y la que NO es cuadrada tiene que quedar en
        # `auto`, o el rótulo estaría anunciando la condición contraria a la que
        # el panel dibuja.
        #
        # Los dos lados se derivan de `ASPECTO_MAX` sobre lo que el registro
        # traiga; ninguno se escribe. Antes acá se exigía el lado cuadrado para
        # las tres filas, de modo que el día en que una se pasó del umbral este
        # test se puso rojo por la rama que existe para ese día.
        for fila, (real, prediccion, _, _) in zip(filas, ejes):
            esperado = (1.0 if fila["razonDeAspecto"] <= ASPECTO_MAX
                        else "auto")
            for panel in (real, prediccion):
                assert panel.get_aspect() == esperado, (
                    fila["normalizacion"], fila["razonDeAspecto"],
                    panel.get_aspect(), esperado)

        # El eje de la fase 2 es UNO para las tres filas, derivado del registro,
        # y no recorta ninguna curva.
        compartido = toy_run.limite_fase2(filas)
        for fila, (*_, fase_dos) in zip(filas, ejes):
            assert fase_dos.get_ylim() == (0.0, compartido), (
                fila["normalizacion"], fase_dos.get_ylim(), compartido)
        mayor = max(float(fila["histFase2"][:, i].max())
                    for fila in filas
                    for i in [toy_run.TRAJECTORY_HISTORY_FIELDS.index(c)
                              for c in toy_run.SERIES_FASE_DOS])
        assert compartido >= mayor, (
            f"el eje compartido ({compartido}) queda por debajo de la serie más "
            f"alta ({mayor}): estaría recortando la curva que el panel existe "
            "para mostrar")
        assert compartido == pytest.approx(mayor * toy_run.HOLGURA_EJE), (
            compartido, mayor)

        # 8 --- el pie: una línea, con la forma del de CREDA.
        pies = [t.get_text() for t in fig.texts if t is not fig._suptitle]
        assert len(pies) == 1, pies
        assert "\n" not in pies[0], pies[0]
        for palabra in ("AdaBN", "ausente", "POR EJE", "están todas acá"):
            assert palabra not in pies[0], (palabra, pies[0])
    finally:
        plt.close(fig)

    # El control de la lectura compartida: tiene que MOVERSE entre cuadros en
    # alguna fila, o sería una constante que cumple la igualdad sola.
    cuadros = toy_run.cuadros_comunes(len(fila["pasos"]) for fila in filas)
    assert cuadros > 1, "con un solo cuadro este control no mide nada"
    fig = toy_run.figura_del_cuadro(registro, familia, filas, cuadros - 1)
    try:
        ultimas = [fig.axes[r * columnas].get_title().split("\n")[1]
                   for r in range(len(filas))]
    finally:
        plt.close(fig)
    assert ultimas != lineas_compartidas, (
        f"la lectura de las dos columnas es la misma en el cuadro 0 y en el "
        f"{cuadros - 1}: no está leyendo el cuadro")


def test_a_shared_reading_pinned_to_one_frame_is_caught(registro,
                                                        filas_animadas,
                                                        tmp_path):
    """Clavá la lectura en el cuadro 0 y el control de arriba se cae.

    El rojo de la última aserción del test anterior. Sin esto, «la lectura se
    mueve entre cuadros» sería una propiedad de los datos y no de que el panel
    la lea donde está parado --- que es justo lo que una sesión posterior rompe
    al cachear la lectura por fila.
    """
    import matplotlib.pyplot as plt

    mutado = _toy_run_mutado(tmp_path,
                             "{lectura_del_cuadro(fila, k)}",
                             "{lectura_del_cuadro(fila, 0)}",
                             "toy_run_lectura_clavada_en_el_cuadro_cero")
    familia, filas = filas_animadas["familia"], filas_animadas["filas"]
    columnas = len(toy_run.GRID_COLUMNS)
    cuadros = toy_run.cuadros_comunes(len(fila["pasos"]) for fila in filas)
    assert cuadros > 1, "con un solo cuadro esta mutación no cambia nada"

    lecturas = []
    for k in (0, cuadros - 1):
        fig = mutado.figura_del_cuadro(registro, familia, filas, k)
        try:
            lecturas.append([fig.axes[r * columnas].get_title().split("\n")[1]
                             for r in range(len(filas))])
        finally:
            plt.close(fig)
    assert lecturas[0] == lecturas[1], (
        "la mutación no clavó la lectura: o el cabezal decide igual en los dos "
        "cuadros de todas las filas, y entonces el control de arriba no prueba "
        "nada, o la lectura no sale de donde este test cree")


def test_the_row_label_says_when_the_box_stopped_being_square(registro,
                                                              filas_animadas):
    """El otro polo del rótulo de fila: la línea condicional de CREDA.

    `cuadro_config` agrega «(ejes N:1, no cuadrados)» entre el nombre y el paso
    cuando la caja dejó de ser cuadrada, porque ahí las distancias que se ven no
    son las que hay. Es SU línea y es condicional en SU figura, así que acá es
    condicional también.

    **Los dos lados se miden, y ninguno depende de lo que el registro del día
    traiga.** La versión anterior exigía que las tres filas fueran cuadradas y
    fabricaba la no cuadrada: eso valía mientras la grilla entera estuviera
    debajo de `ASPECTO_MAX`, y se puso rojo el día en que tres de nueve celdas
    lo pasaron --- con el registro intacto y la rama condicional haciendo
    exactamente lo que fue escrita para hacer. Así que acá se arma cada polo
    desde el que falte: el lado que el registro ya tiene se mide sobre sus
    propias filas, y el que no, se fabrica. El test no puede quedarse sin
    polos, y tampoco puede pasar midiendo uno solo.

    Se mueve la lectura de la fila, no el módulo: lo que se prueba es que el
    rótulo LEE `cajaCuadrada` y que el panel DIBUJA lo que el rótulo anuncia.
    """
    import matplotlib.pyplot as plt

    familia, filas = filas_animadas["familia"], filas_animadas["filas"]
    # Lo que el registro trae, derivado del umbral de CREDA y no de una lista.
    cuadradas = [f for f in filas if f["razonDeAspecto"] <= ASPECTO_MAX]
    por_eje = [f for f in filas if f["razonDeAspecto"] > ASPECTO_MAX]

    def _rotulo_esperado(fila: dict) -> str:
        condicional = ("" if fila["cajaCuadrada"] else
                       f"\n(ejes {fila['razonDeAspecto']:.0f}:1, no cuadrados)")
        return (f"{toy_tables.NOMBRE_FILA[fila['normalizacion']]}{condicional}\n"
                f"paró en el paso {fila['pasoElegido']}")

    def _medir(grilla: list) -> None:
        """Cada fila de la grilla, rótulo y dibujo, contra su propio lado."""
        fig = toy_run.figura_del_cuadro(registro, familia, grilla, 0)
        try:
            for r, fila in enumerate(grilla):
                eje = fig.axes[r * len(toy_run.GRID_COLUMNS)]
                assert eje.get_ylabel() == _rotulo_esperado(fila), (
                    fila["normalizacion"], eje.get_ylabel(),
                    _rotulo_esperado(fila))
                # Y el rótulo tiene que estar diciendo la verdad sobre el
                # DIBUJO, que es la mitad que un rótulo solo no prueba.
                assert eje.get_aspect() == (1.0 if fila["cajaCuadrada"]
                                            else "auto"), (
                    fila["normalizacion"], fila["cajaCuadrada"],
                    eje.get_aspect())
        finally:
            plt.close(fig)

    # El lado cuadrado. Si el registro no trajera ninguno, se fabrica, para que
    # el test no se quede sin ese polo cuando la grilla entera se pase.
    if not cuadradas:
        cuadradas = [dict(filas[0], cajaCuadrada=True, razonDeAspecto=1.3)]
    _medir(cuadradas)

    # Y el lado por eje, con el mismo criterio al revés.
    if not por_eje:
        por_eje = [dict(filas[0], cajaCuadrada=False, razonDeAspecto=7.7)]
    _medir(por_eje)

    # Los dos polos se midieron de verdad, y son DISTINTOS: sin esto el test
    # pasaría habiendo dibujado la misma grilla dos veces.
    assert cuadradas and por_eje
    assert all(f["cajaCuadrada"] for f in cuadradas), cuadradas
    assert not any(f["cajaCuadrada"] for f in por_eje), por_eje
    # Y la línea condicional aparece en uno y no en el otro, que es lo que la
    # hace condicional en vez de un renglón fijo.
    assert "no cuadrados" not in _rotulo_esperado(cuadradas[0])
    assert "no cuadrados" in _rotulo_esperado(por_eje[0])


def test_credas_literal_axis_would_clip_this_terms_own_curve(registro,
                                                             filas_animadas,
                                                             tmp_path):
    """Poné el `[0, 1]` literal de CREDA y la curva se recorta.

    El rojo de «el eje se DERIVA y no se copia». CREDA puede fijar `0` a `1`
    porque su término está mapeado a ese rango por las Ecs. 35-36; el de acá no,
    y medido sobre este registro llega a 1.2457. Sin este test, «derivado» sería
    una elección de implementación que una sesión posterior simplifica al
    literal de al lado, y lo que se vería después es una meseta donde hay un
    pico.

    Se mide sobre la familia cuya serie pasa de 1, elegida del registro y no
    escrita: si ninguna lo hiciera, el literal no recortaría nada y este test no
    probaría nada, y entonces lo dice en vez de pasar.
    """
    import matplotlib.pyplot as plt

    indices = [toy_run.TRAJECTORY_HISTORY_FIELDS.index(c)
               for c in toy_run.SERIES_FASE_DOS]
    por_familia = {}
    for fam in toy_tables.familias(registro):
        filas = [toy_run._fila_animada(registro, fam, norma)
                 for norma in toy_tables.normalizaciones(registro)]
        por_familia[fam] = (filas, max(float(f["histFase2"][:, i].max())
                                       for f in filas for i in indices))
    desbordan = {fam: alto for fam, (_, alto) in por_familia.items() if alto > 1.0}
    assert desbordan, (
        "ninguna serie de la fase 2 pasa de 1.0 en este registro, así que el "
        "literal de CREDA no recortaría nada y este test no mide la razón por "
        f"la que el eje se deriva: {[a for _, a in por_familia.values()]}")
    familia = sorted(desbordan)[0]
    filas, alto = por_familia[familia][0], desbordan[sorted(desbordan)[0]]

    mutado = _toy_run_mutado(tmp_path,
                             "a.set_ylim(0.0, tope_y)",
                             "a.set_ylim(0.0, 1.0000)",
                             "toy_run_eje_literal_de_creda")
    fig = mutado.figura_del_cuadro(registro, familia, filas, 0)
    try:
        topes = {fig.axes[r * len(toy_run.GRID_COLUMNS) + 3].get_ylim()[1]
                 for r in range(len(filas))}
    finally:
        plt.close(fig)
    assert topes == {1.0}, topes
    assert alto > 1.0, alto
    # Y el derivado, sobre la misma familia, la contiene.
    assert toy_run.limite_fase2(filas) > alto, (toy_run.limite_fase2(filas), alto)


def test_a_phase_two_panel_drawn_past_the_chosen_step_is_caught(registro,
                                                                filas_animadas,
                                                                tmp_path):
    """Dejá que el panel pase del paso elegido y su título deja de ser cierto.

    El rojo del techo que hace verdadero el título de CREDA. «hasta el paso que
    eligió» vale de un panel que PARA ahí; con `max` en vez de `min` el panel
    dibuja hasta el cuadro y la frase de arriba contradice la curva de abajo,
    que es exactamente lo que había antes de adoptar las dos cosas juntas.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    familia, filas = filas_animadas["familia"], filas_animadas["filas"]
    cuadros = toy_run.cuadros_comunes(len(fila["pasos"]) for fila in filas)
    k = cuadros - 1
    assert any(fila["pasos"][k] > fila["pasoElegido"] for fila in filas), (
        "ninguna fila corre más allá del paso que eligió, así que el techo no "
        "recorta nada acá y esta mutación no distingue nada")

    def _hasta_donde(modulo):
        fig = modulo.figura_del_cuadro(registro, familia, filas, k)
        try:
            columnas = len(toy_run.GRID_COLUMNS)
            fin = []
            for r in range(len(filas)):
                eje = fig.axes[r * columnas + 3]
                xs = [float(np.asarray(l.get_xdata()).max())
                      for l in eje.get_lines() if len(l.get_xdata())]
                fin.append(max(xs) if xs else 0.0)
            return fin
        finally:
            plt.close(fig)

    propio = _hasta_donde(toy_run)
    for fila, hasta in zip(filas, propio):
        assert hasta <= fila["pasoElegido"], (
            fila["normalizacion"], hasta, fila["pasoElegido"],
            "el panel dibuja más allá del paso que eligió y su título dice que "
            "no")
    mutado = _toy_run_mutado(tmp_path,
                             'tope = min(fila["pasos"][k], fila["pasoElegido"])',
                             'tope = max(fila["pasos"][k], fila["pasoElegido"])',
                             "toy_run_fase_dos_sin_techo")
    suelto = _hasta_donde(mutado)
    assert any(s > p for s, p in zip(suelto, propio)), (suelto, propio)


def test_the_gif_is_written_with_more_than_one_frame_and_its_declared_pause(
        registro, tmp_path):
    """The file on disk carries the frames and the pause, read BACK out of it.

    `imageio` does not refuse a duration it cannot honour -- it writes a file
    with no time assigned -- and the encoder merges identical consecutive
    frames, so a pause implemented as a repeated last frame disappears without
    anything failing. Both are read off the written file.
    """
    from PIL import Image as ImagenPIL

    familia = toy_tables.familias(registro)[0]
    animacion = toy_run.animar_familia(registro, familia, tmp_path,
                                       al_paso=lambda *_: None)
    assert animacion.ruta.exists()
    with ImagenPIL.open(animacion.ruta) as leido:
        assert leido.n_frames == animacion.cuadros
        assert leido.n_frames > 1, (
            "un GIF de un cuadro no es una animación, y nada en el archivo lo "
            "diría")
        leido.seek(leido.n_frames - 1)
        assert leido.info.get("duration") == toy_run.MS_PAUSE
        leido.seek(0)
        assert leido.info.get("duration") == toy_run.MS_FRAME


def test_an_animation_of_a_row_that_left_no_frame_is_refused(registro):
    """A row with zero frames is a defect of the artefact, never a short GIF."""
    assert toy_run.cuadros_comunes([5, 3, 7]) == 3
    with pytest.raises(ValueError):
        toy_run.cuadros_comunes([5, 0, 7])
    with pytest.raises(ValueError):
        toy_run.cuadros_comunes([])


def test_the_notebook_shows_the_animated_grid_inline_and_archives_it(cuaderno,
                                                                     registro):
    """Every family's block builds its GIF, prints where it archived it, and
    SHOWS it -- a cell that only printed the path would have reported a
    filename."""
    codigo = _codigo(cuaderno)
    for familia in toy_tables.familias(registro):
        assert f'toy_run.animar_familia(REGISTRO, "{familia}", FIGURAS)' in codigo, familia
    assert codigo.count("display(Image(filename=") == len(
        toy_tables.familias(registro))


def test_the_notebook_renders_the_intervention_ladder_for_every_family(
        cuaderno, registro):
    """The ladder is CREDA's `método` column, and it is in every family's block."""
    codigo = _codigo(cuaderno)
    for familia in toy_tables.familias(registro):
        assert (f'toy_tables.render_familia_escalera(REGISTRO, "{familia}")'
                in codigo), familia


# ------------------------------------------- la tabla de CREDA, fila por fila
#
# `CREDA/Notebooks/Toyset_Plot.ipynb` cell 16 renders `tabla_dataset` with eight
# columns and seven rows, and every test in this section is about one property of
# THAT shape. It is the shape the operator asked for, column for column, which is
# why the header is written out here once and compared whole rather than probed
# for a column at a time: a table missing one column, or carrying an extra, or
# carrying the eight in another order, reads as CREDA's beside CREDA's and is
# not.


def _encabezado_de_creda(registro: dict) -> list[str]:
    """CREDA's own eight headers, in CREDA's own order.

    The two role words are the ones THIS record partitions on and not CREDA's
    literal `test`: a header naming a role this split does not carry would be a
    header nobody measured, which is the defect `rol_publicado` exists to
    refuse. Everything else is CREDA's, verbatim.
    """
    rol = toy_tables.rol_publicado(registro)
    return ["método", "normalización", "época fuente", "paso que eligió",
            f"fuente · {rol}", f"blanco · {rol}", "vigía: fuente conservada",
            f"blanco pico · {rol} (oráculo)"]


def test_the_family_table_has_exactly_credas_eight_columns_in_credas_order(
        registro):
    """Eight headers, CREDA's, in CREDA's order, in every family's block.

    Compared whole and as a SEQUENCE. A set comparison would pass a reordering,
    and a per-column `in` would pass a table that carried CREDA's eight plus
    four of its own -- which is precisely what this table was before: seventeen
    columns, of which eight were these.
    """
    esperado = _encabezado_de_creda(registro)
    assert len(esperado) == len(set(esperado)) == 8, esperado
    for familia in toy_tables.familias(registro):
        cabecera, _ = _parsear(
            toy_tables.render_familia_escalera(registro, familia))
        assert cabecera == esperado, (
            f"la tabla de {familia} trae {cabecera} y la de CREDA trae "
            f"{esperado}: ocho columnas, en ese orden")


def _filas_por_metodo(registro: dict, familia: str) -> dict:
    """What architecture each rung got a row for, read off the rendered grid."""
    cabecera, filas = _parsear(
        toy_tables.render_familia_escalera(registro, familia))
    metodos = _columna(cabecera, filas, "método")
    arquitecturas = _columna(cabecera, filas, "normalización")
    salida: dict[str, list[str]] = {}
    for metodo, arquitectura in zip(metodos, arquitecturas):
        salida.setdefault(metodo, []).append(arquitectura)
    return salida


def test_the_floor_rung_has_one_row_per_declared_architecture(registro):
    """`.predict` runs on every network, so it gets a row for every one.

    Read as the sequence of architectures under that rung's own label, in the
    record's order. CREDA's own three `.predict` rows are the three starting
    points the two rungs below them are read against, so a missing one takes a
    comparison away rather than a line.
    """
    esperado = [toy_tables.NOMBRE_FILA[norma]
                for norma in toy_tables.normalizaciones(registro)]
    assert len(esperado) > 1, (
        "el registro declara una sola arquitectura, así que «una fila por "
        "arquitectura» no distingue nada: revisar el artefacto")
    for familia in toy_tables.familias(registro):
        por_metodo = _filas_por_metodo(registro, familia)
        for escalon in (toy_tables.ESCALON_FASE_1, toy_tables.ESCALON_METODO):
            rotulo = toy_tables.ROTULO_ESCALON[escalon]
            assert por_metodo.get(rotulo) == esperado, (
                f"en {familia}, el escalón {escalon} trae "
                f"{por_metodo.get(rotulo)} y las arquitecturas declaradas son "
                f"{esperado}")


def test_adabn_has_one_row_and_it_is_the_architecture_that_carries_buffers(
        registro):
    """One AdaBN row, for the architecture whose backbone holds re-estimable
    buffers, and that architecture is identified by MEASUREMENT.

    `adabnLayers` is what the run found to re-estimate, so the rung follows
    that count and never the layer's name. Both halves are required: the row
    exists for every architecture the record measured buffers in, and for no
    other.
    """
    for familia in toy_tables.familias(registro):
        con_buffers = [norma for norma in toy_tables.normalizaciones(registro)
                       if all(celda["adabnLayers"] > 0
                              for celda in toy_tables.celdas(registro, familia)
                              if celda["normalization"] == norma)]
        assert len(con_buffers) == 1, (
            f"el artefacto trae {con_buffers} con capas reestimables en "
            f"{familia}; este test pide exactamente una para distinguir «una "
            "fila de AdaBN» de «una por arquitectura»")
        por_metodo = _filas_por_metodo(registro, familia)
        rotulo = toy_tables.ROTULO_ESCALON[toy_tables.ESCALON_ADABN]
        assert por_metodo.get(rotulo) == [
            toy_tables.NOMBRE_FILA[norma] for norma in con_buffers], (
            f"en {familia}, AdaBN trae las filas {por_metodo.get(rotulo)} y el "
            f"registro mide buffers en {con_buffers}")
        assert (toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
                == con_buffers)


def test_an_architecture_with_no_reestimable_buffers_contributes_no_adabn_row(
        registro):
    """Flip the MEASUREMENT, and the AdaBN row moves with it.

    This is the red for the test above and it is driven on the record rather
    than on a name: the architecture that carries buffers has its count zeroed
    and loses its AdaBN row, and one that carries none is given a count and an
    accuracy and gains one. A renderer that decided by the layer's name would
    print the same table under both mutations.

    The second half is what makes the first mean anything. With only the
    zeroing, «no AdaBN row» would be satisfied by a renderer that never emits
    one at all.
    """
    familia = toy_tables.familias(registro)[0]
    con_buffers = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
    sin_buffers = [norma for norma in toy_tables.normalizaciones(registro)
                   if norma not in con_buffers]
    assert con_buffers and sin_buffers, (con_buffers, sin_buffers)
    rotulo = toy_tables.ROTULO_ESCALON[toy_tables.ESCALON_ADABN]

    apagado = copy.deepcopy(registro)
    for clave, celda in apagado["cells"].items():
        if celda["normalization"] in con_buffers:
            celda["adabnLayers"] = 0
            celda["adabnTargetAccuracy"] = None
    assert toy_tables.arquitecturas_con_escalon_adabn(apagado, familia) == []
    cabecera, filas = _parsear(
        toy_tables.render_familia_escalera(apagado, familia))
    assert cabecera == _encabezado_de_creda(registro), (
        "la mutación movió las columnas y tendría que haber movido sólo las "
        "filas")
    assert rotulo not in _columna(cabecera, filas, "método"), (
        "se puso en cero el conteo de capas reestimables de todas las "
        "arquitecturas y AdaBN sigue teniendo fila: la fila no sale de la "
        "medición")
    assert len(filas) == 2 * len(toy_tables.normalizaciones(registro))

    encendido = copy.deepcopy(registro)
    for celda in encendido["cells"].values():
        if celda["normalization"] in sin_buffers:
            celda["adabnLayers"] = 1
            celda["adabnNormLayers"] = max(celda["adabnNormLayers"], 1)
            celda["adabnTargetAccuracy"] = celda["predictTargetAccuracy"]
    assert (toy_tables.arquitecturas_con_escalon_adabn(encendido, familia)
            == toy_tables.normalizaciones(registro))
    cabecera, filas = _parsear(
        toy_tables.render_familia_escalera(encendido, familia))
    ganadas = [arq for met, arq in zip(_columna(cabecera, filas, "método"),
                                       _columna(cabecera, filas,
                                                "normalización"))
               if met == rotulo]
    assert ganadas == [toy_tables.NOMBRE_FILA[norma] for norma
                       in toy_tables.normalizaciones(registro)], (
        "se le dieron capas reestimables a toda la grilla y AdaBN no ganó "
        "filas: la fila no sale de la medición")


def test_a_record_that_declares_buffers_and_reports_no_adabn_accuracy_is_refused(
        registro):
    """The two record fields have to agree, or the row would be a dash.

    A dash under `blanco` on an AdaBN row says «AdaBN was measured and came out
    empty», which is a different fact from «AdaBN has no rung here» -- and the
    second is what the ABSENCE of the row says. So a record whose layer count
    and whose accuracy disagree is refused rather than rendered.
    """
    familia = toy_tables.familias(registro)[0]
    roto = copy.deepcopy(registro)
    for celda in roto["cells"].values():
        if celda["adabnLayers"] > 0:
            celda["adabnTargetAccuracy"] = None
    with pytest.raises(ValueError, match="acierto de AdaBN"):
        toy_tables.render_familia_escalera(roto, familia)


def test_the_header_line_takes_its_seed_count_and_chance_from_the_record(
        registro):
    """CREDA's header line, and every part of it moves when the record moves.

    Three claims, because the line carries three readings and a literal in any
    of them would be right today: the seed count, the chance level, and the
    role. The dispersion clause is the fourth and is driven by its own test
    below.
    """
    escala = registro["scale"]
    familia = toy_tables.familias(registro)[0]
    linea = toy_tables.encabezado_familia(registro, familia)
    assert linea.startswith(f"**{familia} · ")
    assert f"azar {toy_tables._num(1.0 / escala['classes'])}" in linea
    assert f"{toy_tables.rol_publicado(registro)}, sólo reporte**" in linea

    mas_clases = copy.deepcopy(registro)
    mas_clases["scale"]["classes"] = escala["classes"] + 1
    movida = toy_tables.encabezado_familia(mas_clases, familia)
    assert (f"azar {toy_tables._num(1.0 / (escala['classes'] + 1))}" in movida
            and movida != linea), (
        "se movió el número de clases y el azar del encabezado no se movió: "
        "está escrito y no calculado")

    mas_semillas = copy.deepcopy(registro)
    mas_semillas["scale"]["report_seeds"] = list(escala["report_seeds"]) + [7]
    con_dos = toy_tables.encabezado_familia(mas_semillas, familia)
    assert f"· {len(escala['report_seeds']) + 1} semillas ·" in con_dos, con_dos
    assert "· 1 semilla ·" in linea, (
        "el artefacto declara más de una semilla de reporte, así que este test "
        "no distingue el conteo derivado del literal: revisar el artefacto")


def test_at_one_report_seed_the_table_prints_the_mean_and_claims_no_spread(
        registro):
    """`± 0.0000` would claim a dispersion was measured and came out zero.

    What happened at one seed is that nothing was measured, so the reduction
    prints the mean alone and the header says how many seeds it ran over. Both
    halves: the sign is absent at one seed and present at two, over the same
    values, so its absence is a reading of the count and not of the renderer.
    """
    assert len(registro["scale"]["report_seeds"]) == 1
    familia = toy_tables.familias(registro)[0]
    salida = toy_tables.render_familia_escalera(registro, familia)
    assert "±" not in salida, (
        "con una sola semilla de reporte la tabla imprime un ±: eso afirma una "
        "dispersión medida que vale cero, y lo que pasó es que no se midió "
        "ninguna")
    assert "sin dispersión" in salida

    dos = copy.deepcopy(registro)
    dos["scale"]["report_seeds"] = [0, 7]
    for clave in list(dos["cells"]):
        familia_, norma, brazo, _ = clave.split(toy_tables.SEP)
        gemela = copy.deepcopy(dos["cells"][clave])
        gemela["seed"] = 7
        # Un valor movido en la celda gemela, para que la desviación de la
        # pareja no sea cero: con dos celdas idénticas el `±` saldría `0.0000`
        # y el polo de este test no distinguiría «imprime el signo» de «no lo
        # imprime».
        gemela["sourceAccuracy"] = min(1.0, gemela["sourceAccuracy"] + 0.25)
        dos["cells"][toy_tables.SEP.join((familia_, norma, brazo, "7"))] = gemela
    con_dos = toy_tables.render_familia_escalera(dos, familia)
    assert "±" in con_dos, (
        "con dos semillas de reporte la tabla no imprime ninguna dispersión: "
        "entonces su ausencia a una semilla no es una lectura del conteo")
    assert "media ± desvío" in con_dos


def test_the_oracle_column_is_the_peak_of_the_series_the_run_records(registro):
    """The peak fills on the rung that ran a phase 2 and dashes on the two that
    did not, and BOTH halves are read off the artefact rather than from a
    literal.

    The column reads `SERIE_BLANCO` by name and `toy_run.run_cell` now records
    it -- one reading of the target's TEST role per phase-2 step -- so what used
    to be a dash in every row is the maximum of that series. The dash that
    remains is CREDA's own (`toy_plots.tabla_dataset`): `.predict` and AdaBN are
    read at the step-0 checkpoint and have no trajectory, so the method's peak
    printed on their row would attribute this run's best pass to an
    intervention that took no step.

    Three poles, because the first alone proves nothing:

    - the dash count is DERIVED from the ladder, so a column that dashed one row
      too many or too few is red;
    - the peak is the series' own maximum, recomputed here from the record;
    - with the field REMOVED from a copy the whole column dashes again, which is
      what keeps the fill from being a number the renderer invents.
    """
    assert toy_tables.SERIE_BLANCO not in toy_run.TRAJECTORY_HISTORY_FIELDS, (
        "the per-step trace carries the series as well, so it travels twice "
        "and two renderings of one measurement can drift apart")
    series_por_celda = {clave: celda[toy_tables.SERIE_BLANCO]
                        for clave, celda in registro["cells"].items()
                        if toy_tables.SERIE_BLANCO in celda}
    assert len(series_por_celda) == len(registro["cells"]), (
        f"solo {len(series_por_celda)} de {len(registro['cells'])} celdas del "
        f"artefacto traen {toy_tables.SERIE_BLANCO}: la corrida que lo escribe "
        "no llegó a este registro")
    assert all(serie for serie in series_por_celda.values()), (
        "alguna celda trae la serie vacía, así que su máximo no existe")

    rol = toy_tables.rol_publicado(registro)
    titulo = f"blanco pico · {rol} (oráculo)"
    familia = toy_tables.familias(registro)[0]
    cabecera, filas = _parsear(
        toy_tables.render_familia_escalera(registro, familia))
    columna = _columna(cabecera, filas, titulo)
    metodo = _columna(cabecera, filas, "método")

    # Qué filas tienen que traer número y cuáles raya, DERIVADO de la escalera:
    # la del método y ninguna otra.
    rotulo_metodo = toy_tables.ROTULO_ESCALON[toy_tables.ESCALON_METODO]
    con_numero = [valor for valor, quien in zip(columna, metodo)
                  if quien == rotulo_metodo]
    con_raya = [valor for valor, quien in zip(columna, metodo)
                if quien != rotulo_metodo]
    assert con_numero and con_raya, (columna, metodo)
    assert set(con_raya) == {toy_tables.NADA}, (
        f"una fila que no corrió fase 2 imprime un pico: {con_raya}")
    assert toy_tables.NADA not in con_numero, (
        f"la fila del método sigue con raya habiendo serie en el registro: "
        f"{con_numero}")

    # Y el valor: el máximo de la serie de cada celda de esa arquitectura,
    # recompuesto acá desde el registro y no leído de la tabla.
    for fila in _filas_del_metodo(registro, familia):
        picos = [max(celda[toy_tables.SERIE_BLANCO]) for celda in fila["celdas"]]
        assert toy_tables._media_desvio(picos) in con_numero, (
            fila["norma"], toy_tables._media_desvio(picos), con_numero)

    # El polo que mantiene la raya como lectura y no como literal: sin el campo,
    # la columna entera vuelve a la raya.
    sin_serie = copy.deepcopy(registro)
    for celda in sin_serie["cells"].values():
        celda.pop(toy_tables.SERIE_BLANCO, None)
    cabecera, filas = _parsear(
        toy_tables.render_familia_escalera(sin_serie, familia))
    assert set(_columna(cabecera, filas, titulo)) == {toy_tables.NADA}, (
        "se le quitó la serie al registro y la columna siguió imprimiendo un "
        "pico, así que el número no viene del campo que declara leer")


def _filas_del_metodo(registro: dict, familia: str) -> list:
    """The ladder's rows for the one rung that ran a phase 2."""
    return [fila for fila in toy_tables._filas_escalera(registro, familia)
            if fila["escalon"] == toy_tables.ESCALON_METODO]


def test_the_eight_columns_name_the_record_fields_they_read(registro):
    """Every column declares what it reads, so what a table computes with is
    readable without reading its lambdas.

    Asked of the built columns and not of the output. `phase1ValidAccuracy` is
    deliberately NOT expected here: it was read by the retired selection-role
    grid, and after that retirement no column of this block reads it -- the
    field is still in the record, it is simply not rendered. The one column with
    no `lee` is `método`, whose value is the ladder's own label and no reading of
    the record.
    """
    campos = {campo
              for grupos in [_grupos_de_la_escalera(registro)]
              for _, columnas, _ in grupos
              for columna in columnas
              for campo in columna.lee}
    assert "sourceAccuracy" in campos
    assert "phase1ValidAccuracy" not in campos, (
        "una columna volvió a leer la lectura del rol de selección, que el "
        "operador retiró de este bloque")
    assert "targetAccuracy" in campos and "watchmanRestored" in campos
    assert toy_tables.SERIE_BLANCO in campos, (
        "la columna del oráculo no declara el campo que lee, así que su raya no "
        "es la lectura de una ausencia declarada")


def _grupos_de_la_escalera(registro: dict) -> list:
    """The groups `render_familia_escalera` hands the builder, captured.

    Captured by watching the one builder rather than rebuilt here: a copy of
    the column list written in this file would verify this file against itself.
    """
    capturados: list = []
    real = toy_tables._bloque

    def espia(grupos):
        capturados.extend(grupos)
        return real(grupos)

    toy_tables._bloque = espia
    try:
        toy_tables.render_familia_escalera(registro,
                                           toy_tables.familias(registro)[0])
    finally:
        toy_tables._bloque = real
    return capturados


# --------------------------------- la escalera, derivada y no escrita como tres

def _modulo_mutado(ruta: Path, tmp_path, antes: str, despues: str, nombre: str):
    """Any module of this package, loaded from a SAME-LENGTH mutation.

    `_toy_run_mutado` above is this, bound to one file; the three measured
    reasons for the shape are written there and hold here unchanged. Generalized
    because the derivation under test lives in `toy_tables` and the figure's
    lives in `toy_run`, and two copies of a loader are two things that drift.
    """
    import importlib.util
    import shutil
    import sys as _sys

    fuente = ruta.read_text(encoding="utf-8")
    assert len(antes) == len(despues), (len(antes), len(despues))
    assert fuente.count(antes) == 1, (antes, fuente.count(antes))
    destino = tmp_path / f"{nombre}.py"
    destino.write_text(fuente.replace(antes, despues), encoding="utf-8")
    assert destino.stat().st_size == ruta.stat().st_size
    assert _md5(destino) != _md5(ruta), (
        "la mutación no movió el md5: el reemplazo no llegó al archivo")
    shutil.rmtree(tmp_path / "__pycache__", ignore_errors=True)
    spec = importlib.util.spec_from_file_location(nombre, str(destino))
    modulo = importlib.util.module_from_spec(spec)
    _sys.modules[nombre] = modulo
    try:
        spec.loader.exec_module(modulo)
    finally:
        _sys.modules.pop(nombre, None)
    return modulo


def _filas_de_creda(registro: dict, familia: str) -> int:
    """How many rows CREDA's shape gives this record, derived.

    Two bands of one row per declared architecture -- the floor and the method --
    plus one AdaBN row per architecture the record measured buffers in. Written
    once here because three tests compare against it, and three copies of an
    arithmetic are three things that drift.
    """
    return (2 * len(toy_tables.normalizaciones(registro))
            + len(toy_tables.arquitecturas_con_escalon_adabn(registro, familia)))


def test_the_tables_row_count_follows_the_record_and_is_not_written_down(
        registro):
    """Drop a declared normalization from a COPY and the table has to shrink.

    The row count is two bands of one row per declared architecture plus AdaBN's
    own, and the architectures come from the record's own `normalizations`.
    Nothing may write the count: a literal seven would be RIGHT today, which is
    exactly what makes it dangerous -- the count here was a literal nine over a
    grid that then changed shape.

    Driven as a shrink and not a grow because a grow would need cells the
    artefact does not hold, and `claves_de_celda` refuses those by design: the
    refusal is a different claim, pinned by its own test above.
    """
    normas = toy_tables.normalizaciones(registro)
    assert len(normas) > 2, (
        f"el registro declara {len(normas)} normalizaciones; con menos de tres "
        "este test no distingue un conteo derivado de uno escrito")
    familia = toy_tables.familias(registro)[0]

    _, filas = _parsear(toy_tables.render_familia_escalera(registro, familia))
    assert len(filas) == _filas_de_creda(registro, familia), (
        len(filas), len(normas))

    menos = copy.deepcopy(registro)
    sobrante = normas[-1]
    menos["flags"]["normalizations"] = {
        k: v for k, v in menos["flags"]["normalizations"].items()
        if k != sobrante}
    cabecera, cortas = _parsear(
        toy_tables.render_familia_escalera(menos, familia))
    assert len(cortas) == _filas_de_creda(menos, familia), (
        f"se sacó `{sobrante}` de las normalizaciones declaradas y la tabla "
        f"siguió con {len(cortas)} filas: el conteo no sale del registro")
    assert len(cortas) < len(filas), (len(cortas), len(filas))
    assert (toy_tables.NOMBRE_FILA[sobrante]
            not in _columna(cabecera, cortas, "normalización"))

    # Y la tabla A de los tests declarados, por el mismo camino: sus filas son
    # una por arquitectura declarada más la que promedia. La tabla B no entra
    # acá --- sus filas son los pares de la escalera, que no dependen de
    # cuántas arquitecturas hay, y de qué depende tiene su propio test.
    _, tabla_a = _parsear(
        toy_tables.render_familia_sin_contra_con(registro, familia))
    assert len(tabla_a) == len(normas) + 1
    _, tabla_a_corta = _parsear(
        toy_tables.render_familia_sin_contra_con(menos, familia))
    assert len(tabla_a_corta) == len(normas), (
        f"se sacó `{sobrante}` de las normalizaciones declaradas y la tabla A "
        f"siguió con {len(tabla_a_corta)} filas: el conteo no sale del registro")


def test_a_table_whose_normalizations_are_written_down_is_caught(registro,
                                                                 tmp_path):
    """Write the pair in and the table stops following the record.

    The red for the test above, and the mutation is the exact defect: a pair of
    row names written where the record's list belongs. Under it the cell walk
    loses one whole architecture while the declared list still names three, so
    the table is asked for a declared architecture it has no cell of -- and it
    REFUSES, naming the architecture, instead of rendering its remaining rows
    correctly and saying nothing about the one that went.

    Every other assertion in this file stays green under the mutation if the
    refusal is removed: the rows that remain are correct and in order.
    """
    normas = toy_tables.normalizaciones(registro)
    assert set(normas) != {"ninguna", "batchnorm"}, (
        "el registro declara justo el par que la mutación escribe, así que no "
        "cambia nada y este test no mide nada")
    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        'for normalizacion in banderas["normalizations"]:',
        'for normalizacion in ("ninguna", "batchnorm")  :',
        "toy_tables_con_el_par_escrito")
    familia = toy_tables.familias(registro)[0]
    assert mutado.normalizaciones(registro) == normas, (
        "la mutación movió la lectura de las normalizaciones declaradas; "
        "tendría que haber movido sólo el recorrido de las celdas")
    with pytest.raises(KeyError, match="no tiene ninguna celda suya"):
        mutado.render_familia_escalera(registro, familia)
    # Y el control de que la mutación es la que recortó el recorrido: las dos
    # arquitecturas que el par escrito sí nombra siguen teniendo celdas.
    assert len(mutado.celdas(registro, familia)) == 2 * len(
        _escala_semillas(registro))


def _escala_semillas(registro: dict) -> list:
    return list(registro["scale"]["report_seeds"])


def test_the_ladder_rungs_are_the_declared_ones_in_the_declared_order(registro,
                                                                      tmp_path):
    """The three rungs, in `ESCALERA`'s order, under CREDA's own labels.

    The ORDER is the reading: a ladder read from the top says nothing, because
    the question each rung answers is what the one below it left on the table.
    So the `método` column is asserted as a SEQUENCE against the declaration --
    the floor's band, then AdaBN's, then the method's -- which catches a
    reversal, a sort, an omission and a regrouping.

    The red is the quietest defect the column has: one that names something
    ELSE. Under the mutation the column is populated, has the right number of
    rows, keeps the architecture beside it, keeps every number in the row, and
    prints the record's field names where the rung's label belongs. No other
    assertion in this file sees it.
    """
    nombres = [escalon for escalon, _, _ in toy_tables.ESCALERA]
    rotulos = [rotulo for _, rotulo, _ in toy_tables.ESCALERA]
    campos = [campo for _, _, campo in toy_tables.ESCALERA]
    assert len(nombres) == 3 and nombres != sorted(nombres), (
        "los escalones están en orden alfabético, así que una inversión sería "
        "indistinguible de un orden declarado y este test no mediría nada")
    assert set(rotulos).isdisjoint(campos), (rotulos, campos)
    assert len(set(rotulos)) == 3, rotulos

    for familia in toy_tables.familias(registro):
        con_adabn = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
        cuantas = {toy_tables.ESCALON_ADABN: len(con_adabn)}
        esperado = [rotulo for escalon, rotulo, _ in toy_tables.ESCALERA
                    for _ in range(cuantas.get(
                        escalon, len(toy_tables.normalizaciones(registro))))]
        cabecera, filas = _parsear(
            toy_tables.render_familia_escalera(registro, familia))
        assert _columna(cabecera, filas, "método") == esperado, familia

    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        '        Columna("método", lambda r: ROTULO_ESCALON[r["escalon"]]),',
        '        Columna("método", lambda r: r["campo"]),                  ',
        "toy_tables_escalon_mal_nombrado")
    familia = toy_tables.familias(registro)[0]
    cabecera, filas = _parsear(mutado.render_familia_escalera(registro, familia))
    mal = _columna(cabecera, filas, "método")
    assert set(mal) == set(campos), mal
    assert mal != _columna(*_parsear(
        toy_tables.render_familia_escalera(registro, familia)), "método")
    # Y el control de que la mutación tocó SÓLO esa columna: la tabla sigue
    # trayendo las mismas filas y los mismos aciertos.
    rol = toy_tables.rol_publicado(registro)
    assert (_columna(cabecera, filas, f"blanco · {rol}")
            == _columna(*_parsear(
                toy_tables.render_familia_escalera(registro, familia)),
                f"blanco · {rol}"))


def test_a_table_whose_rung_bands_are_crossed_is_caught(registro, tmp_path):
    """The red for the two shape tests above: invert the guard that decides
    which rung reaches which architecture.

    Under the mutation the table still has CREDA's eight columns and every
    number in every row is still the right number for the architecture beside
    it -- and the BANDS are swapped: the floor and the method reach only the architecture
    AdaBN has a rung on, and AdaBN reaches the other two. Nothing else in this
    file sees it.

    The inversion rather than a deletion, on purpose: a deleted guard would make
    the table refuse or come out empty, which every other test sees. An
    inversion keeps every property except the one under test, so a green from
    the two tests above means they are measuring the band and not the table.
    """
    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        'if escalon == ESCALON_ADABN and norma not in con_adabn:',
        'if escalon != ESCALON_ADABN and norma not in con_adabn:',
        "toy_tables_con_las_bandas_cruzadas")
    familia = toy_tables.familias(registro)[0]
    con_adabn = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
    normas = toy_tables.normalizaciones(registro)
    assert len(con_adabn) < len(normas), (con_adabn, normas)

    cabecera, filas = _parsear(mutado.render_familia_escalera(registro, familia))
    assert cabecera == _encabezado_de_creda(registro), (
        "la mutación movió las columnas y tendría que haber movido sólo las "
        "bandas")
    bandas: dict[str, list[str]] = {}
    for metodo, arquitectura in zip(_columna(cabecera, filas, "método"),
                                    _columna(cabecera, filas, "normalización")):
        bandas.setdefault(metodo, []).append(arquitectura)
    piso = toy_tables.ROTULO_ESCALON[toy_tables.ESCALON_FASE_1]
    adabn = toy_tables.ROTULO_ESCALON[toy_tables.ESCALON_ADABN]
    assert bandas[piso] == [toy_tables.NOMBRE_FILA[n] for n in con_adabn], (
        f"la mutación no cruzó las bandas: `.predict` sigue con {bandas[piso]}, "
        "así que el verde del test del piso no prueba que la banda sea derivada")
    # Y AdaBN, que la inversión deja alcanzando TODAS: su condición pasó a ser
    # «es AdaBN», que es verdadera en su propia banda, así que nunca saltea.
    assert bandas[adabn] == [toy_tables.NOMBRE_FILA[n] for n in normas], (
        f"AdaBN quedó con {bandas[adabn]} y la inversión tendría que dejarlo "
        "alcanzando todas las arquitecturas")
    assert len(filas) == 2 * len(con_adabn) + len(normas)


def test_adabns_architecture_is_decided_by_the_measurement_and_not_by_the_name(
        registro, tmp_path):
    """The red for «derived by buffer presence, never by name».

    The mutation is the exact defect and it is INVISIBLE on this artefact: the
    one architecture that carries buffers here is the one whose name a
    by-name renderer would look for, so both modules print the identical table.
    What distinguishes them is a record where the two answers differ -- another
    architecture given a measured buffer count -- and there the real module
    gains its row and the mutated one does not.

    The first half is what makes the second mean anything: without it, «the
    tables differ» could be any difference the mutation happened to cause.
    """
    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        'medidos = {celda["adabnLayers"] > 0 for celda in suyas}',
        'medidos = {norma == "batchnorm"     for celda in suyas}',
        "toy_tables_que_decide_por_el_nombre")
    familia = toy_tables.familias(registro)[0]
    assert (mutado.render_familia_escalera(registro, familia)
            == toy_tables.render_familia_escalera(registro, familia)), (
        "la mutación ya cambia la tabla sobre este artefacto, así que lo que el "
        "test mide abajo no es la diferencia entre medir y nombrar")

    con_buffers = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
    otras = [norma for norma in toy_tables.normalizaciones(registro)
             if norma not in con_buffers]
    assert otras, "toda arquitectura ya tiene buffers: no hay nada que encender"
    encendido = copy.deepcopy(registro)
    for celda in encendido["cells"].values():
        if celda["normalization"] in otras:
            celda["adabnLayers"] = 1
            celda["adabnNormLayers"] = max(celda["adabnNormLayers"], 1)
            celda["adabnTargetAccuracy"] = celda["predictTargetAccuracy"]

    rotulo = toy_tables.ROTULO_ESCALON[toy_tables.ESCALON_ADABN]

    def _filas_de_adabn(modulo) -> list[str]:
        cabecera, filas = _parsear(
            modulo.render_familia_escalera(encendido, familia))
        return [arq for met, arq in zip(_columna(cabecera, filas, "método"),
                                        _columna(cabecera, filas,
                                                 "normalización"))
                if met == rotulo]

    assert _filas_de_adabn(toy_tables) == [
        toy_tables.NOMBRE_FILA[norma]
        for norma in toy_tables.normalizaciones(registro)]
    assert _filas_de_adabn(mutado) == [
        toy_tables.NOMBRE_FILA[norma] for norma in con_buffers], (
        "la versión que decide por el nombre dio las mismas filas que la que "
        "mide: entonces el verde de arriba no prueba que la fila salga de la "
        "medición")


def test_a_header_that_stopped_being_credas_is_caught(registro, tmp_path):
    """The red for the eight-header comparison: move one character of one title.

    One character, and ASCII so the byte count does not move either: what has to
    be proven is that the comparison is on the exact text at the exact position
    and not on «a column whose name contains the word». Under the mutation the
    table has eight columns, the right rows and the right numbers.
    """
    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        'Columna("vigía: fuente conservada",',
        'Columna("vigía: fuente conservadX",',
        "toy_tables_con_una_cabecera_movida")
    familia = toy_tables.familias(registro)[0]
    cabecera, filas = _parsear(mutado.render_familia_escalera(registro, familia))
    assert len(cabecera) == 8 and len(filas) == _filas_de_creda(registro,
                                                                familia)
    assert cabecera != _encabezado_de_creda(registro), (
        "se movió un carácter de una cabecera y la comparación no lo vio: "
        "entonces no está comparando el encabezado de CREDA")


def test_the_control_architectures_absence_from_adabn_is_read_off_measured_rows(
        registro):
    """The control row has no AdaBN row, and the reason is stated where it fits.

    The join, over the record the generator left rather than over a fixture: a
    backbone that normalizes and keeps nothing has to come back from the RUN as
    `adabnTargetAccuracy: None` with `adabnNormLayers` non-zero, and the
    document has to say that its missing AdaBN row is the control of the grid
    and not a gap. Both halves, because either one alone is satisfiable without
    the other -- a record could carry the fields and the document could drop
    them, and a document could state a reason the record never produced.

    **It is the ARCHITECTURE TABLE's two measured rows that say it.** CREDA's
    eight columns carry no provenance column, so the per-cell reason left the
    family table; it moved to a computed conclusion, and then the conclusions
    were removed because CREDA's document prints none. What is left saying it is
    the pair of MEASURED rows of `render_arquitecturas` -- `adabnNormLayers` and
    `adabnLayers`, both read off the run -- and they are enough: the control is
    the architecture that normalizes and holds nothing re-estimable, which is
    exactly a row with the first count above zero and the second at zero. The
    row that NAMES it is a declared constant (`ESTADO_DE_NORMA`) and is
    deliberately not what this test reads, because asserting a constant proves
    nothing. What is gone and is reported rather than replaced: the per-family
    sentence that said it over that family's own cells.

    And the BatchNorm pole is required in the same pass: with no architecture
    carrying buffers, «AdaBN has no rung here» would be true of the whole grid
    and would say nothing about any backbone.
    """
    normas = toy_tables.normalizaciones(registro)
    sin_buffers, con_buffers = [], []
    for norma in normas:
        celdas = [c for c in toy_tables.celdas(registro)
                  if c["normalization"] == norma]
        assert celdas, norma
        if all(c["adabnTargetAccuracy"] is None for c in celdas):
            sin_buffers.append(norma)
            assert all(c["adabnLayers"] == 0 for c in celdas), norma
        else:
            con_buffers.append(norma)
            assert all(c["adabnLayers"] > 0 for c in celdas), norma
    assert con_buffers, (
        "ninguna arquitectura del registro tiene capas reestimables, así que "
        "la ausencia de la fila de AdaBN no distingue nada")
    assert sin_buffers, "ninguna arquitectura se queda sin fila de AdaBN"

    # La que normaliza y no guarda nada: es el control, y el registro la
    # distingue de la que no normaliza por `adabnNormLayers`.
    control = [norma for norma in sin_buffers
               if all(c["adabnNormLayers"] > 0 for c in toy_tables.celdas(registro)
                      if c["normalization"] == norma)]
    assert control, (
        f"ninguna de {sin_buffers} normaliza sin guardar nada: la comparación "
        "que separa «normalizar ayuda» de «reestimar ayuda» no está en este "
        "registro")

    rotulo = toy_tables.ROTULO_ESCALON[toy_tables.ESCALON_ADABN]
    for familia in toy_tables.familias(registro):
        por_metodo = _filas_por_metodo(registro, familia)
        for norma in sin_buffers:
            assert (toy_tables.NOMBRE_FILA[norma]
                    not in por_metodo.get(rotulo, [])), (familia, norma)
        for norma in con_buffers:
            assert (toy_tables.NOMBRE_FILA[norma]
                    in por_metodo.get(rotulo, [])), (familia, norma)

    # Y la tabla de arquitecturas distingue las tres con sus dos filas medidas:
    # la que no normaliza, la que normaliza y guarda, y el control --- que
    # normaliza y no guarda ---. Leido de la tabla EMITIDA y no del registro
    # otra vez, que es lo que hace de esto un test del documento.
    cabecera, filas = _parsear(toy_tables.render_arquitecturas(registro))
    normalizan = _fila_llamada(cabecera, filas, "capas que normalizan")
    reestimables = _fila_llamada(cabecera, filas, "capas que AdaBN puede reestimar")
    for norma in control:
        assert int(normalizan[norma]) > 0 and int(reestimables[norma]) == 0, (
            f"la tabla de arquitecturas no muestra a `{norma}` como el control "
            f"de la grilla: normaliza {normalizan[norma]} capas y AdaBN puede "
            f"reestimar {reestimables[norma]}")
    for norma in con_buffers:
        assert int(reestimables[norma]) > 0, norma
    # El polo que hace de las dos filas una distincion y no una coincidencia:
    # alguna arquitectura tiene que normalizar MENOS que el control, o «el
    # control normaliza y no guarda» seria cierto de toda la grilla.
    assert any(int(normalizan[norma]) < int(normalizan[control[0]])
               for norma in normas if norma != control[0]), (
        "todas las arquitecturas normalizan lo mismo, así que la primera fila "
        "no distingue al control de ninguna otra")


# ------------------------------------------------- los tests declarados
#
# `### <familia> — los tests declarados` is CREDA's TWO tables, and what each
# test below holds is one of the three mappings that decided them: which pair
# `con CREDA` / `sin CREDA` is here, which architecture the three-rung table is
# built for, and what `PROMEDIO` averages. It was one table of nine rows with a
# `por qué no hay test` column whose every row held the same paragraph, so the
# last test here is the one that keeps the reason under the table.

#: How many whitespace-separated tokens a cell of either table may hold. Three,
#: and the number is READ off the widest thing these tables print rather than
#: chosen: the architecture labels (`sin normalización`, two) and the interval
#: (`[+0.1007, +0.1793]`, two). A reason is a paragraph, so no threshold near
#: this one can be satisfied by a prose column -- which the reachability pole of
#: that test measures rather than assumes.
PALABRAS_POR_CELDA = 3


def _tablas_declaradas(registro: dict, familia: str) -> tuple[str, str]:
    """The two tables of the subsection, in the order a reader meets them."""
    return (toy_tables.render_familia_sin_contra_con(registro, familia),
            toy_tables.render_familia_tests(registro, familia))


def _celdas_de(markdown: str) -> list[str]:
    """Every data cell of every grid of one rendering, headers excluded."""
    return [celda.strip() for _, filas in _grillas(markdown)
            for fila in filas for celda in fila]


def _registro_con_semillas(registro: dict, semillas: list[int], valores) -> dict:
    """The real record re-seeded, with each cell's three rungs set by `valores`.

    `valores(normalizacion, indice)` returns the floor, the AdaBN and the method
    readings of that cell, so a test can build a record whose paired differences
    say what it needs to distinguish -- which is the only way to tell a
    denominator from another denominator, since both give the same mean.

    Built from the REAL record and not from a fixture: every other field a
    renderer reads stays what the generator wrote, so what moves is only what
    the test is about.
    """
    muchos = copy.deepcopy(registro)
    muchos["scale"]["report_seeds"] = list(semillas)
    for clave, celda in list(registro["cells"].items()):
        familia, norma, brazo, _ = clave.split(toy_tables.SEP)
        for indice, semilla in enumerate(semillas):
            nueva = copy.deepcopy(celda)
            piso, adabn, metodo = valores(norma, indice)
            nueva["predictTargetAccuracy"] = piso
            nueva["targetAccuracy"] = metodo
            if celda["adabnTargetAccuracy"] is not None:
                nueva["adabnTargetAccuracy"] = adabn
            muchos["cells"][toy_tables.SEP.join(
                (familia, norma, brazo, str(semilla)))] = nueva
    return muchos


def _sin_buffers(registro: dict) -> dict:
    """The record with no architecture carrying anything re-estimable."""
    apagado = copy.deepcopy(registro)
    for celda in apagado["cells"].values():
        celda["adabnLayers"] = 0
        celda["adabnTargetAccuracy"] = None
    return apagado


def _con_buffers_en_todas(registro: dict) -> dict:
    """The record with every architecture carrying measured buffers.

    The other pole of the derivation: the architectures whose backbone keeps
    nothing are given a measured count, so «table B renders the ones that carry
    buffers» can be told apart from «table B renders BatchNorm».
    """
    encendido = copy.deepcopy(registro)
    for celda in encendido["cells"].values():
        if celda["adabnTargetAccuracy"] is None:
            celda["adabnLayers"] = 1
            celda["adabnNormLayers"] = max(celda["adabnNormLayers"], 1)
            celda["adabnTargetAccuracy"] = celda["predictTargetAccuracy"]
    return encendido


def test_the_declared_tests_subsection_emits_exactly_two_tables(registro,
                                                                tmp_path):
    """Two tables under the one heading, counted, and the refusal when the
    second one cannot be built.

    Counting is the claim: CREDA's subsection is two tables and this one was
    ONE, so «the tables are there» is satisfied by the shape that was wrong. The
    count of table B's grids is DERIVED from the same measurement that decides
    which architectures it covers, so a record where two architectures carry
    buffers gives three tables under the heading and this test says so rather
    than failing.

    And the other pole: with no architecture carrying buffers there is no
    three-rung table to build, so what the subsection emits is table A and one
    SENTENCE -- never a grid of dashes, which would say the three rungs were
    compared and came out empty.

    That the sentence goes through the one builder is no longer observable from
    its output: `_bloque([])` carried the scale stamp, the operator retired it,
    and an empty block now adds nothing. What holds the routing today is
    `test_every_renderer_goes_through_the_one_builder`, which watches the call;
    the assertion that used to read the stamp off this sentence is deleted
    rather than replaced by one that cannot fail.
    """
    for familia in toy_tables.familias(registro):
        con_adabn = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
        tabla_a, tabla_b = _tablas_declaradas(registro, familia)
        assert len(_grillas(tabla_a)) == 1, familia
        assert len(_grillas(tabla_b)) == len(con_adabn), (familia, con_adabn)
        assert len(_grillas(tabla_a)) + len(_grillas(tabla_b)) == 2, (
            f"la subsección de {familia} no emite dos tablas: este artefacto "
            f"tiene {len(con_adabn)} arquitectura(s) con escalón de AdaBN")

    apagado = _sin_buffers(registro)
    familia = toy_tables.familias(registro)[0]
    assert toy_tables.arquitecturas_con_escalon_adabn(apagado, familia) == []
    tabla_a, tabla_b = _tablas_declaradas(apagado, familia)
    assert len(_grillas(tabla_a)) == 1
    assert _grillas(tabla_b) == [], (
        "sin ninguna arquitectura con buffers la tabla B trajo una grilla: una "
        "grilla de rayas dice que los tres escalones se compararon y salieron "
        "vacíos, que es otra cosa que no tener tres escalones")
    assert "no tiene escalón en ninguna" in tabla_b, tabla_b

    # El rojo: invertir el filtro que decide qué arquitecturas tienen tabla.
    # Una inversión y no un borrado --- un filtro borrado haría una tabla por
    # arquitectura, que es el conteo que el otro polo de arriba ya pide ---, así
    # que lo que queda es una subsección con las arquitecturas EQUIVOCADAS y el
    # mismo número de grillas en ningún caso.
    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        '        if bloque["normalizacion"] not in con_adabn:',
        '        if bloque["normalizacion"]     in con_adabn:',
        "toy_tables_con_el_filtro_invertido")
    familia = toy_tables.familias(registro)[0]
    con_adabn = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
    otras = [norma for norma in toy_tables.normalizaciones(registro)
             if norma not in con_adabn]
    cruzado = mutado.render_familia_tests(registro, familia)
    assert len(_grillas(cruzado)) == len(otras), (
        "la mutación no cruzó el filtro: entonces el conteo de arriba no "
        "prueba que las tablas salgan de la medición de buffers")
    assert len(_grillas(cruzado)) != len(con_adabn), (
        f"este artefacto tiene tantas arquitecturas con escalón como sin él "
        f"({con_adabn} / {otras}), así que la inversión no se ve en el conteo")


def test_table_a_has_one_row_per_declared_architecture_then_the_average(
        registro, tmp_path):
    """Table A's rows are CREDA's: one per declared architecture, in the
    record's order, and then `PROMEDIO`.

    A SEQUENCE and not a set: `PROMEDIO` is a reduction over the rows above it,
    so a reader who met it first would read the architectures as a breakdown of
    something stated before them. Its label is CREDA's own word and its row key
    is `None`, which is what keeps anything from confusing it with a declared
    architecture.

    The red is that reordering, and it is the one defect of this table that
    every other assertion survives: under the mutation the rows are the same
    rows, with the same numbers and the same footnote, and the reduction is
    printed above the rows it reduces.
    """
    normas = toy_tables.normalizaciones(registro)
    esperado = ([toy_tables.NOMBRE_FILA[norma] for norma in normas]
                + [toy_tables.ROTULO_PROMEDIO])
    for familia in toy_tables.familias(registro):
        cabecera, filas = _parsear(
            toy_tables.render_familia_sin_contra_con(registro, familia))
        assert _columna(cabecera, filas, "normalización") == esperado, familia
        res = toy_tables.comparar_familia_sin_contra_con(registro, familia)
        assert [fila["clave"] for fila in res["filas"]] == normas + [None], (
            f"la fila que promedia de {familia} trae una clave de "
            "arquitectura: algo puede confundirla con una")

    antes = ("    filas.append(_fila(\n"
             "        ROTULO_PROMEDIO, None,\n"
             "        _dentro_de_la_semilla(series, normas, 0, bloques),\n"
             "        _dentro_de_la_semilla(series, normas, 1, bloques)))")
    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        antes, antes.replace("filas.append(_fila(", "filas[:0] = [_fila(")
                    .replace("bloques)))", "bloques))]"),
        "toy_tables_con_el_promedio_arriba")
    familia = toy_tables.familias(registro)[0]
    cabecera, filas = _parsear(
        mutado.render_familia_sin_contra_con(registro, familia))
    movido = _columna(cabecera, filas, "normalización")
    assert movido[0] == toy_tables.ROTULO_PROMEDIO, (
        f"la mutación no movió la fila que promedia ({movido}): entonces el "
        "verde de arriba no prueba que la secuencia esté asertada")
    assert movido != esperado
    # Y el control de que la mutación movió SÓLO el orden: las mismas filas,
    # con los mismos números.
    assert sorted(movido) == sorted(esperado)


def test_the_average_row_pairs_inside_the_seed_and_never_pools_the_cells(
        registro, tmp_path):
    """`PROMEDIO` averages the architectures WITHIN each seed, so its pairs are
    as many as there are report seeds and never as many as there are cells.

    Driven over a record built so the two answers DIFFER, because on any record
    where they do not the test would pass over the wrong one: the point estimate
    is the same either way -- the mean of the averaged differences is the mean
    of all of them -- so what distinguishes them is the denominator and the win
    count, and those only move when the per-seed averages do not share the sign
    of the pairs inside them.

    Both answers are COMPUTED here, from the same record, so the one that is
    wrong is measured rather than remembered.
    """
    normas = toy_tables.normalizaciones(registro)
    assert len(normas) >= 2, (
        "con una sola arquitectura el promedio dentro de la semilla y el "
        "agrupado son el mismo conjunto: este test no distinguiría nada")
    ultima = normas[-1]

    def valores(norma: str, indice: int):
        # Semilla 0: todas suben 0.15. Semilla 1: dos suben 0.05 y la última
        # baja 0.30, de modo que el promedio de la semilla 1 queda NEGATIVO
        # mientras la mayoría de sus pares de celda-semilla es positiva.
        piso = 0.50
        if indice == 0:
            return piso, piso, 0.65
        return piso, piso, (0.20 if norma == ultima else 0.55)

    dos = _registro_con_semillas(registro, [0, 1], valores)
    familia = toy_tables.familias(registro)[0]
    res = toy_tables.comparar_familia_sin_contra_con(dos, familia)
    promedio = res["filas"][-1]
    assert promedio["rotulo"] == toy_tables.ROTULO_PROMEDIO

    # La alternativa, calculada del mismo registro: los pares de celda-semilla.
    agrupados = [celda["targetAccuracy"] - celda["predictTargetAccuracy"]
                 for celda in toy_tables.celdas(dos, familia)]
    assert len(agrupados) == 2 * len(normas)
    gana_agrupado = sum(1 for d in agrupados if d > 0.0)
    assert gana_agrupado != 1, (
        "el conteo agrupado coincide con el de dentro de la semilla, así que "
        "este registro no distingue los dos denominadores")

    assert promedio["n"] == 2, (
        f"la fila que promedia dice {promedio['n']} pares y las semillas de "
        "reporte son dos: está agrupando las celdas")
    assert promedio["gana"] == 1, (
        f"la fila que promedia gana {promedio['gana']} de {promedio['n']} y "
        f"dentro de la semilla gana una de dos; agrupado gana "
        f"{gana_agrupado} de {len(agrupados)}")
    # El estimador puntual sale igual de las dos formas, que es exactamente por
    # qué lo que hay que testear es el denominador.
    assert promedio["delta"] == pytest.approx(statistics.fmean(agrupados))

    cabecera, filas = _parsear(
        toy_tables.render_familia_sin_contra_con(dos, familia))
    assert _columna(cabecera, filas, "gana")[-1] == "1/2"
    assert f"{gana_agrupado}/{len(agrupados)}" not in _columna(
        cabecera, filas, "gana")

    # El rojo: la versión que agrupa. Misma longitud para que el .pyc no pueda
    # quedar viejo, y el relleno va al final de la última línea.
    antes = ('    return [statistics.fmean([series[norma][cual][bloque] '
             'for norma in normas])\n'
             '            for bloque in range(bloques)]')
    crudo = ('    return [series[norma][cual][bloque] '
             'for bloque in range(bloques)\n'
             '            for norma in normas]')
    assert len(crudo) < len(antes), (len(crudo), len(antes))
    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        antes, crudo + " " * (len(antes) - len(crudo)),
        "toy_tables_que_agrupa_las_celdas")
    agrupa = mutado.comparar_familia_sin_contra_con(dos, familia)["filas"][-1]
    assert agrupa["n"] == len(agrupados), (
        "la mutación no agrupó: entonces el verde de arriba no prueba que el "
        "promedio sea dentro de la semilla")
    assert agrupa["gana"] == gana_agrupado
    assert agrupa["delta"] == pytest.approx(promedio["delta"]), (
        "la mutación movió el estimador puntual, así que lo que distingue a "
        "las dos versiones no es sólo el denominador y este test mide otra cosa")


def test_table_b_renders_only_the_architectures_with_reestimable_buffers(
        registro, tmp_path):
    """Table B's architectures are DERIVED from measured buffer presence.

    CREDA renders BatchNorm because that is the one architecture where all three
    rungs exist, and «where all three exist» is a property of the backbone that
    the run measures cell by cell -- never the layer's name. Deciding by name
    would give a three-rung table to the architecture that normalizes and keeps
    nothing, whose middle rung would be the floor under another name.

    Three poles. The architectures that carry buffers get a table and the others
    get none; a record where every architecture carries them gets one table
    each, so the count follows the measurement rather than the artefact; and the
    by-name version does NOT gain them, which is what makes the first two mean
    something -- on this artefact the two answers coincide, so a test that only
    read this record could not tell them apart.
    """
    normas = toy_tables.normalizaciones(registro)
    familia = toy_tables.familias(registro)[0]
    con_buffers = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
    otras = [norma for norma in normas if norma not in con_buffers]
    assert con_buffers and otras, (con_buffers, otras)

    tabla_b = toy_tables.render_familia_tests(registro, familia)
    assert len(_grillas(tabla_b)) == len(con_buffers)
    for norma in con_buffers:
        assert toy_tables.NOMBRE_FILA[norma] in tabla_b, norma
    for norma in otras:
        assert toy_tables.NOMBRE_FILA[norma] not in tabla_b, (
            f"`{norma}` tiene tabla de tres escalones y su backbone no guarda "
            "nada reestimable: el escalón del medio sería el piso con otro "
            "nombre")

    encendido = _con_buffers_en_todas(registro)
    assert (toy_tables.arquitecturas_con_escalon_adabn(encendido, familia)
            == normas)
    assert len(_grillas(toy_tables.render_familia_tests(
        encendido, familia))) == len(normas)

    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        'medidos = {celda["adabnLayers"] > 0 for celda in suyas}',
        'medidos = {norma == "batchnorm"     for celda in suyas}',
        "toy_tables_tests_que_decide_por_el_nombre")
    assert (mutado.render_familia_tests(registro, familia)
            == tabla_b), (
        "la mutación ya cambia la tabla sobre este artefacto, así que lo que "
        "sigue no mide la diferencia entre medir y nombrar")
    assert len(_grillas(mutado.render_familia_tests(
        encendido, familia))) == len(con_buffers), (
        "la versión que decide por el nombre dio las mismas tablas que la que "
        "mide: entonces el verde de arriba no prueba que salgan de la medición")


def test_table_b_has_one_row_per_declared_pair_in_the_declared_order(
        registro, tmp_path):
    """One row per declared pair, in `pares_de_la_escalera`'s order, with the
    more intervened rung named FIRST -- and the order does not follow the winner.

    If it followed the winner an always-positive sign would say nothing, so the
    control is a record where the winner is the OTHER rung: the deltas come out
    negative and the two columns stay exactly where they were.

    The red is the quietest defect these two columns have: a loop that reads the
    pair the other way round. Under it the table has the same rows, the same
    intervals and the same win counts, with every sign flipped and every pair
    named backwards -- which is how `+0.0819` beside `.predict` came to read as
    `.predict` winning in this lineage.
    """
    pares = toy_tables.pares_de_la_escalera()
    nombres = [escalon for escalon, _, _ in toy_tables.ESCALERA]
    for primero, segundo in pares:
        assert nombres.index(primero) > nombres.index(segundo), (primero, segundo)

    def _columnas_del_par(modulo, registro_, familia_):
        cabecera, filas = _parsear(
            modulo.render_familia_tests(registro_, familia_))
        return list(zip(_columna(cabecera, filas, "gana si Δ>0"),
                        _columna(cabecera, filas, "contra")))

    for familia in toy_tables.familias(registro):
        con_adabn = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
        for _, filas in _grillas(toy_tables.render_familia_tests(
                registro, familia)):
            assert len(filas) == len(pares), (familia, len(filas))
        assert _columnas_del_par(toy_tables, registro, familia) == list(pares), (
            familia, con_adabn)

    # El ganador invertido: el piso por encima del método en toda celda.
    familia = toy_tables.familias(registro)[0]
    def _al_revés(norma: str, indice: int):
        return 0.70, 0.60, 0.40 + 0.01 * indice

    invertido = _registro_con_semillas(registro, [0, 1], _al_revés)
    cabecera, filas = _parsear(
        toy_tables.render_familia_tests(invertido, familia))
    deltas = _columna(cabecera, filas, "Δ = 1ª − 2ª")
    assert all(valor.startswith("-") for valor in deltas), (
        f"el registro invertido no invirtió los ganadores ({deltas}), así que "
        "el orden que sigue no prueba que sea fijo")
    assert _columnas_del_par(toy_tables, invertido, familia) == list(pares), (
        "el orden del par siguió al ganador")

    mutado = _modulo_mutado(
        Path(toy_tables.__file__), tmp_path,
        "        for primero, segundo in pares_de_la_escalera():",
        "        for segundo, primero in pares_de_la_escalera():",
        "toy_tables_con_el_par_al_revés")
    assert _columnas_del_par(mutado, registro, familia) == [
        (segundo, primero) for primero, segundo in pares], (
        "la mutación no dio vuelta el par: entonces el verde de arriba no "
        "prueba que las dos columnas estén en el orden declarado")


def test_neither_declared_test_table_carries_a_prose_column(registro):
    """No cell of either table holds a sentence, and each reason is under its
    table exactly once.

    This is the defect the two tables replaced: a `por qué no hay test` column
    whose every row held the same paragraph, which is one explanation rendered
    once per row. CREDA's footnotes are one line under each table.

    Both halves, because either alone is satisfiable without the other: a table
    could drop the column and lose the reason entirely -- a dash with nothing
    beside it reads as a number that failed to render -- and a table could state
    the reason under it and keep the column as well.

    The threshold is reachable and that is MEASURED rather than argued: the same
    predicate is run over a grid built here with a reason in its one cell, and
    it has to fire.
    """
    for familia in toy_tables.familias(registro):
        for tabla in _tablas_declaradas(registro, familia):
            largas = [celda for celda in _celdas_de(tabla)
                      if len(celda.split()) > PALABRAS_POR_CELDA]
            assert largas == [], (
                f"{familia} trae {largas[:1]} en una celda: una columna de "
                "prosa es una explicación impresa una vez por fila")

        # Los motivos de las filas que las dos tablas IMPRIMEN: la tabla B
        # cubre las arquitecturas con escalón de AdaBN y nada más, así que los
        # motivos de las otras no son suyos --- y, retiradas las conclusiones,
        # ya no los dice nadie. El test de abajo sigue pidiendo que no estén
        # acá, que es la mitad que se puede sostener.
        con_adabn = toy_tables.arquitecturas_con_escalon_adabn(registro, familia)
        de_b = {fila["razon"]
                for bloque in toy_tables.comparar_familia(
                    registro, familia)["porArquitectura"]
                if bloque["normalizacion"] in con_adabn
                for fila in bloque["filas"] if fila["razon"]}
        de_a = {fila["razon"] for fila
                in toy_tables.comparar_familia_sin_contra_con(
                    registro, familia)["filas"] if fila["razon"]}
        assert de_a and de_b, (
            f"alguna de las dos tablas de {familia} no trae ninguna fila con "
            "motivo, así que que el motivo esté una sola vez no dice nada "
            "sobre este artefacto")
        tabla_a, tabla_b = _tablas_declaradas(registro, familia)
        # Una vez por TABLA, que es dónde va un pie: cada tabla se lee sola, y
        # un motivo que estuviera sólo bajo la otra dejaría rayas sin explicar
        # en la que el lector tiene delante. Lo que no puede es estar una vez
        # por FILA, que es la columna que estas dos tablas reemplazaron.
        for tabla, motivos, filas_con in (
                (tabla_a, de_a, len(_grillas(tabla_a)[0][1])),
                (tabla_b, de_b, len(_grillas(tabla_b)[0][1]))):
            assert filas_con > 1, (
                "la tabla tiene una fila sola, así que una vez por tabla y una "
                "vez por fila son lo mismo y este test no distingue nada")
            for motivo in motivos:
                assert tabla.count(motivo) == 1, (
                    f"el motivo «{motivo[:40]}…» aparece "
                    f"{tabla.count(motivo)} veces en una tabla de {familia} "
                    f"que tiene {filas_con} filas: va una vez, en el pie")
        # Y el motivo de un par que no se puede restar no está en ninguna de
        # las dos: no hay fila suya, así que una frase sobre él debajo de una
        # tabla hablaría de algo que esa tabla no muestra. Lo decía la
        # conclusión, que se retiró con las otras tres de este documento ---
        # así que hoy ese motivo no se renderiza en ninguna parte, y lo que
        # queda asertado es la mitad que sigue siendo cierta: que no se colo
        # en las tablas que sí están.
        ajenos = {fila["razon"]
                  for bloque in toy_tables.comparar_familia(
                      registro, familia)["porArquitectura"]
                  if bloque["normalizacion"] not in con_adabn
                  for fila in bloque["filas"]
                  if fila["razon"] and fila["delta"] is None}
        for motivo in ajenos:
            assert motivo not in tabla_a and motivo not in tabla_b, motivo
        assert ajenos, f"{familia} no trae ningún par sin segundo escalón"

    # Y que el predicado dispara: la forma que esto reemplazó, armada acá.
    motivo = next(iter(motivos))
    vieja = f"|por qué no hay test|\n|---|\n|{motivo}|\n|{motivo}|"
    assert [celda for celda in _celdas_de(vieja)
            if len(celda.split()) > PALABRAS_POR_CELDA], (
        "el umbral de palabras no ve una columna de prosa, así que el verde de "
        "arriba pasaría con la columna puesta")


def test_the_declared_tests_refuse_to_print_a_statistic_that_cannot_move(
        registro):
    """Below the declared floor of blocks, the test columns are DASHED.

    Measured before it was written, and this is where the measurement lives:
    over three treatments and ONE block, `scipy.stats.friedmanchisquare`
    returns the same chi-square and the same p for every input, because the
    within-block ranks are always a permutation of 1..k; `scipy.stats.wilcoxon`
    over one pair returns p = 1 for every difference; `scipy.stats.ttest_rel`
    over one pair returns `nan`; and `scipy.stats.shapiro` returns `nan` below
    three values, which is why that one has a floor of its own. A number that
    cannot come out different is not a measurement, for exactly the reason an
    assertion that cannot fail proves nothing.

    So the constants are measured here too. If a future scipy made any of them
    move, this test reddens and the floor gets re-argued -- which is better than
    a floor nobody can check.
    """
    import math

    import scipy.stats as st

    constantes = {st.friedmanchisquare(*trio).pvalue
                  for trio in (([1.0], [2.0], [3.0]), ([3.0], [2.0], [1.0]),
                               ([0.0], [100.0], [50.0]),
                               ([0.4444], [0.3333], [0.3333]))}
    assert len(constantes) == 1, (
        f"Friedman sobre un bloque ya no es constante ({constantes}): hay que "
        "releer por qué el piso de bloques existe")
    assert {st.wilcoxon([d]).pvalue for d in (0.1, -0.9, 0.0001)} == {1.0}
    assert math.isnan(st.ttest_rel([0.4], [0.3]).pvalue)
    # Y Shapiro, que tiene su propio piso: `nan` por debajo de tres valores y
    # un número real desde tres. Un `nan` impreso donde va una p se lee como un
    # número, así que la columna va con raya y el motivo va debajo.
    assert all(math.isnan(st.shapiro([0.1 * k for k in range(n)]).pvalue)
               for n in range(1, toy_tables.BLOQUES_MINIMOS_SHAPIRO)), (
        "scipy ya calcula Shapiro por debajo de "
        f"{toy_tables.BLOQUES_MINIMOS_SHAPIRO} valores: hay que releer su piso")
    assert not math.isnan(st.shapiro(
        [0.1, 0.25, 0.4]).pvalue)
    # Y que con DOS bloques sí se mueve, que es lo que hace del piso un piso.
    assert (st.friedmanchisquare([1.0, 1.0], [2.0, 2.0], [3.0, 3.0]).pvalue
            != st.friedmanchisquare([1.0, 3.0], [2.0, 2.0], [3.0, 1.0]).pvalue)

    bloques = len(registro["scale"]["report_seeds"])
    assert bloques < toy_tables.BLOQUES_MINIMOS, (
        f"el registro trae {bloques} bloques y el piso es "
        f"{toy_tables.BLOQUES_MINIMOS}: este test está escrito para el caso de "
        "abajo del piso y hay que releerlo")
    con, sin = (toy_tables._llano(escalon)
                for escalon in toy_tables.PAREJA_SIN_CON)
    for familia in toy_tables.familias(registro):
        cabecera, filas = _parsear(
            toy_tables.render_familia_sin_contra_con(registro, familia))
        for titulo in ("IC95 de Δ", f"t({bloques - 1})", "p", "Shapiro p",
                       "Wilcoxon p"):
            columna = _columna(cabecera, filas, titulo)
            assert set(columna) == {toy_tables.NADA}, (
                f"la tabla A de {familia} imprime {titulo!r} con {bloques} "
                "bloque(s): ese número es una constante del número de bloques "
                "y no una medición")
        deltas = _columna(cabecera, filas, f"Δ = {con} − {sin}")
        assert all(d != toy_tables.NADA for d in deltas), (
            f"la tabla A de {familia} no imprime la diferencia pareada, que es "
            "lo único que a un bloque es exacto")

        cabecera, filas = _parsear(
            toy_tables.render_familia_tests(registro, familia))
        for titulo in ("IC95 de Δ", "p (Wilcoxon)",
                       "p Holm (familia de 0)"):
            columna = _columna(cabecera, filas, titulo)
            assert set(columna) == {toy_tables.NADA}, (
                f"la tabla B de {familia} imprime {titulo!r} con {bloques} "
                "bloque(s)")
        deltas = _columna(cabecera, filas, "Δ = 1ª − 2ª")
        assert any(d != toy_tables.NADA for d in deltas), (
            f"la tabla B de {familia} se quedó sin lo único que a esta escala "
            "es exacto")
        comparacion = toy_tables.comparar_familia(registro, familia)
        assert all(bloque["omnibus"] is None
                   for bloque in comparacion["porArquitectura"])
        assert all(bloque["familiaDeHolm"] == 0
                   for bloque in comparacion["porArquitectura"])


def test_the_declared_tests_do_run_once_there_are_enough_blocks(registro):
    """The other pole: give it blocks and the statistics appear.

    Without this, «dashed below the floor» would be satisfied by two tables that
    can never compute anything, and the subsection would be a permanent apology.
    Driven over a record built from the real one with extra report seeds -- the
    pilot runs one, and at one every statistic is a constant.

    The values are SPREAD and not copied: a copy would give paired differences
    with no dispersion at all, which the renderer refuses for its own declared
    reason, and then this test would be measuring that refusal instead of the
    test running.

    Shapiro is asked at its own floor, which is higher than the other two: at
    two blocks the t and the Wilcoxon are there and the Shapiro is a dash with
    its reason, and at five all four are numbers.
    """
    familia = toy_tables.familias(registro)[0]

    def valores(norma: str, indice: int):
        return 0.30 + 0.01 * indice, 0.35 + 0.02 * indice, 0.40 + 0.03 * indice

    muchos = _registro_con_semillas(registro, [0, 1, 2, 3, 4], valores)
    comparacion = toy_tables.comparar_familia(muchos, familia)
    con_omnibus = [bloque for bloque in comparacion["porArquitectura"]
                   if bloque["omnibus"] is not None]
    assert con_omnibus, (
        "con cinco bloques y tres escalones no salió ningún Friedman: la mitad "
        "que calcula no se ejercita nunca y la tabla sería una disculpa "
        "permanente")
    for bloque in con_omnibus:
        assert bloque["omnibus"]["gl"] == len(bloque["tratamientos"]) - 1
        assert bloque["omnibus"]["mejorRango"] == len(bloque["tratamientos"])
        rangos = bloque["omnibus"]["rangos"]
        assert sorted(rangos) == sorted(bloque["tratamientos"])
        # Los rangos medios de un bloque suman lo mismo que los rangos de uno:
        # son una permutación de 1..k promediada, así que su suma es k(k+1)/2.
        tratamientos = len(bloque["tratamientos"])
        assert sum(rangos.values()) == pytest.approx(
            tratamientos * (tratamientos + 1) / 2)
        assert bloque["familiaDeHolm"] > 0
        for fila in bloque["filas"]:
            if fila["p"] is None:
                continue
            assert fila["ic"] is not None
            assert fila["holm"] is not None
            assert fila["holm"] >= fila["p"] - 1e-12, (fila["holm"], fila["p"])
            assert fila["mediana"] is not None

    # La tabla B: la línea de Friedman va ARRIBA de la grilla y no en una
    # columna. Leído de lo que se emitió: el estadístico está antes de la regla
    # de la tabla, y ninguna cabecera lo nombra.
    tabla_b = toy_tables.render_familia_tests(muchos, familia)
    chi = f"{con_omnibus[0]['omnibus']['chi2']:.2f}"
    assert f"χ²({con_omnibus[0]['omnibus']['gl']}) = {chi}" in tabla_b, tabla_b
    cabecera, filas = _parsear(tabla_b)
    assert tabla_b.index(chi) < tabla_b.index("|---|"), (
        "el estadístico de Friedman no está arriba de la grilla")
    assert not [titulo for titulo in cabecera if "χ²" in titulo], cabecera
    # Y sobre cuántos bloques se tomó, en esta misma línea, que es donde CREDA
    # lo imprime (`... · 10 bloques de semilla · rangos medios ...`). Estaba en
    # el encabezado de la familia arriba del bloque; el operador retiró ese
    # encabezado, así que sin esto el bloque no diría sobre cuántos bloques
    # reducen sus números.
    linea_del_omnibus = next(linea for linea in tabla_b.splitlines()
                             if chi in linea)
    assert toy_tables._cuantos_bloques(muchos) in linea_del_omnibus, (
        f"la línea del omnibus es «{linea_del_omnibus}» y no dice sobre "
        f"cuántos bloques se tomó")
    # Acá iba «y la conclusión no repite este estadístico»: una medición se
    # renderiza una vez, y la conclusión de los tests era el único otro lugar
    # que lo calculaba. Retirada la conclusión, no hay nada que pueda repetirlo
    # y la aserción no sería violable --- que es la razón por la que se saca en
    # vez de reescribirse. Lo que sostiene la afirmación hoy es estructural:
    # `test_nothing_in_this_document_renders_a_conclusion`. Una sustituta que
    # buscara el mismo string en las otras tablas del módulo se midió y no
    # distingue nada: `chi` es un número de dos decimales y colisiona con el de
    # otra familia por azar.
    for titulo in ("IC95 de Δ", "p (Wilcoxon)"):
        assert any(valor != toy_tables.NADA
                   for valor in _columna(cabecera, filas, titulo)), titulo

    # Y la tabla A, cuyas cinco columnas de test son las de CREDA.
    cabecera, filas = _parsear(
        toy_tables.render_familia_sin_contra_con(muchos, familia))
    for titulo in ("IC95 de Δ", "t(4)", "p", "Shapiro p", "Wilcoxon p"):
        assert any(valor != toy_tables.NADA
                   for valor in _columna(cabecera, filas, titulo)), titulo

    # El piso propio de Shapiro: a dos bloques el t está y el Shapiro no, con
    # su motivo debajo de la tabla y no en una fila.
    dos = _registro_con_semillas(registro, [0, 1], valores)
    assert toy_tables.BLOQUES_MINIMOS_SHAPIRO > 2
    cabecera, filas = _parsear(
        toy_tables.render_familia_sin_contra_con(dos, familia))
    assert set(_columna(cabecera, filas, "Shapiro p")) == {toy_tables.NADA}
    assert all(valor != toy_tables.NADA
               for valor in _columna(cabecera, filas, "t(1)"))
    assert "Shapiro-Wilk no se calcula" in toy_tables.render_familia_sin_contra_con(
        dos, familia)


def test_the_pair_of_the_two_arm_table_is_the_method_against_the_floor(
        registro):
    """Which pair CREDA's `con CREDA` / `sin CREDA` is HERE, and that the
    columns are named for what they hold.

    CREDA subtracts `acc_predict` -- its step-0 reading -- from `acc_te`, so the
    pair is the regularizer on against nothing on, and the two rungs that are
    those two things here are the ladder's method and its floor. AdaBN is
    neither: it re-estimates the target's running statistics, which is an
    intervention of its own, and it exists only where the backbone keeps
    buffers, so a column of it would be empty in most rows of a table that has
    one row per architecture.

    Derived from `ESCALERA` and asserted against the record fields each rung
    reads, so a renamed rung or a re-pointed field moves this test rather than
    leaving the columns naming something else.
    """
    assert toy_tables.PAREJA_SIN_CON == (toy_tables.ESCALON_METODO,
                                         toy_tables.ESCALON_FASE_1)
    assert toy_tables.ESCALON_ADABN not in toy_tables.PAREJA_SIN_CON
    con, sin = toy_tables.PAREJA_SIN_CON
    familia = toy_tables.familias(registro)[0]
    cabecera, filas = _parsear(
        toy_tables.render_familia_sin_contra_con(registro, familia))
    assert con in cabecera and sin in cabecera, cabecera
    assert toy_tables.ESCALON_ADABN not in cabecera, cabecera

    # Y que cada columna trae el campo de SU escalón, celda por celda.
    for norma, fila in zip(toy_tables.normalizaciones(registro), filas):
        celdas = [celda for celda in toy_tables.celdas(registro, familia)
                  if celda["normalization"] == norma]
        for escalon in (con, sin):
            esperado = toy_tables._num(statistics.fmean(
                [celda[toy_tables._campo_de(escalon)] for celda in celdas]))
            assert fila[cabecera.index(escalon)] == esperado, (norma, escalon)


def test_holm_leaves_an_untestable_pair_out_of_its_own_denominator(registro):
    """A pair that produced no p is not part of the family being corrected.

    Counting it in `m` would correct for a comparison that does not exist and
    loosen the others; inventing `p = 1.0` for it would assert that it was
    measured and not rejected. CREDA's own rule, with its own reason, and the
    arithmetic is asserted against the step-down formula rather than against
    numbers written here.
    """
    assert toy_tables._holm([None, None]) == [None, None]
    crudas = [0.01, 0.04, None]
    ajustadas = toy_tables._holm(crudas)
    assert ajustadas[2] is None
    assert ajustadas[0] == pytest.approx(2 * 0.01)
    assert ajustadas[1] == pytest.approx(max(2 * 0.01, 1 * 0.04))
    # Si el `None` contara en `m`, el primero sería `3 * 0.01`.
    assert ajustadas[0] != pytest.approx(3 * 0.01)
    # Monótono y topado en uno.
    assert toy_tables._holm([0.9, 0.95]) == [1.0, 1.0]
