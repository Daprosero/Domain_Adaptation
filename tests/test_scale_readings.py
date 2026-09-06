"""De qué corrida salen los números que alguien mira.

Un DESTINO es dónde una corrida escribe y ya tiene su regla
(`config.DESTINOS`, en `test_label_noise.py`). Una LECTURA es de qué corrida se
LEE lo que después se muestra, y no tenía ninguna. Los dos defectos son la misma
coordenada perdida y ninguno se parece al otro cuando falla: un destino
equivocado escribe encima de otra corrida, una lectura equivocada no escribe
nada y renderiza una ausencia honesta ---«no hay búsqueda», «sin checkpoints»---
que se lee exactamente igual que «esto todavía no corrió».

Que es lo que pasó. El ensayo corrió entero, y el informe que el dueño abre para
decidir qué arreglar antes de gastar cuota mostraba 6 tablas de 57 salidas y 23
renglones de «no hay nada que mostrar». Ni una sola falla: `search_record()` sin
coordenada leía el registro de la corrida COMPLETA, que no existe porque lo que
corrió fue un ensayo, y todo lo de abajo dijo la verdad sobre el archivo
equivocado.

La regla y sus dos mitades viven en `config.LECTURAS_SIN_COORDENADA`, arriba de
la excepción que declara. Acá se comprueban contra las firmas vivas y contra el
árbol de sintaxis de todo lo que puede llamarlas.
"""

import ast
import inspect
import json
from pathlib import Path

import pytest

from MIL_CREDA_Benchmark import config, contamination, harness, latent

#: El repositorio, desde este archivo. Igual que en `test_label_noise.py`: un
#: cuaderno no se importa, y su `source` es el único lugar donde está su código.
_REPOSITORIO = Path(__file__).resolve().parents[1]

#: Cómo se llama la coordenada de escala en una firma.
COORDENADA = "pilot"


def _fuentes():
    """`(archivo, código)` de todo lo que puede leer una corrida.

    El paquete, `tools/` y CADA celda de código de CADA cuaderno. Los cuadernos
    entran porque ahí estaban tres de los cuatro defectos: un cuaderno no se
    importa, así que ninguna regla que sólo mire `.py` los ve.

    `tests/` queda afuera a propósito, igual que en la regla de los destinos:
    acá se redirige `config.RESULTS` con `monkeypatch` en casi todos los
    fixtures, que es justo lo que el paquete no puede hacer.
    """
    for ruta in sorted((_REPOSITORIO / "src" / "MIL_CREDA_Benchmark").glob("*.py")):
        yield ruta.name, ruta.read_text(encoding="utf-8")
    for ruta in sorted((_REPOSITORIO / "tools").glob("*.py")):
        yield ruta.name, ruta.read_text(encoding="utf-8")
    for cuaderno in sorted((_REPOSITORIO / "MIL-CREDA" / "Notebooks").glob("*.ipynb")):
        documento = json.loads(cuaderno.read_text(encoding="utf-8"))
        for celda in documento.get("cells", []):
            if celda.get("cell_type") != "code":
                continue
            # Las mágicas de IPython y los `!` no son Python y romperían el
            # parseo; nada que lea una corrida vive en una de ellas.
            yield cuaderno.name, "\n".join(
                linea for linea in "".join(celda.get("source", [])).splitlines()
                if not linea.lstrip().startswith(("%", "!")))


def _puertas() -> dict[tuple[str, str], dict]:
    """Cada entrada que lleva escala, sacada de su propia firma.

    La clave es `(archivo, nombre)` y no el nombre solo: `latent.available`
    lleva escala y `contamination.available` no lleva ninguna, y una regla que
    resolviera por nombre pelado marcaría a la segunda por el defecto de la
    primera.

    `vigente` es la mitad que decide qué exige la regla: `pilot=None` significa
    «el que rige» y hace legal la llamada pelada; cualquier otro valor por
    omisión significa «la corrida completa» y obliga a decirlo.
    """
    puertas = {}
    for archivo, fuente in _fuentes():
        if not archivo.endswith(".py"):
            continue
        for nodo in ast.walk(ast.parse(fuente)):
            if not isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            posicionales = [a.arg for a in nodo.args.args]
            nombres = posicionales + [a.arg for a in nodo.args.kwonlyargs]
            if COORDENADA not in nombres:
                continue
            omision = dict(zip(
                posicionales[len(posicionales) - len(nodo.args.defaults):],
                nodo.args.defaults))
            omision.update(zip((a.arg for a in nodo.args.kwonlyargs),
                               nodo.args.kw_defaults))
            valor = omision.get(COORDENADA)
            puertas[(archivo, nodo.name)] = {
                "posicionales": posicionales,
                "nombres": nombres,
                "vigente": isinstance(valor, ast.Constant) and valor.value is None,
            }
    return puertas


def _resolver(nodo: ast.Call, archivo: str, puertas: dict):
    """Qué puerta llama esta expresión, o `None` si no llama a ninguna.

    `mod.f(...)` resuelve contra `mod.py`; `f(...)` contra el archivo donde
    está escrita. Una celda de cuaderno no define nada, así que sólo llega acá
    por la primera forma, que es como se llama a un paquete desde afuera.
    """
    if isinstance(nodo.func, ast.Attribute):
        if not isinstance(nodo.func.value, ast.Name):
            return None
        return puertas.get((f"{nodo.func.value.id}.py", nodo.func.attr))
    if isinstance(nodo.func, ast.Name):
        return puertas.get((archivo, nodo.func.id))
    return None


def _llamadas_sin_escala(puertas: dict | None = None) -> dict[str, str]:
    """Cada llamada que deja la escala decidida por la firma, con su puerta.

    Sólo cuentan las puertas cuya omisión significa «la corrida completa»: si
    significa «el que rige», la llamada pelada ES la forma correcta.
    """
    puertas = puertas if puertas is not None else _puertas()
    sueltas = {}
    for archivo, fuente in _fuentes():
        try:
            arbol = ast.parse(fuente)
        except SyntaxError:                                  # pragma: no cover
            continue
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            puerta = _resolver(nodo, archivo, puertas)
            if puerta is None or puerta["vigente"]:
                continue
            atados = (set(puerta["posicionales"][:len(nodo.args)])
                      | {kw.arg for kw in nodo.keywords})
            if COORDENADA not in atados:
                sueltas[f"{archivo}: {ast.unparse(nodo)}"] = puerta
    return sueltas


class TestCadaLecturaDiceDeQueCorridaSale:
    """La regla entera, derivada de la firma y del árbol, nunca de una lista."""

    def test_hay_puertas_que_derivar(self):
        """El polo que hace medible a todo lo demás.

        Sin esto, un `_puertas()` que devolviera `{}` ---un `glob` mal escrito,
        un `SyntaxError` tragado--- dejaría la regla verde sin haber mirado
        nada, que es la única forma en que este archivo puede mentir.
        """
        puertas = _puertas()
        assert len(puertas) >= 20, (
            f"sólo {len(puertas)} entradas llevan escala; el recorrido no está "
            f"leyendo el paquete")
        # Contra el disco y no contra un número escrito: `7` quedó viejo el día
        # que la búsqueda, el barrido y el diagnóstico tuvieron cada uno su
        # cuaderno propio, y un literal acá vuelve a quedar viejo en el próximo.
        en_disco = {ruta.name for ruta in
                    (_REPOSITORIO / "MIL-CREDA" / "Notebooks").glob("*.ipynb")}
        cuadernos = {a for a, _ in _fuentes() if a.endswith(".ipynb")}
        assert cuadernos == en_disco, (
            f"el recorrido no ve {sorted(en_disco - cuadernos)}")

    def test_ninguna_lectura_deja_la_escala_a_la_firma(self):
        """Lo que quede suelto tiene que estar declarado, y declararse cuesta
        escribir por qué.

        Rojo alcanzable: sacarle `pilot=` a cualquier llamada a `search_record`,
        `latent_grid`, `floors_agree`, `correspondence_grid`, `checkpoint_for`,
        `available` o `ceilings_in_force`, en un módulo, en `tools/` o en una
        celda de cualquiera de los siete cuadernos.
        """
        encontradas = set(_llamadas_sin_escala())
        declaradas = set(config.LECTURAS_SIN_COORDENADA)
        assert encontradas - declaradas == set(), (
            "estas llamadas dejan la escala decidida por la firma, así que leen "
            "la corrida completa sin decirlo -> "
            f"{sorted(encontradas - declaradas)}")
        assert declaradas - encontradas == set(), (
            "estas exclusiones le sobrevivieron a su llamada -> "
            f"{sorted(declaradas - encontradas)}")

    def test_toda_exclusion_dice_por_que(self):
        """La forma de `DESTINOS_SIN_COORDENADA` y de `CUADERNOS_SIN_PASO`: un
        nombre pelado en una lista y un olvido se leen igual."""
        sin_razon = [clave for clave, razon
                     in config.LECTURAS_SIN_COORDENADA.items()
                     if not razon.strip()]
        assert not sin_razon, f"excluidas sin decir por qué -> {sin_razon}"

    def test_la_omision_que_se_repliega_lo_hace_desde_config(self):
        """Las dos mitades de la regla existen las dos.

        Si NINGUNA puerta se replegara, la regla de arriba sería «pasá la
        coordenada siempre» y pasaría igual; si TODAS se replegaran, no habría
        nada que gobierne. Las dos tienen que existir o la regla no separa nada.
        """
        puertas = _puertas()
        vigentes = {n for (_, n), p in puertas.items() if p["vigente"]}
        completas = {n for (_, n), p in puertas.items() if not p["vigente"]}
        assert "search_record" in vigentes and "level_dir" in vigentes, (
            f"lo que se muestra tiene que replegarse al vigente -> {vigentes}")
        assert {"results_for", "models_for", "checkpoint_for"} <= completas, (
            f"lo que gobierna no puede replegarse -> {completas}")


class TestElRegistroQueGobiernaSigueSiendoCiegoAlEnsayo:
    """El polo contrario, y el que importa más.

    Un repliegue que llegara hasta acá sería mucho peor que el defecto que
    arregla: los techos de un ensayo ---tres épocas--- gobernando una campaña
    real, aceptados por el mismo rechazo que existe para impedirlo.
    """

    def test_la_campana_pide_el_registro_completo_por_su_nombre(self):
        """Derivado del cuerpo de `campaign`, no de una lista: la llamada que
        alimenta el rechazo por `atRequiredScale` tiene que nombrar la escala.

        Rojo alcanzable: volver esa línea a `search_record()`, que hoy
        contestaría con el registro del ensayo.
        """
        fuente = inspect.getsource(harness.campaign)
        assert "atRequiredScale" in fuente
        llamadas = [ast.unparse(n) for n in ast.walk(ast.parse(inspect.getsource(
            harness.campaign).lstrip()))
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "search_record"]
        assert llamadas, "`campaign` ya no lee el registro de la búsqueda"
        assert all("pilot=False" in ll for ll in llamadas), (
            f"la lectura que gobierna dejó de nombrar su escala -> {llamadas}")

    def test_pedir_la_escala_completa_no_ve_el_ensayo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
        monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD",
                            tmp_path / "ceilings.pilot.json")
        config.CEILINGS_PILOT_RECORD.write_text(
            json.dumps({"creda": {"ceiling": 0.01, "epochs": 3,
                                  "atRequiredScale": False}}), encoding="utf-8")
        assert harness.search_record(pilot=False) is None
        assert harness.search_record(pilot=True) is not None
        assert harness.search_record() is not None      # el vigente, para mostrar


class TestUnEnsayoNoSePuedeLeerComoUnaCorridaCompleta:
    """El repliegue sin la marca sería peor que la tabla vacía: no ser citable
    está bien, ser confundible no.

    Se reusa el mecanismo que ya existía para la campaña
    (`contamination.source_note`) en vez de inventar un segundo aviso, así que
    las dos mitades del informe avisan igual y con las mismas palabras.
    """

    def _solo_ensayo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
        monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD",
                            tmp_path / "ceilings.pilot.json")
        config.CEILINGS_PILOT_RECORD.write_text(json.dumps(
            {"creda": {"ceiling": 0.01, "epochs": 3, "seeds": [0],
                       "atRequiredScale": False,
                       "requiredScale": {"epochs": 20, "trials": 30}}}),
            encoding="utf-8")

    def test_el_aviso_grita_ensayo_cuando_lo_es(self, tmp_path, monkeypatch):
        self._solo_ensayo(tmp_path, monkeypatch)
        nota = harness.search_source_note()
        assert "ENSAYO" in nota and "**" in nota
        assert "3 épocas" in nota
        assert "no se citan como resultados" in nota

    def test_el_aviso_se_escribe_igual_cuando_no_lo_es(self, tmp_path, monkeypatch):
        """Siempre y no sólo en el caso malo: un aviso que aparece únicamente
        cuando algo anda mal no le enseña a nadie qué es lo que vigila."""
        monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
        monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD",
                            tmp_path / "ceilings.pilot.json")
        config.CEILINGS_RECORD.write_text(json.dumps(
            {"creda": {"ceiling": 0.01, "epochs": 20, "seeds": [0, 1, 2],
                       "atRequiredScale": True}}), encoding="utf-8")
        nota = harness.search_source_note()
        assert nota.strip()
        assert "ENSAYO" not in nota
        assert "20 épocas" in nota

    def test_sin_ningun_registro_lo_dice_y_no_finge_una_tabla(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
        monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD",
                            tmp_path / "ceilings.pilot.json")
        assert "Sin búsqueda de techos" in harness.search_source_note()

    def test_el_informe_muestra_el_aviso_antes_de_cada_tabla_de_techos(self):
        """Derivado del cuaderno y no de la memoria de quien lo escribió: cada
        celda que renderiza el registro de la búsqueda tiene que mostrar la
        procedencia también.

        Rojo alcanzable: borrar un `show(PROCEDENCIA_TECHOS)` de cualquiera de
        las dos celdas, o renderizar una tercera tabla de techos sin él.
        """
        documento = json.loads((_REPOSITORIO / "MIL-CREDA" / "Notebooks"
                                / "Benchmark_Report_v1.ipynb")
                               .read_text(encoding="utf-8"))
        celdas = ["".join(c.get("source", [])) for c in documento["cells"]
                  if c.get("cell_type") == "code"]
        assert any("PROCEDENCIA_TECHOS = harness.search_source_note()" in c
                   for c in celdas), "el informe no resuelve la procedencia"
        rinden = [c for c in celdas if "render_ceilings" in c]
        assert rinden, "el informe ya no muestra la rejilla de techos"
        sin_aviso = [c.strip()[:60] for c in rinden
                     if "PROCEDENCIA_TECHOS" not in c]
        assert not sin_aviso, (
            f"estas celdas muestran techos sin decir de qué escala -> {sin_aviso}")


class TestElCuadernoLatenteLeeLosPesosDeSuPropiaCorrida:
    """Las tres entradas que dibujan paneles llevan la escala, o el ensayo mira
    un árbol vacío y dibuja una grilla apagada sin un solo error."""

    @pytest.mark.parametrize("entrada", ["latent_grid", "correspondence_grid",
                                         "floors_agree"])
    def test_toda_entrada_que_dibuja_toma_la_escala(self, entrada):
        firma = inspect.signature(getattr(latent, entrada))
        assert {"rate", "pilot"} <= set(firma.parameters), (
            f"`latent.{entrada}` dibuja paneles y no puede decir de qué corrida "
            f"salen los pesos -> {list(firma.parameters)}")

    def test_el_cuaderno_le_pasa_su_propia_escala_a_las_tres(self):
        documento = json.loads((_REPOSITORIO / "MIL-CREDA" / "Notebooks"
                                / "Benchmark_Latent_v1.ipynb")
                               .read_text(encoding="utf-8"))
        celdas = "\n".join("".join(c.get("source", [])) for c in documento["cells"]
                           if c.get("cell_type") == "code")
        arbol = ast.parse("\n".join(l for l in celdas.splitlines()
                                    if not l.lstrip().startswith(("%", "!"))))
        vistas = set()
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            nombre = getattr(nodo.func, "attr", None)
            if nombre not in ("latent_grid", "correspondence_grid", "floors_agree"):
                continue
            vistas.add(nombre)
            texto = ast.unparse(nodo)
            assert "ES_ENSAYO" in texto or "pilot" in texto, (
                f"esta llamada dibuja con los pesos de otra corrida -> {texto}")
        assert vistas == {"latent_grid", "correspondence_grid", "floors_agree"}, (
            f"el cuaderno ya no llama a las tres -> {sorted(vistas)}")


def test_la_forma_del_repliegue_es_una_sola_en_todo_el_repositorio():
    """Dos ortografías de una regla es el defecto, no la regla.

    `Benchmark_Search_Report_v1` escribía el repliegue a mano ---`search_record() or
    search_record(pilot=True)`--- y por eso era el único cuaderno que mostraba
    sus tablas; el informe no lo tenía y no mostraba ninguna. El repliegue vive
    ahora en la firma, una vez, y escribirlo a mano otra vez cae acá.
    """
    a_mano = {}
    for archivo, fuente in _fuentes():
        try:
            arbol = ast.parse(fuente)
        except SyntaxError:                                  # pragma: no cover
            continue
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.BoolOp) or not isinstance(nodo.op, ast.Or):
                continue
            llamadas = [v for v in nodo.values if isinstance(v, ast.Call)]
            nombres = {getattr(v.func, "attr", getattr(v.func, "id", None))
                       for v in llamadas}
            # DOS llamadas, no una: `search_record(...) or {}` es un valor por
            # omisión para un `None`, no un repliegue entre dos escalas, y
            # marcarlo confundiría las dos cosas justo donde importa.
            if len(llamadas) >= 2 and len(nombres) == 1 and nombres <= {
                    "search_record", "level_dir", "load", "in_force"}:
                a_mano[f"{archivo}: {ast.unparse(nodo)}"] = sorted(nombres)
    assert not a_mano, (
        "estos escriben a mano el repliegue que la firma ya hace, que es la "
        f"segunda ortografía de una misma regla -> {a_mano}")


def test_la_puerta_que_se_repliega_lo_hace_por_el_resolutor_que_ya_existia():
    """`search_record` no inventa su propio orden de preferencia.

    Si lo escribiera aparte, «el completo gana siempre que exista» viviría en
    dos lugares y uno de los dos envejecería. Lo resuelve
    `config.ceilings_record_in_force`, que es el mismo que ya alimenta
    `ceilings_on_record` y `ceilings_provenance`.
    """
    assert "ceilings_record_in_force" in inspect.getsource(harness.search_record)
    assert "ceilings_record_in_force" in inspect.getsource(config.ceilings_on_record)


def test_el_gemelo_de_la_campana_sigue_teniendo_la_misma_forma():
    """La forma se copió de `contamination`, así que las dos tienen que seguir
    siendo la misma cosa: si una se mueve, la otra tiene que moverse con ella."""
    assert (inspect.signature(contamination.level_dir).parameters["pilot"].default
            is None)
    assert (inspect.signature(harness.search_record).parameters["pilot"].default
            is None)


def test_la_escala_configurada_se_lee_de_las_dos_constantes_que_la_declaran(
        monkeypatch):
    """`is_pilot_scale()` contra las dos constantes, medido y no leído.

    El encabezado de `config` dice que dos constantes separan al ensayo de la
    corrida completa, `EPOCHS` y `SEEDS`, y ninguna otra. Esta es la única
    lectura de esa regla, y las dos mitades se comprueban por separado porque
    cada una puede romperse sola: un predicado que mirara sólo las épocas diría
    «completa» sobre veinte épocas con una semilla, que es un ensayo.

    Y la lectura es «por debajo de», no «distinto de»: por encima de la escala
    declarada no hay ensayo que valga --- una corrida más larga que la completa
    no es una rehearsal --- y `<` lo dice mientras `!=` diría lo contrario.

    Rojo alcanzable: mirar una sola de las dos constantes, comparar por
    igualdad, o dar vuelta el sentido.
    """
    monkeypatch.setattr(config, "EPOCHS", config.FULL_EPOCHS)
    monkeypatch.setattr(config, "SEEDS", list(config.FULL_SEEDS))
    assert config.is_pilot_scale() is False, "la escala completa se leyó como ensayo"

    monkeypatch.setattr(config, "EPOCHS", config.FULL_EPOCHS - 1)
    assert config.is_pilot_scale() is True, "pocas épocas no se leyeron como ensayo"

    monkeypatch.setattr(config, "EPOCHS", config.FULL_EPOCHS)
    monkeypatch.setattr(config, "SEEDS", list(config.FULL_SEEDS)[:-1])
    assert config.is_pilot_scale() is True, "pocas semillas no se leyeron como ensayo"

    # Por ENCIMA de la escala declarada tampoco hay ensayo, y es la única mitad
    # que separa `<` de `!=`: sin este caso las dos ortografías dan lo mismo en
    # todo lo de arriba y la afirmación sobre el sentido no estaría medida.
    monkeypatch.setattr(config, "EPOCHS", config.FULL_EPOCHS + 1)
    monkeypatch.setattr(config, "SEEDS", list(config.FULL_SEEDS) + [99])
    assert config.is_pilot_scale() is False, (
        "una corrida más larga que la completa se leyó como ensayo")

    # y lo que el repositorio declara hoy: las dos por debajo, o sea un ensayo
    monkeypatch.undo()
    assert config.EPOCHS < config.FULL_EPOCHS
    assert len(config.SEEDS) < len(config.FULL_SEEDS)
    assert config.is_pilot_scale() is True
