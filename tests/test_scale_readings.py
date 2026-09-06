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
        alimenta el rechazo por `atRequiredScale` tiene que nombrar la escala
        COMPLETA, y ninguna otra lectura puede ocupar su lugar.

        `campaign` lee el registro dos veces y son dos preguntas distintas: la
        que GOBIERNA ---`searched`, sobre la que se exige `atRequiredScale`--- y
        la EVIDENCIA que se sella en `ceilingSearch`, que describe de dónde
        salió el escalar con el que esta corrida corrió. Compartían una
        variable, y compartirla sellaba una campaña de ensayo con la rejilla de
        la búsqueda completa. Se afirman las dos por separado y por su rol, no
        por «todas las llamadas dicen lo mismo»: esa forma anterior era verde
        mientras las dos preguntas fueran una, y se habría puesto en rojo
        justamente al separarlas.

        Rojo alcanzable, y son dos: volver `searched` a `search_record()` ---que
        hoy contestaría con el registro del ensayo y dejaría que los techos de
        un ensayo gobiernen una campaña real---, o volver el sello a `searched`.
        """
        fuente = inspect.getsource(harness.campaign)
        assert "atRequiredScale" in fuente
        arbol = ast.parse(fuente.lstrip())

        gobierna = [ast.unparse(nodo.value) for nodo in ast.walk(arbol)
                    if isinstance(nodo, ast.Assign)
                    and any(getattr(t, "id", None) == "searched"
                            for t in nodo.targets)]
        assert gobierna, "`campaign` ya no lee el registro que gobierna"
        assert all("pilot=False" in ll for ll in gobierna), (
            f"la lectura que gobierna dejó de nombrar su escala -> {gobierna}")

        # y el rechazo se alimenta de ESA lectura y no de la otra
        (rechazo,) = [ast.unparse(nodo.value) for nodo in ast.walk(arbol)
                      if isinstance(nodo, ast.Assign)
                      and any(getattr(t, "id", None) == "under"
                              for t in nodo.targets)]
        assert "searched" in rechazo, (
            f"el rechazo por escala dejó de leer el registro que gobierna -> "
            f"{rechazo}")

        sello = [ast.unparse(kw.value) for nodo in ast.walk(arbol)
                 if isinstance(nodo, ast.Call)
                 for kw in nodo.keywords if kw.arg == "ceilingSearch"]
        assert sello, "`campaign` ya no sella el registro en la reducción"
        assert all("reduction.pilot" in ll for ll in sello), (
            f"la evidencia sellada no es la de esta corrida -> {sello}")
        assert not any("pilot=False" in ll for ll in sello), (
            f"la evidencia volvió a ser la de la lectura que gobierna -> {sello}")

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
        # Derivado del árbol y no de la cadena literal: la llamada tiene que
        # existir Y nombrar la escala de la corrida que el informe dibuja. La
        # forma literal pasaba con una llamada pelada, que es lo que había ---
        # la rejilla de la búsqueda COMPLETA encabezando las tablas de una
        # campaña de ensayo.
        resuelven = []
        for celda in celdas:
            try:
                arbol = ast.parse("\n".join(
                    l for l in celda.splitlines()
                    if not l.lstrip().startswith(("%", "!"))))
            except SyntaxError:                              # pragma: no cover
                continue
            for nodo in ast.walk(arbol):
                if (isinstance(nodo, ast.Assign)
                        and any(getattr(t, "id", None) == "PROCEDENCIA_TECHOS"
                                for t in nodo.targets)
                        and isinstance(nodo.value, ast.Call)):
                    resuelven.append(ast.unparse(nodo.value))
        assert resuelven, "el informe no resuelve la procedencia"
        assert all("search_source_note" in r and "pilot=" in r
                   for r in resuelven), (
            "el aviso de procedencia no dice de qué escala habla, así que puede "
            f"describir un registro que esta corrida no consumió -> {resuelven}")
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


def test_la_preferencia_entre_los_dos_registros_se_consulta_en_un_solo_lugar():
    """Nadie escribe su propio orden de preferencia entre los dos registros.

    «El completo gana siempre que exista» es UNA regla, y vivía escrita a mano
    en `harness.search_record` además de en `config.ceilings_record_in_force`:
    dos ortografías, y la que quede vieja se lee igual de verde que la que no.
    Hoy la consulta `config.ceilings_record_at` --- la única puerta por la que
    una escala se vuelve un registro para todo lo que LEE --- y todos los
    lectores pasan por ella.

    Derivado del árbol y no de una lista: un lector nuevo que vuelva a
    preguntar por el vigente por su cuenta aparece acá sin que nadie lo agregue.

    Rojo alcanzable: volver a escribir la rama `if pilot is None:
    ceilings_record_in_force()` adentro de cualquier lector.
    """
    consultan = set()
    for archivo, fuente in _fuentes():
        try:
            arbol = ast.parse(fuente)
        except SyntaxError:                                  # pragma: no cover
            continue
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for llamada in ast.walk(nodo):
                if (isinstance(llamada, ast.Call)
                        and getattr(llamada.func, "attr",
                                    getattr(llamada.func, "id", None))
                        == "ceilings_record_in_force"):
                    consultan.add(f"{archivo}: {nodo.name}")
    assert consultan == {"config.py: ceilings_record_at"}, (
        "la preferencia entre los dos registros se consulta desde más de un "
        f"lugar, así que está escrita más de una vez -> {sorted(consultan)}")

    # Y la puerta la usan los cuatro lectores, o «una sola vez» sería cierto
    # porque nadie la usa.
    for lector in (config.ceilings_on_record, config.ceilings_by_transfer_on_record,
                   config.ceilings_provenance, harness.search_record):
        assert "ceilings_record_at" in inspect.getsource(lector), (
            f"`{lector.__name__}` resuelve el registro por su cuenta")


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


# --------------------------------------- la escala que el ámbito ya tiene
#
# La segunda mitad de la regla. `_llamadas_sin_escala` mira las puertas cuya
# omisión significa «la corrida completa»; una puerta cuya omisión significa «el
# que rige» acepta la llamada pelada, y ahí vivía el defecto que quedaba:
# `config.ceilings_on_record()` no tenía coordenada NINGUNA, así que una campaña
# de ensayo pedía sus techos y se llevaba los de la búsqueda completa.


def _cuadernos_con_escala() -> set[str]:
    """Los cuadernos que declaran `ES_ENSAYO` en alguna de sus celdas.

    El ámbito de un cuaderno es el CUADERNO y no la celda: `ES_ENSAYO` se
    declara una vez, arriba, y se usa quince celdas más abajo. Esta regla
    miraba celda por celda y por eso no veía nada de eso --- la celda de techos
    de `Benchmark_Campaign_v1` no declara `ES_ENSAYO`, la declara la celda 3 ---
    así que devolverle esa celda a la llamada pelada pasaba por acá sin una
    palabra. Lo agarró otro test, y «otro test lo agarra» no es lo que esta
    regla afirma de sí misma.
    """
    con_escala = set()
    for archivo, fuente in _fuentes():
        if not archivo.endswith(".ipynb"):
            continue
        try:
            arbol = ast.parse(fuente)
        except SyntaxError:                                  # pragma: no cover
            continue
        if any(isinstance(nodo, ast.Assign)
               and any(getattr(t, "id", None) == "ES_ENSAYO"
                       for t in nodo.targets)
               for nodo in ast.walk(arbol)):
            con_escala.add(archivo)
    return con_escala


def _ambitos_con_escala(archivo: str, arbol: ast.AST,
                        cuadernos: set[str] | None = None) -> list[ast.AST]:
    """Los trozos de este archivo que ya saben a qué corrida pertenecen.

    Una función con parámetro `pilot`, una cuyo cuerpo nombra `reduction.pilot`
    ---la escala viaja adentro de la reducción, que es como la lleva casi todo
    el paquete--- o cualquier celda de un cuaderno que declare `ES_ENSAYO`.
    Fuera de éstos nadie sabe de qué corrida se habla y «cuál rige» es la
    pregunta correcta.
    """
    if archivo.endswith(".ipynb"):
        cuadernos = cuadernos if cuadernos is not None else _cuadernos_con_escala()
        return [arbol] if archivo in cuadernos else []
    ambitos = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        argumentos = ({a.arg for a in nodo.args.args}
                      | {a.arg for a in nodo.args.kwonlyargs})
        if COORDENADA in argumentos or "reduction.pilot" in ast.unparse(nodo):
            ambitos.append(nodo)
    return ambitos


def _llamadas_que_no_reenvian(puertas: dict | None = None) -> set[str]:
    """Cada llamada a una puerta VIGENTE hecha desde un ámbito que ya tiene escala.

    Sólo las vigentes: para las otras ya rige `_llamadas_sin_escala`, y marcar
    las mismas dos veces confundiría dos reglas que fallan distinto.
    """
    puertas = puertas if puertas is not None else _puertas()
    cuadernos = _cuadernos_con_escala()
    sueltas = set()
    for archivo, fuente in _fuentes():
        try:
            arbol = ast.parse(fuente)
        except SyntaxError:                                  # pragma: no cover
            continue
        for ambito in _ambitos_con_escala(archivo, arbol, cuadernos):
            for nodo in ast.walk(ambito):
                if not isinstance(nodo, ast.Call):
                    continue
                puerta = _resolver(nodo, archivo, puertas)
                if puerta is None or not puerta["vigente"]:
                    continue
                atados = (set(puerta["posicionales"][:len(nodo.args)])
                          | {kw.arg for kw in nodo.keywords})
                if COORDENADA not in atados:
                    sueltas.add(f"{archivo}: {ast.unparse(nodo)}")
    return sueltas


class TestQuienSabeSuEscalaLaDice:
    """La mitad que faltaba, y la que se llevó los techos de la otra corrida."""

    def test_hay_ambitos_con_escala_que_recorrer(self):
        """El polo que hace medible a la regla de abajo.

        Sin esto, un `_ambitos_con_escala` que devolviera `[]` ---un cambio de
        nombre en `ES_ENSAYO`, un `walk` mal escrito--- dejaría la regla verde
        sin haber mirado nada, que es la única forma en que puede mentir.
        """
        vistos = {}
        for archivo, fuente in _fuentes():
            try:
                arbol = ast.parse(fuente)
            except SyntaxError:                              # pragma: no cover
                continue
            cuantos = len(_ambitos_con_escala(archivo, arbol))
            if cuantos:
                vistos[archivo] = vistos.get(archivo, 0) + cuantos
        assert sum(vistos.values()) >= 20, (
            f"el recorrido casi no encuentra ámbitos con escala -> {vistos}")
        # Y de los dos lados: el paquete y los cuadernos. Los cuadernos son la
        # mitad que ninguna regla que sólo mire `.py` puede ver, y ahí estaba
        # `ceilings_on_record()` pelado en la campaña y en el barrido.
        assert any(a.endswith(".py") for a in vistos), vistos
        assert any(a.endswith(".ipynb") for a in vistos), vistos

    def test_ninguna_lectura_que_ya_tiene_escala_deja_de_reenviarla(self):
        """Lo que quede suelto tiene que estar declarado, y declararse cuesta
        escribir por qué.

        Rojo alcanzable: sacarle `pilot=` a `config.ceilings_on_record(...)` en
        la celda de techos de la campaña o del barrido, a
        `config.ceilings_by_transfer_on_record(...)` en `with_ceilings_in_force`
        o en `campaign`, a `config.ceilings_provenance(...)` en
        `search_source_note`, o a `harness.search_record(...)` en el informe.
        """
        encontradas = _llamadas_que_no_reenvian()
        declaradas = set(config.LECTURAS_QUE_NO_REENVIAN)
        assert encontradas - declaradas == set(), (
            "estas llamadas se hacen desde un ámbito que YA sabe a qué corrida "
            "pertenece y no lo dicen, así que leen el registro que rige en vez "
            f"del suyo -> {sorted(encontradas - declaradas)}")
        assert declaradas - encontradas == set(), (
            "estas exclusiones le sobrevivieron a su llamada -> "
            f"{sorted(declaradas - encontradas)}")

    def test_toda_exclusion_de_reenvio_dice_por_que(self):
        sin_razon = [clave for clave, razon
                     in config.LECTURAS_QUE_NO_REENVIAN.items()
                     if not razon.strip()]
        assert not sin_razon, f"excluidas sin decir por qué -> {sin_razon}"

    def test_ningun_cuaderno_se_escribe_su_escala_a_mano(self):
        """La escala es un modo del RECORRIDO, así que ningún cuaderno la fija.

        `ES_ENSAYO` sale siempre de una lectura: `config.is_pilot_scale()` en
        los cuatro que corren, o el `pilot` del registro que se está dibujando
        en los que informan. Una constante ahí es la forma exacta que tenía el
        defecto que esto cierra --- `ES_ENSAYO = True` en el cuaderno de la
        búsqueda, con el que NINGÚN cuaderno podía correr la búsqueda completa,
        así que a escala completa los techos salían de la biblioteca y todo el
        resto del recorrido de un cuaderno.

        Rojo alcanzable: escribir `ES_ENSAYO = True` (o `False`) en cualquiera
        de los siete cuadernos.
        """
        a_mano = {}
        derivadas = 0
        for archivo, fuente in _fuentes():
            if not archivo.endswith(".ipynb"):
                continue
            try:
                arbol = ast.parse(fuente)
            except SyntaxError:                              # pragma: no cover
                continue
            for nodo in ast.walk(arbol):
                if not isinstance(nodo, ast.Assign):
                    continue
                if not any(getattr(t, "id", None) == "ES_ENSAYO"
                           for t in nodo.targets):
                    continue
                if isinstance(nodo.value, ast.Constant):
                    a_mano[f"{archivo}: {ast.unparse(nodo)}"] = archivo
                else:
                    derivadas += 1
        assert derivadas >= 6, (
            f"el recorrido casi no ve declaraciones de escala -> {derivadas}")
        assert not a_mano, (
            "estos cuadernos se escriben la escala en vez de recibirla, así que "
            f"el modo del recorrido no los alcanza -> {sorted(a_mano)}")


class TestLasDosMitadesDelTechoSalenDelMismoRegistro:
    """El valor más sensible del cálculo, y su procedencia.

    Un techo llega a la reducción en dos mitades ---el agrupado y el pick por
    transferencia--- leídas en dos llamadas separadas. La guarda débil es
    afirmar que la reducción «lleva techos»: pasa con las dos mitades llenas,
    pasa con las dos mitades sacadas del registro de OTRA corrida, y pasa con
    una mitad de cada registro. Lo que se afirma acá es de qué archivo salió
    cada mitad, y que es el mismo archivo que la escala de la corrida nombra.
    """

    def _dos_registros(self, tmp_path, monkeypatch):
        """Los DOS registros en disco a la vez, con valores distintos.

        Con un solo registro toda esta clase es verde por vacuidad: no hay
        segundo archivo del que leer mal. Y los valores tienen que diferir --- un
        fixture que escribiera el mismo techo en los dos haría verde justamente
        a la lectura equivocada, que es como este defecto sobrevivió.
        """
        monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
        monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD",
                            tmp_path / "ceilings.pilot.json")
        config.CEILINGS_RECORD.write_text(json.dumps({
            "milcreda": {"ceiling": 0.90, "epochs": 20, "seeds": [0, 1, 2],
                         "atRequiredScale": True,
                         "requiredScale": {"epochs": 20, "trials": 30},
                         "byTransfer": {"M->U": 0.91}}}), encoding="utf-8")
        config.CEILINGS_PILOT_RECORD.write_text(json.dumps({
            "milcreda": {"ceiling": 0.11, "epochs": 3, "seeds": [0],
                         "atRequiredScale": False,
                         "requiredScale": {"epochs": 20, "trials": 30},
                         "byTransfer": {"M->U": 0.12}}}), encoding="utf-8")
        return {True: (0.11, 0.12), False: (0.90, 0.91)}

    @pytest.mark.parametrize("escala", [True, False])
    def test_una_reduccion_lleva_las_dos_mitades_de_su_propia_escala(
            self, escala, tmp_path, monkeypatch):
        """Las dos mitades, contra el registro que la reducción declara.

        Rojo alcanzable, y son dos rojos distintos: sacarle `pilot=` a
        `config.ceilings_on_record` adentro de `ceilings_in_force` mueve la
        mitad agrupada, y sacárselo a `config.ceilings_by_transfer_on_record`
        adentro de `with_ceilings_in_force` mueve la otra. Cada mutación deja
        pasar a la guarda débil ---la reducción sigue llevando techos, y siguen
        siendo números plausibles--- y cae acá.

        La mutación que la guarda débil NO sobrevive y ésta sí distingue:
        moverle la escala a UNA sola de las dos. La reducción queda con el techo
        agrupado de un experimento y el pick de otro, los dos existen, los dos
        son plausibles, y ninguna afirmación sobre «lleva techos» los separa.
        """
        import torch

        esperado = self._dos_registros(tmp_path, monkeypatch)
        agrupado, por_transferencia = esperado[escala]

        reduccion = harness.Reduction(pilot=escala)
        salida = harness.with_ceilings_in_force(
            reduccion, torch.device("cpu"), progress=lambda *_: None)

        assert salida.pilot is escala
        assert salida.ceilings == {"milcreda": agrupado}, (
            "la mitad agrupada salió del registro de la otra escala")
        assert salida.ceilingsByTransfer == {"milcreda": {"M->U": por_transferencia}}, (
            "el pick por transferencia salió del registro de la otra escala")

        # Y las dos del MISMO archivo, dicho como una sola afirmación: cada
        # registro escribe su pooled y su pick con un dígito distinto, así que
        # dos mitades de archivos distintos no pueden pasar esto por casualidad.
        otro_agrupado, otro_pick = esperado[not escala]
        assert salida.ceilings["milcreda"] != otro_agrupado
        assert salida.ceilingsByTransfer["milcreda"]["M->U"] != otro_pick

    def test_la_guarda_debil_pasa_con_las_dos_mitades_del_vecino(
            self, tmp_path, monkeypatch):
        """Lo que la guarda débil no puede ver, medido y no argumentado.

        «La reducción lleva techos» es verdad de una reducción de ENSAYO
        cargada con el registro COMPLETO: las dos mitades están, no están
        vacías, y son números plausibles. Este test construye exactamente esa
        reducción a mano y comprueba que la guarda débil la acepta --- si algún
        día dejara de aceptarla, la afirmación de arriba sobre qué separa a una
        de la otra habría dejado de ser cierta y este archivo estaría
        exagerando lo que mide.
        """
        from dataclasses import replace

        esperado = self._dos_registros(tmp_path, monkeypatch)
        completo_agrupado, completo_pick = esperado[False]

        # una reducción de ENSAYO con las dos mitades de la corrida COMPLETA
        equivocada = replace(
            harness.Reduction(pilot=True),
            ceilings={"milcreda": completo_agrupado},
            ceilingsByTransfer={"milcreda": {"M->U": completo_pick}})

        assert equivocada.ceilings, "la guarda débil"
        assert equivocada.ceilingsByTransfer, "la guarda débil"
        # y sin embargo no es el registro de su escala
        agrupado_de_su_escala, _ = esperado[True]
        assert equivocada.ceilings != {"milcreda": agrupado_de_su_escala}


def test_el_recorrido_entero_lee_el_registro_de_su_escala_en_los_dos_modos():
    """La propiedad de punta a punta, sobre TODOS los lectores y de una sola vez.

    La afirmación del dueño ---en ensayo todo consume lo del ensayo; a escala
    completa no se usa nada del ensayo--- escrita como una comprobación y no
    como una por función: **agregarle a un disco el registro del vecino no
    puede cambiar nada de lo que este modo lee.**

    Se mide sobre cuatro estados del disco y sin saber qué forma devuelve cada
    lector, que es lo que la hace sobrevivir al próximo:

    1. sin ningún registro --- la respuesta base de cada lector;
    2. sólo el del ensayo --- `pilot=True` tiene que MOVERSE de esa base y
       `pilot=False` tiene que quedarse exactamente en ella;
    3. sólo el completo --- lo mismo al revés;
    4. los dos a la vez --- cada escala tiene que contestar *idéntico* a lo que
       contestaba cuando el otro archivo no existía.

    El paso 4 es el que importa y ninguno de los otros lo implica: un lector
    que se replegara al vigente pasa 2 y 3 enteros ---en cada uno hay un solo
    archivo, así que el repliegue acierta--- y sólo falla acá.

    Los lectores se DERIVAN: cualquier función del paquete que lleve `pilot` y
    resuelva por `ceilings_record_at` entra sola. Una lista escrita a mano se
    quedaría vieja con el próximo lector, que es exactamente cómo
    `ceilings_on_record` pasó tanto tiempo sin coordenada.

    Rojo alcanzable: sacarle la coordenada a cualquier lector, o hacer que
    `ceilings_record_at` ignore su argumento.
    """
    import inspect
    import tempfile

    completo = {"milcreda": {"ceiling": 0.93, "epochs": 21, "seeds": [0, 1, 2],
                             "atRequiredScale": True,
                             "requiredScale": {"epochs": 21, "trials": 31},
                             "byTransfer": {"M->U": 0.93}}}
    ensayo = {"milcreda": {"ceiling": 0.11, "epochs": 3, "seeds": [7],
                           "atRequiredScale": False,
                           "requiredScale": {"epochs": 21, "trials": 31},
                           "byTransfer": {"M->U": 0.11}}}

    lectores = {}
    for modulo in (config, harness):
        for nombre, entrada in vars(modulo).items():
            if not inspect.isfunction(entrada) or nombre.startswith("_"):
                continue
            if COORDENADA not in inspect.signature(entrada).parameters:
                continue
            try:
                cuerpo = inspect.getsource(entrada)
            except OSError:                                  # pragma: no cover
                continue
            if "ceilings_record_at" in cuerpo:
                lectores[f"{modulo.__name__.split('.')[-1]}.{nombre}"] = entrada
    assert len(lectores) >= 4, (
        f"el recorrido no encontró los lectores del registro -> {lectores}")

    with tempfile.TemporaryDirectory() as carpeta:
        raiz = Path(carpeta)
        anterior = (config.CEILINGS_RECORD, config.CEILINGS_PILOT_RECORD)
        config.CEILINGS_RECORD = raiz / "ceilings.json"
        config.CEILINGS_PILOT_RECORD = raiz / "ceilings.pilot.json"

        def leer(escala):
            return {n: f(pilot=escala) for n, f in sorted(lectores.items())}

        try:
            # 1. sin nada
            vacio = {escala: leer(escala) for escala in (True, False)}

            # 2. sólo el del ensayo
            config.CEILINGS_PILOT_RECORD.write_text(json.dumps(ensayo),
                                                    encoding="utf-8")
            solo_ensayo = leer(True)
            assert leer(False) == vacio[False], (
                "con sólo el registro de ENSAYO en disco, la escala completa "
                "contestó algo: se está sirviendo del archivo del vecino")
            movidos = [n for n in solo_ensayo if solo_ensayo[n] != vacio[True][n]]
            assert len(movidos) == len(lectores), (
                f"estos lectores no vieron su propio registro -> "
                f"{sorted(set(solo_ensayo) - set(movidos))}")
            config.CEILINGS_PILOT_RECORD.unlink()

            # 3. sólo el completo
            config.CEILINGS_RECORD.write_text(json.dumps(completo),
                                              encoding="utf-8")
            solo_completa = leer(False)
            assert leer(True) == vacio[True], (
                "con sólo el registro COMPLETO en disco, la escala de ensayo "
                "contestó algo: se está sirviendo del archivo del vecino")
            movidos = [n for n in solo_completa
                       if solo_completa[n] != vacio[False][n]]
            assert len(movidos) == len(lectores), (
                f"estos lectores no vieron su propio registro -> "
                f"{sorted(set(solo_completa) - set(movidos))}")

            # 4. los dos a la vez, que es el paso que los otros tres no implican
            config.CEILINGS_PILOT_RECORD.write_text(json.dumps(ensayo),
                                                    encoding="utf-8")
            assert leer(True) == solo_ensayo, (
                "aparecer el registro COMPLETO cambió lo que lee el modo de "
                "ensayo")
            assert leer(False) == solo_completa, (
                "aparecer el registro de ENSAYO cambió lo que lee el modo "
                "completo")
            assert solo_ensayo != solo_completa, (
                "los dos registros contestan igual, así que este test no "
                "puede distinguir nada")

            # El aviso que encabeza las tablas, que es lo que un lector humano
            # ve: con los dos archivos en disco no puede decir «completa»
            # sobre una corrida de ensayo.
            assert "ENSAYO" in harness.search_source_note(pilot=True)
            assert "ENSAYO" not in harness.search_source_note(pilot=False)
            assert "3 épocas" in harness.search_source_note(pilot=True)
            assert "21 épocas" in harness.search_source_note(pilot=False)

            # Y el resolutor sigue existiendo y sigue contestando «cuál rige»:
            # sin esto, «cada uno el suyo» podría haberse conseguido rompiendo
            # la omisión, que es la pregunta que un informe sin corrida propia
            # todavía necesita hacer.
            assert config.ceilings_provenance()["source"] == "full"
            config.CEILINGS_RECORD.unlink()
            assert config.ceilings_provenance()["source"] == "pilot"
        finally:
            config.CEILINGS_RECORD, config.CEILINGS_PILOT_RECORD = anterior
