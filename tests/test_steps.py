"""Los pasos locales que este repositorio le ofrece a la forja.

Nivel 1 - invariantes de estructura. No ejecuta ningún cuaderno: ejecutar uno
tarda minutos y lo que acá se afirma es qué se ofrece y qué no, no qué imprime.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import pytest

import MIL_CREDA as paquete
from MIL_CREDA_Benchmark import steps

#: La raíz del repositorio, derivada del módulo que ya la conoce
#: (`steps.CUADERNOS` es `<raíz>/MIL-CREDA/Notebooks`) y nunca escrita.
_RAIZ = steps.CUADERNOS.parents[1]

#: Los pasos que legítimamente no corren ningún cuaderno, cada uno con su
#: razón al lado --- la misma forma que `steps.CUADERNOS_SIN_PASO`, y por la
#: misma razón: un nombre pelado en una exención y un olvido se leen igual.
#:
#: Está VACÍO, y vacío es una afirmación: hoy todo paso declarado nombra un
#: cuaderno entre sus raíces.
#:
#: Tenía dos, `campaign` y `mechanisms`, con la razón «el artefacto que manda
#: a un worker no es un cuaderno, es un `run-config.json` que nombra
#: `{module, function}` directamente». La premisa era verdadera y se midió; la
#: conclusión no se seguía. Un `run-config.json` no puede nombrar un `.ipynb`
#: --- y por eso `harness.run_campaign_shard`/`run_mechanism_sweep_shard`
#: siguen intactas y siguen siendo ese camino --- pero las dos formas no
#: compiten: la remota manda una función a otra máquina, el cuaderno ejercita
#: acá el artefacto que un lector abre. Con la exención puesta, `verify`
#: reportaba los dos pasos bajo `undeclaredStepNotebooks` y el ensayo los
#: recorría probando la biblioteca.
#:
#: La forma sigue disponible: un paso puede eximirse acá, con su razón al
#: lado. Lo que no puede es eximirse sin decir por qué.
PASOS_SIN_CUADERNO_A_PROPOSITO: dict[str, str] = {}


class PasosDeclaradosTests(unittest.TestCase):

    def test_cada_paso_declarado_resuelve_a_un_llamable(self):
        for nombre, entrada in paquete.__steps__.items():
            with self.subTest(paso=nombre):
                self.assertEqual(entrada["module"], "MIL_CREDA_Benchmark.steps")
                funcion = getattr(steps, entrada["function"], None)
                self.assertTrue(callable(funcion),
                                f"{entrada['function']} no es llamable")

    def test_cada_paso_toma_cero_argumentos(self):
        """La forja llama sin argumentos: la declaración no lleva `kwargs`."""
        for nombre, entrada in paquete.__steps__.items():
            with self.subTest(paso=nombre):
                firma = inspect.signature(getattr(steps, entrada["function"]))
                self.assertEqual(list(firma.parameters), [])

    def test_la_campana_corre_su_cuaderno_en_vez_de_computar_en_su_lugar(self):
        """El defecto que el ensayo existe para no tener.

        `campana` llamaba a `harness.campaign()` directamente, así que el
        cuaderno que se envía --- el único de este árbol que se envía --- era el
        único que el ensayo nunca ejercitaba: `Benchmark_Campaign_v1.ipynb`
        llegó a tener cero celdas ejecutadas mientras el paso reportaba
        `returned`. Un ensayo que computa por su cuenta prueba la biblioteca y
        deja sin probar el artefacto.

        Las dos mitades se afirman por separado porque cada una puede volver
        sola: que corra el cuaderno, y que no vuelva a computar al lado de él.
        Ninguna de las dos se lee de la otra --- un paso que hiciera las dos
        cosas correría la campaña dos veces y reportaría `returned`.

        La biblioteca no se toca y este test no la vigila: la celda 8 del
        cuaderno llama a `harness.campaign()`, así que borrarla rompería lo que
        se quiere correr.

        Rojo alcanzable: devolver `harness.campaign(...)` al cuerpo del paso,
        apuntarlo a otro cuaderno, o sacarle el `_ejecutar`.
        """
        import ast

        entrada = paquete.__steps__["results"]
        self.assertEqual(entrada["function"], "resultados")
        self.assertEqual(
            _cuadernos_nombrados_por_los_pasos().get("Results.ipynb"),
            "resultados", "los resultados no corren su propio cuaderno")

        fuente = Path(steps.__file__).read_text(encoding="utf-8")
        (definicion,) = [nodo for nodo in ast.parse(fuente).body
                         if isinstance(nodo, ast.FunctionDef)
                         and nodo.name == "resultados"]
        computa = [nodo for nodo in ast.walk(definicion)
                   if isinstance(nodo, ast.Call)
                   and isinstance(nodo.func, ast.Attribute)
                   and nodo.func.attr in ("campaign", "run_search",
                                          "search_ceilings", "run_one")]
        self.assertEqual(computa, [],
                         "los resultados computan al lado del cuaderno que corre")

    def test_la_busqueda_de_ensayo_corre_su_cuaderno_en_vez_de_computar_en_su_lugar(
            self):
        """La reversión, medida: la búsqueda tiene cuaderno propio y lo corre.

        Este test afirmaba lo contrario --- que `ensayo_de_busqueda` se quedaba
        en biblioteca porque el único cuaderno de la búsqueda era su INFORME, y
        apuntar los dos pasos ahí le habría dado dos dueños a una raíz
        declarada. El razonamiento era bueno y el hecho que lo sostenía dejó de
        ser cierto: la búsqueda tiene el suyo, y cada raíz sigue con un dueño.

        Dos mitades, cada una capaz de volver sola: que el paso corra
        `Benchmark_Ceiling_Search.ipynb`, y que no compute al lado del cuaderno
        que corre.

        Eran tres. La tercera afirmaba que el INFORME de la búsqueda
        (`Benchmark_Search_Report_v1.ipynb`, paso `search-report`) no volviera
        a buscar --- abrirlo tenía que costar lo que cuesta leer y no lo que
        cuesta correr la búsqueda. Ese cuaderno fue retirado entero junto con
        su paso y su función, así que la afirmación no quedó débil: se quedó
        sin sujeto. Su razón vive hoy en
        `CUADERNOS_QUE_NO_EXISTEN_A_PROPOSITO`, y
        `test_toda_exencion_sigue_nombrando_algo_que_de_verdad_no_existe` es lo
        que se pone en rojo si el nombre vuelve.

        Rojo alcanzable: devolver `harness.run_search(...)` al cuerpo del paso,
        o apuntarlo a otro cuaderno.
        """
        import ast

        entrada = paquete.__steps__["search-pilot"]
        self.assertEqual(entrada["function"], "ensayo_de_busqueda")
        corridos = _cuadernos_nombrados_por_los_pasos()
        self.assertEqual(corridos.get("Benchmark_Ceiling_Search.ipynb"),
                         "ensayo_de_busqueda",
                         "la búsqueda no corre su propio cuaderno")

        fuente = Path(steps.__file__).read_text(encoding="utf-8")
        (definicion,) = [nodo for nodo in ast.parse(fuente).body
                         if isinstance(nodo, ast.FunctionDef)
                         and nodo.name == "ensayo_de_busqueda"]
        computa = [nodo for nodo in ast.walk(definicion)
                   if isinstance(nodo, ast.Call)
                   and isinstance(nodo.func, ast.Attribute)
                   and nodo.func.attr in ("campaign", "run_search",
                                          "search_ceilings", "run_one")]
        self.assertEqual(computa, [],
                         "el paso computa al lado del cuaderno que corre")

    def test_cada_paso_declara_exactamente_un_cuaderno_entre_sus_raices(self):
        """Los dos ejes de cada mitad, cada uno con su artefacto, mirado desde
        los pasos.

        `test_el_cuaderno_que_corre_cada_paso_esta_entre_sus_raices` va de los
        cuadernos hacia los pasos, así que un paso que no corra ninguno queda
        fuera de su recorrido y su silencio se lee igual que un acierto. Éste va
        al revés y por eso ve lo que aquél no: un paso que computa sin abrir
        ningún cuaderno.

        Es lo que costaba: la búsqueda, el barrido y el diagnóstico computaban
        en la biblioteca y tenían cuaderno sólo del lado que dibuja, así que el
        ensayo los recorría probando la biblioteca y dejando sin ejercitar el
        artefacto --- que es el que un lector abre y el que después se manda.

        Exactamente uno y no «al menos uno»: dos cuadernos en un mismo paso son
        dos dueños de una ejecución `--inplace`, y la segunda le pisa a la
        primera lo único que deja.

        Un paso puede eximirse declarándose en `PASOS_SIN_CUADERNO_A_
        PROPOSITO`, con su razón al lado -- la misma disciplina que
        `steps.CUADERNOS_SIN_PASO` ya impone del otro lado. Eximir cuesta
        escribir por qué; un nombre pelado no alcanza.

        Rojo alcanzable: declarar un paso que compute sin correr un cuaderno
        y sin eximirlo, sacarle el `Notebooks/...` a las raíces de un paso no
        eximido, o eximir uno sin razón.
        """
        sin_cuaderno, con_varios = [], []
        for nombre, entrada in paquete.__steps__.items():
            if nombre in PASOS_SIN_CUADERNO_A_PROPOSITO:
                continue
            cuadernos = [raiz for raiz in entrada.get("produces", [])
                         if raiz.endswith(".ipynb")]
            if not cuadernos:
                sin_cuaderno.append(nombre)
            elif len(cuadernos) > 1:
                con_varios.append((nombre, cuadernos))
        self.assertEqual(sin_cuaderno, [],
                         "estos pasos no abren ningún cuaderno, así que lo que "
                         "el ensayo ejercita en ellos es la biblioteca y no el "
                         "artefacto")
        self.assertEqual(con_varios, [],
                         "un paso con dos cuadernos ejecuta `--inplace` dos "
                         "veces y la segunda le pisa a la primera")
        for nombre, razon in PASOS_SIN_CUADERNO_A_PROPOSITO.items():
            with self.subTest(paso=nombre):
                self.assertIn(nombre, paquete.__steps__,
                             "la exención le sobrevivió al paso")
                self.assertTrue(razon.strip(), "eximido sin decir por qué")

    # `test_el_cuaderno_de_campana_deriva_su_escala_en_vez_de_escribir_en_la_completa`
    # is removed. `campana`/`Benchmark_Campaign_v1.ipynb` and the
    # `campaign-local` step are retired: nothing in this stretch's restructured
    # `__steps__` calls `harness.campaign()` from a notebook any more (see
    # `resultados`'s own docstring in `steps.py`). `Results.ipynb` derives
    # its scale through `cargar_corridas()` -- a resolver, not a hand-built
    # `Reduction(pilot=...)` -- and that discipline is already covered
    # generically, notebook-name-agnostic, by
    # `test_scale_readings.py::TestCadaLecturaDiceDeQueCorridaSale.test_ninguna_lectura_deja_la_escala_a_la_firma`.

    def test_todo_cuaderno_del_arbol_lo_corre_un_paso_o_dice_por_que_no(self):
        """El defecto no es un cuaderno sin paso: es que nada lo note.

        `Benchmark_Noise_Diagnostic_Report_v1.ipynb` vivió con cinco celdas y ninguna
        ejecutada -- la tabla y la conclusión que separan *falló el término* de
        *le faltó coeficiente*, computadas y sin dibujar -- porque ningún paso
        lo nombraba y ninguna afirmación miraba el disco. Las dos mitades se
        derivan: los cuadernos que se corren salen del árbol de `steps` a través
        de `__steps__`, los que existen salen del directorio, y lo que sobra
        tiene que ser exactamente lo que `CUADERNOS_SIN_PASO` excluye.

        Una lista de dos nombres escrita acá sería el mismo defecto una
        indirección más allá. Por eso el resto se compara contra la exclusión
        declarada, y la exclusión tiene que traer su razón: excluir cuesta
        escribir por qué, y un nombre pelado no alcanza.

        Rojo alcanzable: sacarle el `_ejecutar` a un paso de informe, agregar un
        cuaderno de resultados que nadie corra, o excluir uno sin razón.
        """
        corridos = _cuadernos_nombrados_por_los_pasos()
        en_disco = {ruta.name for ruta in steps.CUADERNOS.glob("*.ipynb")}
        self.assertTrue(en_disco, f"{steps.CUADERNOS} no tiene cuadernos")

        self.assertEqual(en_disco - set(corridos), set(steps.CUADERNOS_SIN_PASO),
                         "un cuaderno de este árbol no lo corre ningún paso y "
                         "tampoco está declarado como excluido")
        # Y al revés: un paso que nombre un cuaderno que no está falla recién al
        # correrlo, cuando el flujo que lo precede ya se gastó.
        self.assertEqual(set(corridos) - en_disco, set(),
                         "un paso corre un cuaderno que no está en el árbol")

        for cuaderno, razon in steps.CUADERNOS_SIN_PASO.items():
            with self.subTest(cuaderno=cuaderno):
                self.assertTrue((steps.CUADERNOS / cuaderno).is_file(),
                                "la exclusión le sobrevivió al cuaderno")
                self.assertTrue(razon.strip(), "excluido sin decir por qué")

    def test_un_cuaderno_ausente_se_rechaza_en_vez_de_correr(self):
        with self.assertRaises(FileNotFoundError):
            steps._ejecutar("no_existe.ipynb")


#: Los `.ipynb` que este repositorio nombra a propósito sin que existan, cada
#: uno con su razón al lado --- la misma forma que `CUADERNOS_SIN_PASO`, y por
#: la misma razón: un nombre pelado en una exención y un olvido se leen igual.
CUADERNOS_QUE_NO_EXISTEN_A_PROPOSITO: dict[str, str] = {
    "no_existe.ipynb": ("el cuaderno que `_ejecutar` tiene que rechazar; "
                        "que exista sería la falla que ese test mide"),
    # Los nombres LIBERADOS, que la prosa del repositorio nombra para decir de
    # dónde viene cada renombre. Están acá y no borrados del texto porque un
    # nombre liberado que vuelve es peor que uno colgado: una referencia vieja
    # sigue resolviendo, contra otro artefacto, y nadie ve nada. Declarados así,
    # `test_toda_exencion_sigue_nombrando_algo_que_de_verdad_no_existe` se pone
    # en rojo el día que alguien recicle uno.
    "Benchmark_Search_v1.ipynb": (
        "liberado: era el INFORME de la búsqueda y pasó a llamarse "
        "`Benchmark_Search_Report_v1.ipynb`, porque su nombre prometía la "
        "búsqueda y lo que hacía era presentarla. Ese sucesor tampoco existe "
        "ya --- ver su propia entrada acá abajo --- y ninguno de los dos se "
        "recicla para el cuaderno que sí la corre"),
    "Benchmark_Search_Report_v1.ipynb": (
        "retirado entero, con su paso (`search-report`) y su función "
        "(`steps.informe_de_busqueda`): el informe de la búsqueda no es un "
        "resultado de este paper. Lo que la búsqueda deja es el registro de "
        "techos, que ya atraviesa el recorrido entero --- lo consumen el "
        "barrido y la campaña ---, y lo que un lector juzga es la corrida que "
        "corrió bajo esos techos, no la elección de los techos presentada "
        "como experimento propio. `tables.render_ceilings`/"
        "`render_ceilings_by_transfer`/`conclusion_ceilings`/"
        "`conclusion_ceilings_by_transfer` siguen declarados y siguen con sus "
        "propias pruebas (`tests/test_ceiling_readers.py`); lo que se fue es "
        "el cuaderno que los mostraba"),
    "Benchmark_Search_Pilot_v1.ipynb": (
        "liberado: era el cuaderno que corre la búsqueda y fijaba `pilot=True` "
        "adentro, así que su nombre afirmaba una escala que no le toca elegir "
        "--- y mientras la afirmaba ningún cuaderno podía correr la búsqueda "
        "completa. Hoy se llama `Benchmark_Ceiling_Search.ipynb` y recibe su "
        "escala"),
    # Los seis borrados junto con la reestructuración de este stretch (commit
    # `2f9bf32`, "delete every artefact the new structure will not
    # overwrite"): sus checkpoints, registros y cuadernos medían el árbol de
    # r17 y ninguno se regenera en el lugar. `Benchmark_Noise_Sweep.ipynb`
    # -- el séptimo que ese commit retiró -- no entra acá: este stretch lo
    # repuso, y su nombre vuelve a resolver contra el árbol.
    "Benchmark_Campaign_v1.ipynb": (
        "borrado junto con el resto: era el cuaderno que corría la campaña "
        "completa (`harness.campaign()`) y el `campaign-local` step que lo "
        "ejecutaba. Sin sucesor notebook-driven -- el `campaign` step que "
        "este stretch agrega a `__steps__` corre remoto, sin cuaderno propio "
        "(ver su entrada más abajo)"),
    "Benchmark_Report_v1.ipynb": (
        "borrado: era el informe de la campaña completa. "
        "`Results.ipynb` lo reemplaza, junto con "
        "`Benchmark_Latent_v1.ipynb`, en un solo cuaderno -- las seis "
        "secciones que reemplazan a las dos leen el mismo registro y se "
        "citan entre sí"),
    "Benchmark_Latent_v1.ipynb": (
        "borrado: era el análisis latente por separado. Ver "
        "`Benchmark_Report_v1.ipynb`, con quien se fusionó en "
        "`Results.ipynb`"),
    "Benchmark_Noise_Report_v1.ipynb": (
        "borrado junto con el eje de ruido de dos pasos (barrido + informe): "
        "`noise-sweep` (`barrido_de_ruido`) hoy corre y presenta en un solo "
        "cuaderno, `Benchmark_Noise_Sweep.ipynb`"),
    "Benchmark_Noise_Diagnostic_Search_v1.ipynb": (
        "borrado junto con el diagnóstico de ruido entero: re-buscaba el "
        "techo de la adaptación bajo material contaminado, lo que "
        "contradice la decisión de que la búsqueda del techo corre siempre "
        "sobre material limpio (ver la nota de retiro en `config.py`)"),
    "Benchmark_Noise_Diagnostic_Report_v1.ipynb": (
        "borrado junto con el diagnóstico de ruido entero -- ver "
        "`Benchmark_Noise_Diagnostic_Search_v1.ipynb`"),
}


class OrdinalesDelRecorridoTests(unittest.TestCase):
    """El orden del recorrido, declarado en `advances`, y lo que tiene que cumplir.

    Un paso sin ordinal corre sin puerta y la forja lo dice así: queda AFUERA
    del recorrido ordenado. Cuatro de los diez que hubo estaban afuera --- todo
    el eje del ruido --- y el lugar que les tocaba no era una decisión libre: el
    barrido va entre la búsqueda y la campaña.

    Ningún número se escribe acá abajo: los ordinales se contrastan contra
    `len(__steps__)` y contra la cadena derivada, así que retirar un paso o
    agregar uno no deja una cuenta vieja en verde. Un título que diga «los
    diez» sí envejece, y por eso estos no lo dicen.

    Las dos afirmaciones de abajo son distintas y ninguna se lee de la otra. Que
    todos tengan ordinal no dice nada sobre el orden, y un orden que respete
    la cadena puede tener un hueco o un empate.
    """

    #: El ordinal de cada paso, leído de la declaración y nunca escrito acá: una
    #: copia y el original se pueden desalinear, y entonces este test afirmaría
    #: que dos listas de este archivo coinciden entre sí.
    def _ordinales(self) -> dict[str, int]:
        return {nombre: entrada.get("advances")
                for nombre, entrada in paquete.__steps__.items()}

    def test_todo_paso_lleva_ordinal_y_son_uno_a_N_sin_repetir(self):
        """Tantos ordinales como pasos declarados, sin hueco y sin empate.

        `N` es `len(__steps__)` y nunca un literal: el día que un paso se
        retira ---`search-report` se retiró--- una cuenta escrita acá seguiría
        en verde sobre una numeración con un agujero.

        Rojo alcanzable: sacarle el `advances` a cualquier paso, darle a dos el
        mismo número, o retirar un paso sin renumerar los que quedan.
        """
        ordinales = self._ordinales()
        sin_ordinal = sorted(n for n, o in ordinales.items()
                             if not isinstance(o, int) or isinstance(o, bool))
        self.assertEqual(sin_ordinal, [],
                         "un paso sin ordinal queda fuera del recorrido ordenado")
        self.assertEqual(sorted(ordinales.values()),
                         list(range(1, len(paquete.__steps__) + 1)),
                         "los ordinales no son 1..N sin repetir")

    def test_ningun_ordinal_pone_a_un_paso_delante_de_lo_que_lee(self):
        """El orden se contrasta contra la cadena DERIVADA y no contra una lista.

        `steps.predecesores` sale de `reads` y `produces`; nadie nombra a nadie.
        Un ordinal que dejara a un paso corriendo antes de lo que lee es peor
        que no tener ordinal: sin ordinal el paso corre sin puerta y se ve; con
        el ordinal equivocado la puerta lo deja pasar y afirma que estaba en su
        lugar.

        Rojo alcanzable: intercambiar los ordinales de `noise-sweep` y
        `noise-report`, o bajar `report` por debajo de `campaign-local`.
        """
        ordinales = self._ordinales()
        for nombre in paquete.__steps__:
            for predecesor in steps.predecesores(nombre):
                with self.subTest(paso=nombre, predecesor=predecesor):
                    self.assertLess(
                        ordinales[predecesor], ordinales[nombre],
                        f"{nombre} corre en el puesto {ordinales[nombre]} y lee "
                        f"lo que produce {predecesor}, que corre en el "
                        f"{ordinales[predecesor]}")

    def test_cada_ordinal_cae_en_el_item_que_atestigua_el_cuaderno_de_ese_paso(self):
        """Las dos mitades del recorrido dicen lo mismo.

        `advances` no es un número suelto: nombra el ítem de la secuencia de
        posición para el que ese paso produce evidencia, y la forja hace ese
        join (`_pilot_notebooks`). Declarar el ordinal de un lado y no moverlo
        del otro deja a un paso produciendo evidencia para el ítem de otro, sin
        que nada se ponga rojo.

        Sólo los ítems que atestiguan un CUADERNO entran: el del registro de
        techos y el del shard atestiguan otra cosa, y exigirles un cuaderno
        sería pedirle a este join algo que no afirma.

        Rojo alcanzable: renumerar `advances` sin volver a correr `position`, o
        insertar un ítem en la secuencia sin renumerar `advances`.
        """
        secuencia = _secuencia_de_posicion()
        self.assertEqual(len(secuencia), len(paquete.__steps__),
                         "la secuencia y los pasos declarados ya no son tantos")
        for nombre, entrada in paquete.__steps__.items():
            testigo = secuencia[entrada["advances"]]
            if not testigo.endswith(".ipynb"):
                continue
            with self.subTest(paso=nombre):
                self.assertEqual(Path(testigo).name, steps.cuaderno_de(nombre))


def _secuencia_de_posicion() -> dict[int, str]:
    """`{ordinal: operando del testigo}`, leído del bloque que `position` escribe.

    Del archivo y no de una copia: `AGREED.md` es lo que un humano lee y lo que
    la forja mide, así que es la mitad que tiene que coincidir con `advances`.
    """
    import re

    texto = (_RAIZ / "MIL-CREDA" / "AGREED.md").read_text(encoding="utf-8")
    bloque = texto.split("<!-- position ")[1].split("<!-- /position -->")[0]
    encontrados = {}
    for linea in bloque.splitlines():
        hallazgo = re.match(r"^- \[.\] (\d+)\. .* `@\w+(?::level)? (.+)`$", linea)
        if hallazgo:
            encontrados[int(hallazgo.group(1))] = hallazgo.group(2)
    assert encontrados, "no se leyó ningún ítem del bloque de posición"
    return encontrados


class ReferenciasACuadernosTests(unittest.TestCase):
    """Todo `.ipynb` que este repositorio se nombra a sí mismo está en el árbol.

    La afirmación no es «el cuaderno del paso existe» --- eso lo dice ya
    `test_todo_cuaderno_del_arbol_lo_corre_un_paso_o_dice_por_que_no`, y sigue
    verde mientras cualquier OTRO archivo siga nombrando el nombre viejo. Lo
    que se afirma acá es que ningún lugar del repositorio quedó nombrando un
    cuaderno que ya no está: el paso, la declaración, un test, la prosa de
    `AGREED.md` o el texto de otro cuaderno.

    Es la mitad que un renombre rompe y nada más mira. `Benchmark_Search_v1`
    pasó a llamarse `Benchmark_Search_Report_v1` y su ruta aparecía además
    como testigo en `AGREED.md`, donde ningún test la habría leído.
    """

    #: Dónde puede vivir un cuaderno de este repositorio. Dos, porque `CREDA/`
    #: es trabajo previo que este proyecto no edita y sus cuadernos igual se
    #: nombran desde `src/CREDA/schedules.py`.
    DIRECTORIOS = ("MIL-CREDA/Notebooks", "CREDA/Notebooks")
    #: Lo que no se lee: producto, entornos y cachés. `Results/` y `Models/`
    #: quedan afuera porque son salida y no texto de este repositorio.
    SALTEADOS = {".git", ".venv", "__pycache__", ".pytest_cache", ".scratch",
                 ".ipynb_checkpoints", ".benchmark-data", ".implementation",
                 "Results", "Models", ".domain-adaptation-cache", ".atl"}
    LEIDOS = {".py", ".md", ".txt", ".cfg", ".toml", ".ipynb"}

    def _texto_del_repositorio(self):
        """`[(ruta relativa, texto)]` de todo lo que puede nombrar un cuaderno.

        Un `.ipynb` se abre como JSON y se devuelve sólo el `source` de sus
        celdas: leer su archivo crudo mezclaría el código con las salidas
        guardadas, y una salida vieja nombrando un cuaderno viejo no es una
        referencia que nadie siga.
        """
        import json

        raiz = _RAIZ
        for ruta in sorted(raiz.rglob("*")):
            if not ruta.is_file() or ruta.suffix not in self.LEIDOS:
                continue
            if self.SALTEADOS & set(ruta.relative_to(raiz).parts):
                continue
            if ruta.suffix == ".ipynb":
                documento = json.loads(ruta.read_text(encoding="utf-8"))
                texto = "\n".join("".join(c.get("source", []))
                                  for c in documento["cells"])
            else:
                texto = ruta.read_text(encoding="utf-8")
            yield ruta.relative_to(raiz).as_posix(), texto

    def test_ninguna_referencia_a_un_cuaderno_quedo_apuntando_a_la_nada(self):
        """Rojo alcanzable: renombrar un cuaderno y arreglar sólo `steps.py`,
        dejando la declaración, un test o el testigo de `AGREED.md` nombrando
        el nombre viejo.
        """
        import re

        en_disco = {ruta.name for directorio in self.DIRECTORIOS
                    for ruta in (_RAIZ / directorio).glob("*.ipynb")}
        self.assertTrue(en_disco, "no se encontró ningún cuaderno en el árbol")

        colgadas = []
        for archivo, texto in self._texto_del_repositorio():
            for nombre in re.findall(r"[A-Za-z0-9_.-]+\.ipynb", texto):
                if nombre in en_disco or nombre in CUADERNOS_QUE_NO_EXISTEN_A_PROPOSITO:
                    continue
                colgadas.append(f"{archivo} nombra {nombre}")
        self.assertEqual(colgadas, [],
                         "estas referencias apuntan a un cuaderno que no está")

    def test_toda_exencion_sigue_nombrando_algo_que_de_verdad_no_existe(self):
        """Exentar cuesta escribir por qué, y la exención no le sobrevive al
        hecho: el día que uno de estos nombres exista, la exención lo estaría
        tapando en vez de declararlo.

        Rojo alcanzable: crear `no_existe.ipynb`, o exentar un nombre sin razón.
        """
        for nombre, razon in CUADERNOS_QUE_NO_EXISTEN_A_PROPOSITO.items():
            self.assertTrue(razon.strip(), f"{nombre} exento sin decir por qué")
            for directorio in self.DIRECTORIOS:
                self.assertFalse((_RAIZ / directorio / nombre).exists(),
                                 f"{nombre} existe: la exención lo tapa")


class RaicesDeclaradasTests(unittest.TestCase):
    """Qué escribe cada paso, declarado donde la forja lo lee.

    La forja fotografía la carpeta de producto antes y después de cada corrida
    y grada lo que cambió contra las raíces que el paso declara. Un paso sin
    `produces` no se grada: una corrida que no escribió nada se lee igual que
    una que produjo todo, y una que escribió en el árbol del vecino igual que
    una que se quedó en el suyo. Las dos fallas ya pasaron acá, las dos en
    silencio y las dos reportando `outcome: "returned"`.

    Las cuatro afirmaciones se derivan --- de `__steps__`, del árbol de
    `steps.py` y de `config` --- y ninguna lleva una lista escrita a mano: una
    lista acá sería el mismo defecto una indirección más allá.
    """

    def test_cada_paso_declara_las_raices_que_escribe(self):
        """Presente y con forma, para todos, sin nombrar a ninguno.

        La forma es la que `cmd_step` valida del otro lado: una lista NO vacía
        de cadenas no vacías. La lista vacía no es "este paso no escribe nada"
        --- la forja la rechaza con `STEP_MALFORMED` y `verify` la cuenta como
        no declarada --- así que un paso que no escribiera nada no tendría
        cómo decirlo y no hay ninguno acá.

        Rojo alcanzable: agregar un paso sin `produces`, o vaciarle la lista a
        cualquiera de los declarados.
        """
        self.assertTrue(paquete.__steps__, "no hay pasos declarados")
        for nombre, entrada in paquete.__steps__.items():
            with self.subTest(paso=nombre):
                raices = entrada.get("produces")
                self.assertIsInstance(raices, list,
                                      f"{nombre} no declara qué escribe")
                self.assertTrue(raices, f"{nombre} declara una lista vacía")
                for raiz in raices:
                    self.assertIsInstance(raiz, str)
                    self.assertTrue(raiz.strip(), f"{nombre} declara una raíz vacía")

    def test_ninguna_raiz_declarada_sale_de_la_carpeta_de_producto(self):
        """Relativa a `<producto>/` y sin salirse, medido y no leído.

        No se inspeccionan los caracteres de la cadena: se compone contra
        `config.PRODUCT` y se comprueba que el resultado siga adentro. Un
        `..` en el medio, una raíz absoluta o un `/` al principio se caen ahí,
        que es la misma prueba que hace la forja antes de correr el paso.

        Rojo alcanzable: declarar `/tmp/algo`, `../otro-repo/Results` o
        `/Results/Benchmark`.
        """
        from MIL_CREDA_Benchmark import config

        producto = config.PRODUCT.resolve()
        for nombre, raiz in _raices_declaradas():
            with self.subTest(paso=nombre, raiz=raiz):
                self.assertFalse(Path(raiz).is_absolute(),
                                 "una raíz absoluta no es relativa a nada")
                self.assertNotIn("..", Path(raiz).parts, "la raíz se sale trepando")
                self.assertIn(producto, (producto / raiz).resolve().parents,
                              f"{raiz} cae fuera de {producto}")

    def test_ninguna_raiz_es_de_dos_pasos(self):
        """La colisión que toda esta declaración existe para hacer visible.

        No alcanza con que las cadenas sean distintas: la forja decide la
        pertenencia por SEGMENTOS (`_owns`), así que una raíz que contiene a la
        de otro paso se traga sus escrituras y las lee como propias. Eso es lo
        que pasó de verdad --- el barrido escribiendo en el directorio de
        checkpoints de la campaña --- y por eso acá se prohíbe la contención y
        no sólo la igualdad.

        `Results/Benchmark/latent` y `Results/Benchmark/latent.json` conviven
        justamente porque la lectura es por segmentos y no por prefijo de
        cadena.

        Rojo alcanzable: copiarle las raíces al paso de al lado, o declarar el
        directorio padre de una raíz ajena.
        """
        declaradas = _raices_declaradas()
        colisiones = [f"{paso} declara {raiz}, que se come {otra} de {otro_paso}"
                      for paso, raiz in declaradas
                      for otro_paso, otra in declaradas
                      if paso != otro_paso and _contiene(raiz, otra)]
        self.assertEqual(colisiones, [], "dos pasos se disputan una raíz")

    def test_el_cuaderno_que_corre_cada_paso_esta_entre_sus_raices(self):
        """Se ejecuta `--inplace`, así que el cuaderno es producto suyo.

        Los dos lados se derivan: qué cuaderno corre cada paso sale del árbol
        de `steps.py` (`_cuadernos_nombrados_por_los_pasos`), y las raíces de
        la declaración. Un paso que copiara las raíces del vecino queda con el
        cuaderno del vecino declarado y el suyo no, y eso se ve acá aunque las
        cadenas copiadas fueran únicas.

        Rojo alcanzable: declarar `Notebooks/` a secas, nombrar el cuaderno del
        vecino, u olvidarse del cuaderno en un paso que lo ejecuta.
        """
        for cuaderno, funcion in _cuadernos_nombrados_por_los_pasos().items():
            paso = next(nombre for nombre, entrada in paquete.__steps__.items()
                        if entrada["function"] == funcion)
            with self.subTest(paso=paso, cuaderno=cuaderno):
                self.assertIn(f"Notebooks/{cuaderno}",
                              paquete.__steps__[paso]["produces"],
                              f"{paso} ejecuta {cuaderno} en el lugar y no lo declara")

    def test_las_raices_de_las_corridas_son_las_que_compone_config(self):
        """Cada destino, recompuesto desde el helper que lo elige.

        Ninguna de estas rutas está escrita acá: salen de `results_for`,
        `models_for` y `ceilings_record_for`, que son la autoridad sobre dónde
        cae una corrida. Si alguien mueve el segmento `Pilot/`, cambia
        `NOISE_REPORTED` o le cambia el formato a `rho{...}`, la declaración
        deja de nombrar el lugar donde el paso escribe y esto se pone en rojo
        --- que es la única forma de que un literal en `__init__.py` siga
        siendo verdad.

        La escala es específica a propósito: los cuatro pasos que corren a
        escala de ensayo declaran sólo ese árbol, mientras que los cuadernos del
        veredicto, que dibujan sobre la corrida vigente, declaran las dos
        escalas. No hay una raíz que cubra las dos sin cubrir también el árbol
        de todos los demás pasos.

        `campaign-local` lleva además su cuaderno, porque lo corre `--inplace`.
        No está escrito acá tampoco: sale del árbol de `steps.py`, igual que
        todo el resto. Lo que se afirma es la POSICIÓN y que no sobre ninguna
        otra raíz --- que el cuaderno esté declarado ya lo dice
        `test_el_cuaderno_que_corre_cada_paso_esta_entre_sus_raices`, y las dos
        cosas se rompen por separado.

        Rojo alcanzable: declarar la raíz completa donde el paso escribe la de
        ensayo, o al revés.
        """
        from MIL_CREDA_Benchmark import config

        def relativa(ruta) -> str:
            return ruta.relative_to(config.PRODUCT).as_posix()

        def cuaderno_de(funcion: str) -> str:
            (nombre,) = [c for c, f in _cuadernos_nombrados_por_los_pasos().items()
                         if f == funcion]
            return f"Notebooks/{nombre}"

        rho = config.NOISE_REPORTED
        curva = config.results_for(0.0, "curve", True).parent

        esperado = {
            "search-pilot": [relativa(config.ceilings_record_for(True)),
                             cuaderno_de("ensayo_de_busqueda")],
            "noise-sweep": [relativa(curva),
                            relativa(config.models_for(0.0, "curve", True).parent),
                            cuaderno_de("barrido_de_ruido")],
            # `results` es local y presenta lo que ya exista: su única raíz
            # propia, además de su cuaderno, es el árbol de figuras sin
            # segmento de escala -- ver `config.DESTINOS_SIN_COORDENADA` para
            # la nota completa sobre por qué no lleva coordenada.
            "results": ["Results/figures", cuaderno_de("resultados")],
        }
        for paso, raices in esperado.items():
            with self.subTest(paso=paso):
                self.assertEqual(paquete.__steps__[paso]["produces"], raices)


def _raices_declaradas() -> list[tuple[str, str]]:
    """`[(paso, raíz)]` para todo lo que `__steps__` declara, en orden."""
    return [(nombre, raiz)
            for nombre, entrada in paquete.__steps__.items()
            for raiz in entrada.get("produces", [])]


def _contiene(raiz: str, otra: str) -> bool:
    """Si `otra` cae bajo `raiz`, por segmentos y nunca por prefijo de cadena.

    La misma lectura que hace `_owns` del otro lado, escrita acá porque la
    forja no es importable desde este intérprete: `Results/one` no se come a
    `Results/one-more`, y esa diferencia es la que separa un guarda de una
    comparación de cadenas con forma de guarda.
    """
    partes, otras = Path(raiz).parts, Path(otra).parts
    return otras[:len(partes)] == partes


def _celdas_de_codigo(cuaderno: str) -> list[str]:
    """Las celdas de código de un cuaderno, como texto.

    Un cuaderno es JSON, así que leer su archivo con `rg` o buscarle una
    subcadena al texto crudo mezcla el código con las salidas guardadas y con el
    escape de cada línea. Acá se abre como lo que es y se devuelve sólo lo que
    se ejecuta.
    """
    import json

    documento = json.loads(
        (steps.CUADERNOS / cuaderno).read_text(encoding="utf-8"))
    return ["".join(celda["source"]) for celda in documento["cells"]
            if celda["cell_type"] == "code"]


#: Lo que marca a las celdas de arranque, que son DOS: la que la forja es dueña
#: ---resuelve la raíz contra `Path.cwd()`, que adentro de la suite es la raíz
#: del repositorio y no la carpeta del cuaderno--- y la que le sigue, que
#: manosea `sys.path`. Adentro de la suite el paquete ya está importado, así que
#: ninguna de las dos hace falta y las dos hacen daño.
_ARRANQUE = ("resolve_repository_root", "sys.path.insert")


def _correr_las_celdas(cuaderno: str, ambito: dict | None = None) -> dict:
    """Ejecuta las celdas de código de un cuaderno menos las de arranque.

    Cuáles son se deriva de su contenido y no de un índice --- una celda que se
    agregue arriba correría el índice y dejaría este helper saltándose otra
    cosa --- y se EXIGE que cada marca aparezca: una marca que ya no está en
    ningún cuaderno filtra cero celdas y se lee exactamente igual que un filtro
    que anda.

    Las celdas se compilan juntas y no de a una: un cuaderno es un solo
    programa, y ejecutar la tercera sin la segunda probaría algo que nadie va a
    correr nunca.
    """
    ambito = {} if ambito is None else ambito
    fuentes = _celdas_de_codigo(cuaderno)
    celdas = [c for c in fuentes if not any(m in c for m in _ARRANQUE)]
    faltan = [m for m in _ARRANQUE if not any(m in c for c in fuentes)]
    assert not faltan, (
        f"{cuaderno} ya no tiene ninguna celda con {faltan}, así que este filtro "
        "dejó de sacar la de arranque y la corre igual. Pasó: la marca era "
        "`find_repository`, los cuadernos adoptaron la celda de la forja, "
        "ninguna la definía más, el filtro no sacó nada y la suite siguió verde")
    exec(compile("\n".join(celdas), f"<{cuaderno}>", "exec"), ambito)
    return ambito


def _cuadernos_nombrados_por_los_pasos() -> dict[str, str]:
    """Qué cuaderno corre cada paso DECLARADO: `{cuaderno: función}`.

    Leído del código y no de una lista escrita a mano que puede quedar vieja, y
    de `__steps__` hacia adentro y no de cualquier literal del módulo: una
    función que corriera un cuaderno sin estar declarada no lo pone al alcance
    de nadie, y una mención en una docstring no lo ejecuta.
    """
    import ast

    declaradas = {entrada["function"] for entrada in paquete.__steps__.values()}
    fuente = Path(steps.__file__).read_text(encoding="utf-8")
    corridos: dict[str, str] = {}
    for definicion in ast.parse(fuente).body:
        if (not isinstance(definicion, (ast.FunctionDef, ast.AsyncFunctionDef))
                or definicion.name not in declaradas):
            continue
        for nodo in ast.walk(definicion):
            if not (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)
                    and nodo.func.id == "_ejecutar"):
                continue
            (argumento,) = nodo.args
            assert isinstance(argumento, ast.Constant), (
                f"{definicion.name} nombra su cuaderno con algo que no es un "
                "literal, así que esta lectura dejó de verlo")
            corridos[argumento.value] = definicion.name
    return corridos


# --------------------------------------------------------------- el eje de ruido
#
# Los dos pasos del eje se ejecutan de verdad, con `harness` sustituido: lo que
# se afirma acá es a qué nivel corre cada uno, sobre qué transferencia, con qué
# brazos, cuántas mediciones paga y qué deja escrito. Nada de eso se puede leer
# de las constantes -- `NOISE_DIAGNOSTIC_LEVEL` puede estar bien mientras el paso
# arma su reducción con otra cosa, y las dos cosas dan una suite verde.


def _quien_llama(fuente: str, nombre: str) -> set[str]:
    """Las funciones de nivel superior que llaman a `nombre`, leídas del árbol.

    Del código y no de una lista escrita a mano: un tercer lector aparecería sin
    que nadie actualice la lista, que es exactamente el caso que esta afirmación
    existe para atrapar.
    """
    import ast

    arbol = ast.parse(fuente)
    llaman = set()
    for definicion in arbol.body:
        if not isinstance(definicion, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for nodo in ast.walk(definicion):
            if (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)
                    and nodo.func.id == nombre):
                llaman.add(definicion.name)
    return llaman


def _sin_maquina(monkeypatch, tmp_path):
    """El dispositivo, el ambiente y el árbol de salida, fuera del camino.

    `RESULTS` redirigido y no sólo `PRODUCT`: `results_for` deriva todo de
    `RESULTS`, así que redirigir el otro dejaría el paso escribiendo en el árbol
    de la corrida real.
    """
    import torch

    from MIL_CREDA_Benchmark import config, harness

    monkeypatch.setattr(config, "PRODUCT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models" / "Benchmark")
    monkeypatch.setattr(harness, "resolve_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(harness, "environment", lambda: {"stub": True})


# `_correr_el_diagnostico` and the noise diagnostic (its notebook,
# `NOISE_DIAGNOSTIC_ARMS`/`NOISE_DIAGNOSTIC_LEVEL`, and the `noise-diagnostic`/
# `noise-diagnostic-report` steps) are removed entirely with this stretch.
# Reason, from `config.py`'s own retirement note: the diagnostic re-searched
# the ceiling under contamination, which contradicts the decision that the
# search always runs on clean material.


# `test_el_diagnostico_corre_en_el_tope_del_rango_y_paga_una_sola_medicion` and
# `test_los_numeros_del_diagnostico_no_entran_en_las_tablas_del_veredicto` are
# removed along with the noise diagnostic itself -- see the retirement note
# above `_correr_el_diagnostico` used to sit at.

def _correr_el_barrido(monkeypatch, tmp_path) -> dict:
    """Ejecuta el cuaderno del barrido sin máquina, y devuelve lo que pidió.

    Las corridas se registran con la lectura de techos que había en el momento
    de pedirlas, que es lo que permite afirmar «una sola vez, ARRIBA del bucle»
    en vez de «una sola vez en total»: las dos son verdes hoy y se rompen
    distinto.
    """
    from MIL_CREDA_Benchmark import bags, config, harness

    _sin_maquina(monkeypatch, tmp_path)
    def _registro(pilot=None):
        escalas["registro"].append(pilot)
        return {"stub": True}

    monkeypatch.setattr(harness, "search_record", _registro)
    monkeypatch.setattr(harness, "search_ceilings", lambda *a, **k: pytest.fail(
        "el barrido lanzó una búsqueda: mediría el ruido y el coeficiente a la vez"))
    monkeypatch.setattr(harness, "with_ceilings_in_force",
                        lambda *a, **k: pytest.fail(
                            "el barrido resolvió techos en vez de leerlos: a "
                            "escala completa esa llamada es la búsqueda entera"))
    monkeypatch.setattr(bags, "build", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(harness, "run_one", lambda *a, **k: {"seconds": 1.0})

    lecturas = {"agrupado": 0, "por_transferencia": 0}
    escalas = {"agrupado": [], "por_transferencia": [], "registro": []}
    corridas = []

    # Un valor DISTINTO por escala en cada mitad. Un doble que contestara lo
    # mismo por las dos haría verde a un cuaderno que lee el registro
    # equivocado: la lectura existiría, contaría uno, y el número que llega a
    # `campaign` sería el correcto por casualidad. Con dos valores, el que llega
    # dice de qué archivo salió.
    AGRUPADO = {True: {"milcreda": 1e-2, "creda": 1e-4},
                False: {"milcreda": 9e-1, "creda": 9e-1}}
    POR_TRANSFERENCIA = {True: {"milcreda": {"M->U": 1e-2}},
                         False: {"milcreda": {"M->U": 9e-1}}}

    def _agrupado(pilot=None):
        lecturas["agrupado"] += 1
        escalas["agrupado"].append(pilot)
        return AGRUPADO[bool(pilot)]

    def _por_transferencia(pilot=None):
        lecturas["por_transferencia"] += 1
        escalas["por_transferencia"].append(pilot)
        return POR_TRANSFERENCIA[bool(pilot)]

    monkeypatch.setattr(config, "ceilings_on_record", _agrupado)
    monkeypatch.setattr(config, "ceilings_by_transfer_on_record", _por_transferencia)

    def _campaign(reduccion, dispositivo, **kwargs):
        corridas.append({"reduction": reduccion, "lecturas": dict(lecturas),
                         **kwargs})
        return {"runs": []}

    monkeypatch.setattr(harness, "campaign", _campaign)

    ambito = _correr_las_celdas("Benchmark_Noise_Sweep.ipynb")
    return {"corridas": corridas, "lecturas": lecturas, "escalas": escalas,
            "agrupadoPorEscala": AGRUPADO,
            "porTransferenciaPorEscala": POR_TRANSFERENCIA,
            "devuelto": ambito["corridos"], "ambito": ambito}


def test_el_barrido_lee_los_techos_una_vez_y_los_mantiene_en_los_cinco_niveles(
        tmp_path, monkeypatch, capsys) -> None:
    """El coeficiente elegido en limpio, aplicado sucio, sobre una transferencia.

    Buscar por nivel multiplicaría 2 familias x 6 transferencias x 30 trials x 20
    épocas por cinco: otra campaña entera antes de la campaña. Así que el barrido
    no busca nada -- lee el registro UNA vez, antes del bucle, y esa misma
    lectura viaja a los cinco niveles. Que sea una sola lectura es la mitad que
    puede romperse sola: leer adentro del bucle daría los mismos números hoy y
    dejaría el barrido a merced de un registro que cambie a mitad de corrida.

    Y corre sobre M->U y ninguna otra, en cada uno de los cinco niveles: la
    transferencia está fijada por la menor distancia entre dominios, que es una
    propiedad del material y no de ninguna medición. Una transferencia ya cerca
    de su piso en rho 0 no tiene de dónde caer y no puede mostrar curva.

    Se ejecutan las celdas del cuaderno y no el paso: desde que el paso sólo
    corre `_ejecutar`, todo esto vive en el cuaderno.

    Rojo alcanzable: buscar techos adentro del barrido, releerlos por nivel,
    correr las seis transferencias, o saltearse un nivel declarado.
    """
    from MIL_CREDA_Benchmark import config

    corrido = _correr_el_barrido(monkeypatch, tmp_path)
    capsys.readouterr()
    corridas, lecturas = corrido["corridas"], corrido["lecturas"]

    # los techos se leen una sola vez, antes del bucle
    assert lecturas == {"agrupado": 1, "por_transferencia": 1}
    # y ninguna adentro: las cinco pasadas vieron la misma cuenta, y es la final
    assert [c["lecturas"] for c in corridas] == [lecturas] * len(config.NOISE_LEVELS)

    # un nivel declarado por corrida, en el orden declarado, y ninguno de más
    assert [c["reduction"].labelNoise for c in corridas] == config.NOISE_LEVELS
    assert sorted(corrido["devuelto"]) == sorted(
        f"{t:g}" for t in config.NOISE_LEVELS)
    assert config.NOISE_LEVELS[0] == 0.0, "el primer nivel es el limpio"

    # y los mismos techos en los cinco
    techos = {id(c["reduction"].ceilings) for c in corridas}
    assert len(techos) == 1, "algún nivel corrió con otros techos"
    for corrida in corridas:
        assert corrida["reduction"].ceilings == {"milcreda": 1e-2, "creda": 1e-4}
        assert corrida["reduction"].ceilingsByTransfer == {"milcreda": {"M->U": 1e-2}}
        # sobre M->U y ninguna otra, y escrito como curva y no como campaña
        assert corrida["transfers"] == [config.NOISE_TRANSFER]
        assert corrida["reduction"].kind == "curve"

    # la transferencia es la del eje, no una elegida acá
    assert config.NOISE_TRANSFER == ("M", "U")
    assert len(corridas) == len(config.NOISE_LEVELS) == 5
    # una sola transferencia y no las seis del veredicto
    assert len(config.VERDICT_TRANSFERS) > 1


# ------------------------------------------------- la escala de cada cuaderno
#
# UNA regla y no dos, y antes eran dos opuestas. La escala es un modo del
# RECORRIDO: los cuatro cuadernos que corren ---la búsqueda, la campaña, el
# barrido y el diagnóstico--- la derivan de `config.is_pilot_scale()`, y los
# cuatro pasos que los ejecutan se niegan cuando esa lectura no es la del
# ensayo, porque sus `produces` nombran la raíz de ensayo y ninguna otra.
#
# La búsqueda era la excepción: fijaba `True` adentro de su cuaderno, con el
# argumento de que derivarla dejaría que el tamaño de la CAMPAÑA decidiera si
# se lanza la corrida larga. El argumento era bueno y su conclusión costaba más
# de lo que compraba --- con `True` fijo NINGÚN cuaderno podía correr la
# búsqueda completa, así que a escala completa los techos salían de la
# biblioteca y todo el resto del recorrido de un cuaderno ---, y la autorización
# que ese `True` protegía ya vivía, para los otros tres, en la guarda del paso.


def _correr_la_busqueda(monkeypatch, tmp_path, escala_de_ensayo: bool) -> dict:
    """Ejecuta el cuaderno de la búsqueda con la escala configurada que se pida."""
    from MIL_CREDA_Benchmark import config, harness

    _sin_maquina(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "is_pilot_scale", lambda: escala_de_ensayo)

    pedidas = []

    def _run_search(shard=None, pilot=False):
        pedidas.append({"shard": shard, "pilot": pilot})
        return {"milcreda": {"ceiling": 1e-2}, "creda": {"ceiling": 1e-4}}

    monkeypatch.setattr(harness, "run_search", _run_search)
    monkeypatch.setattr(harness, "search_ceilings", lambda *a, **k: pytest.fail(
        "el cuaderno del ensayo buscó por su cuenta en vez de pedir `run_search`"))
    monkeypatch.setattr(harness, "with_ceilings_in_force", lambda *a, **k: pytest.fail(
        "el cuaderno del ensayo llamó a `with_ceilings_in_force`: sin "
        "`ceilings.json` esa llamada ES la búsqueda completa, unas nueve horas y "
        "media que nadie autorizó"))

    ambito = _correr_las_celdas("Benchmark_Ceiling_Search.ipynb")
    return {"pedidas": pedidas, "ambito": ambito}


def test_el_cuaderno_de_la_busqueda_sigue_la_escala_configurada_y_su_paso_se_niega(
        tmp_path, monkeypatch, capsys) -> None:
    """Las dos mitades, que sólo valen juntas --- la misma forma que el barrido.

    **El cuaderno deriva.** Con la escala configurada en la completa pide la
    búsqueda COMPLETA y escribe `ceilings.json`; en ensayo pide el ensayo y
    escribe `ceilings.pilot.json`. Este test afirmaba lo contrario --- que con
    la escala en la completa el cuaderno seguía pidiendo el ensayo --- y esa era
    la regla vieja. Lo que costaba está medido: con `True` fijo adentro, ningún
    cuaderno del árbol podía correr la búsqueda a escala completa, así que a
    escala completa los techos venían de `harness.run_search` mientras cada otra
    pieza del recorrido venía de un cuaderno. Es la misma divergencia entre lo
    que el ensayo ejercita y lo que la corrida real hace, un nivel más abajo.

    **El paso se niega.** La autorización que aquel `True` protegía no se
    perdió: vive donde ya vivía para los otros tres, en la guarda del paso, que
    declara `ceilings.pilot.json` y ninguna otra raíz. Un cuaderno de búsqueda
    es el lugar más fácil del árbol para que entre la corrida larga sin que
    nadie la pida ---`ceilings.json` son unas nueve horas y media y gobierna
    toda campaña---, y ya pasó una vez con `with_ceilings_in_force` lanzándola
    desde una celda sin decirlo.

    Y el destino sale de la MISMA respuesta que la escala, no de otra: un
    cuaderno que ensayara y escribiera igual en `ceilings.json` gastaría la
    respuesta de la búsqueda con la de un ensayo. Por eso las dos escalas se
    corren acá y no sólo la configurada de hoy: a escala de ensayo un `True`
    fijo y una lectura derivada dan el mismo número, y sólo una de las dos es
    una puerta.

    Rojo alcanzable: volver a fijar `ES_ENSAYO` en el cuaderno, componer el
    destino con una constante en vez de con `ES_ENSAYO`, o sacarle la guarda de
    escala al paso.
    """
    from MIL_CREDA_Benchmark import config, steps

    for escala in (True, False):
        corrido = _correr_la_busqueda(monkeypatch, tmp_path, escala)
        capsys.readouterr()
        assert corrido["pedidas"] == [{"shard": None, "pilot": escala}], (
            f"con la escala configurada en {escala!r} el cuaderno pidió otra "
            f"cosa -> {corrido['pedidas']}")
        assert corrido["ambito"]["ES_ENSAYO"] is escala
        assert corrido["ambito"]["DESTINO"] == config.ceilings_record_for(escala)
        assert (corrido["ambito"]["DESTINO"]
                != config.ceilings_record_for(not escala))

    # la mitad del paso: se niega antes de abrir el cuaderno
    monkeypatch.setattr(config, "is_pilot_scale", lambda: False)
    monkeypatch.setattr(steps, "_ejecutar", lambda nombre: pytest.fail(
        f"el paso abrió {nombre} a escala completa, fuera de sus raíces: "
        "`ceilings.json` es el registro que gobierna toda campaña"))
    with pytest.raises(SystemExit) as caido:
        steps.ensayo_de_busqueda()
    assert "escala" in str(caido.value)


def test_el_barrido_sigue_la_escala_configurada_y_su_paso_se_niega(
        tmp_path, monkeypatch, capsys) -> None:
    """La regla contraria, y su otra mitad, que sólo valen juntas.

    El barrido es una forma de campaña y escribe donde una campaña escribe, así
    que deriva su escala de la misma lectura que la campaña
    (`config.is_pilot_scale()`) en vez de fijarla. Fijada en `True`, una corrida
    de veinte épocas y treinta semillas se archivaba bajo `Pilot/`: una medición
    completa etiquetada como ensayo, que es la falla inversa de la que
    `is_pilot_scale` existe para impedir y del mismo tamaño.

    Derivar sola sería peor que fijar: `produces` nombra el árbol de ENSAYO y
    ninguno más, así que a escala completa el cuaderno escribiría donde nadie lo
    vigila y la forja lo leería como `foreign`. Por eso el paso se niega antes
    de abrir el cuaderno, y las dos mitades se afirman juntas porque cada una
    sin la otra es un defecto.

    (Esta prueba cubría también al diagnóstico de ruido; se retiró junto con
    él -- ver la nota de retiro sobre `_correr_el_diagnostico`.)

    Rojo alcanzable: fijar `ES_ENSAYO = True` en el cuaderno, o sacarle la
    guarda de escala al paso.
    """
    from MIL_CREDA_Benchmark import config, steps

    # la mitad del cuaderno: sigue la lectura, no una constante
    monkeypatch.setattr(config, "is_pilot_scale", lambda: False)
    corrido = _correr_el_barrido(monkeypatch, tmp_path)
    capsys.readouterr()
    assert corrido["corridas"], "el barrido no corrió ningún nivel"
    for corrida in corrido["corridas"]:
        assert corrida["reduction"].pilot is False, (
            "el barrido fijó su escala en vez de leerla, así que a escala "
            "completa archivaría bajo `Pilot/`")

    # la mitad del paso: se niega antes de abrir el cuaderno
    monkeypatch.setattr(steps, "_ejecutar", lambda nombre: pytest.fail(
        f"el paso abrió {nombre} a escala completa, fuera de sus raíces"))
    with pytest.raises(SystemExit) as caido:
        steps.barrido_de_ruido()
    assert "escala" in str(caido.value)


# ----------------------------------------------------------- el ensayo remoto
#
# La regla, en las palabras del dueño: el ensayo que ocurre EN EL WORKER tiene
# que usar únicamente los resultados previos de los cuadernos que corrieron en
# modo COMPLETO --- salvo el primero, que no depende de ningún otro cuaderno.
#
# Lo que estas pruebas tienen que separar es un ensayo que LEYÓ de uno que leyó
# LO QUE CORRESPONDE. Una afirmación de que «el ensayo leyó algo» la pasa un
# ensayo que se llevó `SMOKE_CEILINGS` --- el neutral declarado del módulo --- o
# el registro del ensayo de al lado, que es exactamente lo que la regla prohíbe.
# Por eso cada doble de acá contesta un valor DISTINTO por escala, y lo que se
# afirma es cuál de los dos llegó.


def _sin_ensayo_remoto(monkeypatch) -> None:
    """El modo apagado, dicho y no supuesto.

    `config.is_rehearsal()` lee el entorno del proceso, y la suite corre adentro
    de uno que puede traerlo puesto de afuera --- una consola que ensayó a mano,
    un `ensayo_remoto` que murió sin restaurar. Sin esto, la mitad de estas
    pruebas mediría el entorno de quien las corre.
    """
    from MIL_CREDA_Benchmark import config

    monkeypatch.delenv(config.REHEARSAL_ENV, raising=False)


def test_la_cadena_de_pasos_se_deriva_de_las_dos_listas(monkeypatch) -> None:
    """Quién depende de quién, leído de `reads` contra `produces` y de nada más.

    Lo que se afirma es la cadena entera y no una arista: la búsqueda no depende
    de nadie ---escribe el registro de techos y no lo lee--- `verification`
    tampoco, porque corre la suite sobre `src/MIL_CREDA`, que viaja con el clon y
    no lo produce ningún paso, y `campaign`/`mechanisms` tampoco: los dos
    consumen el registro de techos a escala COMPLETA, y ningún paso declarado
    lo produce -- la búsqueda completa es la entrada única de `__records__`,
    corrida por fuera de este recorrido (ver la docstring de `steps.campana`).
    Los cuatro son los únicos exentos de la regla, y esa lista sale de que su
    `reads` esté vacío.

    Y se afirma que es DERIVADA: un paso inventado acá, que lea una raíz que
    `report` escribe, queda encadenado a `report` sin que nadie toque una lista de
    nombres. Una implementación que llevara la cadena escrita a mano pasaría todo
    lo de arriba y se caería acá.

    Rojo alcanzable: escribir la cadena a mano, comparar raíces por prefijo de
    cadena en vez de por segmentos, o dejar de normalizar la escala --- con
    cualquiera de las tres, `noise-sweep` deja de ver a `search-pilot`, porque
    lo que la búsqueda declara es `ceilings.pilot.json` y lo que el barrido lee es
    `ceilings.json`.
    """
    esperada = {
        "verification": (),
        "search-pilot": (),
        "noise-sweep": ("search-pilot",),
        "campaign": (),
        "mechanisms": (),
        # Encuentro y no orden canónico: `predecesores` recorre `reads` en el
        # orden declarado y devuelve cada dueño la primera vez que aparece.
        # `runs.jsonl` es el primer `reads` de `results` y lo produce
        # `campaign`, así que sale primero; `attention_mechanisms.json` es
        # el cuarto `reads` y lo produce `mechanisms`.
        "results": ("campaign", "search-pilot", "mechanisms", "noise-sweep"),
    }
    assert set(esperada) == set(paquete.__steps__), (
        "se declaró un paso nuevo y nadie dijo qué consume")
    for paso, previos in esperada.items():
        assert steps.predecesores(paso) == previos, paso

    sin_predecesor = [p for p in paquete.__steps__ if not steps.predecesores(p)]
    assert sin_predecesor == ["verification", "search-pilot", "campaign",
                              "mechanisms"], (
        "cambió quién está exento de la regla, y eso es una decisión, no un "
        f"detalle -> {sin_predecesor}")

    # y es derivada: un paso nuevo se encadena solo
    inventados = dict(paquete.__steps__)
    inventados["paso-inventado"] = {
        "module": "MIL_CREDA_Benchmark.steps", "function": "resultados",
        "reads": ["Results/Noise/curve"], "produces": []}
    monkeypatch.setattr(paquete, "__steps__", inventados)
    assert steps.predecesores("paso-inventado") == ("noise-sweep",), (
        "la cadena no se deriva de las dos listas")


def test_la_normalizacion_de_escala_sale_de_las_puertas_y_no_de_una_ortografia(
) -> None:
    """Las dos reglas que dejan comparables `produces` y `reads`, recompuestas.

    Las dos listas hablan escalas distintas a propósito: `produces` nombra el
    árbol de ENSAYO en los cuatro pasos que computan, porque es donde escriben, y
    `reads` nombra siempre la corrida completa, que es lo único que un ensayo
    remoto puede consumir. Compararlas pide traducir una a la otra, y la
    traducción no puede estar deletreada: las dos reglas se recomponen acá desde
    las puertas de `config` que las deciden.

    Son dos y no una porque la escala vive en dos lugares distintos: `results_for`
    y `models_for` la meten como un SEGMENTO de directorio, y
    `ceilings_record_for` la mete en el NOMBRE DEL ARCHIVO, así que la primera
    regla no lo toca.

    Rojo alcanzable: mover el segmento `Pilot/` a otro nivel en `results_for`, o
    renombrar `ceilings.pilot.json`, sin mover `raiz_a_escala_completa`.
    """
    from MIL_CREDA_Benchmark import config

    producto = config.PRODUCT
    for puerta in (config.results_for, config.models_for):
        for tasa, forma in ((0.0, "campaign"), (config.NOISE_REPORTED, "campaign"),
                            (0.0, "curve")):
            ensayo = puerta(tasa, forma, True).relative_to(producto).as_posix()
            completa = puerta(tasa, forma, False).relative_to(producto).as_posix()
            assert steps.raiz_a_escala_completa(ensayo) == completa, (
                f"{puerta.__name__}({tasa}, {forma!r}) no se traduce")
            # ya completa, se queda quieta
            assert steps.raiz_a_escala_completa(completa) == completa

    ensayo = config.ceilings_record_for(True).relative_to(producto).as_posix()
    completa = config.ceilings_record_for(False).relative_to(producto).as_posix()
    assert ensayo != completa, (
        "el registro de techos dejó de separar sus dos escalas")
    assert steps.raiz_a_escala_completa(ensayo) == completa
    assert steps.raiz_a_escala_completa(completa) == completa


# `test_el_ensayo_remoto_de_la_campana_corre_bajo_los_techos_de_la_busqueda_completa`
# is removed along with `campana`/`Benchmark_Campaign_v1.ipynb`/the
# `campaign-local` step (see the retirement note above
# `test_todo_cuaderno_del_arbol_lo_corre_un_paso_o_dice_por_que_no`).


def test_el_ensayo_remoto_del_barrido_corre_bajo_los_techos_de_la_busqueda_completa(
        tmp_path, monkeypatch, capsys) -> None:
    """Lo mismo para el barrido, y su guarda con él.

    El barrido pregunta por el registro antes de leerlo, y las dos mitades tienen
    que moverse juntas: una guarda que preguntara por el archivo del ensayo
    negaría cada ensayo remoto por la ausencia de un archivo que ese ensayo no va
    a abrir, y una que preguntara por «alguno de los dos» dejaría pasar un barrido
    que después corre bajo un registro que no le pidieron.

    Rojo alcanzable: dejar `pilot=ES_ENSAYO` en cualquiera de las tres lecturas
    de la celda 4, o desacoplar la guarda de la lectura.
    """
    from MIL_CREDA_Benchmark import config

    _sin_ensayo_remoto(monkeypatch)
    local = _correr_el_barrido(monkeypatch, tmp_path)
    capsys.readouterr()
    assert local["escalas"]["registro"] == [True]
    for corrida in local["corridas"]:
        assert corrida["reduction"].ceilings == local["agrupadoPorEscala"][True]

    monkeypatch.setenv(config.REHEARSAL_ENV, "1")
    remoto = _correr_el_barrido(monkeypatch, tmp_path)
    capsys.readouterr()
    assert remoto["corridas"], "el barrido no corrió ningún nivel"
    assert remoto["escalas"]["registro"] == [False], (
        "la guarda del barrido preguntó por un archivo que el cuaderno no abre")
    for corrida in remoto["corridas"]:
        assert corrida["reduction"].ceilings == remoto["agrupadoPorEscala"][False], (
            "el ensayo remoto del barrido no corrió bajo los techos completos")
        assert corrida["reduction"].ceilings != remoto["agrupadoPorEscala"][True]
        assert corrida["reduction"].pilot is True


# `test_el_ensayo_remoto_del_diagnostico_lee_la_linea_limpia_del_barrido_completo`
# is removed along with the noise diagnostic itself (see the retirement note
# above).


def test_el_ensayo_remoto_se_niega_cuando_falta_una_salida_de_escala_completa(
        tmp_path, monkeypatch) -> None:
    """El estado ordinario al principio de un recorrido, y lo que NO se hace con él.

    Que la salida completa del paso de arriba todavía no esté no es un defecto:
    es lo que pasa hasta que ese paso corre. Lo que no puede pasar es que el
    ensayo siga igual --- ni cayendo al árbol de ensayo, ni al neutral declarado
    del módulo ---, porque las dos caídas lo dejan en verde sin haber tocado nada
    de lo que dice probar, que es justamente la falla que este ensayo existe para
    no tener.

    Y el mensaje nombra la raíz que falta y el paso que la escribe, porque un
    rechazo que dijera «falta una entrada» deja al operador buscando cuál.

    Rojo alcanzable: devolver un resultado en vez de negarse, negarse sin nombrar
    la raíz, o dejar que el paso corra y lea lo que haya.
    """
    _sin_ensayo_remoto(monkeypatch)
    _sin_maquina(monkeypatch, tmp_path)
    monkeypatch.setattr(steps, "_ejecutar", lambda nombre: pytest.fail(
        f"el ensayo abrió {nombre} sin la entrada que dice consumir"))

    faltan = steps.entradas_faltantes("noise-sweep")
    assert faltan == [{"root": "Results/Benchmark/ceilings.json",
                       "producedBy": "search-pilot"}], faltan

    with pytest.raises(SystemExit) as caido:
        steps.ensayo_remoto("noise-sweep")
    mensaje = str(caido.value)
    assert "Results/Benchmark/ceilings.json" in mensaje
    assert "search-pilot" in mensaje

    # y el archivo del ENSAYO no alcanza: es exactamente la sustitución
    # prohibida. Las dos raíces salen de la declaración y no de una constante
    # del módulo: `CEILINGS_RECORD` se congela al importar y `_sin_maquina`
    # redirige `RESULTS`, así que preguntarle a la constante miraría el árbol de
    # la corrida real desde adentro de un test.
    (ensayo,) = [r for r in paquete.__steps__["search-pilot"]["produces"]
                 if r.endswith(".json")]
    (completa,) = paquete.__steps__["noise-sweep"]["reads"]
    assert steps.raiz_a_escala_completa(ensayo) == completa

    (tmp_path / ensayo).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ensayo).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        steps.ensayo_remoto("noise-sweep")

    # con la salida COMPLETA en disco deja de faltar
    (tmp_path / completa).write_text("{}", encoding="utf-8")
    assert steps.entradas_faltantes("noise-sweep") == []


def test_el_paso_sin_predecesor_ensaya_el_cable_y_no_abre_ningun_cuaderno(
        tmp_path, monkeypatch) -> None:
    """La única rama exenta, y sale de la cadena y no de un nombre.

    Un paso sin nada arriba no tiene qué consumir, así que lo que queda por probar
    es que el cable lleva corriente en ESTA máquina --- que es lo que `run_smoke`
    hace, y por qué su independencia de todo registro sigue siendo correcta
    exactamente acá y en ningún otro paso.

    Se afirma sobre los pasos cuyo `reads` está vacío, tal como la declaración los
    dice hoy, y no sobre `'search-pilot'` escrito acá: el día que la búsqueda
    empiece a consumir algo, este test la saca sola de la rama.

    Rojo alcanzable: mandar un paso con predecesores a `run_smoke`, o hacer que el
    exento abra un cuaderno.
    """
    from MIL_CREDA_Benchmark import harness

    _sin_ensayo_remoto(monkeypatch)
    _sin_maquina(monkeypatch, tmp_path)
    monkeypatch.setattr(steps, "_ejecutar", lambda nombre: pytest.fail(
        f"un paso sin predecesor abrió {nombre}"))
    pedidos: list = []
    monkeypatch.setattr(harness, "run_smoke",
                        lambda *a, **k: pedidos.append(True) or {"stub": True})

    exentos = [p for p in paquete.__steps__ if not steps.predecesores(p)]
    assert exentos, "no quedó ningún paso exento y la rama es inalcanzable"
    for paso in exentos:
        salida = steps.ensayo_remoto(paso)
        assert salida["shape"] == "wire" and salida["consumed"] == []
    assert len(pedidos) == len(exentos)


def test_todo_paso_con_predecesor_o_lee_la_escala_de_entrada_o_se_niega(
        tmp_path, monkeypatch) -> None:
    """La otra rama, y su negativa cuando el cuaderno todavía no sabe leer.

    Un cuaderno que compone TODAS sus lecturas con su propia escala no puede ser
    ensayado contra la corrida completa: en el worker abriría el árbol de ensayo,
    que está vacío, o el completo y escribiría encima. Los dos son peores que
    negarse, así que se niega y dice qué le falta al cuaderno.

    La lista de cuáles saben no está escrita acá: sale de leer el cuaderno. Por
    eso convertir uno lo habilita y este test lo sigue solo, en vez de ponerse en
    rojo por un cambio que es una mejora.

    El que computa Y tiene predecesor sí tiene que saber, y se nombra: es el
    único que se envía al worker con algo previo que consumir, y el único
    donde el ensayo compra algo --- los otros leen y dibujan en segundos, sin
    GPU, o no tienen predecesor (`search-pilot`, exento por la otra rama de
    esta misma regla). Eran tres remotos con predecesor antes de esta
    reestructuración (`campaign-local`, `noise-sweep`, `noise-diagnostic`);
    `campaign-local` y el diagnóstico se retiraron, así que sólo queda
    `noise-sweep`.

    Rojo alcanzable: sacarle `upstream_pilot_scale` al cuaderno del barrido, o
    dejar que un paso que no sabe leer corra igual.
    """
    _sin_ensayo_remoto(monkeypatch)
    _sin_maquina(monkeypatch, tmp_path)
    abiertos: list = []
    monkeypatch.setattr(steps, "_ejecutar", lambda nombre: abiertos.append(nombre))

    assert steps.honra_la_escala_de_entrada("noise-sweep"), (
        "noise-sweep se envía al worker y su cuaderno no sabe leer la escala "
        "de entrada")

    for paso in paquete.__steps__:
        if not steps.predecesores(paso) or steps.honra_la_escala_de_entrada(paso):
            continue
        with pytest.raises(SystemExit) as caido:
            steps.ensayo_remoto(paso)
        assert steps.cuaderno_de(paso) in str(caido.value), paso
        assert "upstream_pilot_scale" in str(caido.value), paso
    assert abiertos == [], f"un paso que no sabe leer abrió su cuaderno -> {abiertos}"


def test_todo_paso_que_lee_dice_de_que_arbol_salieron_sus_numeros() -> None:
    """Leer el árbol completo y caer al del ensayo es correcto; hacerlo callado no.

    Los cuadernos locales ---los que dibujan, no los que computan--- ya
    prefieren la corrida completa y caen al ensayo cuando no hay ninguna. Eso es
    lo que los hace útiles mientras la campaña completa todavía no volvió del
    worker: se miran las tablas y las figuras con lo que haya.

    Lo que no puede pasar es que la caída sea invisible. Una figura del ensayo
    tiene la misma forma que la completa y números que no se citan, así que el
    único lugar donde la diferencia existe es el encabezado. Sin él, el informe
    cambia de fuente sin que nadie lo toque y sigue leyéndose igual.

    La partición se DERIVA y no se escribe, sobre TODO paso con `reads`, sin una
    lista de cuáles: quien nombra `config.upstream_pilot_scale()` lee un árbol
    exacto y no eligió nada; quien no la nombra resolvió su propia fuente y
    tiene que decir cuál.

    Rojo alcanzable: sacarle el `source_note` a cualquier paso que resuelve su
    propia fuente, o agregar un paso que lea y no declare.
    """
    assert steps.notas_de_fuente(), (
        "ninguna función `*source_note` en el paquete: el control no tiene con "
        "qué reconocer un estampado y estaría en verde por vacío")

    exactos, resuelven = [], []
    for paso, entrada in paquete.__steps__.items():
        if not entrada.get("reads"):
            continue
        (exactos if steps.honra_la_escala_de_entrada(paso) else resuelven).append(paso)

    assert exactos, "ningún paso compone sus lecturas con la escala del recorrido"
    assert resuelven, "ningún paso resuelve su propia fuente: la partición no parte"

    callados = [paso for paso in resuelven if not steps.declara_su_fuente(paso)]
    assert not callados, (
        "estos pasos eligen entre la corrida completa y el ensayo, y no dicen "
        "cuál eligieron:\n" + "\n".join(
            f"  {paso} -- {steps.cuaderno_de(paso)}" for paso in callados))


def test_el_ensayo_remoto_marca_el_modo_para_el_kernel_y_deja_el_entorno_como_estaba(
        tmp_path, monkeypatch) -> None:
    """El modo viaja por el entorno porque quien lo lee es OTRO proceso.

    `_ejecutar` lanza `nbconvert`, y el kernel del cuaderno no hereda ni un
    argumento de esta llamada: hereda el entorno. Así que la marca tiene que estar
    puesta MIENTRAS el paso corre --- no antes, no después --- y eso es lo que se
    afirma, desde adentro del doble que reemplaza a la ejecución.

    Y se restaura lo que hubiera, en vez de borrarse: un `ensayo_remoto` que
    limpiara a secas apagaría el modo de un ensayo que lo envuelva, y ese apagón
    se lee igual que un cuaderno que decidió leer el árbol de ensayo.

    Rojo alcanzable: poner la marca antes de la guarda de entradas, borrarla con
    `pop` incondicional, o pasarle el modo al paso por argumento.
    """
    from MIL_CREDA_Benchmark import config

    _sin_ensayo_remoto(monkeypatch)
    _sin_maquina(monkeypatch, tmp_path)
    (completa,) = paquete.__steps__["noise-sweep"]["reads"]
    (tmp_path / completa).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / completa).write_text("{}", encoding="utf-8")

    # La precondición del propio paso ---que exista el registro a la escala que
    # el cuaderno lee--- tiene sus tests aparte; acá se sustituye para que lo
    # único que decida el resultado sea la marca de modo. La escala con la que se
    # la pregunta sí se afirma: es la otra mitad del mismo mecanismo.
    from MIL_CREDA_Benchmark import harness

    preguntas: list = []
    monkeypatch.setattr(harness, "search_record", lambda pilot=None: (
        preguntas.append(pilot) or {"stub": True}))

    visto: list = []
    monkeypatch.setattr(steps, "_ejecutar", lambda nombre: (
        visto.append(config.is_rehearsal()) or nombre))

    salida = steps.ensayo_remoto("noise-sweep")
    assert visto == [True], "el kernel del cuaderno no habría visto el modo"
    assert preguntas == [False], (
        "la precondición del paso preguntó por un registro que el ensayo remoto "
        f"no va a abrir -> {preguntas}")
    assert salida["shape"] == "notebook"
    assert salida["consumed"] == ["Results/Benchmark/ceilings.json"]
    assert salida["predecessors"] == ["search-pilot"]
    assert config.is_rehearsal() is False, "el modo quedó puesto después"

    # y lo que hubiera antes se restaura, en vez de borrarse
    monkeypatch.setenv(config.REHEARSAL_ENV, "1")
    steps.ensayo_remoto("noise-sweep")
    assert config.is_rehearsal() is True


def _stamped(entry: dict) -> dict:
    """`entry`, con la procedencia que `ceiling_record.stamp` estampa hoy.

    Este archivo escribe registros de techos a mano para el único test de acá
    abajo que necesita uno; sin esto `config.ceilings_on_record`/
    `harness.search_record` lo verían como escrito antes de que el sello
    existiera y se negarían por eso, no por lo que el test quiere ejercitar.
    """
    from MIL_CREDA_Benchmark import config as _config

    return {**entry, "revision": _config.REVISION,
            "kernelSigma": _config.KERNEL_SIGMA,
            "attentionGamma": _config.ATTENTION_GAMMA,
            "attentionTemperature": _config.ATTENTION_TEMPERATURE}


def test_el_ensayo_remoto_no_ablanda_la_guarda_que_gobierna_la_campana(
        tmp_path, monkeypatch) -> None:
    """Lo que este modo NO toca, afirmado y no supuesto.

    `campaign()` lee `search_record(pilot=False)` por su nombre y exige
    `atRequiredScale` sobre ESE archivo, porque un registro de ensayo no gobierna
    una campaña real. El ensayo remoto empuja las lecturas hacia el registro
    completo, que es la dirección en la que una implementación descuidada haría
    esa exigencia relativa a la escala --- «estamos en ensayo, con el techo del
    ensayo alcanza» --- y ahí la sustitución que el rechazo existe para impedir
    entra por la puerta de al lado.

    Se corre con el modo PUESTO y con una reducción de ensayo, que es la
    combinación en la que un aflojamiento sería invisible.

    Rojo alcanzable: derivar el `pilot=False` de `campaign()` de
    `upstream_pilot_scale()`, o saltear la exigencia cuando la reducción es de
    ensayo.
    """
    import json as _json

    import torch

    from MIL_CREDA_Benchmark import config, harness

    monkeypatch.setenv(config.REHEARSAL_ENV, "1")
    # El árbol de salida fuera del camino, igual que todos los pasos de acá.
    # `campaign()` hace `results_for(...).mkdir()` y `models_for(...).mkdir()`
    # antes de sus tres rechazos, y esta reducción es de ENSAYO: sin esto los dos
    # directorios que se creaban eran justo los que guardan la corrida de ensayo
    # que este repositorio tiene en disco.
    _sin_maquina(monkeypatch, tmp_path)
    registro = tmp_path / "ceilings.json"
    registro.write_text(_json.dumps({
        "creda": _stamped({"ceiling": 1e-4, "atRequiredScale": False}),
        "milcreda": _stamped({"ceiling": 1.0, "atRequiredScale": True}),
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", registro)
    # `reduccion.pilot=True` hace que el rechazo de sello a la entrada de
    # `campaign()` mire también `search_record(pilot=True)` -- el registro de
    # ENSAYO, que este test nunca redirige, así que sin esto leería el
    # `ceilings.pilot.json` real de este repositorio, viejo y sin sello, y se
    # negaría por esa razón antes de llegar a la guarda de escala que el test
    # quiere ejercitar.
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")

    reduccion = harness.Reduction(ceilings={"creda": 1e-4, "milcreda": 1.0},
                                  pilot=True)
    with pytest.raises(SystemExit) as caido:
        harness.campaign(reduccion, torch.device("cpu"), arms=["B"],
                         progress=lambda *a: None)
    assert "below scale" in str(caido.value)
    assert "creda" in str(caido.value)


# `_correr_las_celdas_de_la_campana` and the two tests that exercised
# `Benchmark_Campaign_v1.ipynb`'s two passes
# (`test_la_campana_corre_las_dos_pasadas_y_la_segunda_es_la_contaminada`,
# `test_la_pasada_contaminada_reusa_los_techos_limpios_sin_volver_a_buscar`)
# are removed along with `campana`/that notebook/the `campaign-local` step
# (see the retirement note above
# `test_todo_cuaderno_del_arbol_lo_corre_un_paso_o_dice_por_que_no`).


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------- el paso orquesta, el cuaderno corre
#
# Las tres afirmaciones de abajo cierran el defecto que `verify` reportaba bajo
# `undeclaredStepNotebooks`: `campaign` y `mechanisms` --- dos de seis pasos ---
# llamaban a `harness.run_campaign_shard()`/`run_mechanism_sweep_shard()` desde
# la biblioteca, sin cuaderno propio. El ensayo los recorría probando la
# biblioteca y el artefacto que llevaría el mismo trabajo a otra máquina no lo
# ejecutaba nadie.
#
# Ninguna nombra un cuaderno: las tres recorren `__steps__` entero y resuelven
# el cuaderno de cada paso con `steps.cuaderno_de`. Un paso nuevo entra solo.


def _entradas_que_entrenan() -> set[str]:
    """Las funciones de `harness` que llegan a `run_one`, cerradas transitivamente.

    DERIVADO y no una lista: `run_one` es donde se entrena un modelo, así que
    «esta función entrena» es «desde acá se llega hasta ahí», y la clausura lo
    contesta sola. Una lista escrita a mano se queda vieja en el sentido que no
    se ve --- una entrada nueva que entrena no entra, y el control sigue verde
    sobre un cuaderno que ella nunca miró.
    """
    import ast

    arbol = ast.parse(
        Path(_harness_modulo().__file__).read_text(encoding="utf-8"))
    definidas = {nodo.name: nodo for nodo in arbol.body
                 if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef))}
    llama = {}
    for nombre, nodo in definidas.items():
        salidas = set()
        for hijo in ast.walk(nodo):
            if isinstance(hijo, ast.Call):
                llamado = (hijo.func.id if isinstance(hijo.func, ast.Name)
                           else getattr(hijo.func, "attr", None))
                if llamado in definidas:
                    salidas.add(llamado)
        llama[nombre] = salidas

    entrenan = {"run_one"}
    creciendo = True
    while creciendo:
        creciendo = False
        for nombre, salidas in llama.items():
            if nombre not in entrenan and salidas & entrenan:
                entrenan.add(nombre)
                creciendo = True
    return entrenan


def _harness_modulo():
    from MIL_CREDA_Benchmark import harness

    return harness


def _llamadas_del_cuaderno(cuaderno: str) -> tuple[set[str], list]:
    """`({nombres llamados}, [árbol por celda])` de un cuaderno.

    Los nombres vienen con y sin dueño (`harness.campaign` y `campaign`), para
    que una llamada importada por su nombre corto no se escape del recorrido.
    """
    import ast

    nombres, arboles = set(), []
    for celda in _celdas_de_codigo(cuaderno):
        limpia = "\n".join(linea for linea in celda.splitlines()
                           if not linea.lstrip().startswith(("%", "!")))
        try:
            arbol = ast.parse(limpia)
        except SyntaxError:                                  # pragma: no cover
            continue
        arboles.append(arbol)
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            if isinstance(nodo.func, ast.Attribute):
                nombres.add(nodo.func.attr)
                if isinstance(nodo.func.value, ast.Name):
                    nombres.add(f"{nodo.func.value.id}.{nodo.func.attr}")
            elif isinstance(nodo.func, ast.Name):
                nombres.add(nodo.func.id)
    return nombres, arboles


def test_ningun_paso_computa_al_lado_del_cuaderno_que_corre() -> None:
    """La mitad que el paso debe: orquestar y no computar.

    `campana` y `mecanismos_de_atencion` devolvían `harness.run_campaign_shard()`
    y `harness.run_mechanism_sweep_shard()` desde el cuerpo del paso, sin abrir
    ningún cuaderno. Eso ya estaba afirmado, paso por paso, para `results` y
    para `search-pilot`; escrito así, un paso NUEVO que computara entraba sin
    que nada lo mirara --- y entraron dos.

    Acá el recorrido es `__steps__` entero y el conjunto prohibido se deriva
    (`_entradas_que_entrenan`), así que ni el paso ni la entrada hacen falta
    escribirlos.

    Las guardas no cuentan y no son una excepción escrita: `barrido_de_ruido`
    llama a `harness.search_record(...)` para negarse antes de abrir su
    cuaderno, y `search_record` no llega a `run_one`, así que la clausura no la
    incluye. Lo que se prohíbe es ENTRENAR al lado del cuaderno, no leer.

    Rojo alcanzable: devolver `harness.run_campaign_shard()` al cuerpo de
    `campana`, o poner `harness.campaign(...)` en el de cualquier otro paso.
    """
    import ast

    entrenan = _entradas_que_entrenan()
    assert {"run_one", "campaign", "run_campaign_shard",
            "run_mechanism_sweep_shard", "run_search"} <= entrenan, (
        "la clausura no reconoce las entradas que sí entrenan, así que este "
        f"control estaría verde por vacío -> {sorted(entrenan)}")

    fuente = Path(steps.__file__).read_text(encoding="utf-8")
    definidas = {nodo.name: nodo for nodo in ast.parse(fuente).body
                 if isinstance(nodo, ast.FunctionDef)}

    computan = {}
    for paso, entrada in paquete.__steps__.items():
        cuerpo = definidas[entrada["function"]]
        adentro = sorted({
            nodo.func.attr for nodo in ast.walk(cuerpo)
            if isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr in entrenan})
        if adentro:
            computan[paso] = adentro
    assert not computan, (
        "estos pasos entrenan en el cuerpo del paso en vez de dejar que lo "
        f"haga el cuaderno que corren -> {computan}")


def test_el_cuaderno_de_cada_paso_que_entrena_llama_a_la_biblioteca_y_no_la_copia() -> None:
    """La otra mitad: el cuaderno orquesta, la biblioteca computa.

    Un paso puede correr su cuaderno y el cuaderno reimplementar el bucle
    adentro, y entonces el ensayo ejercita el artefacto y mide otra cosa que la
    corrida real --- la misma bifurcación que `unreachedModules` vigila entre
    un brazo y los módulos del método, un nivel más abajo.

    Lo prohibido es `run_one` y sus vecinos de UNA corrida: son el cuerpo del
    bucle, así que un cuaderno que los alcanza ya lo forkeó. Llamar a una
    ENTRADA (`run_search`, `run_campaign_shard`, `run_mechanism_sweep_shard`,
    `campaign`) es exactamente lo correcto y es lo que se exige del otro lado.

    Cada nombre prohibido se resuelve contra el módulo vivo: un renombre deja
    este control en rojo en vez de vaciarlo en silencio, que es la única forma
    en que una lista escrita a mano falla sin que se vea.

    **Se mira TODO cuaderno de TODO paso, y no sólo los que llaman a una
    entrada que entrena.** Esa era la versión anterior y la mutación se la
    comió: reemplazar `harness.run_mechanism_sweep_shard()` por
    `harness.run_mechanism(...)` en la celda de la corrida ---que es
    exactamente el fork que esto existe para atrapar--- dejaba al cuaderno sin
    ninguna entrada de la clausura, así que salía del conjunto examinado y el
    control pasaba en VERDE. Un conjunto derivado de lo que el cuaderno llama
    no puede vigilar lo que el cuaderno llama. Medido, no razonado: la
    mutación corrió y pasó.

    Hoy ninguno de los seis cuadernos alcanza una primitiva ---medido, no
    supuesto--- así que la regla más fuerte es también la verdadera, y un
    cuaderno que dibuja y de verdad necesitara materializar una bolsa se exime
    escribiendo por qué, como todo lo demás acá.

    Y es la mitad que le cierra la puerta de atrás a
    `test_todo_cuaderno_que_entrena_imprime_la_escala_a_la_que_entrena`: con
    las primitivas prohibidas para todos, la única forma de entrenar adentro de
    un cuaderno es llamar a una entrada, que es justo la condición con la que
    aquel elige a quién mirar.

    Rojo alcanzable: reemplazar `harness.run_campaign_shard()` en la celda de
    la corrida por `harness.run_one(...)`, o por `harness.run_mechanism(...)`
    en la del barrido de mecanismos.
    """
    from MIL_CREDA_Benchmark import bags, harness, wiring

    primitivas = {"harness": (harness, ("run_one", "run_mechanism")),
                  "wiring": (wiring, ("build",)),
                  "bags": (bags, ("build",))}
    prohibidas = set()
    for modulo_nombre, (modulo, nombres) in primitivas.items():
        for nombre in nombres:
            assert callable(getattr(modulo, nombre, None)), (
                f"{modulo_nombre}.{nombre} ya no existe: este control estaría "
                "prohibiendo un nombre que no es nada")
            prohibidas.add(f"{modulo_nombre}.{nombre}")

    miradas, forkeados = [], {}
    for paso in paquete.__steps__:
        cuaderno = steps.cuaderno_de(paso)
        assert cuaderno, f"{paso} dejó de declarar exactamente un cuaderno"
        llamadas, _ = _llamadas_del_cuaderno(cuaderno)
        miradas.append(cuaderno)
        copiadas = sorted(llamadas & prohibidas)
        if copiadas:
            forkeados[cuaderno] = copiadas
    assert len(miradas) == len(paquete.__steps__) and miradas, (
        "el recorrido no abrió un cuaderno por paso declarado, así que este "
        f"control estaría verde sin haber mirado todo -> {miradas}")
    assert not forkeados, (
        "estos cuadernos alcanzan el cuerpo de una corrida en vez de llamar a "
        f"la entrada de biblioteca -> {forkeados}")


def test_todo_cuaderno_que_entrena_imprime_la_escala_a_la_que_entrena() -> None:
    """Una salida de ensayo y una completa tienen la misma forma.

    El único lugar donde la diferencia existe es lo que el cuaderno imprime, y
    por eso se exige que la escala LLEGUE a quien mira: una escala leída y
    nunca mostrada deja este control en verde sobre un cuaderno que no dice
    nada --- la misma falla que `steps.declara_su_fuente` ya evita del lado de
    la fuente, un paso más allá.

    El conjunto se deriva de la clausura y los nombres de escala del módulo
    vivo: el que entrena tiene que decirlo, el que sólo lee y dibuja no le debe
    nada a este control.

    Rojo alcanzable: sacarle el `print` de la escala a cualquiera de los
    cuadernos que entrenan, o dejar la escala calculada en una variable que
    ninguna celda muestre.
    """
    import ast

    from MIL_CREDA_Benchmark import config

    escalas = tuple(nombre for nombre in
                    ("EPOCHS", "SEEDS", "FULL_EPOCHS", "FULL_SEEDS",
                     "is_pilot_scale", "upstream_pilot_scale")
                    if hasattr(config, nombre))
    assert len(escalas) == 6, (
        f"`config` dejó de declarar alguna escala que este control lee -> "
        f"{escalas}")

    entrenan = _entradas_que_entrenan()
    mostradores = {"print", "show", "display"}

    miradas, mudos = [], []
    for paso in paquete.__steps__:
        cuaderno = steps.cuaderno_de(paso)
        llamadas, arboles = _llamadas_del_cuaderno(cuaderno)
        if not (llamadas & entrenan):
            continue
        miradas.append(cuaderno)

        # Los nombres que llevan una escala adentro: la escala misma, y toda
        # variable asignada desde una expresión que la nombra.
        portadores = set(escalas)
        for arbol in arboles:
            for nodo in ast.walk(arbol):
                if not isinstance(nodo, ast.Assign):
                    continue
                dentro = {hijo.id for hijo in ast.walk(nodo.value)
                          if isinstance(hijo, ast.Name)}
                dentro |= {getattr(hijo, "attr", "")
                           for hijo in ast.walk(nodo.value)
                           if isinstance(hijo, ast.Attribute)}
                if dentro & portadores:
                    portadores |= {destino.id for destino in nodo.targets
                                   if isinstance(destino, ast.Name)}

        dicho = False
        for arbol in arboles:
            for nodo in ast.walk(arbol):
                if not (isinstance(nodo, ast.Call)
                        and (getattr(nodo.func, "id", None) in mostradores
                             or getattr(nodo.func, "attr", None) in mostradores)):
                    continue
                nombrados = {hijo.id for hijo in ast.walk(nodo)
                             if isinstance(hijo, ast.Name)}
                nombrados |= {getattr(hijo, "attr", "")
                              for hijo in ast.walk(nodo)
                              if isinstance(hijo, ast.Attribute)}
                if nombrados & portadores:
                    dicho = True
        if not dicho:
            mudos.append(cuaderno)

    assert miradas, (
        "ningún cuaderno entrena, así que este control estaría verde sin "
        "haber mirado nada")
    assert not mudos, (
        "estos cuadernos entrenan y no dicen a qué escala, así que una salida "
        f"de ensayo se lee igual que una completa -> {mudos}")
