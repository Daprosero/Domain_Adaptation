"""Los pasos locales que este repositorio le ofrece a la forja.

Nivel 1 - invariantes de estructura. No ejecuta ningún cuaderno: ejecutar uno
tarda minutos y lo que acá se afirma es qué se ofrece y qué no, no qué imprime.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import pytest

import MIL_CREDA_Benchmark as paquete
from MIL_CREDA_Benchmark import steps

#: La raíz del repositorio, derivada del módulo que ya la conoce
#: (`steps.CUADERNOS` es `<raíz>/MIL-CREDA/Notebooks`) y nunca escrita.
_RAIZ = steps.CUADERNOS.parents[1]


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

        La biblioteca no se toca y este test no la vigila: la celda 7 del
        cuaderno llama a `harness.campaign()`, así que borrarla rompería lo que
        se quiere correr.

        Rojo alcanzable: devolver `harness.campaign(...)` al cuerpo del paso,
        apuntarlo a otro cuaderno, o sacarle el `_ejecutar`.
        """
        import ast

        entrada = paquete.__steps__["campaign-local"]
        self.assertEqual(entrada["function"], "campana")
        self.assertEqual(
            _cuadernos_nombrados_por_los_pasos().get("Benchmark_Campaign_v1.ipynb"),
            "campana", "la campaña no corre su propio cuaderno")

        fuente = Path(steps.__file__).read_text(encoding="utf-8")
        (definicion,) = [nodo for nodo in ast.parse(fuente).body
                         if isinstance(nodo, ast.FunctionDef)
                         and nodo.name == "campana"]
        computa = [nodo for nodo in ast.walk(definicion)
                   if isinstance(nodo, ast.Call)
                   and isinstance(nodo.func, ast.Attribute)
                   and nodo.func.attr in ("campaign", "run_search",
                                          "search_ceilings", "run_one")]
        self.assertEqual(computa, [],
                         "la campaña computa al lado del cuaderno que corre")

    def test_la_busqueda_de_ensayo_corre_su_cuaderno_en_vez_de_computar_en_su_lugar(
            self):
        """La reversión, medida: la búsqueda tiene cuaderno propio y lo corre.

        Este test afirmaba lo contrario --- que `ensayo_de_busqueda` se quedaba
        en biblioteca porque el único cuaderno de la búsqueda era su INFORME, y
        apuntar los dos pasos ahí le habría dado dos dueños a una raíz
        declarada. El razonamiento era bueno y el hecho que lo sostenía dejó de
        ser cierto: la búsqueda tiene el suyo, y cada raíz sigue con un dueño.

        Las tres mitades, cada una capaz de volver sola: que el paso corra
        `Benchmark_Ceiling_Search_v1.ipynb`, que no compute al lado del cuaderno
        que corre, y que el informe siga sin ejecutar ninguna búsqueda ---
        abrirlo tiene que seguir costando lo que cuesta leer, no lo que cuesta
        buscar.

        La mención en prosa no cuenta: la celda del informe NOMBRA la llamada
        que tenía comentada, para decir por qué ya no está, y eso es texto. Lo
        que se afirma es el árbol de sintaxis, así que descomentarla es rojo y
        contarla es verde.

        Rojo alcanzable: devolver `harness.run_search(...)` al cuerpo del paso,
        apuntarlo al informe, o volver a poner la llamada viva en el informe.
        """
        import ast

        entrada = paquete.__steps__["search-pilot"]
        self.assertEqual(entrada["function"], "ensayo_de_busqueda")
        corridos = _cuadernos_nombrados_por_los_pasos()
        self.assertEqual(corridos.get("Benchmark_Ceiling_Search_v1.ipynb"),
                         "ensayo_de_busqueda",
                         "la búsqueda no corre su propio cuaderno")
        self.assertEqual(corridos.get("Benchmark_Search_Report_v1.ipynb"),
                         "informe_de_busqueda",
                         "el informe de la búsqueda dejó de tener su dueño")

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

        arbol = ast.parse("\n".join(
            _celdas_de_codigo("Benchmark_Search_Report_v1.ipynb")))
        llamadas = {getattr(nodo.func, "attr", getattr(nodo.func, "id", None))
                    for nodo in ast.walk(arbol) if isinstance(nodo, ast.Call)}
        for computo in ("run_search", "search_ceilings", "with_ceilings_in_force"):
            self.assertNotIn(computo, llamadas,
                             "el informe de la búsqueda volvió a buscar: "
                             "abrirlo cuesta lo que cuesta correrla")

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

        Rojo alcanzable: declarar un paso que compute sin correr un cuaderno, o
        sacarle el `Notebooks/...` a las raíces de cualquiera de los diez.
        """
        sin_cuaderno, con_varios = [], []
        for nombre, entrada in paquete.__steps__.items():
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

    def test_el_cuaderno_de_campana_deriva_su_escala_en_vez_de_escribir_en_la_completa(
            self):
        """Dónde escribe el cuaderno lo decide la escala, y una sola expresión.

        `Reduction.pilot` decide el árbol y `EPOCHS`/`SEEDS` deciden la escala,
        y hasta acá nada unía las dos: el cuaderno construía su reducción sin
        `pilot`, o sea `False`, así que una corrida de tres épocas y una semilla
        escribía en `Results/Benchmark/` --- el árbol de la corrida completa ---
        y sus números quedaban ahí para que alguien los citara. Es la falla que
        la docstring del propio campo nombra.

        Las dos mitades: el cuaderno pasa `pilot=` derivado de
        `config.is_pilot_scale()`, y el paso se niega cuando esa lectura dice
        que la escala es la completa, porque sus `produces` nombran el árbol de
        ensayo y ninguno más.

        Rojo alcanzable: sacarle el `pilot=` a la `Reduction` del cuaderno,
        escribir la regla a mano en vez de leerla, o sacarle la guarda al paso.
        """
        import ast

        celdas = "\n".join(_celdas_de_codigo("Benchmark_Campaign_v1.ipynb"))
        arbol = ast.parse(celdas)
        reducciones = [nodo for nodo in ast.walk(arbol)
                       if isinstance(nodo, ast.Call)
                       and isinstance(nodo.func, ast.Attribute)
                       and nodo.func.attr == "Reduction"]
        self.assertTrue(reducciones, "el cuaderno de campaña no arma reducción")
        for llamada in reducciones:
            with self.subTest(linea=llamada.lineno):
                nombres = {k.arg for k in llamada.keywords}
                self.assertIn("pilot", nombres,
                              "la reducción no dice a qué escala escribe")

        # y el nombre que pasa sale de la única lectura de la regla
        asignado = [nodo for nodo in ast.walk(arbol)
                    if isinstance(nodo, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "ES_ENSAYO"
                            for t in nodo.targets)]
        self.assertTrue(asignado, "el cuaderno no deriva su escala")
        for nodo in asignado:
            with self.subTest(linea=nodo.lineno):
                self.assertIn("is_pilot_scale", ast.dump(nodo.value),
                              "la escala se escribió a mano en vez de leerse")

        # la guarda del paso, sobre la misma lectura
        fuente = Path(steps.__file__).read_text(encoding="utf-8")
        (definicion,) = [nodo for nodo in ast.parse(fuente).body
                         if isinstance(nodo, ast.FunctionDef)
                         and nodo.name == "campana"]
        self.assertIn("is_pilot_scale", ast.dump(definicion),
                      "el paso no se niega a escala completa")

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
        "liberado: era el INFORME de la búsqueda y hoy se llama "
        "`Benchmark_Search_Report_v1.ipynb`, porque su nombre prometía la "
        "búsqueda y lo que hacía era presentarla. No se recicla para el "
        "cuaderno que sí la corre"),
    "Benchmark_Search_Pilot_v1.ipynb": (
        "liberado: era el cuaderno que corre la búsqueda y fijaba `pilot=True` "
        "adentro, así que su nombre afirmaba una escala que no le toca elegir "
        "--- y mientras la afirmaba ningún cuaderno podía correr la búsqueda "
        "completa. Hoy se llama `Benchmark_Ceiling_Search_v1.ipynb` y recibe su "
        "escala"),
}


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
        """Presente y con forma, para los diez, sin nombrar a ninguno.

        La forma es la que `cmd_step` valida del otro lado: una lista NO vacía
        de cadenas no vacías. La lista vacía no es "este paso no escribe nada"
        --- la forja la rechaza con `STEP_MALFORMED` y `verify` la cuenta como
        no declarada --- así que un paso que no escribiera nada no tendría
        cómo decirlo y no hay ninguno acá.

        Rojo alcanzable: agregar un paso sin `produces`, o vaciarle la lista a
        cualquiera de los diez.
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
        ensayo = config.results_for(0.0, "campaign", True)
        contaminada = config.results_for(rho, "campaign", True)
        curva = config.results_for(0.0, "curve", True).parent

        esperado = {
            "search-pilot": [relativa(config.ceilings_record_for(True)),
                             cuaderno_de("ensayo_de_busqueda")],
            # Las dos pasadas, cada una con su árbol: la campaña corre la rejilla
            # entera en `NOISE` y otra vez en `NOISE_REPORTED`, y el segundo árbol
            # es el que el informe y el latente leen. `Probe_results.json` es uno
            # solo porque `campaign()` lo ancla en la raíz limpia para las dos.
            "campaign-local": [f"{relativa(ensayo)}/runs.jsonl",
                               f"{relativa(ensayo)}/summary.json",
                               f"{relativa(ensayo)}/shard.json",
                               f"{relativa(contaminada)}/runs.jsonl",
                               f"{relativa(contaminada)}/summary.json",
                               f"{relativa(contaminada)}/shard.json",
                               f"{relativa(ensayo.parent)}/Probe_results.json",
                               relativa(config.models_for(0.0, "campaign", True)),
                               relativa(config.models_for(rho, "campaign", True)),
                               cuaderno_de("campana")],
            "noise-sweep": [relativa(curva),
                            relativa(config.models_for(0.0, "curve", True).parent),
                            cuaderno_de("barrido_de_ruido")],
            "noise-diagnostic": [
                f"{relativa(config.results_for(0.0, 'curve', True).parents[1])}"
                "/diagnostic.json",
                cuaderno_de("diagnostico_de_ruido")],
        }
        for paso, raices in esperado.items():
            with self.subTest(paso=paso):
                self.assertEqual(paquete.__steps__[paso]["produces"], raices)

        # Las dos escalas de los cuadernos, cada una compuesta por el helper.
        for pilot in (False, True):
            raiz = relativa(config.results_for(0.0, "campaign", pilot))
            with self.subTest(cuaderno="report", pilot=pilot):
                for hoja in ("curves", "report.txt", "report.md"):
                    self.assertIn(f"{raiz}/{hoja}",
                                  paquete.__steps__["report"]["produces"])
            with self.subTest(cuaderno="report", pilot=pilot, rho=rho):
                self.assertIn(
                    f"{relativa(config.results_for(rho, 'campaign', pilot))}/curves",
                    paquete.__steps__["report"]["produces"])
            with self.subTest(cuaderno="latent", pilot=pilot, rho=rho):
                self.assertIn(
                    f"{relativa(config.results_for(rho, 'campaign', pilot))}/latent",
                    paquete.__steps__["latent"]["produces"])

        # La mitad limpia del latente escribe en el árbol COMPLETO aunque lea
        # los pesos del ensayo: el cuaderno nombra `config.RESULTS` a secas.
        completo = relativa(config.RESULTS)
        for hoja in ("latent", "latent.json", "latent.md"):
            self.assertIn(f"{completo}/{hoja}",
                          paquete.__steps__["latent"]["produces"])


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


def _correr_las_celdas(cuaderno: str, ambito: dict | None = None) -> dict:
    """Ejecuta las celdas de código de un cuaderno menos la de arranque.

    Ésa busca el repositorio en el disco y manosea `sys.path`, y adentro de la
    suite el paquete ya está importado. Cuál es se deriva de su contenido y no
    de un índice --- una celda que se agregue arriba correría el índice y
    dejaría este helper saltándose otra cosa.

    Las celdas se compilan juntas y no de a una: un cuaderno es un solo
    programa, y ejecutar la tercera sin la segunda probaría algo que nadie va a
    correr nunca.
    """
    ambito = {} if ambito is None else ambito
    celdas = [c for c in _celdas_de_codigo(cuaderno) if "find_repository" not in c]
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


def _correr_el_diagnostico(monkeypatch, tmp_path) -> dict:
    """Ejecuta el cuaderno del diagnóstico sin máquina, y devuelve lo que pidió.

    Sus celdas y no una lectura de su texto: `NOISE_DIAGNOSTIC_LEVEL` puede
    estar bien mientras el cuaderno arma su reducción con otra cosa, y las dos
    cosas dan una suite verde.
    """
    from MIL_CREDA_Benchmark import contamination, harness

    _sin_maquina(monkeypatch, tmp_path)

    buscadas, campanas = [], []

    def _search_ceilings(reduccion, dispositivo, **kwargs):
        buscadas.append({"reduction": reduccion, **kwargs})
        return {"milcreda": {"ceiling": 1e-2, "byTransfer": {"M->U": 1e-2}}}

    monkeypatch.setattr(harness, "search_ceilings", _search_ceilings)
    monkeypatch.setattr(harness, "campaign",
                        lambda *a, **k: campanas.append((a, k)))
    monkeypatch.setattr(contamination, "load", lambda *a, **k: None)

    ambito = _correr_las_celdas("Benchmark_Noise_Diagnostic_Search_v1.ipynb")
    return {"buscadas": buscadas, "campanas": campanas,
            "registro": ambito["registro"], "ambito": ambito}


def test_el_diagnostico_corre_en_el_tope_del_rango_y_paga_una_sola_medicion(
        tmp_path, monkeypatch, capsys) -> None:
    """Dónde corre el diagnóstico, sobre qué, con quiénes, y cuánto cuesta.

    El nivel es el tope del rango declarado y no un número escrito acá: en el
    extremo el coeficiente está bajo la mayor presión, así que un techo
    re-buscado que no recupera nada ahí no recupera nada en ningún lado, y la
    lectura no depende de dónde eligió mirar nadie. El tope lo fija el rango y no
    un resultado.

    Necesita tres puntos y paga uno solo. Lo que se afirma acá es la MEDICIÓN
    nueva: una sola llamada a la búsqueda, sobre la transferencia de la curva y
    ninguna otra, y ninguna campaña. Los otros dos puntos ya están en el
    registro del barrido -- este cuaderno los lee, no los vuelve a correr.

    `D` y `G` y nadie más: los dos métodos completos, uno por familia, y los
    únicos que llevan coeficiente. `A` y `B` no tienen término de adaptación al
    que re-buscarle un techo.

    Se ejecutan las celdas del cuaderno y no el paso: desde que el paso sólo
    corre `_ejecutar`, lo que decide todo esto vive en el cuaderno, y un test
    contra el paso mediría `nbconvert`.

    Rojo alcanzable: correr en el medio del rango, buscar sobre las seis
    transferencias, llamar a `campaign`, o agregar un brazo sin coeficiente.
    """
    from MIL_CREDA_Benchmark import config

    corrido = _correr_el_diagnostico(monkeypatch, tmp_path)
    capsys.readouterr()
    registro, buscadas = corrido["registro"], corrido["buscadas"]

    # el tope del rango, leído del rango y no escrito
    assert config.NOISE_DIAGNOSTIC_LEVEL == config.NOISE_LEVELS[-1]
    assert config.NOISE_DIAGNOSTIC_LEVEL == max(config.NOISE_LEVELS)
    assert registro["level"] == config.NOISE_DIAGNOSTIC_LEVEL

    # una sola medición nueva, y ninguna campaña
    assert len(buscadas) == 1, "el diagnóstico paga más de una búsqueda"
    assert corrido["campanas"] == [], "el diagnóstico corrió una campaña que no le toca"
    (buscada,) = buscadas
    assert buscada["transfers"] == [config.NOISE_TRANSFER]
    assert buscada["noise"] == config.NOISE_DIAGNOSTIC_LEVEL
    assert buscada["reduction"].labelNoise == config.NOISE_DIAGNOSTIC_LEVEL

    # sobre M->U, la misma transferencia que recorre la curva
    assert config.NOISE_TRANSFER == ("M", "U")
    assert registro["transfer"] == "M->U"

    # los dos completos, uno por familia
    assert list(registro["arms"]) == list(config.NOISE_DIAGNOSTIC_ARMS) == ["D", "G"]
    familias = {config.ARMS_BY_ID[a]["adaptation"] for a in registro["arms"]}
    assert familias == {"creda", "milcreda"}
    for arm in registro["arms"]:
        spec = config.ARMS_BY_ID[arm]
        assert spec["adaptation"] is not None, "un brazo sin techo que re-buscar"
        assert spec["weighting"], "no es el método completo de su familia"
        assert spec["selection"] is None, "una ablación de selección, no el completo"


def test_los_numeros_del_diagnostico_no_entran_en_las_tablas_del_veredicto(
        tmp_path, monkeypatch, capsys) -> None:
    """Diagnóstico y nunca veredicto, afirmado donde puede fallar.

    Lo único que decide es si vale reestructurar para techos por nivel, y eso se
    sostiene en cuatro cosas a la vez, cada una verde por su cuenta mientras la
    de al lado está rota:

    * escribe UN `diagnostic.json` y ningún `runs.jsonl`, así que no hay corridas
      suyas que ninguna tabla pueda agregar;
    * escribe bajo la raíz del ENSAYO, no encima del árbol de la campaña;
    * los árboles del nivel que diagnostica quedan intactos, así que el barrido
      no hereda un directorio que él no escribió;
    * su registro se lee en una sola expresión, que alimenta dos presentadores, y
      ninguno de los dos aparece en los cuadernos del veredicto.

    No deja pesos tampoco: `keeps_checkpoints` dice que no a este nivel, y un
    nivel que escribiera 8 GB que nadie abre dejaría un directorio que parece
    evidencia.

    Rojo alcanzable: escribir el registro bajo la raíz de la campaña, dejar que
    el cuaderno corra una campaña al nivel de diagnóstico, o mostrar
    `render_diagnostic` en el informe del veredicto.
    """
    import json
    from pathlib import Path as _Path

    from MIL_CREDA_Benchmark import config, harness, steps, tables

    monkeypatch.setattr(harness, "campaign", lambda *a, **k: pytest.fail(
        "el diagnóstico no corre campañas"))
    corrido = _correr_el_diagnostico(monkeypatch, tmp_path)
    capsys.readouterr()
    registro = corrido["registro"]

    escritos = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    assert escritos == ["diagnostic.json"]
    assert list(tmp_path.rglob("runs.jsonl")) == []
    assert list(tmp_path.rglob("summary.json")) == []

    # bajo la raíz del ensayo, que es de donde `_diagnostic_record` lo lee
    (escrito,) = list(tmp_path.rglob("diagnostic.json"))
    assert escrito.parent == config.results_for(0.0, "curve", True).parents[1]
    assert "Pilot" in escrito.parts

    # el nivel que diagnostica queda sin árbol: sus números no son una corrida
    nivel = config.NOISE_DIAGNOSTIC_LEVEL
    for kind in ("curve", "campaign"):
        for pilot in (False, True):
            assert not config.results_for(nivel, kind, pilot).exists()
    assert not config.keeps_checkpoints(nivel)
    assert not config.models_for(nivel, pilot=True).exists()

    # y lo dice de sí mismo, en el propio registro
    assert "diagnosticOnly" in registro
    assert registro["diagnosticOnly"] == json.loads(
        escrito.read_text(encoding="utf-8"))["diagnosticOnly"]

    # una sola expresión lo lee, y alimenta dos presentadores y nada más
    fuente = _Path(tables.__file__).read_text(encoding="utf-8")
    assert fuente.count('"diagnostic.json"') == 1
    assert _quien_llama(fuente, "_diagnostic_record") == {
        "render_diagnostic", "conclusion_diagnostic"}

    # y ninguno de los dos llega a un cuaderno del veredicto
    for cuaderno in ("Benchmark_Report_v1.ipynb", "Benchmark_Latent_v1.ipynb"):
        texto = (steps.CUADERNOS / cuaderno).read_text(encoding="utf-8")
        assert "render_diagnostic" not in texto
        assert "conclusion_diagnostic" not in texto


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

    ambito = _correr_las_celdas("Benchmark_Noise_Sweep_v1.ipynb")
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

    ambito = _correr_las_celdas("Benchmark_Ceiling_Search_v1.ipynb")
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


def test_el_barrido_y_el_diagnostico_siguen_la_escala_configurada_y_su_paso_se_niega(
        tmp_path, monkeypatch, capsys) -> None:
    """La regla contraria, y su otra mitad, que sólo valen juntas.

    Los dos son formas de una campaña y escriben donde una campaña escribe, así
    que derivan su escala de la misma lectura que la campaña
    (`config.is_pilot_scale()`) en vez de fijarla. Fijada en `True`, una corrida
    de veinte épocas y treinta semillas se archivaba bajo `Pilot/`: una medición
    completa etiquetada como ensayo, que es la falla inversa de la que
    `is_pilot_scale` existe para impedir y del mismo tamaño.

    Derivar sola sería peor que fijar: `produces` nombra el árbol de ENSAYO y
    ninguno más, así que a escala completa el cuaderno escribiría donde nadie lo
    vigila y la forja lo leería como `foreign`. Por eso el paso se niega antes
    de abrir el cuaderno --- la misma forma que `campana` ---, y las dos mitades
    se afirman juntas porque cada una sin la otra es un defecto.

    Rojo alcanzable: fijar `ES_ENSAYO = True` en cualquiera de los dos
    cuadernos, o sacarle la guarda de escala a cualquiera de los dos pasos.
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

    monkeypatch.setattr(config, "is_pilot_scale", lambda: False)
    diagnostico = _correr_el_diagnostico(monkeypatch, tmp_path)
    capsys.readouterr()
    (buscada,) = diagnostico["buscadas"]
    assert buscada["reduction"].pilot is False
    assert buscada["pilot"] is False
    (escrito,) = list(tmp_path.rglob("diagnostic.json"))
    assert "Pilot" not in escrito.parts, (
        "el diagnóstico compuso su destino con una constante")

    # la mitad del paso: se niega antes de abrir el cuaderno
    monkeypatch.setattr(steps, "_ejecutar", lambda nombre: pytest.fail(
        f"el paso abrió {nombre} a escala completa, fuera de sus raíces"))
    for paso in (steps.barrido_de_ruido, steps.diagnostico_de_ruido):
        with pytest.raises(SystemExit) as caido:
            paso()
        assert "escala" in str(caido.value)


# --------------------------------------------------- las dos pasadas de la campaña
#
# El cuaderno se ejecuta de verdad --- sus celdas, no una lectura de su texto ---
# con `harness` sustituido: lo que se afirma es a qué niveles corre la campaña, en
# qué árbol cae cada pasada y de dónde salen los techos con los que corre. Nada de
# eso se puede leer de las constantes: `NOISE_REPORTED` puede valer 0.2 mientras el
# cuaderno corre dos veces el nivel limpio, y las dos cosas dan una suite verde.


def _correr_las_celdas_de_la_campana(monkeypatch, tmp_path) -> dict:
    """Ejecuta el cuaderno de la campaña sin máquina, y devuelve lo que pidió.

    Todas las celdas menos la de arranque: ésa busca el repositorio en el disco y
    manosea `sys.path`, y adentro de la suite el paquete ya está importado. Cuál
    es se deriva de su contenido y no de un índice --- una celda que se agregue
    arriba correría el índice y dejaría este helper leyendo otra cosa.

    Las corridas se registran con la lectura de techos que había en el momento de
    pedirlas, que es lo que permite afirmar «una sola vez, antes del bucle» en vez
    de «una sola vez en total»: las dos son verdes hoy y se rompen distinto.
    """
    import json

    from MIL_CREDA_Benchmark import bags, config, harness

    _sin_maquina(monkeypatch, tmp_path)
    # `_sin_maquina` deja un ambiente de una sola clave y la celda 2 lo imprime
    # por campo, así que acá lleva los tres campos que esa celda nombra.
    monkeypatch.setattr(harness, "environment", lambda: {
        "platform": "stub", "torch": "stub", "selfHosted": False})
    escalas = {"agrupado": [], "por_transferencia": [], "registro": []}

    def _registro(pilot=None):
        escalas["registro"].append(pilot)
        return {"stub": True}

    monkeypatch.setattr(harness, "search_record", _registro)
    monkeypatch.setattr(harness, "search_ceilings", lambda *a, **k: pytest.fail(
        "la campaña lanzó una búsqueda: mediría el método y la falta de "
        "coeficiente a la vez, y a escala completa son nueve horas y media"))
    monkeypatch.setattr(harness, "run_search", lambda *a, **k: pytest.fail(
        "la campaña lanzó una búsqueda"))
    monkeypatch.setattr(bags, "build", lambda *a, **k: {"stub": True})
    monkeypatch.setattr(harness, "run_one", lambda *a, **k: {"seconds": 1.0})

    # Un valor DISTINTO por escala, por el mismo motivo que en el barrido: un
    # doble que contestara lo mismo por las dos dejaría verde a una campaña de
    # ensayo corriendo bajo los techos de la búsqueda COMPLETA.
    AGRUPADO = {True: {"milcreda": 1e-2, "creda": 1e-4},
                False: {"milcreda": 9e-1, "creda": 9e-1}}
    POR_TRANSFERENCIA = {True: {"milcreda": {"M->U": 1e-2}},
                         False: {"milcreda": {"M->U": 9e-1}}}
    techos = AGRUPADO[True]
    por_transferencia = POR_TRANSFERENCIA[True]
    lecturas = {"agrupado": 0, "por_transferencia": 0, "en_vigor": 0}

    def _agrupado(pilot=None):
        lecturas["agrupado"] += 1
        escalas["agrupado"].append(pilot)
        return AGRUPADO[bool(pilot)]

    def _por_transferencia(pilot=None):
        lecturas["por_transferencia"] += 1
        escalas["por_transferencia"].append(pilot)
        return POR_TRANSFERENCIA[bool(pilot)]

    def _en_vigor(reduccion, dispositivo, **kwargs):
        from dataclasses import replace as _replace

        lecturas["en_vigor"] += 1
        return _replace(reduccion, ceilings=techos,
                        ceilingsByTransfer=por_transferencia)

    monkeypatch.setattr(config, "ceilings_on_record", _agrupado)
    monkeypatch.setattr(config, "ceilings_by_transfer_on_record", _por_transferencia)
    monkeypatch.setattr(harness, "with_ceilings_in_force", _en_vigor)

    corridas: list[dict] = []

    def _campaign(reduccion, dispositivo, **kwargs):
        corridas.append({"reduction": reduccion, "lecturas": dict(lecturas)})
        return {"stub": reduccion.labelNoise}

    monkeypatch.setattr(harness, "campaign", _campaign)

    # Las corridas que cada pasada va a LEER, escritas acá y no por el doble de
    # `campaign`: si las escribiera el doble, el cuaderno leería el archivo que
    # él mismo eligió y la lectura no probaría ningún destino. Con una cantidad
    # distinta por nivel, lo que imprime dice cuál de los dos árboles abrió.
    escala = config.is_pilot_scale()
    lineas = {config.NOISE: 2, config.NOISE_REPORTED: 3}
    raices = {}
    for nivel, cuantas in lineas.items():
        raiz = config.results_for(nivel, "campaign", escala)
        raiz.mkdir(parents=True, exist_ok=True)
        (raiz / "runs.jsonl").write_text(
            "".join(json.dumps({"nivel": nivel, "i": i}) + "\n"
                    for i in range(cuantas)), encoding="utf-8")
        raices[nivel] = raiz

    ambito = _correr_las_celdas("Benchmark_Campaign_v1.ipynb")

    return {"corridas": corridas, "lecturas": lecturas, "raices": raices,
            "lineas": lineas, "escala": escala, "techos": techos,
            "porTransferencia": por_transferencia, "escalas": escalas,
            "agrupadoPorEscala": AGRUPADO,
            "porTransferenciaPorEscala": POR_TRANSFERENCIA, "ambito": ambito}


def test_la_campana_corre_las_dos_pasadas_y_la_segunda_es_la_contaminada(
        tmp_path, monkeypatch, capsys) -> None:
    """Dos pasadas, y la segunda al nivel que el informe muestra.

    La campaña es cada transferencia a UNA tasa, así que los dos niveles del
    informe son dos pasadas de esa misma forma. Corría sólo la limpia, y por eso
    las celdas contaminadas del informe y la mitad contaminada del latente decían
    «no hay corridas»: leen `results_for(NOISE_REPORTED, "campaign", ...)`, que
    ningún paso escribía.

    Lo que se afirma no es «corre a más de un nivel» --- eso lo cumple un cuaderno
    que repita dos veces el limpio, que es exactamente la segunda pasada que no
    sirve para nada ---, sino que el segundo nivel es `NOISE_REPORTED`, que es
    distinto del primero, y que cada pasada abrió el árbol de SU tasa: las dos
    `runs.jsonl` traen distinta cantidad de líneas, así que lo que el cuaderno
    imprime dice cuál abrió.

    Y el destino de la contaminada es una raíz declarada de este paso. Sin ese
    último tramo la campaña podría escribir donde nadie la vigila y la forja lo
    leería como `foreign`.

    Rojo alcanzable: sacarle el bucle a la celda 7, poner `(config.NOISE,
    config.NOISE)` en `NIVELES`, componer la raíz desde la constante en vez de
    desde la reducción de la pasada, o sacarle las raíces contaminadas a
    `produces`.
    """
    from MIL_CREDA_Benchmark import config

    corrido = _correr_las_celdas_de_la_campana(monkeypatch, tmp_path)
    salida = capsys.readouterr().out
    corridas = corrido["corridas"]

    # los dos niveles, en orden, y el segundo no es el primero otra vez
    assert [c["reduction"].labelNoise for c in corridas] == [
        config.NOISE, config.NOISE_REPORTED]
    assert config.NOISE_REPORTED != config.NOISE, (
        "el nivel contaminado es el limpio: no hay segunda pasada que valga")
    assert len(corridas) == 2

    # la misma forma y la misma escala en las dos: una campaña, no un barrido
    for corrida in corridas:
        assert corrida["reduction"].kind == "campaign"
        assert corrida["reduction"].pilot == corrido["escala"]
        assert corrida["reduction"].seeds == list(config.SEEDS)

    # cada pasada abrió el árbol de su propia tasa, medido por lo que leyó ahí
    for nivel, raiz in corrido["raices"].items():
        assert str(raiz) in salida, f"la pasada a ρ={nivel:g} no nombró {raiz}"
        assert f"{corrido['lineas'][nivel]} corridas" in salida, (
            f"la pasada a ρ={nivel:g} leyó otro árbol que el suyo")
    assert corrido["raices"][config.NOISE] != corrido["raices"][config.NOISE_REPORTED]

    # y el árbol de la contaminada es raíz declarada de este paso
    contaminada = corrido["raices"][config.NOISE_REPORTED]
    declaradas = paquete.__steps__["campaign-local"]["produces"]
    relativa = contaminada.relative_to(config.PRODUCT).as_posix()
    for hoja in ("runs.jsonl", "summary.json", "shard.json"):
        assert f"{relativa}/{hoja}" in declaradas, (
            f"la pasada contaminada escribe {relativa}/{hoja} sin declararlo")
    assert config.models_for(
        config.NOISE_REPORTED, "campaign", corrido["escala"]
    ).relative_to(config.PRODUCT).as_posix() in declaradas
    assert config.keeps_checkpoints(config.NOISE_REPORTED), (
        "la pasada contaminada no guardaría pesos y el latente no tendría qué leer")


def test_la_pasada_contaminada_reusa_los_techos_limpios_sin_volver_a_buscar(
        tmp_path, monkeypatch, capsys) -> None:
    """El coeficiente se elige en limpio una vez, y las dos pasadas corren bajo él.

    Es la decisión que el eje del ruido ya declara --- los techos salen de la
    búsqueda en limpio y se mantienen fijos --- y lo que esa decisión cuesta lo
    separa `noise-diagnostic`, que re-busca a su nivel y no gobierna el registro.
    Acá se afirma la mitad que puede romperse sola: que la segunda pasada no
    dispare una búsqueda nueva.

    Y no alcanza con contarlas al final. Los techos se leen UNA vez y ARRIBA del
    bucle: leerlos por pasada daría los mismos números hoy, dejaría la campaña a
    merced de un registro que cambie a mitad de corrida y --- a escala completa,
    donde la rama que resuelve techos es `with_ceilings_in_force` --- sería la
    puerta por la que la búsqueda entera entra sin que nadie la autorice. Por eso
    cada corrida trae la cuenta de lecturas que había cuando se la pidió: las dos
    tienen que traer la misma, y esa misma tiene que ser la final.

    Rojo alcanzable: mover la celda de los techos adentro del bucle, armar una
    `Reduction` nueva por pasada en vez de un `replace`, o buscar techos al nivel
    contaminado.
    """
    from MIL_CREDA_Benchmark import config

    corrido = _correr_las_celdas_de_la_campana(monkeypatch, tmp_path)
    capsys.readouterr()
    corridas, lecturas = corrido["corridas"], corrido["lecturas"]

    # una sola resolución de techos en todo el cuaderno...
    assert sum(lecturas.values()) in (1, 2), lecturas
    assert lecturas["en_vigor"] + lecturas["agrupado"] == 1, (
        "los techos se resolvieron más de una vez")

    # ...y ninguna adentro del bucle: las dos pasadas vieron la misma cuenta,
    # y esa cuenta es la final
    assert [c["lecturas"] for c in corridas] == [lecturas, lecturas]

    # los mismos techos, el mismo objeto, en las dos pasadas
    assert len({id(c["reduction"].ceilings) for c in corridas}) == 1
    for corrida in corridas:
        assert corrida["reduction"].ceilings == corrido["techos"]
        assert corrida["reduction"].ceilingsByTransfer == corrido["porTransferencia"]

    # y los techos son los del registro, no unos buscados acá: `search_ceilings`
    # está sustituida por una falla, así que llegar hasta acá ya lo dice
    assert corridas[-1]["reduction"].labelNoise == config.NOISE_REPORTED
    assert corridas[-1]["reduction"].ceilings == corridas[0]["reduction"].ceilings


if __name__ == "__main__":
    unittest.main()
