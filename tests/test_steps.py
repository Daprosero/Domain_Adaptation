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

    def test_los_cuadernos_que_los_pasos_nombran_existen(self):
        for nombre in ("verification.ipynb", "Benchmark_Search_v1.ipynb",
                       "Benchmark_Report_v1.ipynb", "Benchmark_Latent_v1.ipynb"):
            with self.subTest(cuaderno=nombre):
                self.assertTrue((steps.CUADERNOS / nombre).is_file(),
                                f"{steps.CUADERNOS / nombre} no existe")

    def test_la_campana_no_se_ofrece_como_paso_local(self):
        """La ausencia es la decisión, no un olvido.

        `Benchmark_Campaign_v1.ipynb` lanza al servicio remoto. Un paso local
        que lo corriera saltearía la aprobación humana por envío, que es la
        única precondición que un llamador no puede satisfacer por su cuenta.
        Sin esta afirmacion la omisión se lee como un descuido y alguien la
        "completa".
        """
        self.assertTrue((steps.CUADERNOS / "Benchmark_Campaign_v1.ipynb").is_file(),
                        "el cuaderno de campaña no está: esta prueba no probaría nada")
        self.assertNotIn("Benchmark_Campaign_v1.ipynb",
                         _cuadernos_nombrados_por_los_pasos())

    def test_todo_cuaderno_del_arbol_lo_corre_un_paso_o_dice_por_que_no(self):
        """El defecto no es un cuaderno sin paso: es que nada lo note.

        `Benchmark_Noise_Diagnostic_v1.ipynb` vivió con cinco celdas y ninguna
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


def test_el_diagnostico_corre_en_el_tope_del_rango_y_paga_una_sola_medicion(
        tmp_path, monkeypatch) -> None:
    """Dónde corre el diagnóstico, sobre qué, con quiénes, y cuánto cuesta.

    El nivel es el tope del rango declarado y no un número escrito acá: en el
    extremo el coeficiente está bajo la mayor presión, así que un techo
    re-buscado que no recupera nada ahí no recupera nada en ningún lado, y la
    lectura no depende de dónde eligió mirar nadie. El tope lo fija el rango y no
    un resultado.

    Necesita tres puntos y paga uno solo. Lo que se afirma acá es la MEDICIÓN
    nueva: una sola llamada a la búsqueda, sobre la transferencia de la curva y
    ninguna otra, y ninguna campaña. Los otros dos puntos ya están en el
    registro del barrido -- este paso los lee, no los vuelve a correr.

    `D` y `G` y nadie más: los dos métodos completos, uno por familia, y los
    únicos que llevan coeficiente. `A` y `B` no tienen término de adaptación al
    que re-buscarle un techo.

    Rojo alcanzable: correr en el medio del rango, buscar sobre las seis
    transferencias, llamar a `campaign`, o agregar un brazo sin coeficiente.
    """
    from MIL_CREDA_Benchmark import config, contamination, harness, steps

    _sin_maquina(monkeypatch, tmp_path)

    buscadas, campanas = [], []

    def _search_ceilings(reduccion, dispositivo, **kwargs):
        buscadas.append({"reduction": reduccion, **kwargs})
        return {"milcreda": {"ceiling": 1e-2, "byTransfer": {"M->U": 1e-2}}}

    monkeypatch.setattr(harness, "search_ceilings", _search_ceilings)
    monkeypatch.setattr(harness, "campaign",
                        lambda *a, **k: campanas.append((a, k)))
    monkeypatch.setattr(contamination, "load", lambda *a, **k: None)

    registro = steps.diagnostico_de_ruido()

    # el tope del rango, leído del rango y no escrito
    assert config.NOISE_DIAGNOSTIC_LEVEL == config.NOISE_LEVELS[-1]
    assert config.NOISE_DIAGNOSTIC_LEVEL == max(config.NOISE_LEVELS)
    assert registro["level"] == config.NOISE_DIAGNOSTIC_LEVEL

    # una sola medición nueva, y ninguna campaña
    assert len(buscadas) == 1, "el diagnóstico paga más de una búsqueda"
    assert campanas == [], "el diagnóstico corrió una campaña que no le toca"
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
        tmp_path, monkeypatch) -> None:
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
    el paso corra una campaña al nivel de diagnóstico, o mostrar
    `render_diagnostic` en el informe del veredicto.
    """
    import json
    from pathlib import Path as _Path

    from MIL_CREDA_Benchmark import config, contamination, harness, steps, tables

    _sin_maquina(monkeypatch, tmp_path)
    monkeypatch.setattr(harness, "search_ceilings",
                        lambda *a, **k: {"milcreda": {"ceiling": 1e-2}})
    monkeypatch.setattr(harness, "campaign", lambda *a, **k: pytest.fail(
        "el diagnóstico no corre campañas"))
    monkeypatch.setattr(contamination, "load", lambda *a, **k: None)

    registro = steps.diagnostico_de_ruido()

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


def test_el_barrido_lee_los_techos_una_vez_y_los_mantiene_en_los_cinco_niveles(
        tmp_path, monkeypatch) -> None:
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

    Rojo alcanzable: buscar techos adentro del barrido, releerlos por nivel,
    correr las seis transferencias, o saltearse un nivel declarado.
    """
    from dataclasses import replace as _replace

    from MIL_CREDA_Benchmark import config, harness, steps

    _sin_maquina(monkeypatch, tmp_path)

    lecturas = {"agrupado": 0, "por_transferencia": 0}
    corridas = []

    monkeypatch.setattr(harness, "search_record", lambda pilot=False: {"stub": True})
    monkeypatch.setattr(harness, "search_ceilings", lambda *a, **k: pytest.fail(
        "el barrido lanzó una búsqueda: mediría el ruido y el coeficiente a la vez"))

    def _agrupado():
        lecturas["agrupado"] += 1
        return {"milcreda": 1e-2, "creda": 1e-4}

    def _por_transferencia():
        lecturas["por_transferencia"] += 1
        return {"milcreda": {"M->U": 1e-2}}

    monkeypatch.setattr(config, "ceilings_on_record", _agrupado)
    monkeypatch.setattr(config, "ceilings_by_transfer_on_record", _por_transferencia)

    def _campaign(reduccion, dispositivo, **kwargs):
        corridas.append({"reduction": reduccion, **kwargs})
        return {"runs": []}

    monkeypatch.setattr(harness, "campaign", _campaign)

    devuelto = steps.barrido_de_ruido()

    # los techos se leen una sola vez, antes del bucle
    assert lecturas == {"agrupado": 1, "por_transferencia": 1}

    # un nivel declarado por corrida, en el orden declarado, y ninguno de más
    assert [c["reduction"].labelNoise for c in corridas] == config.NOISE_LEVELS
    assert sorted(devuelto) == sorted(f"{t:g}" for t in config.NOISE_LEVELS)
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
    assert _replace  # el paso arma cada nivel por `replace`, no por mutación


if __name__ == "__main__":
    unittest.main()
