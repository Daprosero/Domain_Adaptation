"""Niveles, método por método y transferencia por transferencia.

`verdict.py` responde *quién gana cada peldaño*. Esto responde la otra pregunta,
la que un lector se hace primero — *dónde queda cada método* — y es un módulo
aparte a propósito: una tabla de niveles no puede atribuir nada, y una tabla de
peldaños no puede decir qué tan bueno es nadie. Se imprimen las dos, una después
de la otra.

Solo biblioteca estándar, como `verdict.py`: producir los números necesita torch,
ordenarlos no.

El texto visible está en español porque el informe se lee en español. Las claves
de los registros, los identificadores y los nombres de los métodos siguen en
inglés: son contrato de datos, no prosa.

Tres cosas que este módulo se niega a hacer, cada una por una razón que vale la
pena decir una vez:

* Imprime un decimal. Con `EVAL_BAGS` bolsas de evaluación la exactitud se mueve
  de a `100 / EVAL_BAGS` puntos, así que un segundo decimal describiría una
  precisión que la medición no tiene.
* Su `±` es la dispersión **entre semillas**, que es la que consume la regla de
  veredicto. El `±` del artículo de referencia es la dispersión entre lotes de
  evaluación dentro de una corrida: un número que se ve razonable por pocas
  repeticiones que se hayan corrido, que es exactamente la propiedad equivocada
  para una tabla que se lee al lado de un piloto.
* Nunca esconde el sello de piloto. Por debajo del piso de repeticiones
  declarado lo dice en el encabezado e imprime la tabla igual, porque el piloto
  tiene que ser el mismo programa que la campaña.
"""

from __future__ import annotations

import collections

import math
import re
from typing import Iterable

from MIL_CREDA_Benchmark import ceiling_record, config

#: Métricas que se informan como porcentaje en lugar de como fracción.
PERCENT = ("targetAccuracy", "sourceAccuracy")

#: Cómo se llama cada dimensión en el informe, y en qué unidad.
SPANISH = {
    "targetAccuracy": ("exactitud en destino", "%"),
    "sourceAccuracy": ("exactitud en fuente", "%"),
    "seconds": ("tiempo de entrenamiento", "s"),
    "peakMiB": ("memoria pico", "MiB"),
    "parameters": ("parámetros", ""),
    "contribution": ("peso del término de adaptación", ""),
}

BETTER = {config.HIGHER: "más alto es mejor", config.LOWER: "más bajo es mejor",
          None: "descriptivo: se informa, no se disputa"}

#: El azar de la regla que separa fuente de destino. Son dos etiquetas —de qué
#: dominio viene cada punto— y no las tres bases del protocolo, así que sale de
#: acá y no de `config.DOMAINS`.
DOMAIN_CHANCE = 1.0 / 2

#: A partir de acá se dice que la atención dejó de repartir. Es un umbral de
#: lectura, no una medición: va escrito una vez y se lo puede discutir.
UNIFORM_ATTENTION = 0.99

#: Cuántos puntos porcentuales tienen que separar los dos cambios relativos para
#: leerlos como distintos. Por debajo de esto las dos distancias se movieron
#: parejo y lo que cambió fue la escala del espacio, no su geometría. Declarado
#: acá y no repetido en la conclusión: el objetivo lo anuncia y la conclusión lo
#: aplica, y si fueran dos números el informe prometería un umbral y usaría otro.
SCALE_TOLERANCE = 1.0


def spread(values: list[float]) -> dict:
    """Media, dispersión y pico de una celda, sobre sus repeticiones."""
    n = len(values)
    if not n:
        return {"mean": float("nan"), "stdev": 0.0, "max": float("nan"), "n": 0}
    mean = sum(values) / n
    stdev = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "stdev": stdev, "max": max(values), "n": n}


def cells(runs: Iterable[dict], metric: str) -> dict:
    """{(brazo, transferencia): spread} sobre cada repetición de esa celda."""
    gathered: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        gathered.setdefault((run["arm"], run["transfer"]), []).append(float(run[metric]))
    return {key: spread(values) for key, values in gathered.items()}


def table(runs: Iterable[dict], metric: str) -> list[dict]:
    """Una fila por método: cada transferencia, el promedio y el pico.

    `avg` promedia las medias por transferencia, así una transferencia cuenta una
    vez por muchas repeticiones que haya corrido. `max` promedia los máximos por
    transferencia — el método en su semilla más afortunada de cada una — y existe
    para que el pico quede documentado como número acá, en lugar de colarse en
    una figura eligiendo el mejor modelo.
    """
    runs = list(runs)
    grid = cells(runs, metric)
    shares = cells(runs, "contribution")
    labels = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS]

    rows = []
    for arm in config.ARM_ORDER:
        present = [grid[(arm, label)] for label in labels if (arm, label) in grid]
        if not present:
            continue
        rows.append({
            "arm": arm,
            "name": config.NAME_OF[arm],
            "byTransfer": {label: grid.get((arm, label)) for label in labels},
            "avg": sum(c["mean"] for c in present) / len(present),
            "max": sum(c["max"] for c in present) / len(present),
            "share": (sum(shares[(arm, label)]["mean"] for label in labels
                          if (arm, label) in shares)
                      / max(1, sum(1 for label in labels if (arm, label) in shares))),
        })
    return rows


def _scaled(value: float, metric: str) -> float:
    return value * 100.0 if metric in PERCENT else value


def _notes_block(notes: list[str], markdown: bool) -> str:
    """El encabezado de una tabla, en un formato que no se aplaste al renderizarse.

    En Markdown dos líneas seguidas se juntan en un solo párrafo, y ahí el sello de
    piloto deja de leerse como una advertencia aparte justo cuando más importa que
    se lea. Como cita, cada línea sigue siendo una línea.
    """
    if markdown:
        return "\n".join(f"> {note}  " for note in notes)
    return "\n".join(notes)


def stamp(reduction: dict, markdown: bool = False) -> str:
    """Los límites de la corrida, dichos una vez arriba de todo y calculados.

    Antes iban repetidos en el encabezado de cada tabla. Un aviso que aparece ocho
    veces enseña a saltearlo, que es exactamente lo contrario de para lo que está:
    la octava vez ya nadie lo lee, y la que importaba era una de esas ocho. Va una
    sola vez, antes que cualquier número, y desde ahí ata todo lo que sigue.

    Calculado y no escrito: es la única forma de que no pueda quedar describiendo
    una corrida distinta de la que produjo las tablas de abajo.
    """
    seeds = reduction.get("seeds", config.SEEDS)
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else int(seeds or 0)
    notes = [f"{reduction.get('backbone', config.BACKBONE)}  ·  "
             f"{reduction.get('epochs', config.EPOCHS)} épocas  ·  "
             f"{n_seeds} repetición(es)  ·  "
             f"{reduction.get('revision', config.REVISION)}",
             f"la exactitud se mueve de a {100 / config.EVAL_BAGS:.2f} puntos sobre "
             f"{config.EVAL_BAGS} bolsas de evaluación: nada por debajo de eso lo "
             f"resuelve una transferencia sola",
             *_stamp(reduction)]
    return _notes_block(notes, markdown)


def _repetitions(reduction: dict) -> int:
    """Cuántas repeticiones hay detrás de una tabla, leídas de su propia reducción."""
    seeds = reduction.get("seeds", config.SEEDS)
    return len(seeds) if isinstance(seeds, (list, tuple)) else int(seeds or 0)


def _stamp(reduction: dict) -> list[str]:
    """Lo que separa un piloto de un resultado. Lo consume `stamp`, una vez."""
    seeds = reduction.get("seeds", config.SEEDS)
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else int(seeds or 0)
    notes = []
    if n_seeds < 3:
        notes.append(f"!! {n_seeds} repetición(es): el ± de abajo es cero por "
                     f"construcción, no por acuerdo. Son estimaciones puntuales, "
                     f"no resultados.")
    if n_seeds < len(config.FULL_SEEDS):
        notes.append(f"!! piloto: el protocolo declara {len(config.FULL_SEEDS)} "
                     f"repeticiones y {config.FULL_EPOCHS} épocas. Nada de esto es "
                     f"un resultado.")
    return notes


#: Contra qué valor se compara cada lectura del informe. Es lo que un lector que no
#: conoce la métrica necesita para saber si un número es bueno: una dirección le
#: dice para qué lado mirar y nada sobre dónde termina lo bueno.
#:
#: Cada hito se calcula de `config` y nunca se escribe. Tipeado envejecería igual
#: que una medición tipeada: el día que cambien las clases o las bolsas de
#: evaluación, la frase seguiría nombrando el azar viejo.
def objective(key: str, markdown: bool = True) -> str:
    """Qué valor buscamos en la lectura que sigue."""
    clase = 1.0 / config.CLASSES
    transferencias = len(config.VERDICT_TRANSFERS)
    metas = {
        "seconds":
            "**Buscamos el número más bajo**, y sobre todo que la diferencia entre "
            "métodos sea chica: uno más lento sigue siendo utilizable, uno diez "
            "veces más lento deja de serlo.",
        "sourceAccuracy":
            f"**Buscamos que sea alta y que no caiga** al sumar la adaptación. El "
            f"piso es el azar de acertar una clase entre {config.CLASSES}, o sea "
            f"{clase:.3f}; el techo es 1.000.",
        "targetAccuracy":
            f"**Buscamos el valor más alto posible**, entre el azar de "
            f"{clase:.3f} y 1.000 — y leído junto a la tabla de fuente, porque una "
            f"subida acá pagada con una caída allá no es adaptación.",
        "rungs":
            f"**Buscamos acuerdo, no magnitud**: que el peldaño se incline para el "
            f"mismo lado en las {transferencias} transferencias, o sea "
            f"{transferencias}/{transferencias} o 0/{transferencias}. Cualquier "
            f"cosa entre medio —3/{transferencias}, por ejemplo— promedia parecido "
            f"y no dice nada.",
        "supervised":
            "**Buscamos que la curva baje y se quede baja.** Que caiga más rápido "
            "es mejor, y que no se desestabilice al sumarle la adaptación: si el "
            "ajuste se deshace, el término de adaptación lo rompió.",
        "adaptation":
            "**Buscamos que la curva viva dentro de [0, 1] y tienda a cero**, y que "
            "ocupe la misma parte del intervalo en las seis transferencias. Salirse "
            "de la banda o cambiar de escala entre pares de dominios es el hallazgo.",
        "contribution":
            f"**Buscamos que ningún método quede en cero y que los valores sean "
            f"comparables entre sí.** En cero, «el término no hizo nada» y «el "
            f"término no pesó nada» son la misma figura; con órdenes distintos "
            f"entre métodos, el peldaño le acredita al mecanismo lo que hizo la "
            f"escala.",
        "floors":
            "**Buscamos que la diferencia entre los dos pisos sea menor que la que "
            "los separa de cualquier método con adaptación.** Si lo es, la segunda "
            "columna no muestra nada que la primera no muestre y se colapsa.",
        "grid":
            "**Buscamos que los dos dominios se mezclen dentro de cada clase y que "
            "las clases sigan separadas entre sí.** Mezclar todo también junta los "
            "dominios, y eso no es alinear sino colapsar.",
        "gains":
            f"**Buscamos que la fuente no caiga y que el destino suba**, las dos "
            f"contra el propio piso del método —el mismo brazo con la adaptación "
            f"apagada— y dentro de cada transferencia. Es lo que un regularizador "
            f"tiene que dar: si baja la fuente, lo que hubo no fue adaptación sino "
            f"un intercambio. **Y buscamos acuerdo antes que magnitud**: las "
            f"{transferencias} transferencias inclinándose para el mismo lado. "
            f"Ninguno de los dos promedios decide por sí solo, y por eso van los "
            f"dos: el de puntos dice cuánto suma en una transferencia típica, el "
            f"de porcentajes cuánto suma **relativo a la dificultad de cada una** "
            f"—los pisos van de 23% a 81%— y pueden salir con signos opuestos "
            f"sobre los mismos datos. Cuando eso pasa, el rango es lo que hay que "
            f"leer: un `de +15 a -5` dice que el promedio no es la historia.",
        "geometry.ratio":
            "**Buscamos que tienda a cero**, y solo vale si el denominador se "
            "sostiene: por debajo de 1.000 la misma clase está más junta entre "
            "dominios que dos clases distintas dentro de uno, que es lo que se "
            "quiere. Pero una razón también cae si todo colapsa.",
        "geometry.distances":
            f"**Buscamos que la primera baje más que la segunda**, y el «más» es "
            f"lo único que se puede leer: una distancia en un embedding no tiene "
            f"escala fija, así que ninguno de los dos números significa nada por "
            f"su valor absoluto ni se compara entre métodos. Las tablas de abajo "
            f"están en esas unidades sin escala y se leen **contra el piso del "
            f"propio método, transferencia por transferencia**: `CREDA` y `CREDA*` "
            f"contra `Baseline`, toda la familia `MIL-` contra `MIL-Baseline`. La "
            f"conclusión reporta esos dos cambios en porcentaje respecto de ese "
            f"piso, y llama alineación a que el de la misma clase entre dominios "
            f"quede **más de {SCALE_TOLERANCE:g} punto porcentual por debajo** del "
            f"de las clases distintas. Dentro de ese punto las dos se movieron "
            f"parejo: cambió la escala del espacio y la razón de arriba no tenía "
            f"por qué moverse. Ojo con lo que esto **no** dice: no pide que la "
            f"segunda se quede quieta. Puede caer mucho y aun así haber "
            f"alineación, siempre que la primera caiga más — que la segunda se "
            f"desplome es un colapso y se cuenta aparte, en la conclusión de la "
            f"razón.",
        "domainSeparability":
            f"**Buscamos que se acerque a {DOMAIN_CHANCE:.3f}**, que es el azar de "
            f"decidir entre dos dominios. No más bajo: por debajo del azar la regla "
            f"acierta al revés, que es otra forma de decir que la información de "
            f"dominio sigue ahí. Un 1.000 es un dominio perfectamente reconocible.",
        "correspondence":
            f"**Buscamos que el triángulo destacado caiga entre círculos de su "
            f"mismo color**, y que eso pase más en la columna con término local que "
            f"en la de al lado. El azar de acertar la clase es {clase:.3f}.",
        "correspondence.massOnTrueClass":
            f"**Buscamos que supere claramente {clase:.3f}**, que es el azar de una "
            f"clase entre {config.CLASSES}, y que suba más en el método que declara "
            f"el término local que en el mismo método sin él. El máximo es 1.000.",
        "attentionSpread":
            f"**Buscamos un valor lejos de 1.000.** La entropía está normalizada, "
            f"así que 1.000 es la media uniforme: la atención dejó de elegir y "
            f"cualquier lectura de la masa de arriba queda en duda. Es descriptivo, "
            f"no se disputa, pero condiciona lo anterior.",
        "ceilings":
            f"**Buscamos que el criterio se incline**, no un número en particular. "
            f"El neutro es {config.RAMP_CEILING:g}: un techo que aterriza ahí "
            f"confirma la normalización por medición y no por argumento. Lo que "
            f"invalida la elección es que la búsqueda no distinga —ahí el ganador "
            f"lo pone la regla y no el criterio—, y la tabla lo dice en la columna "
            f"que corresponda a su motor: una rejilla plana con semillas que no "
            f"coinciden, o una meseta ancha. **Meseta uno significa que el criterio "
            f"decidió.** Sobre el rango continuo la meseta la define la resolución "
            f"del instrumento, {config.SEARCH_RESOLUTION:g}, que es una bolsa de "
            f"las {config.VALID_BAGS} del rol de búsqueda: dos techos que difieren "
            f"en menos que eso no son distinguibles por la medición.",
        "ceilings.byTransfer":
            f"**Buscamos saber cuántas de las {transferencias} transferencias "
            f"corren a un techo elegido mirándolas.** La búsqueda mide "
            f"{len(config.SEARCH_TRANSFERS)}; las otras "
            f"{transferencias - len(config.SEARCH_TRANSFERS)} heredan, y esa "
            f"herencia es una aplicación fuera de muestra. Lo que hay que ver es "
            f"si alguna de las medidas se aparta del ganador agrupado: si ninguna "
            f"lo hace, separar las dos lecturas no cambió nada y la familia sigue "
            f"corriendo a un coeficiente único; si alguna se aparta, deja de "
            f"hacerlo y su promedio entre transferencias mezcla dos escalares.",
    }
    texto = metas.get(key)
    if texto is None:
        return f"(sin objetivo declarado para `{key}`)"
    return f"> {texto}" if markdown else texto


def render_ceilings(record: dict | None, markdown: bool = False) -> str:
    """La rejilla de la búsqueda, una fila por familia y una columna por techo.

    Recibe el registro en lugar de leerlo: este módulo es solo biblioteca estándar
    y el registro lo lee `harness.search_record`, que necesita torch para todo lo
    demás. Y recibirlo entero, no solo el ganador: un techo elegido entre cuatro
    puntajes idénticos y uno elegido por una diferencia real son el mismo número y
    no la misma evidencia, así que la fila muestra la rejilla y marca cuál ganó.
    """
    # Dos formas, dos tablas. Una rejilla tiene columnas: los mismos techos en
    # todas las familias, así que la fila se lee de izquierda a derecha y la
    # inclinación se ve. Una búsqueda por trials no las tiene — cada familia y
    # cada transferencia visitaron puntos distintos de un rango continuo — y
    # forzarla a columnas inventaría un eje compartido que nadie midió.
    if record and all(ceiling_record.kind_of(e) == ceiling_record.KIND_OPTUNA
                      for e in record.values()):
        return _render_ceilings_trials(record, markdown=markdown)
    if not record:
        return ("Sin búsqueda de techos: no hay rejilla que mostrar. La campaña se "
                "niega a correr hasta que exista.")
    grid = sorted({fila["ceiling"] for entrada in record.values()
                   for fila in entrada["grid"]})
    columns = ["Familia", "Brazo", *(f"{c:g}" for c in grid)]

    def cells(entrada: dict) -> list[str]:
        puntajes = {fila["ceiling"]: fila.get(entrada["criterion"])
                    for fila in entrada["grid"]}
        out = []
        for techo in grid:
            valor = puntajes.get(techo)
            texto = "—" if valor is None else f"{_scaled(valor, entrada['criterion']):.1f}"
            # El elegido se marca en la propia celda: una columna «elegido» aparte
            # repetiría un número que la fila ya muestra, y son dos cosas que se
            # pueden mover por separado.
            if techo == entrada["ceiling"]:
                texto = f"**{texto}**" if markdown else f"[{texto}]"
            out.append(texto)
        return out

    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for familia, entrada in sorted(record.items()):
            lines.append("| " + " | ".join(
                [f"`{familia}`", f"`{entrada['arm']}`", *cells(entrada)]) + " |")
        return "\n".join(lines)

    width = max(14, max((len(f) for f in record), default=14) + 2)
    lines = [f"{'Familia':<{width}}{'Brazo':>8}"
             + "".join(f"{c:g}".rjust(12) for c in grid)]
    for familia, entrada in sorted(record.items()):
        lines.append(f"{familia:<{width}}{entrada['arm']:>8}"
                     + "".join(cell.rjust(12) for cell in cells(entrada)))
    return "\n".join(lines)


def _render_ceilings_trials(record: dict, markdown: bool = False) -> str:
    """La búsqueda por trials, una fila por familia y transferencia.

    Lo que hay que poder leer de un vistazo no es el techo: es **cuán ancha fue
    la meseta**. Un techo elegido entre uno es una medición; el mismo número
    elegido entre nueve es la regla hablando, y los dos se imprimen igual si la
    tabla muestra solo el ganador.
    """
    filas = []
    for familia, entrada in sorted(record.items()):
        detalle = entrada.get("perTransfer") or {}
        for etiqueta in sorted(detalle):
            d = detalle[etiqueta]
            meseta = d.get("plateau") or []
            filas.append({
                "familia": familia, "brazo": entrada.get("arm", "?"),
                "transferencia": etiqueta,
                "techo": d.get("ceiling"),
                "valor": d.get("value"),
                "mejor": d.get("best"),
                "meseta": len(meseta),
                "regla": bool(d.get("decidedByFlatRule")),
                "trials": d.get("trials"),
            })
    if not filas:
        return "(la búsqueda por trials no dejó detalle por transferencia)"

    columnas = ["Familia", "Brazo", "Transferencia", "Techo", "Criterio",
                "Meseta", "Trials"]

    def celda(f) -> list[str]:
        techo = "—" if f["techo"] is None else f"{f['techo']:g}"
        if f["regla"]:
            techo = f"**{techo}**" if markdown else f"[{techo}]"
        valor = "—" if f["valor"] is None else f"{_scaled(f['valor'], 'targetAccuracy'):.1f}"
        # El ancho de la meseta y no un si/no: «1» dice que el criterio decidio,
        # y «9» dice cuanto no distinguio, que es informacion que un booleano
        # tira.
        return [f"`{f['familia']}`", f"`{f['brazo']}`", f["transferencia"],
                techo, valor, str(f["meseta"]), str(f["trials"] or "—")]

    if markdown:
        lineas = ["| " + " | ".join(columnas) + " |",
                  "|" + "|".join(["---"] * len(columnas)) + "|"]
        lineas += ["| " + " | ".join(celda(f)) + " |" for f in filas]
        lineas += ["", "En negrita el techo que puso la regla de meseta y no el "
                       "criterio. **Meseta** es cuántos techos el ruido estimado no "
                       "distinguió del mejor: uno significa que el criterio decidió."]
        return "\n".join(lineas)

    anchos = [max(len(c), *(len(celda(f)[i]) for f in filas)) + 2
              for i, c in enumerate(columnas)]
    out = ["".join(c.ljust(w) for c, w in zip(columnas, anchos))]
    out += ["".join(v.ljust(w) for v, w in zip(celda(f), anchos)) for f in filas]
    return "\n".join(out)


def conclusion_ceilings(record: dict | None) -> str:
    """Si el criterio eligió el techo o lo eligió la regla.

    Vale para las dos formas de registro y las dice distinto a propósito. Una
    rejilla repite semillas y puede informar si coincidieron; una búsqueda por
    trials no repite nada y en su lugar informa el ruido que el GP estimó, que es
    lo que hace su papel. Normalizar los dos a un vocabulario común haría que un
    lector comparara tres repeticiones con treinta evaluaciones de puntos
    distintos como si fueran la misma cantidad de evidencia.

    Calculada y no escrita: es la lectura que decide si el escalar que va a gobernar
    toda la campaña se apoya en una medición o en un criterio de escritorio, y esa
    distinción sobrevive a que los números cambien solo si se recalcula con ellos.

    Lo que el registro no dice se informa como no dicho. Un registro escrito por una
    versión anterior de la búsqueda no trae cómo se desempató ni si las semillas
    coincidieron, y suponer que no hubo empate sería afirmar «elegido por una
    diferencia real» sobre algo que nunca lo dijo: una explicación inventada para
    tapar un hueco se lee como hallazgo y se usa como tal.
    """
    if not record:
        return "Sin búsqueda: ningún techo está elegido y nada de abajo puede correr."
    partes, por_regla = [], []
    for familia, entrada in sorted(record.items()):
        elec = ceiling_record.choice_of(entrada)
        rejilla = elec["kind"] == ceiling_record.KIND_GRID
        if rejilla and "decidedByTieBreak" not in entrada:
            como = "sin que el registro diga cómo se desempató"
        elif elec["byRule"]:
            como = (f"por desempate entre {elec['amongst']} techos empatados" if rejilla
                    else f"por la regla de meseta, entre {elec['amongst']} techos que "
                         f"el ruido estimado no distingue")
        else:
            como = "por una diferencia en el criterio"
        if rejilla:
            evidencia = (f"{elec['count']} repetición(es) de "
                         f"{entrada.get('epochs', '?')} épocas")
            acuerdo = {True: " y las semillas coinciden",
                       False: " y las semillas **no** coinciden entre sí",
                       None: " y el registro no dice si las semillas coincidieron",
                       }[elec["agreement"]]
        else:
            evidencia = (f"{elec['count']} trial(s) de "
                         f"{entrada.get('epochs', '?')} épocas")
            # Nunca «las semillas coinciden»: no hay semillas. Lo que ocupa ese
            # lugar es la resolución del criterio, y hay que nombrarla por lo que
            # es. Esta línea decía «el GP estimó un ruido de …» — falso: el ancho
            # de la meseta sale del instrumento y no del modelo, que fue la
            # decisión explícita de no hacerlo depender de qué tan bien ajustó.
            # Una frase que sobrevive al cambio de su propio mecanismo describe
            # el que ya no está.
            acuerdo = ("" if elec["noise"] is None
                       else f" y la resolución del criterio es {elec['noise']:.4g}")
        escala = ("" if entrada.get("atRequiredScale")
                  else ", **por debajo de la escala que su respuesta necesita**")
        neutro = (" — que es el neutro, así que la normalización queda confirmada "
                  "por medición" if entrada["ceiling"] == entrada.get("neutral") else "")
        partes.append(
            f"**{familia}** se queda en {entrada['ceiling']:g}{neutro}, elegido {como} "
            f"sobre el rol `{entrada.get('role', '?')}` con "
            f"{evidencia}{acuerdo}{escala}.")
        if elec["byRule"]:
            por_regla.append((familia, rejilla))
    if por_regla:
        nombres = ", ".join(sorted(f for f, _ in por_regla))
        cual = ("la rejilla no se inclinó" if all(r for _, r in por_regla)
                else "la búsqueda no distinguió")
        partes.append(
            f"Para {nombres} {cual}: el techo lo puso la regla y no el criterio, "
            f"así que sostiene menos de lo que un número elegido parece sostener.")
    return " ".join(partes)


def render_ceilings_by_transfer(record: dict | None, transfers: list[str],
                                markdown: bool = False) -> str:
    """Qué techo rige en cada transferencia, una fila por familia.

    Los brazos no entran: dentro de una familia todos heredan el mismo techo en
    la misma transferencia, así que una fila por brazo repetiría seis veces la
    misma medición y daría a entender que son seis decisiones.

    La marca distingue las dos lecturas. Una transferencia que la búsqueda midió
    corre a su propio ganador; una que nunca vio corre al ganador agrupado, fuera
    de muestra. Es la diferencia entre un número elegido mirando esa
    transferencia y uno heredado de otras dos, y sin la marca las seis celdas se
    leen como si todas se hubieran medido.
    """
    if not record:
        return ("Sin búsqueda de techos: ninguna transferencia tiene techo elegido.")
    columns = ["Familia", *transfers]

    def cells(entrada: dict) -> list[str]:
        medidos = entrada.get("byTransfer") or {}
        out = []
        for label in transfers:
            valor = medidos.get(label, entrada["ceiling"])
            texto = f"{valor:g}"
            if label in medidos:
                texto = f"**{texto}**" if markdown else f"[{texto}]"
            out.append(texto)
        return out

    if markdown:
        head = ["| " + " | ".join(columns) + " |",
                "|" + "|".join(["---"] * len(columns)) + "|"]
        for familia, entrada in sorted(record.items()):
            head.append("| " + " | ".join([f"`{familia}`", *cells(entrada)]) + " |")
        return "\n".join(head)

    width = max(14, max((len(f) for f in record), default=14) + 2)
    out = [f"{'Familia':<{width}}" + "".join(f"{c:>10}" for c in transfers)]
    for familia, entrada in sorted(record.items()):
        out.append(f"{familia:<{width}}"
                   + "".join(f"{c:>10}" for c in cells(entrada)))
    return "\n".join(out)


def conclusion_ceilings_by_transfer(record: dict | None,
                                    transfers: list[str]) -> str:
    """La regla que reparte los techos, y qué transferencias se apartan.

    La regla se escribe porque es una decisión y nadie puede deducirla de la
    tabla. Lo que la acompaña se calcula: cuántas transferencias se midieron,
    cuántas heredan, y cuáles de las medidas eligieron algo distinto del ganador
    agrupado. Esa última es la que importa — si ninguna se aparta, separar las
    dos lecturas no cambió un solo número y decirlo es más honesto que dejar la
    marca sugiriendo que sí.
    """
    if not record:
        return "Sin búsqueda: no hay regla que aplicar todavía."
    regla = ("En las transferencias que la búsqueda midió rige el ganador de esa "
             "transferencia, por la misma lectura apareada y el mismo desempate. "
             "En las restantes rige el ganador de las medidas tomadas juntas: es "
             "una aplicación fuera de muestra y se declara como tal, porque ese "
             "escalar no se eligió mirándolas.")
    partes = [regla]
    for familia, entrada in sorted(record.items()):
        medidos = entrada.get("byTransfer") or {}
        if not medidos:
            partes.append(
                f"**{familia}**: el registro no trae elecciones por transferencia, "
                f"así que las {len(transfers)} corren al ganador agrupado "
                f"({entrada['ceiling']:g}).")
            continue
        aparte = sorted(l for l, v in medidos.items() if v != entrada["ceiling"])
        heredan = len([l for l in transfers if l not in medidos])
        if aparte:
            detalle = ", ".join(f"`{l}` a {medidos[l]:g}" for l in aparte)
            partes.append(
                f"**{familia}**: {len(medidos)} medida(s), {heredan} heredada(s) a "
                f"{entrada['ceiling']:g}, y {len(aparte)} de las medidas elige otro "
                f"techo — {detalle}. Ahí la familia deja de correr a un coeficiente "
                f"único, así que su promedio entre transferencias mezcla dos "
                f"escalares; dentro de cada transferencia todos los brazos siguen "
                f"compartiendo el techo, que es lo que mantiene atribuible cada "
                f"peldaño.")
        else:
            partes.append(
                f"**{familia}**: {len(medidos)} medida(s) y {heredan} heredada(s), "
                f"todas en {entrada['ceiling']:g}. Ninguna transferencia medida se "
                f"aparta del ganador agrupado, así que separar las dos lecturas no "
                f"cambió ningún techo de esta familia.")
    return " ".join(partes)


def render(runs: Iterable[dict], metric: str, reduction: dict,
           markdown: bool = False) -> str:
    """Una fila por método: sus seis transferencias y, al final, el promedio.

    Nada de máximos ni de participaciones. El máximo de N repeticiones crece con
    la dispersión del propio método, así que impreso al lado de una media favorece
    al brazo más ruidoso — es la misma razón por la que se guarda el artefacto
    mediano y nunca el mejor. La participación del término en el objetivo tiene su
    propia figura, que es donde se lee como trayectoria y no como un número suelto.

    El promedio va último porque es un resumen: se lee después de aquello que
    resume, no antes.
    """
    rows = table(runs, metric)
    labels = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS]
    title, unit = SPANISH.get(metric, (metric, ""))
    seeds = reduction.get("seeds", config.SEEDS)
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else int(seeds or 0)

    # Sin encabezado propio: qué mide esta tabla lo dice el párrafo de arriba, y
    # bajo qué límites corrió lo dice el sello, una vez, antes de todo.

    decimals = 1 if metric in PERCENT else 2

    def cell(entry) -> str:
        if entry is None:
            return "—"
        return (f"{_scaled(entry['mean'], metric):.{decimals}f} ± "
                f"{_scaled(entry['stdev'], metric):.{decimals}f}")

    columns = ["Método"] + labels + ["Prom."]
    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            values = [cell(row["byTransfer"][label]) for label in labels]
            lines.append("| " + " | ".join(
                [f"`{row['name']}`", *values,
                 f"**{_scaled(row['avg'], metric):.{decimals}f}**"]) + " |")
        return "\n".join(lines)

    width = max(14, max((len(r["name"]) for r in rows), default=14) + 2)
    lines = [f"{'Método':<{width}}"
             + "".join(f"{label:>16}" for label in labels) + f"{'Prom.':>10}"]
    for row in rows:
        values = "".join(f"{cell(row['byTransfer'][label]):>16}" for label in labels)
        lines.append(f"{row['name']:<{width}}{values}"
                     + f"{_scaled(row['avg'], metric):>10.{decimals}f}")
    return "\n".join(lines)


def conclusion(runs: Iterable[dict], metric: str, reduction: dict) -> str:
    """La lectura más relevante de la tabla, calculada y no escrita a mano.

    Una conclusión escrita a mano es una segunda fuente de verdad: se desactualiza
    en silencio y se le cree igual. Esta se recalcula con la tabla, así que no
    puede alejarse de lo que describe.
    """
    rows = table(runs, metric)
    if not rows:
        return "Sin corridas: no hay nada que concluir."
    better = config.DIMENSIONS.get(metric, config.HIGHER)
    reverse = better == config.HIGHER
    ordered = sorted(rows, key=lambda r: r["avg"], reverse=reverse)
    best, worst = ordered[0], ordered[-1]
    title, unit = SPANISH.get(metric, (metric, ""))

    def show(value: float) -> str:
        return (f"{_scaled(value, metric):.1f}{unit}" if metric in PERCENT
                else f"{value:.2f}{unit}")

    def vanishes(value: float) -> bool:
        """Si la distancia se redondea a cero, decirlo en palabras y no en cifra.

        `0.0` no es una distancia, es la ausencia de una, y escribirlo como número
        finge una precisión que el redondeo ya se comió. Además es un número de la
        tabla: imprimirlo la repite sin agregar nada.

        Se lee del número que `show` imprime y no de su cadena. Recortar caracteres
        funcionaba para los porcentajes y fallaba en silencio para cualquier
        métrica sin `%` — `0.00s` no se reconocía —, que es la forma exacta de
        error que este archivo existe para no cometer.
        """
        printed = re.search(r"-?\d+(?:\.\d+)?", show(abs(value)))
        return printed is not None and float(printed.group()) == 0.0

    by_name = {row["name"]: row for row in rows}
    # Los extremos por nombre y la distancia entre ellos. Imprimir los dos valores
    # sería reponer el máximo y el mínimo de la tabla, que ya están ahí a la vista;
    # lo que la tabla no dice es cuánto los separa, porque exige restar.
    spread = abs(best["avg"] - worst["avg"])
    lines = [f"Mejor promedio: **{best['name']}**; peor: {worst['name']}, "
             + ("sin distancia apreciable entre ellos." if vanishes(spread)
                else f"a {show(spread)} de distancia.")]

    # Cada método completo contra su propio piso, que es la única lectura que
    # separa lo que aporta la adaptación de lo que aporta la representación.
    for arm, floor in (("D", "A"), ("G", "B")):
        pair = (config.NAME_OF[arm], config.NAME_OF[floor])
        if pair[0] in by_name and pair[1] in by_name:
            delta = by_name[pair[0]]["avg"] - by_name[pair[1]]["avg"]
            if vanishes(delta):
                lines.append(f"{pair[0]} no se separa de su piso {pair[1]}.")
                continue
            direction = "por encima de" if (delta > 0) == reverse else "por debajo de"
            lines.append(f"{pair[0]} queda {show(abs(delta))} {direction} su piso "
                         f"{pair[1]}.")

    if _stamp(reduction):
        lines.append(f"Con {_repetitions(reduction)} repetición(es) esto es una "
                     f"estimación puntual y no un veredicto: la dispersión es cero "
                     f"por construcción, no por acuerdo. Más repeticiones lo "
                     f"refuerzan o lo cambian.")
    return " ".join(lines)


# ----------------------------------------------------------------- los peldaños

def rung_name(left: str, right: str) -> str:
    """`Baseline → MIL-Baseline`, no `A->B`. Un identificador no es un nombre."""
    return f"{config.NAME_OF.get(left, left)} → {config.NAME_OF.get(right, right)}"


def render_rungs(summary: dict, metric: str, markdown: bool = False) -> str:
    """Un peldaño por fila, una transferencia por columna: la diferencia medida.

    Es la misma forma que la tabla de niveles, con el mismo orden de lectura. La
    diferencia se resta en el orden en que el peldaño se lee — izquierda menos
    derecha — así que un valor **negativo** significa que el de la derecha quedó
    por encima. El nombre del peldaño dice `izquierda → derecha`, y restar al revés
    obligaba a invertir mentalmente cada celda contra el título de su propia fila.
    """
    grid = summary["grid"]
    labels = [t for t in [f"{s}->{d}" for s, d in config.VERDICT_TRANSFERS] if t in grid]
    title, unit = SPANISH.get(metric, (metric, ""))
    rows = []
    for left, right, reading in config.LADDER:
        values = {}
        for label in labels:
            cell = grid[label]
            if left in cell and right in cell:
                values[label] = (cell[left][metric]["mean"] - cell[right][metric]["mean"])
        if not values:
            continue
        rows.append({"name": rung_name(left, right), "reading": reading,
                     "values": values,
                     "avg": sum(values.values()) / len(values),
                     # Las veces que ganó la izquierda, que es el lado positivo
                     # de la misma resta que imprime la fila. Contar el otro lado
                     # obligaría a invertir el signo contra el título de la columna.
                     "leans": sum(1 for v in values.values() if v > 0)})
    if not rows:
        return "(sin peldaños medibles)"

    def show(value) -> str:
        return "—" if value is None else f"{_scaled(value, metric):+.1f}"

    # `a favor` se queda y no es un extra: cuenta en cuántas transferencias se
    # inclinó para el mismo lado, y con este piso de repeticiones es lo único que
    # carga peso. Seis acuerdos y tres contra tres promedian parecido y dicen
    # cosas opuestas; sin esta columna el promedio no se puede leer.
    columns = ["Peldaño"] + labels + ["Prom.", "gana izq."]
    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(
                [row["name"], *(show(row["values"].get(l)) for l in labels),
                 f"**{show(row['avg'])}**",
                 f"{row['leans']}/{len(row['values'])}"]) + " |")
        return "\n".join(lines)

    width = max(len(r["name"]) for r in rows) + 2
    lines = [f"{'Peldaño':<{width}}" + "".join(f"{l:>10}" for l in labels)
             + f"{'Prom.':>10}{'gana izq.':>11}"]
    for row in rows:
        lines.append(f"{row['name']:<{width}}"
                     + "".join(f"{show(row['values'].get(l)):>10}" for l in labels)
                     + f"{show(row['avg']):>10}"
                     + f"{row['leans']}/{len(row['values']):<8}")
        lines.append(f"{'':<{width}}    {row['reading']}")
    return "\n".join(lines)


#: Prepended to every `render_per_run` table, in both formats — the one place a
#: reader who skips straight to the numbers still meets the reason there is no
#: `±` column here, before they can mistake a per-machine row for a pooled one.
_PER_RUN_NOTE = ("cada fila es una corrida en su propia máquina, nunca "
                  "promediada con otra: no hay una columna de método que hable "
                  "por todas.")


def render_per_run(grid_per_run: dict, metric: str, markdown: bool = False) -> str:
    """Every `perRun` reading, one row per run, tagged with its own environment.

    A `perRun` dimension has no mean this report is willing to print: the
    replication that created the category (see `shards.merge`) found no value
    of `seconds`/`peakMiB` stable enough to stand for the method, or even for
    one fixed machine across two of its own runs. Pooling `n` readings into a
    single `mean ± stdev`, the way `render` does for a `poolable` dimension,
    would print a number that describes none of the runs behind it and reads
    exactly as rigorous as one that does — the dispersion column makes it look
    more trustworthy, not less misleading.

    So every row here names the run that produced it instead: its own
    environment and seed. There is no aggregate row and no `±` anywhere in the
    output — a reader who wants a summary has to build one themselves, in full
    view of how many machines and readings it would be standing in for.
    """
    title, unit = SPANISH.get(metric, (metric, ""))
    decimals = 1 if metric in PERCENT else 2
    rows = []
    for transfer, cell in grid_per_run.items():
        for arm in config.ARM_ORDER:
            readings = cell.get(arm, {}).get(metric)
            if not readings:
                continue
            for reading in sorted(readings, key=lambda r: (r["env"], r["seed"])):
                rows.append({
                    "arm": config.NAME_OF.get(arm, arm),
                    "transfer": transfer,
                    "env": reading["env"],
                    "seed": reading["seed"],
                    "value": _scaled(reading["value"], metric),
                })
    if not rows:
        return "(sin corridas medidas)"

    note = _notes_block([_PER_RUN_NOTE], markdown)
    columns = ["Método", "Transferencia", "Entorno", "Semilla", f"{title} ({unit})" if unit else title]
    if markdown:
        lines = [note, "",
                 "| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join([
                f"`{row['arm']}`", row["transfer"], f"`{row['env']}`",
                str(row["seed"]), f"{row['value']:.{decimals}f}"]) + " |")
        return "\n".join(lines)

    width = max(14, max(len(r["arm"]) for r in rows) + 2)
    lines = [note, "",
             f"{'Método':<{width}}{'Transferencia':<16}{'Entorno':<16}"
             f"{'Semilla':>8}{columns[-1]:>16}"]
    for row in rows:
        lines.append(f"{row['arm']:<{width}}{row['transfer']:<16}{row['env']:<16}"
                     f"{row['seed']:>8}{row['value']:>16.{decimals}f}")
    return "\n".join(lines)


def conclusion_per_run(metric: str) -> str:
    """Why this section prints no best/worst, unlike `conclusion`.

    `render_per_run` already refuses to average; a conclusion that then
    printed 'best average / worst average' over the same readings would take
    back with prose exactly what the table just refused to claim with
    numbers. There is nothing this function computes — the readings above are
    the entire finding.
    """
    title, unit = SPANISH.get(metric, (metric, ""))
    return (f"Sin conclusión: {title} no se promedia entre máquinas — cada "
            f"corrida es la lectura de su propio entorno, no una propiedad del "
            f"método ni de la máquina que la corrió. Ver la tabla de arriba, "
            f"corrida por corrida.")


def conclusion_rungs(summary: dict, metric: str) -> str:
    """Qué peldaño se movió más y cuál coincidió en todas las transferencias."""
    grid = summary["grid"]
    labels = [t for t in [f"{s}->{d}" for s, d in config.VERDICT_TRANSFERS] if t in grid]
    readings = []
    for left, right, reading in config.LADDER:
        values = [grid[l][left][metric]["mean"] - grid[l][right][metric]["mean"]
                  for l in labels if left in grid[l] and right in grid[l]]
        if values:
            readings.append({"name": rung_name(left, right), "reading": reading,
                             "left": config.NAME_OF.get(left, left),
                             "right": config.NAME_OF.get(right, right),
                             "avg": sum(values) / len(values),
                             # Idem: las veces que ganó la izquierda.
                             "leans": sum(1 for v in values if v > 0), "n": len(values)})
    if not readings:
        return "Sin peldaños medibles."

    strongest = max(readings, key=lambda r: abs(r["avg"]))
    unanimous = [r for r in readings if r["leans"] in (0, r["n"])]
    # Quién quedó adelante, no cuánto se movió la resta. Un peldaño que "se movió
    # 3.4" no dice de qué lado, y el lector termina reconstruyendo el signo contra
    # el nombre de la fila — que es exactamente el trabajo que la conclusión existe
    # para ahorrarle.
    ahead = strongest["right"] if strongest["avg"] < 0 else strongest["left"]
    behind = strongest["left"] if strongest["avg"] < 0 else strongest["right"]
    lines = [f"El peldaño que más separa es **{strongest['name']}**: "
             f"**{ahead}** queda "
             f"{abs(_scaled(strongest['avg'], metric)):.1f} por encima de {behind}, "
             f"y eso lee {strongest['reading']}."]
    if unanimous:
        # Los nombres, no sus valores: la tabla de arriba ya los imprimió, y
        # repetirlos acá los pone en dos lugares que pueden separarse.
        lines.append("Se inclinan igual en las " + str(readings[0]["n"]) +
                     " transferencias: " + ", ".join(r["name"] for r in unanimous) +
                     ".")
    else:
        lines.append("Ningún peldaño se inclina igual en todas las transferencias, "
                     "que es lo que haría falta para leer algo de un solo sentido.")
    if _stamp(summary["reduction"]):
        lines.append("Con esta cantidad de repeticiones lo que carga peso es la "
                     "coincidencia entre transferencias, no la magnitud: seis "
                     "acuerdos y tres contra tres promedian parecido y dicen cosas "
                     "distintas.")
    return " ".join(lines)


# --------------------------------------------------------------- fase dos

def _reach(record: dict, path: str):
    """`geometry.ratio` dentro de una lectura anidada, o nada si el brazo no la tiene."""
    value = record
    for step in path.split("."):
        if not isinstance(value, dict) or step not in value:
            return None
        value = value[step]
    return float(value) if isinstance(value, (int, float)) else None


def render_readings(readings: Iterable[dict], path: str, title: str,
                    markdown: bool = False) -> str:
    """Una medición de fase dos en la misma forma que las tablas de fase uno.

    El `±` de acá **no** es el de las tablas de fase uno: es la dispersión entre
    los checkpoints guardados de esa celda. La fase uno mide la variabilidad de la
    exactitud; esto mide la variabilidad de la geometría, y nada establece que una
    siga a la otra.

    Un brazo sin lectura para esta cantidad — uno sin término local no tiene
    correspondencia — recibe una celda vacía, no un cero.
    """
    readings = list(readings)
    labels = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS]
    gathered: dict[tuple[str, str], list[float]] = {}
    for reading in _own_medians(readings):
        value = _reach(reading, path)
        if value is not None:
            gathered.setdefault((reading["arm"], reading["transfer"]), []).append(value)

    rows = []
    for arm in config.ARM_ORDER:
        by_label = {label: (spread(gathered[(arm, label)]) if (arm, label) in gathered
                            else None) for label in labels}
        present = [c for c in by_label.values() if c]
        if not present:
            continue
        rows.append({"name": config.NAME_OF[arm], "cells": by_label,
                     "avg": sum(c["mean"] for c in present) / len(present),
                     "n": max(c["n"] for c in present)})

    def cell(entry) -> str:
        return "—" if entry is None else f"{entry['mean']:.3f} ± {entry['stdev']:.3f}"

    columns = ["Método"] + labels + ["Prom."]
    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(
                [f"`{row['name']}`", *(cell(row["cells"][l]) for l in labels),
                 f"**{row['avg']:.3f}**"]) + " |")
        return "\n".join(lines)

    width = max(14, max((len(r["name"]) for r in rows), default=14) + 2)
    lines = [f"{'Método':<{width}}" + "".join(f"{l:>18}" for l in labels)
             + f"{'Prom.':>10}"]
    for row in rows:
        lines.append(f"{row['name']:<{width}}"
                     + "".join(f"{cell(row['cells'][l]):>18}" for l in labels)
                     + f"{row['avg']:>10.3f}")
    return "\n".join(lines)


def best_transfers(runs: Iterable[dict], count: int | None = None,
                   metric: str = "targetAccuracy") -> list[str]:
    """Las transferencias donde los métodos llegan más alto, calculadas de la campaña.

    Es una elección hecha por el resultado, y por eso va declarada en cada pie de
    figura. Lo que la hace defendible en esta figura y no en otras: el espacio
    latente de una transferencia donde todos los métodos quedan cerca del azar es
    la foto de un modelo que no aprendió, y de ahí no se lee nada sobre alineación.

    Lo que esta elección nunca toca es **qué sorteo** se muestra. Eso sigue siendo
    la semilla de exhibición, elegida por una regla que no favorece a nadie:
    elegir el sorteo por el resultado es como una figura deja de poder salir mal.
    """
    count = count or config.FIGURE_TRANSFER_COUNT
    grid = cells(runs, metric)
    by_transfer: dict[str, list[float]] = {}
    for (_, transfer), entry in grid.items():
        by_transfer.setdefault(transfer, []).append(entry["mean"])
    ordered = sorted(by_transfer, key=lambda t: sum(by_transfer[t]) / len(by_transfer[t]),
                     reverse=True)
    return ordered[:count]


def ranking(runs: Iterable[dict], metric: str = "targetAccuracy",
            exclude: Iterable[str] = ()) -> list[str]:
    """Los brazos por media de `metric`, el mejor primero."""
    excluded = set(exclude)
    return [row["arm"] for row in
            sorted(table(runs, metric), key=lambda r: r["avg"], reverse=True)
            if row["arm"] not in excluded]


# ------------------------------------------- lo que dice cada figura, calculado

def _paired(by_arm_transfer: dict, left: str, right: str) -> list[tuple[str, float]]:
    """La diferencia entre dos brazos **dentro de cada transferencia**.

    Promediar valores crudos entre transferencias mete la dificultad de cada una
    en la dispersión y ahoga el efecto: una transferencia donde todo el mundo saca
    0,27 y otra donde todo el mundo saca 0,80 no son comparables sumadas. La
    diferencia medida dentro de una misma transferencia cancela esa dificultad, y
    lo que carga peso es que las transferencias **coincidan**, no el promedio.
    """
    shared = sorted(set(by_arm_transfer.get(left, {})) & set(by_arm_transfer.get(right, {})))
    return [(transfer, by_arm_transfer[right][transfer] - by_arm_transfer[left][transfer])
            for transfer in shared]


def _agreement(differences: list[tuple[str, float]], tolerance: float) -> str:
    """Cómo se leen varias diferencias apareadas, sin promediarlas a la nada."""
    if not differences:
        return "sin transferencias en común"
    up = [t for t, d in differences if d > tolerance]
    down = [t for t, d in differences if d < -tolerance]
    flat = [t for t, d in differences if abs(d) <= tolerance]
    detail = ", ".join(f"{t} {d:+.3f}" for t, d in differences)
    if len(up) == len(differences):
        return f"a favor en las {len(differences)} transferencias ({detail})"
    if len(down) == len(differences):
        return f"en contra en las {len(differences)} transferencias ({detail})"
    if len(flat) == len(differences):
        return f"plano en todas, dentro de ±{tolerance:.3f} ({detail})"
    return (f"las transferencias no coinciden — {len(up)} a favor, {len(down)} en "
            f"contra, {len(flat)} planas ({detail}). Un promedio acá diría 'no hace "
            f"nada' y estaría tapando que una transferencia sí se movió")


def paired_gains(runs: Iterable[dict], dimension: str) -> list[dict]:
    """Cada método contra su propio piso, apareado dentro de cada transferencia.

    Apareado y no crudo: una transferencia donde todos sacan 0.23 y otra donde
    todos sacan 0.81 no son comparables sumadas, y la diferencia dentro de cada
    una cancela esa dificultad. Cada par comparte transferencia **y** semilla, o
    sea el mismo sorteo y la misma partición.

    Tres resúmenes, y ninguno alcanza solo:

    * `mean` — puntos porcentuales, con su error **entre transferencias**. No es
      el error de juntar los 180 pares: la transferencia es un escenario y no una
      repetición, y agruparlos afirma una estabilidad entre escenarios que nadie
      midió. Sobre esta campaña los dos difieren por un factor de tres.
    * `pct` — la media de las ganancias relativas. Dice otra cosa y puede decirla
      con el signo contrario: los pisos van de 23% a 81%, así que una pérdida de
      cinco puntos sobre el piso más bajo pesa cuatro veces más como razón que
      como diferencia. Sobre esta campaña la media de puntos da `+0.56` y la de
      porcentajes `-3.61`. Se reportan las dos porque elegir una es elegir una
      respuesta.
    * `span` y el acuerdo — de cuánto a cuánto, y en cuántas transferencias gana,
      pierde o empata. El rango es lo que frena a un lector que solo mira el
      promedio: `de +15.1 a -5.5` se entiende en una lectura y ya avisa que la
      media no es la historia.
    """
    rows = runs if isinstance(runs, list) else list(runs)
    idx = {(r["arm"], r["transfer"], r["seed"]): r for r in rows}
    labels = [f"{a}->{b}" for a, b in config.VERDICT_TRANSFERS]

    out = []
    for arm, floor in config.FLOOR_OF.items():
        per: dict[str, dict] = {}
        for label in labels:
            deltas, floors = [], []
            for (a, t, seed), row in idx.items():
                if a != arm or t != label:
                    continue
                base = idx.get((floor, t, seed))
                if base is None or dimension not in row or dimension not in base:
                    continue
                deltas.append(100.0 * (float(row[dimension]) - float(base[dimension])))
                floors.append(100.0 * float(base[dimension]))
            if not deltas:
                continue
            mean, err = _spread_of(deltas)
            base_mean = sum(floors) / len(floors)
            per[label] = {"mean": mean, "error": err, "n": len(deltas),
                          "pct": (100.0 * mean / base_mean) if base_mean else None,
                          "verdict": ("gana" if mean > 2 * err else
                                      "pierde" if mean < -2 * err else "empata")}
        if not per:
            continue
        means = [c["mean"] for c in per.values()]
        pcts = [c["pct"] for c in per.values() if c["pct"] is not None]
        counted = collections.Counter(c["verdict"] for c in per.values())
        out.append({
            "arm": arm, "floor": floor, "cells": per,
            "mean": sum(means) / len(means),
            # Entre transferencias, nunca sobre los pares agrupados.
            "error": _spread_of(means)[1],
            "pct": (sum(pcts) / len(pcts)) if pcts else None,
            "span": (max(means), min(means)),
            "agreement": {k: counted.get(k, 0) for k in ("gana", "pierde", "empata")},
            "transfers": len(per),
        })
    return out


def _spread_of(values: list[float]) -> tuple[float, float]:
    """Media y error estándar de la media. Con un solo valor el error es cero y
    se devuelve como tal: es un hecho sobre la muestra, no un veredicto."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var) / math.sqrt(n)


def render_gains(runs: Iterable[dict], dimension: str, title: str,
                 markdown: bool = False) -> str:
    """La tabla de ganancias: cada método menos su propio piso.

    Va después de la tabla de exactitud y no en lugar de ella. Aquella dice
    dónde está cada método; esta dice qué agregó el término, que es la pregunta
    que la primera no puede contestar porque los pisos de las dos familias están
    separados por diecisiete puntos.
    """
    rows = paired_gains(runs, dimension)
    if not rows:
        return f"(sin pares para {dimension})"
    labels = [f"{a}->{b}" for a, b in config.VERDICT_TRANSFERS]

    def cell(entry) -> str:
        if entry is None:
            return "—"
        body = f"{entry['mean']:+.2f} ± {entry['error']:.2f}"
        if entry["verdict"] != "empata":
            body = f"**{body}**" if markdown else body
        pct = "" if entry["pct"] is None else f"{entry['pct']:+.1f}%"
        return f"{body}<br>{pct}" if markdown else f"{body}"

    def summary(row) -> tuple[str, str, str]:
        pct = "—" if row["pct"] is None else f"{row['pct']:+.2f}%"
        hi, lo = row["span"]
        # La misma regla que en las celdas. Poner la media siempre en negrita
        # contradice la leyenda que las explica: el lector la aplica y lee como
        # significativo un promedio que no lo es.
        body = f"{row['mean']:+.2f} ± {row['error']:.2f}"
        if markdown and abs(row["mean"]) > 2 * row["error"]:
            body = f"**{body}**"
        return (body, pct, f"de {hi:+.1f} a {lo:+.1f}")

    def words(row) -> str:
        ag, n = row["agreement"], row["transfers"]
        return (f"gana en {ag['gana']} de {n}, pierde en {ag['pierde']}, "
                f"empata en {ag['empata']}")

    ordered = sorted(rows, key=lambda r: config.ARM_ORDER.index(r["arm"]))
    if markdown:
        columns = ["Método vs su piso"] + labels + ["Media (pts)", "% medio",
                                                    "Rango", "Acuerdo"]
        lines = [f"**{title}**", "",
                 "| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in ordered:
            m, pct, span = summary(row)
            lines.append("| " + " | ".join([
                f"`{config.NAME_OF[row['arm']]}` vs `{config.NAME_OF[row['floor']]}`",
                *(cell(row["cells"].get(l)) for l in labels),
                m, pct, span, words(row)]) + " |")
        lines += ["", "En negrita lo que supera dos errores estándar. El error de "
                      "la media es **entre transferencias**, no sobre los pares "
                      "agrupados: la transferencia es un escenario y no una "
                      "repetición. Los dos promedios pueden discrepar en el signo "
                      "— el de puntos y el de porcentajes miden cosas distintas — "
                      "y por eso van los dos."]
        return "\n".join(lines)

    width = max(len(f"{config.NAME_OF[r['arm']]} vs {config.NAME_OF[r['floor']]}")
                for r in ordered) + 2
    lines = [title, "",
             f"{'Método vs su piso':<{width}}" + "".join(f"{l:>16}" for l in labels)
             + f"{'Media':>16}{'% medio':>10}{'Rango':>20}  Acuerdo"]
    for row in ordered:
        m, pct, span = summary(row)
        name = f"{config.NAME_OF[row['arm']]} vs {config.NAME_OF[row['floor']]}"
        lines.append(f"{name:<{width}}"
                     + "".join(f"{cell(row['cells'].get(l)):>16}" for l in labels)
                     + f"{m:>16}{pct:>10}{span:>20}  {words(row)}")
    return "\n".join(lines)


def conclusion_geometry(readings: Iterable[dict]) -> str:
    """Qué dice la grilla, sacado de las mediciones y no de mirarla.

    Una línea por método contra su propio piso, y las diferencias son apareadas
    dentro de cada transferencia: una transferencia donde todos sacan 0,27 y otra
    donde todos sacan 0,80 no son comparables sumadas.

    Alineación y colapso se separan acá y no en la razón sola. La razón es
    `cruzada / entre clases`, así que baja tanto si los dominios se juntaron como
    si el espacio entero se encogió. Se llama colapso cuando la distancia entre
    clases cayó fuerte **y** la razón no mejoró: ahí lo que se juntó fue todo.
    """
    readings = _own_medians(readings)
    ratio: dict[str, dict[str, float]] = {}
    apart: dict[str, dict[str, float]] = {}
    for reading in readings:
        geometry = reading.get("geometry", {})
        if "ratio" in geometry:
            ratio.setdefault(reading["arm"], {})[reading["transfer"]] = float(geometry["ratio"])
        if "betweenClasses" in geometry:
            apart.setdefault(reading["arm"], {})[reading["transfer"]] = float(
                geometry["betweenClasses"])
    if not ratio:
        return "Sin lecturas de geometría: no hay nada que concluir."

    lines = ["Cada método contra su propio piso, transferencia por transferencia. "
             "«Alinea» = la razón bajó; «colapsa» = la razón no bajó y la distancia "
             "entre clases cayó más del 10%.",
             "",
             f"{'Método':<14}{'piso':<14}{'alinea':>8}{'plano':>7}{'empeora':>9}"
             f"{'colapsa':>9}"]

    for arm, floor in config.FLOOR_OF.items():
        if arm not in ratio or floor not in ratio:
            continue
        aligns = flat = worse = collapses = 0
        for transfer, change in _paired(ratio, floor, arm):
            shrink = apart.get(arm, {}).get(transfer, 0.0)
            base = apart.get(floor, {}).get(transfer, 0.0)
            shrank = bool(base) and shrink < base * 0.9
            if change < -0.02:
                aligns += 1
            elif change > 0.02:
                worse += 1
                collapses += shrank
            else:
                flat += 1
                collapses += shrank
        lines.append(f"{config.NAME_OF[arm]:<14}{config.NAME_OF[floor]:<14}"
                     f"{aligns:>8}{flat:>7}{worse:>9}{collapses:>9}")

    lines.append("")
    lines.append("Lo que carga peso no es el promedio sino que las transferencias "
                 "coincidan: un método que alinea en una y empeora en otra no está "
                 "diciendo nada todavía.")
    _pilot_note(lines, readings)
    return "\n".join(lines)


def _by_arm(readings: Iterable[dict], path: str) -> dict[str, dict[str, float]]:
    """{brazo: {transferencia: valor}} promediando los checkpoints de cada celda.

    Solo la mediana propia de cada celda. Los extras que una promoción apareada
    le agrega a un piso los eligió el orden de accuracy de otro brazo, así que
    promediar la fila del piso sobre ellos no la estima mejor: estima otra cosa.
    """
    gathered: dict[tuple[str, str], list[float]] = {}
    for reading in _own_medians(readings):
        value = _reach(reading, path)
        if value is not None:
            gathered.setdefault((reading["arm"], reading["transfer"]), []).append(value)
    by_arm: dict[str, dict[str, float]] = {}
    for (arm, transfer), values in gathered.items():
        by_arm.setdefault(arm, {})[transfer] = sum(values) / len(values)
    return by_arm


def _own_medians(readings: Iterable[dict]) -> list[dict]:
    """Solo los checkpoints que son la mediana de su propia celda.

    Una promoción apareada agrega a cada piso las semillas que eligieron los
    brazos que dependen de él, para que la diferencia apareada exista. Esas
    semillas las eligió el orden de accuracy de **otro** brazo, así que no son una
    muestra imparcial de las corridas del piso: promediar la fila del piso sobre
    ellas no la estima mejor, estima otra cosa.

    Un checkpoint sin la marca se conserva. Una corrida de una sola máquina no
    tiene registro de promoción, y ahí todo lo que hay en disco es la mediana de
    su celda porque `keep_median()` es lo único que lo escribió.
    """
    return [r for r in readings if r.get("median", True)]


def _repetitions(readings: Iterable[dict]) -> int:
    """Cuántas repeticiones sostiene de verdad la tabla que se está sellando.

    Contadas de los datos, nunca de la configuración. `len(config.SEEDS)` es lo
    que se pidió; esto es lo que llegó, que es la única cifra que un sello puede
    afirmar. Cuando las celdas no coinciden se reporta el **mínimo**, porque la
    celda más flaca es la que acota la afirmación.
    """
    per_cell: dict[tuple, set] = {}
    for reading in readings:
        per_cell.setdefault((reading["arm"], reading["transfer"]), set()).add(
            reading["seed"])
    return min((len(seeds) for seeds in per_cell.values()), default=0)


def _pilot_note(lines: list[str], readings: Iterable[dict] | None = None,
                repetitions: int | None = None) -> list[str]:
    """El sello, cuando la corrida no alcanza la escala que el protocolo declara.

    `readings` o `repetitions`, y ninguno de los dos es opcional en la práctica:
    llamarlo sin datos no puede contar nada, así que no sella en vez de sellar un
    número inventado.
    """
    if repetitions is None:
        if readings is None:
            # No poder contar no es haber contado bien. Un sello ausente se lee
            # como una corrida a escala completa, que es lo contrario del hecho.
            lines.append("Escala no verificable: esta conclusión no recibió las "
                         "mediciones con las que contar repeticiones.")
            return lines
        repetitions = _repetitions(_own_medians(readings))
    if repetitions and repetitions < len(config.FULL_SEEDS):
        lines.append(f"Piloto de {repetitions} repetición(es): estimación "
                     f"puntual, todavía no un veredicto.")
    return lines


def conclusion_distances(readings: Iterable[dict]) -> str:
    """Cuál de las dos distancias hizo el trabajo, que la razón sola no puede decir.

    La razón es `cruzada / entre clases`, así que baja por dos motivos opuestos:
    porque la misma clase se juntó entre dominios, o porque el espacio entero se
    encogió y arrastró el denominador. Acá se separan, y por eso esta conclusión
    no repite la de la razón: aquella cuenta cuántas transferencias mejoraron,
    esta dice de dónde salió la mejora.
    """
    readings = list(readings)
    cross = _by_arm(readings, "geometry.crossDomainSameClass")
    apart = _by_arm(readings, "geometry.betweenClasses")
    if not cross:
        return "Sin lecturas de distancias: no hay nada que concluir."

    lines = []
    for arm, floor in config.FLOOR_OF.items():
        if arm not in cross or floor not in cross:
            continue
        shared = sorted(set(cross[arm]) & set(cross[floor]) & set(apart.get(arm, {}))
                        & set(apart.get(floor, {})))
        if not shared:
            continue
        # Relativo y apareado dentro de cada transferencia: una distancia cruda en
        # un embedding no se compara entre modelos, y el cambio porcentual sí.
        near = [(cross[arm][t] - cross[floor][t]) / cross[floor][t] for t in shared
                if cross[floor][t]]
        far = [(apart[arm][t] - apart[floor][t]) / apart[floor][t] for t in shared
               if apart[floor][t]]
        if not near or not far:
            continue
        near_avg = 100 * sum(near) / len(near)
        far_avg = 100 * sum(far) / len(far)
        if near_avg < far_avg - SCALE_TOLERANCE:
            verdict = "la misma clase se juntó más de lo que se encogió el espacio"
        elif abs(near_avg - far_avg) <= SCALE_TOLERANCE:
            verdict = ("las dos se movieron parejo: el espacio cambió de escala y "
                       "la razón no tendría por qué haberse movido")
        else:
            verdict = "las clases se juntaron entre sí más que la misma clase entre dominios"
        lines.append(f"{config.NAME_OF[arm]} contra {config.NAME_OF[floor]}: "
                     f"misma clase entre dominios {near_avg:+.1f}%, clases distintas "
                     f"{far_avg:+.1f}% sobre {len(shared)} transferencia(s) — {verdict}.")

    if not lines:
        return "Ningún método comparte transferencias con su piso: nada que separar."
    return " ".join(_pilot_note(lines, readings))


def conclusion_separability(readings: Iterable[dict]) -> str:
    """Si la representación se volvió más difícil de atribuir a su dominio.

    Lo que se lee no es la exactitud sino su distancia al azar, porque la
    dirección declarada es *hacia el azar* y no *hacia abajo*: un clasificador que
    quedara por debajo del azar estaría igual de lejos de ser invariante.
    """
    readings = list(readings)
    values = _by_arm(readings, "domainSeparability")
    if not values:
        return "Sin lecturas de separabilidad: no hay nada que concluir."

    gap = {arm: {t: abs(v - DOMAIN_CHANCE) for t, v in by_transfer.items()}
           for arm, by_transfer in values.items()}
    lines = []
    for arm, floor in config.FLOOR_OF.items():
        if arm not in gap or floor not in gap:
            continue
        differences = _paired(gap, floor, arm)
        if not differences:
            continue
        lines.append(f"{config.NAME_OF[arm]} contra {config.NAME_OF[floor]}, "
                     f"distancia al azar ({DOMAIN_CHANCE:.3f}): "
                     f"{_agreement(differences, 0.01)}.")
    if not lines:
        return "Ningún método comparte transferencias con su piso: nada que aparear."
    lines.insert(0, "Una diferencia negativa es mejor: la regla de dominio quedó "
                    "más cerca del azar que en el mismo método sin adaptación.")
    return " ".join(_pilot_note(lines, readings))


def conclusion_mass(readings: Iterable[dict]) -> str:
    """La afirmación del término local contra su propio azar, transferencia por
    transferencia.

    El piso no tiene término local, así que acá no hay peldaño posible: lo único
    contra lo que se puede leer esta cantidad es el azar, y el azar sale del
    registro y no de una constante escrita acá.
    """
    readings = list(readings)
    values = _by_arm(readings, "correspondence.massOnTrueClass")
    if not values:
        return "Ningún brazo guardado tiene término local: no hay masa que concluir."

    chances = [_reach(r, "correspondence.chance") for r in readings]
    chance = next((c for c in chances if c is not None), 1.0 / config.CLASSES)
    lines = []
    for arm in config.ARM_ORDER:
        if arm not in values:
            continue
        by_transfer = values[arm]
        above = [t for t, v in by_transfer.items() if v > chance]
        mean = sum(by_transfer.values()) / len(by_transfer)
        lines.append(f"{config.NAME_OF[arm]} supera el azar en {len(above)} de "
                     f"{len(by_transfer)} transferencias")
    ordered = sorted((a for a in values), key=lambda a: -_mean(values[a]))
    lines.insert(0, f"La masa más alta es la de **{config.NAME_OF[ordered[0]]}**, "
                    f"contra un azar de {chance:.3f}.")
    return " ".join(_pilot_note(lines, readings))


def _mean(by_transfer: dict) -> float:
    return sum(by_transfer.values()) / len(by_transfer)


def conclusion_attention(readings: Iterable[dict]) -> str:
    """Si la atención está repartiendo o pesando todo por igual.

    Es descriptiva y no se disputa: no hay un lado que gane. Lo que sí se puede
    afirmar es lo que la sección afirma — un valor pegado al máximo significa que
    la atención se volvió la media uniforme que venía a mejorar.
    """
    readings = list(readings)
    values = _by_arm(readings, "attentionSpread")
    if not values:
        return "Ningún brazo guardado usa atención: no hay dispersión que concluir."

    lines = []
    for arm in config.ARM_ORDER:
        if arm not in values:
            continue
        by_transfer = values[arm]
        mean = sum(by_transfer.values()) / len(by_transfer)
        low, high = min(by_transfer.values()), max(by_transfer.values())
        if mean >= UNIFORM_ATTENTION:
            reading = "está pesando todas las instancias casi por igual, que es la media uniforme"
        elif mean <= 1.0 - UNIFORM_ATTENTION:
            reading = "apoya la bolsa en unas pocas instancias"
        else:
            reading = "reparte de forma desigual sin apoyarse en unas pocas"
        lines.append(f"{config.NAME_OF[arm]} {reading}")
    lines.insert(0, "La entropía está normalizada, así que su máximo es uno y ahí "
                    "la atención no estaría haciendo nada. Cómo quedó cada uno:")
    return " ".join(_pilot_note(lines, readings))


def _computes(arm: str) -> str:
    """Qué computa un método, derivado de lo que declara y no escrito al lado.

    La figura de correspondencia muestra tres columnas que se parecen, y el nombre
    del método no dice cuál de las tres tiene el término local. Escribirlo a mano
    en la tabla sería una segunda fuente de verdad sobre algo que `config.ARMS` ya
    afirma: acá se arma con lo declarado, así que un método que cambie de mecanismo
    cambia también su etiqueta.
    """
    spec = config.ARMS_BY_ID[arm]
    if not spec["adaptation"]:
        return "sin adaptación: solo fuente"
    parts = [f"adaptación {spec['adaptation']}",
             "ponderada" if spec["weighting"] else "sin ponderar",
             "con término local" if spec["local"] else "sin término local"]
    return ", ".join(parts)


def render_correspondence(scored: Iterable[dict], markdown: bool = False) -> str:
    """Los aciertos y la masa de la figura de bolsas, como números.

    Vivían dentro de la figura, como pie de cada panel. Ahí obligaban a leer una
    cifra dentro de un dibujo —que es donde peor se compara— y de paso hacían que
    las tres columnas se vieran igual de afirmativas. La figura queda para lo que
    una figura hace bien, que es mostrar si los sujetos se asocian; los números
    quedan acá, donde se comparan de una fila a otra.
    """
    scored = list(scored)
    if not scored:
        return "(sin paneles medidos)"

    transfers = list(dict.fromkeys(row["transfer"] for row in scored))
    by_arm: dict[str, dict[str, dict]] = {}
    for row in scored:
        by_arm.setdefault(row["arm"], {})[row["transfer"]] = row

    rows = []
    for arm in config.BAG_PANELS:
        if arm not in by_arm:
            continue
        cells_ = by_arm[arm]
        masses = [c["mass"] for c in cells_.values()]
        rows.append({
            "name": config.NAME_OF[arm], "computes": _computes(arm),
            "hits": {t: (f"{cells_[t]['hits']}/{cells_[t]['classes']}"
                         if t in cells_ else "—") for t in transfers},
            "mass": sum(masses) / len(masses),
        })
    if not rows:
        return "(sin paneles medidos)"

    columns = ["Método", "Qué computa"] + transfers + ["Masa prom."]
    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(
                [f"`{row['name']}`", row["computes"],
                 *(row["hits"][t] for t in transfers),
                 f"**{row['mass']:.3f}**"]) + " |")
        return "\n".join(lines)

    width = max(14, max(len(r["name"]) for r in rows) + 2)
    computes = max(len(r["computes"]) for r in rows) + 2
    lines = [f"{'Método':<{width}}{'Qué computa':<{computes}}"
             + "".join(f"{t:>10}" for t in transfers) + f"{'Masa prom.':>12}"]
    for row in rows:
        lines.append(f"{row['name']:<{width}}{row['computes']:<{computes}}"
                     + "".join(f"{row['hits'][t]:>10}" for t in transfers)
                     + f"{row['mass']:>12.3f}")
    return "\n".join(lines)


def conclusion_correspondence(scored: Iterable[dict],
                              readings: Iterable[dict] | None = None) -> str:
    """Qué dice la figura de bolsas, sacado de sus propios aciertos y masas."""
    scored = list(scored)
    if not scored:
        return "Sin paneles medidos: no hay nada que concluir."
    mass: dict[str, dict[str, float]] = {}
    hits: dict[str, dict[str, str]] = {}
    for row in scored:
        mass.setdefault(row["arm"], {})[row["transfer"]] = float(row["mass"])
        hits.setdefault(row["arm"], {})[row["transfer"]] = f"{row['hits']}/{row['classes']}"

    chance = 1.0 / config.CLASSES
    # Quién acertó más, no cuánto acertó cada uno: eso es la tabla de arriba, y
    # decirlo otra vez acá es la misma medición en dos lugares.
    def total(arm: str) -> int:
        return sum(int(hits[arm][t].split("/")[0]) for t in hits[arm])

    present = [a for a in config.BAG_PANELS if a in hits]
    top = max(total(a) for a in present)
    leading = [a for a in present if total(a) == top]
    # Un empate se dice, no se rompe. `max` devuelve el primero de los iguales, así
    # que nombrar al ganador sin mirar esto declara vencedor al que quedó antes en
    # la lista de paneles — un veredicto producido por el orden de una constante.
    if len(leading) == 1:
        lines = [f"Empareja mejor **{config.NAME_OF[leading[0]]}**, sumando sus "
                 f"transferencias; el azar de acertar una clase es {chance:.3f}."]
    else:
        lines = [f"Empatan sumando sus transferencias: "
                 + ", ".join(config.NAME_OF[a] for a in leading)
                 + f". De acá no sale quién empareja mejor; el azar de acertar una "
                   f"clase es {chance:.3f}."]

    # El peldaño: el término local solo se ve si el completo separa del mismo
    # método sin él, y separa *dentro* de cada transferencia.
    with_local = next((a for a in config.BAG_PANELS
                       if a in mass and config.ARMS_BY_ID[a]["local"]), None)
    without = next((a for a in config.BAG_PANELS
                    if a in mass and config.ARMS_BY_ID[a]["adaptation"]
                    and not config.ARMS_BY_ID[a]["local"]), None)
    if with_local and without:
        lines.append(f"El término local, aislado como {config.NAME_OF[with_local]} "
                     f"contra {config.NAME_OF[without]}: "
                     f"{_agreement(_paired(mass, without, with_local), 0.02)}.")
    # `scored` ya está agregado por celda y no lleva semilla, así que la
    # cuenta sale de las mismas mediciones que sellan a todas las demás.
    _pilot_note(lines, readings)
    return " ".join(lines)


def conclusions(record: dict) -> dict:
    """Cada conclusión del informe, a partir de un registro y de nada más.

    Un solo punto de entrada y no una lista de funciones, porque una verificación
    que tiene que adivinar firmas termina informando «no se pudo ejercitar» y eso
    se lee como un aprobado. Acá el destino cablea sus propias conclusiones detrás
    de una llamada, y quien verifica solo tiene que invocarla dos veces: una con el
    registro y otra con sus números permutados. Si el texto sale igual, la
    conclusión no está atada a nada.

    Devuelve solo las que el registro puede alimentar: pedir una conclusión sobre
    algo que no se midió es distinto de una que no cambia, y confundirlas haría que
    una fase todavía no corrida se informe como un defecto.
    """
    produced: dict[str, str] = {}
    runs = record.get("runs")
    reduction = record.get("reduction") or {}
    if isinstance(runs, list) and runs:
        for metric in ("seconds", "sourceAccuracy", "targetAccuracy"):
            produced[f"niveles:{metric}"] = conclusion(runs, metric, reduction)
    # El panorama sigue en el registro y ya no se concluye. Promediar cada peldaño
    # sobre las seis transferencias a la vez respondía una pregunta que las tablas
    # de peldaños de cada dominio ya contestan por separado, y la respondía peor:
    # una media no distingue un peldaño que se inclinó igual en las seis de uno que
    # quedó tres y tres. Los datos quedan para quien quiera rehacer esa lectura.
    readings = record.get("readings")
    if isinstance(readings, list) and readings:
        produced["geometría"] = conclusion_geometry(readings)
        produced["distancias"] = conclusion_distances(readings)
        produced["separabilidad"] = conclusion_separability(readings)
        produced["masa"] = conclusion_mass(readings)
        produced["atención"] = conclusion_attention(readings)
    scored = record.get("correspondence")
    if isinstance(scored, list) and scored:
        produced["correspondencia"] = conclusion_correspondence(scored)
    # La búsqueda del techo vive en su propio registro y entra por la reducción de
    # la campaña, que la copia entera. Se concluye acá y no aparte para que la
    # permutación de números la ejercite como a cualquier otra: es la conclusión
    # sobre la que descansa el escalar de todas las demás.
    searched = reduction.get("ceilingSearch")
    if isinstance(searched, dict) and searched:
        produced["techos"] = conclusion_ceilings(searched)
    return produced
