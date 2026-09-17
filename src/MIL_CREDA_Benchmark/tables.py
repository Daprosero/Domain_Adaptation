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
from MIL_CREDA_Benchmark import pooling as _pooling

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

def _min_reachable_attention_entropy(m: int, gamma: float, tau_att: float) -> float:
    """El piso que la entropía normalizada de Eq. (16) puede alcanzar sobre una
    bolsa de `m` instancias, bajo `gamma` y `tau_att` -- nunca cero, y computado
    acá en vez de escrito a mano.

    r21, en el párrafo que sigue a la Ec. (16) (l.~499): "los logits de la
    Ec. 15 están acotados: dos logits de una misma bolsa difieren a lo sumo
    en 2 + gamma". Minimizar la entropía de un softmax sujeto solo a esa cota
    de rango se alcanza en un vértice de la caja que la cota define -- todas
    las instancias en uno de los dos extremos del logit -- y por simetría el
    valor en un vértice depende solo de cuántas instancias caen en cada
    extremo, así que barrer ese conteo es barrer todos los vértices y no una
    esquina plausible de la caja.
    """
    if m < 2:
        return 1.0
    half_spread = (2.0 + gamma) / 2.0
    hi = math.exp(half_spread / tau_att)
    lo = math.exp(-half_spread / tau_att)
    best = 1.0
    for k in range(1, m):
        denom = k * hi + (m - k) * lo
        w_hi, w_lo = hi / denom, lo / denom
        entropy = -(k * w_hi * math.log(w_hi) + (m - k) * w_lo * math.log(w_lo))
        best = min(best, entropy / math.log(m))
    return best


def _bag_size_for(arm: str) -> int:
    """Cuántas instancias `weights_for` reparte para este brazo -- `len(beta)`
    en `latent.attention_spread` -- nunca una sola cuenta para todos. Un brazo
    que selecciona reparte sobre su propio presupuesto (`spec["budget"]`, si la
    declaración lo lleva), no sobre la bolsa entera: el piso alcanzable de su
    entropía es OTRO número, más chico, porque hay menos instancias entre las
    que repartir. Ningún nombre de brazo se supone acá: se lee de
    `config.ARMS_BY_ID[arm]`, y si la declaración de hoy no lleva ningún brazo
    que seleccione, esta rama nunca se toma.
    """
    spec = config.ARMS_BY_ID[arm]
    budget = spec.get("selection") and spec.get("budget")
    if not budget:
        return config.INSTANCES_PER_BAG
    return min(int(budget), config.INSTANCES_PER_BAG)


#: El piso alcanzable de `attentionSpread` bajo los hiperparámetros neutros de
#: hoy, por brazo -- computado de la cota que r21 declara para el logit, nunca
#: supuesto en 0 y nunca un solo número para todos: un brazo que selecciona
#: reparte sobre su propio presupuesto de instancias y no sobre las
#: `INSTANCES_PER_BAG` de la bolsa completa, así que su piso es otro. Con
#: `ATTENTION_GAMMA = 0` y `ATTENTION_TEMPERATURE = 1` el rango alcanzable de
#: cada uno es angosto y pegado a uno: "lejos del máximo" nunca significa
#: "cerca de cero". Ningún id de brazo está escrito acá: la declaración de hoy
#: (`config.ARMS`) puede no llevar ningún brazo que seleccione, y entonces este
#: diccionario coincide con `MIN_ATTENTION_SPREAD` para todos.
MIN_ATTENTION_SPREAD_BY_ARM = {
    arm["id"]: _min_reachable_attention_entropy(
        _bag_size_for(arm["id"]), config.ATTENTION_GAMMA, config.ATTENTION_TEMPERATURE)
    for arm in config.ARMS if arm["attention"] == "learned"
}

#: El piso sobre una bolsa completa (`INSTANCES_PER_BAG` instancias), sin
#: seleccionar -- lo que todo brazo salvo `SU`/`SA`/`SK` reparte sobre, y el
#: valor que `MIN_ATTENTION_SPREAD_BY_ARM` ya guarda para cada uno de ellos.
#: Mantenido con este nombre porque es el que la prosa general (`objective`,
#: el encabezado de `conclusion_attention`) cita como referencia.
MIN_ATTENTION_SPREAD = _min_reachable_attention_entropy(
    config.INSTANCES_PER_BAG, config.ATTENTION_GAMMA, config.ATTENTION_TEMPERATURE)

#: A partir de acá se lee la atención como pegada a la media uniforme. Es un
#: umbral de lectura, no una medición, y cae dentro del rango alcanzable
#: (`MIN_ATTENTION_SPREAD` a 1.000): r21 l.501 dice que un valor así es lo
#: esperado en una bolsa dispersa sin grupo dominante, y no es, por sí solo,
#: un fallo de la atención -- ahí r21 lo empareja con una autosimilitud BAJA
#: de la propia bolsa, no con la masa de correspondencia. Esa autosimilitud
#: es una lectura distinta, y este informe no la calcula ni la imprime.
UNIFORM_ATTENTION = 0.99


def _selecting_arms_floor_clause() -> str:
    """La cláusula que nombra el piso de un brazo que selecciona, o nada.

    Ningún id de brazo se escribe acá: se recorre `config.ARMS` y se nombra
    cualquiera cuyo `spec["selection"]` no sea `None`, cualquiera que sea su
    id o cuántos sean. La declaración de hoy no lleva ninguno, así que esto
    devuelve la cadena vacía y la frase de `objective("attentionSpread")` sale
    sin esa cláusula -- en vez de nombrar un brazo retirado o morir buscando
    una constante que la declaración ya no lleva.
    """
    selecting = [arm for arm in config.ARMS if arm.get("selection") is not None]
    if not selecting:
        return ""
    names = ", ".join(sorted(arm["id"] for arm in selecting))
    floors = {MIN_ATTENTION_SPREAD_BY_ARM.get(arm["id"], MIN_ATTENTION_SPREAD)
              for arm in selecting}
    piso = f"{min(floors):.3f}" if len(floors) == 1 else (
        f"entre {min(floors):.3f} y {max(floors):.3f}")
    return f", {piso} sobre la bolsa reducida que seleccionan {names}"

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
    """{(brazo, transferencia): spread} sobre cada repetición de esa celda.

    Es el único lugar por donde pasa toda tabla que sale de las corridas crudas
    —`table`, y detrás de ella `render`, `conclusion`, `ranking` y
    `best_transfers`—, así que la negativa a agrupar una dimensión `perRun` va
    acá y no repetida en cada una. Puesta sólo en las de arriba, la próxima
    función que lea corridas nacería sin ella y nadie lo notaría hasta ver el
    número impreso.
    """
    _refuse_pooling(metric)
    gathered: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        gathered.setdefault((run["arm"], run["transfer"]), []).append(float(run[metric]))
    return {key: spread(values) for key, values in gathered.items()}


def _mean_se(values: list[float]) -> tuple[float, float]:
    """Media y su error estándar, sobre las repeticiones de un mismo brazo.

    Con menos de dos valores el error es cero: es un hecho sobre la muestra
    -- una sola lectura no tiene con qué estimar su propia dispersión -- y no
    un veredicto de precisión perfecta.
    """
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var) / math.sqrt(n)


def table(runs: Iterable[dict], metric: str) -> list[dict]:
    """Una fila por método: cada transferencia, el promedio y el pico.

    `avg` promedia las medias por transferencia, así una transferencia cuenta una
    vez por muchas repeticiones que haya corrido. `max` promedia los máximos por
    transferencia — el método en su semilla más afortunada de cada una — y existe
    para que el pico quede documentado como número acá, en lugar de colarse en
    una figura eligiendo el mejor modelo.

    `se` es el error estándar de la media del brazo, sobre TODAS sus corridas
    crudas (cada transferencia, cada semilla, agrupadas en una sola muestra) --
    la cantidad que `rank_groups` necesita para decidir si dos brazos se
    distinguen, y que ni `cells` ni el resto de esta fila calculan.
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
        raw = [float(run[metric]) for run in runs if run["arm"] == arm]
        _, se = _mean_se(raw)
        rows.append({
            "arm": arm,
            "name": config.NAME_OF[arm],
            "byTransfer": {label: grid.get((arm, label)) for label in labels},
            "avg": sum(c["mean"] for c in present) / len(present),
            "max": sum(c["max"] for c in present) / len(present),
            "se": se,
            "n": len(raw),
            "share": (sum(shares[(arm, label)]["mean"] for label in labels
                          if (arm, label) in shares)
                      / max(1, sum(1 for label in labels if (arm, label) in shares))),
        })
    return rows


def rank_groups(rows: list[dict], better: str | None = config.HIGHER) -> dict[str, int]:
    """{brazo: puesto}, compartiendo puesto cuando dos brazos no se distinguen.

    Ordena por `avg` y camina la lista: un brazo se une al puesto del anterior
    cuando la brecha entre los dos no supera su error estándar COMBINADO --
    `sqrt(se_a^2 + se_b^2)`, el error estándar de la diferencia de dos medias
    independientes -- y abre un puesto nuevo cuando sí la supera. Compartir es
    por lo tanto una propiedad de **vecinos**: una cadena de tres brazos cada
    uno indistinguible del siguiente comparte un solo puesto aunque los dos
    extremos sí se distingan entre sí, porque nada en los datos separa a
    ningún par adyacente de esa cadena.

    Con `se=0` en todas las filas -- un piloto de una repetición -- el error
    combinado es siempre cero y cada brazo abre su propio puesto: sin
    dispersión medida no hay empate que conceder, y el sello de piloto ya
    dice que estas cifras no son un veredicto.
    """
    reverse = better != config.LOWER
    ordered = sorted(rows, key=lambda r: r["avg"], reverse=reverse)
    places: dict[str, int] = {}
    place = 1
    for index, row in enumerate(ordered):
        if index > 0:
            previous = ordered[index - 1]
            combined_se = math.sqrt(row.get("se", 0.0) ** 2 + previous.get("se", 0.0) ** 2)
            if abs(row["avg"] - previous["avg"]) > combined_se:
                place += 1
        places[row["arm"]] = place
    return places


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
        "noise":
            f"**Buscamos que la caída sea chica y que el orden entre métodos no se "
            f"invierta.** Contaminado el material de entrenamiento, todos caen; lo "
            f"que separa a un método es cuánto. El piso sigue siendo el azar de "
            f"{clase:.3f}: un método que llega ahí dejó de decidir. Las filas "
            f"`{NOISE_DIRTY}` de las tablas de esta sección salieron de la campaña "
            f"con las etiquetas de entrenamiento contaminadas a "
            f"ρ={config.NOISE_REPORTED:g}; las `{NOISE_CLEAN}`, del material "
            f"limpio. La tasa va acá y no en la tabla: es una sola para toda la "
            f"sección, y escrita por fila sería el mismo hecho tantas veces como "
            f"filas haya.",
        "noise.floor.source":
            f"**Buscamos ver si el ruido perjudica incluso al método que no "
            f"adapta, en fuente.** El piso entrena sólo con etiquetas de fuente, "
            f"así que una caída acá no puede venir de ningún término de "
            f"adaptación -- mide cuánto empeora la propia representación cuando "
            f"el material de entrenamiento está contaminado. El piso de la "
            f"lectura sigue siendo el azar de {clase:.3f}, y lo esperable es que "
            f"esta curva caiga menos que la de destino: la fuente nunca se "
            f"contaminó a sí misma, sólo el rótulo que la acompaña.",
        "noise.floor.target":
            f"**Buscamos ver si el ruido perjudica incluso al método que no "
            f"adapta, en destino.** La misma pregunta que la curva de fuente, "
            f"sobre el dominio que el piso nunca ve durante el entrenamiento: "
            f"si ni siquiera el destino de un método que no adapta se mueve con "
            f"el ruido, la contaminación no está llegando a la representación "
            f"en absoluto. El piso de la lectura sigue siendo el azar de "
            f"{clase:.3f}.",
        "latent.grid.clean":
            "**Buscamos ver si las clases se mezclan entre dominios y siguen "
            "separadas entre sí, sobre material limpio.** Un panel por método, "
            "el espacio original primero: «alineado» no se ve sin un «no "
            "alineado» al lado, que es lo que el primer panel de la fila da.",
        "latent.grid.noisy":
            "**Buscamos cuánta de la mezcla entre clases que se veía en limpio "
            "sobrevive cuando el entrenamiento se ensucia.** La misma rejilla, "
            "sobre el material contaminado: lo que importa acá no es el panel "
            "por sí solo sino la comparación con la rejilla limpia de arriba.",
        "correspondence.grid.clean":
            "**Buscamos que el triángulo destacado caiga entre círculos de su "
            "mismo color, en material limpio**, y que eso pase más en la "
            "columna con término local que en la de al lado -- esa comparación "
            "es la razón de que el piso, la versión sin término local y la "
            "completa compartan la misma fila.",
        "correspondence.grid.noisy":
            "**Buscamos si la correspondencia local sobrevive al ruido mejor "
            "que la global.** La misma figura, sobre el material contaminado, "
            "leída junto a la limpia y nunca por separado.",
        "floors":
            "**Buscamos que la diferencia entre los dos pisos sea menor que la que "
            "los separa de cualquier método con adaptación.** Si lo es, la segunda "
            "columna no muestra nada que la primera no muestre y se colapsa.",
        "grid":
            "**Buscamos que los dos dominios se mezclen dentro de cada clase y que "
            "las clases sigan separadas entre sí.** Mezclar todo también junta los "
            "dominios, y eso no es alinear sino colapsar.",
        "grid.contaminated":
            "**Buscamos que sobreviva la misma estructura que en la mitad limpia**: "
            "los dominios mezclados dentro de cada clase y las clases separadas "
            "entre sí. Lo que se lee es cuánto se degrada, y si se degrada menos "
            "donde el término local está declarado.",
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
            f"propio método, transferencia por transferencia**: toda la familia "
            f"`MIL-` contra `MIL-Baseline`. La "
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
            f"**Descriptivo: no hay un valor que se persiga.** La entropía está "
            f"normalizada, así que 1.000 es la media uniforme, y bajo los "
            f"hiperparámetros neutros de hoy (gamma={config.ATTENTION_GAMMA:g}, "
            f"tau_att={config.ATTENTION_TEMPERATURE:g}) el piso que Eq. (16) puede "
            f"alcanzar depende de cuántas instancias el brazo reparte entre sí: "
            f"{MIN_ATTENTION_SPREAD:.3f} sobre una bolsa completa de "
            f"{config.INSTANCES_PER_BAG} instancias"
            f"{_selecting_arms_floor_clause()} -- nunca 0.000, en "
            f"ningún caso: r21 acota el logit en 2 + gamma en el párrafo que sigue "
            f"a la Ec. (16), así que el rango alcanzable es angosto y pegado a uno. "
            f"r21 l.501 dice además que un peso casi uniforme es lo esperado en una "
            f"bolsa dispersa sin grupo dominante -- ahí empareja ese peso con una "
            f"autosimilitud BAJA de la propia bolsa, una lectura distinta de la "
            f"de arriba -- y no es, por sí solo, un fallo de la atención.",
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


def _stale_families(record: dict | None) -> list[str]:
    """Cada familia de `record` cuyo sello discrepa con el `config` VIGENTE.

    `harness.search_record()` ya etiqueta cada entrada con `currentStamp` en
    vez de negarse -- reporta y no decide, la misma forma que `latent.available()`
    usa para una lista de checkpoints, y no la que `latent.load()` usa para uno
    que se va a medir. Estas dos funciones renderizan un informe, no cargan un
    registro para calcular con él, así que siguen esa misma forma: no se niegan,
    imprimen un aviso tan visible como el que ya existe para un registro de
    ensayo (`search_source_note`) -- un `verify` en frío no puede diferenciar
    entre "nunca se buscó" y "se buscó y quedó viejo" si las dos se leen igual.
    """
    return sorted(family for family, entry in (record or {}).items()
                  if isinstance(entry, dict) and ceiling_record.stamp_drift(entry))


def _stale_notice(record: dict | None, markdown: bool = False) -> str:
    """El aviso de sello vencido, sin separador propio, o cadena vacía.

    Sin `\\n` al final a propósito: cada llamador junta este texto con lo
    suyo de una forma distinta -- una tabla lo antepone con salto de línea,
    una conclusión en prosa lo junta con espacio -- y un separador fijo acá
    sería el correcto para uno solo de los dos.
    """
    stale = _stale_families(record)
    if not stale:
        return ""
    nombres = ", ".join(f"`{f}`" if markdown else f for f in stale)
    return (f"**Sello vencido** para {nombres}: buscado bajo una revisión o "
            f"unos hiperparámetros que el `config` vigente ya no lleva. Los "
            f"números de abajo describen esa búsqueda, no la que correría "
            f"hoy.")


def render_ceilings(record: dict | None, markdown: bool = False) -> str:
    """La rejilla de la búsqueda, una fila por familia y una columna por techo.

    Recibe el registro en lugar de leerlo: este módulo es solo biblioteca estándar
    y el registro lo lee `harness.search_record`, que necesita torch para todo lo
    demás. Y recibirlo entero, no solo el ganador: un techo elegido entre cuatro
    puntajes idénticos y uno elegido por una diferencia real son el mismo número y
    no la misma evidencia, así que la fila muestra la rejilla y marca cuál ganó.

    **Nunca la renderiza como vigente sin decirlo.** `_stale_notice` antepone un
    aviso cuando alguna familia quedó con un sello que el `config` de hoy ya no
    sostiene -- ver esa función para por qué es un aviso y no un rechazo.
    """
    aviso = _stale_notice(record, markdown=markdown)
    aviso = (aviso + ("\n\n" if markdown else "\n")) if aviso else ""
    # Dos formas, dos tablas. Una rejilla tiene columnas: los mismos techos en
    # todas las familias, así que la fila se lee de izquierda a derecha y la
    # inclinación se ve. Una búsqueda por trials no las tiene — cada familia y
    # cada transferencia visitaron puntos distintos de un rango continuo — y
    # forzarla a columnas inventaría un eje compartido que nadie midió.
    if record and all(ceiling_record.kind_of(e) == ceiling_record.KIND_OPTUNA
                      for e in record.values()):
        return aviso + _render_ceilings_trials(record, markdown=markdown)
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
        return aviso + "\n".join(lines)

    width = max(14, max((len(f) for f in record), default=14) + 2)
    lines = [f"{'Familia':<{width}}{'Brazo':>8}"
             + "".join(f"{c:g}".rjust(12) for c in grid)]
    for familia, entrada in sorted(record.items()):
        lines.append(f"{familia:<{width}}{entrada['arm']:>8}"
                     + "".join(cell.rjust(12) for cell in cells(entrada)))
    return aviso + "\n".join(lines)


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
    aviso = _stale_notice(record)
    partes, por_regla = [aviso] if aviso else [], []
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
        # Las etiquetas nombran QUIÉN se aparta; el factor dice CUÁNTO, y es lo
        # único de los dos que la tabla no muestra. Listar el techo de cada
        # transferencia que se aparta sería la tabla otra vez en prosa: el
        # lector ya la tiene arriba, y una conclusión que la repite dejó de
        # concluir.
        fuera = ""
        if heredan:
            cuantas = ("La restante corre" if heredan == 1
                       else f"Las {heredan} restantes corren")
            fuera = (f" {cuantas} al ganador agrupado, "
                     f"{entrada['ceiling']:g}, fuera de muestra.")
        if aparte:
            etiquetas = ", ".join(f"`{l}`" for l in aparte)
            valores = list(medidos.values())
            factor = max(valores) / min(valores) if min(valores) > 0 else None
            reparto = (f"Entre el techo más alto y el más bajo de la fila hay un "
                       f"factor {factor:.1f}. " if factor else "")
            partes.append(
                f"**{familia}**: {len(medidos)} medida(s), {heredan} heredada(s), y "
                f"{len(aparte)} de las medidas elige otro techo — {etiquetas}."
                f"{fuera} {reparto}Ahí la familia deja de correr a un coeficiente "
                f"único, así que su promedio entre transferencias mezcla escalares "
                f"distintos; dentro de cada transferencia todos los brazos siguen "
                f"compartiendo el techo, que es lo que mantiene atribuible cada "
                f"peldaño.")
        else:
            partes.append(
                f"**{familia}**: {len(medidos)} medida(s) y {heredan} heredada(s), "
                f"todas en {entrada['ceiling']:g}. Ninguna transferencia medida se "
                f"aparta del ganador agrupado, así que separar las dos lecturas no "
                f"cambió ningún techo de esta familia.")
    return " ".join(partes)


def per_run_dimensions() -> list[str]:
    """Las dimensiones que la declaración del propio banco dice que no se agrupan.

    Reexportada desde `pooling`, que es donde vive ahora. Sigue acá porque los
    llamadores de este módulo la piden por este nombre, y porque una tabla que
    quiere saber qué no puede promediar la tiene a mano.
    """
    return _pooling.per_run_dimensions()


#: Cómo se encabeza la columna de ruido y cómo se lee cada bloque. Declarado una
#: vez porque el encabezado y las celdas tienen que decir la misma cosa: `sin` no
#: es «sin medir», es el material limpio medido.
#:
#: Dice `sin`/`con` y no la tasa porque la tasa es fija: escrita en cada fila,
#: `con` y `0.2` serían el mismo hecho dos veces, y una medición renderizada dos
#: veces es lo que el contrato del informe cuenta como defecto. El número va una
#: sola vez por sección, calculado, en el encuadre que esa sección ya tiene.
NOISE_COLUMN = "Ruido"
NOISE_CLEAN = "sin"
NOISE_DIRTY = "con"

#: Cuánto mide la columna de ruido en la forma de ancho fijo. Los dos valores
#: tienen tres letras, así que el ancho lo pone el encabezado.
_NOISE_WIDTH = len(NOISE_COLUMN) + 2


def _level_or_note(rate: float):
    """El registro contaminado que rige, o el texto que explica por qué no hay.

    Una sola declaración para las cinco tablas que ahora llevan bloque `con`.
    Antes cada gemela `_at` traía su propia copia de estas dos frases, y cinco
    copias son cinco textos que pueden separarse: el día que una campaña
    ausente dejara de decirse igual, el lector no tendría cómo saber si las dos
    secciones miraron lo mismo.

    Lo que devuelve no es «no hay tabla» sino «no hay bloque». La tabla existe
    igual con el bloque `sin` solo: quitarla entera le sacaría las filas limpias
    a la corrida que tiene la mitad del material, y ese estado es alcanzable.
    """
    from MIL_CREDA_Benchmark import contamination as axis

    nivel = axis.load(rate, "campaign")
    if nivel is None:
        return None, (f"Sin bloque `{NOISE_DIRTY}`: la campaña a ρ={rate:g} "
                      f"todavía no corrió. No está vacío: no existe.")
    if axis.mismatched(nivel):
        return None, (f"Sin bloque `{NOISE_DIRTY}`: el registro hallado para "
                      f"ρ={rate:g} declara `labelNoise={axis.stated_rate(nivel)}`. "
                      f"Un directorio no es evidencia y las dos cosas no "
                      f"coinciden: nada se dibuja hasta que se resuelva cuál "
                      f"está mal.")
    return nivel, None


def _with_note(lines: list[str], note: str | None) -> str:
    """La tabla, y debajo el motivo por el que le falta el bloque `con`."""
    if note:
        lines = lines + ["", note]
    return "\n".join(lines)


def _noise_title(title: str, rate: float | None, markdown: bool) -> str:
    """La línea de título de una tabla de dos bloques, con la tasa dicha una vez.

    Acá y en ningún otro lado de la tabla. La columna `Ruido` dice de qué
    material salió cada fila; **cuánto** ruido es un solo hecho sobre la
    campaña entera, y repetirlo por fila lo convertiría en tantas copias como
    filas haya. Sale de `rate`, que es el valor con el que se cargaron las
    lecturas, así que no hay forma de que la línea nombre una tasa y la tabla
    lleve otra.
    """
    head = f"**{title}**" if markdown else title
    if rate is None:
        return head
    return (f"{head} — el bloque `{NOISE_DIRTY}` corrió con las etiquetas de "
            f"entrenamiento contaminadas a ρ={float(rate):g}.")


def _refuse_pooling(metric: str) -> None:
    """La negativa, para toda función de este módulo que agrupe lo que le pasan.

    La guarda nació sobre `render` sola, y `render` no era la única que agrupaba:
    `conclusion` promedia las mismas celdas y las imprime como «mejor y peor»,
    los peldaños restan dos medias, y las ganancias apareadas sacan media, error
    y rango. Una sola de esas protegida se lee como la clase entera cubierta, que
    es la forma exacta en que el hueco se vuelve invisible.

    El texto ya no está escrito acá sino en `pooling`, y la mudanza tiene su
    propia razón: cubierta la familia del veredicto, la del ruido seguía sin
    guardia porque agrega por `contamination.by_arm` y nunca pasa por `cells`.
    Dejar la negativa en un módulo que formatea obligaba a `contamination` y a
    `figures` a importarlo para conseguir una regla que es de la declaración, o
    a copiar el párrafo --- y cinco copias son cinco textos que pueden
    separarse.
    """
    _pooling.refuse(metric)


def render(runs: Iterable[dict], metric: str, reduction: dict,
           rate: float | None = None, markdown: bool = False) -> str:
    """Una fila por método y por material: sus seis transferencias y el promedio.

    Nada de máximos ni de participaciones. El máximo de N repeticiones crece con
    la dispersión del propio método, así que impreso al lado de una media favorece
    al brazo más ruidoso — es la misma razón por la que se guarda el artefacto
    mediano y nunca el mejor. La participación del término en el objetivo tiene su
    propia figura, que es donde se lee como trayectoria y no como un número suelto.

    El promedio va último porque es un resumen: se lee después de aquello que
    resume, no antes.

    **Con `rate` dibuja los dos materiales en una sola tabla**: primero todas las
    filas `sin` y después todas las `con`, con la columna `Ruido` adelante. Antes
    eran dos llamadas —esta y una gemela `render_at` que cargaba el registro
    contaminado y volvía a llamar acá—, y esa gemela existía sólo para que el
    chequeo de duplicación no leyera dos renderizaciones del mismo número donde
    hay dos números distintos. Con una llamada no hay nada que distinguir: una
    cantidad, una tabla. El registro contaminado se pide por su tasa porque es
    así como está guardado, y la guarda que compara el `labelNoise` declarado
    contra el directorio se dice una vez, en `_level_or_note`.

    La tasa no entra en ninguna fila. Es fija, así que escrita por fila sería el
    mismo hecho repetido; el número lo dice el encuadre de la sección —
    `objective("noise")`, calculado de la declaración — una sola vez.

    Y se niega de plano ante una dimensión declarada `perRun`. `seconds` y
    `peakMiB` -- las dos que fueron `perRun` -- ya no están en la declaración
    ni en el objetivo de este informe; la guarda queda igual, general, para
    cualquier dimensión que una declaración futura marque `perRun`. Un
    promedio que la declaración prohíbe se imprime igual de convincente que
    uno que permite.

    La negativa ya no está escrita acá sino en `cells`, por donde esta función
    pasa antes de imprimir nada. Repetida en las dos era una guarda que no se
    podía probar: quitarla de acá dejaba la suite entera en verde, porque la de
    abajo la tapaba — un candado idéntico a uno vivo y que no cierra nada.

    **Lleva un puesto por bloque, calculado con `rank_groups` y nunca a mano.**
    Cada bloque -- `sin` y, si lo hay, `con` -- se ordena y se reparte puestos
    por su cuenta: el material limpio y el contaminado no comparten error
    estándar, así que compartir puesto en uno no dice nada sobre el otro. Dos
    brazos comparten puesto cuando la brecha entre sus promedios no supera su
    error estándar combinado, y ahí lo dice el propio número: dos filas con el
    mismo puesto son, con la evidencia de esta corrida, indistinguibles.
    """
    labels = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS]
    title, unit = SPANISH.get(metric, (metric, ""))
    seeds = reduction.get("seeds", config.SEEDS)
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else int(seeds or 0)
    better = config.DIMENSIONS.get(metric, config.HIGHER)

    # Sin encabezado propio: qué mide esta tabla lo dice el párrafo de arriba, y
    # bajo qué límites corrió lo dice el sello, una vez, antes de todo.

    nota = None
    bloques = [(NOISE_CLEAN, table(runs, metric))]
    if rate is not None:
        nivel, nota = _level_or_note(rate)
        if nivel is not None:
            bloques.append((NOISE_DIRTY, table(nivel["runs"], metric)))

    return _render_ranked_blocks(bloques, metric, labels, better, markdown, note=nota)


def _render_ranked_blocks(bloques: list[tuple[str, list[dict]]], metric: str,
                          labels: list[str], better: str | None,
                          markdown: bool = False, note: str | None = None,
                          row_header: str = "Método") -> str:
    """El cuerpo compartido de toda tabla de dos bloques con puesto por bloque.

    Extraído de `render` para que `render` (por brazo declarado) y una
    comparación por mecanismo de atención (sección 4, por nombre de mecanismo
    y no por brazo) impriman exactamente la misma forma sin duplicar el
    formato: una cantidad, una manera de mostrarla.
    """
    rows = []
    for ruido, filas in bloques:
        puestos = rank_groups(filas, better) if better in (config.HIGHER, config.LOWER) else {}
        for row in filas:
            rows.append(dict(row, noise=ruido, place=puestos.get(row["arm"])))

    decimals = 1 if metric in PERCENT else 2

    def cell(entry) -> str:
        if entry is None:
            return "—"
        return (f"{_scaled(entry['mean'], metric):.{decimals}f} ± "
                f"{_scaled(entry['stdev'], metric):.{decimals}f}")

    def place_cell(row) -> str:
        return "—" if row["place"] is None else str(row["place"])

    columns = [NOISE_COLUMN, row_header] + labels + ["Prom.", "Puesto"]
    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            values = [cell(row["byTransfer"][label]) for label in labels]
            lines.append("| " + " | ".join(
                [row["noise"], f"`{row['name']}`", *values,
                 f"**{_scaled(row['avg'], metric):.{decimals}f}**",
                 place_cell(row)]) + " |")
        return _with_note(lines, note)

    width = max(14, max((len(r["name"]) for r in rows), default=14) + 2)
    lines = [f"{NOISE_COLUMN:<{_NOISE_WIDTH}}{row_header:<{width}}"
             + "".join(f"{label:>16}" for label in labels)
             + f"{'Prom.':>10}{'Puesto':>8}"]
    for row in rows:
        values = "".join(f"{cell(row['byTransfer'][label]):>16}" for label in labels)
        lines.append(f"{row['noise']:<{_NOISE_WIDTH}}{row['name']:<{width}}{values}"
                     + f"{_scaled(row['avg'], metric):>10.{decimals}f}"
                     + f"{place_cell(row):>8}")
    return _with_note(lines, note)


def conclusion(runs: Iterable[dict], metric: str, reduction: dict) -> str:
    """La lectura más relevante de la tabla, calculada y no escrita a mano.

    Una conclusión escrita a mano es una segunda fuente de verdad: se desactualiza
    en silencio y se le cree igual. Esta se recalcula con la tabla, así que no
    puede alejarse de lo que describe.

    Se niega ante una dimensión `perRun` por la misma vía que la tabla: ordena
    por `avg`, que es el promedio que `render` no imprime, y decirlo en prosa no
    lo vuelve defendible. La guarda está en `cells`, una sola vez, y llega hasta
    acá por `table`.
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

    # Cada brazo adaptado contra su propio piso, que es la única lectura que
    # separa lo que aporta la adaptación de lo que aporta la representación.
    # `config.FLOOR_OF` entero y no un par escrito a mano: qué brazo es "el
    # completo" es una lectura de la declaración de hoy, y una declaración que
    # cambie de arms no debe obligar a tocar esta función para seguir siendo
    # correcta.
    for arm, floor in sorted(config.FLOOR_OF.items()):
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


# ------------------------------------------------- mecanismos de atención

#: La ruta donde vive el barrido de mecanismos, declarada acá porque `render_
#: mechanisms` y la celda del cuaderno tienen que nombrar el mismo archivo. No
#: hay productor todavía -- ningún módulo de `wiring`/`harness` corre el
#: método completo bajo un mecanismo de atención distinto del propio -- así
#: que esta ruta puede no existir nunca, y eso se reporta y no se disimula.
MECHANISM_RECORD = "Results/Benchmark/attention_mechanisms.json"


def mechanism_table(runs: Iterable[dict], metric: str, mechanisms: list[str]) -> list[dict]:
    """Como `table`, pero la fila es un mecanismo de atención y no un brazo declarado.

    La sección 4 compara MECANISMOS dentro del único brazo completo -- nunca
    un id de `config.ARMS` -- así que lee `run["mechanism"]` en vez de
    `run["arm"]`, y el orden de las filas es el que `mechanisms` declare: el
    nombre de cada mecanismo lo trae el propio registro, nunca esta función.
    """
    runs = list(runs)
    grid: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        grid.setdefault((run["mechanism"], run["transfer"]), []).append(float(run[metric]))
    labels = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS]

    rows = []
    for mechanism in mechanisms:
        present = [spread(grid[(mechanism, label)]) for label in labels
                   if (mechanism, label) in grid]
        if not present:
            continue
        raw = [value for label in labels for value in grid.get((mechanism, label), [])]
        _, se = _mean_se(raw)
        rows.append({
            "arm": mechanism,
            "name": mechanism,
            "byTransfer": {label: (spread(grid[(mechanism, label)])
                                   if (mechanism, label) in grid else None)
                          for label in labels},
            "avg": sum(c["mean"] for c in present) / len(present),
            "se": se,
            "n": len(raw),
        })
    return rows


def render_mechanisms(record: dict | None, metric: str, markdown: bool = False) -> str:
    """Sección 4: un mecanismo de atención por fila, sobre el brazo completo.

    `record` es `{"mechanisms": [...], "clean": [runs...], "noisy": [runs...]}`
    -- cada `run` lleva `mechanism`, `transfer` y la métrica, exactamente como
    un `run` de `runs.jsonl` lleva `arm` en su lugar. Sin registro, o sin la
    clave `mechanisms`, se dice llanamente que el barrido no corrió: ningún
    nombre de mecanismo (ABMIL publicado, ABMIL con compuerta, max, mean, o el
    propio de este método) está escrito en este módulo, así que no hay nada
    que esta función pueda inventar para llenar la tabla.
    """
    if not record or not record.get("mechanisms"):
        return ("Sin barrido de mecanismos de atención: la comparación no "
                "corrió todavía. No es una tabla vacía: es que la corrida no "
                "existe.")
    mechanisms = list(record["mechanisms"])
    better = config.DIMENSIONS.get(metric, config.HIGHER)
    labels = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS]
    bloques = [(NOISE_CLEAN, mechanism_table(record.get("clean") or [], metric, mechanisms))]
    if record.get("noisy"):
        bloques.append((NOISE_DIRTY, mechanism_table(record["noisy"], metric, mechanisms)))
    return _render_ranked_blocks(bloques, metric, labels, better, markdown,
                                 row_header="Mecanismo")


def conclusion_mechanisms(record: dict | None, metric: str) -> str:
    """Cuál mecanismo queda adelante, calculado de `render_mechanisms` y no escrito a mano."""
    if not record or not record.get("mechanisms"):
        return ("Sin barrido: no hay mecanismo que comparar todavía.")
    mechanisms = list(record["mechanisms"])
    rows = mechanism_table(record.get("clean") or [], metric, mechanisms)
    if not rows:
        return "El registro no trae corridas limpias: no hay nada que concluir."
    better = config.DIMENSIONS.get(metric, config.HIGHER)
    reverse = better == config.HIGHER
    ordered = sorted(rows, key=lambda r: r["avg"], reverse=reverse)
    best, worst = ordered[0], ordered[-1]
    decimals = 1 if metric in PERCENT else 2
    spread_value = abs(best["avg"] - worst["avg"])
    return (f"En material limpio, el mecanismo que queda adelante es "
            f"**{best['name']}**; el que queda atrás, {worst['name']}, a "
            f"{_scaled(spread_value, metric):.{decimals}f}{'%' if metric in PERCENT else ''} "
            f"de distancia.")


def conclusion_normalization(runs: Iterable[dict], key: str) -> str:
    """Si la curva de `key` se quedó dentro de [0, 1) en toda esta corrida.

    La única lectura que la sección 6 pide -- no cuál método tiene la curva
    más baja, sino si la normalización que la Ec. (39) promete se sostiene.
    Barre cada punto de cada trayectoria de cada corrida, así que un solo
    paso de un solo brazo que se saliera del intervalo alcanza para que esta
    conclusión lo diga.
    """
    valores = [float(punto[key]) for run in runs for punto in run.get("curve", [])
              if key in punto]
    if not valores:
        return "Sin puntos de curva para esta cantidad: no hay nada que comprobar."
    bajo, alto = min(valores), max(valores)
    if 0.0 <= bajo and alto < 1.0:
        return (f"La curva se queda dentro de [0, 1) en toda esta corrida: "
                f"mínimo {bajo:.4f}, máximo {alto:.4f}. La normalización se "
                f"sostiene.")
    return (f"La curva SALE de [0, 1) en esta corrida: mínimo {bajo:.4f}, "
            f"máximo {alto:.4f}. Eso es el hallazgo, no un detalle de "
            f"implementación.")


# ----------------------------------------------------------------- lecturas anidadas

def _reach(record: dict, path: str):
    """`geometry.ratio` dentro de una lectura anidada, o nada si el brazo no la tiene."""
    value = record
    for step in path.split("."):
        if not isinstance(value, dict) or step not in value:
            return None
        value = value[step]
    return float(value) if isinstance(value, (int, float)) else None


def render_readings(readings: Iterable[dict], path: str, title: str,
                    contaminated: Iterable[dict] = (), rate: float | None = None,
                    markdown: bool = False) -> str:
    """Una medición de fase dos, limpia y contaminada, en una tabla de dos bloques.

    Antes eran dos tablas: una por tasa, separadas por un párrafo y con la misma
    forma justamente para que el lector no tuviera que traducir entre ellas. Eso
    seguía pidiéndole que recordara una fila mientras bajaba a buscar la otra.
    Acá el ruido es una **columna** —la primera— y la tabla va en dos bloques:
    todas las filas `sin` y después todas las `con`, en el mismo orden de brazos.
    Un brazo se lee por columna, no por adyacencia: la fila `con` de `MIL-CREDA`
    está tantas filas más abajo como brazos tenga el bloque.

    Al haber una sola renderización por cantidad, el chequeo de duplicación deja
    de tener dos llamadas que distinguir: hay una, y lleva adentro los dos
    números. Lo que antes obligaba a nombrar la contaminada aparte desapareció
    con la segunda tabla.

    La columna dice `sin`/`con` y no la tasa. La tasa es fija, así que
    escrita en cada fila sería el mismo hecho repetido tantas veces como filas
    haya, y el contrato del informe cuenta una medición renderizada dos veces
    como defecto. El número va **una vez**, calculado de `rate`, en la línea de
    título de esta tabla —que ya existía como parámetro y no se imprimía— y no
    tipeado en ningún lado.

    El `±` de acá **no** es el de las tablas de fase uno: es la dispersión entre
    los checkpoints guardados de esa celda. La fase uno mide la variabilidad de la
    exactitud; esto mide la variabilidad de la geometría, y nada establece que una
    siga a la otra.

    Un brazo sin lectura para esta cantidad — uno sin término local no tiene
    correspondencia — recibe una celda vacía, no un cero.

    Sin lecturas contaminadas imprime el bloque `sin` solo y no se niega: la
    campaña a ρ puede todavía no haber dejado checkpoints, y ese estado es
    alcanzable —el cuaderno lo tiene guardado en cada llamada— así que una tabla
    de un bloque es una respuesta y no una falla.
    """
    readings, sucias = list(readings), list(contaminated)
    if sucias and rate is None:
        raise ValueError(
            "hay lecturas contaminadas y ninguna tasa que nombrar: un bloque "
            "`con` sin su ρ no dice de qué campaña salió.")
    labels = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS]

    def reunir(source: list[dict]) -> dict[tuple[str, str], list[float]]:
        gathered: dict[tuple[str, str], list[float]] = {}
        for reading in _own_medians(source):
            value = _reach(reading, path)
            if value is not None:
                gathered.setdefault((reading["arm"], reading["transfer"]), []).append(value)
        return gathered

    bloques = [(NOISE_CLEAN, reunir(readings))]
    if sucias:
        bloques.append((NOISE_DIRTY, reunir(sucias)))

    rows = []
    for ruido, gathered in bloques:
        for arm in config.ARM_ORDER:
            by_label = {label: (spread(gathered[(arm, label)]) if (arm, label) in gathered
                                else None) for label in labels}
            present = [c for c in by_label.values() if c]
            if not present:
                continue
            rows.append({"noise": ruido, "name": config.NAME_OF[arm],
                         "cells": by_label,
                         "avg": sum(c["mean"] for c in present) / len(present),
                         "n": max(c["n"] for c in present)})

    def cell(entry) -> str:
        return "—" if entry is None else f"{entry['mean']:.3f} ± {entry['stdev']:.3f}"

    columns = [NOISE_COLUMN, "Método"] + labels + ["Prom."]
    if markdown:
        lines = [_noise_title(title, rate if sucias else None, markdown), "",
                 "| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(
                [row["noise"], f"`{row['name']}`",
                 *(cell(row["cells"][l]) for l in labels),
                 f"**{row['avg']:.3f}**"]) + " |")
        return "\n".join(lines)

    width = max(14, max((len(r["name"]) for r in rows), default=14) + 2)
    lines = [_noise_title(title, rate if sucias else None, markdown), "",
             f"{NOISE_COLUMN:<{_NOISE_WIDTH}}{'Método':<{width}}"
             + "".join(f"{l:>18}" for l in labels) + f"{'Prom.':>10}"]
    for row in rows:
        lines.append(f"{row['noise']:<{_NOISE_WIDTH}}{row['name']:<{width}}"
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


def _repetitions_measured(readings: Iterable[dict]) -> int:
    """Cuántas repeticiones sostiene de verdad la tabla que se está sellando.

    Se llamaba `_repetitions`, igual que la función de fase uno cien líneas más
    arriba, y en un módulo eso no son dos funciones: la de abajo tapa a la de
    arriba. `conclusion()` pedía la de la reducción y recibía ésta, que espera
    lecturas, y moría con `string indices must be integers` --- en la conclusión
    de CADA métrica del informe. Ninguna suite lo vio porque ninguna ejecuta el
    cuaderno; lo encontró el piloto la primera vez que algo lo corrió entero.

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
        repetitions = _repetitions_measured(_own_medians(readings))
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

    **Nombra el azar y no lo imprime.** El valor lo dicen ya el objetivo de la
    sección y la línea de título de su tabla, las dos calculados de
    `DOMAIN_CHANCE`; repetirlo acá era un tercer lugar para un mismo número y
    uno de los que el chequeo de duplicación cuenta. Con la conclusión cruzada
    compuesta en la misma frase —una tabla, una conclusión— ese tercer lugar
    llevaba el texto por encima del límite: la lectura entera se caía por un
    número que la tabla ya tenía arriba.
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
                     f"distancia al azar: "
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
    """Dónde cae cada brazo en el rango que Eq. (16) puede alcanzar, no si
    "falló".

    Descriptiva y no se disputa: no hay un lado que gane, y un valor pegado a
    la media uniforme no es, por sí solo, un fallo de la atención -- r21 l.501
    dice que es lo esperado en una bolsa dispersa sin grupo dominante. Bajo
    los hiperparámetros neutros el piso alcanzable
    (`MIN_ATTENTION_SPREAD_BY_ARM`) está lejos de cero para cada brazo, así
    que "lejos de la media uniforme" nunca significa "cerca de cero" tampoco
    -- y el piso NO es el mismo número para todos: un brazo que selecciona
    reparte sobre menos instancias, así que el suyo es otro.
    """
    readings = list(readings)
    values = _by_arm(readings, "attentionSpread")
    if not values:
        return "Ningún brazo guardado usa atención: no hay dispersión que concluir."

    lines = []
    for arm in config.ARM_ORDER:
        if arm not in values:
            continue
        floor = MIN_ATTENTION_SPREAD_BY_ARM.get(arm, MIN_ATTENTION_SPREAD)
        bag_size = _bag_size_for(arm)
        span = 1.0 - floor
        by_transfer = values[arm]
        mean = sum(by_transfer.values()) / len(by_transfer)
        position = (mean - floor) / span if span else 1.0
        if mean >= UNIFORM_ATTENTION or position >= 2 / 3:
            reading = ("reparte casi por igual entre todas las instancias, cerca de "
                       "la media uniforme")
        elif position <= 1 / 3:
            reading = ("queda cerca del extremo más concentrado que estos "
                       "hiperparámetros permiten alcanzar")
        else:
            reading = "reparte de forma desigual, a mitad de camino del rango alcanzable"
        lines.append(f"{config.NAME_OF[arm]} (sobre {bag_size} instancias, piso "
                     f"{floor:.3f}) {reading}")
    lines.insert(0, f"Bajo gamma={config.ATTENTION_GAMMA:g} y "
                    f"tau_att={config.ATTENTION_TEMPERATURE:g}, el piso que Eq. (16) "
                    f"alcanza depende de sobre cuántas instancias reparte cada brazo -- "
                    f"nunca menos que ese piso, y un valor cercano a uno es lo esperado "
                    f"en una bolsa dispersa (r21 l.501), no un fallo por sí solo. Cómo "
                    f"quedó cada uno dentro de su propio rango alcanzable:")
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


def render_bag_neighbors(entries: Iterable[dict], markdown: bool = False) -> str:
    """Por cada bolsa de destino (evaluación), sus 5 bolsas fuente más cercanas.

    `entries` es lo que devuelve `latent.top_k_source_bags`: una lista de
    `{"targetBag", "targetLabel", "neighbours": [{"sourceBag", "kernel",
    "trueClass"}, ...]}`, ya en orden de cercanía. Cada vecino marca si es de
    la clase verdadera, para leer de un vistazo si el top-5 de una bolsa es
    mayormente de su propia clase o no.
    """
    entries = list(entries)
    if not entries:
        return ("Sin bolsas de evaluación medidas: no hay vecinos que mostrar. "
                "No es una tabla vacía: es que esta lectura no corrió.")
    columns = ["Bolsa destino", "Clase", "Vecinos fuente (kernel, clase)"]

    def vecino(n) -> str:
        marca = "✓" if n["trueClass"] else "✗"
        return f"{n['sourceBag']} ({n['kernel']:.3f}, {marca})"

    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for entry in entries:
            vecinos = ", ".join(vecino(n) for n in entry["neighbours"])
            lines.append(f"| {entry['targetBag']} | {entry['targetLabel']} | {vecinos} |")
        lines += ["", "✓ marca un vecino de la misma clase que la bolsa de destino; "
                      "✗ de otra. El número entre paréntesis es el kernel de bolsa "
                      "de la Sección 3, no una distancia -- más alto es más cerca."]
        return "\n".join(lines)

    lines = [f"{'Bolsa destino':<15}{'Clase':<8}Vecinos fuente (kernel, clase)"]
    for entry in entries:
        vecinos = ", ".join(vecino(n) for n in entry["neighbours"])
        lines.append(f"{entry['targetBag']:<15}{entry['targetLabel']:<8}{vecinos}")
    return "\n".join(lines)


def conclusion_bag_neighbors(entries: Iterable[dict]) -> str:
    """Qué fracción del top-5 de cada bolsa de destino es de su propia clase."""
    entries = list(entries)
    if not entries:
        return "Sin bolsas medidas: no hay nada que concluir."
    total = sum(len(entry["neighbours"]) for entry in entries)
    correct = sum(1 for entry in entries for n in entry["neighbours"] if n["trueClass"])
    per_bag = [sum(1 for n in entry["neighbours"] if n["trueClass"]) / len(entry["neighbours"])
              for entry in entries if entry["neighbours"]]
    all_correct = sum(1 for share in per_bag if share == 1.0)
    none_correct = sum(1 for share in per_bag if share == 0.0)
    return (f"En {correct} de {total} vecinos (top-{len(entries[0]['neighbours'])}) el "
            f"kernel eligió una bolsa fuente de la misma clase. "
            f"{all_correct} de {len(per_bag)} bolsas de destino tienen el top-5 "
            f"entero de su propia clase; {none_correct} no tienen ninguno.")


def render_source_bag_usage(usage: dict, markdown: bool = False) -> str:
    """Por cada bolsa fuente, cuántas bolsas de destino la usaron -- la tabla inversa.

    `usage` es lo que devuelve `latent.source_bag_usage`: `{bolsa_fuente:
    cuenta}`, con cuenta cero incluida. Ordenada de menor a mayor uso, porque
    lo que esta tabla existe para exponer son las bolsas fuente que nadie usa,
    y esas son las que hay que ver primero.
    """
    if not usage:
        return ("Sin bolsas fuente medidas: no hay uso que mostrar. No es una "
                "tabla vacía: es que esta lectura no corrió.")
    ordered = sorted(usage.items(), key=lambda kv: (kv[1], kv[0]))
    unused = sum(1 for _, count in ordered if count == 0)
    columns = ["Bolsa fuente", "Bolsas destino que la usaron"]
    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for bag, count in ordered:
            lines.append(f"| {bag} | {count} |")
        lines += ["", f"{unused} de {len(ordered)} bolsas fuente no aparecen en "
                      f"el top-`k` de ninguna bolsa de destino."]
        return "\n".join(lines)

    lines = [f"{'Bolsa fuente':<14}Bolsas destino que la usaron"]
    lines += [f"{bag:<14}{count}" for bag, count in ordered]
    lines.append(f"\n{unused} de {len(ordered)} bolsas fuente no aparecen en el "
                 f"top-`k` de ninguna bolsa de destino.")
    return "\n".join(lines)


def conclusion_source_bag_usage(usage: dict) -> str:
    """Cuántas bolsas fuente quedan afuera de todo top-5, calculado y no leído a ojo."""
    if not usage:
        return "Sin bolsas fuente medidas: no hay nada que concluir."
    ordered = sorted(usage.items(), key=lambda kv: kv[1])
    unused = [bag for bag, count in ordered if count == 0]
    most_used_bag, most_used_count = max(usage.items(), key=lambda kv: kv[1])
    share = len(unused) / len(usage)
    return (f"{len(unused)} de {len(usage)} bolsas fuente "
            f"({share:.0%}) no aparecen en el top-`k` de ninguna bolsa de "
            f"destino. La más usada es la bolsa {most_used_bag}, en el top-`k` "
            f"de {most_used_count} bolsas de destino.")


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
        # `seconds` y `peakMiB` estaban en esta lista y no volvieron: eran las
        # dos dimensiones `perRun` de la declaración, y `conclusion` se niega
        # a agruparlas. La declaración de hoy ya no las lleva -- se retiraron
        # del banco entero, no sólo de este informe --, así que lo que queda
        # es lo que la declaración sí agrupa.
        for metric in ("sourceAccuracy", "targetAccuracy"):
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


# ------------------------------------------------------- el eje de contaminación

def conclusion_noise_floor(metric: str, arms: list[str] | None = None) -> str:
    """Cuánto cae la exactitud del piso entre el primer y el último nivel.

    Abre el informe: mide si el ruido perjudica incluso al método que no
    adapta. `arms=None` toma `sorted(set(config.FLOOR_OF.values()))` -- los
    pisos declarados hoy, nunca un id escrito acá -- así que una declaración
    con más de un piso concluye sobre todos, uno por frase.
    """
    from MIL_CREDA_Benchmark import contamination as noise_axis

    _refuse_pooling(metric)
    arms = sorted(set(config.FLOOR_OF.values())) if arms is None else list(arms)
    if not arms:
        return ("Ningún piso está declarado (`config.FLOOR_OF` está vacío): no "
                "hay brazo sobre el que leer esta curva.")
    rows = {row["arm"]: row for row in noise_axis.degradation(metric)}
    present = [arm for arm in arms if arm in rows]
    if not present:
        return ("Sin registro en el eje de ruido para ningún piso declarado: la "
                "curva no corrió todavía. No es una figura vacía: es que la "
                "corrida no existe.")
    decimals = 1 if metric in PERCENT else 2
    partes = []
    for arm in present:
        row = rows[arm]
        partes.append(
            f"**{row['name']}** cae de {_scaled(row['clean'], metric):.{decimals}f} "
            f"a {_scaled(row['worst'], metric):.{decimals}f} "
            f"({_scaled(row['fall'], metric):+.{decimals}f})")
    faltan = [arm for arm in arms if arm not in rows]
    cola = ""
    if faltan:
        nombres = ", ".join(config.NAME_OF.get(arm, arm) for arm in faltan)
        cola = f" Sin registro para {nombres}."
    return ("El ruido perjudica incluso al método que no adapta: "
            + "; ".join(partes) + "." + cola)


def conclusion_versus_clean(metric: str, rate: float) -> str:
    """Lo que el bloque contaminado dice y el limpio no puede decir.

    Deliberadamente no vuelve a enumerar la tabla de arriba. Una conclusión que
    repite los números que tiene al lado dejó de concluir: lo que agrega esta es
    la **resta** entre los dos bloques, que la tabla no hace.

    (Antes esta frase decía «lo que ninguna de las dos tablas contiene por
    separado». Las dos tablas son una sola desde que el ruido es una columna,
    así que lo que quedó en pie no es que el número esté repartido entre dos
    renderizaciones, sino que no está en ninguna: una tabla de niveles no
    resta.)
    """
    _refuse_pooling(metric)
    from MIL_CREDA_Benchmark import contamination as noise_axis

    # Las dos CAMPAÑAS, no el barrido: esta conclusión va debajo de una tabla de
    # seis transferencias, y leer la curva -- que corre una sola -- pondría dos
    # formas distintas a los dos lados de una resta.
    limpio = noise_axis.load(0.0, kind="campaign")
    sucio = noise_axis.load(rate, kind="campaign")
    if limpio is None or sucio is None:
        cual = "limpio" if limpio is None else f"ρ={rate:g}"
        return (f"Falta el registro {cual}, así que no hay comparación: una tabla "
                f"sola no dice cuánto se movió nada.")

    antes = noise_axis.by_arm(limpio["runs"], metric)
    despues = noise_axis.by_arm(sucio["runs"], metric)
    comunes = [arm for arm in config.ARM_ORDER if arm in antes and arm in despues]
    if not comunes:
        return ("Los dos niveles corrieron brazos disjuntos: no hay ningún par "
                "sobre el que restar.")

    decimals = 1 if metric in PERCENT else 2
    caidas = {arm: antes[arm] - despues[arm] for arm in comunes}
    menor = min(comunes, key=lambda a: caidas[a])
    mayor = max(comunes, key=lambda a: caidas[a])
    return (
        f"Contra el bloque `{NOISE_CLEAN}` de arriba, a ρ={rate:g} el que menos "
        f"pierde es "
        f"`{config.NAME_OF.get(menor, menor)}` "
        f"({_scaled(caidas[menor], metric):+.{decimals}f}) y el que más "
        f"`{config.NAME_OF.get(mayor, mayor)}` "
        f"({_scaled(caidas[mayor], metric):+.{decimals}f}). La tabla de arriba "
        f"pone los dos materiales en dos bloques; la resta entre ellos es lo "
        f"único que no dice."
    )


def conclusion_readings_versus_clean(limpias: Iterable[dict], sucias: Iterable[dict],
                                     path: str, rate: float) -> str:
    """Cuánto movió la contaminación una lectura de fase dos, brazo por brazo.

    No repite la tabla que tiene arriba. Lo que agrega es la **resta**: la tabla
    pone los dos materiales de cada brazo en dos bloques, y eso es todo lo que
    hace: quien quiera la distancia todavía tiene que restar dos números a ojo,
    y ahora con un bloque de por medio. Acá está hecha, y con el brazo que más se
    movió y el que menos nombrados. En fase dos importa más que en fase uno,
    porque una geometría que no se mueve bajo ruido y una exactitud que sí se
    mueve son dos hechos distintos sobre el mismo modelo.

    (Antes esta frase decía «lo que ninguna de las dos tablas contiene por
    separado». Las dos tablas son una sola desde que el ruido es una columna, así
    que lo que quedó en pie no es que el número esté repartido entre dos
    renderizaciones, sino que no está en ninguna: una tabla de lecturas no resta.)

    Sobre las medianas de cada celda y no sobre todo lo promovido: los extras
    fueron elegidos por el ordenamiento de los brazos dependientes, así que son
    una muestra sesgada de ese piso y promediarlos estima otra cosa.
    """
    def por_brazo(readings):
        reunido: dict[str, list[float]] = {}
        for reading in _own_medians(list(readings)):
            value = _reach(reading, path)
            if value is not None:
                reunido.setdefault(reading["arm"], []).append(value)
        return {arm: sum(v) / len(v) for arm, v in reunido.items() if v}

    antes, despues = por_brazo(limpias), por_brazo(sucias)
    comunes = [a for a in config.ARM_ORDER if a in antes and a in despues]
    if not comunes:
        return (f"No hay ningún brazo con lectura en las dos tasas, así que no "
                f"hay par sobre el que restar. Una punta sola no mide distancia.")

    movimientos = {a: despues[a] - antes[a] for a in comunes}
    quieto = min(comunes, key=lambda a: abs(movimientos[a]))
    movido = max(comunes, key=lambda a: abs(movimientos[a]))
    return (
        f"Entre ρ=0 y ρ={rate:g}, `{config.NAME_OF.get(movido, movido)}` es el que "
        f"más se mueve ({movimientos[movido]:+.4g}) y "
        f"`{config.NAME_OF.get(quieto, quieto)}` el que menos "
        f"({movimientos[quieto]:+.4g}). La tabla de arriba pone los dos "
        f"materiales en dos bloques; la resta entre ellos es lo único que no "
        f"dice."
    )


def render_correspondence_contaminated(scored: Iterable[dict], rate: float,
                                       markdown: bool = False) -> str:
    """Los mismos aciertos y la misma masa, sobre la campaña contaminada.

    Declarado aparte de `render_correspondence`: el chequeo de duplicación mira
    la llamada, y acá las dos mitades llamaban a la misma función sin nombrar
    ninguna dimensión, de modo que la contaminada se leía como una segunda
    renderización de la tabla limpia. No lo es: una mide sobre material limpio y
    la otra a ρ, y son dos números distintos. Nombrarlo aparte no esquiva el
    chequeo, dice lo que efectivamente hay, y lo deja intacto para el caso que sí
    tiene que atrapar.

    Las otras cinco tablas salieron de este arreglo: llevan ahora una columna
    `Ruido` con un bloque por material y una sola llamada por cantidad. Esta no
    se unificó --- sus filas son sujetos y no brazos, así que el bloque `con` no
    tendría con qué fila del `sin` aparearse --- y por eso el par de nombres
    sigue existiendo acá y sólo acá.

    La tabla es idéntica porque tiene que serlo: dos formas distintas para la
    misma cantidad obligarían al lector a traducir entre ellas para comparar,
    que es justamente lo que estas dos tablas existen para no pedir.
    """
    scored = list(scored)
    if not scored:
        return (f"La campaña a ρ={rate:g} todavía no dejó checkpoints, así que no "
                f"hay segunda tabla. No está vacía: no existe.")
    return render_correspondence(scored, markdown=markdown)


#: El peldaño del peso por confianza: el brazo sin pesar contra el que pesa.
#: Declarado acá y no adivinado de `ARMS`, porque «cuál es el par que difiere
#: sólo en el peso» es una lectura de la formulación y no una propiedad que se
#: pueda derivar de un diccionario.
def conclusion_with_noise(runs: Iterable[dict], metric: str, reduction: dict,
                          rate: float) -> str:
    """Dónde quedó cada método y cuánto lo movió la contaminación, junto.

    La primera mitad es `conclusion`: los extremos de la tabla y cada método
    contra su propio piso. La segunda es `conclusion_versus_clean`: la resta
    entre el bloque `sin` y el `con`, que es lo único que la tabla no hace.
    """
    return (f"{conclusion(runs, metric, reduction)} "
            f"{conclusion_versus_clean(metric, rate)}")


READING_CONCLUSIONS = {
    "geometry.ratio": conclusion_geometry,
    "domainSeparability": conclusion_separability,
    "correspondence.massOnTrueClass": conclusion_mass,
    "attentionSpread": conclusion_attention,
}


def conclusion_readings_with_noise(limpias: Iterable[dict],
                                   sucias: Iterable[dict],
                                   path: str, rate: float) -> str:
    """Qué dice una lectura de fase dos y cuánto la movió la contaminación.

    Se niega ante una ruta que no tiene conclusión limpia propia en vez de
    devolver sólo la cruzada: una conclusión que aparece a veces entera y a
    veces a medias se lee igual en los dos casos.
    """
    if path not in READING_CONCLUSIONS:
        raise KeyError(
            f"`{path}` no tiene conclusión limpia propia, así que no hay nada "
            f"que componer. Las dos distancias se concluyen juntas con "
            f"`conclusion_distances`; las rutas con conclusión propia son "
            f"{', '.join(sorted(READING_CONCLUSIONS))}.")
    limpias, sucias = list(limpias), list(sucias)
    limpia = READING_CONCLUSIONS[path](limpias)
    if not sucias:
        return (f"{limpia} Sin la corrida contaminada no hay resta que hacer: "
                f"la tabla de arriba lleva un solo bloque.")
    return (f"{limpia} "
            f"{conclusion_readings_versus_clean(limpias, sucias, path, rate)}")
