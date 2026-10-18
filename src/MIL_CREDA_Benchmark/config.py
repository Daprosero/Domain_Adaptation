"""Every number this comparison was decided with, in one place.

Two constants separate the pilot from the full configuration — `EPOCHS` and
`SEEDS` — and nothing else. That is deliberate: the pilot has to exercise the
same path the full run will, or it proves nothing about it.

    pilot:  EPOCHS = 3,  SEEDS = [0]
    full:   EPOCHS = 20, SEEDS = range(30)

The run count at either scale is `len(ARMS) * len(TRANSFERS) * len(SEEDS)` --
computed by `sizing()` below rather than typed here, because a number typed
beside this docstring would drift the day an arm is added or removed and
nothing would notice.

Read the header of any summary before reading its numbers. A table produced with
one seed carries a threshold of zero, so every row declares a winner from a bare
difference; the summary says so in its header and in its JSON, and that stamp is
the only thing standing between a pilot and a misquote.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

# ---------------------------------------------------------------- scale knobs

#: Passes over the source training bags. The lambda ramp is off during the first
#: epoch and effectively on afterwards, so three is the smallest count that still
#: trains with the adaptation term active.
EPOCHS = 3

#: What varies between repetitions: the stratified draw, the composition of the
#: bags, the split and the initialization of the head and the relevance selector.
SEEDS = [0]

#: The scale the verdict requires, declared separately from the scale running now.
#: Without both there is nothing to compare: a record of the pilot and a configuration
#: set to the pilot agree with each other, and everything reads as finished. With
#: both, a run below these is reported as a pilot and never quoted as a result.
#:
#: Thirty repetitions is what 36 evaluation bags need to separate three points. The
#: epoch count is the open question: the source term saturates by the third, so
#: anything past it buys ramp time for the adaptation rather than fit.
FULL_SEEDS = list(range(30))
FULL_EPOCHS = 20


def is_pilot_scale() -> bool:
    """Si la escala configurada ahora mismo es la del ensayo y no la completa.

    Una sola lectura de la regla que el encabezado de este modulo ya declara:
    dos constantes separan al ensayo de la corrida completa, `EPOCHS` y `SEEDS`,
    y ninguna otra. Escrita de nuevo en cada llamador serian dos ortografias de
    lo mismo, y la que quede vieja se lee igual de verde que la que no.

    Existe porque `Reduction.pilot` --- que decide DONDE escribe una corrida ---
    no se derivaba de la escala en ningun lado: `Benchmark_Campaign_v1.ipynb`
    construia su reduccion sin `pilot`, asi que una corrida de tres epocas y una
    semilla escribia en `Results/Benchmark/`, el arbol de la corrida completa, y
    sus numeros quedaban ahi para que alguien los citara. La docstring del campo
    ya decia que un registro que no dice que es de ensayo es exactamente como un
    numero de piloto termina citado como resultado.
    """
    return EPOCHS < FULL_EPOCHS or len(SEEDS) < len(FULL_SEEDS)


#: La variable de entorno que marca esta corrida como un ENSAYO REMOTO: el paso
#: corriendo a escala reducida en el worker, contra lo que los pasos anteriores
#: dejaron a escala COMPLETA, para saber que anda antes de gastar la cuota.
#:
#: Una variable de entorno y no una constante de este archivo porque el ensayo
#: no es un modo del REPOSITORIO sino de UNA ejecucion: el mismo commit clonado
#: en el worker corre primero el ensayo y despues la corrida real, y un archivo
#: editado entre las dos serian dos pines distintos --- o sea, dos codigos, y la
#: corrida real dejaria de ser la que el ensayo probo. `steps.ensayo_remoto` es
#: quien la pone, para el subproceso que ejecuta el cuaderno y para nadie mas.
REHEARSAL_ENV = "MIL_CREDA_REHEARSAL"


def is_rehearsal() -> bool:
    """Si esta ejecucion es un ensayo remoto.

    Leida del entorno en cada llamada y no fijada al importar: la suite prende y
    apaga el modo alrededor de un cuaderno que ya corrio otras veces en el mismo
    proceso, y un valor congelado al importar haria que el primer test decidiera
    por todos los demas.
    """
    return os.environ.get(REHEARSAL_ENV) == "1"


def upstream_pilot_scale() -> bool:
    """El `pilot` con el que se LEE lo que produjo un paso ANTERIOR.

    Dos coordenadas y no una, y hasta ahora era una sola: `is_pilot_scale()`
    decidia a la vez a que escala corre este paso y de que arbol lee lo que el
    anterior dejo. Mientras las dos preguntas viven en el mismo recorrido eso es
    correcto ---en ensayo todo corre y consume lo del ensayo--- y es exactamente
    lo que el recorrido local hace.

    El ensayo REMOTO es la otra combinacion, y no existia: corre a escala
    reducida ---asi cuesta minutos y no horas--- y tiene que consumir lo que los
    pasos anteriores dejaron a escala COMPLETA, porque de eso se trata. Un
    ensayo que corre contra las salidas de OTRO ensayo prueba que el cable lleva
    corriente y no prueba nada sobre la corrida que va a seguirlo: el archivo que
    la corrida real va a abrir es el completo, y es el unico que puede tener la
    forma equivocada, la version vieja o el brazo que falta.

    Y por eso lee `False` y nunca "el que rija": `contamination.in_force` cae al
    ensayo cuando no hay corrida completa, en silencio, y un ensayo que se apoya
    en esa caida pasa sin haber tocado nada de lo que dice probar. Quien se niega
    cuando la salida completa no esta es `steps.ensayo_remoto`, antes de abrir el
    cuaderno; esto solo dice de que arbol se lee.

    Fuera del ensayo remoto no cambia nada: es `is_pilot_scale()`, la misma
    lectura de siempre.
    """
    return False if is_rehearsal() else is_pilot_scale()


# --------------------------------------------------- la escala de la ejecucion
#
# `is_pilot_scale()` lee un dial del ARCHIVO -- `EPOCHS`/`SEEDS` -- y ese dial
# es una propiedad del REPOSITORIO, no de una corrida particular. Para el
# recorrido LOCAL eso es exactamente correcto: es la misma persona editando el
# mismo checkout entre un ensayo y el siguiente. Para un worker remoto es la
# falla que el comentario de `REHEARSAL_ENV` ya describe para su propia
# variable, un nivel mas abajo: el commit que la forja clono queda fijo desde
# que se lo pidieron, asi que ese commit no puede tener una escala distinta
# para el ensayo y para la corrida real sin dejar de ser el commit que se
# pidio. Un cuaderno que sólo sabe leer `is_pilot_scale()` lee "ensayo" en
# CUALQUIER worker -- hoy, `EPOCHS=3` y `SEEDS=[0]` son literales del archivo,
# y nada que la sumision declare puede moverlos -- asi que a escala completa
# los techos, la campaña y la comparacion de mecanismos venian de la
# biblioteca mientras el cuaderno que un lector abre nunca corria a esa
# escala. Las tres funciones de esta seccion son la lectura que falta: de que
# EJECUCION se trata, nunca de que archivo.

#: El directorio en el que la forja clono el commit fijado para ESTA
#: submision. `runner_invoke.py` lo exporta con este mismo nombre para el
#: kernel de cada cuaderno -- la celda 0 de cada uno ya lo lee para resolver
#: `ROOT`, con la misma disciplina de "ausente es LOCAL, nunca `se adivina`"
#: que esta lectura hereda en vez de reinventar. Su presencia es el unico
#: hecho que distingue un worker de un portatil: nadie mas la pone.
CLONE_ROOT_ENV = "FORGE_CLONE_ROOT"

#: El modo de ESTA submision, y no del repositorio -- `"smoke"` en un ensayo
#: remoto, ausente en una submision ordinaria. `runner_invoke.py`'s propio
#: `submission_environment()` documenta por que la ausencia nunca es una
#: cadena vacia: un `os.environ.get` tiene que poder distinguir "no se
#: declaro modo" de "se declaro y da falso", y una cadena vacia confundiria
#: las dos. Ausente NUNCA significa "local" -- sólo `CLONE_ROOT_ENV` decide
#: eso -- significa "esta submision, sea local o remota, no se declaro un
#: ensayo".
RUN_MODE_ENV = "FORGE_RUN_MODE"

#: Las unidades que ESTA submision posee: un arreglo JSON de strings, o la
#: variable ausente por completo. `runner_invoke.py` las serializa con
#: `json.dumps` en vez de unirlas con un separador por la razon que su propio
#: comentario da -- un separador invita la ambiguedad que un JSON ya
#: resuelve -- y la ausencia de la CLAVE, nunca el valor, es lo unico que
#: separa "no se declararon unidades" de "se declararon cero". Esta lectura
#: preserva esa distincion en vez de colapsarla, la misma disciplina que
#: `is_rehearsal()` ya aplica sobre su propia variable.
RUN_UNITS_ENV = "FORGE_RUN_UNITS"


def execution_is_pilot_scale() -> bool:
    """Si ESTA EJECUCION corre a escala de ensayo -- la pregunta que
    `is_pilot_scale()` no puede contestar por si sola en un worker.

    Tres casos, y los tres se distinguen:

    - `CLONE_ROOT_ENV` ausente: esto corre en el portatil de una persona, y
      el recorrido LOCAL no cambia ni un bit -- se repliega a
      `is_pilot_scale()`, exactamente la lectura de siempre. Abrir un
      cuaderno a mano hoy sigue haciendo lo mismo que hacia ayer.
    - `CLONE_ROOT_ENV` puesto y `RUN_MODE_ENV == "smoke"`: un ensayo remoto,
      la contraparte exacta de `is_rehearsal()` para la escala -- corre a
      escala de ensayo PORQUE lo que se esta probando es si el cable lleva
      corriente, nunca si el resultado se puede citar.
    - `CLONE_ROOT_ENV` puesto y cualquier otro modo, o ninguno: una submision
      ordinaria, remota y real, y la unica lectura que `is_pilot_scale()` no
      podia dar jamas desde un cuaderno -- escala COMPLETA, sin que nadie
      tenga que editar `EPOCHS`/`SEEDS` en el archivo para conseguirla.
    """
    if not os.environ.get(CLONE_ROOT_ENV):
        return is_pilot_scale()
    return os.environ.get(RUN_MODE_ENV) == "smoke"


#: La autorizacion EXPLICITA para que ESTA EJECUCION lance la rejilla
#: completa (30 semillas, 20 epocas) sin `CLONE_ROOT_ENV` -- o sea, sin que la
#: forja ya la haya autorizado por su cuenta con su propio token de un solo
#: uso. Una variable de entorno y no una constante de este archivo, por
#: exactamente la razon que el comentario de `REHEARSAL_ENV` ya escribio para
#: la suya: autorizar una campaña completa es un hecho de UNA EJECUCION,
#: nunca del repositorio. Una constante puesta en `True` seguiria diciendo
#: que si mucho despues de que la persona que la puso dejara de mirar la
#: maquina -- exactamente el pin editado entre el ensayo y la corrida real
#: que `REHEARSAL_ENV` rehusa ser. Puesta a mano, en el entorno del proceso
#: que lanza, para esa lanzada y ninguna otra.
LOCAL_FULL_SCALE_ENV = "MIL_CREDA_LOCAL_FULL_SCALE_AUTHORIZED"


def full_scale_launch_is_authorized() -> bool:
    """Si ESTA EJECUCION puede lanzar la rejilla completa (30 semillas, 20
    epocas) -- la puerta que ninguna funcion que recibe `pilot=` tenia, y que
    `is_pilot_scale()` no puede dar por si sola: es una comparacion contra
    las dos constantes de ESTE archivo, `EPOCHS`/`SEEDS`, y una llamada con
    `pilot=False` la deja de acuerdo con lo que acaba de pasar sin que nadie
    haya tocado el archivo.

    Dos casos, igual que `execution_is_pilot_scale()` distingue:

    - `CLONE_ROOT_ENV` puesto: esto corre en un worker, y la forja ya emitio
      su propio token de un solo uso ANTES de la sumision -- negar la corrida
      aca romperia exactamente el camino remoto que este proyecto persigue.
      Autoriza sin mirar `RUN_MODE_ENV`: un worker que efectivamente llama a
      esta funcion con `pilot=False` no esta ensayando -- un ensayo remoto
      prueba el cable con `run_smoke()`, nunca con la rejilla completa -- asi
      que no hay un segundo caso de worker que distinguir aqui.
    - `CLONE_ROOT_ENV` ausente: esto corre en el portatil de una persona, y
      nada afuera de este proceso miro la corrida. Autoriza solo
      `LOCAL_FULL_SCALE_ENV`, puesta a mano para esta ejecucion.
    """
    if os.environ.get(CLONE_ROOT_ENV):
        return True
    return os.environ.get(LOCAL_FULL_SCALE_ENV) == "1"


def transfer_label(transfer: tuple[str, str]) -> str:
    """The one spelling of a transfer used as a key anywhere.

    Written once because the record, the search's progress lines and the
    ceiling lookup all have to agree on it. Two of them agreeing and the
    third not would make a per-transfer ceiling silently fall back to the
    pooled one, which is the failure that reads as success: the run
    proceeds and reports a number.

    Moved down from `harness.py` (D3, layering): `config.py` imports
    nothing from the package, and `harness` imports `config` -- so the one
    spelling has to live here for `harness.transfer_label` to be a plain
    re-export rather than a second definition.
    """
    return f"{transfer[0]}->{transfer[1]}"


def _seed_units_from(crudo: str) -> "list[int]":
    """The seed axis's populated branch, byte for byte `execution_seed_
    units()`'s old body -- these messages predate the `axis` parameter and
    have to keep saying exactly what they said before it existed
    (Requirement "The seed axis keeps its exact existing behavior",
    `specs/run-unit-axis-resolution`)."""
    try:
        decodificado = json.loads(crudo)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{RUN_UNITS_ENV}={crudo!r} no es JSON valido; una submision "
            "que declara unidades tiene que declararlas en la forma que "
            "`runner_invoke.py` serializa, o esta lectura no puede saber a "
            "que semillas corresponden") from exc
    if not isinstance(decodificado, list):
        raise ValueError(
            f"{RUN_UNITS_ENV}={crudo!r} no es un arreglo; `runner_invoke.py` "
            "siempre serializa `units` como una lista de strings")
    semillas: list[int] = []
    for unidad in decodificado:
        if not isinstance(unidad, str):
            raise ValueError(
                f"la unidad {unidad!r} de {RUN_UNITS_ENV} no es un string; "
                "`runner_invoke.py` sólo serializa strings, y un elemento "
                "que no lo sea es una submision que no siguio su propio "
                "contrato")
        try:
            semillas.append(int(unidad))
        except ValueError as exc:
            raise ValueError(
                f"la unidad {unidad!r} de {RUN_UNITS_ENV} no es una semilla "
                "entera; este repositorio declara el eje de reparto como "
                "semillas y ninguna otra cosa, asi que una unidad que no lo "
                "sea no tiene una corrida que correr") from exc
    return semillas


def _transfer_units_from(crudo: str) -> "list[tuple[str, str]]":
    """The transfer axis's populated branch: every element has to be the
    canonical label `transfer_label()` produces (`"M->U"`), validated
    against `SEARCH_TRANSFERS` -- the closed set this search actually
    covers (D1). A positional index is deliberately not a valid unit here:
    `"0"` is a valid seed too, and the whole point of the label vocabulary
    is that a transfer unit is self-identifying across axes -- `int("M->U")`
    already raises, so a step that mixes up which axis it declared dies on
    the seed branch instead of quietly running the wrong search.
    """
    try:
        decodificado = json.loads(crudo)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{RUN_UNITS_ENV}={crudo!r} is not valid JSON; a submission "
            "declaring units has to declare them in the shape "
            "`runner_invoke.py` serializes, or this read cannot know which "
            "transfers it names") from exc
    if not isinstance(decodificado, list):
        raise ValueError(
            f"{RUN_UNITS_ENV}={crudo!r} is not an array; `runner_invoke.py` "
            "always serializes `units` as a list of strings")
    by_label = {transfer_label(t): t for t in SEARCH_TRANSFERS}
    transferencias: list[tuple[str, str]] = []
    for unidad in decodificado:
        if not isinstance(unidad, str):
            raise ValueError(
                f"the unit {unidad!r} of {RUN_UNITS_ENV} is not a string; "
                "`runner_invoke.py` only serializes strings, and an element "
                "that is not one is a submission that did not follow its "
                "own contract")
        if unidad not in by_label:
            raise ValueError(
                f"the unit {unidad!r} of {RUN_UNITS_ENV} is not a transfer "
                f"this search declares; available: {sorted(by_label)}. A "
                "unit read for the transfer axis is never coerced to the "
                "nearest valid label -- a misread transfer does not fail, "
                "it searches a different one and returns a number shaped "
                "exactly like the right one.")
        transferencias.append(by_label[unidad])
    return transferencias


def execution_units_for(axis: str) -> "list[int] | list[tuple[str, str]] | None":
    """`RUN_UNITS_ENV` for THIS submission, in the vocabulary of the axis
    the STEP declares -- received, never inferred from the content (D3).

    No default for `axis`: a caller that forgets to name one gets a
    refusal, never the seed behaviour this repository used to give
    unconditionally. `axis` is checked against
    `__benchmark__["distribution"]["axes"]` -- the declaration, not a local
    tuple -- so a step naming an axis nobody declared cannot fall through
    to seeds.

    The three-way distinction `execution_seed_units()` already made is
    preserved for every axis, not only the seed one:

    - Absent: this submission never declared units, so there is nothing to
      split. Returns `None`, read by the caller as "the whole grid" --
      exactly the default `seeds=None`/`transfers=None` already give.
    - Declared empty (`"[]"`): ZERO units were decided, a fact distinct
      from nobody deciding anything. Returns `[]`, a real list and never
      `None`.
    - Declared with content: every element is validated against THIS
      axis's own vocabulary -- an integer seed for `"seed"`, a canonical
      transfer label for `"transfer"`. Anything else -- invalid JSON, a
      non-list value, a non-string element, or a string outside the axis's
      vocabulary -- is REFUSED, never guessed: a misread unit does not
      fail, it runs a different experiment and returns a number shaped
      exactly like the right one.
    """
    from MIL_CREDA_Benchmark import __benchmark__
    axes = (__benchmark__.get("distribution") or {}).get("axes") or {}
    if axis not in axes:
        raise ValueError(
            f"refusing to resolve axis {axis!r}: it is not one of the axes "
            f"declared in `distribution.axes` ({sorted(axes)}); a step "
            "cannot read units for an axis nobody declared.")

    crudo = os.environ.get(RUN_UNITS_ENV)
    if crudo is None:
        return None

    if axis == "seed":
        return _seed_units_from(crudo)
    if axis == "transfer":
        return _transfer_units_from(crudo)
    raise ValueError(
        f"refusing to resolve axis {axis!r}: it is declared, but no unit "
        "vocabulary is wired for it yet.")


def execution_seed_units() -> "list[int] | None":
    """`RUN_UNITS_ENV` for THIS submission, in the seed axis's vocabulary --
    one unit IS a seed.

    A thin delegate to `execution_units_for("seed")` (D3): the three-way
    absent/declared-empty/populated distinction and every message it
    raises live there now, moved without editing a character.
    `run_campaign_shard`/`run_mechanism_sweep_shard` are the two in-repo
    callers, and neither observes a behaviour change.
    """
    return execution_units_for("seed")


# ------------------------------------------------------------------- material

REVISION = "research-concept-r33.md"

#: Every domain supplies its own bags; a transfer names which is source and which
#: is target. All three hold far more than the 3600 images a domain contributes.
#:
#: Nine more codes join the three digit domains below: `toy.py`'s own synthetic
#: manifolds, with no pretrained backbone to hide behind. Every measurement of
#: the adaptation term so far ran under `resnet18` -- 11M ImageNet-pretrained
#: parameters on 32x32 digit images -- which makes the objection that a bare
#: normalization statistic is already sufficient unanswerable at that cost: the
#: backbone is strong enough to absorb almost anything. These three toy families
#: remove it. Lowercase is a source, uppercase is a translation-only target, and
#: the digit suffix is a translation+rotation+ambient-noise target -- see
#: `TOY_TRANSFERS` for which pairs are declared, and `toy.py` for the generator.
DOMAINS = {
    "M": "MNIST", "U": "USPS", "S": "SVHN",
    "r": "SwissRoll", "R": "SwissRollTranslated", "1": "SwissRollRotatedNoisy",
    "t": "Tori", "T": "ToriTranslated", "2": "ToriRotatedNoisy",
    "n": "Moons", "N": "MoonsTranslated", "3": "MoonsRotatedNoisy",
}

TRANSFERS = [("M", "U"), ("U", "M"), ("M", "S"), ("S", "M"), ("U", "S"), ("S", "U")]

#: The toy counterpart of `TRANSFERS`: one source-to-target pair per manifold
#: family, never mixed across families (a swiss-roll source has no business
#: adapting to a moons target). `harness.py`'s digit campaign reads `TRANSFERS`
#: alone and never this list, so declaring it here changes nothing about that
#: campaign -- these nine domain codes exist for `toy.py`/`bags.py` and the
#: tests that exercise them, not for the digit grid.
TOY_TRANSFERS = [("r", "R"), ("r", "1"), ("t", "T"), ("t", "2"), ("n", "N"), ("n", "3")]

CLASSES = 10

#: A bag is a subject: 30 instances of one class, drawn at random inside it. No
#: instance carries a label of its own, and no instance comes from another class.
INSTANCES_PER_BAG = 30

#: Twelve per class, so the draw is stratified and balanced rather than
#: proportional. The local correspondence is undefined for a class with no source
#: bag, so class coverage is a requirement of the formulation and not a
#: convenience.
#:
#: It was ten until the ceiling search needed a role of its own. The two extra
#: bags a class contributes fund that role outright, so nothing was taken from
#: training or from the verdict: 64 / 20 / 36 where it used to be 64 / — / 36.
#: The material allows it — the domains hold far more than they contribute, and
#: USPS is the one that binds at 542 images in its smallest class, which is 18
#: bags of thirty. Twelve sits under that with room, and the other two domains
#: are not close to a limit.
BAGS_PER_CLASS = 12
BAGS_PER_DOMAIN = BAGS_PER_CLASS * CLASSES          # 120
IMAGES_PER_DOMAIN = BAGS_PER_DOMAIN * INSTANCES_PER_BAG  # 3600

#: Three roles, drawn identically in both domains, and disjoint.
#:
#: Training fits. Selection is where the ceiling search looks, and it exists
#: because the search chooses by outcome: a coefficient picked on the material the
#: verdict is read from makes the verdict read a decision it already made. The
#: evaluation role is never seen before the verdict.
#:
#: The 36 evaluation bags are what let 30 seeds resolve three points, and they are
#: untouched — the selection role is funded by the two extra bags per class, not
#: taken from anywhere. An 80/20 would have left 20 and resolved five.
#:
#: Twenty selection bags across two search transfers and 30 seeds is 60
#: measurements per ceiling, which separates about three points between grid
#: points. That is enough to pick one scalar out of five and not enough to be
#: read as a result, which is exactly what it is for.
TRAIN_BAGS = 64
VALID_BAGS = 20
EVAL_BAGS = 36

# -------------------------------------------------------------- toy domains

#: Deterministic from the domain code alone, never from a campaign's own
#: `seed` -- `bags.rebuild` reconstructs a toy domain from `(domain, seed)`
#: it is given in the manifest plus whatever `toy.py` generates, and the
#: generator must therefore answer the same way on every call regardless of
#: which seed a particular bag draw happened to use. One shared constant
#: rather than one per family: the three generators consume independent
#: `numpy.random.Generator` streams of their own (offset by family and by
#: domain code), so a single seed cannot make two families collide.
TOY_SEED = 20260925

#: How many points `toy.py` generates for each of a class's `BAGS_PER_CLASS`
#: subjects. Forced by the funding arithmetic `bags._select` already enforces
#: for the digit domains, carried over unchanged: a class must offer at least
#: `BAGS_PER_CLASS * INSTANCES_PER_BAG` images for its own bags plus
#: `bags._reserve_size(NOISE_CAP)` held back for contamination at the top
#: rung. With `INSTANCES_PER_BAG = 30`, `BAGS_PER_CLASS = 12`,
#: `NOISE_ROLES = ("train", "eval")` and `NOISE_CAP = 0.5`:
#:
#:     noise_instances(0.5) = 15 per bag
#:     _noised_bag_count() = TRAIN_BAGS + EVAL_BAGS = 64 + 36 = 100
#:     reserve = ceil(100 * 15 / CLASSES) + INSTANCES_PER_BAG = 150 + 30 = 180
#:     needed per class = BAGS_PER_CLASS * INSTANCES_PER_BAG = 360
#:     360 + 180 = 540, spread over 12 subjects = 45 per subject, exactly
#:
#: So 45 is the floor, not a margin above it: a smaller value refuses mid-draw
#: at the top contamination rung, which `test_toy_domains.py` pins directly
#: against `bags._reserve_size(NOISE_CAP)` rather than against this number
#: restated. `toy.py` never hardcodes the per-class total either -- it reads
#: `BAGS_PER_CLASS * TOY_POINTS_PER_SUBJECT`, so the two cannot drift apart.
TOY_POINTS_PER_SUBJECT = 45

#: The constant added to every ambient coordinate of a translation-only (`R`,
#: `T`, `N`) or translation+rotation+noise (`1`, `2`, `3`) target, relative to
#: a source domain. Large enough that an unadapted model genuinely degrades:
#: each class is one equal-width band of the manifold parameter (10 bands over
#: roughly `3*pi ~= 9.4` for the swiss roll, `pi` for one moons branch, `2*pi`
#: for one torus's major angle), so a shift of this size is several multiples
#: of a single class's own band width in every family -- large enough to move
#: a point's *apparent* position past its neighbouring classes' bands, which
#: is exactly the failure an unadapted encoder cannot see past. A per-coordinate
#: constant on purpose: `BatchNorm1d`'s running mean can absorb exactly this,
#: which is the whole point of declaring `R`/`T`/`N` as translation-only targets
#: a normalization statistic alone should already repair.
TOY_TRANSLATION_SCALE = 5.0

#: The angle `toy.py` rotates a translation+rotation+noise target (`1`, `2`,
#: `3`) by, about the manifold's own "width" axis (the swiss roll's height, a
#: torus's minor angle) so the rotation mixes exactly two of the ambient
#: coordinates and leaves the third untouched. 45 degrees: a real mixing
#: rather than a near-identity or a near-reflection, and -- the reason this
#: axis exists at all -- NOT expressible as a per-coordinate affine map.
#: `BatchNorm1d` is one scale and one shift per coordinate; a rotation matrix
#: with a nonzero off-diagonal block cannot be written that way at any angle
#: other than a multiple of 90 degrees, and 45 sits exactly at the point of
#: maximum mixing between those multiples.
TOY_ROTATION_ANGLE = math.pi / 4

#: The standard deviation of the ambient Gaussian noise `toy.py` adds, per
#: coordinate, only to a translation+rotation+noise target (`1`, `2`, `3`).
#: Modest relative to a class's own band width in every family (a few percent
#: of `TOY_TRANSLATION_SCALE`): enough to blur the manifold -- so the rotated
#: target is not merely a noiseless rigid copy of the source -- without
#: destroying the class structure the band assignment depends on, which the
#: generator itself never re-derives after translating/rotating a point.
TOY_NOISE_SIGMA = 0.3

# ---------------------------------------------------------------- label noise

#: The fraction of a training bag's instances replaced by images of another
#: class. The bag's label never changes: bags are pure and no instance carries a
#: label of its own, so there is nothing to flip. What contamination corrupts is
#: the evidence, not the answer.
#:
#: The same replacement is two different perturbations, and that asymmetry is the
#: experiment rather than a wrinkle in it. `wiring.py` broadcasts the bag's label
#: to all `INSTANCES_PER_BAG` instances, so for an instance-unit arm those
#: replaced instances carry a genuinely wrong label; for a bag-unit arm the label
#: stays at the bag and the replacements are witnesses the attention may learn to
#: downweight.
NOISE = 0.0

#: Every level the axis runs over, `0.0` first. The clean campaign is the first
#: point of the curve rather than a separate document — a second set of notebooks
#: differing in one parameter forks from the first day. The last level, `0.5`,
#: is the tie `NOISE_CAP` admits: under the single donor `NOISE_PAIRING`
#: declares for each label, the bag's own class and its paired donor land at
#: exactly fifteen instances each -- the rung where the label stops being
#: decidable from composition alone, which is the error this axis exists to
#: force rather than a side effect of raising the rate far enough.
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]

#: The one donor class each label's bags are contaminated from. An involution
#: on the ten classes with no fixed point -- five mutual pairs -- and DERIVED
#: from confusability in the data rather than chosen from a prior about which
#: digits look alike, and never from a trained model, because a declaration
#: must not depend on a run.
#:
#: Two independent class-centroid cosine proxies decided it: raw pixels
#: exactly as the bags hand them to the encoder (no model needed at all), and
#: the declared backbone's UNTRAINED ImageNet features (closer to what the
#: encoder actually sees, at the cost of inheriting the backbone choice). Each
#: proxy is normalized per domain and averaged over the three domains, and the
#: matching itself is solved exactly rather than approximated -- all 945
#: perfect matchings on ten classes, enumerated.
#:
#: The two proxies disagree on their own optimum, so neither alone is
#: defensible: the backbone proxy's own winner falls to rank 754 of 945 under
#: pixels, and declaring it would have declared the proxy rather than the
#: confusability. The averaged optimum below is the only matching near-optimal
#: under BOTH -- ranks 2 and 19 of 945 -- and it stays rank 2 under pixels
#: alone, which matters because that proxy needs no backbone while three are
#: declared. It carries the two most confusable pairs of the averaged matrix,
#: 3-5 (cosine distance 0.0546) and 7-9.
#:
#:     0-6   1-4   2-8   3-5   7-9
NOISE_PAIRING = [6, 4, 8, 5, 1, 3, 0, 9, 2, 7]

#: Past this the donor class would outnumber the bag's own and the label would
#: stop being defensible. Under the single donor `NOISE_PAIRING` declares for
#: each label, the two counts a composition reader faces are exact:
#: `rate * INSTANCES_PER_BAG` for the donor against
#: `(1 - rate) * INSTANCES_PER_BAG` for the bag's own class, so plurality
#: breaks exactly at one half rather than approximately -- a claim this
#: docstring made too early once already, back when the donor was still spread
#: across the other nine classes and the plurality in fact only broke at rho
#: 0.90. `rate <= NOISE_CAP` admits the tie itself: at 0.5 the two counts are
#: equal and the bag's own class is still half the evidence, not yet the
#: minority. Declared as a cap and enforced, not left to whoever edits
#: `NOISE_LEVELS` next.
NOISE_CAP = 0.5

#: Which roles the noise reaches: `train` and `eval`, in both domains, with ONE
#: shared draw across every arm -- the same instances are replaced for `B`, `E`,
#: `F` and `G` alike, because the draw is a property of the material
#: (`bags.build` is called once per domain/seed and the same `BagSet` is handed
#: to every arm) and never a property of which arm is training.
#:
#: `valid` stays clean, and the reason is not that contaminating it would corrupt
#: a measurement: it is that nothing reads it under noise. The search runs always
#: on clean material and its values are used unchanged when the material is
#: corrupted, so a contaminated `valid` would be drawn and never opened. `eval`
#: IS contaminated, because the sharpest half of this question is whether the
#: method still decides when the evidence it is judged on is corrupted too.
#:
#: One rate for source and target alike. The two are not the same perturbation --
#: the target trains unsupervised, through `pseudolabel` (Eq. 22) and
#: `confidences` (Eq. 24), so contaminating it corrupts the conditional the
#: adaptation term aligns to rather than any label -- but separating the rates
#: would make the sweep two-dimensional and multiply a campaign that already
#: costs `len(ARMS) * len(TRANSFERS) * len(FULL_SEEDS)` runs. Which of the two
#: domains hurts more is a rung of its own, later, on one transfer.
NOISE_ROLES = ("train", "eval")


def noise_instances(rate: float) -> int:
    """How many of a bag's instances a rate replaces.

    Exact at every declared level: the levels are tenths and a half, and
    `INSTANCES_PER_BAG` is thirty, so nothing rounds. Refuses past the cap
    rather than clamping, because a run silently held at 0.5 while its record
    says 0.7 is a table nobody can attribute. The cap itself is admitted --
    `<=`, not `<` -- because at the tie the bag's own class is still half the
    evidence, which is defensible; one step past it the donor would be the
    majority and it would stop being.
    """
    if not 0.0 <= rate <= NOISE_CAP:
        raise ValueError(
            f"noise rate {rate} is outside [0, {NOISE_CAP}]; past the cap the "
            f"donor class would outnumber the bag's own and its label would "
            f"stop being defensible"
        )
    return round(rate * INSTANCES_PER_BAG)


# --------------------------------------------------- the noise axis, on report

#: The contaminated level the report and latent notebooks show beside `0.0`. Both
#: render each table twice rather than once at a level chosen afterwards: picking
#: it once the degradation curve exists would put whichever level flatters the
#: method into the headline table, chosen by outcome. The midpoint of
#: `NOISE_LEVELS` is arithmetic, and nothing the run produces can have decided it.
NOISE_REPORTED = NOISE_LEVELS[len(NOISE_LEVELS) // 2]

#: Where the degradation curve is measured. One transfer, and the smallest domain
#: gap rather than the best result: the rule is about the instrument, since a
#: transfer already near its floor at `0.0` has no room to fall and cannot show a
#: curve. The gap is a property of the material and not of any measurement.
NOISE_TRANSFER = ("M", "U")

# The noise diagnostic -- the re-search that separated *the term failed* from
# *the coefficient was too small* -- is removed. Reason: it re-searched the
# ceiling under contamination, which contradicts the decision that the search
# always runs on clean material (see `harness.search_ceilings`'s own doctrine
# and `__benchmark__["search"]`'s `what`). Its config entries
# (`NOISE_DIAGNOSTIC_ARMS`, `NOISE_DIAGNOSTIC_LEVEL`), its steps
# (`noise-diagnostic`, `noise-diagnostic-report`) and its notebooks are gone with
# it.

# -------------------------------------------------------------------- network

#: `"resnet18"`/`"resnet50"`/`"vit_tiny_patch16_224"` build `CREDA.models.
#: FeatureExtractor` exactly as before; `"mlp"`/`"mlp-nobn"` (`mlp.py`, read by
#: `wiring._resolve_encoder`) build a small plain MLP instead, for the toy
#: domains above -- the two differ in EXACTLY their `BatchNorm1d` layers and
#: nothing else, which is the whole instrument this pair of backbones exists
#: to be: if a normalization statistic alone already recovers a translated
#: target, `"mlp"` should and `"mlp-nobn"` should not.
BACKBONE = "resnet18"
PRETRAINED = True          # ImageNet weights, identical on both sides

#: The MLP backbones' hidden width. Deliberately small -- toy domains are meant
#: to be solved by the method's own kernels, never brute-forced by backbone
#: capacity the way an 11M-parameter ImageNet resnet can.
#:
#: Thirty-two, and the number is MEASURED rather than chosen. Swept 8/16/32/64/128
#: over three seeds, reading source-against-source accuracy for BOTH backbone
#: variants, since what makes the pair a fair instrument is that NEITHER variant
#: is left unable to learn its own domain. Criterion: maximise the WORSE of the
#: two, breaking ties on the smaller gap. Measured (worse-of-two, by width):
#:   toroides  0.852 / 0.917 / 0.926 / 0.907 / 0.917
#:   rollo     0.806 / 0.889 / 0.880 / 0.870 / 0.815
#:   lunas     0.806 / 0.815 / 0.824 / 0.796 / 0.787
#: So 8 is too small and 128 too large in all three, while 16, 32 and 64 are
#: indistinguishable inside a dispersion of +-0.013 to +-0.082. One width for all
#: three columns rather than three per-column values: nothing in the measurement
#: justifies differentiating them, and a single value removes a free parameter
#: instead of adding three decisions to defend. Thirty-two is the toroides' own
#: best and the other two accept it at no measurable cost.
#:
#: The gap between the two variants runs -0.065 to +0.074 across all fifteen
#: (dataset, width) cells with no systematic direction, which is the evidence
#: that giving both the identical architecture is fair: the variant without
#: BatchNorm is not handicapped at LEARNING, only at transferring.
MLP_HIDDEN_WIDTH = 32

#: The MLP backbones' `output_dim`, read by every caller that already reads
#: `FeatureExtractor.output_dim` (`Arm.head`, `self.creda`'s instance reshape,
#: and `Arm.encoder.output_dim` elsewhere) -- half the hidden width, for the same
#: reason `MLP_HIDDEN_WIDTH` is small.
MLP_OUTPUT_DIM = 16

#: Cada cuantos pasos del optimizador los parametros de `theta_s` se transmiten
#: a `theta_t`. Procedimiento de ENTRENAMIENTO, no matematica: la Ec. (13) sigue
#: declarando dos codificadores que no comparten ningun tensor, y despues de
#: cada transmision vuelven a divergir. La decision y sus mediciones estan en
#: `AGREED.md`; esto es solo el intervalo.
#:
#: **Declarado fijo en vez de buscado, y la razon es medida** --- pero lo que se
#: midio es el NUMERO DE TRANSMISIONES, no el intervalo, y la diferencia importa:
#: un intervalo cuenta pasos, y cuantos pasos tiene una corrida depende de su
#: escala. A 20 epocas hay 140 pasos y a 3 hay 21, con
#: `TRAIN_BAGS`/`BAGS_PER_STEP` de hoy.
#:
#: A 20 epocas los intervalos 1, 7, 35 y 70 midieron 1.0000 los cuatro (2 a 140
#: transmisiones) contra 0.2500 sin transmitir. A 3 epocas, barrido sobre los
#: cuatro intervalos que ALCANZAN a disparar:
#:
#:     transmisiones    U->M      M->S
#:               21    0.9444    0.5278
#:                7    0.9167    0.4722
#:                3    0.8889    0.5000
#:                1    0.2500    0.0833   <- una sola, y en el ultimo paso
#:                0    0.1111    0.0833   <- control
#:
#: De 3 a 21 transmisiones el resultado es plano y ni siquiera monotono (en M->S
#: tres le gana a siete), asi que el intervalo exacto no es lo que decide. Lo que
#: decide es que haya MAS DE UNA y que ocurran DURANTE el entrenamiento: una sola
#: en el ultimo paso no compra nada en M->S y casi nada en U->M. El umbral esta
#: entre 1 y 3 transmisiones.
#:
#: Siete es un paso por epoca con `TRAIN_BAGS`/`BAGS_PER_STEP` de hoy, elegido
#: por ser el mas legible de la region plana, nunca por ser mejor. **Y es el
#: borde de esa region a escala de piloto**: tres transmisiones. Bajar `EPOCHS`
#: por debajo de 3 dejaria menos, y un intervalo mayor a los pasos totales no
#: transmite NUNCA sin que nada avise.
#:
#: La condicion que el mecanismo necesita es que los dos codificadores tengan la
#: MISMA ARQUITECTURA, no que reciban la misma entrada: medido sobre `resnet18`,
#: las sesenta formas de parametros son identicas con entradas de 32, 64 y 96
#: pixeles, porque la red cierra con `AdaptiveAvgPool2d`. Un `vit_tiny` cambiaria
#: un solo tensor, `pos_embed`, que depende de la cantidad de parches.
SOURCE_TO_TARGET_SYNC_STEPS = 7
FEATURE_DIM = 512          # resnet18's pooled width, the head's input

#: Hidden width of the relevance selector's R_phi of Eq. (15). It has no
#: counterpart on the other side, so it is declared and never tuned.
ATTENTION_WIDTH = 128

#: gamma of Eq. (15): the weight of the within-bag consensus term against the
#: learned relevance R_phi. The proposal fixes it as a hyperparameter during
#: training but gives it no value. Zero sits at the neutral where Eq. (15)
#: reduces to R_phi with the l1-normalized v_R (r25 Eq. (15)) -- not to a bare,
#: unconstrained v_R^T tanh(...), which is what a prior revision's attention
#: computed and which this one no longer does at any value of gamma. The
#: l1-ball reparametrization is not gated by gamma; only the consensus term
#: is. It stays tunable and is not asserted as the last word on it.
ATTENTION_GAMMA = 0.0

#: tau_att of Eq. (16): the attention temperature. The proposal fixes it as a
#: hyperparameter during training but gives it no value. One sits at the
#: neutral where Eq. (16) reduces to a plain softmax of Eq. (15)'s logit (no
#: temperature scaling at all) -- and, at ATTENTION_GAMMA = 0.0 as well, of
#: R_phi with the l1-normalized v_R alone, per the note above. It stays
#: tunable and is not asserted as the last word on it.
ATTENTION_TEMPERATURE = 1.0

#: The local temperature of Eq. (28). Also without a counterpart in CREDA, so it
#: is fixed at one and reported as fixed rather than chosen.
TAU_LOCAL = 1.0

#: sigma of Eq. (14): the one bandwidth of the one instance kernel. r25
#: Eq. (23) is explicit that a single sigma governs the three blocks of a
#: class's mixed matrix "ya que los tres derivan del mismo kernel de
#: instancia", and r25 Eq. (15) says the attention consensus reuses that
#: same kernel and its bandwidth rather than a second one of its own -- so
#: one constant serves the consensus term inside the attention logit (Eq. 15),
#: the top-k selection ranking, every bag kernel block Eq. (23) builds
#: (K_ss, K_st, K_tt) -- the three blocks of a class's mixed matrix are
#: defined there, not at Eqs.
#: (17)-(18), which define the bag kernel and its representation in general --
#: and the local correspondence's own kernel evaluations (Eq. 28, 31). Every
#: caller passes this explicitly, with no default anywhere in the call
#: chain -- a constant carries no gradient, so there is no question of it
#: being learned, and the value below is not asserted as final.
#:
#: Measured once, by the median heuristic prior work already applies per
#: batch (`CREDALoss._compute_sigma`, and MIL-CREDA's own former per-call
#: rule before this constant replaced it): sqrt(median(||h_a - h_a'||^2)
#: + 1e-6) over the off-diagonal pairwise squared distances of embeddings
#: h = F_theta(x), computed on the pilot configuration's source training
#: material -- domain M (MNIST), seed SEEDS[0] = 0, the 64 TRAIN_BAGS x 30
#: INSTANCES_PER_BAG = 1920 images that role draws -- in the regime training
#: actually uses: the encoder in TRAINING mode (so BatchNorm normalizes with
#: each chunk's own batch statistics rather than accumulated running ones),
#: passed through in BAGS_PER_STEP = 10-bag / 300-image chunks, walked in
#: `train_idx`'s own sequential order -- not the class-stratified shuffle
#: `training_step` actually draws its batches in via `balanced_batches`, which
#: this measurement did not reproduce. An earlier measurement of this same
#: constant ran the encoder in
#: eval mode instead -- accumulated running statistics rather than the batch
#: statistics training actually normalizes with -- and got 8.709090275783575
#: from a median squared distance of 75.848252431748; measured in the regime
#: above, the median squared distance is 1305.7392578125 and the constant
#: below follows from it. This is a placeholder to be tuned with Optuna
#: alongside the ceiling search -- not added to that search by this stretch
#: of work, which was not asked to make that change.
#:
#: `tools/measure_kernel_sigma.py` is this procedure, committed, not only
#: described: seed the global generator with `SEEDS[0]`, build the clean
#: material for domain M, build any arm (the encoder is constructed first,
#: before any arm-specific parameter, so which arm is irrelevant), leave the
#: model in training mode, walk the 64 training bags through
#: `instance_embeddings` in `BAGS_PER_STEP`-bag chunks in `train_idx`'s own
#: order under `torch.no_grad()`, concatenate the 1920 embeddings, and take
#: `sqrt(median(off-diagonal squared distance) + 1e-6)` directly over that set
#: -- every tensor left in the float32 dtype `bags.build` and an ordinary
#: campaign both already use, never upcast to this package's own
#: `MIL_CREDA_Benchmark.DTYPE`, which governs a different module's internal
#: tensors and not what a campaign trains with. Run today it measures
#: 36.135013580322266, 7.245e-07 away from the value below; stable across
#: repeated runs on this machine, so the gap is not run-to-run noise here, but
#: a gap this small is exactly what unordered floating-point summation inside
#: a multi-threaded BLAS matrix multiply can produce between environments
#: without the procedure itself differing. The script prints both numbers
#: rather than asserting they must agree.
KERNEL_SIGMA = 36.135014304860874

#: Whether the instance kernel kappa^I (Eq. 14) evaluates on the L2-normalized
#: instance embedding rather than the raw one `Arm.instance_embeddings` hands
#: it. `KERNEL_SIGMA` above is a constant CHOSEN BEFORE TRAINING, and the
#: kernel's regime is set by the ratio of the embedding's own scale to that
#: constant -- a ratio training is free to move. Measured on toy datasets this
#: session: the trained encoder's median instance-embedding distance ended at
#: 7.03 with BatchNorm and 38.97 without -- same architecture, same data, a
#: 5.5x drift -- and two of six cells ended with a dead or saturated kernel
#: because of it.
#:
#: Normalizing bounds ||h - h'||^2 to [0, 4] by construction, which pins the
#: scale this sigma governs. But a global normalization would also change
#: what the classification head eats (Eq. 19's `z = sum_a beta_a h_a`, read
#: by `self.head`), and the head is the one consumer that wants the scale
#: free -- so this flag governs only the kernel's own view of the embedding;
#: see `Arm._kernel_embedding` for the fork and every routed call site.
#:
#: Run-level, never per-arm: a per-arm flag would let two adjacent rungs of
#: the ablation ladder differ in the component under test AND in the
#: normalization, so no difference between them could be attributed to
#: either one -- the same confound this repository already walked into once
#: with the learning rate.
#:
#: Defaulted to the CURRENT behaviour (off, not normalized) so every existing
#: number stays reproducible and the full suite stays green with no flag set;
#: `test_kernel_normalization.py`'s own bit-identical-with-the-flag-off test
#: is what a later change to this default has to keep passing.
NORMALIZE_KERNEL_EMBEDDING = False

#: The stabilizer inside a logarithm, shared by two call sites rather than
#: private to either: `wiring.py` passes it as eps_src, `source_loss`'s (and
#: `source_bound`'s) own stabilizer for Eq. (21) -- what fixes B_src's scale --
#: and, for an instance-unit arm's `forward`, it also keeps the averaged
#: instance distribution off zero before it is logged. A prior version of this
#: comment said Eq. (21) "normalizes its own", which read as though this
#: constant played no part in it; it does, at the one call site that matters:
#: `wiring.py`'s `source_loss(..., config.EPSILON)`.
EPSILON = 1e-8

#: The stabilizer of the DISPERSION probe's divisor, and nothing else.
#:
#: **It was `eps_loc`, Eq. (38)'s own stabilizer, until 2026-10-18.** r32
#: replaced that equation's confidence-weighted quotient with the uniform mean
#: `(1/N_t) sum_j l_loc,j`, so there is no `sum_j w_j^t` left to keep away from
#: zero and `eps_loc` left the formulation with it. `local_loss` no longer
#: takes an epsilon at all, which is why no call site passes this any more.
#:
#: What still reads it is `wiring._dispersion_normalized_local_term`, where the
#: divisor is a MEASURED dispersion that genuinely can reach zero -- that is
#: the regime the probe exists to visit. Kept as its own name rather than
#: folded into `EPSILON`: that one stabilizes Eq. (21)'s logarithm, this one a
#: probe's denominator, and nothing requires them to move together.
EPSILON_LOCAL = 1e-8

#: Whether the local term's first half (Eq. (38)'s `l_loc,j`) normalizes
#: `d_j^2 = ||Psi(B_j^t) - mu_j^s||^2` by the target representations' own
#: MEASURED DISPERSION instead of the revision's constant 2.
#:
#: r25:1047 proves `<Psi, mu> > 0` (every inner product in H_I is strictly
#: positive and both are convex combinations with non-negative weights),
#: hence `d_j^2 < 2` -- the tight bound `normalized_local_distance` divides
#: by. That bound stops the term from DIVERGING; it does nothing to stop it
#: from DEGENERATING. Minimizing `d_j^2` alone has a degenerate global
#: minimum: collapse every bag representation to one point in H_I and
#: `d_j^2 -> 0` for every bag, a perfect local score carrying zero
#: information. Dividing by a constant does not move where that minimum
#: sits -- only WHAT is in the denominator can.
#:
#: Measured, without BatchNorm, on the target embedding's covariance
#: participation ratio (1 = all variance in one direction, 16 = isotropic):
#:
#:              B      E      F      G
#:     rollo    2.09   2.09   2.09   1.74
#:     toroides 2.55   2.56   2.56   1.07
#:     lunas    1.32   1.32   1.32   1.15
#:
#: `E` and `F` leave it bit-identical to the floor; only `G` -- the arm
#: carrying the local term -- lowers it, and on the rollo it does so while
#: LOSING transfer accuracy against the floor (0.4722 -> 0.2222). The
#: remedy this flag probes: normalize `d_j^2` by the bags' actual spread
#: rather than the constant bound, so collapse shrinks the numerator and
#: the denominator together instead of paying for free.
#:
#: **The denominator carries gradient. It must.** `wiring._target_dispersion`
#: is NOT detached: with `L = d_j^2 / D` and `D` detached, descent on `d_j^2`
#: alone is scaled by `1/D`, which EXPLODES as the cloud collapses (`D -> 0`)
#: -- a detached denominator is a positive feedback loop into the exact
#: collapse this flag exists to resist, not merely an inert rescaling. Live,
#: a second gradient term appears that pushes to INCREASE `D`; that term is
#: the entire remedy. See `wiring._target_dispersion`'s own docstring for the
#: full derivation, and `tests/test_local_dispersion.py`'s own
#: `test_the_dispersion_denominator_carries_gradient_and_resists_collapse`
#: for the measurement.
#:
#: **This is a PROBE on the harness side, not a change to the mathematics.**
#: `MIL_CREDA.local_term.normalized_local_distance` is never touched, stays
#: bound to Eq. (38), and keeps running by default (flag off). And the cost
#: is paid in the open, not hidden behind the probe succeeding: dividing by
#: a MEASURED dispersion instead of the declared constant 2 LOSES Eq. (38)'s
#: `[0, 1]` guarantee -- the ratio is unbounded above whenever the measured
#: dispersion is small, which is exactly the regime a probe of collapse has
#: to visit. `wiring._milcreda_term` restates this at the call site and
#: applies no clamp; if this probe ever earned a remedy, the lost bound is
#: part of what a deliberation would have to decide about, not a detail to
#: paper over here.
#:
#: Run-level, never per-arm, for the same reason `NORMALIZE_KERNEL_EMBEDDING`
#: above is one: a per-arm flag would make two adjacent rungs of the ladder
#: differ in the component under test AND in the normalization, so no
#: difference between them could be attributed to either one.
#:
#: Defaulted to the CURRENT behaviour (off, divide by the constant 2) so
#: every number measured so far stays reproducible and the full suite stays
#: green with no flag set; the bit-identical-with-the-flag-off invariant in
#: `tests/test_local_dispersion.py` is what a later change to this default
#: has to keep passing.
NORMALIZE_LOCAL_BY_DISPERSION = False

# ------------------------------------------------------- schedules, shared by all

#: The adaptation coefficient is one object with two knobs — how fast it grows
#: and how far — and `CREDA.schedules.creda_ramp` is where both live. Both families
#: drive it from there, so the two sides of the comparison share one
#: implementation rather than two copies of the same formula.
#:
#: Nothing multiplies the schedule afterwards. Three factors, two of them pinned
#: at one and therefore invisible, is how a scale error hides: the supervised
#: term of Eq. (21) spent a revision at 18.42 times its stated weight because the
#: coefficient was spread across places nobody read together.

#: How far. One is the neutral of Eq. (39), and from r17 that is a statement about
#: the objective rather than about this setting. Eq. (36) normalizes the global
#: score by the conservative bounds, Eq. (38) bounds the local term, and Eq. (21)
#: is divided by its own supremum B_src — so all three terms live in [0, 1) and a
#: coefficient of one weighs them equally, which is what the normalization is for.
#:
#: It is identical for every arm that has an adaptation term, and it is not chosen
#: by looking at outcomes. Tuning it on one side while the other keeps whatever
#: its author chose would make the two arms differ in two things; held identical,
#: the normalization shows up in the results instead of being compensated for.
#:
#: CREDA's own published ceiling for these domains is `creda_lambda_special`
#: = 1e-4, and running it there was measured to be inert: at 1e-4, 1e-2 and 1e-1
#: every adapted arm scored exactly what its own floor scored. That measurement
#: was taken against the UN-NORMALIZED objective — before Eq. (21) was divided
#: by its own supremum B_src and the three terms of Eq. (39) were brought onto a
#: common scale — so it is historical record and not a reading of the objective
#: this comment describes above. What it established still stands: a comparison
#: against a CREDA whose term changes no decision is a comparison against the
#: source-only floor with extra wall time.
#:
#: What it no longer supports is the closing clause it used to carry, which had
#: the two families sharing this one ceiling. They share none: each searches its
#: own and its derivations inherit it, and
#: `harness.ceiling_for(reduction, family, transfer)` is where any arm's
#: coefficient is read from. This constant is the neutral that search is read
#: against, not the value either family runs at. The report prints the ceiling
#: each family actually found rather than leaving it to whoever knows CREDA
#: to notice.
RAMP_CEILING = 1.0

# --------------------------------------------------------- the ceiling search
#
# Each family looks for its own ceiling and keeps it for its derivations. A
# shared ceiling equalizes the coefficient and unequalizes the balance: the two
# objectives sit a factor of B_src apart, so the same number puts adaptation at
# about 85% of one objective and 10% of the other. Searching per family
# equalizes what actually matters, which is where each method operates.
#
# One search per family, inherited. If every arm found its own, B->E would
# differ in two things and no rung would be attributable. The consequence is
# declared rather than hidden: E and F carry no local term, so the ceiling found
# on the complete method is not necessarily theirs, and that is the price of a
# ladder that can be read.

#: The interval, and it is not arbitrary: the endpoints are the two declared
#: defaults. CREDA's published `creda_lambda_special` at the bottom, the neutral
#: of a normalized Eq. (39) at the top. Whatever comes out sits between two
#: values that were already defensible, so the search cannot invent one.
CEILING_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

#: Which arm each family searches with: the complete method, not an ablation.
SEARCH_ARMS = {"milcreda": "G"}

#: The search runs once, at the scale the campaign runs at, and never at pilot
#: scale. Its epoch count is `FULL_EPOCHS` and not `EPOCHS`, deliberately: the
#: ramp climbs on the fraction of training elapsed, so at three epochs it is
#: saturated by the second and every ceiling is reached almost immediately. A
#: ceiling found there describes a landscape the campaign never trains in.
#:
#: Which is why it is not a knob of the pilot. The pilot is the campaign at a
#: smaller scale, and a pilot that re-searched would be a different program from
#: the one it exists to rehearse. Both read `CEILINGS` below.
SEARCH_EPOCHS = FULL_EPOCHS
SEARCH_SEEDS = [0, 1, 2]

#: The scale the search's answer requires, declared separately from the scale it
#: is running at — the same pairing the campaign has, for the same reason. With
#: only one of them, a ceiling found at three epochs and the configuration agree
#: with each other and everything reads as finished. With both, `atRequiredScale`
#: lands in the record and the campaign refuses a ceiling searched below it.
#:
#: Three repetitions is not elegance, it is the floor. The ceiling is measured on
#: 20 validation bags per transfer, so one seed leaves the granularity at five
#: points and the argmax over five cells is picked by noise.
FULL_SEARCH_EPOCHS = FULL_EPOCHS
FULL_SEARCH_SEEDS = 3

#: What the search found, once, and what everything else runs at. Empty means it
#: has not been run: `campaign` refuses rather than searching on the spot, so a
#: campaign can never quietly fund its own coefficient out of the run it is about
#: to report.
#:
#: Filled in from `Results/Benchmark/ceilings.json`, which keeps the whole grid
#: and not only the winner — a ceiling chosen between four identical scores and
#: one chosen by a real difference are the same number and not the same evidence.
#: Filled in at the end of this file, once the paths it reads from exist.
CEILINGS: dict[str, float] = {}

#: The ceiling of each family on each transfer the search actually measured.
#: Empty for a transfer the search never saw, and that emptiness is the rule:
#: `ceiling_for` falls back to `CEILINGS` there, which is the pooled winner of
#: the searched transfers applied out of sample. Kept apart from `CEILINGS`
#: rather than folded into it so the two readings stay distinguishable in the
#: record — a value chosen by looking at that transfer and one inherited from
#: two others are the same number and not the same evidence.
CEILINGS_BY_TRANSFER: dict[str, dict[str, float]] = {}

#: Which transfers the search runs on. This is cost and not insulation: the
#: selection role is what keeps the search away from the verdict's material, so
#: the search may look at any transfer it likes as long as it looks at validation
#: bags. Two is what it costs to pick one scalar, and one easy transfer with one
#: hard one keeps the choice from being fitted to a single difficulty.
#: Todas. La rejilla medía dos y las otras cuatro heredaban su techo sin que
#: nadie lo comprobara ahí, y las dos derrotas significativas de `MIL-CREDA`
#: están las dos en transferencias heredadas. Medir las seis elimina la
#: aplicación fuera de muestra: `ceiling_for` deja de tener rama agrupada
#: alcanzable.
#:
#: Buscar sobre transferencias que el veredicto también juzga no filtra nada. Lo
#: que mantiene disjunto el material es el **rol** —la búsqueda lee `valid`, el
#: veredicto lee `eval`— y eso vale igual para las seis.
SEARCH_TRANSFERS = list(TRANSFERS)   # las mismas que VERDICT_TRANSFERS

#: Qué motor elige el techo. `grid` es la rejilla de techos fijos repetida sobre
#: semillas, y sigue existiendo porque escribió el registro que gobierna la
#: campaña vigente. `optuna` busca sobre un rango continuo con trials.
SEARCH_ENGINE = "optuna"

#: El rango sobre el que se busca, en escala logarítmica. Son los extremos de la
#: rejilla que reemplaza: lo que cambia es que adentro ya no hay cinco puntos
#: sino un continuo.
#:
#: `lambda_glob` de la Ec. (39) -- Decision: leído por `ceiling_for`/
#: `reduction.ceilings`. Ver `CEILING_LOCAL_RANGE` para el segundo coeficiente
#: que esta constante ya no tiene que cubrir sola.
CEILING_RANGE = (1e-4, 1.0)

#: `lambda_loc` de la Ec. (39): el segundo coeficiente que este mismo `objective.
#: total_objective` siempre aceptó por separado (su propia firma y docstring lo
#: dicen) y que hasta esta etapa el harness financiaba con el MISMO número que
#: `lambda_glob` -- una rampa, un techo, pasado dos veces. Ese acoplamiento era el
#: defecto, no una decisión de la Ec. (39): el comentario que solía vivir en esta
#: sección, más abajo, argumentaba lo contrario y quedó falsificado por este mismo
#: cambio; ver la nota nueva donde estaba.
#:
#: **Su propio rango, no un alias del global, y la diferencia es la decisión.**
#: Escrito `= CEILING_RANGE` quedaba acoplado a un número que tiene un paper
#: detrás: el piso de `CEILING_RANGE` es el techo PUBLICADO de CREDA. Mover ese
#: piso --legítimo, porque responde a CREDA-- arrastraba en silencio al
#: coeficiente local, que es exactamente lo que este cambio acaba de declarar
#: independiente. Dos números iguales hoy y una sola decisión mañana.
#:
#: **El tope, 1.0, no se presta del global: es el neutro de la Ec. (39) misma**
#: (`MILCREDA_CEILING`). Peso uno significa "el término entra tal como la
#: ecuación lo escribe"; pasar de ahí afirmaría que el término local necesita más
#: peso que el supervisado, y esa es una decisión que nadie tomó.
#:
#: **El piso, 1e-4, es una ELECCIÓN declarada y no una medición**, y se dice así
#: de frente: CREDA no tiene término local, así que no hay ningún valor publicado
#: del que partir --- ni acá ni en ninguna parte. Lo que el número dice es
#: "suficientemente chico para que el término esté apagado", y las cuatro décadas
#: que abre en escala logarítmica son el espacio donde la búsqueda decide.
#:
#: **Y la búsqueda avisa si la ventana quedó mal puesta**: un ganador pegado a
#: cualquiera de los dos bordes es la señal de que el rango, y no el criterio,
#: eligió. Ese es el control que reemplaza al piso medido que este término no
#: puede tener.
CEILING_LOCAL_RANGE = (1e-4, 1.0)

# ---------------------------------------------- las otras siete dimensiones
#
# The search covers eight dimensions now, not six: the two ramp ceilings above
# (`CEILING_RANGE`/`CEILING_LOCAL_RANGE`), the two ramp growth rates below
# (`RAMP_DELTA_RANGE`/`RAMP_DELTA_LOCAL_RANGE`), and the four that follow them.
# Every one is an Eq. (39)/Eq. (15)/Eq. (16)/Eq. (28) constant this file
# otherwise declares fixed.
#
# `lambda_glob`/`lambda_loc` used NOT to be among them, on the premise that both
# came out of the one shared ramp (`harness.ramp`, then `total_objective(...,
# coefficient, coefficient)`) and a second, independent coefficient for each
# would change Eq. (39) itself rather than search over it. That premise was
# backwards: Eq. (39) already names two coefficients -- `MIL_CREDA.objective.
# total_objective`'s own signature and docstring always took `lambda_global`
# and `lambda_local` apart -- so giving each its own schedule is not a change to
# the equation, it is the harness finally matching an equation it had been
# under-applying. `ATTENTION_WIDTH` stays out and fixed for the reason this
# section always gave: it sizes `R_phi`'s hidden layer, an architectural choice
# with no equation attached, never a hyperparameter Eq. (15) itself names.
#
# Each range is declared beside where it came from, the same discipline
# `CEILING_RANGE`'s own comment already applies -- log scale where the constant
# is inherently a ratio or a bandwidth, linear where it is a bounded mixing
# weight.

#: How fast the ramp climbs. CREDA's own `delta = 20` is the interior point this
#: range is built around, not its edge: the existing note on `RAMP_DELTA` below
#: already measures what happens at the two ends of a wide range -- at three
#: epochs `delta = 20` reaches 0.9975 by the second, i.e. a near-instant step
#: function, and a much smaller delta would still be ramping past the twentieth
#: epoch of a full run. `(1.0, 100.0)`, log scale, spans "barely past the
#: neutral by the end of a full run" to "a step function in the first epoch" --
#: the two qualitative regimes a growth-rate search has to be able to reach.
RAMP_DELTA_RANGE = (1.0, 100.0)

#: How fast the LOCAL term's ramp climbs -- `lambda_loc`'s own growth rate,
#: independent of `lambda_glob`'s above. Reuses `RAMP_DELTA_RANGE`'s bracket
#: outright rather than re-deriving one, and unlike `CEILING_LOCAL_RANGE` this
#: reuse is not a weaker stand-in: growth SPEED is a property of a schedule
#: climbing from zero to whatever ceiling it was given, over the fraction of
#: training elapsed (`CREDA.schedules.creda_ramp`'s own math), and nothing in
#: that shape is specific to which term or which method the coefficient belongs
#: to. The same "barely past neutral" to "step function in the first epoch"
#: argument `RAMP_DELTA_RANGE` already makes applies unchanged.
RAMP_DELTA_LOCAL_RANGE = RAMP_DELTA_RANGE

#: Decision 1's one bandwidth. `KERNEL_SIGMA`'s own docstring already declares it
#: "a placeholder to be tuned with Optuna alongside the ceiling search", measured
#: once by the median heuristic at 36.135014304860874. The range is one order of
#: magnitude either side of that measured value, log scale, because a kernel
#: bandwidth is a ratio quantity (it rescales a squared distance) and an
#: order-of-magnitude sweep is the standard way to bracket a median-heuristic
#: estimate without asserting the heuristic itself is exactly right.
KERNEL_SIGMA_RANGE = (KERNEL_SIGMA / 10.0, KERNEL_SIGMA * 10.0)

#: gamma of Eq. (15): the weight of the within-bag consensus term against the
#: learned relevance R_phi. `ATTENTION_GAMMA`'s own docstring already declares
#: zero the neutral (R_phi alone, l1-normalized); the mixing weight is not
#: declared with an upper bound past one anywhere in the revision, and `(0.0,
#: 1.0)` is the natural closed range for a term that never exceeds giving the
#: consensus term the whole logit.
ATTENTION_GAMMA_RANGE = (0.0, 1.0)

#: tau_att of Eq. (16): the attention temperature. `ATTENTION_TEMPERATURE`'s own
#: docstring declares one the neutral (a plain softmax, no scaling). `(0.1,
#: 10.0)`, log scale, is the ordinary bracket for a softmax temperature: at the
#: low end the distribution is far sharper than the neutral, at the high end far
#: flatter, and the neutral sits at the geometric range's own centre.
ATTENTION_TEMPERATURE_RANGE = (0.1, 10.0)

#: The local temperature of Eq. (28). `TAU_LOCAL`'s own docstring declares one
#: the fixed neutral, with no counterpart in CREDA to measure a range from --
#: it plays the identical role Eq. (16)'s temperature plays, scaling a softmax
#: rather than a kernel bandwidth or a mixing weight, so it is given the same
#: bracket for the same reason.
TAU_LOCAL_RANGE = (0.1, 10.0)

#: Cuántas evaluaciones por `(familia, transferencia)`. Una por trial: la
#: repetición que daban las semillas la reemplaza el término de ruido que el GP
#: estima para decidir dónde mirar.
#:
#: **Cuarenta y no treinta, y el número es del operador.** Separar los dos
#: coeficientes de adaptación llevó el espacio de seis dimensiones a ocho
#: (`CEILING_RANGE`/`CEILING_LOCAL_RANGE` y `RAMP_DELTA_RANGE`/
#: `RAMP_DELTA_LOCAL_RANGE`), y treinta evaluaciones sobre ocho dimensiones son
#: menos de cuatro por dimensión. La decisión fue subir a cuarenta y explícitamente
#: no más: cuarenta son cinco por dimensión, sigue siendo una exploración fina, y
#: el operador la eligió sabiéndolo.
#:
#: **Lo que este número NO resuelve, dicho acá para que nadie lo lea como que sí.**
#: La búsqueda no se reparte entre máquinas --- `run_search` no toma `seeds`, su
#: `shard` es sólo una etiqueta de directorio --- así que las seis transferencias
#: por cuarenta trials corren en una sola sesión o no corren. Cuánto cuesta eso en
#: la máquina que va a correrlo no está medido: la única proyección que existe es
#: una regla de tres desde el ensayo local, y la propia forja se niega a proyectar
#: porque el registro no guarda la duración de la corrida que lo produjo. Ese
#: número se mide en el ensayo remoto, y es el que decide si ocho dimensiones son
#: alcanzables.
#:
#: El ensayo se queda en cuatro y no acompaña la subida, a propósito: existe para
#: probar que el cable anda, no para elegir nada. Subirlo lo haría más caro sin
#: hacerlo más útil, y lo acercaría a parecer una búsqueda de verdad.
SEARCH_TRIALS = 40
PILOT_SEARCH_TRIALS = 4

#: La única semilla que cada trial evalúa. Declarada y no sorteada: dos trials
#: sobre semillas distintas medirían el techo y el sorteo a la vez, que es la
#: confusión que la comparación apareada de la rejilla existía para evitar.
SEARCH_SEED = 0

#: La diferencia más chica que el criterio puede expresar sobre el rol de
#: búsqueda: con `VALID_BAGS` bolsas, la exactitud se mueve de a `1/VALID_BAGS`.
#: Es lo que define la meseta, y es una propiedad del instrumento y no una
#: cantidad ajustada — dos techos que difieren en menos de una bolsa no son
#: distinguibles por la medición, opine lo que opine el GP.
SEARCH_RESOLUTION = 1.0 / VALID_BAGS

#: And the verdict keeps all six. An earlier draft withheld the two the search
#: used, which was the right instinct against the wrong leak: with the roles
#: already disjoint by bag, withholding them bought nothing and cost a third of
#: the units the paired reading rests on. The ceiling being chosen on two
#: transfers and applied to six is not a leak — it is an out-of-sample
#: application, and the report says so.
VERDICT_TRANSFERS = TRANSFERS

#: What the search maximizes, and where. Target accuracy is the outcome the
#: campaign is about; the validation role is the only place the search may read
#: it, because the evaluation role is not seen before the verdict.
SEARCH_CRITERION = "targetAccuracy"
SEARCH_ROLE = "valid"

#: How fast. CREDA's own `delta`, the shape of `get_lambda`: zero at the first
#: epoch, approaching the ceiling afterwards. Applied to both sides so the
#: schedule is not one more difference.
#:
#: The curve runs on the fraction of training elapsed, so a short run is not a
#: slower version of a long one — at three epochs it is already at 0.9975 by the
#: second, while twenty epochs take about five to get there. The pilot therefore
#: exercises almost no warm-up, which is worth remembering before reading a pilot
#: as evidence about a schedule.
RAMP_DELTA = 20

#: One fixed, declared learning rate -- CREDA's own `creda_lr_special` -- for
#: every arm, with no decay. `LR_ALPHA`/`LR_BETA` and the decay of `get_eta` are
#: removed: this stretch's own decision is one fixed rate rather than a schedule,
#: and it is not searched -- the ceiling search's six dimensions
#: (`RAMP_CEILING`, `RAMP_DELTA`, `KERNEL_SIGMA`, `ATTENTION_GAMMA`,
#: `ATTENTION_TEMPERATURE`, `TAU_LOCAL`) are all Eq. (39)/Eq. (15)/Eq. (16)/
#: Eq. (28) constants; the optimizer's own rate is not one of them.
LR = 1e-3

#: Held equal across the two units. One bag per class covers every class in every
#: step, which the local correspondence requires; the instance arms take the same
#: number of images so neither side is measured against a different optimizer.
BAGS_PER_STEP = CLASSES                                   # 10
IMAGES_PER_STEP = BAGS_PER_STEP * INSTANCES_PER_BAG       # 300

# ----------------------------------------------------------------------- arms
#
# unit        which statistical unit the arm trains on
# adaptation  None, "creda" or "milcreda"
# weighting   confidence weighting of the target blocks
# local       the subject-to-subject correspondence, which CREDA has no analogue of
# attention   how a bag becomes a representation, for bag-unit arms
#
# Normalization is part of the architecture and not a per-arm choice: every
# adapted arm's target forward runs through its OWN target encoder's
# normalization layers, in training mode, exactly as the source forward runs
# through the source encoder's -- each domain trains its own encoder's
# statistics (Eq. 13 splits the encoder; it does not split the treatment of
# normalization within one). There is no separate "frozen"/"live" knob here --
# the only normalization fact any arm declares is the floor's own (Decision
# 2): it never encodes a target image during training at all, so it never
# reaches a normalization layer with one either.

#: selection   which instances of a bag the arm is allowed to look at: None for
#:             all of them, or a rule that keeps `SELECT_K` of the `INSTANCES_PER_BAG`
#:
#: The display name is what every table and figure prints, and the asterisks in it
#: count what the arm lacks: `CREDA*` is CREDA without the confidence weighting,
#: `MIL-CREDA**` lacks the local term and the weighting, `MIL-CREDA*` lacks only
#: the local term, and an unmarked name is the complete method.

#: `normalization` is `"shared"` for every declared arm: the target forward
#: passes through the target encoder's own normalization layers, in training
#: mode, exactly as the source forward passes through the source encoder's --
#: each domain trains its own encoder's statistics, never a frozen or
#: per-domain-switched copy (the `target_embeddings` seam above `wiring.Arm`
#: states this; Eq. (13) is what splits the encoder itself into two).
#:
#: `GN` is retired, and the whole `"sourceBatch"` axis goes with it. It was the
#: one arm that declared it -- its target forward normalized with the CURRENT
#: SOURCE BATCH statistics of that same step, so its normalization layers never
#: learned from the target at all -- and it was the only thing that separated it
#: from `G`. `wiring.Arm._source_embeddings`, `_encode_with_frozen_stats` and
#: `_batchnorm_modules` existed to serve exactly that one arm and are retired
#: with it; nothing else reached any of the three. The `normalization` key stays
#: declared on the four remaining arms, at one value on all of them, and no code
#: reads it today: with the axis gone there is nothing left to branch on, and
#: whether the key itself should go is a decision about the arm declaration's
#: shape rather than part of retiring this arm.
ARMS = [
    {"id": "B", "name": "MIL-Baseline", "label": "source-only (bags)",
     "unit": "bag", "adaptation": None, "weighting": False, "local": False,
     "attention": "learned", "selection": None},
    {"id": "E", "name": "MIL-CREDA**", "label": "MIL-CREDA global, unweighted",
     "unit": "bag", "adaptation": "milcreda", "weighting": False, "local": False,
     "attention": "learned", "selection": None},
    {"id": "F", "name": "MIL-CREDA*", "label": "MIL-CREDA global, weighted",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": False,
     "attention": "learned", "selection": None},
    {"id": "G", "name": "MIL-CREDA", "label": "MIL-CREDA full (global + local)",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned", "selection": None},
]

ARMS_BY_ID = {arm["id"]: arm for arm in ARMS}

#: The display order of every table and figure: the order the arms are declared in.
ARM_ORDER = [arm["id"] for arm in ARMS]
NAME_OF = {arm["id"]: arm["name"] for arm in ARMS}

#: Selection arms (`SU`, `SA`, `SK`) and their budget (`SELECT_K`,
#: `SELECTION_SEED`) are removed: every remaining arm keeps every instance of a
#: bag. `wiring.Arm.select` still exists and still returns `H` unchanged for
#: every declared arm's `spec["selection"] is None`.
#:
#: What each rung of the ladder reads. A comparison is only attributable when its
#: two arms differ in one thing, so the pairs are written out rather than left to
#: whoever reads the table.
LADDER = [
    ("B", "E", "qué compra el término global, sin ponderar"),
    ("E", "F", "qué compra la ponderación por confianza en MIL-CREDA"),
    ("F", "G", "qué compra la correspondencia local"),
]

#: Which direction wins each dimension. The parameter count is reported and not
#: contested: it is a fact rather than a contest. Time and memory
#: (`seconds`/`peakMiB`) are removed from this comparison entirely -- not one
#: dimension, not one record field, not one reader of either.
HIGHER, LOWER, DESCRIPTIVE = "higher", "lower", None
DIMENSIONS = {
    "targetAccuracy": HIGHER,
    "sourceAccuracy": HIGHER,
    "contribution": DESCRIPTIVE,
    #: `contributionGlobal`/`contributionLocal` (the two terms `contribution`
    #: sums, on their own coefficient each -- see `harness.run_one`'s own
    #: docstring) are deliberately NOT declared here. This dict is
    #: `summarize()`'s own iteration set (`harness.py`: `for dimension in
    #: config.DIMENSIONS: ... r[dimension]`) and `shards.partition`'s
    #: (`harness.py:2236`) -- both HARD-require every run dict to carry every
    #: key declared here, across the whole campaign grid and every shard.
    #: Widening that set is a decision about pooling and shard-agreement
    #: policy for two new dimensions, which belongs to whoever owns
    #: `MIL_CREDA_Benchmark/__init__.py`'s `distribution` declaration (outside
    #: this stretch's authorized scope) and not to this constant alone. The two
    #: fields still exist, on every run's own record (`harness.run_one`'s
    #: returned dict) -- just not pooled, tabled or shard-compared yet.
    #: The supervised magnitude and the adaptation's share of the objective.
    #: `contribution` alone cannot separate a term that commanded nothing from a
    #: term that was scaled to nothing, and both print as a small number. The two
    #: are descriptive because neither has a better direction: a large share is
    #: not a better method, it is a differently balanced objective, and a rung
    #: whose two arms differ in share is a rung whose reading has to say so.
    "supervised": DESCRIPTIVE,
    "adaptationShare": DESCRIPTIVE,
    "parameters": DESCRIPTIVE,
}

#: Whose weights phase 2 needs, and how many repetitions of each: the ones closest
#: to the median target accuracy, never the best. The best of thirty is an extreme
#: of thirty draws, and its latent space describes the luckiest run rather than
#: the method.
#:
#: Every arm gets three, and the reason is which arms the figures will need: the
#: latent grid shows the three best adaptations, and which three those are is only
#: known once the campaign has ranked them. Keeping weights for four arms and
#: discovering afterwards that the ranking names a fifth would cost the whole run
#: again. Three is the smallest count that still gives every latent measurement a
#: dispersion. At roughly 45 MB each this is about 8 GB, written as the run goes,
#: all of it local and ignored by git — cheap against a day and a half of compute.
CHECKPOINTS = {arm["id"]: 3 for arm in ARMS}

#: Which floor each adapted arm is read against: same unit, same everything, with
#: the adaptation term switched off.
FLOOR_OF = {"G": "B", "F": "B", "E": "B"}

# ------------------------------------------------------------------- figures

#: How many transfers the figures show. Three and not six: six rows at a legible
#: panel size do not fit on a page, and the tables already carry every transfer.
FIGURE_TRANSFER_COUNT = 3

#: And which three: the ones where the methods reach the highest mean target
#: accuracy, computed from the campaign rather than written here. Two things
#: follow from that and both have to be said rather than assumed.
#:
#: It is a choice made by the outcome, so it is declared in every caption. What
#: makes it defensible is that the alternative is worse for this particular
#: figure: the latent space of a transfer where every method sits near chance is a
#: picture of a model that did not learn, and nothing about alignment can be read
#: off it. Showing where adaptation actually happened is the informative choice as
#: long as nobody is told it was the neutral one.
#:
#: What it must never touch is *which draw* is shown. That stays the display seed,
#: chosen by a rule that favours no method, because choosing the draw by the
#: outcome is how a figure stops being able to come out wrong.
FIGURE_TRANSFER_RULE = "mayor exactitud media en destino sobre todos los métodos"

#: The columns of the latent grid after the shared original space: the floor and
#: all three MIL-CREDA. The floor is what makes the rest readable — "aligned"
#: cannot be seen without a "not aligned" beside it.
#:
#: One floor and not two, because there is one unit left: with the instance-unit
#: arms undeclared, `MIL-Baseline` is the only source-only column the grid has to
#: draw. The comparison that measured whether two floors were redundant is gone
#: with its second floor; what it found while it had one is kept here as the
#: record it left: drawn at the instance level on the pilot, the two differed by
#: up to 0.38 in distance ratio and 0.07 in domain separability, so they were
#: never redundant. Declaring a second floor again means writing that comparison
#: again, which is the honest cost of having retired one.
#:
#: The selecting arms are left out because they differ from `G` in their instance
#: budget rather than in what they align: a phase-one question, not a picture.
LATENT_PANELS = ["B", "E", "F", "G"]

#: Whether `MIL-Baseline` is redundant with `Baseline` once both are drawn at the
#: instance level is a measurement, not an assumption: the two train the same
#: encoder through different objectives — per instance against per bag through the
#: attention pooling — so their instance embeddings need not agree at all. The
#: notebook measures how far apart they are and says so, and this constant is
#: what the answer is checked against rather than a belief about it.
FLOORS_AGREE_WITHIN = 0.05

#: Every panel is drawn at the **instance** level, bag-unit arms included. Every
#: arm encodes instances — that is where Eq. (13) applies — so it is a space they
#: all have, and it is the only way the panels carry the same number of points.
#: One point per subject beside one point per instance made the CREDA columns look
#: like they covered the space and the MIL columns look sparse, which is the
#: statistical unit drawn rather than anything about alignment.
#:
#: The bag-level view is not lost: the phase-two tables measure each arm in its own
#: unit, which is where that distinction belongs.
LATENT_UNIT = "instance"

#: Points per domain in each panel of the grid, stratified by class. Every panel
#: gets the same number so no column looks denser than another; 300 is thirty per
#: class, enough to see a cluster and few enough that UMAP is quick over 21 panels.
LATENT_POINTS = 300

#: The bag figure is about the local correspondence, so it shows the arms that
#: differ in it rather than the arms that rank highest: the floor, the same method
#: without the local term, and the complete one. If the middle panel looks like
#: the right one, the local term is doing nothing visible — which is the whole
#: reason the figure is worth drawing.
BAG_PANELS = ["B", "F", "G"]

#: One bag of each class is highlighted, the same bags in every panel, and it is
#: the median of its class by correspondence mass. The best bag of each class
#: would produce a clean pairing under every arm, including the floor, and a
#: figure that cannot come out wrong is not measuring anything.
BAGS_HIGHLIGHTED_PER_CLASS = 1

# ---------------------------------------------------------------------- paths

#: This file sits at <repo>/src/MIL_CREDA_Benchmark/config.py, so the repository
#: is two levels up. The product folder is named with a hyphen, which is legal in
#: a directory and not in an identifier — hence the two spellings.
REPOSITORY = Path(__file__).resolve().parents[2]
PRODUCT = REPOSITORY / "MIL-CREDA"

RESULTS = PRODUCT / "Results" / "Benchmark"
MODELS = PRODUCT / "Models" / "Benchmark"
DATA_CACHE = REPOSITORY / ".benchmark-data"

#: `RESULTS` already ends in `Benchmark`; appending it again buried the record one
#: level deeper than the contract declares, and the search wrote there without a
#: word. Every path here hangs off a constant that says where it points.
CEILINGS_RECORD = RESULTS / "ceilings.json"

#: Donde escribe el ensayo local de la busqueda, separado del registro real y con
#: su propio nombre. Dos archivos y no uno, porque son dos experimentos: este
#: corre a `PILOT_SEARCH_EPOCHS` epocas y su respuesta no se puede citar. A escala
#: piloto la rampa se satura en la segunda epoca y todo techo se alcanza casi
#: enseguida, asi que contesta sobre un paisaje donde nada mas entrena. Escribirlo
#: sobre `CEILINGS_RECORD` haria que una campana completa consumiera ese valor sin
#: una palabra, que es exactamente la falla que el sello de piloto existe para
#: impedir, un experimento mas arriba.
CEILINGS_PILOT_RECORD = RESULTS / "ceilings.pilot.json"

#: La escala propia del ensayo. Declarada aparte de `SEARCH_EPOCHS`/`SEARCH_SEEDS`
#: y nunca derivada de ellas: bajarle la escala a la busqueda real seria cambiar
#: el experimento; esto es un experimento distinto que comparte su programa.
PILOT_SEARCH_EPOCHS = EPOCHS
PILOT_SEARCH_SEEDS = [0]




#: Which levels keep their weights, and the two are not a preference: they are
#: exactly the levels the latent notebook renders, so the list is derived from
#: what the report already declares rather than typed beside it.
#:
#: The other levels run and record their runs -- the degradation curve is read
#: from `runs.jsonl` and needs every level -- but write no checkpoints at all. A
#: campaign keeps `len(ARMS) * len(TRANSFERS) * 3` weights at roughly 45 MB each,
#: about 8 GB a level; keeping five would be 40 GB of which three levels would
#: never be opened by anything.
CHECKPOINT_LEVELS = [NOISE_LEVELS[0], NOISE_REPORTED]


def keeps_checkpoints(rate: float) -> bool:
    """Whether a campaign at this rate has a reader for the weights it would keep."""
    return any(abs(rate - level) < 1e-12 for level in CHECKPOINT_LEVELS)


def results_for(rate: float, kind: str = "campaign",
                pilot: bool = False) -> "Path":
    """Where a run of this shape, at this rate, of this kind, writes.

    Three coordinates and not one, and each was added because two runs collided
    on the one before it.

    `rate == 0` with `kind == "campaign"` and no pilot returns `RESULTS`
    unchanged: every path already on disk, every notebook naming one and the
    `records` block of the declaration point there. The axis is an addition, not
    a relocation.

    `kind` separates the degradation sweep from the campaign. Both can stand at
    the same rate and they are not the same experiment -- the sweep is ONE
    transfer across every level, the campaign is every transfer at one level --
    and `runs.jsonl` is opened `"w"`, so whichever ran second would truncate the
    first in silence.

    `pilot` separates a rehearsal from the run it rehearses, for exactly that
    reason and one worse: when a pilot and a real campaign share a destination,
    the one that overwrites is the cheap one. Only the ROOT moves; the shape
    underneath is identical, because a pilot that also rearranged its files
    would not be the same program as the run it claims to rehearse.

    Everything is derived from `RESULTS` rather than rebuilt from `PRODUCT`, so
    a caller that redirects `RESULTS` -- every test here does -- redirects all
    of it and not the clean campaign alone.
    """
    if kind not in ("campaign", "curve"):
        raise ValueError(f"unknown run kind {kind!r}; known: 'campaign', 'curve'")
    base = RESULTS.parent
    if pilot:
        base = base / "Pilot"
    if kind == "curve":
        return base / "Noise" / "curve" / f"rho{rate:g}".replace(".", "p")
    if not rate:
        return base / RESULTS.name if pilot else RESULTS
    return base / "Noise" / f"rho{rate:g}".replace(".", "p")


def models_for(rate: float, kind: str = "campaign",
               pilot: bool = False) -> "Path":
    """Where the run of this shape keeps its checkpoints.

    The same three coordinates as `results_for`, mirroring its destinations
    exactly, and the same reasons. It took two for a while, and its docstring
    said "the same two as `results_for`" while `results_for` took three: a
    pilot sweep then wrote its rate-0 level -- the one level where a curve and
    a campaign agree on every other coordinate -- straight into the campaign's
    own checkpoint directory, relabelling ten manifests `"kind": "curve"`. The
    weights were byte-identical, so nothing raised and nothing was reported.
    `latent.available()` globs a directory, so two runs sharing one would hand
    the analysis a mixed set with no way to tell which run each checkpoint came
    from -- and unlike a truncated `runs.jsonl`, that failure is silent and
    renders.
    """
    if kind not in ("campaign", "curve"):
        raise ValueError(f"unknown run kind {kind!r}; known: 'campaign', 'curve'")
    base = MODELS.parent
    if pilot:
        base = base / "Pilot"
    if kind == "curve":
        return base / "Noise" / "curve" / f"rho{rate:g}".replace(".", "p")
    if not rate:
        return base / MODELS.name if pilot else MODELS
    return base / "Noise" / f"rho{rate:g}".replace(".", "p")


def ceilings_record_for(pilot: bool) -> "Path":
    """El archivo al que le corresponde escribir a esta corrida de la busqueda.

    Un solo lugar donde `pilot` se convierte en un camino, y todo lo que elige
    destino pasa por aca: el escritor, el parcial y el lector. Estaba repetido
    en los tres, y un test que mockeaba el del medio dejaba pasar una mutacion
    en el escritor — cada mitad verificada contra su propio fixture y la union
    entre ellas sin verificar, que es la unica cosa que la regla decia.
    """
    return CEILINGS_PILOT_RECORD if pilot else CEILINGS_RECORD


def noise_axis_for(pilot: bool) -> "Path":
    """La raiz del eje de ruido de esta corrida: lo que resume a los cinco niveles.

    El barrido escribe un arbol por nivel (`results_for(rate, "curve", pilot)`);
    lo que resume a los cinco --- la curva de degradacion, el diagnostico --- no
    pertenece a ninguno, asi que vive en el padre de todos ellos. La tasa y la
    forma no son coordenadas de este destino: es uno por corrida del eje, no uno
    por nivel. `pilot` si lo es, y esa es toda la razon por la que esto existe:
    la expresion estaba escrita a mano en tres lugares --- el paso, la tabla que
    la lee y el cuaderno --- y el cuaderno la componia desde `PRODUCT`, sin
    `pilot` ninguno, asi que el resumen de un barrido de ensayo caia donde va el
    de la corrida completa.
    """
    return results_for(0.0, "curve", pilot).parents[1]


def ceilings_record_in_force() -> tuple["Path | None", str]:
    """Que registro de techos rige, y a que titulo.

    `("full"|"pilot"|"none")`, y el archivo. El completo gana siempre que exista:
    un ensayo no desplaza una medicion. Cuando solo esta el ensayo, rige — para
    que un piloto local pueda correr sin haber gastado Kaggle — y su procedencia
    viaja con la reduccion, para que ninguna tabla cite un techo de ensayo como
    si lo hubiera medido la busqueda.

    Devuelve la procedencia y no solo el camino, porque "de donde salio este
    escalar" es la pregunta que el registro tiene que poder contestar. Un
    resolutor que devolviera el mapping a secas haria indistinguibles los dos
    casos justo donde importa.
    """
    completo, ensayo = ceilings_record_for(False), ceilings_record_for(True)
    if completo.exists():
        return completo, "full"
    if ensayo.exists():
        return ensayo, "pilot"
    return None, "none"


def ceilings_record_at(pilot: "bool | None" = None) -> tuple["Path | None", str]:
    """El registro del que lee ESTA corrida, y a que titulo.

    La misma pareja que devuelve `ceilings_record_in_force`, con la coordenada
    de escala adelante. Un solo lugar donde `pilot` se vuelve un registro para
    todo lo que LEE, igual que `ceilings_record_for` lo es para todo lo que
    escribe, y por el mismo motivo: la preferencia --- el completo le gana al
    ensayo siempre que exista --- vive una sola vez, en
    `ceilings_record_in_force`, y esto la consulta en vez de volver a
    escribirla.

    `pilot=None` es *el que rige* y es la omision: la pregunta de un resolutor,
    "cual manda", que es lo correcto para lo que se muestra sin pertenecer a
    ninguna corrida. Cualquier otro valor es la corrida que pregunta por SU
    registro y por ninguno de los dos: a escala de ensayo el del ensayo, a
    escala completa el completo, y la ausencia del propio se contesta con
    `"none"` en vez de repartir el del vecino. Esa es toda la diferencia entre
    un resolutor y una corrida, y es la que faltaba: `ceilings_on_record` no
    tenia como recibirla, asi que una campana de ensayo pedia sus techos y se
    llevaba los de la corrida completa sin una palabra.
    """
    if pilot is None:
        return ceilings_record_in_force()
    record = ceilings_record_for(pilot)
    if not record.exists():
        return None, "none"
    return record, "pilot" if pilot else "full"


def ceilings_provenance(pilot: "bool | None" = None) -> dict:
    """De donde salieron los techos de esta corrida, para estampar en la reduccion.

    `pilot=None` describe el registro vigente; con la coordenada describe el de
    esa escala. Un aviso que dijera "busqueda completa" sobre una corrida de
    ensayo seria peor que no avisar: es exactamente el sello que
    `search_source_note` existe para poner.
    """
    record, kind = ceilings_record_at(pilot)
    out = {"source": kind, "record": str(record) if record else None,
           "epochs": None, "seeds": None}
    if record is None:
        return out
    import json as _json
    found = _json.loads(record.read_text(encoding="utf-8"))
    # Las claves que la busqueda ya escribe por familia, no una envoltura
    # inventada aca: `epochs`, `seeds` y `atRequiredScale` viven al nivel de cada
    # familia desde que el registro existe.
    entry = next((e for e in found.values() if isinstance(e, dict)), None)
    if entry:
        out["epochs"] = entry.get("epochs")
        out["seeds"] = len(entry.get("seeds") or [])
        out["atRequiredScale"] = entry.get("atRequiredScale")
        out["requiredScale"] = entry.get("requiredScale")
    return out


#: The same mapping `ceiling_record.STAMP_FIELDS` declares, duplicated here
#: rather than imported. `ceiling_record` imports THIS module at its own top
#: level, and `ceilings_on_record()` below is called from `CEILINGS.update(...)`
#: at this module's own top level (a few hundred lines down) -- so an import of
#: `ceiling_record` reached from inside `_refuse_on_stamp_drift`, even a local
#: one, is reached while `config` is still mid-import whenever `ceiling_record`
#: is the caller's first import of the two, and `ceiling_record` itself is then
#: only half-defined. Two constants that have to agree is the cost of not
#: creating that cycle; `tests/test_ceiling_record.py` and this module's own
#: tests hold the two mappings equal so a field added to one and not the other
#: is caught rather than silently checked on one side and not the other.
#: `kernelSigma`/`attentionGamma`/`attentionTemperature` are no longer here:
#: the search now explores all three itself (alongside `rampDelta`/
#: `tauLocal`), per transfer, so an entry's own value is that search's
#: winner rather than a fixed backdrop it ran under -- comparing it against
#: this module's bare declared default would flag drift on every search that
#: found anything else. `revision` is the one genuine backdrop left.
_STAMP_FIELDS: dict[str, str] = {
    "revision": "REVISION",
}


def _refuse_on_stamp_drift(found: dict, record: "Path") -> None:
    """Refuse when any family in `found` disagrees with the CURRENT config's
    stamp, or was written before a stamp existed at all.

    The check `harness.ceiling_for` and `harness.ceilings_in_force` already
    make (via `ceiling_record.stamp_drift`, the same rule, see `_STAMP_FIELDS`
    above for why this cannot call that function directly), moved to where
    these two functions actually read the file -- both were reachable straight
    from a notebook (`Benchmark_Campaign_v1.ipynb`'s `ES_ENSAYO` branch,
    `Benchmark_Noise_Sweep.ipynb`) building `ceilings=`/`ceilingsByTransfer=`
    from these two calls directly, never through `harness.with_ceilings_in_force`,
    so neither of those refusals ever ran: a campaign built that way trained
    every arm under a ceiling searched under a `KERNEL_SIGMA`/`ATTENTION_GAMMA`/
    `ATTENTION_TEMPERATURE`/`REVISION` the current config no longer carries, and
    nothing said so.

    **Called only when the caller named a scale.** Both callers below take this
    only when `pilot is not None` -- an explicit `True`/`False`, the shape every
    caller feeding a `Reduction` actually passes (`ceilings_in_force`'s own
    `pilot` parameter defaults to `False`, never `None`, and so do both
    notebook call sites this closes). `pilot=None` is the *vigente* resolver
    reading (`ceilings_record_at`'s own docstring), used for reporting what
    governs right now rather than for anything about to run under it --
    `CEILINGS` below is the one caller that matters: it fills at IMPORT, before
    any `Reduction` exists to run, and refusing there would make importing this
    module itself depend on a ceiling record's freshness. `campaign()`'s own
    up-front stamp check (`harness.py`) is what actually stands between a stale
    `CEILINGS` default and a run: nothing reaches `run_one` without passing it.
    """
    def has_drift(entry: dict) -> bool:
        for field, source in _STAMP_FIELDS.items():
            current = globals()[source]
            if field not in entry or entry[field] != current:
                return True
        return False

    drifted = sorted(family for family, entry in found.items()
                     if isinstance(entry, dict) and has_drift(entry))
    if drifted:
        raise SystemExit(
            f"refusing to read ceilings from {record}: stamped under a "
            f"revision or hyperparameters the current config no longer "
            f"carries: {', '.join(drifted)}.\n"
            "  Re-run `harness.search_ceilings(...)` under today's config, or "
            "delete the stale record to search again."
        )


def ceilings_on_record(pilot: "bool | None" = None) -> dict[str, float]:
    """The searched ceilings, read from the record the search wrote.

    Read and not remembered, like everything else here. A constant typed in by
    hand would be a second source of truth for a measured value: it goes stale in
    silence the first time the search is re-run and is believed anyway.

    Public, and re-read on every call, because `CEILINGS` below is filled once at
    import. A caller that runs the search in the same process — a notebook, which
    is the only place that ever does — would otherwise hold the empty mapping this
    module was imported with and hand it to a campaign that refuses it.

    `pilot` es la coordenada que faltaba, y la omision sigue siendo el registro
    VIGENTE porque esa es la pregunta de un resolutor: `CEILINGS`, mas abajo, se
    llena al importar y no pertenece a ninguna corrida todavia. Una corrida si
    pertenece a una escala, y la dice: sin poder decirla, una campana de ensayo
    corria bajo los techos de la busqueda COMPLETA --- medidos en otro
    experimento, a veinte epocas --- y su propio registro de ensayo quedaba en
    disco sin que nada lo leyera.
    """
    record, _ = ceilings_record_at(pilot)
    if record is None:
        return {}
    import json as _json
    found = _json.loads(record.read_text(encoding="utf-8"))
    if pilot is not None:
        _refuse_on_stamp_drift(found, record)
    return {family: entry["ceiling"] for family, entry in found.items()}


def ceilings_by_transfer_on_record(
        pilot: "bool | None" = None) -> dict[str, dict[str, float]]:
    """The per-transfer picks, read from the same record.

    A record written before this key existed simply has none, and that is not an
    error: an absent mapping makes every transfer fall back to the pooled winner,
    which is exactly what such a record meant when it was written. Read on every
    call, for the same reason `ceilings_on_record` is.

    **La misma coordenada que `ceilings_on_record`, y por una razon propia.** Las
    dos mitades de un mismo techo --- el agrupado y el pick por transferencia ---
    se leen en dos llamadas separadas, y una reduccion que tomara cada mitad de
    un registro distinto llevaria adentro dos experimentos mezclados en el valor
    mas sensible del calculo. Que hoy coincidan no es una propiedad de las dos
    funciones: es una propiedad de que las dos preguntaban lo mismo. Con la
    coordenada las dos preguntan lo mismo *que la corrida*, que es lo que hay que
    sostener.
    """
    record, _ = ceilings_record_at(pilot)
    if record is None:
        return {}
    import json as _json
    found = _json.loads(record.read_text(encoding="utf-8"))
    if pilot is not None:
        _refuse_on_stamp_drift(found, record)
    return {family: dict(entry.get("byTransfer") or {})
            for family, entry in found.items()}


#: The other seven dimensions the eight-dimensional search explores beside the
#: GLOBAL ceiling (`ceiling_for`/`reduction.ceilings`/`ceilingsByTransfer`), in
#: the order `harness.search_ceilings_trials` declares them. Named once here so
#: `hyper_by_transfer_on_record`/`hyper_pooled_on_record` and
#: `harness.Reduction`'s own field list cannot drift apart on which seven they
#: mean.
#:
#: `ceilingLocal` and `rampDeltaLocal` are the two this stretch of work added:
#: the LOCAL term's own ceiling and growth rate, resolved through this exact
#: same per-transfer/pooled-fallback machinery `hyper_for` already gave the
#: other five rather than through a second copy of `ceiling_for`'s own
#: stamp-drift refusal. That refusal still protects both halves of the split:
#: `ceiling_for`'s check runs against `reduction.ceilingSearch`, which stamps
#: the whole family's search (both coefficients, searched together, in one
#: record) and not the GLOBAL ceiling alone, so a stale record is caught before
#: either coefficient is read, exactly the reasoning `hyper_for`'s own
#: docstring already gives for the other five.
HYPER_DIMENSIONS = ("rampDelta", "rampDeltaLocal", "ceilingLocal", "kernelSigma",
                    "attentionGamma", "attentionTemperature", "tauLocal")


def hyper_by_transfer_on_record(
        pilot: "bool | None" = None) -> dict[str, dict[str, dict[str, float]]]:
    """The OTHER seven searched dimensions' per-transfer winners, read from the
    same record `ceilings_by_transfer_on_record` reads its own pick from.

    `harness.search_ceilings_trials` writes `rampDelta`/`rampDeltaLocal`/
    `ceilingLocal`/`kernelSigma`/`attentionGamma`/`attentionTemperature`/
    `tauLocal` into `perTransfer[label]` beside `ceiling` itself -- that
    transfer's own winning trial, entire, not only its (global) ceiling. This
    reads that same sub-dict back, narrowed to `HYPER_DIMENSIONS`, keyed the
    same way `ceilingsByTransfer` is (`{family: {label: {dim: value}}}`). A
    record written before this stretch's eight-dimensional search simply has
    no `perTransfer[label][dim]` for any of the seven, and an absent dimension
    is read exactly like an absent transfer: the caller falls back to the
    pooled winner, and `harness.hyper_for` falls back further, to the declared
    `config`/`Reduction` constant.
    """
    record, _ = ceilings_record_at(pilot)
    if record is None:
        return {}
    import json as _json
    found = _json.loads(record.read_text(encoding="utf-8"))
    if pilot is not None:
        _refuse_on_stamp_drift(found, record)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for family, entry in found.items():
        per_transfer = entry.get("perTransfer") or {}
        result[family] = {
            label: {dim: winner[dim] for dim in HYPER_DIMENSIONS if dim in winner}
            for label, winner in per_transfer.items()
        }
    return result


def hyper_pooled_on_record(pilot: "bool | None" = None) -> dict[str, dict[str, float]]:
    """The OTHER seven searched dimensions' pooled winner, one per family --
    the sibling of `ceilings_on_record`, over `HYPER_DIMENSIONS` instead of
    `ceiling` alone.

    Read for the same reason `ceilings_on_record`'s own pooled reading is:
    a transfer the search never measured falls back to this rather than to
    nothing, out of sample and declared as such.
    """
    record, _ = ceilings_record_at(pilot)
    if record is None:
        return {}
    import json as _json
    found = _json.loads(record.read_text(encoding="utf-8"))
    if pilot is not None:
        _refuse_on_stamp_drift(found, record)
    return {family: {dim: entry[dim] for dim in HYPER_DIMENSIONS if dim in entry}
            for family, entry in found.items()}


# ------------------------------------------------- las puertas de cada destino

#: Cada puerta por la que una coordenada se vuelve un camino, y bajo que nombre
#: recibe cada una. `harness.Reduction` declara las tres coordenadas del destino
#: --- `labelNoise`, `kind`, `pilot` ---; esto dice como se llaman en la puerta.
#:
#: Una entrada con menos de las tres es una afirmacion, no un descuido: el
#: registro de techos se separa por nombre de archivo y no por arbol, y el eje de
#: ruido es uno por corrida y no uno por nivel, asi que a ninguno de los dos le
#: corresponde una tasa ni una forma. Lo que no puede pasar es que una puerta
#: tome una tasa y no tome las otras dos: `models_for` lo hizo, y un nivel limpio
#: del barrido termino escribiendo en el directorio de checkpoints de la campana
#: y reetiquetando diez manifiestos, en silencio.
#:
#: `tests/test_label_noise.py` lee esto por los dos lados: exige que cada firma
#: acepte exactamente estas coordenadas, y que ninguna llamada a una de estas
#: puertas --- en el paquete, en `tools/` o en una celda de cuaderno --- deje una
#: sin pasar.
DESTINOS: dict[str, dict[str, str]] = {
    "results_for": {"labelNoise": "rate", "kind": "kind", "pilot": "pilot"},
    "models_for": {"labelNoise": "rate", "kind": "kind", "pilot": "pilot"},
    "ceilings_record_for": {"pilot": "pilot"},
    "noise_axis_for": {"pilot": "pilot"},
    "shard_paths": {"labelNoise": "noise", "kind": "kind", "pilot": "pilot"},
    "seal_shard_stamp": {"labelNoise": "noise", "kind": "kind", "pilot": "pilot"},
}

#: Los destinos que se componen a mano y no pasan por ninguna puerta, cada uno
#: con su razon al lado. Estar aca cuesta escribir por que, igual que en
#: `steps.CUADERNOS_SIN_PASO` y por el mismo motivo: un nombre pelado en una
#: lista y un olvido se leen igual.
#:
#: La prueba deriva los destinos compuestos a mano --- del arbol de sintaxis del
#: paquete, de `tools/` y de las celdas de los cuadernos, contra las raices que
#: este modulo declara --- y le resta las puertas; lo que sobra tiene que ser
#: exactamente esto. Un destino nuevo escrito a mano trae una clave nueva y cae
#: en rojo hasta que alguien diga por que no lleva escala.
#:
#: La clave es el archivo y la expresion tal como la escribe `ast.unparse`, para
#: que dos destinos distintos en un mismo archivo no compartan una excusa.
DESTINOS_SIN_COORDENADA: dict[str, str] = {
    "harness.py: config.CEILINGS_RECORD": (
        "no es un destino: nombra en un rechazo el registro COMPLETO que "
        "`search_record()` acaba de leer sin argumentos, y ese rechazo solo "
        "puede dispararse cuando ese archivo existe y quedo bajo escala. El de "
        "ensayo no llega ahi, asi que nombrarlo por coordenada seria decir que "
        "el mensaje puede hablar de dos archivos cuando solo habla de uno"),
    "steps.py: Path(__file__).resolve().parents[2] / 'MIL-CREDA' / 'Notebooks'": (
        "los cuadernos son fuente, no producto: hay un arbol por repositorio y "
        "se ejecutan en el lugar, asi que no existe una version de ensayo y otra "
        "completa entre las cuales elegir"),
    "verification.ipynb: ROOT / 'MIL-CREDA' / 'Results' / 'local_distance_bound'": (
        "dibuja una cota del objetivo medida sobre `tests/sweep.py` y no sobre "
        "una corrida: no hay material contaminado, ni forma, ni escala que "
        "llevar, asi que el archivo es uno por repositorio y no uno por corrida"),
    # Los dos cuadernos que `steps.CUADERNOS_SIN_PASO` declara sin paso dejan su
    # salida al lado suyo, y por eso la componen a mano. La razon es la misma
    # para los tres destinos y vale la pena decirla una vez: estos archivos
    # llevan su escala ADENTRO, como campos (`epocas`, `semillas`, `tasas`, y la
    # huella de la sintonizacion), y al abrirlos el cuaderno compara esos campos
    # contra los suyos y descarta el archivo ENTERO si no coinciden. Una
    # coordenada en la ruta serviria para tener dos archivos a la vez; lo que
    # estos cuadernos necesitan es lo contrario, que una corrida de otra escala
    # no se pueda reanudar ni mezclar fila por fila.
    "Ablation_Ladder.ipynb: REPOSITORY / 'MIL-CREDA' / 'Notebooks' / 'ablation_ladder.json'": (
        "la escalera de ablacion no es un producto declarado: no escribe bajo "
        "`results_for(...)` y su salida la lee una persona al lado del "
        "cuaderno. El archivo guarda `epocas`, `semillas` y `tasas`, y el "
        "cuaderno lo descarta entero cuando no coinciden con los suyos, asi "
        "que la escala esta adentro del archivo y no en su nombre"),
    "steps.py: config.PRODUCT": (
        "no es un destino sino la raiz contra la que `entradas_faltantes` "
        "resuelve las raices que un paso DECLARA en `reads`, que ya vienen "
        "escritas en la ortografia de la corrida completa: la escala esta "
        "adentro de la raiz y no al lado de ella, asi que una coordenada aca "
        "seria una segunda respuesta a una pregunta que la raiz ya contesto"),
    # `harness.py: config.PRODUCT / tables.MECHANISM_RECORD` estaba aca, y se
    # retira porque su destino dejo de componerse a mano: `run_mechanism_sweep`
    # ahora lo pide por `results_for(rate=0.0, kind="campaign",
    # pilot=reduction.pilot)`. La excusa decia que `pilot` no era coordenada
    # suya porque "la seccion 4 no declara un `Reduction` de ensayo propio", y
    # eso describia la ausencia de un dial, no una propiedad del registro: en
    # cuanto `run_mechanism_sweep_shard` toma `pilot`, un camino fijo escribe
    # el ensayo encima de la corrida completa. Lo que la excusa decia bien ---
    # que la TASA no es coordenada, porque limpio y contaminado se multiplexan
    # adentro del JSON --- sigue valiendo y esta escrito en el sitio.
    "promote.py: config.PRODUCT / '.remote-execution' / 'campaign'": (
        "es donde el backend remoto desempaqueta lo que devuelve, antes de que "
        "nada haya leido una reduccion: la escala de lo que viene adentro la "
        "deciden los sellos de cada shard, no el directorio que los recibe"),
    "Benchmark_Results.ipynb: config.PRODUCT / 'Results' / 'figures'": (
        "el directorio de figuras del cuaderno de resultados: un solo arbol "
        "compartido, sin segmento de pilot/full ni de tasa -- distingue limpio "
        "de contaminado por NOMBRE de archivo (`correspondence_grid_clean.pdf` / "
        "`correspondence_grid_noisy.pdf`) y no por directorio. Esta declaracion "
        "registra el hecho, no lo avala: `Benchmark_Results.ipynb` no es un archivo "
        "que este stretch posea (ver el mapa de propiedad del cambio), asi que "
        "si una figura de ensayo y una de la corrida completa debieran vivir "
        "aparte, corregirlo es trabajo de quien sea dueno de ese cuaderno"),
    "Toyset_Plot.ipynb: ROOT / 'MIL-CREDA' / 'Notebooks' / 'toy_figures'": (
        "el directorio de la figura de las tres familias duras de CREDA con "
        "cada clase partida en bolsas. No lleva coordenada porque no hay "
        "corrida que coordinar: ese cuaderno no entrena, no compara los "
        "dominios de `DOMAINS` y no escribe nada bajo `results_for(...)`, asi "
        "que no tiene escala, ni tasa de ruido, ni clase de corrida con que "
        "nombrarse. La figura se redibuja entera en cada ejecucion desde los "
        "generadores de `CREDA.toy_experiment`, de modo que no hay dos "
        "versiones entre las cuales elegir, y una coordenada en la ruta "
        "dejaria figuras viejas en el arbol diciendo algo distinto de la tabla "
        "que esta arriba de ellas.\n\n"
        "Anclar la ruta en `steps.CUADERNOS` en vez de acá habria esquivado "
        "este control entero --- sin raiz de `config` y sin segmento de "
        "producto, el destino no se deriva --- y por eso no se hizo: seria "
        "esconder un destino de salida nuevo detras de una declaracion sobre "
        "donde viven los cuadernos, que es el silencio que este control existe "
        "para impedir"),
    "toy_run.py: root / 'MIL-CREDA' / 'Results' / 'Toyset' / scale.name": (
        "el artefacto del toyset, y es el unico de esta lista que SI lleva "
        "coordenada: `scale.name` es `pilot` o `full`, de modo que una corrida "
        "de ensayo no puede pisar a una completa ni mezclarse fila por fila "
        "con ella. Lo que lo trae aca es componerse a mano en vez de pasar por "
        "`results_for(...)`, y eso es deliberado: `results_for` lleva TRES "
        "coordenadas --- tasa de ruido, clase de corrida y ensayo --- que son "
        "las de la campana de digitos, y dos de ellas no existen de este lado. "
        "El toyset no tiene eje de contaminacion y no compara los dominios de "
        "`DOMAINS`, asi que pasar por ahi obligaria a inventarle una tasa y "
        "una clase de corrida para que el nombre saliera, y un cero inventado "
        "en una ruta se lee despues como un cero medido"),
}


#: --------------------------------------------- las lecturas que llevan escala
#:
#: Un DESTINO es donde una corrida escribe; una LECTURA es de qué corrida se
#: leen los números que después alguien mira. Son la misma coordenada y dos
#: defectos distintos, y este repositorio ya tenía cerrado el primero mientras
#: el segundo seguía abierto: el informe del ensayo llamaba a
#: `harness.search_record()` sin coordenada, leía el registro de la corrida
#: COMPLETA ---que no existía, porque lo que corrió fue un ensayo--- y 23
#: renglones de «no hay búsqueda» salían sobre una búsqueda que sí había
#: corrido. El cuaderno latente hacía lo mismo con los pesos y dibujaba la
#: grilla entera apagada.
#:
#: La regla es UNA y se deriva de la firma, no de una lista:
#:
#:   * si la omisión de `pilot` ya significa **el que rige** ---la corrida
#:     completa si existe, el ensayo si no, que es `pilot: bool | None = None`---
#:     entonces una llamada pelada es correcta y es la forma que se prefiere
#:     para todo lo que se muestra. `contamination.level_dir` y
#:     `harness.search_record` la tienen.
#:   * si la omisión significa **la corrida completa** ---`pilot: bool = False`,
#:     que es lo correcto para lo que GOBIERNA--- entonces cada llamada tiene
#:     que decir la escala. `campaign` pide `pilot=False` por su nombre porque
#:     exige `atRequiredScale` sobre ese archivo, y aceptar ahí el del ensayo
#:     dejaría que los techos de un ensayo gobernaran una campaña real: mucho
#:     peor que el defecto que esta regla arregla.
#:
#: `tests/test_scale_readings.py` recorre el paquete, `tools/` y las celdas de
#: los cuadernos, resuelve cada llamada contra la firma viva, y lo que quede
#: suelto tiene que ser exactamente lo de abajo.
#:
#: La clave es el archivo y la expresión tal como la escribe `ast.unparse`,
#: igual que en `DESTINOS_SIN_COORDENADA` y por el mismo motivo: un nombre
#: pelado en una lista y un olvido se leen igual.
LECTURAS_SIN_COORDENADA: dict[str, str] = {
    "harness.py: _partial_path()": (
        "es el valor por omisión de un ayudante privado que nadie llama así: "
        "`_read_partial` y `_write_partial` lo escriben como `path or "
        "_partial_path()` y sus dos únicos llamadores ---la búsqueda, en dos "
        "lugares--- pasan siempre el parcial ya resuelto por `shard_paths` con "
        "las tres coordenadas de la reducción. Darle escala acá sería una "
        "segunda fuente para un archivo que ya viene decidido"),
}


#: ------------------------- las lecturas que TIENEN una escala y no la reenvían
#:
#: La otra mitad de la regla de arriba, y la que faltaba. `LECTURAS_SIN_COORDENADA`
#: mira las puertas cuya omisión significa «la corrida completa»; una puerta cuya
#: omisión significa «el que rige» acepta la llamada pelada, y ahí es donde se
#: escondía el segundo defecto: `config.ceilings_on_record()` no tenía coordenada
#: ninguna, así que una campaña de ENSAYO ---que sabe perfectamente a qué escala
#: corre--- pedía sus techos y se llevaba los de la búsqueda COMPLETA, medida a
#: veinte épocas en otro experimento, mientras su propio `ceilings.pilot.json`
#: quedaba en disco sin que nada lo leyera. Ni un error: los dos números existen
#: y los dos son plausibles.
#:
#: La regla, derivada del árbol y no de una lista: **si el ámbito que rodea a la
#: llamada YA tiene una escala, la llamada tiene que decirla.** Un ámbito tiene
#: escala cuando es una función con parámetro `pilot`, cuando su cuerpo nombra
#: `reduction.pilot`, o cuando es una celda de cuaderno que declara `ES_ENSAYO`.
#: Lo que queda es un resolutor: nadie alrededor sabe de qué corrida se habla, y
#: «cuál rige» es la pregunta correcta.
#:
#: La clave es el archivo y la expresión tal como la escribe `ast.unparse`, igual
#: que en las otras dos reglas y por el mismo motivo.
LECTURAS_QUE_NO_REENVIAN: dict[str, str] = {
    "Benchmark_Ceiling_Search.ipynb: config.ceilings_provenance()": (
        "no habla de la corrida que ese cuaderno está por lanzar sino del "
        "registro que rige AHORA, y lo imprime con esa etiqueta --- «registro "
        "en vigor ahora mismo» --- al lado del destino que sí lleva la escala. "
        "Son dos hechos distintos y el cuaderno los muestra juntos a propósito: "
        "uno dice adónde va a escribir esta corrida, el otro dice qué archivo "
        "gobierna las campañas mientras tanto. Reenviarle la escala haría que "
        "los dos dijeran lo mismo y el segundo dejaría de informar nada"),
    # "Benchmark_Noise_Report_v1.ipynb: contamination.load(tasa)" is removed:
    # that notebook is deleted with this stretch's restructuring (its
    # analysis folds into Benchmark_Results.ipynb), and the pattern it excused --
    # `contamination.load(tasa)` re-reading the same level
    # `curve_is_pilot`/`ES_ENSAYO` just resolved for that cell -- does not
    # appear anywhere on disk any more (measured: neither `contamination.load`
    # nor `curve_is_pilot` is called in `Benchmark_Results.ipynb`). Keeping the
    # exclusion would let it survive its own call, which is exactly the
    # defect `test_ninguna_lectura_que_ya_tiene_escala_deja_de_reenviarla`
    # exists to catch.
}


CEILINGS.update(ceilings_on_record())
CEILINGS_BY_TRANSFER.update(ceilings_by_transfer_on_record())

# All three are ignored by git: this is a preliminary phase for deciding whether
# the strategy holds, not part of the paper's record.


def sizing() -> dict:
    """The shape of one run, so a notebook can print it before spending an hour."""
    steps = -(-TRAIN_BAGS // BAGS_PER_STEP)      # ceil: the last step is short
    return {
        "arms": len(ARMS),
        "transfers": len(TRANSFERS),
        "seeds": len(SEEDS),
        "runs": len(ARMS) * len(TRANSFERS) * len(SEEDS),
        "epochs": EPOCHS,
        "stepsPerEpoch": steps,
        "imagesPerStep": IMAGES_PER_STEP,
        "imagePassesPerRun": EPOCHS * steps * IMAGES_PER_STEP * 2,  # source + target
        "verdictsMeaningful": len(SEEDS) >= 3,
        # And the search's own shape, beside the campaign's. It is the first thing
        # that spends machine time and the one part that has no pilot scale: a
        # notebook forecasting only the grid would put the longest single wait
        # ahead of the estimate that exists to precede it.
        "search": search_sizing(),
    }


def search_sizing() -> dict:
    """What the ceiling search costs, in the same units the grid is forecast in.

    Its epoch count is its own — `SEARCH_EPOCHS`, never the pilot's — so a caller
    scaling from a timed pilot run has to know the ratio rather than assume it.
    """
    # En los ejes que el motor tiene, no en los de la rejilla traducidos. Una
    # busqueda por trials no visita cinco techos tres veces: visita treinta puntos
    # una vez cada uno, y reportar `ceilings: 5` sobre eso seria describir una
    # rejilla que no corrio.
    if SEARCH_ENGINE == "optuna":
        runs = len(SEARCH_ARMS) * len(SEARCH_TRANSFERS) * SEARCH_TRIALS
        return {
            "engine": SEARCH_ENGINE,
            "families": len(SEARCH_ARMS),
            "transfers": len(SEARCH_TRANSFERS),
            "trials": SEARCH_TRIALS,
            "runs": runs,
            "epochs": SEARCH_EPOCHS,
            "atRequiredScale": SEARCH_EPOCHS >= FULL_SEARCH_EPOCHS,
        }
    runs = (len(SEARCH_ARMS) * len(CEILING_GRID)
            * len(SEARCH_TRANSFERS) * len(SEARCH_SEEDS))
    return {
        "engine": SEARCH_ENGINE,
        "families": len(SEARCH_ARMS),
        "ceilings": len(CEILING_GRID),
        "transfers": len(SEARCH_TRANSFERS),
        "seeds": len(SEARCH_SEEDS),
        "runs": runs,
        "epochs": SEARCH_EPOCHS,
        "atRequiredScale": (SEARCH_EPOCHS >= FULL_SEARCH_EPOCHS
                            and len(SEARCH_SEEDS) >= FULL_SEARCH_SEEDS),
    }
