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


# ------------------------------------------------------------------- material

REVISION = "research-concept-r21.md"

#: Every domain supplies its own bags; a transfer names which is source and which
#: is target. All three hold far more than the 3600 images a domain contributes.
DOMAINS = {"M": "MNIST", "U": "USPS", "S": "SVHN"}

TRANSFERS = [("M", "U"), ("U", "M"), ("M", "S"), ("S", "M"), ("U", "S"), ("S", "U")]

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
#: differing in one parameter forks from the first day.
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]

#: Past this the bag's label stops being defensible. Contaminants are drawn from
#: the other `CLASSES - 1` classes, so the bag's own class remains the plurality
#: while the rate is under one half; at that point the label is a coin toss and
#: the curve measures nothing. Declared as a cap and enforced, not left to whoever
#: edits `NOISE_LEVELS` next.
NOISE_CAP = 0.5

#: Which roles the noise reaches: `train` and `eval`, in both domains, with ONE
#: shared draw across every arm -- the same instances are replaced for `B`, `E`,
#: `F`, `G` and `GN` alike, because the draw is a property of the material
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

    Exact at every declared level: the levels are tenths and `INSTANCES_PER_BAG`
    is thirty, so nothing rounds. Refuses above the cap rather than clamping,
    because a run silently held at 0.5 while its record says 0.7 is a table
    nobody can attribute.
    """
    if not 0.0 <= rate < NOISE_CAP:
        raise ValueError(
            f"noise rate {rate} is outside [0, {NOISE_CAP}); past the cap the "
            f"bag's own class stops being the plurality of its instances and its "
            f"label stops being defensible"
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

BACKBONE = "resnet18"
PRETRAINED = True          # ImageNet weights, identical on both sides
FEATURE_DIM = 512          # resnet18's pooled width, the head's input

#: Hidden width of the relevance selector's R_phi of Eq. (15). It has no
#: counterpart on the other side, so it is declared and never tuned.
ATTENTION_WIDTH = 128

#: gamma of Eq. (15): the weight of the within-bag consensus term against the
#: learned relevance R_phi. The proposal fixes it as a hyperparameter during
#: training but gives it no value. Zero sits at the neutral where Eq. (15)
#: reduces to R_phi with the l1-normalized v_R (r21 l.456) -- not to a bare,
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

#: sigma of Eq. (14): the one bandwidth of the one instance kernel. r21 l.715
#: is explicit that a single sigma governs the three blocks of a class's mixed
#: matrix "ya que los tres derivan del mismo kernel de instancia", and l.458
#: says the attention consensus reuses that same kernel and its bandwidth
#: rather than a second one of its own -- so one constant serves the
#: consensus term inside the attention logit (Eq. 15), the top-k selection
#: ranking, every bag kernel block Eq. (23) builds (K_ss, K_st, K_tt) -- the
#: three blocks of a class's mixed matrix are defined there, not at Eqs.
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

#: The stabilizer inside a logarithm, shared by two call sites rather than
#: private to either: `wiring.py` passes it as eps_src, `source_loss`'s (and
#: `source_bound`'s) own stabilizer for Eq. (21) -- what fixes B_src's scale --
#: and, for an instance-unit arm's `forward`, it also keeps the averaged
#: instance distribution off zero before it is logged. A prior version of this
#: comment said Eq. (21) "normalizes its own", which read as though this
#: constant played no part in it; it does, at the one call site that matters:
#: `wiring.py`'s `source_loss(..., config.EPSILON)`.
EPSILON = 1e-8

#: eps_loc: the stabilizer of Eq. (38)'s `local_loss`, declared here rather
#: than left at that function's own default. `local_loss(squared_distances,
#: target_weights, epsilon=1e-8)` has a default of its own -- one a caller
#: could omit without anyone noticing which value governed the run. Passing
#: this explicitly from `wiring.py` is what makes the number a declared fact
#: of the comparison rather than an implicit one; it happens to equal
#: `EPSILON` today, and the two are kept as separate names because they
#: stabilize two different equations and nothing requires them to move
#: together.
EPSILON_LOCAL = 1e-8

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
CEILING_RANGE = (1e-4, 1.0)

# ---------------------------------------------- las otras cinco dimensiones
#
# The search covers six dimensions now, not one: the ramp ceiling above, and the
# five below. Every one is an Eq. (39)/Eq. (15)/Eq. (16)/Eq. (28) constant this
# file otherwise declares fixed -- `lambda_glob`/`lambda_loc` are NOT among them,
# because both come out of the shared ramp (`harness.ramp`, then
# `total_objective(..., coefficient, coefficient)`) and a second, independent
# coefficient for each would be a change to Eq. (39) itself, not a search over
# it. `ATTENTION_WIDTH` also stays out and fixed: it sizes `R_phi`'s hidden
# layer, an architectural choice with no equation attached, never a
# hyperparameter Eq. (15) itself names.
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
SEARCH_TRIALS = 30
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
# adapted arm's target forward runs through the encoder exactly as the source
# forward does, normalization layers in training mode included, so it learns
# from both domains like the rest of the model. There is no separate
# "frozen"/"live" knob here -- the only normalization fact any arm declares is
# the floor's own (Decision 2): it never encodes a target image during
# training at all, so it never reaches a normalization layer with one either.

#: selection   which instances of a bag the arm is allowed to look at: None for
#:             all of them, or a rule that keeps `SELECT_K` of the `INSTANCES_PER_BAG`
#:
#: The display name is what every table and figure prints, and the asterisks in it
#: count what the arm lacks: `CREDA*` is CREDA without the confidence weighting,
#: `MIL-CREDA**` lacks the local term and the weighting, `MIL-CREDA*` lacks only
#: the local term, and an unmarked name is the complete method.

#: `normalization` is `"shared"` for every arm except `GN`: the target forward
#: passes through the encoder's normalization layers exactly as the source
#: forward does, in training mode, learning from both domains like the rest of
#: the model (the `_target_embeddings` doctrine above `wiring.Arm` states this).
#: `GN` is the one exception -- `"sourceBatch"` -- and it is the only thing that
#: separates it from `G`: its normalization layers never learn from the target at
#: all. The target forward of that one step is normalized with the CURRENT
#: SOURCE BATCH statistics of that same step (never the accumulated running
#: statistics -- those were measured to scale the two domains apart before this
#: arm existed), and the running statistics the source forward already updated
#: are left exactly as the source forward last set them: the target forward
#: never calls into a form that would update them again.
ARMS = [
    {"id": "B", "name": "MIL-Baseline", "label": "source-only (bags)",
     "unit": "bag", "adaptation": None, "weighting": False, "local": False,
     "attention": "learned", "selection": None, "normalization": "shared"},
    {"id": "E", "name": "MIL-CREDA**", "label": "MIL-CREDA global, unweighted",
     "unit": "bag", "adaptation": "milcreda", "weighting": False, "local": False,
     "attention": "learned", "selection": None, "normalization": "shared"},
    {"id": "F", "name": "MIL-CREDA*", "label": "MIL-CREDA global, weighted",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": False,
     "attention": "learned", "selection": None, "normalization": "shared"},
    {"id": "G", "name": "MIL-CREDA", "label": "MIL-CREDA full (global + local)",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned", "selection": None, "normalization": "shared"},
    {"id": "GN", "name": "MIL-CREDA-GN",
     "label": "MIL-CREDA full, target normalized by source-batch statistics",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned", "selection": None, "normalization": "sourceBatch"},
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
FLOOR_OF = {"G": "B", "F": "B", "E": "B", "GN": "B"}

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


#: The other five dimensions the six-dimensional search explores beside the
#: ceiling, in the order `harness.search_ceilings_trials` declares them. Named
#: once here so `hyper_by_transfer_on_record`/`hyper_pooled_on_record` and
#: `harness.Reduction`'s own field list cannot drift apart on which five they
#: mean.
HYPER_DIMENSIONS = ("rampDelta", "kernelSigma", "attentionGamma",
                    "attentionTemperature", "tauLocal")


def hyper_by_transfer_on_record(
        pilot: "bool | None" = None) -> dict[str, dict[str, dict[str, float]]]:
    """The OTHER five searched dimensions' per-transfer winners, read from the
    same record `ceilings_by_transfer_on_record` reads its own pick from.

    `harness.search_ceilings_trials` writes `rampDelta`/`kernelSigma`/
    `attentionGamma`/`attentionTemperature`/`tauLocal` into `perTransfer[label]`
    beside `ceiling` itself -- that transfer's own winning trial, entire, not
    only its ceiling. This reads that same sub-dict back, narrowed to
    `HYPER_DIMENSIONS`, keyed the same way `ceilingsByTransfer` is
    (`{family: {label: {dim: value}}}`). A record written before this stretch's
    six-dimensional search simply has no `perTransfer[label][dim]` for any of
    the five, and an absent dimension is read exactly like an absent transfer:
    the caller falls back to the pooled winner, and `harness.hyper_for` falls
    back further, to the declared `config` constant.
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
    """The OTHER five searched dimensions' pooled winner, one per family --
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
    "steps.py: config.PRODUCT": (
        "no es un destino sino la raiz contra la que `entradas_faltantes` "
        "resuelve las raices que un paso DECLARA en `reads`, que ya vienen "
        "escritas en la ortografia de la corrida completa: la escala esta "
        "adentro de la raiz y no al lado de ella, asi que una coordenada aca "
        "seria una segunda respuesta a una pregunta que la raiz ya contesto"),
    "harness.py: config.PRODUCT / tables.MECHANISM_RECORD": (
        "la escritura del mismo registro `Benchmark_Results.ipynb` ya lee sin "
        "escala, excusada abajo (`Benchmark_Results.ipynb: config.PRODUCT / "
        "tables.MECHANISM_RECORD`): `run_mechanism_sweep` multiplexa limpio "
        "y contaminado ADENTRO del JSON (`clean`/`noisy`), nunca por "
        "directorio ni por nombre de archivo, así que no hay coordenada de "
        "tasa que llevar en la ruta -- y `pilot`/`kind` tampoco: la sección "
        "4 no declara un `Reduction` de ensayo propio, corre siempre a la "
        "escala que `reduction.epochs`/`seeds` diga"),
    "promote.py: config.PRODUCT / '.remote-execution' / 'campaign'": (
        "es donde el backend remoto desempaqueta lo que devuelve, antes de que "
        "nada haya leido una reduccion: la escala de lo que viene adentro la "
        "deciden los sellos de cada shard, no el directorio que los recibe"),
    "Benchmark_Results.ipynb: config.PRODUCT / 'Results' / 'figures'": (
        "el directorio de figuras del cuaderno de resultados: un solo arbol "
        "compartido, sin segmento de pilot/full ni de tasa -- distingue limpio "
        "de contaminado por NOMBRE de archivo (`latent_grid_clean.pdf` / "
        "`latent_grid_noisy.pdf`) y no por directorio. Esta declaracion "
        "registra el hecho, no lo avala: `Benchmark_Results.ipynb` no es un archivo "
        "que este stretch posea (ver el mapa de propiedad del cambio), asi que "
        "si una figura de ensayo y una de la corrida completa debieran vivir "
        "aparte, corregirlo es trabajo de quien sea dueno de ese cuaderno"),
    "Benchmark_Results.ipynb: config.PRODUCT / tables.MECHANISM_RECORD": (
        "una lectura, no una escritura -- `tables.MECHANISM_RECORD` es la "
        "cadena fija que ese modulo ya declara "
        "(`Results/Benchmark/attention_mechanisms.json`), sin escala propia, "
        "la misma forma que `harness.py: config.CEILINGS_RECORD` ya tiene "
        "excusada arriba: nombra el registro de la corrida COMPLETA y nunca "
        "el de ensayo"),
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
