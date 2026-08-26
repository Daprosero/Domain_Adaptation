"""Every number this comparison was decided with, in one place.

Two constants separate the pilot from the full configuration — `EPOCHS` and
`SEEDS` — and nothing else. That is deliberate: the pilot has to exercise the
same path the full run will, or it proves nothing about it.

    pilot:  EPOCHS = 3,  SEEDS = [0]           60 runs
    full:   EPOCHS = 20, SEEDS = range(30)   1800 runs

Read the header of any summary before reading its numbers. A table produced with
one seed carries a threshold of zero, so every row declares a winner from a bare
difference; the summary says so in its header and in its JSON, and that stamp is
the only thing standing between a pilot and a misquote.
"""

from __future__ import annotations

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

# ------------------------------------------------------------------- material

REVISION = "research-concept-r17.md"

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

# -------------------------------------------------------------------- network

BACKBONE = "resnet18"
PRETRAINED = True          # ImageNet weights, identical on both sides
FEATURE_DIM = 512          # resnet18's pooled width, the head's input

#: Hidden width of the relevance selector of Eq. (14). It has no counterpart on
#: the other side, so it is declared and never tuned.
ATTENTION_WIDTH = 128

#: The local temperature of Eq. (28). Also without a counterpart in CREDA, so it
#: is fixed at one and reported as fixed rather than chosen.
TAU_LOCAL = 1.0

#: The stabilizer inside a logarithm. Eq. (18) normalizes its own; this one only
#: keeps the averaged instance distribution off zero before it is logged.
EPSILON = 1e-8

# ------------------------------------------------------- schedules, shared by all

#: The adaptation coefficient is one object with two knobs — how fast it grows
#: and how far — and `CREDA.schedules.creda_ramp` is where both live. Both families
#: drive it from there, so the two sides of the comparison share one
#: implementation rather than two copies of the same formula.
#:
#: Nothing multiplies the schedule afterwards. Three factors, two of them pinned
#: at one and therefore invisible, is how a scale error hides: the supervised
#: term of Eq. (18) spent a revision at 18.42 times its stated weight because the
#: coefficient was spread across places nobody read together.

#: How far. One is the neutral of Eq. (39), and from r17 that is a statement about
#: the objective rather than about this setting. Eq. (36) normalizes the global
#: score by the conservative bounds, Eq. (38) bounds the local term, and Eq. (18)
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
#: every adapted arm scored exactly what its own floor scored. A comparison
#: against a CREDA whose term changes no decision is a comparison against the
#: source-only floor with extra wall time, so both sides run at this ceiling and
#: the report says so rather than leaving it to whoever knows CREDA to notice.
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
SEARCH_ARMS = {"creda": "D", "milcreda": "G"}

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
SEARCH_TRANSFERS = [("M", "U"), ("S", "M")]

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

#: CREDA's own `creda_lr_special` and the decay of `get_eta`.
LR = 1e-3
LR_ALPHA = 20
LR_BETA = 0.75

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

#: selection   which instances of a bag the arm is allowed to look at: None for
#:             all of them, or a rule that keeps `SELECT_K` of the `INSTANCES_PER_BAG`
#:
#: The display name is what every table and figure prints, and the asterisks in it
#: count what the arm lacks: `CREDA*` is CREDA without the confidence weighting,
#: `MIL-CREDA**` lacks the local term and the weighting, `MIL-CREDA*` lacks only
#: the local term, and an unmarked name is the complete method.

ARMS = [
    {"id": "A", "name": "Baseline", "label": "source-only (instances)",
     "unit": "instance", "adaptation": None, "weighting": False, "local": False,
     "attention": None, "selection": None},
    {"id": "C", "name": "CREDA*", "label": "CREDA, unweighted",
     "unit": "instance", "adaptation": "creda", "weighting": False, "local": False,
     "attention": None, "selection": None},
    {"id": "D", "name": "CREDA", "label": "CREDA, weighted (full)",
     "unit": "instance", "adaptation": "creda", "weighting": True, "local": False,
     "attention": None, "selection": None},
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
    {"id": "SU", "name": "MIL-CREDA-U", "label": "MIL-CREDA full, regular selection",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned", "selection": "regular"},
    {"id": "SA", "name": "MIL-CREDA-A", "label": "MIL-CREDA full, arbitrary selection",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned", "selection": "arbitrary"},
    {"id": "SK", "name": "MIL-CREDA-K", "label": "MIL-CREDA full, top-K selection",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned", "selection": "topk"},
]

ARMS_BY_ID = {arm["id"]: arm for arm in ARMS}

#: The display order of every table and figure: the order the arms are declared in.
ARM_ORDER = [arm["id"] for arm in ARMS]
NAME_OF = {arm["id"]: arm["name"] for arm in ARMS}

#: How many of a bag's instances a selecting arm keeps. The three selecting arms
#: hold this budget fixed and differ only in the rule that picks them, so the rung
#: between any two of them is attributable to the rule. `SK -> G` is the separate
#: question of what the budget itself costs.
SELECT_K = 10

#: The arbitrary selection is drawn once from a generator of its own and never
#: again. Two reasons: re-drawing every step would test noise rather than test the
#: rule, and drawing from the training generator would shift every later draw, so
#: the rung would credit the selection with what the offset did.
SELECTION_SEED = 20250812

#: What each rung of the ladder reads. A comparison is only attributable when its
#: two arms differ in one thing, so the pairs are written out rather than left to
#: whoever reads the table.
LADDER = [
    ("A", "B", "qué compra la representación por bolsas, con la adaptación apagada"),
    ("A", "C", "qué compra la alineación de CREDA, sin ponderar"),
    ("C", "D", "qué compra la ponderación por confianza en CREDA"),
    ("B", "E", "qué compra el término global, sin ponderar"),
    ("E", "F", "qué compra la ponderación por confianza en MIL-CREDA"),
    ("F", "G", "qué compra la correspondencia local"),
    ("C", "E", "el mismo peldaño, construido de dos maneras: sin ponderar"),
    ("D", "F", "el mismo peldaño, construido de dos maneras: ponderado"),
    ("D", "G", "mano a mano, cada método completo"),
    # The three below hold the instance budget at SELECT_K and differ only in the
    # rule that spends it, except the last, which is the budget itself.
    ("SU", "SK", "qué compra la selección por atención frente a una regular"),
    ("SA", "SK", "qué compra frente a una selección fija arbitraria"),
    ("SK", "G", "qué cuesta quedarse solo con las mejores instancias frente a quedarse con todas"),
]

#: Which direction wins each dimension. The two costs and the parameter count are
#: reported and not contested: peak memory as measured is Python's heap and says
#: nothing about tensors, and a parameter count is a fact rather than a contest.
HIGHER, LOWER, DESCRIPTIVE = "higher", "lower", None
DIMENSIONS = {
    "targetAccuracy": HIGHER,
    "sourceAccuracy": HIGHER,
    "seconds": LOWER,
    "contribution": DESCRIPTIVE,
    #: The supervised magnitude and the adaptation's share of the objective.
    #: `contribution` alone cannot separate a term that commanded nothing from a
    #: term that was scaled to nothing, and both print as a small number. The two
    #: are descriptive because neither has a better direction: a large share is
    #: not a better method, it is a differently balanced objective, and a rung
    #: whose two arms differ in share is a rung whose reading has to say so.
    "supervised": DESCRIPTIVE,
    "adaptationShare": DESCRIPTIVE,
    "peakMiB": DESCRIPTIVE,
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
FLOOR_OF = {"D": "A", "G": "B", "F": "B", "E": "B", "SU": "B", "SA": "B", "SK": "B",
            "C": "A"}

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

#: The columns of the latent grid after the shared original space: both floors,
#: both CREDA, all three MIL-CREDA. The floors are what make the rest readable —
#: "aligned" cannot be seen without a "not aligned" beside it.
#:
#: Both floors and not one, because whether they are redundant is a measurement
#: and the measurement says they are not: drawn at the instance level on the pilot,
#: their distance ratios differ by up to 0.38 and their domain separabilities by
#: 0.07. They train the same encoder through different objectives — cross-entropy
#: per instance against cross-entropy per bag through the attention pooling — so
#: their instance embeddings had no reason to agree. `latent.floors_agree` runs
#: that check on every campaign rather than leaving it as a belief.
#:
#: The selecting arms are left out because they differ from `G` in their instance
#: budget rather than in what they align: a phase-one question, not a picture.
LATENT_PANELS = ["A", "B", "C", "D", "E", "F", "G"]

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


def ceilings_on_record() -> dict[str, float]:
    """The searched ceilings, read from the record the search wrote.

    Read and not remembered, like everything else here. A constant typed in by
    hand would be a second source of truth for a measured value: it goes stale in
    silence the first time the search is re-run and is believed anyway.

    Public, and re-read on every call, because `CEILINGS` below is filled once at
    import. A caller that runs the search in the same process — a notebook, which
    is the only place that ever does — would otherwise hold the empty mapping this
    module was imported with and hand it to a campaign that refuses it.
    """
    if not CEILINGS_RECORD.exists():
        return {}
    import json as _json
    found = _json.loads(CEILINGS_RECORD.read_text(encoding="utf-8"))
    return {family: entry["ceiling"] for family, entry in found.items()}


def ceilings_by_transfer_on_record() -> dict[str, dict[str, float]]:
    """The per-transfer picks, read from the same record.

    A record written before this key existed simply has none, and that is not an
    error: an absent mapping makes every transfer fall back to the pooled winner,
    which is exactly what such a record meant when it was written. Read on every
    call, for the same reason `ceilings_on_record` is.
    """
    if not CEILINGS_RECORD.exists():
        return {}
    import json as _json
    found = _json.loads(CEILINGS_RECORD.read_text(encoding="utf-8"))
    return {family: dict(entry.get("byTransfer") or {})
            for family, entry in found.items()}


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
    runs = (len(SEARCH_ARMS) * len(CEILING_GRID)
            * len(SEARCH_TRANSFERS) * len(SEARCH_SEEDS))
    return {
        "families": len(SEARCH_ARMS),
        "ceilings": len(CEILING_GRID),
        "transfers": len(SEARCH_TRANSFERS),
        "seeds": len(SEARCH_SEEDS),
        "runs": runs,
        "epochs": SEARCH_EPOCHS,
        "atRequiredScale": (SEARCH_EPOCHS >= FULL_SEARCH_EPOCHS
                            and len(SEARCH_SEEDS) >= FULL_SEARCH_SEEDS),
    }
