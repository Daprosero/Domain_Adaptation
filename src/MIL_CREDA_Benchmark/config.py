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

REVISION = "research-concept-r16.md"

#: Every domain supplies its own bags; a transfer names which is source and which
#: is target. All three hold far more than the 3000 images a domain contributes.
DOMAINS = {"M": "MNIST", "U": "USPS", "S": "SVHN"}

TRANSFERS = [("M", "U"), ("U", "M"), ("M", "S"), ("S", "M"), ("U", "S"), ("S", "U")]

CLASSES = 10

#: A bag is a subject: 30 instances of one class, drawn at random inside it. No
#: instance carries a label of its own, and no instance comes from another class.
INSTANCES_PER_BAG = 30

#: Ten per class, so the draw is stratified and balanced rather than proportional.
#: The local correspondence is undefined for a class with no source bag, so class
#: coverage is a requirement of the formulation and not a convenience.
BAGS_PER_CLASS = 10
BAGS_PER_DOMAIN = BAGS_PER_CLASS * CLASSES          # 100
IMAGES_PER_DOMAIN = BAGS_PER_DOMAIN * INSTANCES_PER_BAG  # 3000

#: Two roles, drawn identically in both domains. Nothing is selected by looking at
#: outcomes any more — lambda and the epoch count are both fixed — so there is no
#: validation role to carve. The 36 evaluation bags are what let 30 seeds resolve
#: three points; an 80/20 would leave 20 and resolve five.
TRAIN_BAGS = 64
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

#: The constant the ramp is multiplied by. CREDA's own value for this setting is
#: `creda_lambda_special` = 1e-4, and it was measured to be inert here: at a
#: ceiling of 1e-4 — and at 1e-2 and 1e-1 — every adapted arm scored exactly what
#: its own floor scored, so the term was computed and changed no decision.
#:
#: One is the smallest ceiling at which both terms move the outcome. The ramp is
#: untouched: same shape, same delta, still zero in the first epoch. Only the
#: ceiling it climbs to changes, from a ten-thousandth to one.
#:
#: The logarithm inside the Renyi entropy already keeps both terms the same order
#: as the supervised loss, so one does not let either overwhelm it. What the
#: logarithm does not do is fix the scale: it bounds the score by ln n, a window
#: that widens with the bags a class contributes — measured at (-3.178, 2.485)
#: for twelve bags a side. Eq. (36) normalizes by exactly those bounds, which is
#: why MIL-CREDA's term is in [0, 1] whatever the class size.
LAMBDA_CONST = 1.0

#: CREDA's own warm-up, `get_lambda`. Zero in the first epoch, effectively one
#: afterwards. Applied to both sides so the schedule is not one more difference.
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
    ("A", "B", "what the bag representation buys, with adaptation off"),
    ("A", "C", "what CREDA's alignment buys, unweighted"),
    ("C", "D", "what confidence weighting buys in CREDA"),
    ("B", "E", "what the global term buys, unweighted"),
    ("E", "F", "what confidence weighting buys in MIL-CREDA"),
    ("F", "G", "what the local correspondence buys"),
    ("C", "E", "the same rung, built two ways: unweighted"),
    ("D", "F", "the same rung, built two ways: weighted"),
    ("D", "G", "head to head, each method complete"),
    # The three below hold the instance budget at SELECT_K and differ only in the
    # rule that spends it, except the last, which is the budget itself.
    ("SU", "SK", "what attention-based selection buys over a regular one"),
    ("SA", "SK", "what it buys over an arbitrary fixed selection"),
    ("SK", "G", "what keeping only the top instances costs against keeping all"),
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
    }
