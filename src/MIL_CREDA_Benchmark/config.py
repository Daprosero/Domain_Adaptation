"""Every number this comparison was decided with, in one place.

Two constants separate the pilot from the full configuration — `EPOCHS` and
`SEEDS` — and nothing else. That is deliberate: the pilot has to exercise the
same path the full run will, or it proves nothing about it.

    pilot:  EPOCHS = 3,  SEEDS = [0]           54 runs
    full:   EPOCHS = 20, SEEDS = range(30)   1620 runs

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

ARMS = [
    {"id": "A", "label": "source-only (instances)",
     "unit": "instance", "adaptation": None, "weighting": False, "local": False,
     "attention": None},
    {"id": "B", "label": "source-only (bags)",
     "unit": "bag", "adaptation": None, "weighting": False, "local": False,
     "attention": "learned"},
    {"id": "C", "label": "CREDA, unweighted",
     "unit": "instance", "adaptation": "creda", "weighting": False, "local": False,
     "attention": None},
    {"id": "D", "label": "CREDA, weighted (full)",
     "unit": "instance", "adaptation": "creda", "weighting": True, "local": False,
     "attention": None},
    {"id": "E", "label": "MIL-CREDA global, unweighted",
     "unit": "bag", "adaptation": "milcreda", "weighting": False, "local": False,
     "attention": "learned"},
    {"id": "F", "label": "MIL-CREDA global, weighted",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": False,
     "attention": "learned"},
    {"id": "G", "label": "MIL-CREDA full (global + local)",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned"},
    {"id": "H1", "label": "MIL-CREDA full, uniform attention",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "uniform"},
    {"id": "H2", "label": "MIL-CREDA full, random attention",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "random"},
]

ARMS_BY_ID = {arm["id"]: arm for arm in ARMS}

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
    ("H1", "G", "what learned attention buys over a uniform bag mean"),
    ("H2", "G", "what learned attention buys over an arbitrary fixed one"),
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
#: The two complete methods get five, because every latent measurement is reported
#: with its dispersion and one model gives none. Their floors get three: they are
#: the reference the adapted space is compared against, and a coarser estimate is
#: enough for a reference. At roughly 45 MB each this is about 4.3 GB, all of it
#: local and ignored by git.
CHECKPOINTS = {"A": 3, "B": 3, "D": 5, "G": 5}

#: Which floor each adapted arm is read against: same unit, same everything, with
#: the adaptation term switched off.
FLOOR_OF = {"D": "A", "G": "B"}

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
