"""The figures, executed rather than described.

Nothing in `figures.py` and none of the grids in `latent.py` was ever run by a
test: every claim about them rested on the constants they read. A constant naming
three arms does not make a grid draw three columns, and `LATENT_PANELS` being
right while `latent_grid` drops the shared original column are the same green
suite.

The drawing itself is stubbed exactly twice -- the UMAP embedding and the trained
checkpoints -- because those need a run on disk and a fitted model, and neither is
what any claim here is about. What is not stubbed is the layout: how many panels,
in which order, out of how many figures, and which column comes from a model at
all.
"""

from __future__ import annotations

import json
import re

import numpy
import pytest
import torch
import torch.nn as nn

from MIL_CREDA_Benchmark import bags, config, figures, latent, tables, wiring

REPORT = config.REPOSITORY / "MIL-CREDA" / "Notebooks" / "Benchmark_Report_v1.ipynb"

TRANSFERS = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS[:3]]


# --------------------------------------------------------------- the noise axis

def _levels(tmp_path, monkeypatch, per_level: dict[float, dict[str, dict[str, float]]]):
    """A results tree per declared level, in the shape the sweep writes one."""
    monkeypatch.setattr(config, "PRODUCT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    for rate, per_arm in per_level.items():
        root = config.results_for(rate, "curve")
        root.mkdir(parents=True, exist_ok=True)
        lines = []
        for arm, values in per_arm.items():
            lines.append(json.dumps({"arm": arm, "transfer": "M->U", **values}))
        (root / "runs.jsonl").write_text("\n".join(lines), encoding="utf-8")
        (root / "summary.json").write_text(
            json.dumps({"reduction": {"labelNoise": rate}, "grid": {}}),
            encoding="utf-8")


def _drawn(figure) -> dict[str, tuple[list[float], list[float]]]:
    """Every labelled series of a one-panel figure, by its legend label."""
    axis = figure.axes[0]
    return {line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
            for line in axis.lines if not line.get_label().startswith("_")}


def test_the_degradation_figure_draws_the_share_against_rho_and_not_only_accuracy(
        tmp_path, monkeypatch) -> None:
    """Two instruments over one axis, and the second is the one that says which
    of the two failures happened.

    Accuracy against rho for every arm says that something fell. The share of the
    objective the adaptation term commanded, against the same rho, is what
    separates a term that stopped working from a term that was never given any
    weight to work with -- and both instruments are already recorded, so the axis
    is the only thing being added.

    Reachable red: draw `targetAccuracy` whatever the caller asked for, and the
    share series comes out as the accuracy one.
    """
    _levels(tmp_path, monkeypatch, {
        0.0: {"A": {"targetAccuracy": 0.80, "adaptationShare": 0.0},
              "G": {"targetAccuracy": 0.82, "adaptationShare": 0.10}},
        0.2: {"A": {"targetAccuracy": 0.60, "adaptationShare": 0.0},
              "G": {"targetAccuracy": 0.75, "adaptationShare": 0.30}},
        0.4: {"A": {"targetAccuracy": 0.40, "adaptationShare": 0.0},
              "G": {"targetAccuracy": 0.70, "adaptationShare": 0.55}},
    })

    accuracy = _drawn(figures.noise_curves("targetAccuracy"))
    assert accuracy[config.NAME_OF["G"]] == ([0.0, 0.2, 0.4], [0.82, 0.75, 0.70])
    assert accuracy[config.NAME_OF["A"]] == ([0.0, 0.2, 0.4], [0.80, 0.60, 0.40])

    share = _drawn(figures.noise_curves("adaptationShare"))
    assert share[config.NAME_OF["G"]] == ([0.0, 0.2, 0.4], [0.10, 0.30, 0.55])
    # every arm that carries the term is on the share figure, and the axis is
    # the same rate axis the accuracy figure ran over
    for arm in (a["id"] for a in config.ARMS if a["adaptation"]):
        if config.NAME_OF[arm] in accuracy:
            assert config.NAME_OF[arm] in share


def test_the_degradation_figure_never_draws_a_level_that_did_not_run(
        tmp_path, monkeypatch) -> None:
    """A blank axis reads as a flat result, so a figure with nothing behind it
    says so in words instead of drawing."""
    _levels(tmp_path, monkeypatch, {})
    figure = figures.noise_curves("targetAccuracy")
    assert not _drawn(figure)
    assert any("ningún nivel" in text.get_text() for text in figure.axes[0].texts)


def test_the_figure_is_written_where_it_is_asked_for_and_only_as_a_pdf(
        tmp_path, monkeypatch) -> None:
    """One extension, so two versions of one figure cannot coexist unnoticed."""
    _levels(tmp_path, monkeypatch,
            {0.0: {"G": {"targetAccuracy": 0.8, "adaptationShare": 0.1}}})
    figures.noise_curves("targetAccuracy", path=tmp_path / "curva" / "ruido.png")
    written = sorted(p.name for p in (tmp_path / "curva").iterdir())
    assert written == ["ruido.pdf"]


def test_the_contribution_panel_reports_the_realized_share_arm_by_arm(
        tmp_path) -> None:
    """The coefficient is fixed at the neutral for every arm, and fixing the
    coefficient does not fix the share: what each term actually commands of its
    own objective has to be reported per arm, or a term scaled to irrelevance is
    inferred rather than seen.

    The curve is the median across seeds with an interquartile band -- never one
    seed's trajectory, which cannot say whether the shape is the method's or the
    draw's.

    Reachable red: drop an arm from the panel, or draw `supervised` where the
    realized contribution belongs.
    """
    runs = tmp_path / "runs.jsonl"
    steps = 4
    shares = {"C": 0.03, "D": 0.20, "G": 0.99}
    with runs.open("w", encoding="utf-8") as handle:
        for arm, share in shares.items():
            for seed in range(3):
                curve = [{"contribution": share + seed / 100,
                          "supervised": 1.0 - share,
                          "adaptation": share} for _ in range(steps)]
                handle.write(json.dumps(
                    {"transfer": "M->U", "arm": arm, "seed": seed,
                     "curve": curve}) + "\n")

    figure = figures.contribution_curves(tmp_path / "contribution.pdf",
                                         arms=tuple(shares), runs=runs)
    drawn = {line.get_label(): list(line.get_ydata())
             for line in figure.axes[0].lines if not line.get_label().startswith("_")}

    assert set(drawn) == {config.NAME_OF[a] for a in shares}
    for arm, share in shares.items():
        # the median of the three seeds, and the whole trajectory rather than a
        # single number
        assert drawn[config.NAME_OF[arm]] == [pytest.approx(share + 0.01)] * steps
    # a term at 0.03 beside one at 0.99 is the reading the panel exists for
    assert min(min(v) for v in drawn.values()) < 0.1 < max(max(v) for v in drawn.values())


# ------------------------------------------------------------- the display seed

def test_the_display_seed_is_the_median_of_the_across_arm_mean() -> None:
    """One draw for the whole grid, chosen by a rule that favours no arm.

    Each arm's own median seed would differ in the method *and* in the draw --
    and because the seed fixes the partition, two panels would not even share
    their bags. So the rule reads every arm at once: the seed whose across-arm
    mean is the median.

    The fixture is built so the two rules disagree. `G`'s own median seed is 2
    and the across-arm median is seed 1, which is what the grid has to draw.

    Reachable red: take the best seed instead of the median, or rank by the
    headline arm's own accuracy.
    """
    runs = [
        {"seed": 0, "arm": "A", "targetAccuracy": 0.10},
        {"seed": 0, "arm": "G", "targetAccuracy": 0.20},   # mean 0.15  (lowest)
        {"seed": 1, "arm": "A", "targetAccuracy": 0.55},
        {"seed": 1, "arm": "G", "targetAccuracy": 0.65},   # mean 0.60  (median)
        {"seed": 2, "arm": "A", "targetAccuracy": 0.95},
        {"seed": 2, "arm": "G", "targetAccuracy": 0.55},   # mean 0.75  (highest)
    ]
    assert latent.display_seed(runs) == 1
    # G's own ordering would have named a different draw, which is the rule this
    # one replaces rather than a coincidence of the fixture.
    only_g = sorted((r for r in runs if r["arm"] == "G"),
                    key=lambda r: r["targetAccuracy"])
    assert only_g[len(only_g) // 2]["seed"] == 2

    # and it does not depend on the order the runs arrive in
    assert latent.display_seed(list(reversed(runs))) == 1


def test_without_a_single_run_there_is_no_seed_to_display() -> None:
    """Zero is a seed the campaign ran, so returning it would be a wrong answer
    rather than a missing one."""
    with pytest.raises(ValueError):
        latent.display_seed([])


# ------------------------------------------------------ the grid drawn stratified

def test_every_panel_draws_the_same_budget_stratified_by_class() -> None:
    """A bag-unit arm contributes one row per subject and an instance-unit arm
    one per instance, so without this the CREDA columns arrive with thirty times
    the points and the eye reads density as coverage.

    Stratified, and that is the half a head-slice would silently drop: a panel
    missing a class is not a sparser panel, it is a different measurement.

    Reachable red: keep the first `budget` rows instead of drawing per class.
    """
    labels = torch.cat([torch.full((count,), class_id)
                        for class_id, count in enumerate([200, 90, 10])])
    rows = torch.arange(len(labels), dtype=torch.float32).reshape(-1, 1)

    kept_rows, kept_labels = latent.equalize(rows, labels, budget=30, seed=7)
    counts = {int(c): int((kept_labels == c).sum()) for c in kept_labels.unique()}
    assert counts == {0: 10, 1: 10, 2: 10}
    assert len(kept_rows) == len(kept_labels) == 30
    # drawn from the whole class and not from its head
    assert kept_rows.max() > 250


def test_a_class_with_less_than_its_share_is_taken_whole_and_never_padded() -> None:
    labels = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(3, dtype=torch.long)])
    rows = torch.arange(len(labels), dtype=torch.float32).reshape(-1, 1)
    _, kept = latent.equalize(rows, labels, budget=20, seed=7)
    assert int((kept == 1).sum()) == 3


# --------------------------------------------------------------- the latent grid

@pytest.fixture
def stubbed(monkeypatch):
    """The two things a grid needs from a finished run, and nothing else.

    `_embed` is UMAP and `load` is a trained checkpoint; both are replaced because
    no claim here is about either. Everything the claims ARE about -- how many
    panels, which columns, which of them comes from a model -- runs for real.
    """
    seen = {"loaded": [], "original": [], "pairs": [], "units": []}

    def _checkpoint_for(arm, transfer, seed, rate=0.0, pilot=False):
        return {"arm": arm, "transfer": transfer, "seed": seed,
                "source": {}, "target": {}, "weights": "none"}

    class _Bagset:
        eval_idx = torch.arange(4)

    def _load(record, device):
        seen["loaded"].append((record["arm"], record["transfer"]))
        return object(), _Bagset(), _Bagset()

    def _represent(model, bagset, positions, device, unit=None):
        seen["units"].append(unit)
        return torch.arange(40.0).reshape(20, 2), torch.arange(20) % config.CLASSES

    def _original_rows(record, budget, seed):
        seen["original"].append(record["transfer"])
        rows = torch.arange(40.0).reshape(20, 2)
        labels = torch.arange(20) % config.CLASSES
        return rows, labels, rows, labels

    def _embed(rows, seed):
        return numpy.asarray(rows).reshape(len(rows), -1)[:, :2]

    monkeypatch.setattr(latent, "checkpoint_for", _checkpoint_for)
    monkeypatch.setattr(latent, "load", _load)
    monkeypatch.setattr(latent, "represent", _represent)
    monkeypatch.setattr(latent, "original_rows", _original_rows)
    monkeypatch.setattr(latent, "_embed", _embed)
    return seen


def test_the_latent_grid_is_the_shared_original_space_and_then_one_column_per_method(
        stubbed, tmp_path) -> None:
    """`Original` is the images themselves, before any model, and it is the
    reference every trained column is read against: "aligned" cannot be seen
    without a "not aligned" beside it.

    Three things are asserted together because each alone passes while the figure
    is wrong: the column exists, it is first, and it is the only one no model was
    loaded for.

    Reachable red: drop the original column, move it to the end, or draw it from
    a checkpoint like the rest.
    """
    figure = latent.latent_grid(tmp_path / "grid.pdf", config.LATENT_PANELS,
                                TRANSFERS, seed=3, device=torch.device("cpu"))
    axes = figure.axes
    columns = 1 + len(config.LATENT_PANELS)
    assert len(axes) == len(TRANSFERS) * columns

    titles = [axis.get_title() for axis in axes[:columns]]
    assert titles == ["Original", *[config.NAME_OF[a] for a in config.LATENT_PANELS]]

    # one model per (arm, transfer) and not one more: the first column is the
    # material, so nothing was fitted to it
    assert sorted(stubbed["loaded"]) == sorted(
        (arm, transfer) for transfer in TRANSFERS for arm in config.LATENT_PANELS)
    assert stubbed["original"] == TRANSFERS

    # rows are transfers, and each row says which one it is
    assert [axes[row * columns].get_ylabel() for row in range(len(TRANSFERS))] == TRANSFERS


def test_the_latent_grid_keeps_both_floors_as_its_first_trained_columns(
        stubbed, tmp_path) -> None:
    """Whether the two floors are redundant is a measurement `latent.floors_agree`
    makes, and only it may retire one of these columns."""
    figure = latent.latent_grid(tmp_path / "grid.pdf", config.LATENT_PANELS,
                                TRANSFERS[:1], seed=3, device=torch.device("cpu"))
    titles = [axis.get_title() for axis in figure.axes]
    floors = [config.NAME_OF[arm["id"]] for arm in config.ARMS
              if arm["adaptation"] is None and arm["id"] in config.LATENT_PANELS]
    assert len(floors) == 2
    for name in floors:
        assert name in titles


# ---------------------------------------------------------- the bag correspondence

def test_the_bag_figure_is_one_figure_of_three_columns_by_three_rows(
        stubbed, tmp_path, monkeypatch) -> None:
    """One figure, not one file per transfer.

    Three columns instead of six separate files is what makes the panels large
    enough to read, and it is also what lets a reader follow one subject across
    the rung: floor, the same method without the local term, and the complete
    one, side by side in one row.

    Reachable red: emit one figure per transfer -- three files land in the
    directory and the axes of the returned one collapse to a single row.
    """
    def _bag_pairs(model, source, target, device):
        rows = torch.arange(20.0).reshape(10, 2)
        labels = torch.arange(10)
        return {"sourceRows": rows, "targetRows": rows,
                "sourceLabels": labels, "targetLabels": labels,
                "nearest": torch.arange(10), "mass": torch.linspace(0.1, 1.0, 10)}

    monkeypatch.setattr(latent, "bag_pairs", _bag_pairs)
    produced = latent.correspondence_grid(tmp_path / "bolsas.pdf", config.BAG_PANELS,
                                          TRANSFERS, seed=3,
                                          device=torch.device("cpu"))

    assert len(config.BAG_PANELS) == 3 and len(TRANSFERS) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bolsas.pdf"]
    assert len(produced["figure"].axes) == 9
    assert len(produced["scored"]) == 9
    assert {row["arm"] for row in produced["scored"]} == set(config.BAG_PANELS)
    assert {row["transfer"] for row in produced["scored"]} == set(TRANSFERS)

    titles = [axis.get_title() for axis in produced["figure"].axes[:3]]
    assert titles == [config.NAME_OF[a] for a in config.BAG_PANELS]


def test_the_highlighted_subject_is_the_median_of_its_class_and_never_the_best() -> None:
    """The best subject of a class pairs cleanly under every arm, including the
    floor that learned no correspondence at all, and a figure that cannot come
    out wrong is not measuring anything.

    Reachable red: take the last of the ordering instead of its middle.
    """
    reference = {"targetLabels": torch.tensor([0, 0, 0, 1, 1, 1]),
                 "mass": torch.tensor([0.9, 0.1, 0.5, 0.2, 0.8, 0.4])}
    chosen = latent.median_bag_per_class(reference)
    assert chosen[0] == 2 and chosen[1] == 5
    assert chosen[0] != 0, "the best of the class was highlighted"


# ------------------------------------------------ the band, and the third panel

def test_a_loss_curve_is_the_median_across_seeds_with_an_interquartile_band() -> None:
    """The curve a reader sees is the median of the repetitions, and the shading
    around it is the interquartile range of the same repetitions at each step.

    One seed's trajectory cannot say whether a shape is the method's or the
    draw's, and concatenating the repetitions would draw thirty runs as one that
    took thirty times as long. So the reduction happens step by step, across
    seeds, and the quantiles are asserted against numbers computed by hand rather
    than against a second implementation of the same interpolation -- which would
    only prove the two agree.

    Three repetitions, sorted `[a, b, c]` at every step: the median is `b`, and
    the quartiles fall midway to each neighbour because the interpolation walks
    `fraction * (n - 1)` positions along the sorted values.

    Reachable red: return the mean where the median belongs, or widen the band
    to the extremes.
    """
    curves = [[{"supervised": 1.0}, {"supervised": 10.0}],
              [{"supervised": 2.0}, {"supervised": 20.0}],
              [{"supervised": 6.0}, {"supervised": 60.0}]]

    low, mid, high = figures.band(curves, "supervised")

    # step 0 over [1, 2, 6]: q1 = 1 + (2-1)/2, median = 2, q3 = 2 + (6-2)/2
    # step 1 over [10, 20, 60]: the same interpolation, ten times over
    assert mid == [2.0, 20.0]
    assert low == [1.5, 15.0]
    assert high == [4.0, 40.0]

    # the median is not the mean, and the band is not the extremes: this fixture
    # is skewed on purpose so the two rules cannot agree by accident
    assert mid != [pytest.approx(3.0), pytest.approx(30.0)]
    assert (low, high) != ([1.0, 10.0], [6.0, 60.0])

    # ordering across seeds is a reduction, not a lookup
    assert figures.band(list(reversed(curves)), "supervised") == (low, mid, high)


def test_a_repetition_that_stopped_early_truncates_the_band_and_never_extends_it() -> None:
    """Extending the short one would invent steps nobody ran."""
    low, mid, high = figures.band(
        [[{"k": 1.0}, {"k": 2.0}, {"k": 3.0}], [{"k": 5.0}]], "k")
    # one step, because the shorter repetition only reached one -- and the two
    # readings at that step are [1, 5], interpolated exactly as above
    assert (low, mid, high) == ([2.0], [3.0], [4.0])


def test_the_contribution_panel_is_shown_beside_the_other_two_curve_figures() -> None:
    """The third panel is not optional and it is not somewhere else.

    Without it, "the term had no effect" and "the term had no weight" are the
    same picture: the supervised and adaptation curves show the trajectories and
    neither says what share of the objective the term actually commanded. The
    panel is beside them -- the same notebook, after both -- because it is read
    against them and not on its own.

    Read from the notebook itself, which is the only place that decides which
    figures the report shows and in what order.

    Reachable red: drop the contribution cell, or move it above the two curves
    it is meant to be read beside.
    """
    cells = json.loads(REPORT.read_text(encoding="utf-8"))["cells"]
    shown = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for name in ("supervised_curves", "adaptation_curves", "contribution_curves"):
            if f"figures.{name}(" in source:
                shown.append(name)

    assert "contribution_curves" in shown, "the report shows no contribution panel"
    assert {"supervised_curves", "adaptation_curves"} <= set(shown)
    # beside them and after them: the share is read against the trajectories
    assert shown.index("contribution_curves") > shown.index("supervised_curves")
    assert shown.index("contribution_curves") > shown.index("adaptation_curves")
    # and it is a real panel of the module, not a name the notebook alone knows
    assert callable(figures.contribution_curves)


# ------------------------------------------------------- what one cell draws with

def _reference_marker(marker: str):
    """The vertices matplotlib gives that marker, so the assertion names the
    shape rather than an opaque path object."""
    import matplotlib.pyplot as plt

    axis = plt.subplots()[1]
    path = axis.scatter([0.0], [0.0], marker=marker).get_paths()[0].vertices
    plt.close(axis.figure)
    return path


def test_colour_is_the_class_and_the_marker_is_the_domain() -> None:
    """Two channels carrying two different facts, and never the same one twice.

    A panel of this grid answers one question -- did the two domains come
    together class by class -- and that needs both facts at once. Colouring by
    domain would make a perfectly aligned panel and a perfectly collapsed one
    look identical; drawing one marker would leave a reader unable to say which
    cloud is which.

    The target is the side being judged, so it carries the heavier ink: larger
    and edged in dark, against source points that are smaller and
    semi-transparent. At grid size a shape is harder to read than a colour, and
    the domain is what the eye is hunting for.

    Reachable red: colour by `domains`, swap the two markers, or draw both
    domains at one size.
    """
    import matplotlib.pyplot as plt

    axis = plt.subplots()[1]
    embedded = numpy.arange(24.0).reshape(12, 2)
    labels = numpy.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    domains = numpy.array([0.0] * 6 + [1.0] * 6)

    latent._draw_cell(axis, embedded, labels, domains)

    source, target = axis.collections
    assert len(axis.collections) == 2, "one scatter per domain, and only two domains"

    # the marker is the domain
    assert numpy.allclose(source.get_paths()[0].vertices, _reference_marker("o"))
    assert numpy.allclose(target.get_paths()[0].vertices, _reference_marker("^"))
    assert numpy.array_equal(source.get_offsets(), embedded[domains == 0])
    assert numpy.array_equal(target.get_offsets(), embedded[domains == 1])

    # the colour is the class -- each domain's own slice of the labels, and not
    # the domain repeated as a colour
    assert numpy.array_equal(source.get_array(), labels[domains == 0])
    assert numpy.array_equal(target.get_array(), labels[domains == 1])
    assert source.cmap.name == target.cmap.name == "tab10"

    # the target carries the heavier ink: larger, opaquer, and edged in dark
    assert float(target.get_sizes()[0]) > float(source.get_sizes()[0])
    assert target.get_alpha() > source.get_alpha()
    assert len(source.get_edgecolors()) == 0, "source points carry no edge"
    edge = target.get_edgecolors()[0][:3]
    assert edge.max() < 0.5, "the target edge is dark, or the marker does not read"

    plt.close(axis.figure)


def test_every_panel_of_the_grid_is_drawn_at_the_instance_level(
        stubbed, tmp_path) -> None:
    """One unit for every column, bag-unit arms included.

    Every arm encodes instances -- Eq. (13) applies identically in both families
    -- so it is a space they all have, and the only one where every panel carries
    the same number of points. Drawn in each arm's own unit, one point per
    subject would sit beside thirty points per subject and the eye would read the
    statistical unit as coverage.

    Asserted over every panel of the grid and not over the constant: `LATENT_UNIT`
    being right while one column reads it from somewhere else is the same green
    suite.

    Reachable red: pass the arm's own unit at either call site, or drop the
    argument so `represent` falls back to its own default.
    """
    latent.latent_grid(tmp_path / "grid.pdf", config.LATENT_PANELS,
                       TRANSFERS, seed=3, device=torch.device("cpu"))

    # two representations per trained panel -- source and target -- and every one
    # of them at the instance level
    assert len(stubbed["units"]) == 2 * len(TRANSFERS) * len(config.LATENT_PANELS)
    assert set(stubbed["units"]) == {"instance"}
    assert config.LATENT_UNIT == "instance"


# ------------------------------------------- the correspondence, measured for real

class _Encoder(nn.Module):
    """Something with an `output_dim` that maps an instance to a row.

    The encoder is a pretrained resnet18 and no claim below is about it. The bag
    kernel, the correspondence and the projection are what run for real.
    """

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


def _domains() -> tuple[bags.BagSet, bags.BagSet]:
    """Two synthetic domains, one bag per class, the second a shifted first.

    Every class carries a source bag because `total_correspondence` refuses a
    step where one does not: a target bag whose class has no source counterpart
    leaves the correspondence undefined rather than empty.

    The target is the source plus noise rather than an independent draw, and
    that is what makes the correspondence measurable at all. Two unrelated
    clouds pair at chance, the hit count comes out zero, and a test asserting
    zero cannot tell a stopped counter from a correct one. At this offset the
    pairing lands three of ten -- neither nothing nor everything.
    """
    count = config.CLASSES
    source_images = torch.randn(count * config.INSTANCES_PER_BAG, 3, 8, 8,
                                generator=torch.Generator().manual_seed(1))
    target_images = source_images + 0.2 * torch.randn(
        source_images.shape, generator=torch.Generator().manual_seed(7))
    members = torch.arange(source_images.shape[0]).reshape(
        count, config.INSTANCES_PER_BAG)
    positions = torch.arange(count)

    def one(domain: str, images: torch.Tensor) -> bags.BagSet:
        return bags.BagSet(domain, images, members, torch.arange(count),
                           positions, positions, positions, {})

    return one("S", source_images), one("T", target_images)


@pytest.fixture
def trained(monkeypatch):
    """A tiny model over synthetic bags, and the two stubs a grid needs.

    `checkpoint_for` and `load` are replaced because they need a finished run on
    disk, and `_embed` because it is UMAP -- the projection is asserted through
    the coordinates it returns rather than through the coordinates it computes.
    Everything else runs: the encoder, the attention pooling, the bag kernel and
    the correspondence.
    """
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder)
    torch.manual_seed(11)
    pool = wiring.Pool(torch.randn(300, 3, 8, 8),
                       torch.arange(300).reshape(10, 30),
                       torch.arange(10) % config.CLASSES)
    model = wiring.build("G", config.CLASSES, pool, pool)
    source, target = _domains()

    seen = {"embedded": []}

    def _checkpoint_for(arm, transfer, seed, rate=0.0, pilot=False):
        return {"arm": arm, "transfer": transfer, "seed": seed}

    def _load(record, device):
        return model, source, target

    def _embed(rows, seed):
        # A deterministic stand-in for UMAP: still a projection of the very rows
        # it was handed, so what the panel scatters can be compared against it.
        projected = numpy.asarray(rows.numpy(), dtype=float)[:, :2] * 3.0 + 1.0
        seen["embedded"].append((rows.shape[0], projected))
        return projected

    monkeypatch.setattr(latent, "checkpoint_for", _checkpoint_for)
    monkeypatch.setattr(latent, "load", _load)
    monkeypatch.setattr(latent, "_embed", _embed)
    return {"model": model, "source": source, "target": target, "seen": seen}


def test_the_nearest_source_bag_is_found_with_the_bag_kernel_in_the_representation_space(
        trained) -> None:
    """The pairing is the method's own geometry, and it is not a distance the
    method never computes.

    Euclidean distance between the bag representations of Eq. (16) is a different
    quantity from the relevance-weighted bag kernel of Eq. (21), and distance in
    the two-dimensional projection is a third: that one would illustrate UMAP
    rather than the correspondence. All three are available at this point in the
    code, which is why the one that is used has to be asserted rather than
    assumed -- the fixture is built so the three answers disagree.

    The joining line is NOT what is checked here. It is drawn only for an arm
    that declares the local term, so two of the three panels of a row carry none
    and a test over the lines would be asserting the panel choice instead.

    Reachable red: pair by Euclidean distance on `Z`, or take the argmin of the
    kernel instead of its argmax.
    """
    from MIL_CREDA.attention import bag_embedding
    from MIL_CREDA.bag_kernel import bag_kernel_matrix

    model, source, target = trained["model"], trained["source"], trained["target"]
    reading = latent.bag_pairs(model, source, target, torch.device("cpu"))

    # the same kernel, rebuilt from the published pieces rather than from the
    # function under test
    H_s = model.instance_embeddings(source.images[source.members[source.train_idx]])
    H_t = model.instance_embeddings(target.images[target.members[target.eval_idx]])
    pairs_s, pairs_t = model.bags_of(H_s), model.bags_of(H_t)
    sigma = wiring._median_sigma(torch.cat([torch.cat([H for H, _ in pairs_s]),
                                            torch.cat([H for H, _ in pairs_t])]))
    K_st = bag_kernel_matrix(pairs_s, pairs_t, sigma)

    assert torch.equal(reading["nearest"], K_st.argmax(dim=0))

    # and the two geometries the method does not use answer differently, so the
    # assertion above is separating them rather than restating a coincidence
    Z_s = torch.stack([bag_embedding(H, w) for H, w in pairs_s])
    Z_t = torch.stack([bag_embedding(H, w) for H, w in pairs_t])
    euclidean = torch.cdist(Z_t, Z_s).argmin(dim=1)
    projected = torch.cdist(Z_t[:, :2], Z_s[:, :2]).argmin(dim=1)
    assert not torch.equal(reading["nearest"], euclidean)
    assert not torch.equal(reading["nearest"], projected)

    # nearest, not farthest
    assert not torch.equal(reading["nearest"], K_st.argmin(dim=0))
    # the representation space and not the projection: the kernel ran over the
    # instance embeddings the model produced, in their full width
    assert H_s.shape[-1] == model.encoder.output_dim > 2


def test_the_measured_correspondence_hit_rate_is_printed_with_the_figure(
        trained, tmp_path) -> None:
    """A number beside the drawing, so the panel carries a reading and not an
    impression.

    Whether the highlighted subject landed with its own class is countable, and
    counted it is comparable from one row to the next -- which is exactly what a
    reader cannot do by eye across nine panels. The count comes out of the same
    call that draws them, so the table and the figure cannot describe different
    runs.

    Reachable red: stop counting the hits in the grid, or drop the column from
    the table that prints them.
    """
    produced = latent.correspondence_grid(
        tmp_path / "bolsas.pdf", config.BAG_PANELS, TRANSFERS[:1], seed=3,
        device=torch.device("cpu"))

    scored = produced["scored"]
    assert scored, "the grid measured nothing to print"

    # what the count has to be, recomputed from the pairing itself: the source
    # bags are one per class in order, so a highlighted subject of class k is
    # paired correctly exactly when its nearest source bag is bag k
    reading = latent.bag_pairs(trained["model"], trained["source"],
                               trained["target"], torch.device("cpu"))
    highlighted = latent.median_bag_per_class(reading)
    expected = sum(int(reading["sourceLabels"][int(reading["nearest"][position])])
                   == class_id for class_id, position in highlighted.items())
    for row in scored:
        assert row["classes"] == len(highlighted)
        assert row["hits"] == expected

    printed = tables.render_correspondence(scored, markdown=True)
    for row in scored:
        assert f"{row['hits']}/{row['classes']}" in printed, printed
        assert config.NAME_OF[row["arm"]] in printed
    # and it is the measurement, not a constant: a different count prints
    # differently
    moved = [dict(row, hits=(row["hits"] + 1) % (row["classes"] + 1)) for row in scored]
    assert tables.render_correspondence(moved, markdown=True) != printed


def test_the_bag_figure_keeps_the_projection_and_never_becomes_a_bipartite_diagram(
        trained, tmp_path) -> None:
    """The panel is a projection of one shared space, not two columns of nodes
    with edges between them.

    A bipartite diagram fixes the two domains on two axes and draws the pairing
    as edges. It reads the pairing off cleanly and loses the only thing the
    figure is for: whether the bags landed near their own class at all. Three
    columns in one figure already makes the panels large enough to read, so the
    projection stays; if it is still a knot, the bipartite is next.

    Asserted where it can fail: both domains go through ONE embedding call, and
    what each panel scatters is exactly the coordinates that call returned --
    never a layout the drawing invented.

    Reachable red: project the two domains separately, or place them at two
    fixed abscissae and keep only the edges.
    """
    produced = latent.correspondence_grid(
        tmp_path / "bolsas.pdf", config.BAG_PANELS, TRANSFERS[:1], seed=3,
        device=torch.device("cpu"))

    embedded = trained["seen"]["embedded"]
    assert len(embedded) == len(config.BAG_PANELS), "one projection per panel"

    axes = produced["figure"].axes
    for axis, (count, projection) in zip(axes, embedded):
        # one call carrying both domains: a bipartite layout has no such shared
        # space to project into
        assert count == config.CLASSES * 2
        assert projection.shape[1] == 2
        cut = count // 2
        drawn = numpy.concatenate([axis.collections[0].get_offsets(),
                                   axis.collections[1].get_offsets()])
        assert numpy.allclose(drawn, projection)
        # both clouds live over the same abscissa range -- two fixed columns
        # would leave the source at one x and the target at another
        source_x = projection[:cut, 0]
        target_x = projection[cut:, 0]
        assert min(source_x.max(), target_x.max()) > max(source_x.min(), target_x.min())


# ------------------------------------- the three lines a figure carries above it

LATENT = config.REPOSITORY / "MIL-CREDA" / "Notebooks" / "Benchmark_Latent_v1.ipynb"


def test_every_figure_carries_the_same_three_lines_a_table_does() -> None:
    """What is looked at, what is being sought, and a conclusion below it.

    The other half of this claim -- that the conclusion is *computed* and not
    typed -- is `test_every_conclusion_the_report_produces_is_read_off_its_own
    _numbers`, which permutes the record and demands every sentence move. That
    one never looks at a notebook, so nothing until now said the three lines
    are there at all: a figure could be displayed bare, with the sentence it
    needs living in a function the notebook never calls, and both suites stay
    green.

    Read from the phase-two notebook, which is where this agreement sits and
    where the figures it names are drawn. `latent_grid`'s own docstring is the
    reason it is a notebook property and not a module one: the figure carries
    no title and no footer on purpose, because "the framing sits directly above
    the figure" -- so the framing is only ever a fact about the cells around it.

    Phase one is out of scope on purpose and not by oversight: the report's
    three curve figures carry the first two lines and no conclusion at all --
    `tables` has no `conclusion_*` for a curve, and the report calls none after
    cells 74, 79 and 84. Asserting the third line over them would fail for
    being right about a gap this agreement's own section does not cover.

    Reachable red: drop a key from `objective`'s `metas` and the figure it
    frames is declared by a placeholder; move a `show(tables.objective(...))`
    below the figure it frames and nothing states what is being sought before
    the picture is already read.
    """
    cells = json.loads(LATENT.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(cell["source"]) for cell in cells]

    # A figure is what reaches `figures.inline`. The notebook also calls
    # `latent.floors_agree`, `latent.analyse` and `latent.bound`, which compute
    # and never draw, so the display is what tells a figure from a helper.
    drawn = [index for index, cell in enumerate(cells)
             if cell["cell_type"] == "code" and "figures.inline(" in sources[index]]
    produced = {name for index in drawn
                for name in re.findall(r"latent\.(\w+_grid)\(", sources[index])}
    assert {"latent_grid", "correspondence_grid"} <= produced, \
        f"the phase-two notebook draws neither of its own figures: {produced}"

    framings = []
    for position, index in enumerate(drawn):
        # 2 · what is being sought -- the figure's own declared objective, and
        #     directly above it, so it is read before the picture is.
        asked = re.findall(r'tables\.objective\(\s*"([^"]+)"\s*\)', sources[index - 1])
        assert len(asked) == 1, \
            f"the figure in cell {index} has no objective in the cell above it"
        key = asked[0]
        framings.append(key)

        stated = tables.objective(key)
        assert "sin objetivo declarado" not in stated, \
            f"the figure framed by `{key}` is declared by a placeholder"
        assert "Buscamos" in stated, f"`{key}` never says which way is better"
        assert len(stated) > 80, f"`{key}` is framed by a line too short to say it"

        # 1 · what is looked at -- the section's own prose, above the objective
        prose = cells[index - 2]
        assert prose["cell_type"] == "markdown", \
            f"the figure in cell {index} opens no section that says what it shows"
        text = "".join(prose["source"])
        assert text.lstrip().startswith("#"), "the framing is not a section of its own"
        body = text.split("\n", 1)[1] if "\n" in text else ""
        assert len(body.strip()) > 200, \
            f"the figure framed by `{key}` is headed but never described"

        # 3 · the conclusion -- below the figure, and before the next one, so it
        #     belongs to this figure and not to the one after it.
        end = drawn[position + 1] if position + 1 < len(drawn) else len(cells)
        concluded = [name for source in sources[index + 1:end]
                     for name in re.findall(r"tables\.(conclusion\w*)\(", source)]
        assert concluded, f"the figure framed by `{key}` concludes nothing"
        for name in concluded:
            assert callable(getattr(tables, name, None)), \
                f"`{name}` is a name the notebook alone knows"

    assert len(set(framings)) == len(framings), \
        f"two figures are framed by one declaration: {framings}"


def test_the_crowded_labels_are_measured_and_not_assumed(tmp_path) -> None:
    """Que dos números se pisen es una afirmación sobre los datos, y el código
    que dibuja la mide.

    Es el polo doble: dos etiquetas en el MISMO punto tienen que quedar neutras,
    y dos separadas tienen que conservar el color de su clase. Con solo la
    primera mitad, pintar todo de gris pasaría la prueba y la figura perdería la
    codificación entera; con solo la segunda, no neutralizar nunca también.

    Rojo alcanzable: devolver 0 sin medir, o neutralizar toda la lista.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from MIL_CREDA_Benchmark import latent as modulo

    figure, axis = plt.subplots(figsize=(4, 4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 10)

    encimadas = [axis.annotate("3", xy=(5, 5), xytext=(4.5, 4.5),
                               textcoords="offset points", color="red"),
                 axis.annotate("8", xy=(5, 5), xytext=(4.5, 4.5),
                               textcoords="offset points", color="blue")]
    aparte = [axis.annotate("1", xy=(0.5, 0.5), xytext=(4.5, 4.5),
                            textcoords="offset points", color="green"),
              axis.annotate("2", xy=(9.5, 9.5), xytext=(4.5, 4.5),
                            textcoords="offset points", color="purple")]

    contadas = modulo._neutralise_crowded_labels(figure, encimadas + aparte)

    assert contadas == 2, "las dos que se pisan, y solo ésas"
    for etiqueta in encimadas:
        assert etiqueta.get_color() == modulo.CROWDED_LABEL_COLOUR
    assert aparte[0].get_color() == "green"
    assert aparte[1].get_color() == "purple"
    plt.close(figure)


def test_only_the_arm_that_asserts_numbers_the_two_ends_of_its_line(
        trained, tmp_path) -> None:
    """El número de clase va en las dos puntas, y solo donde hay afirmación.

    Dos cosas que se pueden romper por separado. Numerar una sola punta deja
    ilegible el emparejamiento errado ---la línea punteada dice «mal» y no
    contra qué---; numerar los tres paneles le presta a los brazos sin término
    local el gesto que solo el término local se ganó, que es exactamente la
    regla por la que la línea tampoco se dibuja ahí.

    Rojo alcanzable: anotar una sola punta, mover el `annotate` fuera del
    `if asserts`, o dibujar el número de la bolsa destino en las dos puntas.
    """
    produced = latent.correspondence_grid(
        tmp_path / "bolsas.pdf", config.BAG_PANELS, TRANSFERS[:1], seed=3,
        device=torch.device("cpu"))

    afirman = [arm for arm in config.BAG_PANELS if config.ARMS_BY_ID[arm]["local"]]
    assert afirman, "ningún panel afirma: la figura no tiene qué numerar"

    reading = latent.bag_pairs(trained["model"], trained["source"],
                               trained["target"], torch.device("cpu"))
    highlighted = latent.median_bag_per_class(reading)

    # dos por clase destacada y por brazo que afirma, ni una más
    assert produced["labels"] == 2 * len(highlighted) * len(afirman)

    for axis, arm in zip(produced["figure"].axes, config.BAG_PANELS):
        numeros = [t for t in axis.texts if t.get_text().isdigit()]
        if config.ARMS_BY_ID[arm]["local"]:
            assert len(numeros) == 2 * len(highlighted), arm
        else:
            assert numeros == [], f"{arm} no afirma emparejamiento y numeró"

    # y el segundo número es el de la pareja, no el de la bolsa destacada: en un
    # emparejamiento errado los dos difieren, y ahí está toda la lectura
    esperados = sorted(
        [str(class_id) for class_id in highlighted]
        + [str(int(reading["sourceLabels"][int(reading["nearest"][position])]))
           for position in highlighted.values()])
    for axis, arm in zip(produced["figure"].axes, config.BAG_PANELS):
        if not config.ARMS_BY_ID[arm]["local"]:
            continue
        assert sorted(t.get_text() for t in axis.texts
                      if t.get_text().isdigit()) == esperados
