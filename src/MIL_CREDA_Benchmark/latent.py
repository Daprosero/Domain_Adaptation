"""Phase two: what the trained models did to the space they represent in.

Accuracy says whether a method wins. It does not say whether it won by aligning
the domains class by class, which is what both formulations claim to do. That
claim lives in the representation, so it is measured there.

Every model is read against its own floor — the same arm with the adaptation term
switched off — because a distance in an embedding has no absolute meaning: what
carries information is how it moved.

Two numbers are needed together and neither means anything alone. If the distance
between the two domains' class centroids falls while the distance between
different classes holds, that is conditional alignment. If both fall, the space
collapsed and the first number alone would have called it a success.

Nothing here re-runs a draw. Every model is loaded beside the manifest that
records the exact images its bags were built from, so the space being measured is
the space that produced the accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from MIL_CREDA.local_term import total_correspondence
from MIL_CREDA_Benchmark import bags, config, figures, wiring


#: What `tools/promote.py` leaves beside the checkpoints it promoted, naming
#: which seeds of each cell are that cell's own median and which were added only
#: to make a dependant arm's paired comparison possible.
PROMOTION_RECORD = "PROMOTION.json"


def _median_seeds(rate: float = 0.0, pilot: bool = False) -> dict | None:
    """Which promoted seeds are each cell's own median, or `None` if unrecorded.

    `None` and "an empty set" are different answers and must not collapse: a
    single-machine run has no promotion record at all, and there every checkpoint
    on disk is its cell's median by construction, because `keep_median()` is what
    put it there and it writes nothing else.
    """
    record = config.models_for(rate, pilot) / PROMOTION_RECORD
    if not record.exists():
        return None
    chosen = json.loads(record.read_text(encoding="utf-8")).get("chosen") or {}
    return {cell: set(entry.get("chosen") or []) for cell, entry in chosen.items()}


def available(rate: float = 0.0, pilot: bool = False) -> list[dict]:
    """Every checkpoint phase one kept at one contamination rate, with what it is.

    `rate` defaults to the clean campaign, which is where every checkpoint on
    disk lived before the noise axis existed. It is a parameter and not a global
    because this globs a directory: a caller reading the clean tree while the
    campaign it means to analyse wrote elsewhere would measure the wrong run and
    `bound()` would only catch it if the two records happened to disagree on a
    field they share.

    Each entry carries `median`: whether this seed is its own cell's median, or
    was promoted only so that some other arm could be read against this one at
    the same seed.

    **Tagged and not filtered**, deliberately. Dropping the extras here would
    make every caller see only medians, including `against_floor()`, whose whole
    purpose is the paired difference those extras exist for — and it would lose
    them silently, which is the one failure mode nothing downstream could detect.
    Each consumer decides instead: a marginal average over a cell reads only the
    medians, a paired difference reads everything.

    The distinction is not tidiness. A floor's extras were selected by the
    *dependant* arms' accuracy orderings, not by its own, so they are a biased
    sample of that floor's outcomes. Averaging a floor's row over them does not
    estimate that row better; it estimates something else.
    """
    medians = _median_seeds(rate, pilot)
    found = []
    for manifest_path in sorted(
            config.models_for(rate, pilot).glob("*.manifest.json")):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        weights = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".pt"))
        if not weights.exists():
            continue
        if medians is None:
            is_median = True
        else:
            cell = f"{record['arm']}|{record['transfer']}"
            is_median = record["seed"] in medians.get(cell, set())
        found.append({**record, "weights": weights, "manifest": manifest_path,
                      "median": is_median})
    return found


class CheckpointsDisagree(RuntimeError):
    """The checkpoints on disk were not produced by the run recorded beside them.

    Separate from "there are no checkpoints": that one is loud and stops the
    notebook at its first cell. This one is silent by construction — a directory
    full of valid checkpoints from an earlier, smaller run loads, measures and
    renders exactly like the right ones, under the record's stamp.
    """


#: The one field of a `reduction` that is expected to differ, and an exclusion
#: rather than an oversight. A checkpoint's manifest records the seeds of the
#: shard that produced it; the record's carries the union across every shard that
#: arrived. Requiring equality there would refuse every distributed run. The seed
#: itself is held to the record instead, one field over: it has to be one the
#: record says was actually run.
PER_SHARD = "seeds"


def disagreements(found: list[dict], summary: dict) -> list[dict]:
    """Where the checkpoints and the record they are read beside disagree.

    Compared field by field over what the two `reduction`s *both* carry, because
    they are not the same shape: a manifest holds a full `Reduction`, while the
    record holds only what merging actually proved. Comparing the union would
    refuse on fields the record never claimed; comparing nothing at all is what
    let a three-epoch pilot be measured under a twenty-epoch stamp.

    Returns the findings rather than raising, so the refusal can be tested apart
    from the message that carries it — `bound()` is what refuses.
    """
    record = summary.get("reduction") or {}
    ran = set(record.get(PER_SHARD) or [])
    found_out: list[dict] = []
    for entry in found:
        mine = entry.get("reduction") or {}
        name = Path(entry["manifest"]).name if entry.get("manifest") else "?"
        for field in sorted((set(mine) & set(record)) - {PER_SHARD}):
            if mine[field] != record[field]:
                found_out.append({"checkpoint": name, "field": field,
                                  "checkpoint_says": mine[field],
                                  "record_says": record[field]})
        if ran and entry.get("seed") not in ran:
            found_out.append({"checkpoint": name, "field": "seed",
                              "checkpoint_says": entry.get("seed"),
                              "record_says": sorted(ran)})
    return found_out


def bound(found: list[dict], summary: dict) -> list[dict]:
    """`found`, once it is established that the record describes it.

    A record with no `reduction`, or one sharing no field with the manifests,
    refuses too. An unprovable precondition is not a satisfied one: passing there
    would make this check silent in exactly the state where nothing is known.
    """
    record = summary.get("reduction") or {}
    if not found:
        raise CheckpointsDisagree(
            "refusing to measure: there are no checkpoints to bind to the record.")
    comparable = {f for entry in found
                  for f in (set(entry.get("reduction") or {}) & set(record))} - {PER_SHARD}
    if not comparable:
        raise CheckpointsDisagree(
            "refusing to measure: the record carries no `reduction` field the "
            "checkpoints also carry, so whether they came from this run cannot be "
            "established at all.\n"
            "  An unchecked precondition is not a satisfied one."
        )
    clashes = disagreements(found, summary)
    if clashes:
        fields = sorted({c["field"] for c in clashes})
        first = clashes[0]
        raise CheckpointsDisagree(
            f"refusing to measure: {len(clashes)} checkpoint(s) disagree with the "
            f"record on {', '.join(fields)}.\n"
            f"  e.g. {first['checkpoint']}: {first['field']} is "
            f"{first['checkpoint_says']!r} but the record says "
            f"{first['record_says']!r}.\n"
            "  These checkpoints were produced by a different run than the one "
            "`summary.json` describes. Promote the ones this record came from "
            "(`tools/promote.py`) rather than measuring these under its stamp."
        )
    return found


def load(record: dict, device: torch.device):
    """One trained arm and the exact material it was trained on."""
    source = bags.rebuild(record["source"], config.DATA_CACHE)
    target = bags.rebuild(record["target"], config.DATA_CACHE)
    model = wiring.build(
        record["arm"], config.CLASSES,
        wiring.Pool(source.images.to(device),
                    source.members[source.train_idx].to(device),
                    source.labels[source.train_idx].to(device)),
        wiring.Pool(target.images.to(device),
                    target.members[target.train_idx].to(device),
                    target.labels[target.train_idx].to(device)),
    ).to(device)
    model.load_state_dict(torch.load(record["weights"], map_location=device))
    model.eval()
    return model, source, target


@torch.no_grad()
def represent(model, bagset: bags.BagSet, positions: torch.Tensor,
              device: torch.device, unit: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows this arm actually aligns, and the class of each.

    A bag-unit arm aligns bag representations, one row per subject. An
    instance-unit arm aligns instances, one row each, carrying the class of the
    bag they came from. Each is **measured** in its own unit and the two are never
    put in the same number.

    `unit` overrides that for drawing only. Every arm encodes instances, so
    forcing the instance unit gives a space every arm has and every panel of a
    grid the same number of points; forcing it for a measurement would put two
    families in one number, which is what the default exists to prevent.
    """
    rows, labels = [], []
    for start in range(0, len(positions), config.BAGS_PER_STEP):
        chunk = positions[start:start + config.BAGS_PER_STEP]
        instances = bagset.images[bagset.members[chunk]].to(device)
        embeddings = model.instance_embeddings(instances)
        classes = bagset.labels[chunk].to(device)
        if (unit or model.spec["unit"]) == "bag":
            Z, _ = model.bag_representations(embeddings)
            rows.append(Z)
            labels.append(classes)
        else:
            rows.append(embeddings.reshape(-1, embeddings.shape[-1]))
            labels.append(classes.repeat_interleave(embeddings.shape[1]))
    return torch.cat(rows).float().cpu(), torch.cat(labels).cpu()


def geometry(source_rows, source_labels, target_rows, target_labels) -> dict:
    """Conditional alignment, or collapse, told apart.

    Raw distances are not comparable between two models: the embedding has no
    fixed scale and each run settles on its own. The ratio is, which is why the
    verdict is read from it and the two distances are reported beside it rather
    than instead of it.
    """
    def centroids(rows, labels):
        return {int(c): rows[labels == c].mean(dim=0)
                for c in labels.unique() if int((labels == c).sum()) > 0}

    mu_s, mu_t = centroids(source_rows, source_labels), centroids(target_rows, target_labels)
    shared = sorted(set(mu_s) & set(mu_t))
    cross = [float(torch.norm(mu_s[c] - mu_t[c])) for c in shared]

    def between(mu):
        keys = sorted(mu)
        return [float(torch.norm(mu[a] - mu[b]))
                for i, a in enumerate(keys) for b in keys[i + 1:]]

    within_source, within_target = between(mu_s), between(mu_t)
    mean_cross = sum(cross) / len(cross) if cross else float("nan")
    apart = within_source + within_target
    mean_apart = sum(apart) / len(apart) if apart else float("nan")
    return {
        "crossDomainSameClass": mean_cross,
        "betweenClasses": mean_apart,
        # Below one means the two domains sit closer to each other, class by
        # class, than different classes sit to one another. That is the shape
        # conditional alignment has; a collapse moves both and leaves it flat.
        "ratio": mean_cross / mean_apart if mean_apart else float("nan"),
        "classes": len(shared),
    }


def separability(source_rows, target_rows, seed: int) -> float:
    """How well a linear rule still tells the two domains apart.

    Chance is one half. Nearer that is a more aligned representation — but on its
    own it cannot distinguish alignment from collapse either, which is why it is
    read together with the ratio above and never by itself.
    """
    X = torch.cat([source_rows, target_rows]).numpy()
    y = torch.cat([torch.zeros(len(source_rows)), torch.ones(len(target_rows))]).numpy()
    model = LogisticRegression(max_iter=2000, random_state=seed)
    return float(cross_val_score(model, X, y, cv=5, scoring="accuracy").mean())


@torch.no_grad()
def correspondence(model, source: bags.BagSet, target: bags.BagSet,
                   device: torch.device) -> dict:
    """How much of Eq. (29)'s mass lands on source subjects of the right class.

    This is the local term's own claim, stated as a number rather than a picture:
    each target subject spreads its correspondence over the source subjects, and
    if the correspondence learned anything, most of that mass sits on subjects of
    the target's true class. Chance is one over the class count.

    The true target labels appear here and nowhere in training. They are read at
    analysis time only, to score a quantity the method produced without them.
    """
    from MIL_CREDA.bag_kernel import bag_kernel_matrix

    source_positions = source.train_idx
    target_positions = target.eval_idx
    H_s = model.instance_embeddings(source.images[source.members[source_positions]].to(device))
    H_t = model.instance_embeddings(target.images[target.members[target_positions]].to(device))

    # Through the arm's own selection, so a selecting arm is scored on the
    # instances it actually looks at. Reading the full bags here would measure a
    # correspondence the trained model never computed.
    bags_s = model.bags_of(H_s)
    bags_t = model.bags_of(H_t)
    sigma = wiring._median_sigma(torch.cat([torch.cat([H for H, _ in bags_s]),
                                            torch.cat([H for H, _ in bags_t])]))
    K_st = bag_kernel_matrix(bags_s, bags_t, sigma)

    from MIL_CREDA.attention import bag_embedding
    Z_t = torch.stack([bag_embedding(H, w) for H, w in bags_t])
    G_t = F.softmax(model.head(Z_t), dim=1)

    source_labels = source.labels[source_positions].to(device)
    truth = target.labels[target_positions].to(device)

    on_truth = []
    for column in range(len(target_positions)):
        pi = total_correspondence(K_st[:, column], source_labels, G_t[column],
                                  config.TAU_LOCAL)
        on_truth.append(float(pi[source_labels == truth[column]].sum()))
    return {
        "massOnTrueClass": sum(on_truth) / len(on_truth),
        "chance": 1.0 / config.CLASSES,
        "subjects": len(on_truth),
    }


@torch.no_grad()
def attention_spread(model, bagset: bags.BagSet, positions: torch.Tensor,
                     device: torch.device) -> float:
    """The entropy of the in-bag weights, normalized to [0, 1].

    One means the arm is weighting every instance alike, which is the uniform
    mean it was supposed to improve on; near zero means it is resting the whole
    subject on a few instances.
    """
    entropies = []
    for start in range(0, len(positions), config.BAGS_PER_STEP):
        chunk = positions[start:start + config.BAGS_PER_STEP]
        instances = bagset.images[bagset.members[chunk]].to(device)
        for _, beta in model.bags_of(model.instance_embeddings(instances)):
            entropy = -(beta * torch.log(beta + config.EPSILON)).sum()
            entropies.append(float(entropy / torch.log(torch.tensor(float(len(beta))))))
    return sum(entropies) / len(entropies)


def analyse(record: dict, device: torch.device) -> dict:
    """One checkpoint, measured."""
    model, source, target = load(record, device)
    source_rows, source_labels = represent(model, source, source.eval_idx, device)
    target_rows, target_labels = represent(model, target, target.eval_idx, device)

    reading = {
        "arm": record["arm"],
        "transfer": record["transfer"],
        "seed": record["seed"],
        "unit": model.spec["unit"],
        "targetAccuracy": record["targetAccuracy"],
        "geometry": geometry(source_rows, source_labels, target_rows, target_labels),
        "domainSeparability": separability(source_rows, target_rows, record["seed"]),
    }
    if model.spec["attention"] is not None:
        reading["attentionSpread"] = attention_spread(model, target, target.eval_idx, device)
    if model.spec["local"]:
        reading["correspondence"] = correspondence(model, source, target, device)
    return reading


def against_floor(readings: list[dict]) -> list[dict]:
    """Each adapted arm beside the floor it should have improved on."""
    indexed = {(r["arm"], r["transfer"], r["seed"]): r for r in readings}
    compared = []
    for (arm, transfer, seed), reading in indexed.items():
        floor_id = config.FLOOR_OF.get(arm)
        floor = indexed.get((floor_id, transfer, seed)) if floor_id else None
        if floor is None:
            continue
        compared.append({
            "arm": arm, "floor": floor_id, "transfer": transfer, "seed": seed,
            "ratio": reading["geometry"]["ratio"],
            "floorRatio": floor["geometry"]["ratio"],
            "ratioChange": reading["geometry"]["ratio"] - floor["geometry"]["ratio"],
            "separability": reading["domainSeparability"],
            "floorSeparability": floor["domainSeparability"],
            "separabilityChange": reading["domainSeparability"] - floor["domainSeparability"],
        })
    return compared


def projection(rows, labels, domains, path: Path, title: str, seed: int,
               caption: str = "") -> "figures.plt.Figure":
    """A picture beside the numbers, never instead of them.

    UMAP and not t-SNE, and the reason is the claim being shown rather than taste.
    t-SNE optimizes local neighbourhoods and does not preserve the distance between
    clusters — the standard caveat is that inter-cluster distance in a t-SNE plot
    means nothing. What the geometry measurements assert is exactly an inter-cluster
    distance: how far the same class sits across domains against how far different
    classes sit from each other. Drawing that on a projection that scrambles it would
    show the reader something the numbers do not say.

    The caption carries the run's bounds, because a figure read without them gets
    misquoted the same way a number does.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(n_components=2, random_state=seed,
                        n_neighbors=min(15, max(2, len(rows) // 4)), min_dist=0.1)
    embedded = reducer.fit_transform(rows.numpy())
    figure, axis = plt.subplots(figsize=(6, 5.4))
    for domain, marker in ((0, "o"), (1, "^")):
        mask = domains == domain
        axis.scatter(embedded[mask, 0], embedded[mask, 1], c=labels[mask],
                     cmap="tab10", marker=marker, s=14, alpha=0.75,
                     label="source" if domain == 0 else "target")
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    if caption:
        # Wrapped rather than shrunk: a caption that runs off the edge loses exactly
        # the bounds it was added to carry, and silently.
        import textwrap
        wrapped = "\n".join(textwrap.wrap(caption, width=72))
        figure.text(0.5, 0.012, wrapped, ha="center", va="bottom",
                    fontsize=7, color="0.35", linespacing=1.5)
    figure.tight_layout(rect=(0, 0.10, 1, 1) if caption else None)
    return figures.emit(figure, path)


# ------------------------------------------------------------- the comparative grid

def display_seed(runs: list[dict]) -> int:
    """The one seed every comparative panel is drawn from.

    Not each arm's own median run, and the reason is the rule the whole ladder
    rests on: a comparison may differ in one thing. Panels taken from different
    seeds would differ in the method *and* in the draw — and because the seed
    fixes the partition, they would not even share their bags, so "this subject
    sits beside the wrong one here and the right one there" would stop being a
    sentence anyone could say.

    The seed is chosen by a rule that favours nobody: the one whose mean target
    accuracy *across every arm* is the median. Choosing it by the headline arm's
    own outcome would let that arm pick the ground it is judged on.
    """
    by_seed: dict[int, list[float]] = {}
    for run in runs:
        by_seed.setdefault(int(run["seed"]), []).append(float(run["targetAccuracy"]))
    if not by_seed:
        raise ValueError("no runs: there is no seed to display")
    ordered = sorted(by_seed, key=lambda s: sum(by_seed[s]) / len(by_seed[s]))
    return ordered[len(ordered) // 2]


def checkpoint_for(arm: str, transfer: str, seed: int,
                   rate: float = 0.0, pilot: bool = False) -> dict | None:
    """The kept checkpoint of one cell, or nothing if that cell kept none.

    `rate`/`pilot` eligen de qué corrida. Sin ellos una figura contaminada se
    dibujaba con los pesos limpios: correcta en forma, de la corrida
    equivocada, y sin un solo error que lo delatara.
    """
    for record in available(rate, pilot):
        if (record["arm"] == arm and record["transfer"] == transfer
                and int(record["seed"]) == seed):
            return record
    return None


def original_rows(record: dict, budget: int, seed: int):
    """The shared original space of the pair: the images themselves, before any model.

    The reference every trained column is read against, and model-free on purpose —
    it has been fitted to this transfer by nothing at all. It is a *shared* space
    because preprocessing already brings both domains to the same tensor shape, so
    the two sets of pixels live in one vector space and can be projected together
    without anything having learned to put them there.

    Representative rather than complete: an evaluation split holds far more
    instances than a panel can show, so the sample is stratified by class and drawn
    from a generator of its own.
    """
    source = bags.rebuild(record["source"], config.DATA_CACHE)
    target = bags.rebuild(record["target"], config.DATA_CACHE)

    def pixels(bagset):
        images = bagset.images[bagset.members[bagset.eval_idx]]
        rows = images.reshape(images.shape[0] * images.shape[1], -1).float()
        labels = bagset.labels[bagset.eval_idx].repeat_interleave(images.shape[1])
        return equalize(rows, labels, budget, seed)

    s_rows, s_labels = pixels(source)
    t_rows, t_labels = pixels(target)
    return s_rows, s_labels, t_rows, t_labels


def _embed(rows: torch.Tensor, seed: int):
    """UMAP, for the reason `projection` already gives: the claim is a distance."""
    import umap
    reducer = umap.UMAP(n_components=2, random_state=seed,
                        n_neighbors=min(15, max(2, len(rows) // 4)), min_dist=0.1)
    return reducer.fit_transform(rows.numpy())


def equalize(rows, labels, budget: int, seed: int):
    """Draw the same number of points for every arm, whatever its unit.

    Without this the grid is unreadable in a way that looks like a finding: an
    instance-unit arm contributes one row per instance and a bag-unit arm one row
    per subject, so at thirty instances a bag the CREDA columns arrive with thirty
    times the points. The eye reads that as *covers the space* and the sparse
    columns as *learned less*, and neither is what happened — it is the statistical
    unit, drawn.

    The subsample is stratified by class and drawn from a generator of its own, so
    it neither drops a class from a panel nor disturbs any other draw.
    """
    if len(rows) <= budget:
        return rows, labels
    generator = torch.Generator().manual_seed(seed)
    per_class = max(1, budget // max(1, int(labels.unique().numel())))
    keep = []
    for class_id in labels.unique():
        positions = (labels == class_id).nonzero().reshape(-1)
        take = min(per_class, positions.numel())
        keep.append(positions[torch.randperm(positions.numel(), generator=generator)[:take]])
    chosen = torch.cat(keep)
    return rows[chosen], labels[chosen]


def _draw_cell(axis, embedded, labels, domains):
    """Class by colour, domain by marker: source circles, target triangles.

    The target markers are larger and edged and the source ones smaller and
    semi-transparent, because at grid size a shape is harder to read than a
    colour and the domain is the thing the reader is looking for.
    """
    for domain, marker, size, alpha, edge in ((0, "o", 12, 0.55, "none"),
                                              (1, "^", 26, 0.95, "0.15")):
        mask = domains == domain
        if not mask.any():
            continue
        axis.scatter(embedded[mask, 0], embedded[mask, 1], c=labels[mask],
                     cmap="tab10", vmin=0, vmax=9, marker=marker, s=size,
                     alpha=alpha, linewidths=0.35, edgecolors=edge)
    axis.set_xticks([])
    axis.set_yticks([])


def latent_grid(path: Path, arms: list[str], transfers: list[str], seed: int,
                device: torch.device) -> "figures.plt.Figure":
    """Rows are transfers, columns are arms, and every panel is the same space.

    No title and no footer. What the marks mean and what bounds the run was made
    under are the framing's job, and the framing sits directly above the figure:
    stating them here as well puts the same claim in two places, where they can
    drift apart and the reader cannot tell which one moved.

    Every panel of the grid is drawn at the instance level, whatever unit its arm
    trains on. Every arm encodes instances — Eq. (13) applies identically in both
    families — so it is a space they all have, and it is the only way the panels
    carry the same number of points. One point per subject beside one point per
    instance made the instance-unit columns look like they covered the space and
    the bag-unit ones look sparse, which is the statistical unit drawn rather than
    anything about alignment.

    Every panel comes from the same seed, so every panel of a row is looking at
    the same subjects.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["Original"] + [config.NAME_OF[a] for a in arms]
    figure, axes = plt.subplots(len(transfers), len(columns),
                                figsize=(2.15 * len(columns), 2.3 * len(transfers)),
                                squeeze=False)

    for row, transfer in enumerate(transfers):
        anchor = next((checkpoint_for(a, transfer, seed) for a in arms
                       if checkpoint_for(a, transfer, seed)), None)
        for column in range(len(columns)):
            axis = axes[row][column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                # The shared original space: the images themselves, before any
                # model. Any checkpoint of this row names the same material, which
                # is why the anchor can be whichever one exists.
                if anchor is None:
                    axis.axis("off")
                    continue
                s_rows, s_labels, t_rows, t_labels = original_rows(
                    anchor, config.LATENT_POINTS, seed)
                _draw_cell(axis, _embed(torch.cat([s_rows, t_rows]), seed),
                           torch.cat([s_labels, t_labels]).numpy(),
                           torch.cat([torch.zeros(len(s_rows)),
                                      torch.ones(len(t_rows))]).numpy())
                if row == 0:
                    axis.set_title(columns[0], fontsize=9)
                continue
            arm = arms[column - 1]
            record = checkpoint_for(arm, transfer, seed)
            if record is None:
                axis.axis("off")
                continue
            model, source, target = load(record, device)
            s_rows, s_labels = represent(model, source, source.eval_idx, device,
                                         unit=config.LATENT_UNIT)
            t_rows, t_labels = represent(model, target, target.eval_idx, device,
                                         unit=config.LATENT_UNIT)
            del model
            # Every panel the same size, so no column looks denser than another.
            s_rows, s_labels = equalize(s_rows, s_labels, config.LATENT_POINTS, seed)
            t_rows, t_labels = equalize(t_rows, t_labels, config.LATENT_POINTS, seed)
            stacked = torch.cat([s_rows, t_rows])
            domains = torch.cat([torch.zeros(len(s_rows)), torch.ones(len(t_rows))]).numpy()
            _draw_cell(axis, _embed(stacked, seed),
                       torch.cat([s_labels, t_labels]).numpy(), domains)
            if row == 0:
                axis.set_title(columns[column], fontsize=9)
        axes[row][0].set_ylabel(transfer, fontsize=9)

    figure.tight_layout()
    return figures.emit(figure, path)


# ------------------------------------------------------- the correspondence figure

@torch.no_grad()
def bag_pairs(model, source: bags.BagSet, target: bags.BagSet, device: torch.device) -> dict:
    """Bag representations, and for each target bag the source bag nearest to it.

    Nearest by the bag kernel of Section 3, in the representation space — never
    Euclidean, which the method does not use, and never in the two-dimensional
    projection, which would illustrate UMAP rather than the correspondence.
    """
    from MIL_CREDA.attention import bag_embedding
    from MIL_CREDA.bag_kernel import bag_kernel_matrix

    s_pos, t_pos = source.train_idx, target.eval_idx
    H_s = model.instance_embeddings(source.images[source.members[s_pos]].to(device))
    H_t = model.instance_embeddings(target.images[target.members[t_pos]].to(device))
    pairs_s, pairs_t = model.bags_of(H_s), model.bags_of(H_t)
    sigma = wiring._median_sigma(torch.cat([torch.cat([H for H, _ in pairs_s]),
                                            torch.cat([H for H, _ in pairs_t])]))
    K_st = bag_kernel_matrix(pairs_s, pairs_t, sigma)

    Z_s = torch.stack([bag_embedding(H, w) for H, w in pairs_s])
    Z_t = torch.stack([bag_embedding(H, w) for H, w in pairs_t])
    G_t = F.softmax(model.head(Z_t), dim=1)
    s_labels = source.labels[s_pos].to(device)
    t_labels = target.labels[t_pos].to(device)

    mass = []
    for column in range(len(t_pos)):
        pi = total_correspondence(K_st[:, column], s_labels, G_t[column], config.TAU_LOCAL)
        mass.append(float(pi[s_labels == t_labels[column]].sum()))

    return {
        "sourceRows": Z_s.float().cpu(),
        "targetRows": Z_t.float().cpu(),
        "sourceLabels": s_labels.cpu(),
        "targetLabels": t_labels.cpu(),
        "nearest": K_st.argmax(dim=0).cpu(),
        "mass": torch.tensor(mass),
    }


def median_bag_per_class(reference: dict) -> dict:
    """One target bag of each class: the median by correspondence mass, never the best.

    The best bag of a class pairs cleanly under every arm, including the floor
    that learned no correspondence at all — a figure that cannot come out wrong
    is not measuring anything. The median is what the arm typically does with
    that class, which is the thing being compared.
    """
    chosen = {}
    for class_id in range(config.CLASSES):
        positions = (reference["targetLabels"] == class_id).nonzero().reshape(-1)
        if not positions.numel():
            continue
        ordered = positions[reference["mass"][positions].argsort()]
        chosen[class_id] = int(ordered[len(ordered) // 2])
    return chosen




#: El color al que caen los números que se pisan. No es el gris de `tab10`
#: ---que es la clase 7 y sería un número mintiendo sobre su clase--- sino uno
#: bastante más oscuro, para que se lea como «acá hay varios» y no como una
#: clase más.
CROWDED_LABEL_COLOUR = "0.25"


def _neutralise_crowded_labels(figure, labels, colour: str = CROWDED_LABEL_COLOUR) -> int:
    """Los números que se pisan pasan a un color neutro, medido y no supuesto.

    Que dos etiquetas se traslapen es una afirmación sobre los datos ---sobre
    dónde cayeron los puntos en esta proyección, con esta semilla--- y no una
    decisión de diseño. Dejarla como creencia la vuelve falsa en cuanto cambia
    la corrida: se pintarían de neutro números que no se pisan, o se dejarían
    de colores números apilados. Así que se mide: se dibuja, se piden las cajas
    reales en coordenadas de pantalla y se neutraliza exactamente lo que se
    superpone.

    Va después de `tight_layout` y no antes. El acomodo mueve los ejes, y una
    caja pedida antes describe una figura que ya no existe --- la medición
    saldría igual de convincente y sería sobre otra imagen.
    """
    if len(labels) < 2:
        return 0
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in labels]
    crowded = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                crowded.add(i)
                crowded.add(j)
    for i in crowded:
        labels[i].set_color(colour)
    return len(crowded)


def correspondence_grid(path: Path, arms: list[str], transfers: list[str], seed: int,
                        device: torch.device, rate: float = 0.0,
                        pilot: bool = False) -> dict:
    """Rows are transfers, columns are arms: one figure, not one file per transfer.

    The panels are chosen by the mechanism rather than by the ranking: the floor,
    the same method without the local term, and the complete one. That is the rung
    the local correspondence lives on, so the figure can come out wrong — and if
    the middle column pairs its subjects as well as the right one does, the local
    term is doing nothing visible.

    The same subject of each class is highlighted in every panel of a row, and it
    is the **median** of its class by correspondence mass, never the best: the best
    subject of a class pairs cleanly under every arm, including the floor that
    learned no correspondence at all, and a figure that cannot come out wrong is
    not measuring anything.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(transfers), len(arms),
                                figsize=(3.5 * len(arms), 3.5 * len(transfers)),
                                squeeze=False)
    palette = plt.get_cmap("tab10")
    scored: list[dict] = []
    etiquetas = []  # los números de clase, para medirles el traslape al final

    for row, transfer in enumerate(transfers):
        readings, present = {}, []
        for arm in arms:
            record = checkpoint_for(arm, transfer, seed, rate, pilot)
            if record is None:
                continue
            model, source, target = load(record, device)
            readings[arm] = bag_pairs(model, source, target, device)
            present.append(arm)
            del model
        if not present:
            for column in range(len(arms)):
                axes[row][column].axis("off")
            continue

        # The subjects come from the floor's reading. Picking them from the arm
        # being praised would let it choose the ground it is judged on.
        highlighted = median_bag_per_class(readings[present[0]])

        for column, arm in enumerate(arms):
            axis = axes[row][column]
            axis.set_xticks([])
            axis.set_yticks([])
            if arm not in readings:
                axis.axis("off")
                continue
            reading = readings[arm]
            cut = len(reading["sourceRows"])
            embedded = _embed(torch.cat([reading["sourceRows"], reading["targetRows"]]), seed)
            s_xy, t_xy = embedded[:cut], embedded[cut:]

            # Cada bolsa con el color de su clase, en los tres paneles. Sin las
            # líneas, el color es lo único que deja juzgar la correspondencia: si
            # el triángulo destacado cayó entre círculos de su propio color, quedó
            # cerca de su clase, y eso se ve sin que el dibujo afirme un
            # emparejamiento que el método no computa. En gris no se veía nada.
            axis.scatter(s_xy[:, 0], s_xy[:, 1],
                         c=[palette(int(k) % 10) for k in reading["sourceLabels"]],
                         marker="o", s=18, linewidths=0, alpha=0.55, zorder=1)
            axis.scatter(t_xy[:, 0], t_xy[:, 1],
                         c=[palette(int(k) % 10) for k in reading["targetLabels"]],
                         marker="^", s=24, linewidths=0, alpha=0.55, zorder=1)

            # Only one column carries a claim, and it is the arm that declares the
            # local term — read from the arm, never from its position, because a
            # reordered panel list would silently move the emphasis onto a method
            # that does not have the term.
            #
            # Drawn identically, the three columns read as if all three paired
            # subjects on purpose. In the two without the local term the line is
            # just the nearest source neighbour: something to look at for whether
            # the subjects associate graphically, and nothing more. In the arm that
            # has the term, that same line IS the term's assertion. So the weight
            # of the ink follows the claim, and correctness coding — solid against
            # dotted — appears only where correctness is being asserted. Every
            # panel's hits still land in the table behind the figure, so nothing
            # the muted columns stop showing is lost.
            asserts = bool(config.ARMS_BY_ID[arm]["local"])

            hits = 0
            for class_id, position in highlighted.items():
                colour = palette(class_id % 10)
                partner = int(reading["nearest"][position])
                correct = int(reading["sourceLabels"][partner]) == class_id
                hits += correct
                # El sujeto destacado se dibuja en los tres paneles: es la misma
                # bolsa a lo largo de toda la fila, elegida por el piso y no por el
                # brazo que se está juzgando, así que seguirla de panel en panel es
                # justamente la comparación que la figura ofrece.
                axis.scatter(*t_xy[position], color=colour, marker="^", s=110,
                             edgecolors="0.15", linewidths=0.7, zorder=3)

                # La pareja y la línea, en cambio, SOLO donde hay correspondencia
                # por bolsas que las sostenga. En un brazo sin término local no
                # existe tal emparejamiento: lo que se trazaría es la vecina más
                # cercana, que es una consecuencia de dónde quedaron los puntos y
                # no algo que el método afirme. Dibujarla le prestaría a esos
                # paneles el gesto que solo el término local se ganó. Lo que se ve
                # ahí es el color: si el triángulo destacado cayó entre círculos de
                # su misma clase, quedó cerca de su clase, y eso el lector lo juzga
                # sin que nadie le trace una conclusión encima.
                if not asserts:
                    continue
                axis.plot([t_xy[position, 0], s_xy[partner, 0]],
                          [t_xy[position, 1], s_xy[partner, 1]],
                          color=colour, linewidth=1.6,
                          linestyle="-" if correct else ":",
                          alpha=0.95, zorder=2)
                axis.scatter(*s_xy[partner], color=colour, marker="o", s=80,
                             edgecolors="0.15", linewidths=0.7, zorder=3)

                # El número de clase en CADA punta de la línea, del color de su
                # propio punto. El color ya marca la clase, pero son diez sobre
                # `tab10` en un panel de 3.5 pulgadas: dos azules distintos son
                # dos clases distintas y el ojo no las separa, así que el color
                # solo alcanza para decir «parecidas» y no «cuál».
                #
                # Y en las dos puntas, no en una. Cuando el emparejamiento es
                # correcto los dos números coinciden y el segundo no agrega
                # nada; cuando es errado ---la línea punteada--- es lo único que
                # dice CONTRA QUÉ clase se emparejó, que es el hallazgo entero.
                # Un solo número deja al lector con «esta se emparejó mal» y sin
                # con qué, que es la mitad de la lectura.
                #
                # Solo en la columna que afirma, como la línea misma: los otros
                # dos paneles no trazan emparejamiento, así que numerar sus
                # puntas les prestaría el gesto que solo el término local se ganó.
                for punto, klass in ((t_xy[position], class_id),
                                     (s_xy[partner],
                                      int(reading["sourceLabels"][partner]))):
                    etiquetas.append(axis.annotate(
                        str(int(klass)),
                        xy=(float(punto[0]), float(punto[1])),
                        xytext=(4.5, 4.5), textcoords="offset points",
                        fontsize=7, fontweight="bold",
                        color=palette(int(klass) % 10), zorder=4))

            share = float(reading["mass"].mean())
            scored.append({"arm": arm, "transfer": transfer, "hits": hits,
                           "classes": len(highlighted), "mass": share})
            if row == 0:
                axis.set_title(config.NAME_OF[arm], fontsize=10)
        axes[row][0].set_ylabel(transfer, fontsize=10)
    # No reserved strips left: the top one held a suptitle and the bottom one a
    # caption, and both moved into the framing above the figure.
    figure.tight_layout()
    # Medido acá y no adentro del bucle: `tight_layout` mueve los ejes, así que
    # una caja pedida antes describe una figura que ya no existe. Y una sola
    # pasada sobre toda la figura en lugar de una por panel --- el traslape se
    # mide en coordenadas de pantalla, donde dos números de paneles vecinos que
    # se pisan se pisan igual.
    crowded = _neutralise_crowded_labels(figure, etiquetas)
    drawn = figures.emit(figure, path)
    return {"path": path.with_suffix(".pdf"), "figure": drawn, "scored": scored,
            # Informado y no callado: cuántos números quedaron neutros es cuánto
            # se apretó la proyección, y un lector que ve grises tiene que poder
            # saber que son eso y no una clase.
            "labels": len(etiquetas), "crowdedLabels": crowded}


@torch.no_grad()
def floors_agree(transfers: list[str], seed: int, device: torch.device,
                 left: str = "A", right: str = "B") -> dict:
    """¿Los dos pisos representan lo mismo a nivel de instancia? Medido, no supuesto.

    La pregunta importa porque de la respuesta depende si una columna de la grilla
    es redundante. Y no se puede contestar mirando el código: los dos entrenan el
    mismo codificador con objetivos distintos — entropía cruzada por instancia
    contra entropía cruzada por bolsa a través del agrupamiento por atención — así
    que sus embeddings de instancia no tienen por qué coincidir.

    Se compara la **geometría**, no las coordenadas: dos codificadores pueden
    aprender el mismo espacio rotado, y comparar coordenadas llamaría distintos a
    dos espacios idénticos. Lo que se compara es la razón entre distancias y la
    separabilidad de dominio, que son las cantidades que la figura muestra.
    """
    readings = []
    for transfer in transfers:
        pair = {}
        for arm in (left, right):
            record = checkpoint_for(arm, transfer, seed)
            if record is None:
                continue
            model, source, target = load(record, device)
            s_rows, s_labels = represent(model, source, source.eval_idx, device,
                                         unit="instance")
            t_rows, t_labels = represent(model, target, target.eval_idx, device,
                                         unit="instance")
            del model
            pair[arm] = {
                "ratio": geometry(s_rows, s_labels, t_rows, t_labels)["ratio"],
                "separability": separability(s_rows, t_rows, seed),
            }
        if len(pair) == 2:
            readings.append({
                "transfer": transfer,
                "ratio": {arm: pair[arm]["ratio"] for arm in pair},
                "separability": {arm: pair[arm]["separability"] for arm in pair},
                "ratioGap": abs(pair[left]["ratio"] - pair[right]["ratio"]),
                "separabilityGap": abs(pair[left]["separability"]
                                       - pair[right]["separability"]),
            })
    if not readings:
        return {"agree": None, "detail": "sin checkpoints de los dos pisos"}

    worst_ratio = max(r["ratioGap"] for r in readings)
    worst_separability = max(r["separabilityGap"] for r in readings)
    agree = (worst_ratio <= config.FLOORS_AGREE_WITHIN
             and worst_separability <= config.FLOORS_AGREE_WITHIN)
    return {
        "agree": agree,
        "left": config.NAME_OF[left],
        "right": config.NAME_OF[right],
        "tolerance": config.FLOORS_AGREE_WITHIN,
        "worstRatioGap": worst_ratio,
        "worstSeparabilityGap": worst_separability,
        "byTransfer": readings,
        "detail": (
            f"{config.NAME_OF[left]} y {config.NAME_OF[right]} representan lo mismo "
            f"a nivel de instancia dentro de {config.FLOORS_AGREE_WITHIN:.2f}: una "
            f"de las dos columnas sería redundante."
            if agree else
            f"{config.NAME_OF[left]} y {config.NAME_OF[right]} NO representan lo "
            f"mismo a nivel de instancia: la razón entre distancias difiere hasta "
            f"{worst_ratio:.3f} y la separabilidad hasta {worst_separability:.3f}, "
            f"contra una tolerancia de {config.FLOORS_AGREE_WITHIN:.2f}. Sacar una "
            f"de las dos columnas pierde el piso de esa familia."),
    }
