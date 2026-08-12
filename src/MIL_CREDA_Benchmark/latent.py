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
from MIL_CREDA_Benchmark import bags, config, wiring


def available() -> list[dict]:
    """Every checkpoint phase one kept, with what it is."""
    found = []
    for manifest_path in sorted(config.MODELS.glob("*.manifest.json")):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        weights = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".pt"))
        if weights.exists():
            found.append({**record, "weights": weights, "manifest": manifest_path})
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
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows this arm actually aligns, and the class of each.

    A bag-unit arm aligns bag representations, one row per subject. An
    instance-unit arm aligns instances, one row each, carrying the class of the
    bag they came from. Each is measured in its own unit and the two are never
    put in the same number.
    """
    rows, labels = [], []
    for start in range(0, len(positions), config.BAGS_PER_STEP):
        chunk = positions[start:start + config.BAGS_PER_STEP]
        instances = bagset.images[bagset.members[chunk]].to(device)
        embeddings = model.instance_embeddings(instances)
        classes = bagset.labels[chunk].to(device)
        if model.spec["unit"] == "bag":
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
    beta_s = [model.weights_for(H) for H in H_s]
    beta_t = [model.weights_for(H) for H in H_t]

    sigma = wiring._median_sigma(torch.cat([H_s.reshape(-1, H_s.shape[-1]),
                                            H_t.reshape(-1, H_t.shape[-1])]))
    K_st = bag_kernel_matrix(list(zip(H_s, beta_s)), list(zip(H_t, beta_t)), sigma)

    from MIL_CREDA.attention import bag_embedding
    Z_t = torch.stack([bag_embedding(H, w) for H, w in zip(H_t, beta_t)])
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
        for H in model.instance_embeddings(instances):
            beta = model.weights_for(H)
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
               caption: str = "") -> Path:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0.10, 1, 1) if caption else None)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
