"""Repository, data, and experiment-artifact path contracts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


BACKBONES = ("resnet18", "resnet50", "vit_tiny_patch16_224")
RESULT_BACKBONE_DIRECTORIES = {
    "resnet18": "Resnet18",
    "resnet50": "Resnet50",
    "vit_tiny_patch16_224": "Transformer",
}
DATASETS = ("MNIST-USPS-SVHN", "ImageCLEF", "Office-Caltech")


def _repository_candidates(start: Path | None = None) -> Iterable[Path]:
    configured_root = os.environ.get("DOMAIN_ADAPTATION_ROOT")
    if configured_root:
        yield Path(configured_root).expanduser()

    current = (start or Path.cwd()).expanduser().resolve()
    yield current
    yield from current.parents

    # Common managed-notebook locations. DOMAIN_ADAPTATION_ROOT takes priority.
    yield Path("/content/Domain_Adaptation")
    yield Path("/kaggle/working/Domain_Adaptation")
    yield Path("/kaggle/input/domain-adaptation")


def resolve_repository_root(start: Path | None = None) -> Path:
    """Return the project root or raise an actionable configuration error."""
    for candidate in _repository_candidates(start):
        candidate = candidate.resolve()
        if (candidate / "src" / "CREDA").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate

    raise FileNotFoundError(
        "Domain_Adaptation repository not found. Set DOMAIN_ADAPTATION_ROOT to "
        "the directory containing pyproject.toml and src/CREDA."
    )


def resolve_artifact_paths(repository_root: Path | None = None) -> dict[str, Path]:
    """Return the canonical repository-owned directories for experiment artifacts."""
    root = repository_root or resolve_repository_root()
    images = root / "Images"
    return {
        "repository": root,
        "results": images / "Results",
        "models": images / "Models",
    }


def load_dataset_results(backbone: str, dataset: str, repository_root: Path | None = None):
    """Load one unified results CSV for a canonical backbone and dataset.

    Results are stored in ``Images/Results/<artifact-directory>/<dataset>.csv``;
    each file is already unified and is intentionally read without concatenation.
    """
    if backbone not in BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Expected one of: {', '.join(BACKBONES)}."
        )
    if dataset not in DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Expected one of: {', '.join(DATASETS)}."
        )

    results_root = resolve_artifact_paths(repository_root)["results"]
    artifact_directory = RESULT_BACKBONE_DIRECTORIES[backbone]
    results_file = results_root / artifact_directory / f"{dataset}.csv"
    if not results_file.is_file():
        raise FileNotFoundError(
            f"Results CSV for backbone '{backbone}' and dataset '{dataset}' was not found at "
            f"{results_file}. Expected one unified CSV in Images/Results/{artifact_directory}/."
        )

    import pandas as pd

    return pd.read_csv(results_file)


def _resolve_existing_directory(label: str, env_var: str, candidates: Iterable[Path]) -> Path:
    configured_path = os.environ.get(env_var)
    search_paths = [Path(configured_path).expanduser()] if configured_path else []
    search_paths.extend(candidates)

    for candidate in search_paths:
        candidate = candidate.expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(
        f"{label} data directory not found. Set {env_var} to the extracted dataset directory. "
        f"Searched: {searched}"
    )


def resolve_imageclef_root(repository_root: Path | None = None) -> Path:
    root = repository_root or resolve_repository_root()
    return _resolve_existing_directory(
        "ImageCLEF",
        "DOMAIN_ADAPTATION_IMAGECLEF_ROOT",
        (
            root / ".domain-adaptation-cache" / "ImageCLEF-DA" / "image_CLEF",
            root / "data" / "ImageCLEF-DA" / "image_CLEF",
            root / "ImageCLEF-DA" / "image_CLEF",
            Path("/content/ImageCLEF-DA/image_CLEF"),
            Path("/kaggle/input/imageclef-da/image_CLEF"),
        ),
    )


def resolve_officecaltech_root(repository_root: Path | None = None) -> Path:
    root = repository_root or resolve_repository_root()
    return _resolve_existing_directory(
        "Office-Caltech",
        "DOMAIN_ADAPTATION_OFFICECALTECH_ROOT",
        (
            root / ".domain-adaptation-cache" / "OfficeCaltechDomainAdaptation" / "images",
            root / "data" / "OfficeCaltechDomainAdaptation" / "images",
            root / "OfficeCaltechDomainAdaptation" / "images",
            Path("/content/OfficeCaltechDomainAdaptation/images"),
            Path("/kaggle/input/office-caltech-domain-adaptation/images"),
        ),
    )


def checkpoint_path(models_root: Path, backbone: str, model_type: str, src: str, tgt: str, component: str | None = None) -> Path:
    """Build the canonical checkpoint path produced by ``run_all_models``."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of: {', '.join(BACKBONES)}.")

    model_name = model_type.lower()
    if model_name == "baseline":
        if component not in {"F", "C"}:
            raise ValueError("Baseline checkpoints require component='F' or component='C'.")
        filename = f"{backbone}_Baseline_{src}_{tgt}_{component}.pth"
    else:
        filename = f"{backbone}_{model_name.upper()}_{src}_{tgt}_weights.pth"
    return Path(models_root) / backbone / filename
