# CREDA: Conditional Rényi α-Entropy Domain Adaptation

This repository contains an implementation of **CREDA** (Conditional Rényi α-Entropy Domain Adaptation), a deep-learning approach for unsupervised image domain adaptation. It combines source-domain classification with class-conditional feature alignment based on Rényi quadratic entropy and confidence-weighted target pseudo-labels.

The implementation follows the article: [Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting](https://www.mdpi.com/2227-7390/13/16/2602).

## Included components

- CREDA model and loss implementations in `src/CREDA/models.py`.
- Training utilities for CREDA and comparison methods in `src/CREDA/training_pipeline.py`.
- Result-generation notebooks under `CREDA/Notebooks/` and stored model/result artifacts under `CREDA/`.
- One toyset training tutorial per method (see below).

## Toyset tutorials: how to train CREDA and MIL-CREDA

Each method has exactly one toyset notebook. It is a complete, readable training example: data, model, loss, optimizer, training loops, evaluation and plots, on the three toyset families (`lunas`, `toroide`, `rollo`) with a single architecture, an MLP with BatchNorm (fixed, not a choice).

| Method | Notebook | What you choose |
|--------|----------|-----------------|
| CREDA | `CREDA/Notebooks/Toyset.ipynb` | `VERSION = "v2"` (default, the current two-phase protocol) or `"v01"` (the single joint training of the previous publication). Each version is one dictionary in an `if/else`. |
| MIL-CREDA | `MIL-CREDA/Notebooks/Toyset.ipynb` | The budget cell; bags, the adapted arm and the declared hyperparameters come from `MIL_CREDA_Benchmark`. |

### Run it

1. Install the dependencies from `requirements.txt` in a virtual environment.
2. Start Jupyter as usual, either from the repository root (`jupyter lab`, then open the notebook) or from the notebook's folder (`cd CREDA/Notebooks && jupyter lab Toyset.ipynb`).
3. Run all cells. The kernel starts in the notebook's folder and the first cell resolves the repository root from there, so no install of `src/` is needed.

Both notebooks train on CPU, download nothing and write no separate result files; models, histories and figures stay in memory. The notebooks are saved with executed outputs. Each starts with one input-data figure: three 3D panels (one per toyset), with source (`x`) and target (`o`) overlaid on the same axes. Source points are coloured by class and target points stay grey, without using target labels. Outputs also include an adaptation GIF and a confidence-`w` animation for each family (six GIFs per method), alongside the static summaries. CREDA reads confidence on points; MIL-CREDA reads it on bag posteriors. Adaptation frames come from real training epochs; the confidence paths are constructed sweeps, not training time.

Rendering controls do not change training: CREDA's `FRAMES` selects captured epochs and `WALK_STEPS` sets the confidence-walk resolution; MIL-CREDA's `BARRIDOS_W` sets its confidence-walk resolution.

### Budgets are didactic, not published results

The defaults are short so the notebooks finish quickly. They are not tuned and they are not the image benchmarks or the stored runs under `*/Results/`; the accuracies a notebook prints belong to that run only.

| Notebook | Defaults (one seed) |
|----------|---------------------|
| CREDA, `v2` | `source_epochs=60` (ceiling, early stop with `source_patience=15`), then `adaptation_epochs=20` |
| CREDA, `v01` | `epochs=60` of joint training |
| MIL-CREDA | `EPOCAS_1 = 60` (ceiling, `PACIENCIA_1 = 15`), then `EPOCAS_2 = 20` |

To change the CREDA budget, set `ADJUSTMENTS` (for example `{"source_epochs": 3, "adaptation_epochs": 2}`); for MIL-CREDA, edit the budget cell.

`MIL-CREDA/AGREED-TOYSET.md` is kept as the historical record of the earlier toyset experiment; the tutorials above replace it as the entrypoint.

## Verified dependencies

`requirements.txt` declares the notebook and runtime dependencies: Jupyter, IPython kernel support, gdown, Matplotlib, NumPy 1.26.4, pandas 2.2.3, scikit-image, scikit-learn, timm, PyTorch, torchcam, torchvision, tqdm, and umap-learn. The project build configuration requires `setuptools>=61`.

## Citation

```bibtex
@Article{math13162602,
AUTHOR = {Pérez-Rosero, Diego Armando and Álvarez-Meza, Andrés Marino and Castellanos-Dominguez, German},
TITLE = {Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting},
JOURNAL = {Mathematics},
VOLUME = {13},
YEAR = {2025},
NUMBER = {16},
ARTICLE-NUMBER = {2602},
URL = {https://www.mdpi.com/2227-7390/13/16/2602},
ISSN = {2227-7390},
ABSTRACT = {Domain adaptation is a key approach to ensure that artificial intelligence models maintain reliable performance when facing distributional shifts between training (source) and testing (target) domains. However, existing methods often struggle to simultaneously preserve domain-invariant representations and discriminative class structures, particularly in the presence of complex covariate shifts and noisy pseudo-labels in the target domain. In this work, we introduce Conditional Rényi α-Entropy Domain Adaptation, named CREDA, a novel deep learning framework for domain adaptation that integrates kernel-based conditional alignment with a differentiable, matrix-based formulation of Rényi’s quadratic entropy. The proposed method comprises three main components: (i) a deep feature extractor that learns domain-invariant representations from labeled source and unlabeled target data; (ii) an entropy-weighted approach that down-weights low-confidence pseudo-labels, enhancing stability in uncertain regions; and (iii) a class-conditional alignment loss, formulated as a Rényi-based entropy kernel estimator, that enforces semantic consistency in the latent space. We validate CREDA on standard benchmark datasets for image classification, including Digits, ImageCLEF-DA, and Office-31, showing competitive performance against both classical and deep learning-based approaches. Furthermore, we employ nonlinear dimensionality reduction and class activation maps visualizations to provide interpretability, revealing meaningful alignment in feature space and offering insights into the relevance of individual samples and attributes. Experimental results confirm that CREDA improves cross-domain generalization while promoting accuracy, robustness, and interpretability.},
DOI = {10.3390/math13162602}
}
```
