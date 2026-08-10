# CREDA: Conditional Rényi α-Entropy Domain Adaptation

This repository contains an implementation of **CREDA** (Conditional Rényi α-Entropy Domain Adaptation), a deep-learning approach for unsupervised image domain adaptation. It combines source-domain classification with class-conditional feature alignment based on Rényi quadratic entropy and confidence-weighted target pseudo-labels.

The implementation follows the article: [Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting](https://www.mdpi.com/2227-7390/13/16/2602).

## Included components

- CREDA model and loss implementations in `src/CREDA/models.py`.
- Training utilities for CREDA and comparison methods in `src/CREDA/training_pipeline.py`.
- Result-generation notebooks under `CREDA/Notebooks/` and stored model/result artifacts under `CREDA/`.

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
