# Domain adaptation for multiple-instance learning (MIL): bag-level and subject-level adaptation, and the closest neighbours of MIL-CREDA

Research run: 2026-10-10. About 21 search/fetch calls. Verification status is marked on every item:
**[V]** = the abstract or proceedings page was fetched or the search result came from the primary page during this run; **[S]** = only a search-result snippet was seen this run (primary page not fetched, or fetch returned 403); **[M]** = from background knowledge, NOT verified this run. Do not cite [M] items without checking them first.

## Q1. Which papers exist on MIL domain adaptation (WSI/histopathology, weakly supervised DA with bags, cross-subject EEG with MIL, set-level DA), 2018–2026?

### Takeaway
The MIL+DA literature is small and scattered. Most of it is histopathology. Most of it aligns **instance (patch) features**, or aligns slide features without class conditioning, using adversarial training (DANN-style). Only one verified work builds **bag-level kernel distances from instance kernels** (Wang & Zhao, users as bags, generalized MMD), and it is not class-conditional and uses no pseudo-labels. I found no paper that does cross-subject EEG domain adaptation with subjects treated as MIL bags.

### Cited Findings

**Histopathology / medical MIL + DA**
- **Hashimoto et al., "Multi-scale Domain-adversarial Multiple-instance CNN for Cancer Subtype Classification with Unannotated Histopathological Images", CVPR 2020 [V].** Bag = one WSI (patches as instances). Source/target = hospitals and specimens whose staining differs. It combines MIL, domain-adversarial learning and multi-scale learning so that features stay stable "against differences in staining conditions among hospitals/specimens". Tested on 196 malignant-lymphoma cases from several hospitals. Family: adversarial (DANN-style) on patch features. Bag labels are available in all domains, so this is closer to multi-domain invariance than to UDA. No pseudo-labels reported. — [CVF open access](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hashimoto_Multi-scale_Domain-adversarial_Multiple-instance_CNN_for_Cancer_Subtype_Classification_with_Unannotated_CVPR_2020_paper.pdf); [arXiv 2001.01599](https://arxiv.org/abs/2001.01599v1)
- **Takahama et al., "Domain Adaptive Multiple Instance Learning for Instance-level Prediction of Pathological Images", ISBI 2023 (oral) [V].** Source has labels for every instance; target has bag (slide) labels only. The goal is instance-level abnormality prediction in the target. The pipeline merges MIL and DA supervision through "pseudo-labels with high confidence". Uses pseudo-labels (instance-level). The abstract does not say whether alignment is adversarial or MMD-based. — [arXiv 2304.03537](https://arxiv.org/abs/2304.03537)
- **Yamaguchi, Iwana, Bise, Harada, Okuo, Tanaka, Shiku, "Domain Adaptation for Ulcerative Colitis Severity Estimation Using Patient-Level Diagnoses", MICCAI-workshop MLMI 2025 [V].** Bag = patient (endoscopic images). Domains = hospitals and imaging devices. The target has patient-level (bag) labels, so this is *weakly supervised* DA, not UDA. Alignment is **class-wise**, on instance features, using "Shared Aggregation Tokens and a Max-Severity Triplet Loss". It is the closest verified example of class-conditional alignment inside a MIL model, but the class information comes from real target bag labels, not pseudo-labels. — [arXiv 2509.14573](https://arxiv.org/abs/2509.14573)
- **Shou et al., "Graph Domain Adaptation with Dual-branch Encoder and Two-level Alignment (DETA) for WSI-based Survival Prediction", arXiv Nov 2024 [V].** WSI graphs across hospitals. It aligns at two levels: (1) **category level**, coupling the two branches to reduce divergence between the domains' category distributions; (2) **feature level**, using adversarial perturbation of source features. Family: adversarial + GNN. Pseudo-labels are not mentioned in the abstract. Its claim to be the first work on WSI domain shift conflicts with Hashimoto 2020 (above). — [arXiv 2411.14001](https://arxiv.org/abs/2411.14001)
- **StructMIL (npj Digital Medicine 2025) [S].** A prototype- and graph-based MIL model with a *domain-generalization* strategy (contrastive alignment + adversarial confusion + consistency regularization). Reported gains: +3.2% cross-centre AUC on Camelyon16 and +7.4% Cohen's κ cross-scanner on PANDA. This is DG (no target data), not UDA. — [Nature npj DM](https://www.nature.com/articles/s41746-025-02289-4)
- **Marikkar, Awais, Atito, "Domain Adaptation Without the Compute Burden for Efficient WSI Analysis", arXiv Mar 2026 [V].** "Domain adaptation" here means adapting the feature extractor to the pathology task with PEFT inside end-to-end MIL (Camelyon16, TCGA, BRACS). It is **not** cross-domain UDA, so do not count it as MIL-UDA. — [arXiv 2603.15774](https://arxiv.org/abs/2603.15774)
- **"Test-Time Instance Selection for Improved WSI Analysis" (TTIS), arXiv Aug 2026 [S].** A training-free, plug-in patch selection step at inference for existing MIL models. Despite the name it is not distribution alignment. It is relevant only as a test-time MIL neighbour. — [arXiv 2608.14759](https://arxiv.org/abs/2608.14759)
- **"Semantic-Anchored Evidential Fusion for Domain-Robust Whole-Slide Survival Analysis", arXiv Jun 2026 [S]**, title only. — [arXiv 2606.19966](https://arxiv.org/pdf/2606.19966)
- **"Hierarchical Prototype-based Domain Priors for MIL in Multimodal Histopathology", arXiv 2026 [S]**, title only. — [arXiv 2604.23982](https://arxiv.org/pdf/2604.23982)
- Deep Microscopy Adaptation Network (MICCAI 2019) adapts WSI-trained models to microscopy images with deep UDA [S]. The snippet does not show whether it uses a MIL formulation. — [Springer](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_40)

**Non-medical MIL + DA (bags = users, sensor windows, etc.)**
- **Wang & Zhao, "Multi-instance Domain Adaptation for Vaccine Adverse Event Detection", WWW 2018 (arXiv 2009.04901) [V].** Bag = user. Instances = messages. Source = VAERS reports, target = social media. **"Message-level generalized MMD distances are synthesized by newly proposed mixed instance kernels to user-level distances"**, which is a bag-level kernel distance built from an instance kernel, structurally the closest verified precedent to Ψ(B) as an RKHS mean embedding. Family: MMD/kernel, optimized with ADMM + CCP. It is not class-conditional, and the abstract mentions no pseudo-labels. — [arXiv 2009.04901](https://arxiv.org/abs/2009.04901)
- **"Multiple-instance domain adaptation for cost-effective sensor-based human activity recognition", Future Generation Computer Systems, 2022 [S].** It proposes a **bag-level adaptation strategy called "Blast"**, combining high-level and bag-level adaptation. Authors, volume and loss family were not verified (the ScienceDirect page returned 403). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X22000814)
- **"Domain Transfer Multi-Instance Dictionary Learning", arXiv 1605.08397 (2016) [S].** Adapts a source multi-instance dictionary and classifier to the target by mapping target bags into a bag-level feature space. — [arXiv 1605.08397](https://arxiv.org/pdf/1605.08397)
- **"Multi-instance transfer metric learning by weighted distribution and consistent MLE" (MITML), 2018 [S].** Uses **bag weighting** to bridge the domains' distributions and learns a metric for classifying target bags. — [HKBU scholars](https://scholars.hkbu.edu.hk/en/publications/multi-instance-transfer-metric-learning-by-weighted-distribution-/)
- **Adaptive p-Posterior Mixture-Model Kernels for MIL, ICML 2008 [S].** Shows that the mechanism linking instances to bag labels differs across domains. It learns domain-specific mechanisms with mixture models and an adaptive **bag kernel**. Pre-2018, but a conceptual ancestor of "the attention/pooling rule can shift across domains". — [ICML 2008 PDF](https://icml.cc/imls/conferences/2008/papers/130.pdf)
- AKT-MIL (Adaptive Knowledge Transfer for MIL, AAAI) [S]: cross-*category* transfer, not cross-domain alignment. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/8895/8754)
- CoDATS-style "Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data" (arXiv 2005.10996) [S]. Its weak supervision is target label proportions, which is related to learning from label proportions but is not MIL. — [arXiv 2005.10996](https://arxiv.org/pdf/2005.10996)

**EEG: subject as bag, and subject-specific encoders**
- **BridgeMIL (Ma et al., arXiv Jul 2026) [V].** Bag = subject, instance = short EEG segment, for disease diagnosis. Two stages: an encoder pretrained with temporal alignment and within-subject sub-bags, then an attention-based MIL aggregator supervised only at subject level. It notes "distinct subject-wise clusters" in the representations but does **no cross-subject domain adaptation**. Three EEG disease datasets; 76.57% mean accuracy. — [arXiv 2607.27274](https://arxiv.org/abs/2607.27274)
- **Lopes, Schwartz, Chevallier, de Camargo, Aristimunha, "Learning aligned EEG representations with subject-specific encoders", arXiv Jun 2026 (rev. Sep 2026) [V].** Cross-subject motor imagery/execution. **Subject-specific encoders/heads** "place each subject close to its own latent manifold". Alignment comes from architectural pressure only, described as complementary to DA such as Euclidean Alignment. No MMD, no adversary, no pseudo-labels. A per-subject low-rank adapter with 2Cr parameters recovers full accuracy. This is the closest verified precedent for MIL-CREDA's per-domain encoders, but it has no bag structure. — [arXiv 2606.16462](https://arxiv.org/abs/2606.16462)

### Inferences
- In what I found, the verified MIL-DA works split three ways: (a) instance-level adversarial (Hashimoto), (b) weakly supervised DA where the target *does* have bag labels (Takahama, Yamaguchi), and (c) kernel or metric bag distances in older non-deep work (Wang & Zhao, MITML, p-posterior kernels). MIL-CREDA's setting has **no target bag labels**, does **class-conditional bag alignment with pseudo-labels**, and treats the **bag as an RKHS mean embedding**. No verified paper sits at that intersection.
- Wang & Zhao 2018 is the strongest "prior art" for a bag-level kernel distance built from instance kernels. A paper on MIL-CREDA should cite it and contrast with it: theirs is marginal, shallow, ADMM-optimized and has no attention.

### Gaps
- No verified paper on cross-subject **emotion** EEG (SEED/DEAP) that uses subjects or trials as MIL bags with DA.
- No full text read for Blast (403), so its alignment loss is unknown.
- I did not find a MICCAI/TMI/MedIA paper titled with "unsupervised domain adaptation" for slide-level MIL classification with an unlabeled target. Such papers may exist under other names (e.g. "stain-invariant MIL", "cross-site MIL"); this needs a dedicated Google Scholar pass.

## Q2. Does any work align bag-level class-conditional distributions, or use kernel mean embeddings of bags (SMM, distribution regression, set kernels, attention-MIL)?

### Takeaway
Kernel mean embeddings of bags are a mature tool for *supervised* learning on distributions (SMM, distribution regression, set kernels). Attention-MIL (Ilse 2018) is formally a weighted mean embedding. I found only one verified case of KME-of-bags used for **domain adaptation** (Wang & Zhao), and it is not class-conditional. Class-conditional alignment inside MIL appears only with real target bag labels (Yamaguchi 2025) or as "category-level" alignment in DETA (2024). Neither uses an RKHS bag embedding.

### Cited Findings
- **Ilse, Tomczak, Welling, "Attention-based Deep MIL", ICML 2018 (PMLR v80) [V, via search result on proceedings PDF].** The bag representation is "a weighted average of instances (low-dimensional embeddings) where the weights are determined by a neural network". With a kernel feature map this is exactly a weighted empirical kernel mean embedding. — [PMLR](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)
- Muandet, Fukumizu, Sriperumbudur, Schölkopf, "Kernel Mean Embedding of Distributions: A Review and Beyond" (FnT ML 2017). It lists domain adaptation and two-sample tests among KME applications. — [arXiv 1605.09522](https://arxiv.org/pdf/1605.09522)
- Wang & Zhao (WWW 2018) build user-level (bag) MMD from message-level kernels via "mixed instance kernels" [V]. — [arXiv 2009.04901](https://arxiv.org/abs/2009.04901)
- Zhang et al., "Domain Adaptation under Target and Conditional Shift", ICML 2013 [S]. A kernel-embedding treatment of conditional/target shift at instance level, and the classical basis for class-conditional KME alignment. — [PMLR v28](http://proceedings.mlr.press/v28/zhang13d.pdf)
- Domain Adaptation with Conditional Transferable Components (Gong et al., ICML 2016) [S]: conditional-shift DA via kernel embeddings, at instance level. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5321138/)
- **[M] not verified this run:** Muandet et al., "Learning from Distributions via Support Measure Machines", NeurIPS 2012. Szabó, Sriperumbudur, Póczos, Gretton, "Learning Theory for Distribution Regression", JMLR 2016. Sutherland/Póczos et al. set-kernel and "deep sets of distributions" work. Blanchard, Lee, Scott (NeurIPS 2011) and Blanchard et al. (JMLR 2021), "marginal transfer learning / domain generalization", which use the KME of each *domain's* marginal as an extra input. Muandet, Balduzzi, Schölkopf, DICA (ICML 2013). These are the right lineage for "a bag is a distribution", but their citations must be checked before use.

### Inferences
- MIL-CREDA's object, a class-conditional distribution **over bag embeddings** where each bag is itself a KME, is a "distribution of distributions". The Gram matrix over bags then corresponds to a second-level (set) kernel in the spirit of SMM and distribution regression. Supervised work on that object exists [M], and an unsupervised marginal version exists (Wang & Zhao) [V]. No class-conditional, pseudo-labelled version was found.

### Gaps
- Not verified this run: whether Sutherland/Póczos or the "distribution regression" community published any DA-specific bag-level method after 2018.

## Q3. Which classic DA methods (DANN, MMD/JAN, CDAN, CAN, DSAN, SHOT, MCC, CORAL) have been applied to MIL, and what problems appeared?

### Takeaway
DANN-style adversarial alignment is the only classic method with a verified, well-cited MIL application (Hashimoto CVPR 2020). MMD reached MIL in a shallow form (Wang & Zhao). I found no verified MIL application of CDAN, CAN, DSAN, SHOT, MCC or CORAL. I also found no paper that explicitly reports few bags per class per batch or attention collapse under domain shift. Those remain plausible but undocumented problems.

### Cited Findings
- DANN on MIL patch features: Hashimoto et al. CVPR 2020 [V]. — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hashimoto_Multi-scale_Domain-adversarial_Multiple-instance_CNN_for_Cancer_Subtype_Classification_with_Unannotated_CVPR_2020_paper.pdf)
- Adversarial feature-level alignment combined with category-level coupling on WSI graphs: DETA 2024 [V]. — [arXiv 2411.14001](https://arxiv.org/abs/2411.14001)
- Generalized MMD lifted to bags: Wang & Zhao [V]. — [arXiv 2009.04901](https://arxiv.org/abs/2009.04901)
- Adversarial confusion + contrastive alignment for MIL *domain generalization*: StructMIL [S]. — [npj DM](https://www.nature.com/articles/s41746-025-02289-4)
- High-confidence pseudo-labels in a MIL+DA pipeline (closest to SHOT- and CAN-style self-training): Takahama ISBI 2023 [V]. — [arXiv 2304.03537](https://arxiv.org/abs/2304.03537)
- Attention instability is recognized as a MIL problem *without* domain shift: ASMIL ("Attention-Stabilized MIL for WSI", arXiv 2026) [S] — [arXiv 2603.06658](https://arxiv.org/pdf/2603.06658). CAR-MIL counterfactual attention regularization (arXiv 2026) [S] — [arXiv 2609.08419](https://arxiv.org/pdf/2609.08419).

### Inferences
- Instance-level vs bag-level shift: the Adaptive p-Posterior kernel paper [S] ([ICML 2008](https://icml.cc/imls/conferences/2008/papers/130.pdf)) states that the instance-to-bag-label mechanism itself differs across domains. That is an argument that aligning instances alone cannot remove bag-level shift, and it supports MIL-CREDA's choice to align at bag level.
- Few bags per class per batch is a structural consequence of bag-level class-conditional alignment: a batch holds B bags, not B instances. This is inferred, not sourced. The attention-stability literature suggests attention collapse is a live risk when the encoders are domain-specific and the pooling is shared.

### Gaps
- I found no source that empirically reports how CDAN, MCC, CORAL or SHOT fail when transplanted to bag embeddings. This is a gap the paper could fill with ablations.

## Q4. Cross-subject EEG emotion/affect DA (SEED, DEAP): which treat the subject or trial as a set?

### Takeaway
The dominant SEED/DEAP DA line is **multi-source**: each source subject is its own domain, aligned to the target at the level of individual samples or features (MMD, joint DA, contrastive). The subject is a *domain* in these methods, never a *bag* with a pooled embedding. No verified emotion-EEG work uses MIL pooling or bag-level kernel embeddings.

### Cited Findings
- **MS-MDA** (Multisource Marginal Distribution Adaptation, cross-subject and cross-session, SEED) [S]: one branch per source subject, marginal (MMD) alignment. — [arXiv 2107.07740](https://arxiv.org/pdf/2107.07740)
- **MSJDA**, multi-source *joint* (marginal + conditional) DA for cross-subject/session emotion EEG [S]. — [PMC9520599](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9520599/)
- **Latent Multi-source Adaptation (LMA)**: multiple domain-invariant latent subspaces [S]. — [PMC9091911](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9091911/)
- **MSMRA**: subjects and sessions as multiple domains, with multi-representation alignment [S]. — [PMC8792438](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8792438/)
- Multi-source co-adaptation mining correlation information [S]. — [PMC8155359](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8155359/)
- Contrastive cross-subject EEG emotion recognition (Sci. Rep. 2025) [S]. — [Nature Sci Rep](https://www.nature.com/articles/s41598-025-13289-5)
- Deep feature aggregation + multi-source DA (Cogn. Neurodyn. 2025) [S]. — [Springer](https://link.springer.com/article/10.1007/s11571-025-10379-y)
- Outside emotion: BridgeMIL [V] treats the subject as a MIL bag but does no DA. Subject-specific encoders [V] give per-subject latent manifolds but no bags. — [arXiv 2607.27274](https://arxiv.org/abs/2607.27274); [arXiv 2606.16462](https://arxiv.org/abs/2606.16462)
- [M] not verified this run: the DAN/DANN baselines on SEED (e.g. Li et al., "domain adaptation for EEG emotion recognition", 2018–2019) and BiDANN/BiHDM (Li et al., TAFFC). Verify before citing.

### Inferences
- MIL-CREDA's subject-as-bag with class-conditional bag alignment is a genuine change of unit: from subject-as-domain over samples to subject-as-sample within a domain. This is a defensible novelty claim for affect EEG, provided a Scholar pass confirms that no SEED/DEAP MIL-DA paper exists.

### Gaps
- No full-text check of whether any SEED/DEAP paper pools trial segments with attention before aligning.

## Q5. Gaps: is there any work combining information-theoretic (Rényi, entropy-based) alignment with MIL?

### Takeaway
I found none. Matrix-based Rényi entropy has been used for (non-MIL) class-conditional DA, in CREDA itself, but I found no paper combining Rényi or matrix-entropy alignment with MIL or bag embeddings. This is the clearest open gap.

### Cited Findings
- **CREDA, "Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting", *Mathematics* 13(16):2602, 2025 [S]** (MDPI page returned 403; details from the indexed snippet). It combines kernel-based **class-conditional** alignment, a differentiable **matrix-based Rényi quadratic entropy** estimator, and entropy weighting that **down-weights low-confidence pseudo-labels**. Evaluated on Digits, ImageCLEF-DA and Office-31, at instance level only, with no MIL. — [DOI 10.3390/math13162602](https://doi.org/10.3390/math13162602); [ResearchGate](https://www.researchgate.net/publication/394905031_Conditional_Domain_Adaptation_with_a-Renyi_Entropy_Regularization_and_Noise-Aware_Label_Weighting)
- Foundations: matrix-based Rényi α-order entropy from the normalized eigenspectrum of a Gram matrix in RKHS, with a multivariate/joint extension (Yu et al.) [V, arXiv abstract page in results]. — [arXiv 1808.07912](https://arxiv.org/abs/1808.07912)
- Computationally efficient approximations for matrix-based Rényi entropy [S] are relevant if bag Gram matrices are large. — [ResearchGate](https://www.researchgate.net/publication/367196167_Computationally_Efficient_Approximations_for_Matrix-Based_Renyi's_Entropy)
- Mansour, Mohri, Rostamizadeh, "Multiple Source Adaptation and the Rényi Divergence" [S]: Rényi divergence in *multi-source* DA theory, not MIL. — [arXiv 1205.2628](https://arxiv.org/pdf/1205.2628)

### Inferences
- Combining (i) bags as attention-weighted RKHS mean embeddings, (ii) class-conditional alignment from pseudo-labels with confidence weights, and (iii) matrix-based Rényi entropy over **bag-level** Gram matrices appears unoccupied in what this run could find. The nearest occupants are one component each: (i) Wang & Zhao 2018 and Ilse 2018, (ii) Yamaguchi 2025 (real labels) and Takahama 2023 (pseudo-labels at instance level), (iii) CREDA 2025 at instance level. Per-domain encoders have an EEG precedent (Lopes et al. 2026).
- The personalized local term, the RKHS distance from each target bag to a personalized source reference, has no verified MIL precedent. Its closest relatives are the multi-source EEG methods that pick or weight source subjects per target (MS-MDA family [S]).

### Gaps
- A Google Scholar search for "Rényi" + "multiple instance" and "Cauchy–Schwarz divergence" + "bag" was not run as a separate query. A negative claim of novelty needs that check plus a citation-forward search on CREDA (DOI above).
- The CREDA full text, including its exact relationship to MIL-CREDA, was not retrieved (403).
