# Source-side weighting in domain adaptation: which methods weight or select source samples, classes or subjects, and how they fare in the MIL-CREDA funnel

Run date: 2026-10-10. Fast run: 16 web searches, no full-PDF reads. Companion to `adversarial.md`, `discrepancy_kernel_ot.md`, `selftraining_sfda_distillation.md` and `eeg_and_mil.md` in this folder. For the methods already surveyed there, F1–F3 are reused from those files (tag **[V-prior]**). **[V-run]** means a search this run confirmed the authors, venue and mechanism from an abstract or proceedings page. **[memory]** means the mechanism detail comes from memory and was not re-checked this run.

**Reference point (MIL-CREDA).** Both domains are weighted. On the target side the weight is w_j = 1 − H₂(g_j)/ln C. On the source side, each target subject j gets its own reference μ_j^s = Σ_i π_ji Ψ(B_i^s), with π_ji = g_{j,c(i)} · softmax_i(bag-kernel similarity) taken over source subjects of the same class. This is a **per-target, within-class, similarity-based selection of source subjects**. Attention β also weights instances inside bags, in both domains.

**Funnel.** F1 = class-conditional (or joint) alignment. F2 = no target labels. F3 = target samples weighted by pseudo-label confidence. **F4 = the source side is weighted or selected** (at sample, class or subject/domain level).

## Q1. Does each method in the survey list weight the source?

### Takeaway
Of the 32 listed methods, only **5 clearly weight the source**. Two do it at domain or subject level (DAMSDAN, DDSPR), two at class level (WMMD, PCT), and one at region or image level (TADA). Three more do it weakly or as a side effect: CDAN+E (each source sample is weighted by its own entropy, not by how relevant it is to the target), ETD (attention on source–target pairs) and LMMD/CPDA (a per-class normalisation, which is not a selection). Every self-training, SFDA and TTA method fails F4 by construction, because those methods either see no source data or treat it uniformly.

### Table (survey methods)

| Name | Family / subfamily | F1 | F2 | F3 target weighting | F4 source weighting | Source-weighting mechanism (one line) | Tag | aplicado en EEG |
|---|---|---|---|---|---|---|---|---|
| CDAN | adversarial / multilinear conditioning | ✓ | ✓ | ✗ | **no** | uniform source in the discriminator loss | [V-prior] | ? |
| CDAN+E | adversarial / conditioning + entropy | ✓ | ✓ | ✓ (1+e^{−H}) | **weak yes** | the same entropy weight 1+e^{−H} is applied to source samples in the discriminator loss. It rewards certainty, not relevance to the target | [V-prior]; per-domain application [memory] | ? |
| MADA | adversarial / per-class discriminators | ✓ | ✓ | ✗ (routing) | **no** | ŷ routes source samples to the class discriminators. That is routing, not weighting | [V-prior] | ? |
| SRADA | adversarial / entropy-reweighted | ✓ | ✓ | inverted | **unclear** | the entropy reweighting of the adversarial loss may cover both domains. Not verified | [memory] | ? |
| TADA | adversarial / transferable attention | ✗ (marginal) | ✓ | ✗ | **yes** | local attention from region-level discriminators and global attention from an image-level discriminator. Both up-weight transferable regions and images, in source and target | [V-run] | ? |
| CAN / CDD | discrepancy / class-conditional MMD | ✓ | ✓ | ✓ (cluster filtering) | **no** | the source uses ground-truth class means, uniformly | [V-prior] | ? |
| DCAN / CMMD | discrepancy / conditional embedding operators | ✓ | ✓ | ✓ (max-prob > 0.95) | **no** | uniform source | [V-prior] | ? |
| TPN | discrepancy / prototypes | ✓ | ✓ | ✓ (threshold 0.6) | **no** | uniform source prototypes | [V-prior] | ? |
| DSAN / LMMD | discrepancy / local MMD | ✓ | ✓ | soft | **no (normalisation only)** | w_i^{s,c} = y_ic / Σ_j y_jc. This is class-balancing, not selection | [V-prior] | ? |
| CPDA | discrepancy / path-signature LMMD | ✓ | ✓ | soft (τ optional) | **no (normalisation only)** | LMMD-style per-class normalisation | [V-prior] | ? |
| WMMD | discrepancy / weighted MMD | weak | ✓ | ✗ | **yes (class level)** | source class priors α_c are reweighted so that the source class mix matches the estimated target mix | [V-prior] | ? |
| JDA | discrepancy / joint MMD (shallow) | ✓ | ✓ | ✗ | **no** | uniform | [V-prior] | sí [V-run, SEED baselines] |
| JDOT | OT / joint cost | joint | ✓ | ✗ | **no** | uniform marginals. The coupling only redistributes mass between pairs | [V-prior] | ? |
| DeepJDOT | OT / joint cost, deep | joint | ✓ | ✗ | **no** | the same as JDOT | [V-prior] | ? |
| ETD | OT / attention-weighted | joint | ✓ | borderline | **weak yes (pair level)** | learned attention reweights the source–target pair costs, so a source sample's influence varies per target | [V-prior] | ? |
| RWOT | OT / reliability-weighted | ✓ | ✓ | ✓ (SSR) | **no** | SSR reliability scores the target samples near the decision boundaries | [V-run] | ? |
| PCT | OT / prototype conditional transport | ✓ | ✓ | soft | **yes (class level)** | estimated target class proportions weight the source class prototypes in the transport | [V-prior]; weighting detail [memory] | ? |
| CREDA | discrepancy / entropy-weighted conditional | ✓ | ✓ | ✓ (1−H₂/log C) | **no** | only the target is weighted | [V-prior] | ? |
| DARES | (as surveyed) | — | — | — | **no evidence** | `creda_lineage_novelty.md` documents no source weighting | [V-prior, partial] | ? |
| PFAN | self-training / prototypes | ✓ | ✓ | ✓ | **no** | uniform source | [V-prior] | ? |
| ProDA | self-training / prototypes (segmentation) | ✓ | ✓ | ✓ | **no** | uniform source | [V-prior] | ? |
| ATDOC-NA | self-training / neighbourhood memory | ✓ | ✓ | ✓ | **no** | uniform source | [V-prior] | ? |
| SHOT++ | SFDA | ✓ | ✓ | ✓ | **no (no source data)** | — | [V-prior] | ? |
| DaC | SFDA | ✓ | ✓ | ✓ | **no** | its "source-like" split is a split of target samples | [V-prior] | ? |
| CoWA-JMDS | SFDA | ✓ | ✓ | ✓ | **no** | — | [V-prior] | ? |
| A2Net | SFDA | ✓ | ✓ | ✓ | **no** | — | [V-prior] | ? |
| PLUE | SFDA | ✓ | ✓ | ✓ | **no** | — | [V-prior] | ? |
| T3A | TTA | ✓ | ✓ | ✓ | **no** | — | [V-prior] | ? |
| DANN-RPLI | EEG / subdomain adversarial | ✓ | ✓ | ✓ (reliability filter) | **no** | — | [V-prior] | sí (SEED/SEED-IV/DEAP) [V-prior] |
| PR-PL | EEG / prototypes + marginal DANN | loose | ✓ | ✓ (thresholds) | **no** | source prototypes are uniform | [V-prior] | sí (SEED/SEED-IV) [V-prior] |
| DAMSDAN | EEG / multi-source | ✓ | ✓ | likely | **yes (domain/subject level)** | MMD-based weights over the source domains (subjects) | [V-prior] | sí (SEED/SEED-IV/FACED) [V-prior] |
| DDSPR | EEG / multi-source | not confirmed | ✓ | ✓ | **yes (domain/subject level)** | dynamic source selection with an "agent" domain strategy | [V-prior] | sí (SEED/SEED-IV) [V-prior] |

### Cited Findings
- TADA uses "transferable local attention generated by multiple region-level domain discriminators" and "transferable global attention generated by single image-level domain discriminator to highlight transferable images" (Wang, Li, Ye, Long, Wang, AAAI 2019). — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4472)
- RWOT's weighted OT is "based on SSR … to reduce negative transfer from samples near decision boundaries in the target domain" (Xu, Liu, Wang, Chen, Wang, CVPR 2020). The weighting is on the target side. — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)
- DAMSDAN (MMD source weights), DDSPR (dynamic source selection), WMMD (class-prior reweighting), PCT (estimated class priors), ETD (attention-weighted OT) and CDAN+E (entropy weight): see the rows in `eeg_and_mil.md`, `discrepancy_kernel_ot.md` and `adversarial.md` in this folder.

### Inferences
- Within the survey, no method passes **F1+F2+F3+F4 with per-target-sample source selection**. DAMSDAN comes closest: it passes F1 and F2, F3 is "likely", and its F4 works at domain level. Its weights are global per source domain, not personalised per target subject.
- CDAN+E's source weight is a confidence weight on the source's own predictions. It is a different object from MIL-CREDA's relevance weight π_ji, so it should not be cited as a precedent for "personalised source selection".

### Gaps
- SRADA: whether its entropy reweighting also covers source samples was not verified this run.
- DARES: no source-side mechanism is documented in the prior notes, and it was not searched this run.

## Q2. Source-weighting families missing from the survey

### Takeaway
Six families weight the source explicitly. **Partial DA** and **label-shift importance weighting** do it at class level, with class weights estimated from averaged target predictions or from a confusion matrix. **Instance importance weighting** (KMM/KLIEP) and **curriculum/adversarial selection** (TCL, CMSS, IWAN, ETN) do it at sample level. **Multi-source DA** (MDAN, DARN, M3SDA) does it at domain level. **Unbalanced/partial OT** does it implicitly, by letting outlier source mass go untransported. Only two candidates also pass F1: **DRCN** (weighted class-wise alignment) and **IWCDAN** (class-ratio weights on CDAN). Neither has explicit target confidence weighting (F3). None builds a per-target-sample source reference.

### Table (missing families)

| Name | Family / subfamily | F1 | F2 | F3 target weighting | F4 source weighting | Source-weighting mechanism | Citation tag | aplicado en EEG |
|---|---|---|---|---|---|---|---|---|
| SAN | partial DA / class-wise adversarial | ✓ (K class discriminators) | ✓ | ✗ (ŷ routing) | **class** | averaged target predictions weight each class's discriminator, which "selects out the outlier source classes" | Cao, Long, Wang, Jordan — CVPR 2018 [V-run] | ? |
| PADA | partial DA / adversarial | ✗ (one discriminator) | ✓ | ✗ | **class** | class weight γ = mean target softmax, normalised. It "down-weigh[s] the data of outlier source classes for training both the source classifier and domain adversary" | Cao, Ma, Long, Wang — ECCV 2018 [V-run] | ? |
| IWAN | partial DA / importance-weighted adversarial | ✗ | ✓ | ✗ | **sample** | an auxiliary discriminator flags source samples "potentially from outlier classes". Their weight is roughly 1 − D(z) [memory] | Zhang, Ding, Li, Ogunbona — CVPR 2018 [V-run] | ? |
| ETN | partial DA / progressive example weighting | ✗ | ✓ | ✗ (entropy minimisation) | **sample** | an auxiliary discriminator, trained with the label predictor, gives a "progressive weighting scheme" that measures how transferable each source example is | Cao, You, Long, Wang, Yang — CVPR 2019 [V-run] | ? |
| BA³US | partial DA / balanced alignment | ✗ | ✓ | inverted (AUS emphasises uncertain samples) | **class + augmentation** | class weights from target predictions. A few source samples are added to the target batch so the label distributions match | Liang et al. — ECCV 2020 [V-run]; full author list [memory] | ? |
| DRCN | partial DA / residual + weighted class-wise alignment | **✓** | ✓ | ✗/unclear | **class** | a "weighted class-wise domain alignment loss" that selects out irrelevant source classes and raises the most relevant ones | Li et al. — TPAMI 2020 (arXiv 2004.04914) [V-run]; authors [memory] | ? |
| KMM | instance importance weighting (shallow) | ✗ (covariate shift) | ✓ | ✗ | **sample** | weights source points so that the weighted source RKHS mean matches the target mean | Huang, Gretton, Borgwardt, Schölkopf, Smola — **NIPS 2006** (not 2007) [V-run] | ? |
| KLIEP | instance importance weighting (density ratio) | ✗ | ✓ | ✗ | **sample** | density-ratio model fitted by minimising KL(target ‖ weighted source) | Sugiyama et al. — NIPS 2007 [memory; venue search inconclusive] | ? |
| TrAdaBoost | instance reweighting / boosting | ✗ | **✗ (needs labelled target)** | ✗ | **sample** | boosting rounds lower the weight of source points misclassified on the target task | Dai, Yang, Xue, Yu — ICML 2007 [V-run] | ? |
| BBSE | label-shift class weighting | n/a (p(x\|y) assumed fixed) | ✓ | ✗ | **class** | w_y = q(y)/p(y), estimated by inverting a black-box predictor's confusion matrix | Lipton, Wang, Smola — ICML 2018 [V-run] | ? |
| IWDAN / IWCDAN | label-shift (GLS) + adversarial DA | IWCDAN ✓, IWDAN ✗ | ✓ | ✗ | **class** | estimated class ratios reweight the source samples inside DANN, CDAN and JAN | Tachet des Combes, Zhao, Wang, Gordon — NeurIPS 2020 [V-run] | ? |
| MDAN | multi-source / adversarial | ✗ | ✓ | ✗ | **domain** | the soft version uses a smooth max over the per-source losses, so harder or closer sources get more weight | Zhao, Zhang, Wu, Moura, Costeira, Gordon — NeurIPS 2018 [V-run venue; mechanism memory] | ? |
| M3SDA | multi-source / moment matching | ✗ | ✓ | ✗ | **domain (at inference)** | per-source classifiers are combined with weights (M3SDA-β) | Peng, Bai, Xia, Huang, Saenko, Wang — ICCV 2019 [V-run venue; weighting memory] | ? |
| DARN | multi-source / bound-optimised aggregation | ✗ | ✓ | ✗ | **domain** | domain aggregation weights are set by optimising a discrepancy bound, trading effective sample size against irrelevant sources | Wen, Greiner, Schuurmans — ICML 2020 [V-run] | ? |
| CMSS | multi-source / curriculum | ✗ | ✓ | ✗ | **sample/domain** | an adversarial Curriculum Manager learns which source samples or domains to align, with no domain labels needed | Yang, Balaji, Lim, Shrivastava — ECCV 2020 [V-run] | ? |
| TCL | curriculum / weakly-supervised DA | ✗ | ✓ | ✗ | **sample** | the curriculum marks which source examples are clean and transferable, and noisy ones are down-weighted | Shu, Cao, Long, Wang — AAAI 2019 [V-run] | ? |
| JUMBOT | unbalanced minibatch OT | joint (DeepJDOT-style cost) | ✓ | ✗ | **sample (implicit)** | KL-relaxed marginals let source mass that does not match the target go untransported | Fatras, Séjourné, Flamary, Courty — ICML 2021 [V-run] | ? |
| m-POT | partial minibatch OT | joint | ✓ | ✗ | **sample (implicit)** | only a fraction of mass is transported, so outlier pairs are dropped | Nguyen et al. — arXiv 2108.09645 (ICML 2022 per [memory]) | ? |

### Cited Findings
- PADA "alleviates negative transfer by down-weighing the data of outlier source classes for training both the source classifier and domain adversary" — [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Zhangjie_Cao_Partial_Adversarial_Domain_ECCV_2018_paper.html)
- SAN "circumvents negative transfer by selecting out the outlier source classes and promotes positive transfer by maximally matching the data distributions in the shared label space" — [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Cao_Partial_Transfer_Learning_CVPR_2018_paper.html)
- IWAN identifies "source samples that are potentially from outlier classes" — [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Importance_Weighted_Adversarial_CVPR_2018_paper.pdf)
- ETN learns "a progressive weighting scheme to quantify the transferability of source examples" with an auxiliary discriminator — [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cao_Learning_to_Transfer_Examples_for_Partial_Domain_Adaptation_CVPR_2019_paper.pdf)
- BA³US "randomly leverages a few source samples to augment the smaller target domain during domain alignment so that classes in different domains are symmetric". Its AUS term "emphasizes uncertain samples" — [arXiv 2003.02541](https://arxiv.org/abs/2003.02541v2); [ECCV 2020](https://mlanthology.org/eccv/2020/liang2020eccv-balanced)
- DRCN uses "a weighted class-wise domain alignment loss" that can "select out the irrelevant source classes and enhance the importance scores of the most relevant source classes". It appeared in TPAMI 2020 — [arXiv 2004.04914](https://arxiv.org/abs/2004.04914)
- KMM reweights training points "such that the means of training and test points in a reproducing kernel Hilbert space (RKHS) are close". It appeared at NIPS 2006 — [NeurIPS proceedings](https://papers.nips.cc/paper/2006/hash/a2186aa7c086b46ad4e8bf81e2a3a19b-Abstract.html)
- TrAdaBoost "iteratively re-weight[s] the source domain data" and uses "a small amount of newly labeled data", so it fails F2 — [ICML 2007](https://mlanthology.org/icml/2007/dai2007icml-boosting); [PDF](https://cse.hkust.edu.hk/~qyang/Docs/2007/tradaboost.pdf)
- BBSE estimates the target p(y) with black-box predictors and only needs their confusion matrices to be invertible — [PMLR 80](https://proceedings.mlr.press/v80/lipton18a.html)
- IWDAN and its siblings use "an estimation of the relative class weights between domains and an appropriate reweighting of samples", applied to JAN, DANN and CDAN — [NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/dfbfa7ddcfffeb581f50edcf9a0204bb-Abstract.html)
- DARN "is able to effectively adjust the weight of each source domain during training" — [PMLR 119](https://proceedings.mlr.press/v119/wen20b.html)
- CMSS "iteratively learns which domains or samples are best suited for aligning to the target" and needs no domain labels — [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2238_ECCV_2020_paper.php)
- MDAN: [NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/717d8b3d60d9eea997b35b02b6a4e867-Abstract.html). M3SDA: [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.html)
- TCL's curriculum identifies "which of the source examples are noiseless and transferable" — [AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4425)
- JUMBOT: "minibatch strategy coupled with unbalanced optimal transport can yield more robust behaviors" for DA — [PMLR 139](https://proceedings.mlr.press/v139/fatras21a.html). Partial minibatch OT — [ar5iv 2108.09645](https://ar5iv.labs.arxiv.org/html/2108.09645)

### Inferences
- The source weights fall on three granularities, and MIL-CREDA's π_ji differs from all of them in one respect:
  - **Class level, global**: PADA, SAN, DRCN, BA³US, BBSE, IWDAN, WMMD, PCT. MIL-CREDA's factor g_{j,c(i)} plays this role, but **per target subject** instead of averaged over the target.
  - **Domain/subject level, global**: MDAN, DARN, M3SDA (and, within the survey list, DAMSDAN and DDSPR). MIL-CREDA's softmax over same-class source subjects plays this role, but again **per target subject** and **restricted to within a class**.
  - **Sample level, global**: KMM, KLIEP, IWAN, ETN, TCL, CMSS. KMM is the closest in form, since it builds a weighted RKHS mean of the source, but it builds one global mean that matches the whole target, not one per target subject.
- The only families with **per-target** source weighting are the OT couplings (JDOT, ETD, JUMBOT) and their barycentric mappings. MIL-CREDA's μ_j^s is structurally a soft, kernel-similarity, class-masked barycentric reference, and these couplings are its nearest formal relatives (consistent with `novelty_two_ideas.md`). None of them adds confidence-weighted target terms (F3).
- Defensible claim: "class-conditional alignment + target confidence weighting + **per-target, class-masked, similarity-based selection of source subjects**" was not found in any method screened (the 32 listed and the 18 added). The defensible novelty is the combination plus the per-target granularity, not "weighting the source" as such. Partial DA, importance weighting and multi-source DA are mature lines that weight the source and must be cited.
- DRCN and IWCDAN are the closest to cite as "class-conditional alignment with source class weights". The paper should state that they weight classes globally and do not weight the target by confidence.

### Gaps
- KLIEP venue and authors were not confirmed this run (the search returned nothing usable). Treat NIPS 2007 as [memory].
- The MDAN soft-max weighting and the M3SDA-β inference weighting were not re-verified from the papers.
- Whether DRCN uses any target confidence threshold (F3) was not checked in the PDF.
- The m-POT venue (ICML 2022) is from memory.

## Note on the "aplicado en EEG" column

Per the user's change, EEG-specific source-selection methods were not searched, and the families above are kept general. The column says **sí** only where an EEG application was verified. That means either an EEG method from `eeg_and_mil.md` [V-prior], or a generic method that a search this run found reported as a SEED baseline. One search this run returned SEED cross-subject comparisons that list DAN, DANN, DCORAL and DDC as single-source baselines ([arXiv 2408.10235](https://arxiv.org/pdf/2408.10235)) and TCA/JDA comparisons ([TANN, arXiv 2009.09585](https://arxiv.org/pdf/2009.09585); [He & Li 2018](https://bcmi.sjtu.edu.cn/home/lubaoliang/papers/2018/HeLi2018Cross-SubjectEmotionRecognitio.pdf)). These attributions are snippet-level and not PDF-verified. **?** means no EEG application was verified this run. It does not mean the method was never used on EEG.
