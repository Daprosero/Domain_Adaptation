# Self-training, pseudo-labelling, source-free, distillation and test-time adaptation: a fixed funnel to position MIL-CREDA

Scope: self-training, pseudo-labelling, source-free (SFDA), teacher–student/distillation, transformer/foundation-model and test-time (TTA) methods for domain adaptation. Every method passes through the same funnel: **Stage 0** (strong or influential) → **F1** (explicit class-conditional structure) → **F2** (no target labels) → **F3** (pseudo-labels weighted or filtered by confidence/uncertainty).

This file complements, and does not repeat, `SOTA adaptación condicional MIL CREDA/weighting_schemes.md`. That note already covers in depth the code of CDAN+E, MCC, SHOT IM, ATDOC-NA, FixMatch/FlexMatch/SoftMatch/UPS, the robust losses, the detach analysis, and the algebra of w = 1 − H2/ln C. Here those methods appear only as funnel rows.

**Verification legend (all checked this run, 2026-10-10).**
- Metadata: **[arXiv]** means title, authors and date came from the arXiv API this run, with the venue from the arXiv comment field. **[CVF]**, **[PMLR]** and **[NeurIPS]** mean the official proceedings page was found this run. **[UNVERIFIED venue]** means the paper exists but its venue was not confirmed this run.
- Math: **[code]** means read in the official repository this run. **[paper]** means read in the paper text this run. **[memory]** means reproduced from memory and not re-read this run.

**MIL-CREDA reference (what "we" do).** Bags of instances. Ψ(B) is an attention-weighted RKHS mean embedding. There are two encoders, and the source encoder is frozen. Each bag is routed to one class by a hard pseudo-label, with confidence w = 1 − H2(g)/ln C, where H2(g) = −ln Σ_k g_k². A per-class matrix-Rényi-2 score is computed on Gram matrices; it is a function of the Cauchy–Schwarz divergence between covariance operators. A confidence gate switches that score on or off. A local term measures the RKHS distance to a personalised source reference built from the full soft prediction.

---

## Q1. Funnel classification of every method (master table)

### Takeaway
Of about 50 methods screened, **10 pass all three filters**: PFAN, CAN, ProDA, ATDOC-NA, SHOT++, DaC, CoWA-JMDS, A2Net (pair level), PLUE and T3A. Two more are borderline: the 2026 dual-teacher DDSR, and TAST, whose filter was not verified. Almost every method that fails is discarded at **F1**. Entropy minimisation, marginal alignment, consistency and EMA-teacher methods have no explicit class-conditional structure, and the TTA family in particular dies there. A smaller group fails at **F3**: these align class-conditionally but trust every pseudo-label equally (MSTN, SHOT, NRC, AaD, G-SFDA, CDTrans, PCL, CS-DA). No method in scope fails at F2, because all are unsupervised on the target.

### Cited Findings

Columns: F1/F2/F3 are ✓ (pass), ✗ (discarded here) or — (not reached). PL means pseudo-label.

**Pseudo-labelling and confidence (UDA)**

| Method | Authors, venue, year | F1 | F2 | F3 | Stage reached / why discarded | PL hard/soft; weight; detached? | MIL bags? |
|---|---|---|---|---|---|---|---|
| CBST | Zou, Yu, Kumar, Wang — ECCV 2018 [arXiv 1810.07911] | ✗ (borderline) | — | — | F1. Its class-wise thresholds are class-balanced **selection**, not alignment of class-conditional structure | Hard; 0/1 by per-class threshold; non-differentiable selection | Yes, with a bag posterior |
| CRST | Zou, Yu, Liu, Kumar, Wang — ICCV 2019 Oral [arXiv 1908.09822] | ✗ (borderline) | — | — | F1. Same as CBST, plus confidence regularisers (LR: soft PLs; MR: L2/ENT/KLD) | Soft (LR) or hard + MR; class-balanced selection | Yes |
| MSTN | Xie, Zheng, Chen, Chen — ICML 2018 [PMLR v80 xie18c] | ✓ | ✓ | ✗ | F3. Per-class centroid alignment, but no PL weighting or filtering. Robustness comes only from the moving-average centroid | Hard; none | Yes (centroids of bag embeddings) |
| **PFAN** | Chen, Xie, Huang, Rong et al. — CVPR 2019 [arXiv 1811.08585] | ✓ | ✓ | ✓ | **Survivor** | Hard; w_j = 1[ψ ≥ τ] with a cosine-to-prototype score and a scheduled τ | Yes |
| **CAN** | Kang, Jiang, Yang, Hauptmann — CVPR 2019 [arXiv 1901.00976] | ✓ | ✓ | ✓ | **Survivor** (kernel class-aware CDD, filtering by distance to cluster) | Hard (clustering); keep if dist < D0; drop classes with too few samples | Yes (kernel on bag embeddings) |
| **ProDA** | Zhang, Zhang, Zhang, Chen et al. — CVPR 2021 per common citation [arXiv 2101.10979; UNVERIFIED venue] | ✓ | ✓ | ✓ | **Survivor** (segmentation) | Soft PL × prototype weight → hard argmax + threshold; weights computed under no_grad | Yes |
| ATDOC-NC | Liang, Hu, Feng — CVPR 2021 [arXiv 2007.04171] | ✓ | ✓ | ✗ | F3. Centroid memory, unweighted CE | Hard; none | Yes |
| **ATDOC-NA** | same | ✓ (neighbourhood aggregation of class predictions) | ✓ | ✓ | **Survivor** | Hard; w = max_c of neighbour-aggregated, sharpened memory; detached (prior note, code) | Yes |
| MCC | Jin, Wang, Long, Wang — ECCV 2020 [arXiv 1912.03699] | ✗ (borderline) | — | — | F1. The class-confusion matrix penalises between-class correlation; nothing aligns per-class structure | Soft; w = 1 + e^{−H}, detached (prior note) | Yes |
| BNM | Cui, Wang, Zhuo, Li et al. — CVPR 2020 Oral [arXiv 2003.12237] | ✗ | — | — | F1. Batch nuclear norm acts on the marginal | No PL | Yes |
| AdaMatch | Berthelot, Roelofs, Sohn, Carlini et al. — ICLR 2022 [arXiv 2106.04732] | ✗ | — | — | F1. Relative confidence threshold plus *marginal* distribution alignment (would pass F3) | Hard; relative threshold | Yes |
| CST | Liu, Wang, Long — NeurIPS 2021 per common citation [arXiv 2103.03571; UNVERIFIED venue] | ✗ | — | — | F1. Cycle (bi-level) PL consistency plus Tsallis entropy regulariser; no class-conditional alignment | Hard PL; no per-sample weight (Tsallis is a regulariser) | Yes |
| PCL | Li, Zhang, Wang, Hou et al. — IJCAI 2024 [arXiv 2111.06021] | ✓ (features clustered around class weights) | ✓ | ✗ | F3. Probabilistic contrast has no PL weighting of its own; FixMatch is added only to some baselines and variants | — | Yes |

**Source-free**

| Method | Authors, venue, year | F1 | F2 | F3 | Stage / why | PL; weight; detached? | MIL? |
|---|---|---|---|---|---|---|---|
| SHOT | Liang, Hu, Feng — ICML 2020 [arXiv 2002.08546] | ✓ (weighted k-means class centroids) | ✓ | ✗ | F3. `unknown_weight = 1 − ent/ln C` is computed but **never returned or used**; only `predict` is returned [code] | Hard, under no_grad | Yes |
| **SHOT++** | Liang, Hu, Wang, He, Feng — TPAMI 2021 [arXiv 2012.07297] | ✓ | ✓ | ✓ | **Survivor**. Labeling transfer: per class, keep the top-t_k lowest-entropy samples as the "easy" split, then MixMatch | Hard split by per-class entropy rank | Yes |
| NRC | Yang, Wang, van de Weijer, Herranz, Jui — NeurIPS 2021 [arXiv 2110.04202] | ✓ (neighbourhood consistency) | ✓ | ✗ (borderline) | F3. Affinity is structural (reciprocal neighbour 1, otherwise r = 0.1), not confidence or uncertainty | Soft neighbour predictions | Partial (needs a kNN graph over bags) |
| AaD | Yang, Wang, Wang, Jui, van de Weijer — NeurIPS 2022 [arXiv 2205.04183] | ✓ (neighbourhood) | ✓ | ✗ | F3. No weighting | Soft | Partial |
| G-SFDA | Yang, Wang, van de Weijer, Herranz, Jui — ICCV 2021 [arXiv 2108.01614] | ✓ (local structure clustering) | ✓ | ✗ | F3 | Soft | Partial |
| **DaC** | Zhang, Chen, Cheng, Li et al. — NeurIPS 2022 [arXiv 2211.06612] | ✓ (prototype-anchored contrast for source-like samples) | ✓ | ✓ | **Survivor**. Source-like split by max-prob ≥ 0.97 [code] | Hard mask under no_grad | Yes |
| **A2Net** | Xia, Zhao, Ding — ICCV 2021 [CVF] | ✓ (contrastive category-wise matching) | ✓ | ✓ (pair level) | **Survivor**. Pairs are filtered by scheduled confidence bounds | Pair sign γ_ij ∈ {1, 0, −1} | Yes |
| U-SFAN | Roy, Trapp, Pilzer, Kannala et al. — ECCV 2022 [arXiv 2208.07591] | ✗ in released code | — | — | F1. The released code weights only the SHOT-IM entropy, w = exp(−H_Laplace), detached [code]. The paper also combines it with nearest-centroid PLs (§ open-set/extension) — that variant would pass F1 and F3, but it was not read in full | Weight exp(−H), detached | Yes |
| **CoWA-JMDS** | Lee, Jung, Yim, Yoon — ICML 2022 [arXiv 2206.06640] | ✓ (per-class GMM in feature space) | ✓ | ✓ | **Survivor** | Hard GMM label; weight JMDS = LPG·MPPL computed under no_grad; **repo default `--coeff` is `LPG`, not `JMDS`** [code] | Yes |
| C-SFDA | Karim, Mithun, Rajvanshi, Chiu et al. — CVPR 2023 [arXiv 2303.17132] | ✗ | — | — | F1. Reliability selection is not class-conditional, and its contrastive loss is label-independent (would pass F3) | Hard binary reliability | Yes |
| **PLUE** | Litrico, Del Bue, Morerio — CVPR 2023 [arXiv 2303.03770] | ✓ (soft kNN voting over a feature bank) | ✓ | ✓ | **Survivor** | Hard refined PL; w = exp(−H/H_max), under no_grad [code] | Partial (feature bank of bags) |
| Co-learn | Zhang, Shen, Foo — ICCV 2023 [arXiv 2212.07585] | not examined | — | — | Metadata only. Listed as a recent SFDA method; not classified | — | — |
| DIFO | Tang, Su, Ye, Zhu — CVPR 2024 [arXiv 2311.16510] | ✗ (on read) | — | — | F1. Mutual-information prompt customisation with CLIP; no explicit class-conditional alignment found in the grep | — | — |
| ProDe | Tang, Su, Gan, Ye et al. — ICLR 2025 Oral [arXiv 2406.01658] | not established | — | — | Proxy (ViL) denoising plus mutual knowledge distillation; class-conditional structure not established this run | — | — |
| SF(DA)² | Hwang, Lee, Shin, Yoon — ICLR 2024 [arXiv 2403.10834] | ✓ (spectral neighbourhood clustering) | ✓ | ✗ (not found) | F3 per grep; no confidence weighting was located | — | Partial |

**Teacher–student and distillation**

| Method | Authors, venue, year | F1 | F2 | F3 | Stage / why | PL; weight | MIL? |
|---|---|---|---|---|---|---|---|
| Self-ensembling (Mean Teacher for DA) | French, Mackiewicz, Fisher — ICLR 2018 [arXiv 1706.05208] | ✗ | — | — | F1. Consistency plus a class-balance loss (marginal); would pass F3 (threshold 0.968) | Soft consistency masked by teacher max-prob | Yes |
| DINE | Liang, Hu, Feng, He — CVPR 2022 [arXiv 2104.01539] | ✗ | — | — | F1. Adaptive label smoothing (top-k teacher), structural mixup and IM regularisation; no class-conditional alignment | Soft distillation | Yes |
| Knowledge Adaptation (KD-DA) | Ruder, Ghaffari, Breslin — arXiv 2017 [arXiv 1702.02052; no venue] | ✗ | — | — | F1. Trust in teachers via domain similarity; adds high-confidence examples (would pass F3) | Hard high-confidence selection | Yes |
| MT-MTDA | Nguyen-Meidine, Belal, Kiran, Dolz et al. — WACV 2021 [arXiv 2007.07077] | ✗ | — | — | F1. Multi-teacher temperature KD | Soft | Yes |
| DTBS (dual teacher) | Huang, Yao, Zhou — ECAI 2024 [arXiv 2401.01066] | ✗ (not established) | — | — | F1 on read: style and illumination teachers with a reweighting step; no class-conditional alignment found (segmentation) | — | — |
| Dual-teacher multi-target (retinal) | Peng, Lin, Cheng, He et al. — MICCAI-MLMI 2022 [arXiv 2203.03631] | not examined | — | — | Metadata only | — | — |
| DDSR, adaptive dual-teacher distillation (black-box) | Zhang, Li, Xue, Cheng et al. — arXiv 2026, under review [arXiv 2603.22908] | ✓ (class-wise prototypes rectify PLs) | ✓ | borderline ✓ | Borderline survivor. Entropy decides the per-sample fusion weights of the source-model and CLIP teachers; whether the prototype stage filters by confidence was not verified | Entropy-adaptive fusion | Yes |

**Transformers and foundation models**

| Method | Authors, venue, year | F1 | F2 | F3 | Stage / why | PL; weight | MIL? |
|---|---|---|---|---|---|---|---|
| CDTrans | Xu, Chen, Wang, Wang et al. — ICLR 2022 [arXiv 2109.06165; paper header] | ✓ (centroid labels; class-consistent cross-attention pairs) | ✓ | ✗ (borderline) | F3. A pair is kept only if the centroid PL **agrees** with the source label: an agreement filter, not a confidence score | Hard | Partial (pairs of bags) |
| TVT | Yang, Liu, Xu, Huang — WACV 2023 per common citation [arXiv 2108.05988; UNVERIFIED venue] | ✗ | — | — | F1. Transferability-aware attention plus discriminative clustering (entropy/diversity) | — | — |
| SSRT | Sun, Lu, Zhang, Ling — CVPR 2022 [arXiv 2204.07683] | ✗ | — | — | F1. KL to perturbed predictions; diversity-triggered reset | Soft | — |
| PMTrans | Zhu, Bai, Wang — CVPR 2023 Highlight [arXiv 2303.13434] | ✗ | — | — | F1. Patch-mix game and mixup losses | — | — |
| MIC | Hoyer, Dai, Wang, Van Gool — CVPR 2023 [arXiv 2212.01322] | ✗ | — | — | F1. Masked consistency to an EMA teacher; would pass F3 (quality weight = ratio of pixels above τ) | Hard teacher PL with quality weight | — |
| DAPL | Ge, Huang, Xie, Lai et al. — arXiv 2022; TNNLS per common citation [arXiv 2202.06687; UNVERIFIED venue] | ✗ | — | — | F1. Domain-specific prompts; would pass F3 (CLIP PL with threshold τ = 0.6 / 0.5) | Hard, thresholded | — |
| PADCLIP | Lai, Vesdapunt, Zhou, Wu et al. — ICCV 2023 [CVF] | ✗ | — | — | F1. Debiased CLIP PLs, with CFM and DCM momenta; would pass F3 (τ = 0.4) | Hard, thresholded | — |

**Test-time adaptation**

| Method | Authors, venue, year | F1 | F2 | F3 | Stage / why | Weight | MIL? |
|---|---|---|---|---|---|---|---|
| Tent | Wang, Shelhamer, Liu, Olshausen, Darrell — ICLR 2021 Spotlight [arXiv 2006.10726] | ✗ | — | — | F1. Entropy minimisation on BN affine parameters | None | — |
| EATA | Niu, Wu, Zhang, Chen et al. — ICML 2022 [arXiv 2204.02610] | ✗ | — | — | F1. Would pass F3 | 1/exp(H − E0), detached [code] | Yes |
| SAR | Niu, Wu, Zhang, Wen et al. — ICLR 2023 Notable-top-5% [arXiv 2302.12400] | ✗ | — | — | F1. Reliable-entropy filter plus SAM | Hard filter on H | — |
| CoTTA | Wang, Fink, Van Gool, Dai — CVPR 2022 [arXiv 2203.13591] | ✗ | — | — | F1. Weight- and augmentation-averaged teacher plus stochastic restore | — | — |
| RoTTA | Yuan, Xie, Li — CVPR 2023 [arXiv 2303.13899] | ✗ (borderline) | — | — | F1. Category-*balanced* memory with uncertainty-aware replacement is class-balanced sampling, not alignment | — | — |
| DeYO | Lee, Jung, Lee, Park et al. — ICLR 2024 Spotlight [arXiv 2403.07366] | ✗ | — | — | F1. Would pass F3 (entropy and PLPD selection and weighting) | — | — |
| **T3A** | Iwasawa, Matsuo — NeurIPS 2021 Spotlight [NeurIPS] | ✓ (per-class pseudo-prototypes) | ✓ | ✓ | **Survivor**. Keeps the filter_K lowest-entropy supports per class [code] | Hard; entropy filter; no gradient at all | Yes |
| TAST | Jang, Chung, Chung — ICLR 2023 per common citation [arXiv 2207.10792; UNVERIFIED venue] | ✓ (prototypes plus nearest neighbours, built on T3A) | ✓ | not verified | Probable survivor; whether the T3A filter is retained was not verified | — | — |

**Adjacent (not self-training, but nearest in mathematics)**

| CS-DA | Yin, Yu, Lin, Liu et al. — UAI 2024 [arXiv 2405.19978] | ✓ (conditional CS on p(ŷ\|z), Gram matrices) | ✓ | ✗ | F3. Uses the **soft** prediction ŷ instead of discrete PLs, with no weighting | Soft, unweighted | Yes |

Metadata sources: every [arXiv …] row is from the arXiv API this run (`https://export.arxiv.org/api/query?id_list=…`). T3A: [NeurIPS 2021 poster](https://neurips.cc/virtual/2021/poster/27675). A2Net: [CVF ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Xia_Adaptive_Adversarial_Network_for_Source-Free_Domain_Adaptation_ICCV_2021_paper.html). PADCLIP: [CVF ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lai_PADCLIP_Pseudo-labeling_with_Adaptive_Debiasing_in_CLIP_for_Unsupervised_Domain_ICCV_2023_paper.html). MSTN: [PMLR v80](https://proceedings.mlr.press/v80/xie18c.html).

### Inferences
- The funnel is front-loaded. F1 removes about 60% of the field, including every "strong" TTA method (Tent, EATA, SAR, CoTTA, DeYO) and every transformer UDA method except CDTrans. Several of these would pass F3 trivially: EATA, SAR, MIC, DAPL, PADCLIP, AdaMatch, French. **Confidence weighting is common; class-conditional alignment combined with confidence weighting is what is rare.**
- "Borderline" verdicts (CBST/CRST, MCC, RoTTA at F1; NRC and CDTrans at F3) come from the strict reading of the funnel. Under a lenient reading, CBST/CRST and RoTTA would pass F1, because class-wise thresholds and class-balanced memory are class-conditional selection. CDTrans would pass F3 if agreement-filtering counts as reliability. A report writer should state which reading is used.

### Gaps
- No published class-conditional **variant** was found for Tent, EATA, SAR, CoTTA, MIC, BNM or MCC. The variant rule therefore produced no extra rows, apart from U-SFAN's centroid-PL combination, which was not fully read.
- Venues for CST, ProDA, TVT, DAPL and TAST were not confirmed this run, because the arXiv comment was empty or listed code only.
- Co-learn, ProDe, the retinal dual-teacher paper and DTBS were only partially examined.

---

## Q2. Core math of the survivors (and of the near-misses that matter for MIL-CREDA)

### Takeaway
The survivors fall into four mechanisms:
- (a) **Prototype or centroid alignment with a hard confidence gate**: PFAN, CAN, SHOT++, DaC.
- (b) **Soft prototype- or GMM-based reliability weights**: ProDA, CoWA-JMDS.
- (c) **Neighbourhood soft-voting with an entropy weight**: ATDOC-NA, PLUE.
- (d) **Optimisation-free per-class prototypes with an entropy filter**: T3A.

Every weight checked in code is computed under `no_grad` or `.detach()`: CoWA, PLUE, DaC, ProDA, T3A, EATA and U-SFAN this run, plus ATDOC and MCC in the prior note. All survivors align **first-order** class structure (means or prototypes) or kernel means (CAN). **None** align class-conditional **second-order** structure (covariance operators) with a confidence gate. CoWA's per-class GMM uses per-class covariances, but only to score samples, not as an alignment target.

### Cited Findings
- **PFAN** [paper]. Score ψ_k(x_j^t) = CS(G(x_j^t), c_k^S), the cosine similarity to the source prototype, with ŷ = argmax_k ψ_k. Selection is w_j = 1 if ψ ≥ τ, otherwise 0, with the schedule τ = 1/(1 + e^{−µ(m+1)}) − 0.01. APA updates each target prototype by an adaptive EMA, ρ = CS(c_k^t, c_k^T), c_k^T ← ρ² c_k^t + (1 − ρ²) c_k^T, and minimises L_apa = Σ_k d(c_k^S, c_k^T). *Plain language: only target samples that already sit close to a source class prototype get a pseudo-label, the bar is lowered step by step, and the source and target prototypes of each class are pulled together.* Hard labels, hard 0/1 selection — [arXiv 1811.08585](https://arxiv.org/abs/1811.08585).
- **MSTN** [paper]. C_k^S ← θ C_k^S + (1 − θ) C_k^S(t) and C_k^T likewise (moving average, θ = 0.7), with L_SM = Σ_k Φ(C_k^S, C_k^T). The authors "doubt the pseudo labels and propose to align the centroid" instead of weighting them. *Plain language: align the running mean of each class across domains; the noise in the pseudo-labels is averaged out rather than filtered.* — [PMLR](https://proceedings.mlr.press/v80/xie18c.html)
- **CAN** [paper + code]. Target clustering is initialised at the source class centres (spherical k-means). The filter keeps T̃ = {(x, ŷ) : dist(φ(x), O^{t(ŷ)}) < D0}, D0 ∈ [0, 1], and drops classes with fewer than `MIN_SN_PER_CLASS` samples. The code is `mask = min_dist < threshold` in `filter_samples`. CDD = intra-class MMD minus inter-class MMD on a multi-kernel RKHS. *Plain language: cluster the target, throw away points far from their cluster centre, then shrink same-class kernel distances and grow different-class ones.* — [arXiv 1901.00976](https://arxiv.org/abs/1901.00976); [code](https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation)
- **ProDA** [paper + code]. ŷ^{(i,k)} = ξ(ω^{(i,k)} p_{t,0}^{(i,k)}), with ω^{(i,k)} = softmax_k(−(‖f − η^{(k)}‖ − min_k′ ‖f − η^{(k′)}‖)·T). In the code: `weight = F.softmax(-feat_proto_distance * self.opt.proto_temperature, dim=1)`, `rectified = weights * threshold_arg`, normalise, argmax, and `threshold_arg[argmax < train_thred] = 250` (ignore). *Plain language: a fixed source soft prediction is re-weighted by how close the feature is to each class prototype, then turned into a hard label and dropped if still unsure.* Prototype weight inside `torch.no_grad()` — [arXiv 2101.10979](https://arxiv.org/abs/2101.10979); [code](https://github.com/microsoft/ProDA/blob/main/models/adaptation_modelv2.py)
- **CoWA-JMDS** [code]. A GMM in feature space, initialised from softmax responsibilities: π = Σ p, µ_k = Σ p_k f / π_k, with per-class covariances, one EM round. Pseudo-label ŷ = argmax γ. LPG = (z_(1) − z_(2)) / max(z_(1) − z_(2)) on log-posterior zz, and MPPL = p_ŷ (the model's probability of the GMM label). JMDS = LPG · MPPL. The weights mix with mixup: `mixed_c = lam*c_batch + (1-lam)*c_batch[shuffle_idx]`, and the KL loss is weighted by mixed_c. The score is computed inside `with torch.no_grad()`. **The argparse default is `--coeff LPG`, so the released script runs LPG-only unless `--coeff JMDS` is passed.** *Plain language: fit one Gaussian per class to target features; a sample is trusted when the GMM's top-two log-posterior gap is large **and** the network agrees with the GMM label.* — [code](https://github.com/Jhyun17/CoWA-JMDS/blob/main/image_target_CoWA.py); [arXiv 2206.06640](https://arxiv.org/abs/2206.06640)
- **SHOT++** [paper]. Per class k, the indices with entropy among the top-t_k smallest form the easy split, with the rest hard. MixMatch (α = 0.75) then propagates labels from easy to hard. *Plain language: within each class, the most certain samples act as a labelled set for semi-supervised learning on the rest.* — [arXiv 2012.07297](https://arxiv.org/abs/2012.07297)
- **SHOT** [code]. Centroids: initc = affᵀ·fea / Σ aff, refined by cdist argmin; classes with count ≤ threshold are dropped. `unknown_weight = 1 - ent / np.log(args.class_num)` (line 265) is computed and not used; the function returns `predict.astype('int')` — [code](https://raw.githubusercontent.com/tim-learn/SHOT/master/object/image_target.py)
- **DaC** [code]. `max_prob, tmp_label = torch.max(p_l, dim=-1); mask = max_prob.ge(args.p_threshold)` with default p_threshold 0.97, under `torch.no_grad()`. Adaptation metric `EMMD` (default) or `LMMD`. *Plain language: very confident samples are treated as source-like and anchor class prototypes; the rest are pulled toward them with contrast plus a memory-bank MMD.* — [code](https://github.com/ZyeZhang/DaC/blob/main/VisDA/target.py); [arXiv 2211.06612](https://arxiv.org/abs/2211.06612)
- **A2Net CCM** [paper]. s_ij = I_iᵀ I_j with I_i = σ(p_i^s). The sign is γ_ij = 1 if s_ij > µ(t), −1 if s_ij < l(t), otherwise 0, with µ(t) = µ0 − λ_µ t and l(t) = l0 + λ_l t; a contrastive loss is applied on the positive pairs. *Plain language: two images are treated as same-class only when their predicted class vectors agree strongly, and the "strongly" bar relaxes over epochs.* — [CVF PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Xia_Adaptive_Adversarial_Network_for_Source-Free_Domain_Adaptation_ICCV_2021_paper.pdf)
- **PLUE** [code]. The refined PL comes from `soft_k_nearest_neighbors` (under `@torch.no_grad()`). The weight is w = exp(−H₂(p)/log₂C), with H₂ here the base-2 *Shannon* entropy, computed under `torch.no_grad()`. The loss is `(w * CE(logits_q, pseudo_labels_w)).mean()` (or with the negative-learning criterion). *Plain language: a neighbour vote gives the label, and its normalised entropy shrinks the loss of uncertain samples exponentially.* — [code](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/main/train_target.py); [arXiv 2303.03770](https://arxiv.org/abs/2303.03770)
- **T3A** [code + paper]. Supports S_k ← S_k ∪ {f(x)/‖f(x)‖} for ŷ = k. Each class keeps the filter_K lowest-entropy supports (`torch.sort(ent_s[y_hat == i])`). Prototypes = suppᵀ·onehot, and prediction = z·normalize(prototypes). No gradient. *Plain language: replace the last layer's class templates with the average of the most confident test features of each class.* — [code](https://github.com/matsuolab/T3A/blob/master/domainbed/adapt_algorithms.py); [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf)
- **ATDOC-NA**: see the prior note (`weighting_schemes.md`). w = max_c (1/K Σ_nbrs m_c), loss Σ w ℓ / Σ w, detached.
- **CDTrans** [paper]. c_k = Σ_t δ_t^k f_t / Σ_t δ_t^k, y_t = argmin_k d(c_k, f_t), one refinement round. "If the pseudo label of the target image is consistent with the label of the source image, this pair would be kept … otherwise … discarded as a noise". A distillation loss runs from the source–target branch (soft) to the target branch — [arXiv 2109.06165](https://arxiv.org/abs/2109.06165)
- **CS-DA** [paper]. It uses D_CS(p^s(z); p^t(z)) plus D_CCS(p^s(ŷ|z); p^t(ŷ|z)), estimated from Gram matrices K (features) and L (predictions), including cross-domain K^{st}, L^{st}. "Instead of the discrete pseudo label … we use the prediction vector … ŷ = g(z) as the target label." — [arXiv 2405.19978](https://arxiv.org/abs/2405.19978)
- **Near-miss weight formulas (fail F1 but are useful comparators).**
  - EATA [code]: `coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))` after the filter `entropys < e_margin` and the cosine redundancy filter — [code](https://github.com/mr-eggplant/EATA/blob/main/eata.py).
  - U-SFAN [code + paper]: `unknown_weight = torch.exp(-la_ent).detach()`, with the paper's "w_i = exp(−H) where H denotes the entropy of the predictive mean" under a Laplace posterior — [code](https://github.com/roysubhankar/uncertainty-sfda/blob/main/cifar/main_cda.py).
  - C-SFDA [paper]: r_i = 1 if conf ≥ τ_c and g_u ≤ τ_u, with τ_c and τ_u the batch means and g_u the std of confidence over augmentations.
  - French [paper]: mask the consistency loss if the teacher max-prob is below 0.968.
  - DAPL [paper]: Σ 1{P(ŷ|x) ≥ τ} log P(ŷ|x).
  - PADCLIP [paper]: L_cp = (1/B) Σ 1[max q_i ≥ τ]·H(p(y|s(x)), q_i), τ = 0.4.
  - MIC [paper]: quality weight = ratio of pixels whose max prob exceeds τ.
- **From memory, not re-read this run** [memory]:
  - Tent: min_{γ,β} −Σ p log p.
  - SAR: E0 = 0.4·ln C as the reliable-entropy threshold (the paper text read this run only states that the entropy lies in (0, ln C)).
  - CBST: min over w and ŷ of the CE on pseudo-labels with class-wise thresholds k_c, chosen so that a portion p of each class is kept.
  - CRST-MRKLD: adds α·KL(u ‖ p) (uniform-to-prediction).
  - AdaMatch: relative threshold τ·mean(max p_source).

### Inferences
- **Hard vs soft.** Every survivor uses **hard** pseudo-labels for routing. Soft information enters only through the weight (ProDA's ω, CoWA's JMDS, PLUE's w) or through soft prototype construction (SHOT/CDTrans centroids δ-weighted). MIL-CREDA matches this pattern: a hard per-class route, a soft confidence weight, and a soft prediction used only for the local reference.
- **Weight shape.** In MIL-CREDA's notation (C = number of classes):
  - PLUE: exp(−H₁/ln C); never zero, at least e^{−1} ≈ 0.37.
  - U-SFAN: exp(−H); unnormalised.
  - EATA: exp(E0 − H); above 1 for confident samples.
  - SHOT/MIL-CREDA family: 1 − H/ln C; zero at uniform.
  - CoWA: a product of a geometric margin and a model probability.
  
  MIL-CREDA's 1 − H2/ln C is the only Rényi-2 form found. It is zero at uniform, so uncertain bags drop out completely. PLUE's exponential form keeps a floor.
- **Detach.** The precedent is uniform: all survivors detach. This supports MIL-CREDA detaching w; see the degeneracy derivation in the prior note.
- **Order of statistics.** PFAN, MSTN, SHOT, DaC, T3A and CDTrans align or use class **means**. CAN aligns class-conditional kernel **mean embeddings** (MMD = distance of RKHS means). CoWA estimates per-class **covariances** but only for scoring. CS-DA matches conditional distributions through Gram matrices, but without per-class routing or weighting.
  
  A per-class second-order (covariance-operator / matrix-Rényi-2) alignment score, *gated* by confidence mass, therefore appears unoccupied among the methods screened. This is an inference from absence in a bounded search, not a proof of novelty.

### Gaps
- The CRST and CBST objectives, the SAR threshold and the AdaMatch threshold were not re-read (tagged [memory]).
- The DaC EMMD formula was not read; whether it is class-conditional or between the source-like and target-specific sets was not established.
- The ProDA paper's exact Eq. 4 was only partially extracted (the PDF text was garbled); the code was used instead.
- TAST's exact filter rule was not read.

---

## Q3. Which survivors are closest to MIL-CREDA, and on which axis

### Takeaway
No single method matches MIL-CREDA on all axes. The closest are:
- **CAN**, for kernel/RKHS class-conditional alignment with a confidence filter;
- **CoWA-JMDS**, for per-class second-order (GMM covariance) modelling, a confidence weight, and a frozen source model;
- **PLUE** and **ATDOC-NA**, for detached entropy-type weights in a normalised weighted loss;
- **CS-DA**, for a Cauchy–Schwarz divergence on Gram matrices, but with no weighting and no per-class routing;
- **ProDA**, for using the full soft prediction together with prototype distances, which parallels MIL-CREDA's personalised local reference.

None of the methods screened operates on MIL bags.

### Cited Findings
- CAN's CDD is intra-class minus inter-class multi-kernel MMD over clustered, filtered target samples — [arXiv 1901.00976](https://arxiv.org/abs/1901.00976)
- CoWA models each class with a Gaussian that has its own covariance (`Covi`, `logdet`, Mahalanobis `mah_dist` in `gmm`) and uses only the source model, no source data — [code](https://github.com/Jhyun17/CoWA-JMDS/blob/main/image_target_CoWA.py)
- CS-DA estimates CS and conditional CS from Gram matrices of features and predictions, using soft predictions instead of discrete pseudo-labels — [arXiv 2405.19978](https://arxiv.org/abs/2405.19978)
- PLUE: w = exp(−H/H_max) multiplies the CE, under no_grad — [code](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/main/train_target.py)
- ProDA: soft prediction × prototype-distance softmax → rectified label — [code](https://github.com/microsoft/ProDA/blob/main/models/adaptation_modelv2.py)
- T3A: per-class prototypes from entropy-filtered supports, with no gradient — [code](https://github.com/matsuolab/T3A/blob/master/domainbed/adapt_algorithms.py)

### Inferences
Position of MIL-CREDA, axis by axis, against the nearest survivor:

| MIL-CREDA component | Nearest published analogue | Difference |
|---|---|---|
| Ψ(B), attention-weighted RKHS mean embedding of bags | CAN (RKHS kernel means), but at instance level | No screened method handles bags |
| Frozen source encoder + trainable target encoder | CoWA, SHOT, NRC/AaD (frozen source *classifier*/hypothesis); A2Net (frozen C_s, learnable C_t) | SFDA methods freeze the classifier, not a whole second encoder; MIL-CREDA's two-encoder split is closer to A2Net's dual classifier |
| Hard per-class routing | PFAN, CAN, SHOT, CDTrans, T3A | Same pattern |
| w = 1 − H2/ln C (Rényi-2) | SHOT's unused 1 − H1/ln C; PLUE exp(−H1/ln C); U-SFAN exp(−H); EATA exp(E0 − H) | Rényi-2 form not found; zero-at-uniform like SHOT's; more lenient than Shannon (prior note) |
| Per-class matrix-Rényi-2 / CS on covariance operators | CS-DA (CS on Gram matrices, not per class, unweighted); CoWA (per-class covariance, scoring only); CAN (per-class first-order kernel means) | **Second-order, per-class, gated alignment not found** |
| Confidence gate per class (Σ w) | CAN's drop of classes with too few samples; SHOT's class-count threshold | Those are hard count gates; MIL-CREDA gates by confidence *mass* |
| Local term: RKHS distance to a personalised source reference from the full soft prediction | ProDA (soft prediction × prototype weights); CDTrans (δ-weighted centroids); ATDOC (neighbour soft memory) | No screened method builds a per-sample *source* reference as a soft mixture of source class embeddings and measures an RKHS distance to it |

- **Likely reviewer challenge.** CoWA-JMDS and CAN will be cited as the "already does class-conditional plus confidence" prior art. The defensible difference is (i) second-order versus first-order or scoring-only, (ii) bags, and (iii) a soft per-sample local reference. Confidence weighting alone is **not** a contribution: 10 survivors already do it.
- **Practical precedent for the gate.** A zero-at-uniform weight makes early training switch the gated terms off; PFAN, A2Net and CBST all use schedules that relax a threshold over time. MIL-CREDA's fixed-scale w has no schedule. This repeats the prior note's SoftMatch remark from the survivors' side.

### Gaps
- No MIL-specific DA paper was screened in this run; that belongs to another researcher's scope. The "extends to bags" column is therefore my inference: a method extends when its weight needs only a bag-level posterior, and partially when it needs augmentations or a kNN graph over instances.
- Novelty claims are bounded by this search (about 50 methods, one extended web query for 2023–2026). A targeted search for "matrix Rényi" or "covariance operator" combined with "domain adaptation" and "pseudo-label" was not run here.

---

## Q4. 2023–2026 methods with confidence-weighted class-conditional alignment

### Takeaway
The verified recent methods that pass all three filters are PLUE (CVPR 2023), with DaC (NeurIPS 2022) and CoWA (ICML 2022) just before the window, plus the borderline 2026 dual-teacher DDSR. Most strong 2023–2026 methods are CLIP/ViL-driven (DIFO, ProDe, PADCLIP, DAPL) or TTA (SAR, RoTTA, DeYO). They fail F1 because they denoise or select pseudo-labels without explicit class-conditional alignment.

### Cited Findings
- PLUE: CVPR 2023; uncertainty (entropy) from neighbour soft-voting weights the PL loss — [arXiv 2303.03770](https://arxiv.org/abs/2303.03770)
- C-SFDA: CVPR 2023; binary reliability from confidence and augmentation std, with curriculum — [arXiv 2303.17132](https://arxiv.org/abs/2303.17132)
- DIFO: CVPR 2024; frozen ViL customised by mutual information — [arXiv 2311.16510](https://arxiv.org/abs/2311.16510)
- ProDe: ICLR 2025 Oral; proxy denoising of ViL predictions plus mutual knowledge distillation — [arXiv 2406.01658](https://arxiv.org/abs/2406.01658)
- SF(DA)²: ICLR 2024; spectral neighbourhood clustering plus implicit feature augmentation — [arXiv 2403.10834](https://arxiv.org/abs/2403.10834)
- DDSR (adaptive dual-teacher, 2026, under review): class-wise prototypes rectify pseudo-labels, and entropy adaptively weights the source-model and CLIP teachers — [arXiv 2603.22908](https://arxiv.org/abs/2603.22908)
- An extended web search surfaced further 2024–2025 candidates (IJCAI 2025 class-imbalanced graph DA with prototype-anchored weighted contrastive alignment; ICLR 2025 PASLE with candidate label sets; uncertainty-guided class-prototype alignment). Only their search snippets were seen, so they are unclassified — [IJCAI 2025 PDF](https://www.ijcai.org/proceedings/2025/0356.pdf); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/ac662d74829e4407ce1d126477f4a03a-Paper-Conference.pdf)
- The prior note flags a 2025 MDPI *Mathematics* paper, "Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting". It is the only Rényi-plus-conditional-plus-weighting hit; its text remained unreadable (HTTP 403) — [DOI 10.3390/math13162602](https://doi.org/10.3390/math13162602)

### Inferences
- The 2023–2026 trend moves confidence estimation from the classifier to external priors (CLIP, augmentation variance, Laplace posteriors). Class-conditional *alignment* has become less central. That leaves second-order class-conditional alignment comparatively unexplored in the recent SFDA and self-training literature screened here.
- The MDPI α-Rényi paper is the single most important unread item for MIL-CREDA's novelty. It should be read before any "first Rényi-weighted conditional alignment" claim is made.

### Gaps
- The IJCAI 2025, ICLR 2025 PASLE and MDPI 2025 papers were not read. Their funnel stage is unknown.
- No 2026 venue-published (non-preprint) method in this family was verified this run.
