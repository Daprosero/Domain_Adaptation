# Novelty check for MIL-CREDA against the CREDA lineage, information-theoretic conditional DA, and confidence-weighted class-conditional alignment in bags

Run date: 2026-10-10. Tags: [verified] = read in this run from the primary source; [abstract only] = only the abstract or metadata was read in this run; [derived] = algebra done here (numerically checked where stated); [UNVERIFIED] = seen only in a search snippet or a prior note, not re-read in this run.

## 1. CREDA in full: loss, α, encoders, pseudo-label weighting, datasets, results

### Takeaway
The full CREDA PDF was read (MDPI's HTML returns 403, but the `mdpi-res.com` PDF downloaded). CREDA is a single shared encoder with hard argmax routing and the weight w = 1 − H₂(g)/log C, which is the same formula MIL-CREDA uses. Its score is Ĩ₂ = ½(H₂(Kₛ)+H₂(K̃ₜ)) − H₂(K_mix), the same functional form as MIL-CREDA's J_c, but its mixture block is not balanced and its cross block is unweighted. α is fixed at 2. It has no normalization, no confidence gate, no local term and no bags. The experiments use one seed and compare only against DANN, ADDA and CDAN+E. The Office-31 numbers come from a 10-class subset.

### Cited Findings
- **Source access.** `https://www.mdpi.com/2227-7390/13/16/2602` (and `/pdf`, `/htm`) return HTTP 403. The PDF at `https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf` returned HTTP 200 (12.5 MB, 29 pp.). Page 8 was rendered to an image to read Eqs. (18)–(23) without extraction errors [verified] — [CREDA PDF](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf); [DOI](https://doi.org/10.3390/math13162602)
- **Bibliographic data.** Pérez-Rosero, Álvarez-Meza, Castellanos-Dominguez (Signal Processing and Recognition Group, Universidad Nacional de Colombia, Manizales). *Mathematics* 2025, 13, 2602. Received 10 Jul 2025, accepted 12 Aug 2025, published 14 Aug 2025, CC BY [verified] — [CREDA PDF](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf)
- **Entropy estimator (Eq. 9).** A = K/tr(K), H_α(A) = (1/(1−α)) log tr(A^α). The paper uses α = 2 in the Frobenius form H₂ = −log tr(ĂᵀĂ) [verified] — same PDF, §2.2, p. 6
- **Architecture: one encoder.** "a deep feature extractor F : X → R^d" with a classifier G : R^d → [0,1]^C (Eqs. 13–14). Source and target go through the same F (Eqs. 15–17 both use F). There is no second or frozen encoder [verified] — same PDF, §2.3, p. 7
- **Routing: hard.** Kₛ_c uses source samples with y_{i,c} = 1. Kₜ_c and Kˢᵗ_c use target samples with argmax_{c′} gᵗ_{j,c′} = c (Eqs. 15–17). The kernel is Gaussian κ_σ [verified] — same PDF, p. 7
- **Confidence weight (Eqs. 18–19).** Ĥ₂(gⱼ) = −log Σ_c g²_{j,c}; Û = Ĥ₂/log C ∈ [0,1]; wⱼ = 1 − Û(gⱼ). This is the same weight as MIL-CREDA's w = 1 − H₂(g)/ln C [verified] — same PDF, p. 8
- **How the weight enters (Eq. 20).** W̃ₜ_c = w̃ₜ_c (w̃ₜ_c)ᵀ and K̃ₜ_c = Kₜ_c ⊙ W̃ₜ_c, so only the target–target block is weighted [verified] — same PDF, p. 8
- **Score (Eq. 21, read from the rendered page).** Ĩ₂(Kₛ_c; K̃ₜ_c) = ½(H₂(Kₛ_c) + H₂(K̃ₜ_c)) − H₂(K_mix_c). The ½ multiplies only the two marginal entropies [verified] — same PDF, p. 8
- **Mixture block (Eq. 22).** K_mix_c = [[Kₛ_c, Kˢᵗ_c], [(Kˢᵗ_c)ᵀ, K̃ₜ_c]]. The cross block Kˢᵗ_c is raw (unweighted). The blocks are not rescaled by cardinality. The paper motivates the block form only as "enables the computation … even when nₜ_c ≠ nₛ_c" [verified] — same PDF, p. 8
- **Total loss (Eq. 23, as printed).** L_CREDA = Σᵢ Σ_c yˢ_{i,c} log G(F(xᵢ)) − λ Σ_c Ĩ₂(Kₛ_c; K̃ₜ_c), with λ ∈ R⁺ [verified] — same PDF, p. 8. As printed, the CE term has no leading minus. If L is minimized, that would minimize log-likelihood, so the printed sign is internally inconsistent. The repository's implementation trains `loss = loss_class + lambda_val * loss_creda` with `creda_c = h_mix - 0.5*(h_s+h_t)` = −Ĩ₂, i.e., it minimizes CE − λĨ₂ (it maximizes Ĩ₂) [verified in local repo `src/CREDA/training_pipeline.py:544`, `src/CREDA/models.py:872`, commit 72f8a17, whose docstring states its defaults are bit-identical to the published behavior] — [repo](https://github.com/Daprosero/Domain_Adaptation)
- **Mini-batch semantics.** All kernels are computed per mini-batch. A class absent from the batch contributes zero. That is the only gating: there is no q_c or minimum count [verified] — same PDF, pp. 8–9
- **Why α = 2.** The paper ties it to the information potential: under a Gaussian kernel the objective "becomes a sum of pairwise similarities" and promotes "tight, pure clusters". It also claims sensitivity to higher-order statistics and consistency inherited from KDE. No α ablation is reported [verified] — same PDF, p. 9
- **Bandwidth.** Median heuristic per batch over the combined source+target features, σ = sqrt(median‖fˢᵢ − fᵗⱼ‖²) + ε (Eq. 31) [verified] — same PDF, p. 17
- **Schedules.** η(p̆) = η₀(1+αp̆)^(−β) and λ(p̆) = (1−e^(−δp̆))/(1+e^(−δp̆)), with α = 20, β = 0.75, δ = 20 (Eq. 29). This "α" is a learning-rate schedule constant, not the Rényi order. Adam is used. Batch size = min(max(16, ⌊0.1N⌋), 64) [verified] — same PDF, pp. 15–16
- **Datasets.** Digits: MNIST, USPS, SVHN, 6 tasks. ImageCLEF-DA: C/I/P, 12 classes, 600 images per domain. Office-31: "we selected a subset of ten shared classes". Each domain is split 70/15/15 with stratification [verified] — same PDF, pp. 10–11, 16. The repository's results files are named `Office-Caltech.csv`, consistent with the 10-class subset [verified, repo tree via GitHub API] — [repo](https://github.com/Daprosero/Domain_Adaptation)
- **Backbones and baselines.** ResNet-18, ResNet-50 and ViT-Tiny (ImageNet-pretrained). The comparators are Source-only, DANN, ADDA and CDAN+E only. There is no MMD-family or class-conditional kernel baseline (no DSAN, CAN, JAN) [verified] — same PDF, §3.2
- **Average accuracy (%), CREDA vs. best other method** [verified, Tables 6–8]:
  - Digits: R18 62.65 vs DANN 59.71; R50 64.07 vs DANN 63.59; ViT-T 47.23 vs Baseline 51.28 (Baseline best).
  - ImageCLEF-DA: R18 73.70 vs ADDA 72.04; R50 81.67 vs DANN 83.15 (DANN best); ViT-T 82.41 vs Baseline 80.19.
  - Office-31 (10 classes): R18 85.97 vs DANN 77.34; R50 92.96 vs DANN/ADDA 91.57; ViT-T 89.31 vs Baseline 85.46.
  - Mean Friedman rank 1.2 ± 0.4 (χ²(4) = 23.21).
  — same PDF, pp. 18–20
- **Statistical protocol.** The "±" values are standard deviations of batch-wise test accuracies ("a proxy for model stability"). Training uses a single global seed of 42. There are no repeated seeds [verified] — same PDF, pp. 16–17
- **Code location.** "https://github.com/Daprosero/Domain_Adaptation" [verified] — same PDF, p. 17 and the Data Availability statement

### Inferences
- MIL-CREDA's confidence weight w, hard argmax routing and the functional form of J_c are inherited from CREDA. None of the three can be claimed as new. The new parts relative to CREDA are elsewhere: bags, a balanced mixture, a weighted cross block if MIL-CREDA adds one, the normalization ℓ_c, the gate q_c, the local term and two encoders.
- **Why the balance matters mathematically [derived].** With trace-normalized blocks given masses ½–½, write V_xx = ‖A_x‖²_F (purities), V_st for the cross term, and D_CS = −log(V_st/√(V_ss V_tt)). Then:

  J = ½(H_s + H_t) − H_mix = log( ½ cosh((H_t − H_s)/2) + ½ e^(−D_CS) )

  Numerically checked on random Gaussian-kernel data: −0.421372 from both sides. At D_CS = 0 this gives U_c = 2 ln cosh((H_t − H_s)/4). As D_CS → ∞ it tends to log(½ cosh(·)) ≥ −ln 2 = L_c. So ℓ_c = 0 exactly when D_CS = 0, with "r" = cosh((H_t − H_s)/2).
- CREDA's unnormalized concatenation instead gives masses proportional to the block traces, i.e., nₛ_c versus Σw² for a unit-diagonal kernel. Its Ĩ₂ therefore has no such closed form: it mixes D_CS with a cardinality- and confidence-dependent mass ratio. This is a real, citable difference, but a technical one.
- CREDA calls Ĩ₂ a "mutual information estimator". Kₛ and Kₜ are Gram matrices over disjoint sample sets, not paired views, so Ĩ₂ is a Jensen–Rényi-type gap between two Gram matrices and their mixture, not MI in the sense of its own Eq. (11). MIL-CREDA should not inherit the MI label.
- The baselines are weak: no DSAN, CAN or CS-DA, one seed, and a 10-class Office-31. CREDA's numbers therefore establish little against the class-conditional kernel family that MIL-CREDA is actually competing with.

### Gaps
- Whether the published CREDA code detaches w from the gradient graph was not checked (the GitHub `CREDA/` folder holds notebooks and weights, and the local `src/CREDA` has since been extended). The paper does not say.
- No ResearchGate, Google Scholar cache or arXiv version was needed once the PDF downloaded. None of them was checked for a different version.

## 2. Works citing CREDA, and the same authors' Rényi/kernel DA work 2023–2026

### Takeaway
CREDA has one indexed citing paper, which cites it only in passing, plus one unpublished downstream code project, DARES. Neither moves CREDA into bags. The authors' other 2023–2026 output uses α-Rényi or kernels for EEG and connectivity, not domain adaptation. One 2026 paper of theirs is MIL but has no DA in its title or abstract metadata.

### Cited Findings
- **Citation counts.** Semantic Scholar: citationCount = 1, the citer being Dardeno, Hughes, Bull, Mills, Dervilis, Worden, "Transfer learning via interpolating structures", arXiv 2603.22621 (Mar 2026). OpenAlex: cited_by_count = 0 [verified] — [S2 API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/math13162602); [OpenAlex](https://api.openalex.org/works/doi:10.3390/math13162602)
- **How Dardeno et al. cite it.** Ref. [34] is grouped with two others as "alternative objectives … (e.g., covariance-based visual alignment, distributed balancing for multi-site healthcare data, and entropy-based alignment strategies) [32–34]". The paper is about structural-health-monitoring transfer via intermediate structures and does not build on CREDA [verified, full text] — [arXiv 2603.22621](https://arxiv.org/pdf/2603.22621)
- **DARES (code project, not a paper).** GitHub repo `liturriago/DARES`, created 2026-08-12, last push 2026-10-07. It "extends the Conditional Rényi α-Entropy Domain Adaptation (CREDA) framework to dense prediction" for deforestation segmentation (Sentinel-2 → Landsat-8). Its loss is L = L_seg + λ_eff(L_align + βL_ac + γL_rep) + λ_em L_em, with "hardened class-conditional order-2 Rényi alignment", "per-pixel Rényi-2 confidence weighting", "anti-collapse spectral entropy floors (H2 ≥ η)", "margin-hinged inter-class target repulsion", a GradNorm-lite trust region, "dense Rényi-EM", and a "Spatially-Stratified, Confidence-Guided Sampling operator (Φ_c)" [verified, README] — [DARES repo](https://github.com/liturriago/DARES)
- **Authors' publication list.** Pérez-Rosero's Semantic Scholar and OpenAlex lists (8 works) contain only CREDA as DA work. They also include "Text-Conditioned Semantic Aggregation for Audio-Based Multiple-Instance Learning in Depression Detection" (IWINAC 2026, DOI 10.1007/978-3-032-27314-7_28) and "Notes on Kernel Methods in Machine Learning" (arXiv 2511.14485) [verified, metadata] — [S2 author](https://api.semanticscholar.org/graph/v1/author/search?query=Diego%20Armando%20P%C3%A9rez-Rosero); [OpenAlex author A5099166547](https://api.openalex.org/works?filter=author.id:A5099166547)
- **Álvarez-Meza 2023–2026, Rényi/kernel-related titles** (Semantic Scholar author 1390049809): T-GARNet "… Gaussian Kernel Connectivity Network with Alpha-Rényi Regularization for EEG-Based ADHD Detection" (*Mathematics* 2025, 10.3390/math13244026); Takens-based kernel transfer entropy (*Sensors* 2025); KCS-FCnet (*Diagnostics* 2023); kernel-regularized EEGNet with centered alignment (*Computers* 2023); "Chained Deep Learning Using Generalized Cross-Entropy for Multiple Annotators" (*Sensors* 2023). None is a domain-adaptation paper by title [verified, metadata] — [S2 author papers](https://api.semanticscholar.org/graph/v1/author/1390049809/papers?fields=title,year,externalIds,venue&limit=200)

### Inferences
- DARES is the closest relative in the lineage and passes all three funnel filters (§4). It adds an inter-class repulsion term and anti-collapse floors that MIL-CREDA lacks. It is instance- or pixel-level, not bag-level. It is a code repository with no paper found, so it should be cited as software with an access date.
- The 2026 IWINAC audio MIL paper shows the group already works on MIL. Whether it involves any domain shift was not checked.

### Gaps
- Google Scholar "cited by" was not queried (no scraping access in this run). Semantic Scholar and OpenAlex may miss MDPI citers in Spanish-language or regional venues.
- The abstracts of the IWINAC 2026 MIL paper and of T-GARNet were not read [abstract not read].
- No published "DARES 2026" paper was found by web search. If one exists, it was not indexed under that name on 2026-10-10.

## 3. Dedicated novelty search for the combinations

### Takeaway
No work was found that combines bag-level (MIL) domain adaptation with Rényi or Cauchy–Schwarz divergences, or with confidence-weighted class-conditional alignment on bag embeddings with unlabeled target bags. MIL+DA exists but uses adversarial training, optimal transport, contrastive objectives, graph coupling, or weak target labels. Matrix-Rényi DA exists only at instance level (CREDA, DARES). As before, this is absence of evidence after targeted web and API searches, not a proof of absence.

### Cited Findings
- **"matrix-based Rényi entropy" + "domain adaptation" + class-conditional.** The only hits were CREDA and its mirrors (ResearchGate 394905031, RePEc), plus Rényi-uncertainty works that do no alignment (e.g., arXiv 2511.04219, active DA for mmWave HAR) [abstract only] — [ResearchGate](https://www.researchgate.net/publication/394905031_Conditional_Domain_Adaptation_with_a-Renyi_Entropy_Regularization_and_Noise-Aware_Label_Weighting); [arXiv 2511.04219](https://arxiv.org/abs/2511.04219)
- **"Rényi" + "multiple instance" + DA.** No bag-level Rényi DA. The hits were Takahama et al. (ISBI 2023, arXiv 2304.03537), Blast (below) and CREDA [abstract only] — [arXiv 2304.03537](https://arxiv.org/abs/2304.03537)
- **"Cauchy–Schwarz" + "bag".** No DA hit. Møllersen et al.'s bag-to-class divergence is MIL classification with divergences between bag distributions, not adaptation (arXiv 1803.02782). The generalized CS divergence (GCSD, arXiv 2405.04061) targets multiple distributions (clustering, multi-source DA), not bags [abstract only] — [arXiv 1803.02782](https://ar5iv.labs.arxiv.org/html/1803.02782); [arXiv 2405.04061](https://arxiv.org/html/2405.04061v3)
- **Blast.** Prabono, Yahya, Lee, "Multiple-instance domain adaptation for cost-effective sensor-based human activity recognition", *Future Generation Computer Systems* 2022, DOI 10.1016/j.future.2022.03.006. The search snippet describes "a bag-level adaptation strategy (Blast) … to perform domain adaptation in bag-level representation". The alignment measure and whether pseudo-labels are used could not be read (ScienceDirect 403, no OpenAlex abstract) [metadata verified; method UNVERIFIED] — [OpenAlex](https://api.openalex.org/works?search=Multiple-instance%20domain%20adaptation%20cost-effective%20sensor-based%20human%20activity%20recognition); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X22000814)
- **TAKT** (MICCAI 2024, arXiv 2303.05780). Teacher–student WSI transfer. A "target-aware feature alignment module" solves optimal transport between source and target features, and teacher bag features supervise the student on the target. Its target task differs from the source task [abstract only] — [arXiv 2303.05780](https://arxiv.org/pdf/2303.05780)
- **DETA** (arXiv 2411.14001). WSI survival analysis as graph DA. Category-level alignment uses "a coupling technique by virtue of the dual-branch structure", and feature-level alignment uses adversarial perturbation. No Rényi or kernel measure is mentioned [abstract only] — [arXiv 2411.14001](https://arxiv.org/abs/2411.14001)
- **Supervised-contrastive WSI DA** (CASEIB 2024, arXiv 2412.04260). Two centers, six skin-cancer subtypes. Adds a DA constraint to SupCon [abstract only] — [arXiv 2412.04260](https://arxiv.org/pdf/2412.04260v1)
- **"stain-invariant MIL".** ULSA (MICCAI 2024) is patch-level stain translation plus consistency learning, not MIL alignment [abstract verified] — [MICCAI 2024 paper 820](https://papers.miccai.org/miccai-2024/820-Paper2012.html)
- **"subject-level … EEG bag".** BridgeMIL (arXiv 2607.27274) treats subjects as bags with two-stage pretraining but performs no domain adaptation. The cross-subject EEG DA hits are all instance-level (multi-source, adversarial, graph) [abstract only] — [arXiv 2607.27274](https://arxiv.org/html/2607.27274v1); [PMC11560783](https://pmc.ncbi.nlm.nih.gov/articles/PMC11560783/)
- **"distribution regression" / "set-level DA" / "kernel mean embedding" + MIL + DA.** Hits were distribution-regression and kernel-embedding works without adaptation (e.g., arXiv 1904.10583, KME of instance-wise predictions in multiple-instance regression) and classical kernel DA under target/conditional shift (Zhang et al., ICML 2013). None aligns bag-embedding distributions per class across domains [abstract only] — [arXiv 1904.10583](https://arxiv.org/abs/1904.10583); [PMLR v28](http://proceedings.mlr.press/v28/zhang13d.pdf)

### Inferences
- The unoccupied cell is narrower than "MIL + DA": per-class alignment of distributions of attention-weighted RKHS bag embeddings, routed by confidence-weighted pseudo-labels on unlabeled target bags, under a Rényi/CS-type measure. Each axis separately is occupied: bag-level DA by Blast, TAKT and DETA; matrix-Rényi conditional DA by CREDA and DARES; confidence-weighted class-conditional kernel DA by CREDA, DSAN and CAN.
- Blast is the one MIL-DA paper whose mechanism could not be read and whose title ("bag-level adaptation") overlaps most. It must be read before the novelty claim is written.

### Gaps
- No Google Scholar searches were run. Web search here is a general engine with US-only results.
- "weakly supervised domain adaptation bags" and "multi-instance transfer learning" were covered only indirectly (Takahama; a hit for "domain transfer multi-instance dictionary learning", Neural Comput. Appl. 2016, not read).
- The Blast full text was inaccessible.

## 4. Information-theoretic conditional DA through the funnel (F1 conditional? F2 no target labels? F3 pseudo-label confidence weighting?)

### Takeaway
Cauchy–Schwarz DA is conditional on the posterior p(ŷ|z), not class-conditional p(z|y), and has no per-sample confidence weights, so it fails F3. RJSD's DA instantiation is a JAN-style joint-layer alignment with no per-class routing or weights, so it fails F1 (strictly) and F3. Of the information-theoretic family, only CREDA and DARES pass all three filters. Of the MMD family, DSAN (LMMD) and CAN (CDD) pass, with F3 satisfied by soft weights (DSAN) or by filtering (CAN).

### Cited Findings
- **CS-DA** — Yin, Yu, Lin, Liu, Sonke, Gavves, "Domain Adaptation with Cauchy–Schwarz Divergence", arXiv 2405.19978, "Accepted by UAI-24" [verified, arXiv metadata and full text in scratch] — [arXiv 2405.19978](https://arxiv.org/html/2405.19978v1)
  - Divergence: D_CS(p;q) = −log[(∫pq)² / (∫p² ∫q²)].
  - Conditional CS: D_CCS(p(y|x); q(y|x)) = −2 log ∬ p(x,y)q(x,y)/(p(x)q(x)) + log ∬ p²(x,y)/p²(x) + log ∬ q²(x,y)/q²(x). It is estimated from a feature Gram K and a Gram L over predicted outputs ŷ = g(z) (App. B, Prop. 6).
  - Objective: min_{f,g₁,g₂} L_cls + λ D_CS(pₛ(z), pₜ(z)) + β Σ_{n=1}^{2} D_CCS(pₛⁿ(ŷ|z), pₜⁿ(ŷ|z)), with two classifiers (MCD-style) and target entropy constraints in L_cls.
  - In plain words: align the feature marginals with CS, and align "how each domain's classifier output varies with the feature" with conditional CS, using soft predictions as the target "label".
  - **F1** yes (posterior-conditional). **F2** yes. **F3** no: soft predictions enter a label kernel, with no confidence weights or thresholds.
- **RJSD** — Hoyos-Osorio & Sánchez Giraldo, "The Representation Jensen-Shannon Divergence", arXiv 2305.16446 (first posted 2023-05-25; no journal reference on arXiv) [verified, full text] — [arXiv 2305.16446](https://arxiv.org/abs/2305.16446)
  - Divergence: D_JS^H(P,Q) = S(½(C_P + C_Q)) − ½[S(C_P) + S(C_Q)], with von Neumann entropy S over uncentered covariance operators. The kernel-matrix estimator is in Prop. 8.
  - DA use (§5.4): replace JAN's MMD with RJSD over the joint covariance of activations in layers L, then min_θ (1/n) Σ J(f_θ(xᵢˢ), lᵢˢ) + β D̂_JS^H(P,Q). Compared against MMD on 4 benchmarks including Office-31.
  - In plain words: JAN's joint-layer alignment with an entropy-based divergence in place of MMD.
  - **F1** joint over layers (soft, implicit), not class-conditional. **F2** yes. **F3** no.
- **Maximizing Conditional Independence for UDA** (arXiv 2203.03212): "maximize the conditional independence of feature and domain given class in the reproducing kernel Hilbert space … viewed as minimizing a surrogate of a certain mutual information between feature and domain" [abstract only] — [arXiv 2203.03212](https://arxiv.org/pdf/2203.03212)
  - **F1** yes. **F2** yes (UDA). **F3** not determinable from the abstract [UNVERIFIED].
- **SIDA, "Surrogate Information Domain Adaptation"**: "maximizing mutual information between features of the same class on both source and target domains" [UNVERIFIED — search snippet only; arXiv id not confirmed] — [search hit](https://arxiv.org/abs/2302.01516v2)
- **CREDA**: F1 yes (hard per-class). F2 yes. F3 yes (w = 1 − H₂/log C on the target block) [verified] — §1 above.
- **DARES**: F1 yes ("class-conditional order-2 Rényi alignment"). F2 yes (UDA). F3 yes ("per-pixel Rényi-2 confidence weighting") [verified, README] — [DARES repo](https://github.com/liturriago/DARES)
- **DSAN / LMMD** — "Deep Subdomain Adaptation Network for Image Classification", arXiv 2106.09388, "published on TNNLS" [metadata verified this run; equation from prior notes, not re-read]
  - d̂ = (1/C) Σ_c ‖Σᵢ wᵢᶜˢ φ(zᵢˢ) − Σⱼ wⱼᶜᵗ φ(zⱼᵗ)‖², with target weights wⱼᶜ = ŷⱼ_c / Σ_k ŷ_k_c from soft predictions.
  - In plain words: per-class MMD in which every target sample contributes to every class in proportion to its predicted probability.
  - **F1** yes. **F2** yes. **F3** yes (soft).
  — [arXiv 2106.09388](https://ar5iv.arxiv.org/html/2106.09388)
- **CAN / CDD** — "Contrastive Adaptation Network for Unsupervised Domain Adaptation", arXiv 1901.00976, "Accepted by CVPR 2019" [metadata verified this run; equation from prior notes]
  - D̂^cdd = (1/M) Σ_c D̂^{cc} − (1/(M(M−1))) Σ_c Σ_{c′≠c} D̂^{cc′}.
  - Target labels come from spherical k-means. Samples far from their cluster are filtered, and classes with too few retained samples are excluded.
  - In plain words: pull same-class source and target together and push different classes apart.
  - **F1** yes. **F2** yes. **F3** yes, by selection rather than weighting.
  — [arXiv 1901.00976](https://arxiv.org/html/1901.00976v2)

### Inferences
- **[derived] Relationship to CS-DA.** Under the balanced mix, MIL-CREDA's J_c is a monotone function of a Cauchy–Schwarz divergence at fixed purity ratio (closed form in §1). That divergence is computed between trace-normalized covariance operators of the bag embeddings under k². Equivalently, for a Gaussian kernel it is the ITL CS divergence between weighted Parzen estimates on Ψ(B) with bandwidth σ/2, since k_σ² = k_{σ/√2} and ∫κ_τ κ_τ ∝ κ_{√2τ}, and the normalizing constants cancel in the CS ratio.
  - MIL-CREDA's global term is therefore best described as a class-conditional (p(Ψ|y)), confidence-weighted CS divergence over bag embeddings with purity-decoupled normalization.
  - CS-DA conditions on the posterior p(ŷ|z), uses soft labels inside a kernel, has no per-sample weights and operates on instances.
  - The divergence family is shared. The conditioning target, routing, weighting and data unit differ.
- RJSD is the principled α = 1 alternative: it is zero iff P = Q under a characteristic κ², at O(n³) cost. Its published DA instantiation is marginal/joint, so a class-conditional RJSD over bags would also be unoccupied, but it would be an alternative to MIL-CREDA, not prior art against it.

### Gaps
- Yu et al. 2023, the source of conditional CS that CS-DA cites, was not re-read.
- The full texts of 2203.03212 and SIDA were not read, so F3 is undetermined for both.
- No other matrix-Rényi DA paper (beyond CREDA and DARES) surfaced in the searches. Searches for "matrix-based entropy domain adaptation" returned only CREDA.

## 5. Final survivors (CREDA, DARES, DSAN/LMMD, CAN/CDD): what MIL-CREDA has that they lack, and vice versa

### Takeaway
Against every survivor, MIL-CREDA's distinguishing content is the bag unit (attention-weighted mean embeddings), the ½–½ balanced mixture that yields the closed-form CS reading, the purity-decoupled normalization ℓ_c, the gate q_c, the personalized local reference, and two encoders with a frozen source. It lacks an inter-class repulsion term (CDD, DARES), soft routing (LMMD), anti-collapse floors (DARES), detached weights (by prior notes), and adaptive thresholds. None of these differences proves superiority. Each is a design choice with a stated mathematical consequence, which an ablation must decide.

### Cited Findings
- CREDA facts: §1 [verified] — [CREDA PDF](https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02602/article_deploy/mathematics-13-02602.pdf)
- DARES facts: README [verified] — [DARES repo](https://github.com/liturriago/DARES)
- LMMD and CDD forms: prior verified notes, metadata re-verified here — [arXiv 2106.09388](https://ar5iv.arxiv.org/html/2106.09388); [arXiv 1901.00976](https://arxiv.org/html/1901.00976v2)

| | CREDA | DARES | DSAN (LMMD) | CAN (CDD) |
|---|---|---|---|---|
| Data unit | instance | pixel (sampled by Φ_c) | instance | instance |
| Encoders | 1 shared | 1 shared (backbone + head) | 1 shared | 1 shared |
| Routing | hard argmax | hard (README: "class-conditional") | soft, all classes | hard, k-means clusters |
| Confidence | w = 1 − H₂/log C on the Kₜ block only; Kˢᵗ raw | per-pixel Rényi-2 weights | implicit via soft ŷ | filtering plus a minimum class count |
| Measure | Ĩ₂ = ½(H_s + H_t) − H_mix, unbalanced mix | order-2 Rényi alignment | MMD² of weighted means | intra − inter MMD |
| Inter-class term | none | margin-hinged target repulsion | none | yes |
| Collapse guard | none | H₂ ≥ η floors, GradNorm trust region | none | none |
| Normalization across classes | none (1/1 sum) | not stated | 1/C | 1/M |

**What MIL-CREDA has that all four lack:**
- **Bags.** Ψ(B) = Σᵢ aᵢ φ(xᵢ) as the alignment unit, so the conditional distributions are distributions of mean embeddings, a second-level kernel construction.
- **Balanced mixture.** J_c = log(½ cosh((H_t − H_s)/2) + ½ e^(−D_CS)) [derived, numerically checked]. This separates "alignment" (D_CS) from "purity mismatch" (H_t − H_s). In CREDA's unbalanced form the two are confounded with nₛ_c / Σw².
- **Normalization ℓ_c = (U_c − J_c)/(U_c − L_c).** It vanishes iff D_CS = 0 and places classes on a common [0,1] scale. No survivor normalizes per class by its own attainable range.
- **Gate q_c = Σw.** It weights classes by confident target mass. CREDA only zeroes absent classes, and CAN uses a hard count threshold.
- **Local term.** An RKHS distance from each target bag to a source reference personalized by its full soft prediction. This is the only per-sample soft use of the full g in MIL-CREDA, and it plays the role LMMD's soft routing plays.
- **Two encoders, source frozen.** With the source encoder frozen, V_ss and H_s are constants per batch composition. Only the target side moves, which removes the "move both domains" degree of freedom that a shared encoder has.

**What the survivors have that MIL-CREDA lacks:**
- **Inter-class separation** (CDD's −Σ_{c≠c′} D^{cc′}; DARES's margin-hinged repulsion). J_c is purely within-class. Nothing in it penalizes two target classes overlapping each other, except through the CE on source and the hard routing.
- **Soft routing** (LMMD). Hard argmax discards the non-maximal mass of g in the global term. MIL-CREDA recovers it only in the local term.
- **Anti-collapse floors** (DARES's H₂ ≥ η). Prior notes derived that the unnormalized gradient rewards purity mismatch. With a frozen source, the only route is for the target class to collapse or disperse. ℓ_c removes the sign problem but, per the prior notes, a purity-dependent gradient remains. DARES's floors are a direct guard against exactly this.
- **Detached weights / adaptive thresholds.** According to the prior report, official code of MCC, ATDOC, SHOT and SoftMatch detaches confidence weights. FlexMatch and SoftMatch adapt thresholds per class. Prior notes state MIL-CREDA's w is not detached [from prior notes, not re-verified here]. CREDA's paper does not say either way.
- **CS-DA's marginal term.** CS-DA also aligns pₛ(z) and pₜ(z) marginally. MIL-CREDA (like CREDA) has no marginal term, so its alignment depends entirely on pseudo-label routing quality early in training.

### Inferences
- An honest novelty statement: *MIL-CREDA lifts CREDA's confidence-weighted, class-conditional matrix-Rényi score from instances to attention-weighted bag embeddings. It balances the source–target mixture so that the score becomes a closed-form function of a Cauchy–Schwarz divergence and a purity ratio, normalizes each class by its attainable range, gates classes by confident mass, adds a personalized local reference, and decouples a frozen source encoder from a trainable target encoder.* Its predecessors' weight formula, hard routing and score form are inherited and must be attributed to CREDA.
- **Claims not to make:**
  - "Rényi sees more structure than MMD": at α = 2 it is a CS/MMD_{k²} relative.
  - "Confidence weighting is new": CREDA, DSAN and DARES all have it.
  - "First conditional DA with Rényi": CREDA is.
  - "First MIL DA": Blast, TAKT and DETA exist.

### Gaps
- No experiment in the literature isolates "balanced vs. unbalanced mix" or "inter-class term vs. none" for a Rényi/CS conditional loss. Both are open ablations, not known results.
- Blast's mechanism (§3) remains the main unread threat to the "bag-level" part of the claim.
