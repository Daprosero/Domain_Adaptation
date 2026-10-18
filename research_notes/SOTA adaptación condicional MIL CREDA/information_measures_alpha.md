# Information-theoretic divergence measures for domain adaptation, and how the Rényi order α is chosen

Scope: catalogue of information measures used in DA / representation alignment (vs. Hilbert-embedding measures), and the evidence on choosing α for matrix-based Rényi entropy as used in CREDA / MIL-CREDA. Research run 2026-10-10. Every URL below was reached in this run unless the item is listed under a "Gaps" heading as not re-fetched.

## Q1. Catalogue of information measures and their DA uses (formula, estimator, sample behaviour, differentiability, density estimation, failure modes)

### Takeaway
The measures split into three estimator families: (i) plug-in on classifier outputs (entropy minimization, IM; no density estimation, cheap, but blind to feature-space geometry); (ii) variational/adversarial (DANN's JS link, f-DAL; need a critic network and minimax training); (iii) kernel / matrix estimators (ITL quadratic Rényi, Cauchy–Schwarz, matrix-based Rényi as in CREDA; closed-form, differentiable, O(n²) Gram cost, bandwidth-sensitive). The Rényi-divergence *bounds* of Mansour et al. 2009 are theory only (no estimator), and they hold for α>1.

### Cited Findings

**Matrix-based Rényi entropy (the estimator CREDA uses)**
- Definition: S_α(A) = 1/(1−α) · log₂ tr(A^α) on a positive definite matrix with tr A ≤ 1 (trace-normalized Gram matrix); it satisfies axioms analogous to Rényi's and avoids estimating the underlying density. — [Sánchez Giraldo, Rao & Príncipe, IEEE TIT 61(1):535–548, 2015 (arXiv 1211.2459)](https://ar5iv.labs.arxiv.org/html/1211.2459); [UKY record](https://scholars.uky.edu/en/publications/measures-of-entropy-from-data-using-infinitely-divisible-kernels/)
- Joint entropy via Hadamard product: S_α(A,B) = S_α((A∘B)/tr(A∘B)); conditional entropy and mutual information follow (I_α = S_α(A)+S_α(B)−S_α(A,B)). — [Sánchez Giraldo et al. 2015](https://ar5iv.labs.arxiv.org/html/1211.2459); MI form also in [Yu, Yu & Príncipe, DIB (ICASSP 2021, arXiv 2102.00533)](https://arxiv.org/pdf/2102.00533v1.pdf)
- Sample behaviour: convergence of the spectrum is O(n^{-1/2}) and *independent of input dimensionality* (Thm 6.2, bound ~ C·sqrt(2 log(2/δ)/N)). — [Sánchez Giraldo et al. 2015](https://ar5iv.labs.arxiv.org/html/1211.2459)
- Differentiability: H_α(A) and H_α(A,B) have analytical gradients, ∂H_α(A)/∂A = α/((1−α) tr(A^α)) · A^{α−1}, enabling end-to-end training. — [DIB, arXiv 2102.00533](https://arxiv.org/pdf/2102.00533v1.pdf)
- Cost: exact computation needs the eigenspectrum, O(n³); randomized approximations reduce to O(n²·s·m), with stochastic trace estimators for integer α and Taylor/Chebyshev polynomial approximations for non-integer α. — [Dong, Gong, Yu & Li, "Optimal Randomized Approximations for Matrix-based Rényi's Entropy", arXiv 2205.07426](https://arxiv.org/abs/2205.07426)
- Multivariate extension (joint entropy of several variables, used for feature selection / multivariate MI). — [Yu, Sánchez Giraldo, Jenssen & Príncipe, IEEE TPAMI (arXiv 1808.07912)](https://ar5iv.labs.arxiv.org/html/1808.07912)

**CREDA (the direct predecessor of MIL-CREDA)**
- "Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting", *Mathematics* 13(16):2602 (MDPI, 2025), Pérez-Rosero et al. Components: deep feature extractor; entropy-weighted down-weighting of low-confidence pseudo-labels; class-conditional alignment loss built as a Rényi-based kernel entropy estimator; uses a differentiable matrix-based formulation of Rényi's *quadratic* entropy; evaluated on Digits, ImageCLEF-DA, Office-31. — [DOI 10.3390/math13162602](https://doi.org/10.3390/math13162602); [ResearchGate](https://www.researchgate.net/publication/394905031_Conditional_Domain_Adaptation_with_a-Renyi_Entropy_Regularization_and_Noise-Aware_Label_Weighting) (full text returned HTTP 403 in this run; details from abstract/search snippet only)
- A downstream implementation (DARES, 2026, Iturriago Salas et al., deforestation segmentation) states it builds on CREDA and uses "hardened class-conditional order-2 Rényi alignment over Gaussian Gram matrices" with a median-heuristic bandwidth; α is fixed at 2, not configurable. — [github.com/liturriago/DARES](https://github.com/liturriago/DARES)

**ITL quadratic Rényi entropy / information potential (Príncipe)**
- Information potential V_α(X)=∫p(x)^α dx, H_α = −log V_α^{1/(α−1)}; for α=2 with a Gaussian Parzen window, Ĥ₂(X) = −log (1/N²)ΣΣ G_{σ√2}(x_i−x_j): a closed-form pairwise-sum estimator (the convolution of two Gaussians is a Gaussian). — [Boccato, ITL lecture notes (Unicamp)](https://www.dca.fee.unicamp.br/~lboccato/topico_12_information_theoretic_learning.pdf); [arXiv 2001.00265 (Fast estimation of ITL descriptors)](https://arxiv.org/pdf/2001.00265); book: Príncipe, *Information Theoretic Learning: Renyi's Entropy and Kernel Perspectives*, Springer 2010 — [library record](https://search.library.wisc.edu/catalog/9910093250202121)

**Cauchy–Schwarz divergence DA**
- D_CS(p,q) = −log[(∫pq)² / (∫p² ∫q²)]; kernel estimator D̂_CS = log(1/M² Σκ(z^s,z^s)) + log(1/N² Σκ(z^t,z^t)) − 2 log(1/MN Σκ(z^s,z^t)); a conditional CS (CCS) term aligns p(y|z); total objective D_CS(marginal) + D_CCS(conditional). No matrix inversion (unlike conditional MMD). Kernel width σ=1 after feature normalization ("a common heuristic"). — [Yin, Yu, Lin, Liu, Sonke & Gavves, UAI 2024, PMLR 244:4011–4040](https://proceedings.mlr.press/v244/yin24a.html); [arXiv 2405.19978](https://arxiv.org/html/2405.19978v1)
- The "tighter than KL" claim is proven **under a Gaussian assumption**: Prop. 1, D_CS(p;q) ≤ min(D_KL(p;q), D_KL(q;p)) for any two Gaussians; Prop. 2, D_TV ≤ sqrt(D_CS) under extra conditions; a non-Gaussian relation (eq. 7) involves constants C₁, C₂. — [arXiv 2405.19978, Sec. 3](https://br.arxiv.org/pdf/2405.19978)
- Reported: CCS has higher test power than conditional MMD (0.72 vs 0.20) on synthetic data with equal marginals and different nonlinear labelling rules (d=10, n=200); KL-based DA "hardly converges" with a nonlinear classifier and is sensitive to architecture. — [arXiv 2405.19978](https://arxiv.org/html/2405.19978v1)
- Note: D_CS is the Rényi-quadratic (α=2) cross-information-potential family, i.e. the same α=2 closed form as ITL. — [arXiv 2405.19978 (cites closed form for Gaussian mixtures, Kampa et al. 2011)](https://br.arxiv.org/pdf/2405.19978)

**Rényi divergence bounds for DA (theory)**
- Mansour, Mohri & Rostamizadeh, "Multiple Source Adaptation and the Rényi Divergence", UAI 2009. D_α(P‖Q) = 1/(α−1) log Σ P^α/Q^{α−1}; α=1 → KL, α=2 → log of expected density ratio, α=∞ → log max ratio. Bounds are stated with d_α = 2^{D_α}. Lemma 1: for any α>1, L_P(h,f) ≤ [d_α(P‖Q) · E_Q L^{α/(α−1)}]^{(α−1)/α} ≤ (d_α(P‖Q) L_Q(h,f))^{(α−1)/α}. Extends to arbitrary targets (not just mixtures). — [arXiv 1205.2628](https://arxiv.org/pdf/1205.2628); [PMLR R7](https://proceedings.mlr.press/r7/mansour09a.html)
- Caution: the similarly named arXiv 0902.3430 ("Domain Adaptation: Learning Bounds and Algorithms", same authors, COLT 2009) is about the *discrepancy distance*, not the Rényi divergence. — [arXiv 0902.3430](https://arxiv.org/abs/0902.3430)

**f-divergences (f-DAL)**
- Acuna, Zhang, Law & Fidler, "f-Domain-Adversarial Learning: Theory and Algorithms", ICML 2021: generalization bound using a variational characterization of f-divergences; recovers Ben-David et al. (2010) as a special case; introduces a correction to Ganin et al.'s adversarial objective; best results with f-divergences not previously used in DA. Estimator: adversarial critic (variational lower bound). — [NVIDIA project page](https://research.nvidia.com/labs/toronto-ai/fDAL); [ICML 2021 poster](https://icml.cc/virtual/2021/poster/9515)

**Amari α-divergence (closest DA α-sweep found)**
- IT-RUDA (Rashidi et al., arXiv 2210.12947, 2022): D_α = 1/(α(α−1)) [∫p^α q^{1−α} − 1] (α→1 KL, α→0 reverse KL); estimated by fitting GMMs to features, no critic/minimax. Office-Home ablation: best α≈0.7–0.9, "not particularly sensitive" in 0.6–0.95; small α over-robust, α→1 loses outlier resilience; robust under growing outlier ratio (partial/open-set DA). — [arXiv 2210.12947](https://arxiv.org/html/2210.12947)

**Information maximization (SHOT)**
- SHOT (Liang et al., ICML 2020) freezes the source classifier and trains the target feature extractor with information maximization plus self-supervised pseudo-labelling. — [arXiv 2002.08546](https://arxiv.org/abs/2002.08546v2); [ICML 2020](https://icml.cc/virtual/2020/poster/53992)

### Inferences
- Draft comparison table for the tutor (information vs Hilbert-embedding measures). Rows are supported by the citations above except where marked [not re-fetched]:

| Measure | Typical DA use | Estimator | Density est.? | Differentiable | Cost | Known failure modes |
|---|---|---|---|---|---|---|
| Shannon entropy min. [not re-fetched] | target confidence | plug-in on softmax | no | yes | O(n) | confirms wrong predictions; collapse to one class |
| IM (entropy − marginal entropy) | SHOT | plug-in on softmax | no | yes | O(n) | needs balanced-class prior; no feature geometry |
| KL / JS (DANN link) [not re-fetched] | marginal alignment | adversarial critic | implicit | yes (minimax) | critic net | unstable minimax; JS saturates on disjoint support |
| f-divergences (f-DAL) | marginal alignment | variational critic | implicit | yes (minimax) | critic net | same minimax issues; choice of f |
| Amari α-div. (IT-RUDA) | robust marginal alignment | GMM fit | yes (parametric) | yes | GMM | GMM misspecification |
| Rényi divergence bounds (Mansour 2009) | theory (multi-source) | none given | — | — | — | bound only, α>1 |
| CS divergence (Yin 2024) | marginal + conditional | kernel (α=2 IP) | no (implicit Parzen) | yes | O(n²) | bandwidth; "tighter than KL" proven for Gaussians |
| Matrix Rényi S_α (CREDA) | class-conditional | trace-normalized Gram, spectrum | no | yes (analytic grad) | O(n²) at α=2, O(n³) else | bandwidth; per-class n small |
| MMD / CMMD (Hilbert) [not re-fetched] | marginal / conditional | kernel mean embedding | no | yes | O(n²) | bandwidth; CMMD needs matrix inverse (per Yin 2024) |
| Wasserstein / OT [not re-fetched] | marginal / joint | Sinkhorn / dual critic | no | yes | O(n²) per iter | entropic bias; cost choice |

- Matrix-based Rényi and CS divergence sit between the two worlds: they are kernel (Hilbert-space) computations with information-theoretic semantics, which is the strongest framing for MIL-CREDA in the tutor's table.
- The dimension-free O(n^{-1/2}) rate matters for MIL-CREDA because per-class bag counts per batch are small; it should be quoted with its assumptions (bounded kernel) when written up.

### Gaps
- Not re-fetched in this run (standard citations, but per the "verify every citation" constraint they are unverified here): Grandvalet & Bengio 2005 (entropy minimization, NeurIPS); Krause, Perona & Gomes 2010 (discriminative clustering by regularized IM, NeurIPS); Ganin et al. 2016 JMLR (DANN and its JS/H-divergence link); Ben-David et al. 2010 (H∆H bound); Jeffreys / symmetric KL in DA; Jensen–Rényi divergence in DA; mutual-information DA methods; Wasserstein/OT DA (Courty et al., Shen et al.). The table rows marked [not re-fetched] rest on these.
- CREDA's full text (MDPI) returned 403; its exact loss formula, bandwidth rule, and whether it reports any α ablation could not be confirmed. A search snippet (source not attributable — it appeared alongside the CREDA and Nyström results) states that α=2 was selected for "strong theoretical grounding and computational advantages" because the quadratic case reduces to "a closed-form expression based on the Frobenius norm of the normalized Gram matrix, avoiding the unstable matrix powers". Treat as unverified until read in the CREDA PDF.
- "Nyström-aware approximations for matrix-based Rényi's entropy" (Neural Networks, 2025, PubMed 41232221) — found but the page was not readable; authors and α content unverified.
- The which-order-appears-in-the-theorems detail of Mansour 2009 beyond Lemma 1 (e.g. whether α=2 is singled out) not checked.

## Q2. How α is chosen in practice (ITL, matrix-based Rényi, IB works) and empirical sensitivity

### Takeaway
There is no derived optimal α. Practice clusters at two values for two different reasons: α=2 (ITL, CS divergence, CREDA, DARES) for closed-form/computational convenience; α≈1.01 (Sánchez Giraldo 2015 experiments, Yu et al. 2019) to approximate Shannon. The only explicit guidance is a heuristic rule from Yu et al. (TPAMI): α<2 (→1⁺) to emphasize tails/multimodality, α=2 "neutral", α>2 for modal behaviour. The one sensitivity study found in the matrix-Rényi line (Yu et al., feature selection) found α=2 *worse* than 1.01 and 0.6.

### Cited Findings
- Sánchez Giraldo et al. note that "for α=2, a rather simple yet elegant plug-in estimator ... can be derived", but develop the theory for all α>0 and run their experiments (independence tests, sample sizes from dozens to thousands, dims 1–3) with α=1.01. — [arXiv 1211.2459](https://ar5iv.labs.arxiv.org/html/1211.2459)
- Yu, Sánchez Giraldo, Jenssen & Príncipe (TPAMI): "If the application requires emphasis on tails of the distribution (rare events) or multiple modalities, α should be less than 2 and possibly approach to 1 from above. α=2 provides neutral weighting. Finally, if the goal is to characterize modal behavior, α should be greater than 2." Primary setting α=1.01, chosen "to approximate the Shannon's information and make this comparison fair". — [ar5iv 1808.07912](https://ar5iv.labs.arxiv.org/html/1808.07912)
- Same paper, sensitivity study over α ∈ {0.6, 1.01, 2} for feature selection: α=0.6 beat 1.01 (attributed to tail emphasis); α=2 "degraded significantly across most datasets"; authors conclude "values of α lower than 2 are preferred" for that classification-based feature-selection task. — [ar5iv 1808.07912](https://ar5iv.labs.arxiv.org/html/1808.07912)
- Same paper, bandwidth rule: σ at "10 to 20 percent of the total range of the Euclidean distances between all pairwise data points", or Silverman's rule. — [ar5iv 1808.07912](https://ar5iv.labs.arxiv.org/html/1808.07912)
- DIB (Yu, Yu & Príncipe): RBF kernel, σ = average over samples of the mean distance to the k=10 nearest neighbours. — [arXiv 2102.00533](https://arxiv.org/pdf/2102.00533v1.pdf)
- ITL: α=2 is used because the Gaussian information potential is closed-form. — [Boccato ITL notes](https://www.dca.fee.unicamp.br/~lboccato/topico_12_information_theoretic_learning.pdf)
- CS-DA (α=2 family) uses σ=1 after normalization; its sensitivity analysis covers the loss weights λ, β (Table 7), not α or σ. — [arXiv 2405.19978](https://br.arxiv.org/pdf/2405.19978)
- CREDA and DARES use α=2 (quadratic); DARES hard-codes it. — [DOI 10.3390/math13162602](https://doi.org/10.3390/math13162602); [DARES repo](https://github.com/liturriago/DARES)
- In a DA setting with the (different) Amari α-divergence, IT-RUDA found a broad flat optimum α≈0.6–0.95 on Office-Home. — [arXiv 2210.12947](https://arxiv.org/html/2210.12947)

### Inferences
- The defensible justification for α=2 in MIL-CREDA is *computational and statistical-stability* (exact, eigendecomposition-free, O(n²), smooth gradient), plus Yu et al.'s "neutral weighting" heuristic — not optimality. Claiming α=2 is "best" would contradict the only matrix-Rényi sensitivity study found (Yu et al.), although that was feature selection, not DA alignment, so it does not transfer directly.
- Because the CREDA loss is an *alignment* objective (compare class-conditional Gram structures), the relevant question is whether a different α changes the ranking of source/target discrepancies, which is exactly what a synthetic sweep can test (Q4).

### Gaps
- No DA paper using *matrix-based* Rényi with an α sweep was found. The DIB paper's α value was not located in the extracted text (it may be in a table/figure); unverified.
- I found no empirical sensitivity analysis in information-bottleneck works beyond the Yu et al. TPAMI study.

## Q3. Theoretical properties as a function of α

### Takeaway
Limits and cost are well established; robustness/outlier claims are heuristic (stated by Yu et al. and IT-RUDA, not proven for the matrix estimator). α=2 is the unique order where the matrix entropy needs no spectrum.

### Cited Findings
- Rényi entropy is non-negative and non-increasing in α (H_{α₁} ≥ H_{α₂} for α₁<α₂); α=0 → log support size, α=1 → Shannon, α=2 → −log collision probability, α=∞ → min-entropy; D_α → KL at α=1, D_∞ = log max density ratio. — [Mansour et al. 2009, Sec. 2](https://arxiv.org/pdf/1205.2628)
- Matrix version: S_α(A) = 1/(1−α) log₂ Σλ_i(A)^α, α ∈ (0,1)∪(1,∞); α→1 is the von Neumann-type limit approximated in practice by α=1.01. — [DIB eq. 2](https://arxiv.org/pdf/2102.00533v1.pdf); [Sánchez Giraldo 2015](https://ar5iv.labs.arxiv.org/html/1211.2459)
- Exact cost O(n³) via eigenspectrum; integer α admits randomized trace estimators, fractional α needs polynomial approximation. — [arXiv 2205.07426](https://arxiv.org/abs/2205.07426)
- Tail vs mode weighting (α<2 tails/multimodality; α>2 modes) — heuristic guidance. — [Yu et al. TPAMI](https://ar5iv.labs.arxiv.org/html/1808.07912)
- Robustness to outliers for orders <1 (Amari α) demonstrated empirically in DA. — [IT-RUDA](https://arxiv.org/html/2210.12947)
- Gradient: ∂S_α/∂A ∝ A^{α−1}/tr(A^α). — [DIB](https://arxiv.org/pdf/2102.00533v1.pdf)

### Inferences (derivations, not citations)
- Because A is symmetric PSD with tr A = 1, tr(A²) = ‖A‖_F², so S₂(A) = −log₂‖A‖_F² = −log₂ Σ_ij A_ij²: no eigendecomposition, O(n²), and the gradient is ∝ A itself (linear), which is the stability argument. For any non-integer α, A^{α−1} requires the spectrum; for α<1 small eigenvalues are raised to a negative power in the gradient, which can blow up when the Gram matrix is near-singular (small σ, duplicated bag embeddings). This gradient-instability point is my derivation; I found no paper measuring it.
- As α→∞, S_α → −log₂ λ_max(A): only the dominant direction of the class-conditional embedding counts; as α→0 it counts rank. α=2 weights eigenvalues by λ², favouring dominant modes over the tail relative to Shannon.
- Concavity of Rényi entropy in the distribution holds for α∈(0,1] (standard result; not re-fetched this run).

### Gaps
- No source found that proves outlier robustness, concavity, or gradient bounds specifically for the *matrix-based* functional as a function of α.
- Concavity statement and the min-entropy interpretation for matrices are standard but not sourced in this run.

## Q4. Designing a synthetic experiment that varies α

### Takeaway
No existing DA paper sweeps α for matrix-based Rényi alignment, so such an experiment would be novel rather than a replication. Closest designs: Sánchez Giraldo 2015 (synthetic rotated ICA distributions, varying n and dimension), Yin et al. 2024 (equal marginals, different conditional labelling rules; measure test power), Yu et al. TPAMI (α ∈ {0.6, 1.01, 2}), IT-RUDA (α grid plus outlier-ratio sweep).

### Cited Findings
- Synthetic independence tests on rotated ICA benchmark distributions, n from dozens to thousands, dims 1–3, comparing against kernel statistics. — [Sánchez Giraldo 2015](https://ar5iv.labs.arxiv.org/html/1211.2459)
- Conditional-shift power test: fixed marginals, nonlinear labelling rules differ, d=10, n=200; metric = probability of rejecting H0 (test power). — [Yin et al. 2024](https://arxiv.org/html/2405.19978v1)
- α grid {0.6, 1.01, 2}, downstream accuracy as metric. — [Yu et al. TPAMI](https://ar5iv.labs.arxiv.org/html/1808.07912)
- α grid with increasing outlier-to-inlier ratio; accuracy curves. — [IT-RUDA](https://arxiv.org/html/2210.12947)
- Randomized estimators make non-integer α affordable at larger n (O(n²sm)). — [arXiv 2205.07426](https://arxiv.org/abs/2205.07426)

### Inferences (proposed design; heuristic, not from a single source)
- Grid: α ∈ {0.5, 0.8, 1.01, 1.5, 2, 3, 5} (covers tail-weighted, Shannon, neutral, modal; mirrors Yu et al.'s rule). Hold σ fixed by one rule (median heuristic as DARES, or Yu's 10–20% range) and also cross σ × α, since α and σ both reshape the spectrum and can confound each other.
- Controlled shifts, each with a scalar severity knob: rotation angle of class-conditional clouds; translation of one class only (class-conditional shift); anisotropic scaling; label-proportion shift (to test whether per-class weighting holds); outlier/heavy-tail contamination fraction (to test the tail claim of α<2); multimodal classes (to test α>2 vs α<2 on modes). For the MIL setting, vary bag size and witness rate so the bag embedding's Gram spectrum changes.
- Metrics: (a) estimator-level — monotonicity / Spearman correlation of the class-conditional Rényi discrepancy with the shift knob, and two-sample test power at fixed n (as Yin et al.); (b) training-level — target accuracy after adaptation, seed variance, gradient-norm statistics and NaN/failure rate across α (tests the stability argument); (c) cost — wall time per step (exact eigh vs Frobenius).
- The honest written justification can then be: "α=2 is chosen for an exact O(n²) eigendecomposition-free loss with a linear gradient; a synthetic sweep shows whether accuracy/test power is flat in α around 2," which converts the tutor's question into a measured claim.

### Gaps
- I found no paper that runs an α sweep for matrix-based Rényi *alignment* or for MIL; whether results are flat in α is unknown.
- No source quantifies the σ–α interaction for the matrix estimator.
