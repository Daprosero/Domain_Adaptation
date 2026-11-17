# Matrix-based Rényi entropy on Gram kernels vs. Hilbert-space embeddings (MMD, kernel mean embeddings, covariance operators, HSIC): theory for MIL-CREDA

Notation used throughout. Points x_i come with weights a_i (with Σa_i = 1). The kernel is normalized, k(x,x) = 1, and has feature map φ. The trace-normalized Gram is A_ij = √(a_i a_j) k(x_i,x_j), so tr A = 1. The weighted uncentered covariance operator ("density operator") is C = Σ_i a_i φ(x_i)⊗φ(x_i). We write V_ss = Σ a_i a_j k_ij², V_tt = Σ b_i b_j k_ij², and V_st = Σ a_i b_j k(x_i, y_j)².

Provenance tags:
- **[fetched]**: the source was opened and read in this run.
- **[snippet]**: only a search-result snippet or an abstract was seen in this run.
- **[not opened]**: a standard reference cited from prior knowledge. The URL was not opened in this run, so check it before quoting.
- **[derived]**: my own algebra. It was checked numerically in this run with `numpy` on random Gaussian-kernel data (n=30, m=40, random weights). The identities matched to about 1e-16.

## Q1. Exact relationships: matrix Rényi ↔ information potentials ↔ CS divergence ↔ MMD ↔ covariance operators ↔ HSIC

### Takeaway
At α=2, every quantity in MIL-CREDA is a rational/log function of three numbers: V_ss, V_tt and V_st. These are the weighted Parzen information potentials of the **squared** kernel k². They are also exactly the Hilbert–Schmidt inner products of the covariance operators: tr(C_s C_t) = V_st. Covariance operators under k are the mean embeddings under k². So at α=2 the score is a nonlinear function of the same three terms that make up MMD²_{k²}. It is not a strictly "more spectral" object. The in-house derivations (J = −JR₂, tr A² = weighted IP, ‖C‖²_F/√(ρ_sρ_t) = CS) are all correct.

### Cited Findings
- **Definition of matrix Rényi entropy.** For A positive definite with tr A = 1, S_α(A) = (1/(1−α)) log₂ tr(A^α). Eq. (12) of Sánchez Giraldo, Rao & Príncipe. — [Sánchez Giraldo et al., arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Origin of the construction.** It generalizes the Parzen information-potential estimator. With K_ij = κ_{√2σ}(x_i,x_j), "Ĥ₂(X) = −log((1/n²) tr(KK)) + C(σ)" (eq. 11). The authors say: "the information potential estimator can be related to the Frobenius norm of the Gram matrix". This is the source for "H₂ = −log‖A‖²_F" being the Rényi quadratic-entropy Parzen estimator. — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Link to the RKHS covariance operator (Prop. 6.1).** Take the empirical operator Ĝ_N = (1/N)Σ φ(x_i)⊗φ(x_i). Its positive eigenvalues λ_i satisfy (1/N)Kα = λα, so "Nλ_i are all the positive eigenvalues of K". Hence tr(Ĝ_N^α) = tr((K/N)^α) (eq. 50). The paper states this for uniform weights 1/N. — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Joint entropy via the Hadamard product.** S_α(A,B) = S_α(A∘B / tr(A∘B)) (eq. 23). This equals the entropy under the product kernel κ₁κ₂.
  - Conditional entropy: S_α(A|B) = S_α(A∘B/tr) − S_α(B) (eq. 31).
  - Mutual information: I_α(A;B) = S_α(A) + S_α(B) − S_α(A∘B/tr) (eq. 33).
  - Both need nonnegative entries and A_ii = 1/n.

  — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Von Neumann limit.** α→1 gives S(ρ) = −tr(ρ log ρ) (eq. 21). The paper says its functional "bears a lot of resemblance" to quantum Rényi entropies. — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Multivariate extension (Yu, Sánchez Giraldo, Jenssen, Príncipe, IEEE TPAMI 42:2960–2966, 2020).** It defines a matrix-based joint entropy among more than two variables. This gives interaction information and total correlation, applied to hyperspectral band selection. — [arXiv:1808.07912, snippet](https://arxiv.org/abs/1808.07912)
- **Representation Jensen–Shannon divergence (RJSD).** This is the closest published object to MIL-CREDA's J. Hoyos-Osorio & Sánchez Giraldo define C_P = E[φ(X)⊗φ(X)], which has unit trace when κ(x,x) = 1.
  - Definition: D_JS^H(P,Q) = S((C_P+C_Q)/2) − ½(S(C_P)+S(C_Q)), with S the von Neumann entropy.
  - Kernel-matrix estimator (Prop. 8): S(K_Z/(n+m)) − [n/(n+m) S(K_X/n) + m/(n+m) S(K_Y/m)], where K_Z is the Gram matrix of the concatenated sample. This is exactly MIL-CREDA's block construction, at α=1.
  - Venue: arXiv:2305.16446v4, Oct 2024. No journal was seen.

  — [arXiv:2305.16446, fetched](https://arxiv.org/html/2305.16446)
- **RJSD vs. MMD.**
  - Lemma 5: "D_JS^H(P,Q) ≥ (1/8)‖C_P − C_Q‖²_HS = (1/8) MMD²_{κ²}(P,Q)".
  - Theorem 11: at power-series order p=1 the estimator equals (1/4)·MMD²_{κ²}.
  - The authors call RJSD "a higher-order extension of the maximum mean discrepancy". This independently confirms that covariance-operator quantities under κ are MMD-type quantities under κ².

  — [arXiv:2305.16446, fetched](https://arxiv.org/html/2305.16446)
- **Representation Jensen–Rényi divergence (Hoyos Osorio, Skean, Brockmeier, Sánchez Giraldo, arXiv:2112.01583, v4 June 2022).** It is built from Gram-matrix eigenvalues and has "properties similar to Jensen-Shannon divergence". The abstract does not say which α range keeps it nonnegative. — [arXiv:2112.01583, snippet](https://arxiv.org/abs/2112.01583)
- **Bach, "Information Theory with Kernel Methods" (arXiv:2202.08545).**
  - The von Neumann entropy and relative entropy of kernel covariance operators "share many of their properties" with Shannon entropy and KL divergence.
  - With tensor-product kernels, the resulting mutual information "perfectly characterize[s] independence" but only partially captures conditional independence.
  - The paper also gives estimation algorithms. Only the abstract was read; check the conditions (universal kernels) in the body.

  — [Bach, arXiv:2202.08545, snippet](https://arxiv.org/abs/2202.08545)
- **Cauchy–Schwarz divergence as Yin et al. (UAI 2024) write it.** D_CS(p;q) = −log((∫pq)² / (∫p² ∫q²)). This is twice Jenssen's single-log form −log(∫pq / √(∫p²∫q²)). Yin et al.: "the CS divergence estimator puts a 'logarithm' on each term of that of MMD". — [Yin et al., arXiv:2405.19978, fetched](https://arxiv.org/html/2405.19978v1)
- **Original CS divergence (Jenssen et al., 2006).** Jenssen, Príncipe, Erdogmus & Eltoft, "The Cauchy–Schwarz divergence and Parzen windowing: connections to graph theory and Mercer kernels", J. Franklin Institute 343(6), 2006. Search results confirm it as the origin of the Parzen/Gram-sum form of CS. — [snippet via arXiv:2301.08970](https://arxiv.org/html/2301.08970v3); [DOI 10.1016/j.jfranklin.2005.10.006, not opened](https://doi.org/10.1016/j.jfranklin.2005.10.006)
- **MMD (Gretton et al., JMLR 2012).** MMD²(P,Q) = ‖μ_P − μ_Q‖²_H = E k(x,x') + E k(y,y') − 2E k(x,y). — [JMLR 13:723–773, not opened](https://jmlr.org/papers/v13/gretton12a.html)
- **HSIC (Gretton et al., ALT 2005).** HSIC = ‖C_XY − μ_X⊗μ_Y‖²_HS. The biased estimator is (1/n²) tr(K H L H). — [ALT 2005, not opened](https://link.springer.com/chapter/10.1007/11564089_7)
- **Other standard references (all not opened):**
  - Fukumizu, Bach & Jordan, JMLR 2004: conditional cross-covariance operators. — [JMLR v5, not opened](https://www.jmlr.org/papers/v5/fukumizu04a.html)
  - Muandet et al., kernel mean embedding survey. — [arXiv:1605.09522, not opened](https://arxiv.org/abs/1605.09522)
  - Príncipe, Information Theoretic Learning, Springer 2010. — [DOI 10.1007/978-1-4419-1570-2, not opened](https://doi.org/10.1007/978-1-4419-1570-2)

### Inferences
Each item below is **[derived]** and was checked numerically in this run unless it says otherwise.

1. **Kernel-trick identity (key result).** tr(C_s C_t) = Σ_{ij} a_i b_j ⟨φ(x_i),φ(y_j)⟩² = Σ a_i b_j k(x_i,y_j)² = V_st. Likewise tr(C_s²) = ‖A_s‖²_F = V_ss = Σ a_i a_j k_ij². This confirms the in-house claim that "tr(A²) = Σ α_iα_j K_ij²".
   - With a Gaussian kernel of width σ, k² is a Gaussian of width σ/√2. So V is the weighted Parzen information potential at width σ/√2, up to a normalization constant.
   - Since φ⊗φ is the feature map of k², the density operator C is the **mean embedding of P under k²**.
2. **Spectral identity.** The trace-normalized Gram of the weighted features [√a_i φ(x_i)] has the same nonzero spectrum as C. This holds for **any** kernel, not only a linear kernel on mean embeddings Ψ, so the linear-kernel assumption in the in-house derivation is sufficient but not necessary.
   - Concatenate the features Φ = [Φ_s Φ_t]/√2. Its Gram is the block matrix ½[[A_s, C],[Cᵀ, A_t]], with cross block C_ij = √(a_i b_j) k(x_i,y_j). Its spectrum equals that of ½(C_s + C_t).
   - Therefore J = ½[H₂(A_s)+H₂(A_t)] − H₂(A_mix) = −JR₂^{(½,½)}(C_s, C_t). This holds **provided** the cross block uses the same kernel and the same √(weight) scaling as the diagonal blocks. If it does not, A_mix is not a valid Gram matrix of the mixture.
3. **Closed form at α=2** (log base 2; numerically identical to the direct computation, −0.4038 vs −0.4038):
   - J = log[(V_ss + V_tt + 2V_st) / (4√(V_ss V_tt))].
   - Decomposition: J = log(½·r + ½·e^{−D_CS}), where r = (V_ss+V_tt)/(2√(V_ssV_tt)) ≥ 1 (AM/GM of the two purities) and D_CS = −log(V_st/√(V_ssV_tt)) is Jenssen's single-log CS divergence.
   - The in-house "‖C‖²_F/√(ρ_sρ_t)" is exp(−D_CS), the cosine between C_s and C_t in Hilbert–Schmidt space.
4. **MMD in information potentials.** MMD²_{k²}(P_s,P_t) = ‖C_s − C_t‖²_HS = V_ss + V_tt − 2V_st. Hence J = log[(2(V_ss+V_tt) − MMD²_{k²}) / (4√(V_ssV_tt))], verified numerically.
   - The MMD under the original k is the ‖μ_s−μ_t‖² analog with k instead of k². It sees first moments of φ, not second moments.
5. **Quadratic Rényi cross-entropy.** H₂(p;q) = −log ∫pq is estimated by −log V_st (Príncipe's ITL; book not opened). So D_CS = H₂(p;q) − ½H₂(p) − ½H₂(q), all at kernel k².
6. **HSIC vs. matrix MI at α=2.** For A_ii = 1/n, tr(A∘B) = 1/n, so I₂(A;B) = log[V_J / (V_X V_Y)].
   - V_J = (1/n²) Σ kx_ij² ky_ij² is the joint IP; V_X and V_Y are the marginal IPs, all with squared kernels.
   - HSIC with kernels kx², ky² is V_J + V_X V_Y − 2V_C, with V_C = (1/n³) Σ_i (Σ_j kx_ij²)(Σ_l ky_il²). This is Príncipe's Euclidean-distance QMI form.
   - So matrix I₂ is a log-ratio that **omits** the cross term V_C. It is neither HSIC nor CS-QMI = log(V_J V_X V_Y / V_C²).
   - Not numerically checked in this run. The V_C term is the standard ITL form, not verified against the book.
7. **Higher α carries more than MMD.** For integer p, tr(C^p) = E[k(x₁,x₂)k(x₂,x₃)…k(x_p,x₁)] over i.i.d. draws. This is a cyclic p-point kernel moment. (Standard algebra, not numerically checked here.)
   - α=2 uses only pairwise k² terms, the same information as MMD_{k²}.
   - Non-integer α and α→1 (von Neumann) depend on the whole spectrum, which is information beyond any single MMD.

### Gaps
- Príncipe 2010 and Jenssen 2006 were not opened. The exact constants (σ√2, normalization) in their CS/IP estimators are from memory.
- The venue of Sánchez Giraldo et al. (IEEE Trans. Inf. Theory 61(1):535–548, 2015) was not verified in this run. The arXiv page did not show the journal reference.
- Whether Yu et al. (AAAI 2021, "Measuring dependence with matrix-based entropy functional") gives a formal HSIC equivalence was not checked.

## Q2. What Rényi offers beyond MMD, and where it is weaker

### Takeaway
The proven advantages are:
- Rényi-style axioms on PSD matrices.
- Invariance to kernel scale (via trace normalization).
- Direct joint, conditional and MI quantities through Hadamard products.
- Concentration of the estimator for α>1.
- At α=1, a valid divergence that lower-bounds the true JSD and upper-bounds MMD²_{k²}/8.

The weaknesses are:
- For α=2 the "spectral" claim is hollow: H₂ is a function of pairwise k² sums, exactly like MMD_{k²}.
- For α>1 the Jensen–Rényi construction is **not** a divergence. It can be negative, and J=0 does not imply P=Q.
- There is no published two-sample test with level/power guarantees for matrix Rényi at α=2. The literature has one for MMD.
- α≠2 needs an O(n³) eigendecomposition.

### Cited Findings
- **Axioms proven (Prop. 3.1).** For tr A = tr B = 1, S_α satisfies:
  - (i) unitary invariance;
  - (ii) continuity of S_α(pA);
  - (iii) S_α(I/n) = log₂ n;
  - (iv) additivity under tensor products: S_α(A⊗B) = S_α(A) + S_α(B);
  - (v) a quasi-arithmetic mean rule for orthogonal mixtures (AB = BA = 0).

  — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Maximum entropy (Prop. 3.2).** For α>1, S_α(A) ≤ S_α(I/n). The proof in the text covers α>1. Rank-1 A gives S_α = 0. — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Joint-entropy inequalities (Prop. 4.1).** For PD A, B with unit trace, nonnegative entries and A_ii = 1/n:
  - S_α(A∘B/tr) ≥ S_α(B);
  - S_α(A∘B/tr) ≤ S_α(A) + S_α(B).

  So conditional entropy is in [0, S_α(A)] and I_α ≥ 0. The proof uses majorization and Schur-concavity of Rényi entropy (Lemma 4.1, "Schur-concave for α > 0"). The authors warn that Schur-concavity "ought not be confused with the more common definition of concavity". — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Kernel normalization requirement.** Â_ij = A_ij / √(A_ii A_jj) (eq. 34), i.e. a constant diagonal. The paper uses infinitely divisible kernels (Sec. 5) to get the nonnegative-entry and Hadamard-power structure. — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Estimation (Thm 6.2, eq. 54).** For a bounded kernel, with probability 1−δ, (Σ(λ_i − λ̂_i)²)^{1/2} ≤ C√(2 log(2/δ)/N). For α>1, |tr(G^α) − tr(Ĝ_N^α)| ≤ αC√(2 log(2/δ)/N). A similar bound holds for the conditional-entropy ratio (Prop. 6.2).
  - Bias: the entropy is the log of tr(Ĝ^α), so it is biased even though E[Ĝ_N] = G. The paper bounds deviation and does not give an unbiased estimator.

  — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **RJSD (α=1) properties.**
  - It is nonnegative, symmetric and bounded by log 2.
  - Theorem 6: if κ² is characteristic, "D_JS^H(P,Q)=0 iff P=Q".
  - Theorem 7: D_JS^H ≤ D_JS (the classical JSD), which allows variational estimation of JSD.
  - Lemma 5: it is ≥ (1/8) MMD²_{κ²}.

  The paper reports gains over MMD in two-sample testing and in domain adaptation (ResNet-50, Office-31). — [arXiv:2305.16446, fetched](https://arxiv.org/html/2305.16446)
- **Characteristic kernels for MMD.** MMD with a characteristic kernel is zero iff P = Q. Gretton et al. give unbiased U-statistic estimators and two-sample tests with consistency guarantees. — [JMLR 2012, not opened](https://jmlr.org/papers/v13/gretton12a.html)
- **CS divergence advantages (Yin et al., UAI 2024).**
  - A generalization bound "theoretically tighter" than KL-based bounds.
  - D_CS ≤ min{D_KL(p;q), D_KL(q;p)}, proven for Gaussians.
  - A conditional CS estimator that "avoids introducing a hyperparameter λ and the necessity of matrix inverse", unlike conditional-MMD/operator approaches.

  — [Yin et al., arXiv:2405.19978, fetched](https://arxiv.org/html/2405.19978v1); [UAI 2024 / PMLR v244, snippet](https://proceedings.mlr.press/v244/yin24a.html)
- **CS inherits scale invariance.** CS is a log-ratio and is invariant to rescaling either density estimate. MMD scales linearly with the kernel amplitude. Yin et al. contrast the "logarithm on each term" with MMD's additive form. — [arXiv:2405.19978, fetched](https://arxiv.org/html/2405.19978v1)

### Inferences
1. **Scale invariance [derived].** Scale invariance means invariance to k → c·k. It is **not** invariance to bandwidth. H₂ depends on σ exactly as V does.
   - Trace normalization makes S_α invariant to the kernel amplitude, and CS is invariant too. MMD is not.
   - MMD with a normalized kernel (k(x,x) = 1) is also bounded, so this advantage is about amplitude-free comparison across classes and batches, not about boundedness.
2. **The α knob (from the definition) [derived].** tr(A^α) = Σλ_i^α. For α>1 the dominant eigenvalues (dominant modes/clusters) dominate. For α<1 small eigenvalues (tail, diversity) gain weight. α→0 counts the rank.
   - Sánchez Giraldo notes that the bound constant can drop to 1 when the dominant eigenvalues fall below α^{−1/(α−1)}.
   - Only α=2 avoids eigendecomposition: O(n²) via the Frobenius norm vs O(n³) for general α.
3. **The "whole spectrum" claim does not hold at α=2 [derived].** Σλ_i² = ‖C‖²_HS is a single quadratic statistic. Q1, inference 4, shows J is a function of (V_ss, V_tt, MMD²_{k²}).
   - What α=2 does add over MMD_{k²} is a nonlinear normalization by the purities √(V_ssV_tt). That is the CS-type cosine plus the purity-mismatch term r.
   - Using k² instead of k means the embedding captures second-order (covariance) structure of φ. Equivalently, it is a first-order embedding under a narrower kernel.
4. **Zero-iff claim for J at α=2 [derived].** −J = 0 iff (V_ss+V_tt)/2 − √(V_ssV_tt) = √(V_ssV_tt) − V_st, i.e. the purity mismatch exactly cancels the CS gap.
   - If the purities are equal (V_ss = V_tt), −J = 0 iff C_s = C_t. If k² is characteristic, that holds iff P_s = P_t (same argument as RJSD Thm 6).
   - In general J = 0 does **not** identify P_s = P_t.
5. **The MIL-CREDA objective is entangled [derived].** Because J = log(½r + ½e^{−D_CS}):
   - Maximizing J rewards smaller D_CS (alignment), which is desired. It also rewards a **larger purity mismatch** r between source and target class operators, which is undesired. A collapsed target class against a diffuse source class increases r.
   - Minimizing J does the opposite.
   - Bounds: log(r/2) < J ≤ log((r+1)/2).
   - A clean alternative at the same O(n²) cost is to use −D_CS directly, which is a log-cosine of C_s and C_t. Another is to use von Neumann RJSD (α=1, O(n³), proven ≥ MMD²_{k²}/8 and zero iff P=Q under characteristic k²).

### Gaps
- No paper was found that gives a two-sample test (null distribution, consistency) for the α=2 matrix-Rényi Jensen divergence. RJSD reports permutation-based tests empirically. Its formal test guarantees were not checked.
- Bias-correction schemes for matrix-Rényi estimators were not found in this run.

## Q3. Concavity of Rényi entropy and the sign of the Jensen–Rényi divergence

### Takeaway
Rényi entropy is concave for α ∈ (0,1]. For α>1 it is only Schur-concave and quasi-concave. Concavity for α>1 holds in general only for binary distributions with α ≤ 2 (Ben-Bassat & Raviv, as cited by Hamza & Krim; the primary statement was not verified here).

Consequently JR_α ≥ 0 is guaranteed for α ≤ 1 but not for α=2 with three or more outcomes. I found an exact counterexample, and the same example transfers to density matrices.

### Cited Findings
- **The Jensen–Rényi divergence (He, Hamza & Krim, 2003).** It was introduced in "A generalized divergence measure for robust image registration", IEEE Trans. Signal Processing 51:1211–1220, April 2003. It was used for ISAR image registration, and its convexity and an upper bound were established. — [search snippet, NCSU repository](https://repository.lib.ncsu.edu/server/api/core/bitstreams/63bdd764-b683-4dca-a1ad-8b7a2586e464/content)
- **Hamza & Krim, ISIT 2003.** "Jensen-Rényi divergence measure: theoretical and computational perspectives". The PDF returned 403, so its exact statements were not read. — [ResearchGate, snippet only](https://www.researchgate.net/publication/240883871_Jensen-Renyi_divergence_measure_theoretical_and_computational_perspectives)
- **Another Jensen–Rényi variant.** Some literature defines it as the average of Rényi divergences to the midpoint, JR_α(p‖q) = ½D_α(p‖m) + ½D_α(q‖m). That version is ≥ 0 for every α by nonnegativity of the Rényi divergence. It is a **different** object from the entropy-difference form R_α(m) − ½R_α(p) − ½R_α(q) used by Hamza–Krim and by MIL-CREDA. — [MetricGate doc (aggregator), snippet](https://metricgate.com/docs/jensen-renyi-divergence/)
- **Schur-concavity.** Sánchez Giraldo et al. prove only Schur-concavity of f_α(p) = (1/(1−α)) log Σp_i^α for α>0 (Lemma 4.1), and explicitly distinguish it from concavity. — [arXiv:1211.2459, fetched](https://arxiv.org/abs/1211.2459)
- **Ben-Bassat & Raviv (1978).** "Rényi's entropy and the probability of error", IEEE Trans. Inf. Theory 24(3):324–331. The usual citation for concavity for α ≤ 1 and binary concavity for α ≤ 2. — [not opened](https://doi.org/10.1109/TIT.1978.1055890)

### Inferences
1. **Exact counterexample to JR₂ ≥ 0 [derived, numerically checked].** Take p = (1,0,0), q = (3/4,1/8,1/8), and midpoint m = (7/8,1/16,1/16).
   - H₂(p) = 0.
   - H₂(q) = −log₂(38/64) = 0.7521.
   - H₂(m) = −log₂(198/256) = 0.3706.
   - JR₂ = 0.3706 − 0.3760 = **−0.0054 < 0**.
2. **Random search [derived, numerically checked].** JR₂ was never negative for k=2 (minimum about −3e−16, consistent with binary concavity at α ≤ 2). It reached −0.0053, −0.033 and −0.065 for k = 3, 5 and 20.
3. **The negativity transfers to MIL-CREDA [derived].** Commuting (diagonal) density matrices reproduce the classical case. A concrete realization:
   - All source mass of a class on one embedding x₀.
   - Target mass 3/4 on x₀ and 1/8 each on two embeddings orthogonal to x₀ and to each other (k ≈ 0, e.g. a narrow Gaussian).

   This gives J = −JR₂ = +0.0054 > 0. In this regime "J > 0" carries no alignment meaning.
4. **Quasi-concavity [derived].** For α>1, Σp^α is convex, so H_α is quasi-concave: H_α(m) ≥ min(H_α(p), H_α(q)). This is weaker than H_α(m) ≥ the average, which is what JR ≥ 0 needs.

### Gaps
- Neither the exact wording nor the theorem number of Ben-Bassat & Raviv's binary α ≤ 2 result was verified (primary not opened). Neither was Hamza–Krim's α range for JR ≥ 0. Treat "concave iff α ≤ 1, binary up to α ≤ 2" as standard but unverified in this run. The numerical results above are consistent with it.

## Q4. CREDA paper and known uses of matrix Rényi / CS divergence in domain adaptation and deep learning

### Takeaway
CREDA appeared in *Mathematics* (MDPI) 13(16):2602, 2025, by Pérez-Rosero, Álvarez-Meza and Castellanos-Domínguez. Its full equations could not be retrieved (MDPI returned 403). Closely related published work:
- CS divergence for domain adaptation (Yin, Yu et al., UAI 2024).
- RJSD with a domain-adaptation application (Hoyos-Osorio & Sánchez Giraldo, arXiv 2023–2024).
- Representation Jensen–Rényi (2021–2022).
- Multivariate matrix Rényi (TPAMI 2020).

### Cited Findings
- **CREDA bibliographic data.** "Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting". Diego Armando Pérez-Rosero, Andrés Marino Álvarez-Meza, German Castellanos-Dominguez. *Mathematics* 13(16), article 2602, 2025. — [IDEAS/RePEc record, fetched](https://ideas.repec.org/a/gam/jmathe/v13y2025i16p2602-d1724330.html)
- **CREDA's components** (from the abstract):
  - (i) a deep feature extractor for domain-invariant representations;
  - (ii) "an entropy-weighted approach that down-weights low-confidence pseudo-labels";
  - (iii) "a class-conditional alignment loss, formulated as a Rényi-based entropy kernel estimator".

  It is described as combining "kernel-based conditional alignment with a differentiable, matrix-based formulation of Rényi's quadratic entropy" (i.e. α=2). It is evaluated on Digits, ImageCLEF-DA and Office-31, with interpretability visualizations. — [search snippet + IDEAS abstract](https://ideas.repec.org/a/gam/jmathe/v13y2025i16p2602-d1724330.html)
- **Yin et al., UAI 2024 (PMLR v244).** Authors: Wenzhe Yin, Shujian Yu, Yicong Lin, Jie Liu, Jan-Jakob Sonke, Efstratios Gavves. The paper estimates marginal and conditional CS divergence between domains with kernel estimators and plugs them into both metric-based and adversarial domain adaptation. Code: github.com/ywzcode/CS-adv. — [PMLR, snippet](https://proceedings.mlr.press/v244/yin24a.html); [arXiv html, fetched](https://arxiv.org/html/2405.19978v1)
- **RJSD.** Evaluated on two-sample testing, distribution-shift detection and unsupervised domain adaptation, where it reportedly outperforms MMD-based baselines. — [arXiv:2305.16446, fetched](https://arxiv.org/html/2305.16446)
- **Related CS-divergence lines seen in search results (titles only):**
  - "The Conditional Cauchy-Schwarz Divergence with Applications to Time-Series Data and Sequential Decision Making" ([arXiv:2301.08970](https://arxiv.org/abs/2301.08970v3));
  - "Cauchy-Schwarz Divergence Information Bottleneck for Regression" ([arXiv:2404.17951](https://arxiv.org/pdf/2404.17951));
  - "Generalized Cauchy-Schwarz Divergence and Its Deep Learning Applications" ([arXiv:2405.04061](https://arxiv.org/pdf/2405.04061)).

  — snippets only

### Inferences
- MIL-CREDA's block-Gram construction is, structurally, the α=2 analog of RJSD's Prop. 8 estimator with equal weights. RJSD is the natural "proven" reference point to cite and possibly compare against.
- The DOI is very likely 10.3390/math13162602, following MDPI's pattern. This is inferred, not verified.

### Gaps
- CREDA's exact loss was not retrieved (MDPI page 403): the kernel, bandwidth rule, how the source/target Gram matrices are combined, the sign convention, the claimed theorems and the reported accuracies. Read the local PDF if available.
- Information-bottleneck uses of matrix Rényi were not fetched in this run. They are known from memory: Yu & Príncipe on autoencoders (Neural Networks 2019); Yu et al. on CNNs (IEEE TNNLS); the deep deterministic information bottleneck (ICASSP 2021).
- No published domain-adaptation paper was found that reports the negativity or entanglement issue of α=2 Jensen–Rényi described in Q2 and Q3.
