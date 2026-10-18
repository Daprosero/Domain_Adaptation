# Novelty check: MIL-CREDA candidate contributions (Idea I demoted, Idea II, Idea III)

Run date: 2026-10-10. Scope was changed mid-run by the coordinator. Idea I is reduced to a short "is it known" note plus the math flags found before the change. Idea II is kept. Idea III ("subject-personalized source reference") was added and checked quickly. The run was time-boxed (about 8 minutes for Idea III), so some primary sources could only be checked at abstract level. Each finding says which level it was checked at.

## Idea I: is the Rényi-2 = Cauchy–Schwarz identity known? (one-line verdict plus math flags)

### Takeaway
It is effectively known, and only the algebra is new. With uncentered covariance operators, ⟨C_P,C_Q⟩_HS is the kernel mean-embedding inner product under k². So the "CS divergence between covariance operators" is the classical ITL/Parzen CS divergence with a squared kernel. The cosine tr(ρσ)/√(trρ² trσ²) is the named "geometric-mean fidelity" F_GM in quantum information. RJSD already uses this exact operator framework (uncentered, trace-normalized, ½–½ mixture, MMD with k²). Novelty risk as a standalone contribution: **high**. It works as a lemma, not as a contribution.

### Cited Findings
- RJSD (Hoyos-Osorio & Sánchez-Giraldo) represents distributions by **uncentered covariance operators** with κ(x,x)=1, so the spectrum sums to 1. It defines representation Rényi entropy S_α(C_P)=1/(1−α) log Tr(C_P^α) and uses the **½–½ mixture** S((C_P+C_Q)/2) − ½(S(C_P)+S(C_Q)). It states that with uncentered operators the HS distance "is equivalent to the maximum mean discrepancy (MMD) with a squared kernel" (Lemma 5: D_JS ≥ ⅛‖C_P−C_Q‖²_HS = ⅛ MMD²_{κ²}). It gives **no α=2 or Cauchy–Schwarz treatment**. Its DA use is JAN-style joint-layer alignment and is **not class-conditional** [full HTML, v4, 23 Oct 2024; a NeurIPS 2024 listing appeared in the search results] — [arXiv 2305.16446v4](https://arxiv.org/html/2305.16446v4); [NeurIPS 2024 virtual](https://neurips.cc/virtual/2024/107791)
- Quantum fidelity review: the "geometric mean fidelity" is ℱ_GM(ρ,σ) := tr(ρσ)/√(tr(ρ²)tr(σ²)), "the Hilbert-Schmidt inner product normalized by the geometric mean of purities". There is also an arithmetic-mean fidelity ℱ_AM = 2tr(ρσ)/(tr ρ² + tr σ²) [full text, ar5iv] — Liang et al., Rep. Prog. Phys. 2019 — [arXiv 1810.08034](https://ar5iv.labs.arxiv.org/html/1810.08034)
- Yu, Li, Løkse, Jenssen, Príncipe extend the CS divergence to conditional distributions, estimated with KDE [abstract] — [arXiv 2301.08970](https://arxiv.org/abs/2301.08970v3)
- The earlier novelty note found that CS-DA (Yin et al., UAI-24, arXiv 2405.19978) aligns p(ŷ|z), not p(z|y) — see `creda_lineage_novelty.md` §4 in this folder (not re-fetched this run).

### Inferences
- [derived] D_CS(C_s,C_t) = −ln[ E_{s,t} κ² / √(E_{s,s}κ² · E_{t,t}κ²) ]. This is exactly the Jenssen-style Parzen CS estimator with kernel κ². e^{−D_CS} = ℱ_GM(ρ_s,ρ_t). The identity rewrites as e^{J_c} = ½cosh(ΔH/2)·(1 + ℱ_AM(ρ_s,ρ_t)), because ℱ_AM = ℱ_GM / cosh(ΔH/2).
- [derived] With the **linear** bag kernel, κ² = ⟨Ψ,Ψ'⟩² is a degree-2 polynomial kernel on Ψ. D_CS = 0 therefore only matches the uncentered second moment E[Ψ⊗Ψ] of the bag embeddings. It does not match the bag distributions. This is the gap Idea II fills.
- **Math flags on (a)–(d)** (checked numerically this run with numpy):
  - (a) is correct as stated (closed form = direct computation, 0.110751 both ways). It needs each block to be trace-normalized **before** the ½–½ mix, independently of n_s, n_t and the weights.
  - (b) The CREDA mix [[K_s, K_st],[K_stᵀ, D K_t D]] can indeed be indefinite: a 5+5 near-aligned example with D = 0.2·I gave a minimum eigenvalue of −3.85. **But "need not peak at alignment" also holds for the balanced J_c.** Example: ρ_s = e₁e₁ᵀ and ρ_t = 0.5·e₁e₁ᵀ + 0.5·I/100 give J_c = +0.111 > 0 = J_c(alignment). Maximizing raw J_c rewards purity imbalance. So (b) does not distinguish CREDA from the balanced mix. Only ℓ (or D_CS itself) peaks at alignment.
  - (c) "zero iff C_s = C_t" must read "iff ρ_s = ρ_t", i.e., C_s ∝ C_t. D_CS is scale-invariant, so the traces (mean ‖Ψ‖²) are not matched. D_CS = 0 forces ΔH = 0, so for fixed ΔH ≠ 0 the bound U = 2 ln cosh(ΔH/4) is a supremum. It is approached only in high or infinite dimension and never attained. L = −ln 2 is attained only when ΔH = 0. For ΔH ≠ 0 the infimum of J is ln(½cosh(ΔH/2)) > L, so ℓ < 1 always. The gradient claim holds under the stop-gradient on ΔH, and U is then constant. It is a surrogate gradient, not ∇ℓ.
  - (d) True but weak as a justification. α=2 is the only order at which the mixture's tr σ^α is bilinear in (ρ_s,ρ_t). This is the matrix analogue of why ITL uses quadratic Rényi with CS. It is a convenience argument, not an optimality argument.

### Gaps
- I did not find a paper that writes the exact J_c = log(½cosh(ΔH/2) + ½e^{−D_CS}) identity. The search was not exhaustive after the scope change.

## Idea II: subject-level conditional alignment with a characteristic two-level kernel

### Takeaway
The ingredients are all established: universal Gaussian kernels on mean embeddings (Christmann & Steinwart), two-level kernels on distributions in DG (Blanchard et al.; DICA), and the injectivity facts from Sriperumbudur et al. I found no prior work that uses a level-2 characteristic kernel for **class-conditional UDA over bags or subjects**. The identifiability statement is a short corollary of known theorems. It is correct only with extra assumptions, listed below. Novelty risk: **medium**. The DA application and the "instance-level CDD/LMMD cannot see set-structured shift" argument are new. A theory referee would call the identifiability result "a direct corollary of Christmann–Steinwart + Sriperumbudur et al."

### Cited Findings
- Christmann & Steinwart give a technique based on Taylor-type kernels that explicitly constructs universal kernels on **compact metric spaces**, including "universal kernels on the set of probability measures". NeurIPS 23 (2010), pp. 406–414 [abstract] — [NeurIPS 2010](https://proceedings.neurips.cc/paper/2010/hash/4e0cb6fb5fb446d1c92ede2ed8780188-Abstract.html)
- Sriperumbudur, Fukumizu & Lanckriet relate universal and characteristic kernels to RKHS embeddings of measures, and to strictly pd, conditionally strictly pd and **integrally strictly pd** kernels. For radial kernels on ℝ^d all these notions are equivalent. JMLR 12(70):2389–2410, 2011 [abstract] — [JMLR](https://jmlr.org/papers/v12/sriperumbudur11a.html)
- Blanchard, Deshmukh, Dogan, Lee & Scott, "Domain Generalization by Marginal Transfer Learning", JMLR 22 (2021) 1–55. They augment the feature space with the marginal distribution and give a universally consistent kernel algorithm, i.e., kernels on (P_X, x) [abstract] — [JMLR](https://www.jmlr.org/papers/v22/17-679.html)
- Muandet, Balduzzi & Schölkopf, DICA, ICML 2013. A kernel method that minimizes dissimilarity (distributional variance) **across domains**, using marginal P(X) and preserving P(Y|X) [abstract] — [PMLR v28](https://proceedings.mlr.press/v28/muandet13.html)
- Blast (Prabono, Yahya, Lee, FGCS 2022). The abstract-level description is a "deep architecture-based multiple instance domain adaptation model that learns domain-invariant latent representation". **The method is still unreadable.** Semantic Scholar has abstract = null and ScienceDirect is paywalled. No kernel-on-bags or class-conditional mechanism could be verified [metadata] — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X22000814); [S2 API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.future.2022.03.006?fields=title,abstract)
- Not re-verified this run (cited from the coordinator's brief and not fetched): Muandet et al. NeurIPS 2012 (SMM), Szabó et al. JMLR 2016 (distribution regression), Wang & Zhao WWW 2018.

### Inferences
- [derived] **Proof route.** C_s^{(2)} = E[φ₂(B)⊗φ₂(B)] with ⟨C_s,C_t⟩_HS = E κ₂(B,B')². κ₂² = exp(−‖Ψ−Ψ'‖²/τ²) is again a Gaussian on embeddings (width τ/√2), so Christmann–Steinwart universality applies to it. Cosine = 1 gives C_s = λC_t, which is the same as μ_{P_s} = λ μ_{P_t} in the RKHS of κ₂². On a compact domain, c-universality gives injectivity on **finite signed** measures (Sriperumbudur et al. 2011). So P_s − λP_t = 0, total mass gives λ = 1, and P_s = P_t. Characteristic alone, meaning injective on probability measures only, does **not** close the λ step. The signed-measure (ISPD / c-universal) version is required.
- **Extra assumptions II needs, which the claim currently omits:**
  1. **Compactness.** The domain of Ψ must be compact. This holds when attention weights are **normalized** (Σa=1, so Ψ is the embedding of a probability measure) and the instance space is compact. With unnormalized weights, Ψ embeds finite measures of varying mass and the set of bags is not the set of probability measures Christmann–Steinwart covers.
  2. **What is identified.** The result is the law of the **attention-weighted empirical measures** Σa_i δ_{x_i}, pushed through the learned encoder. It is not the law of raw bags. Attention is θ-dependent and can collapse onto one instance, which discards bag content. The statement is "equal laws of attention-weighted embedded measures".
  3. **Instance kernel.** Ψ must be injective on probability measures, which needs a characteristic instance kernel (Gaussian works). It also needs continuity of the outer kernel in the weak topology, which is part of the Christmann–Steinwart setting.
  4. **Conditioning.** The target class-conditional is the **pseudo-label-routed** one. Identifiability holds for the routed conditionals, not the true p(B|y). Confidence weights also reweight the empirical measures.
  5. As in Idea I, D_CS sees directions only. Here the λ = 1 step recovers scale because of total mass.
- **Referee attack lines:** "the identifiability theorem is a corollary of [Christmann–Steinwart 2010] + [Sriperumbudur et al. 2011]"; "level-2 kernels for distribution shift already exist in DG ([Blanchard et al. 2021], [DICA 2013])"; "the population guarantee says nothing about finite numbers of subjects per class". The last point is real: the empirical measure per class has few bags, so convergence is at the subject count n, not the instance count.

### Gaps
- Blast's method could not be read again. It is the residual threat for "bag-level DA".
- I found no explicit search hit for "class-conditional MMD on distributions of distributions for UDA". This is absence of evidence after limited queries.

## Idea III: subject-personalized source reference correcting hard routing

### Takeaway
The mechanism is a soft-posterior-weighted RKHS barycenter of source bags per target bag. It is a close cousin of three established patterns: OT barycentric mapping (each sample mapped to a coupling-weighted average of the other domain), prototype alignment with soft target-to-prototype assignment (PCT, ProDA), and similarity-weighted source-subject selection in cross-subject EEG. The specific combination is plausibly unoccupied: per-target-bag barycenter over **source bags (subjects)**, class mass taken from the **full soft posterior**, split within class by bag-kernel softmax, explicitly complementing a hard-routed global term. Novelty risk: **medium-high**. Expect a referee to call it "an entropic/softmax OT barycentric mapping (Courty et al.) restricted per class" or "PCT with subject-level prototypes".

### Cited Findings
- Courty, Flamary, Tuia & Rakotomamonjy, "Optimal Transport for Domain Adaptation", IEEE TPAMI 39(9), 2017. It learns a regularized transport plan and "estimates a barycentric mapping between feature distributions". Each sample is mapped to a plan-weighted average of the other domain's samples, with class regularization on source labels [abstract] — [arXiv 1507.00504](https://ar5iv.arxiv.org/html/1507.00504); [IP Paris portal](https://researchportal.ip-paris.fr/en/publications/optimal-transport-for-domain-adaptation/)
- Tanwisuth et al., "A Prototype-Oriented Framework for UDA" (PCT), NeurIPS 2021. It extracts class prototypes and aligns target features with them, as a memory- and computation-efficient alternative to sample-to-sample statistical distances [abstract; the abstract does not state the transport mechanism] — [arXiv 2110.12024](https://arxiv.org/abs/2110.12024); [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/8edd72158ccd2a879f79cb2538568fdc-Abstract.html)
- Cross-subject EEG uses subject-similarity weighting. Plug-and-Play DA (Zhao et al., AAAI 2021) uses "similarity of private components" to weight source subjects' classifiers. A subject-matching framework uses a reciprocal-exponential similarity to "dynamically select similar source subjects" and weight each source's output [search-result abstract level] — [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16169); [Li et al. 2021 PDF](https://personal.ntu.edu.sg/elpwang/PDF_web/2021_Li_Ruilin_Subject.pdf)
- Not verified this run because of the time box: DeepJDOT (Damodaran et al., ECCV 2018), ProDA (Zhang et al., CVPR 2021), CoWA-JMDS (Lee et al., ICML 2022), NRC (Yang et al., NeurIPS 2021), AaD (Yang et al., NeurIPS 2022).

### Inferences
- **Closest work.**
  - (1) OT barycentric mapping (Courty 2017) and DeepJDOT: same "map each target point to a weighted average of source points" form. Delta: the weights there come from a transport plan with marginal constraints and are global. Here they are a per-target factorized prior, posterior mass × within-class softmax, with no marginal (mass-conservation) constraint, over bag embeddings in the RKHS.
  - (2) PCT / ProDA: soft assignment of target to class prototypes. Delta: the references are **individual source subjects (bags)**, not one class prototype, so the reference is personalized. All C classes contribute through g_j^t rather than the argmax.
  - (3) EEG subject matching: similarity-weighted source subjects. Delta: those works weight **predictions/classifiers** at test time. Here the weights build a **feature-space alignment target** during training.
  - (4) [from memory, unverified] NRC/AaD pull target features toward **target** neighbours. CoWA reweights samples via a GMM. Neither builds a source-subject barycenter.
- [derived] **Math notes.**
  - d_j² is computable in closed form from Gram entries: κ(j,j) − 2Σπ_ji κ(j,i) + Σπ_ji π_ji' κ(i,i').
  - Without a mass constraint, a trivial minimizer exists: if the softmax is sharp, μ_j^s collapses to the nearest source bag, which is self-reinforcing nearest-neighbour matching. If g_j^t is wrong, the loss pulls toward the wrong class's subjects, which is the confirmation bias ProDA was designed to fight.
  - The "correcting hard routing" claim needs evidence that soft class mass gives a different (better) gradient than the hard global term in the misrouted cases. Otherwise it reads as label smoothing of the routing.
  - Π rows must sum to 1 (Σ_c g_jc = 1 and the within-class softmax sums to 1), so μ_j^s is a convex combination. Fine, but state it.
- **Referee attack lines:** "barycentric OT mapping without the transport constraint"; "PCT with per-subject prototypes"; "subject weighting is standard in cross-subject EEG DA"; "no protection against pseudo-label confirmation bias".

### Gaps
- ProDA, DeepJDOT, CoWA, NRC and AaD formulations were not re-read this run. Their deltas above come from known descriptions and should be checked against the papers before writing.
- I did not search for MIL-specific personalized references (bag-to-source-bag barycenters in MIL-DA). The time box ran out.
