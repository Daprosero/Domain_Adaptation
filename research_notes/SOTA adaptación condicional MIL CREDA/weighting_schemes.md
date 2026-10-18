# Sample- and pseudo-label-weighting schemes vs. MIL-CREDA's confidence weights

Reference point throughout: MIL-CREDA weight per target bag
`w_j = 1 − H2(g_j)/ln C`, `H2(g) = −ln Σ_k g_k²`; used in (a) the global alignment term, gated per class by `Σ_j w_j`, and (b) the local term `Σ_j w_j ℓ_j / (Σ_j w_j + ε)`.

Verification status legend used below: **[code]** = checked against the official repository this run; **[paper]** = checked against the paper text this run; **[abs]** = only title/authors/abstract checked this run (formula given from the standard published form, not re-read this run).

## Comparison table

| Scheme | Formula (per sample) | Up-weights / down-weights | Detached? | Degenerate solution & how prevented | Used for |
|---|---|---|---|---|---|
| **MIL-CREDA w** | `w = 1 − H2(g)/ln C = 1 + ln(Σg²)/ln C = log_C(C·Σg²)` | Up: confident bags (w=1 at one-hot); down: near-uniform bags (w=0 at uniform) | To be decided | If differentiable: lower w ⇒ lower gated alignment & reweighted local loss ⇒ incentive toward uniform predictions | Alignment (global gate + local mean) |
| **CDAN+E** (Long 2018) [code] | `w = 1 + e^{−H(g)}`, H Shannon; weights normalized per domain by detached sums | Up: confident samples (w∈[1+1/C, 2]); never zero | **Not detached**: entropy carries a gradient-reversal hook `entropy.register_hook(grl_hook(coeff))` | Floor of 1 keeps every sample; GRL flips the weight gradient (see inference) | Alignment (adversarial discriminator loss) |
| **MCC** (Jin 2020) [code] | `w = 1 + e^{−H(ĝ)}`, ĝ temperature-scaled softmax; renormalized `w ← B·w/Σw` | Up: confident samples in class-confusion matrix | **Detached** explicitly (`.detach()` on the entropy) | Category normalization of the confusion matrix; detach | Regularizer on target predictions (classifier/discriminability) |
| **TADA** local (Wang 2019) [paper] | `w_i^k = 1 − H(d̂_i^k)`, residual `h = (1+w)·f` | Up: regions whose *domain discriminator* is certain | Not stated in paper | Residual `(1+w)` "to mitigate wrong local attention" | Alignment/features |
| **TADA** global [paper] | `m_i = 1 + H(d̂_i)` multiplying the per-sample entropy loss `−Σ_j m_i p_ij log p_ij` | Up: images the domain discriminator *cannot* tell apart (transferable) | Not stated in paper | Floor of 1 | Classification (entropy minimization) |
| **SHOT IM** (Liang 2020) [code] | `L = E[H(p)] − H(E[p])` (entropy minus diversity) | No per-sample weight; pushes every sample to certainty | Entropy **differentiable on purpose** (it is the objective) | Diversity term `H(mean p)` blocks collapse onto one class | Classification (feature extractor) |
| **SHOT pseudo-labels** [code] | Centroid clustering under `torch.no_grad()`; also computes `1 − H(p)/ln C` | Hard labels; low-count classes dropped by threshold | **Detached** (no_grad) | Fixed per-epoch labels, weight `cls_par=0.3` | Classification |
| **ATDOC-NA** (Liang 2021) [code] | `w = max_c (1/K Σ_{nbrs} m_c)`, memory of sharpened softmax `p²/Σ_batch p²`; loss `Σ w ℓ / Σw` | Up: samples whose K-neighbour aggregate is confident | **Detached**: memory filled under `torch.no_grad()`; denominator via `.item()` | Memory momentum, neighbour aggregation, sharpening with column normalization (class balance) | Classification (pseudo-label CE) |
| **BNM** (Cui 2020) [abs] | Maximize nuclear norm `‖A‖_*` of batch softmax matrix A | No per-sample weight | Differentiable (objective) | Nuclear norm couples discriminability and diversity, avoiding single-class collapse | Classification regularizer |
| **CBST** (Zou 2018) [abs] | Self-training; per-class confidence thresholds chosen so a portion p of each class is kept; p grows over rounds | Hard 0/1 selection; balanced across classes | Labels fixed between rounds (non-differentiable selection) | Class-balanced thresholds prevent easy-class domination; curriculum on p | Classification |
| **FixMatch** (Sohn 2020) [code] | `1[max q ≥ τ]·CE(argmax q, p(y|strong(u)))`, τ=0.95 | Hard mask on confident samples | **Detached** (`logits_x_ulb_w.detach()`) | High fixed threshold; weak/strong augmentation asymmetry | Classification |
| **FlexMatch** (Zhang 2021) [abs] | Class-wise thresholds adjusted over time by per-class learning status (curriculum) | Lowers threshold for hard classes | Mask non-differentiable | Per-class curriculum against class imbalance of pseudo-labels | Classification |
| **SoftMatch** (Chen 2023) [code] | `λ(p) = exp(−(min(max p − μ_t, 0))²/(2σ_t²/n_σ²))` — 1 above μ_t, truncated Gaussian below; μ_t, σ_t² EMA (m=0.999) | Soft down-weighting of low-confidence samples | **Detached** (`@torch.no_grad()` + `.detach()`) | Uniform alignment of class marginals; EMA stats | Classification |
| **UPS** (Rizve 2021) [abs] | Select pseudo-label only if confidence high **and** uncertainty (e.g. MC-dropout) low; also negative labels | Hard selection | Non-differentiable selection | Uncertainty gate reduces noisy confident labels | Classification |
| **Focal** (Lin 2017) [abs] | `FL = −(1−p_t)^γ log p_t` | **Down: easy** (high p_t); up: hard | Standard implementations do not detach the modulating factor (not verified this run) | Designed for class imbalance, not noise | Classification loss |
| **GCE / Lq** (Zhang & Sabuncu 2018) [abs] | `L_q = (1 − f_y^q)/q`; `∇L_q = f_y^q·∇CE`; q→0 CE, q=1 MAE | **Up: easy/confident**, down: hard (likely noisy) | Implicit weight inside the loss (differentiable) | Truncated Lq caps loss of very low f_y | Classification loss |
| **MAE** (Ghosh 2017) [abs] | `‖e_y − f‖₁ = 2(1 − f_y)`; gradient = `2 f_y ∇CE` | Up: easy; down: hard | Loss (differentiable) | Symmetric loss ⇒ noise-tolerant; underfits (slow on hard samples) | Classification loss |
| **SCE** (Wang 2019) [abs] | `αCE + β·RCE`, `RCE = −Σ_k f_k log y_k` with `log 0 := A` ⇒ `RCE = −A(1−f_y)` | CE part: hard; RCE part: easy | Loss | RCE's robustness + CE's fitting speed | Classification loss |
| **Taylor CE** (Feng 2020) | `Σ_{i=1}^t (1−f_y)^i / i`; t=1 MAE, t→∞ CE | Interpolates MAE↔CE | Loss | Truncation order t | Classification loss — **citation not verified this run** |
| **PolyLoss** (Leng 2022) [abs] | Poly-1: `CE + ε(1 − p_t)` | ε>0 adds MAE-like (easy) gradient | Loss | — | Classification loss |

## Entropy-based weights in DA (CDAN+E, SHOT, MCC, BNM, ATDOC, TADA/AFN, CBST)

### Takeaway
DA entropy weights are almost always of the form "certain ⇒ more weight", but they differ on detach: MCC and ATDOC detach explicitly, SHOT computes pseudo-labels under no_grad, while CDAN+E leaves the entropy differentiable but routes its gradient through a gradient-reversal hook. SHOT's code already computes the Shannon analogue of MIL-CREDA's weight, `1 − H(p)/ln C`.

### Cited Findings
- CDAN (Long, Cao, Wang; arXiv 1705.10667, NeurIPS 2018) conditions the adversary multilinearly on features and predictions — [arXiv API](http://export.arxiv.org/api/query?id_list=1705.10667)
- CDAN+E official code: `entropy.register_hook(grl_hook(coeff))` then `entropy = 1.0+torch.exp(-entropy)`; per-domain normalization `source_weight / torch.sum(source_weight).detach().item() + target_weight / torch.sum(target_weight).detach().item()`; final loss `torch.sum(weight.view(-1,1) * BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()` — [thuml/CDAN loss.py](https://raw.githubusercontent.com/thuml/CDAN/master/pytorch/loss.py)
- MCC official code: `target_entropy_weight = loss.Entropy(target_softmax_out_temp).detach()`; `1 + torch.exp(-w)`; `train_bs * w / torch.sum(w)`; `cov = (p*w)^T p`; row-normalized; `mcc_loss = (sum(cov) − trace(cov)) / class_num` — [thuml/Versatile-Domain-Adaptation](https://raw.githubusercontent.com/thuml/Versatile-Domain-Adaptation/master/pytorch/train_image_office.py); paper: Jin, Wang, Long, arXiv 1912.03699 — [arXiv API](http://export.arxiv.org/api/query?id_list=1912.03699)
- SHOT IM loss in code: `entropy_loss = mean(Entropy(softmax_out))`; `gentropy_loss = sum(−msoftmax·log msoftmax)`; `entropy_loss -= gentropy_loss`; `obtain_label` runs under `torch.no_grad()`, computes `unknown_weight = 1 - ent / np.log(args.class_num)`, two clustering iterations, class-count threshold; pseudo-label loss scaled by `cls_par` (default 0.3) — [tim-learn/SHOT image_target.py](https://raw.githubusercontent.com/tim-learn/SHOT/master/object/image_target.py); paper Liang, Hu, Feng, arXiv 2002.08546 — [arXiv API](http://export.arxiv.org/api/query?id_list=2002.08546)
- ATDOC-NA code: neighbour search on `features_target.detach()`; `weight_, pred = torch.max(w.mm(mem_cls), 1)`; `classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())`; memory written inside `torch.no_grad()` with `softmax_out**2 / ((softmax_out**2).sum(dim=0))`; ATDOC-NC uses unweighted CE to nearest centroid — [tim-learn/ATDOC demo_uda.py](https://raw.githubusercontent.com/tim-learn/ATDOC/main/demo_uda.py); paper arXiv 2007.04171 — [arXiv API](http://export.arxiv.org/api/query?id_list=2007.04171)
- TADA (Wang, Li, Ye, Long, Wang, AAAI 2019): local attention `w_i^k = 1 − H(d̂_i^k)` (Eq. 3), residual `h_i^k = (1 + w_i^k)·f_i^k` (Eq. 4) "more robust to wrong local attention"; global attention `m_i = 1 + H(d̂_i)` (Eq. 6) in attentive entropy `L_h = −(1/n) Σ_i Σ_j m_i p_ij log p_ij` (Eq. 7); motivation: forcing certainty on dissimilar (non-transferable) target images "may be harmful" — [AAAI paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/4472/4350)
- BNM (Cui et al., CVPR 2020, arXiv 2003.12237) maximizes the batch nuclear norm of the output matrix to improve both discriminability and diversity — [arXiv API](http://export.arxiv.org/api/query?id_list=2003.12237); [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Cui_Towards_Discriminability_and_Diversity_Batch_Nuclear-Norm_Maximization_Under_Label_Insufficient_CVPR_2020_paper.html)
- AFN (Xu et al., arXiv 1811.07456) adapts feature norms progressively to large values — not a sample-weighting scheme — [arXiv API](http://export.arxiv.org/api/query?id_list=1811.07456)
- CBST (Zou, Yu, Kumar; arXiv 1810.07911, ECCV 2018): iterative self-training with class-balanced pseudo-label generation — [arXiv API](http://export.arxiv.org/api/query?id_list=1810.07911)
- A 2025 MDPI *Mathematics* paper, "Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting", uses an entropy-derived confidence to down-weight low-confidence target pseudo-labels — [DOI 10.3390/math13162602](https://doi.org/10.3390/math13162602) (only the search snippet was readable; MDPI returned 403, so its formula and detach choice are unverified)

### Inferences
- **CDAN+E sign argument.** In CDAN the feature extractor maximizes the weighted discriminator loss (via the GRL). The entropy hook reverses the weight-path gradient too. So for the network, raising w (lowering entropy) raises the loss it is maximizing: the weight path rewards **certainty**. MIL-CREDA minimizes its alignment terms directly, with no adversary, so a differentiable w gets the opposite sign and rewards **uncertainty**. CDAN's choice not to detach is therefore no precedent for MIL-CREDA.
- ATDOC-NA's loss `Σ w ℓ / Σ w`, with w detached and the denominator detached, has the same shape as MIL-CREDA's local term. It is the closest published precedent, and it detaches both parts.
- CDAN+E and MCC add a floor: w ≥ 1 (`1 + e^{−H}`, ratio at most 2:1). MIL-CREDA's w can reach 0, so uncertain bags drop out. TADA mitigates the same issue with a residual `(1+w)`.
- TADA's weights use the entropy of the *domain discriminator*, not of the class posterior. They measure transferability, which is a different axis from MIL-CREDA's class-confidence weights.

### Gaps
- No paper text was found that states *why* CDAN puts a GRL hook on the entropy; the sign argument above is my own reading of the code.
- The detach choice for TADA is not stated in the paper, and no official code was located.
- SHOT computes `unknown_weight`; whether anything downstream uses it was not confirmed.
- MCC paper text (temperature value, rationale for detach) not re-read; only code checked.

## Robust losses: weighting or loss? Easy vs. hard

### Takeaway
Every robust loss here is a loss, not an external weight. Its gradient can be written as an implicit, differentiable per-sample weight on the CE gradient. Focal and CE emphasize hard samples. GCE, MAE, RCE and Poly-1 (ε>0) push weight toward easy, confident samples, which is the same direction as MIL-CREDA's w.

### Cited Findings
- Focal loss "down-weights the loss assigned to well-classified examples" (Lin, Goyal, Girshick; arXiv 1708.02002) — [arXiv API](http://export.arxiv.org/api/query?id_list=1708.02002)
- GCE/Lq (Zhang & Sabuncu, NeurIPS 2018): q→0 recovers CE, q=1 is MAE; has "weighted cross-entropy" characteristics; truncated variant — [arXiv 1805.07836](https://arxiv.org/abs/1805.07836)
- Ghosh, Kumar, Sastry (arXiv 1712.09482): a loss "based on mean absolute value of error is inherently robust to label noise" — [arXiv API](http://export.arxiv.org/api/query?id_list=1712.09482)
- SCE (Wang et al., arXiv 1908.06112, ICCV 2019): CE boosted "symmetrically with a noise robust counterpart Reverse Cross Entropy (RCE)" — [arXiv API](http://export.arxiv.org/api/query?id_list=1908.06112)
- PolyLoss (Leng et al., arXiv 2204.12511, ICLR 2022): losses as "a linear combination of polynomial functions" — [arXiv API](http://export.arxiv.org/api/query?id_list=2204.12511)

### Inferences
- Implicit weights relative to ∇CE = −∇log f_y: focal ≈ (1−p_t)^γ (ignoring the derivative of the factor); GCE = f_y^q; MAE = 2f_y; RCE = −A·f_y; Poly-1 = 1 + ε f_y. These follow from the standard formulas.
- GCE with weight f_y^q is the robust-loss analogue of a confidence weight. It is differentiable and still not degenerate, because the weight multiplies the CE gradient toward the *given label*, which raises f_y. A differentiable w on an *alignment* term has no such anchoring label.

### Gaps
- Taylor CE (Feng et al., IJCAI 2020, "Can cross entropy loss be robust to label noise?") was not verified this run.
- The exact symmetric-loss noise-rate condition from Ghosh 2017 was not re-read; only the abstract claim was.
- Whether common focal-loss implementations detach the modulating factor was not checked.

## Pseudo-label confidence schemes (FixMatch, FlexMatch, SoftMatch, UPS)

### Takeaway
All four compute the confidence signal from a stop-gradient copy of the predictions. They differ only in shape: a hard fixed threshold, class-adaptive thresholds, a truncated Gaussian, or a confidence plus uncertainty gate. The official USB implementation detaches in every case checked.

### Cited Findings
- FixMatch (Sohn et al., arXiv 2001.07685) pseudo-labels from weak-augmentation predictions with high-confidence filtering — [arXiv API](http://export.arxiv.org/api/query?id_list=2001.07685); USB code: `probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())`, mask from FixedThresholdingHook, `consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)` — [USB fixmatch.py](https://raw.githubusercontent.com/microsoft/Semi-supervised-learning/main/semilearn/algorithms/fixmatch/fixmatch.py)
- FlexMatch (Zhang et al., arXiv 2110.08263): "flexibly adjust thresholds for different classes at each time step" — [arXiv API](http://export.arxiv.org/api/query?id_list=2110.08263)
- SoftMatch (Chen et al., ICLR 2023) uses a truncated Gaussian to weight samples by confidence, addressing the "quantity-quality trade-off" of thresholding, plus uniform alignment — [arXiv 2301.10921](https://arxiv.org/abs/2301.10921). Code: `mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))`; EMA m=0.999; `update` and `masking` under `@torch.no_grad()`; `logits_x_ulb.detach()`; optional per-class stats — [USB softmatch/utils.py](https://raw.githubusercontent.com/microsoft/Semi-supervised-learning/main/semilearn/algorithms/softmatch/utils.py)
- UPS (Rizve et al., arXiv 2101.06329, ICLR 2021): uncertainty-aware pseudo-label selection — [arXiv API](http://export.arxiv.org/api/query?id_list=2101.06329)

### Inferences
- SoftMatch is the closest "soft" analogue to MIL-CREDA: a continuous weight in [0,1] that equals 1 above a running mean. It is detached, and it adds a marginal-alignment term to avoid class imbalance. MIL-CREDA's per-class gate `Σ w_j` plays a related role.
- Unlike SoftMatch's adaptive μ_t, MIL-CREDA's w has a fixed scale. Early in training, when predictions are near uniform, all w ≈ 0 and the gated terms switch off. SoftMatch's moving μ_t avoids this.

### Gaps
- FlexMatch's mapping function and warmup, and UPS's exact thresholds (τ_p, κ_p) were not re-read this run.
- The noise-robust DA pseudo-labelling literature beyond CBST/ATDOC/SHOT (e.g. CRST, Zou 2019) was not verified.

## Detaching: evidence for and against differentiable confidence weights

### Takeaway
Every pseudo-label weight checked in official code is detached: MCC, ATDOC, SHOT, FixMatch, SoftMatch. Differentiable entropy appears only (i) where entropy itself is the objective, with a diversity counterweight (SHOT IM, TADA's attentive entropy, BNM), or (ii) under a gradient reversal that flips the sign (CDAN+E). No checked source keeps a confidence weight differentiable on a directly minimized alignment loss.

### Cited Findings
- Detached: MCC (`.detach()`), ATDOC (`torch.no_grad()` memory, `.item()` denominator), SHOT pseudo-labels (`torch.no_grad()`), FixMatch (`.detach()`), SoftMatch (`@torch.no_grad()`) — sources as above.
- Differentiable but sign-flipped: CDAN+E entropy hook `grl_hook(coeff)` — [thuml/CDAN](https://raw.githubusercontent.com/thuml/CDAN/master/pytorch/loss.py)
- Differentiable as an objective with an anti-collapse term: SHOT IM (diversity `H(mean p)`) — [SHOT code](https://raw.githubusercontent.com/tim-learn/SHOT/master/object/image_target.py); BNM (nuclear norm couples diversity) — [arXiv 2003.12237](http://export.arxiv.org/api/query?id_list=2003.12237)

### Inferences
- **MIL-CREDA degeneracy, derived.** For the local term L = Σ w_j ℓ_j / (Σ w_j + ε), the gradient is ∂L/∂w_j = (ℓ_j − L)/(Σw + ε) (ignoring ε in L). The optimizer lowers w_j on every bag whose loss exceeds the weighted mean, and lowering w_j means making g_j more uniform. For the global term gated by Σ w_j, lowering all w uniformly shrinks the penalty. Both paths reward uncertainty. Detaching w (stop-gradient, as in MCC, ATDOC and SoftMatch) removes both.
- Alternatives if differentiability is wanted: (a) add a floor, `w' = 1 + w` as in CDAN+E/MCC/TADA; (b) add a SHOT-style diversity or IM term that rewards certainty; (c) put the weight path behind a GRL, as CDAN does.

### Gaps
- No primary source was found that writes out explicitly *why* it detaches confidence weights. The rationale above is derived, not quoted.

## Where MIL-CREDA's w sits; Gini / Tsallis-2 check

### Takeaway
Confirmed: 1 − Σg² is both the Gini impurity and the Tsallis entropy with q=2. But **MIL-CREDA's w is not linear in Σg²**. It is logarithmic: w = 1 + ln(Σg²)/ln C. Because H2 ≤ H1, w_Rényi2 ≥ 1 − H(g)/ln C, so it is more lenient than the Shannon weight SHOT computes.

### Cited Findings
- Tsallis entropy `S_q = k(1 − Σp_i^q)/(q−1)`; q=2 gives `k(1 − Σp_i²)` — [Wikipedia: Tsallis entropy](https://en.wikipedia.org/wiki/Tsallis_entropy)
- Gini impurity `I_G(p) = 1 − Σ p_i²` "corresponds to Tsallis Entropy with deformation coefficient q = 2" — [Wikipedia: Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
- Rényi order-2 (collision) entropy `H₂ = −log Σp_i²`; all orders equal log n at uniform; `log n = H₀ ≥ H₁ ≥ H₂ ≥ H_∞` — [Wikipedia: Rényi entropy](https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy)
- SHOT computes `1 - ent/np.log(class_num)` (Shannon normalized certainty) — [SHOT code](https://raw.githubusercontent.com/tim-learn/SHOT/master/object/image_target.py)

### Inferences
- Algebra: w = 1 − H2/ln C = 1 + ln(Σg²)/ln C = log_C(C·Σg²) = 1 + log_C(1 − Gini(g)). It is monotone in Σg² (and in Gini), but logarithmic. ∂w/∂g_k = 2g_k / (Σg² · ln C).
- The linear Gini/Tsallis-2 certainty would be w_lin = (C·Σg² − 1)/(C − 1) = 1 − Gini/Gini_max. Both equal 0 at uniform and 1 at one-hot. Since log_C(x) ≥ (x−1)/(C−1) on [1, C] (log is concave), w_Rényi2 ≥ w_lin. So the Rényi-2 weight is more lenient than linear Gini, and also more lenient than Shannon normalized certainty (from H2 ≤ H1).
- Ordering at fixed g: Rényi-2 (MIL-CREDA) ≥ Shannon-normalized (SHOT), and Rényi-2 ≥ linear Gini. So MIL-CREDA's w is the most lenient of the three. How Shannon compares with linear Gini was not established.
- Placement: w is a **soft, bounded [0,1], zero-at-uniform class-confidence weight on an alignment loss**. Closest in shape: SoftMatch (soft, [0,1]) and ATDOC-NA (max-prob weight in a normalized weighted mean). Closest in use: CDAN+E (an entropy weight on an alignment loss), which differs in having a floor of 1 and GRL sign reversal. Every close analogue detaches or sign-protects the weight.

### Gaps
- No DA paper using Tsallis-2/Gini or Rényi-2 normalized certainty as a sample weight was verified this run. The MDPI 2025 α-Rényi paper may be one, but it was unreadable (HTTP 403).
- The ordering between Shannon-normalized certainty and linear Gini certainty was not proven here.
