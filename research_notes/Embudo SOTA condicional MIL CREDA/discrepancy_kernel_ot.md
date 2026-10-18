# Discrepancy-based DA (kernel/MMD, moments, OT, divergences) through the conditional / unlabelled-target / confidence-weighting funnel, positioned against MIL-CREDA

Run date: 2026-10-10. Companion to (and not a duplicate of) `../SOTA adaptación condicional MIL CREDA/class_conditional_kernel_da.md`, which already holds the verified JDA / CDD / LMMD / WMMD / JMMD / CPDA formulas; those are reused here by reference.

Verification legend:
- **[V-run]** venue/authors confirmed by a search or a full-text read in *this* run.
- **[V-prior]** confirmed in the prior note (same date, earlier run) and not re-fetched here.
- **[UNVERIFIED]** not confirmed by any fetched source.
- Math tags: **[read]** = equation read in the paper's full text in this run (PDF converted with pdftotext); **[prior-read]** = equation read in the prior run (source cited there); **[from memory]** = not read in any source this run.

Funnel (applied to every entry): **S0** strong/influential → **F1** class-conditional or joint P(Z,Y) alignment? → **F2** target unlabelled (pseudo-labels / clusters / soft predictions)? → **F3** pseudo-labelled target samples weighted or filtered by confidence/uncertainty? I split F3 into **F3-explicit** (threshold, entropy weight, reliability score, sample/class filter) and **F3-soft** (target contribution scaled only by the soft posterior p(c|x) itself, which the brief lists as "soft weights"). A method "survives" if it reaches F3 in either form; F3-explicit survivors are MIL-CREDA's real neighbours.

MIL-CREDA reference: bags, Ψ(B) = attention-weighted RKHS mean embedding; two encoders, source frozen; hard pseudo-label routing per class with w = 1 − H2(g)/ln C; per-class J_c = ½[H2(K_s)+H2(K_t)] − H2(K_mix) on trace-normalised Gram matrices with a ½–½ balanced mix (= log(½r + ½e^{−D_CS})); confidence gate on the class aggregation; local term = RKHS distance to a personalised source reference built from the full soft prediction.

---

## Q1. Where does each method leave the funnel?

### Takeaway
Of ~40 entries, only eight reach F3 with an **explicit** confidence mechanism: CAN/CDD (distance + class-size filtering), DCAN/CMMD (max-prob threshold 0.95), TPN (max-score threshold 0.6), RWOT (soft reliability score inside the OT cost), CREDA (entropy weighting, per abstract), HoMM's target-only clustering term (threshold η; but its cross-domain term is marginal, so HoMM itself exits at F1), and CPDA only when its optional threshold is switched on. LMMD/DSAN, CPDA (default) and PCT survive only through the **soft-posterior** clause. Every shallow JDA-family method (JDA, BDA, VDA, DICD, JGSA, ARTL, MEDA, DJP-MMD) passes F1–F2 and exits at F3 because all hard-pseudo-labelled target samples count equally. No surviving method combines hard per-class routing **and** a continuous entropy-based weight inside the same class-wise discrepancy, operates on MIL bags, or measures a covariance-operator (second-order) divergence per class.

### Cited Findings

**Funnel table.** Stage = last stage passed. "Exit reason" names the failed filter.

| # | Method | Family | Authors, venue, year | Verif. | Stage | Exit reason / survival mode |
|---|---|---|---|---|---|---|
| 1 | TCA | kernel MMD, shallow | Pan, Tsang, Kwok, Yang — IEEE TNN 22(2), 2011 | V-prior | S0 | F1: marginal P(X) only; SSTCA uses source labels only, no target-class conditioning. Conditional continuation = JDA (#2) |
| 2 | JDA | kernel MMD, shallow | Long, J. Wang, Ding, Sun, Yu — ICCV 2013 | V-run | F2 | F3: hard iterated pseudo-labels, all equally weighted |
| 3 | BDA / W-BDA | kernel MMD, shallow | J. Wang, Chen, Hao, Feng, Shen — ICDM 2017 | authors V-run (PDF); venue UNVERIFIED | F2 | F3: μ balances marginal vs conditional; W-BDA class weights are class-prior, not confidence |
| 4 | VDA | kernel MMD, shallow | Tahmoresnezhad & Hashemi — KAIS, 2016/2017 | V-prior | F2 | F3: hard pseudo-labels, no weighting [mechanics from memory] |
| 5 | DICD | kernel MMD + scatter | Li, Song, Huang, Ding, Wu — IEEE TIP 2018 | V-prior | F2 | F3: hard pseudo-labels, no weighting |
| 6 | JGSA | MMD + subspace geometry | J. Zhang, W. Li, Ogunbona — CVPR 2017, pp. 1859–1867 | V-run | F2 | F3: hard pseudo-labels, no weighting. **Two coupled projections (one per domain)** — closest shallow analogue to MIL-CREDA's two encoders |
| 7 | ARTL | MMD + manifold + classifier | Long, J. Wang, Ding, Pan, Yu — IEEE TKDE 26(5):1076–1089, 2014 | V-run | F2 | F3: hard pseudo-labels |
| 8 | MEDA | dynamic MMD + Grassmann | J. Wang, W. Feng, Y. Chen, H. Yu, et al. (remaining authors UNVERIFIED) — ACM MM 2018 | V-prior; first four authors in PDF this run | F2 | F3: adaptive factor μ weights marginal vs conditional, not samples |
| 9 | DJP-MMD | joint-probability MMD | W. Zhang & D. Wu — IJCNN 2020 | authors V-run (PDF); venue V-prior | F2 | F3: hard pseudo-labels |
| 10 | MIDA | HSIC independence | K. Yan, L. Kou, D. Zhang — IEEE T-Cyb 48(1):288–299, 2018 | V-run (search summary; primary page not opened) | S0 | F1: independence from *domain* features, no class conditioning |
| 11 | DAN (MK-MMD) | deep kernel MMD | Long et al. — ICML 2015 (full author list UNVERIFIED) | V-prior | S0 | F1: marginal multi-layer MMD. Conditional continuation = DSAN/LMMD (#16) |
| 12 | RTN | deep MMD + residual classifier | Long, Zhu, J. Wang, Jordan — NIPS 2016 | V-prior | S0 | F1: marginal (tensor) MMD; entropy minimisation is not alignment |
| 13 | JAN / JMMD | deep joint kernel MMD | Long, Zhu, J. Wang, Jordan — ICML 2017 | V-run (PDF header) | F2 | F3: soft predictions enter as one factor of a product kernel; no per-sample weight |
| 14 | WMMD (+ class-specific MMD) | deep weighted MMD | H. Yan, Ding, P. Li, Q. Wang, Y. Xu, Zuo — CVPR 2017 | V-run (PDF header) | F2 | F1 weakly (class-wise prior reweighting α_c); F3: weights are class priors, not confidence. Class-specific MMD journal variant venue UNVERIFIED |
| 15 | CAN / CDD | deep class-conditional kernel | Kang, Jiang, Y. Yang, Hauptmann — CVPR 2019 | V-run (PDF header) | **F3-explicit** | survives: sample filter dist < D0, class filter count > N0 |
| 16 | DSAN / LMMD | deep class-conditional kernel | Zhu, Zhuang, J. Wang, Ke, Chen, Bian, Xiong, He — IEEE TNNLS (arXiv 2021) | V-prior | **F3-soft** | survives via soft-posterior weights (detached in official code) |
| 17 | DCAN / CMMD | conditional embedding operators | Ge, Ren, Dai, H. Yan — Pattern Recognition (PII S0031320322005684) | venue V-prior; year UNVERIFIED (2022/2023) | **F3-explicit** | survives: max-prob > γ0 = 0.95 selects pseudo-labelled targets |
| 18 | TPN | prototype RKHS discrepancy | Pan, Yao, Li, Wang, Ngo, Mei — CVPR 2019 | V-run | **F3-explicit** | survives: pseudo-labels only when max score > 0.6, resampled each iteration |
| 19 | CPDA | class-conditional path-signature MMD | Ott & Mutschler — arXiv 2608.09193, Aug 2026 (preprint) | V-prior; PDF read this run | **F3-soft** (F3-explicit only if τ > 0) | detached soft-posterior weights by default; optional confidence threshold τ, default 0.0 |
| 20 | CORAL | 2nd-order moments | Sun, Feng, Saenko — AAAI 2016 | V-prior | S0 | F1: marginal covariances. No class-conditional CORAL variant verified this run |
| 21 | Deep CORAL | 2nd-order moments, deep | Sun & Saenko — ECCV 2016 **Workshops** (TASK-CV), DOI 10.1007/978-3-319-49409-8_35 | V-run (search summary) | S0 | F1: marginal |
| 22 | CMD | central moments | Zellinger et al. — ICLR 2017 | V-prior | S0 | F1: marginal |
| 23 | HoMM | higher-order moments | C. Chen, Fu, Z. Chen, Jin, Cheng, X. Jin, Hua — AAAI 2020 | V-run | S0 | F1: cross-domain term is marginal. Note: its *target-only* discriminative-clustering term uses a confidence threshold η |
| 23b | COT (class-aware OT + class-aware HMM) — class-conditional variant of HoMM-style matching | class-aware OT + moments | T. Nguyen, V. Nguyen, Le, Zhao, Tran, Phung — arXiv 2401.15952, Jan 2024 | V-run (arXiv); peer-reviewed venue UNVERIFIED | F2 | F3: amortised soft transport net T trained with entropy terms; no confidence weighting of samples |
| 24 | JDOT | joint OT | Courty, Flamary, Habrard, Rakotomamonjy — NIPS 2017 | V-run (PDF) | F2 | F3: predicted labels f(x_t) enter the cost, no sample weighting |
| 25 | DeepJDOT | joint OT, deep | Damodaran, Kellenberger, Flamary, Tuia, Courty — ECCV 2018 | V-run (PDF) | F2 | F3: same as JDOT |
| 26 | ETD | attention-weighted OT | M. Li, Zhai, Luo, Ge, Ren — CVPR 2020 | V-run (PDF) | F2 | F3 (borderline): pair weights come from a learned attention on predictions, not from confidence; entropy is a regulariser |
| 27 | RWOT | reliability-weighted OT | R. Xu, P. Liu, L. Wang, C. Chen, J. Wang — CVPR 2020 | V-run (PDF) | **F3-explicit** | survives: SSR reliability Q(j,c) scales the OT cost |
| 28 | PCT | prototype conditional transport | Tanwisuth, Fan, Zheng, S. Zhang, H. Zhang, B. Chen, Zhou — NeurIPS 2021 | V-run (PDF) | **F3-soft** | survives via soft transport probabilities π(μ_k|f) with estimated class priors; no explicit confidence filter |
| 29 | MARS (OT for conditional matching + label shift) | weighted Wasserstein + proportion estimation | Rakotomamonjy, Flamary, Gasso, El Alaya, Berar, Courty — Machine Learning (Springer), DOI 10.1007/s10994-021-06088-2 | authors V-run (arXiv PDF); journal per ACM DL listing; year UNVERIFIED (2022) | F2 | F1 weakly (class-proportion reweighting of the marginal Wasserstein); F3: no confidence weighting |
| 30 | CS-divergence DA | kernel CS + conditional CS | Yin, S. Yu, Lin, Liu, Sonke, Gavves — UAI 2024 | V-prior; PDF read this run | F2 | F3: soft predictions enter a Gram matrix L; no sample weighting |
| 31 | CREDA | matrix Rényi-2 class-conditional | Pérez-Rosero, Álvarez-Meza, Castellanos-Dominguez — Mathematics 13(16):2602, 2025 | V-prior (abstract only) | **F3-explicit** (per abstract) | entropy-weighted pseudo-labels; full text blocked (403) this run |
| 32 | BSP | spectral penalty | X. Chen, S. Wang, Long, J. Wang — ICML 2019, PMLR 97:1081–1090 | V-run | S0 | F1: penalises top singular values of each domain's feature batch; no class conditioning (used on top of DANN/CDAN) |
| 33 | Prototype-oriented class-conditional clustering transport | prototype OT | Sci. Rep. 2025, s41598-025-21560-y | title/venue from search only | UNVERIFIED | page behind redirect; content not read |

Sources for the table, in order of rows:
- TCA — [IEEE Xplore](https://ieeexplore.ieee.org/document/5640675)
- JDA — [mlanthology ICCV 2013](https://mlanthology.org/iccv/2013/long2013iccv-transfer)
- BDA — [arXiv 1807.00516](https://arxiv.org/pdf/1807.00516)
- VDA — [Unpaywall DOI](https://unpaywall.org/10.1007%2FS10115-016-0944-X)
- DICD — [OpenAlex](https://api.openalex.org/works/doi:10.1109%2FTIP.2018.2839528)
- JGSA — [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Joint_Geometrical_and_CVPR_2017_paper.html), [arXiv 1705.05498](https://arxiv.org/pdf/1705.05498)
- ARTL — [Tsinghua PDF](https://ise.thss.tsinghua.edu.cn/mig/Adaptation%20Regularization%20A%20General%20Framework%20for%20Transfer%20Learning.pdf) (TKDE vol/pages from the search summary)
- MEDA — [arXiv 1807.07258](https://arxiv.org/pdf/1807.07258)
- DJP-MMD — [arXiv 1912.00320](https://arxiv.org/abs/1912.00320)
- MIDA — [arXiv 1603.04535](https://www.arxiv.org/pdf/1603.04535)
- DAN, RTN — [DSAN refs](https://ar5iv.arxiv.org/html/2106.09388), [NeurIPS RTN](https://papers.nips.cc/paper_files/paper/2016/hash/ac627ab1ccbdb62ec96e702f07f6425b-Abstract.html)
- JAN — [arXiv 1605.06636](https://arxiv.org/abs/1605.06636)
- WMMD — [arXiv 1705.00609](https://arxiv.org/pdf/1705.00609), [class-specific MMD (ResearchGate)](https://www.researchgate.net/publication/337237428_Weighted_and_Class-Specific_Maximum_Mean_Discrepancy_for_Unsupervised_Domain_Adaptation)
- CAN — [arXiv 1901.00976](https://arxiv.org/pdf/1901.00976)
- DSAN — [arXiv 2106.09388](https://arxiv.org/pdf/2106.09388), [official code lmmd.py](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/DeepDA/loss_funcs/lmmd.py)
- DCAN — [arXiv 2006.07776](https://arxiv.org/pdf/2006.07776), [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320322005684)
- TPN — [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Pan_Transferrable_Prototypical_Networks_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html), [arXiv 1904.11227](https://arxiv.org/pdf/1904.11227)
- CPDA — [arXiv 2608.09193](https://arxiv.org/pdf/2608.09193)
- CORAL — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/10306)
- Deep CORAL — [mlanthology](https://mlanthology.org/eccv/2016/sun2016eccv-deep/), [arXiv 1607.01719](https://www.arxiv.org/abs/1607.01719)
- CMD — [mlanthology](https://mlanthology.org/iclr/2017/zellinger2017iclr-central)
- HoMM — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/5745), [arXiv 1912.11976](https://arxiv.org/pdf/1912.11976)
- COT — [arXiv 2401.15952](https://arxiv.org/pdf/2401.15952)
- JDOT — [arXiv 1705.08848](https://arxiv.org/pdf/1705.08848)
- DeepJDOT — [arXiv 1803.10081](https://arxiv.org/pdf/1803.10081), [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.html)
- ETD — [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Enhanced_Transport_Distance_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf), [mlanthology](https://mlanthology.org/cvpr/2020/li2020cvpr-enhanced-a/)
- RWOT — [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf), [mlanthology](https://mlanthology.org/cvpr/2020/xu2020cvpr-reliable/)
- PCT — [arXiv 2110.12024](https://arxiv.org/pdf/2110.12024)
- MARS — [arXiv 2006.08161](https://arxiv.org/abs/2006.08161v3), [ACM DL / Machine Learning](https://dl.acm.org/doi/abs/10.1007/s10994-021-06088-2)
- CS-DA — [arXiv 2405.19978](https://arxiv.org/pdf/2405.19978), [mlanthology UAI 2024](https://mlanthology.org/uai/2024/yin2024uai-domain/)
- CREDA — [RePEc/IDEAS](https://ideas.repec.org/a/gam/jmathe/v13y2025i16p2602-d1724330.html)
- BSP — [PMLR](https://proceedings.mlr.press/v97/chen19i.html)
- Sci. Rep. 2025 — [Nature](https://www.nature.com/articles/s41598-025-21560-y)

Correction to the brief: the PCT in the list is **Prototype-oriented Conditional Transport**, NeurIPS 2021, not 2022, per the paper's own footer "35th Conference on Neural Information Processing Systems (NeurIPS 2021)" — [arXiv 2110.12024](https://arxiv.org/pdf/2110.12024). Also, arXiv 2006.08161 (which a search surfaced for "RWOT") is a *different* paper (Rakotomamonjy et al., MARS); RWOT itself is on CVF only — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf).

### Inferences
- The funnel is mostly decided at F3: F1+F2 are passed by every pseudo-label method; what separates MIL-CREDA's neighbours is an explicit reliability mechanism. The F3-explicit set (CDD, DCAN, TPN, RWOT, CREDA) uses three styles: **binary filters** (CDD's D0/N0, DCAN's γ0, TPN's 0.6), **continuous reliability scores** (RWOT's Q), **entropy weights** (CREDA; MIL-CREDA). MIL-CREDA uses both a continuous entropy weight and a binary-like gate on class aggregation, which matches no single precedent.
- JGSA (two coupled, domain-specific projections) and PCT (prototypes held fixed during alignment) are the only discrepancy methods found that are asymmetric in the way MIL-CREDA's frozen source encoder is; neither combines this with confidence weighting.

### Gaps
- BDA ICDM 2017 venue, DCAN publication year, MARS publication year, COT peer-reviewed venue, class-specific-MMD journal venue were not confirmed.
- CREDA full text (encoder layout, exact weight formula) is blocked (MDPI 403 on WebFetch and curl); a local `CREDA/` implementation folder exists in this repo but no Python file containing the loss was found in a shallow search.
- The Sci. Rep. 2025 prototype class-conditional clustering transport paper could not be opened.
- Not searched in this run: JUMBOT, m-POT, other mini-batch-OT DA variants; 2025–2026 class-conditional OT conference papers beyond COT.

---

## Q2. Core math, pseudo-label type, weighting and MIL extensibility, per method

### Takeaway
All kernel class-conditional methods compute some weighted ‖μ_{s,c} − μ_{t,c}‖²_H (JDA, CDD, LMMD, TPN, CPDA) or an operator analogue (DCAN); OT methods put the class information into the transport cost (JDOT/DeepJDOT via predicted-label loss, RWOT via reliability (1−Q), ETD via attention); divergence methods (CS-DA, CREDA) work with log-ratios of Gram sums. Only LMMD (in official code), CPDA (by default) and PCT (prototypes) are documented as detaching weights or references. Every method extends to MIL bags by replacing φ(x) with Ψ(B); what breaks is method-specific (explicit-feature moment methods, Grassmann steps, OT batch size).

### Cited Findings

**A. Shallow kernel MMD (JDA family).** Shared template [prior-read, via Zhuang et al. survey]: min_A Σ_{c=0}^{C} ‖(1/n_c)Σ_{x∈D_s^{(c)}} Aᵀx − (1/m_c)Σ_{x∈D_t^{(c)}} Aᵀx‖², c = 0 the marginal, D_t^{(c)} = targets with **hard** pseudo-label c, labels refreshed by retraining a classifier each iteration — [arXiv 1911.02685](https://arxiv.org/pdf/1911.02685). *Plain language:* distance between the class-c average of the projected source points and the class-c average of the projected target points that the current classifier calls c.
- **TCA**: tr(KLK) − μ tr(K) type objective over the joint kernel [from memory]; marginal only; MIL: bag kernel ⟨Ψ,Ψ'⟩ plugs in directly — [IEEE](https://ieeexplore.ieee.org/document/5640675).
- **JDA**: template above; hard labels, no weights; MIL: per-class MMD on bag embeddings is direct but the projection must act on the bag kernel (kernelised JDA) — [mlanthology](https://mlanthology.org/iccv/2013/long2013iccv-transfer).
- **BDA**: (1−μ)·MMD_marg + μ·Σ_c MMD_c, μ∈[0,1]; JDA = equal weights; W-BDA adds class-prior weights for imbalance [prior-read] — [arXiv 1807.00516](https://arxiv.org/pdf/1807.00516). Hard labels.
- **VDA / DICD**: JDA template plus intra-class compactness (VDA) or inter-class dispersion / intra-class scatter (DICD) [from memory for VDA; DICD's two scatter terms prior-read]; hard labels — [DICD record](https://research.buaa.edu.cn/en/publications/domain-invariant-and-class-discriminative-feature-learning-for-vi/). MIL: scatter matrices need explicit vectors → use Nyström features of Ψ(B).
- **JGSA**: two projections A (source), B (target); objective combines target variance, source between/within-class scatter, marginal + conditional MMD, and subspace divergence ‖A−B‖²_F, solved in closed form [MMD+scatter structure from abstract (V-run); exact terms from memory] — [arXiv 1705.05498](https://arxiv.org/pdf/1705.05498). Hard labels. MIL: two domain-specific projections mirror MIL-CREDA's two encoders; the ‖A−B‖ coupling corresponds to *not* freezing either side.
- **ARTL**: min_f Σ ℓ(f(x_i),y_i) + σ‖f‖²_K + λ·D_{f,K}(J_s,J_t) + γ·M_{f,K}(manifold) [structure V-run from TKDE summary; symbols from memory] — [Tsinghua PDF](https://ise.thss.tsinghua.edu.cn/mig/Adaptation%20Regularization%20A%20General%20Framework%20for%20Transfer%20Learning.pdf). Hard labels. MIL: the kernel classifier accepts a bag kernel directly.
- **MEDA**: ARTL-style classifier on Grassmann (GFK) features with D_f = (1−μ)D_marg + μΣ_c D_c, μ estimated from 𝒜-distances [μ ∈ [0,1] and class-wise sum read this run; 𝒜-distance estimator from memory] — [arXiv 1807.07258](https://arxiv.org/pdf/1807.07258). MIL: GFK needs vector subspaces → breaks on infinite-dimensional Ψ unless approximated.
- **DJP-MMD**: discrepancy on the joint P(X,Y) directly (transferability) minus an inter-class joint discrepancy (discriminability), JDA-style hard labels [prior-read] — [arXiv 1912.00320](https://arxiv.org/abs/1912.00320).
- **MIDA**: max tr(KHK_dH)-type HSIC between projected features and *domain* features [from memory] — [arXiv 1603.04535](https://www.arxiv.org/pdf/1603.04535). MIL: HSIC on bag kernel is direct.

**B. Deep kernel MMD.**
- **DAN**: Σ_l MK-MMD²(D_s^l, D_t^l) with a convex combination of Gaussian kernels [from memory]; marginal. **RTN**: tensor-product MMD of fused layers + residual target classifier + target entropy [from memory].
- **JAN**: JMMD = (1/n_s²)ΣΣ Π_l k^l(z_i^{sl},z_j^{sl}) + (1/n_t²)ΣΣ Π_l k^l(z_i^{tl},z_j^{tl}) − (2/(n_s n_t))ΣΣ Π_l k^l(z_i^{sl},z_j^{tl}), with the softmax layer as one l [prior-read] — [arXiv 1605.06636](https://arxiv.org/abs/1605.06636). *Plain:* MMD on the joint of features and predictions, through a product kernel. Soft, unweighted. MIL: product of bag kernel × prediction kernel is natural.
- **WMMD**: ‖(1/Σα_{y_i})Σ_i α_{y_i^s}φ(x_i^s) − (1/N)Σ_j φ(x_j^t)‖², α_c = ŵ_c^t/w_c^s from hard CEM labels [prior-read] — [arXiv 1705.00609](https://arxiv.org/pdf/1705.00609). *Plain:* reweights source classes so the source mean has the target's class mix. MIL: reweight source bags by α_c.
- **CAN/CDD**: D^{cdd} = (1/M)Σ_c D^{cc} − (1/(M(M−1)))Σ_c Σ_{c'≠c} D^{cc'}, D^{c1c2} = e1+e2−2e3 over label masks [prior-read]. Target labels: spherical k-means initialised at source class centres, cosine dist(a,b) = ½(1 − ⟨a,b⟩/(‖a‖‖b‖)); keep (x_t,ŷ_t) only if dist(φ(x_t), O^{t(ŷ)}) < D0, and keep class c only if its count > N0 [read] — [arXiv 1901.00976](https://arxiv.org/pdf/1901.00976). *Plain:* pull same-class source and target kernel clouds together, push different classes apart, using only confidently clustered targets. Hard labels; binary weights; labels come from an offline clustering step, so they carry no gradient by construction. MIL: spherical k-means on Ψ(B) needs a cosine geometry in the bag-embedding space; class-aware bag sampling.
- **DSAN/LMMD**: d̂ = (1/C)Σ_c ‖Σ_i w_i^{sc}φ(x_i^s) − Σ_j w_j^{tc}φ(x_j^t)‖², w_j^{tc} = ŷ_{jc}/Σ_k ŷ_{kc} (soft) [prior-read]. **Official code**: `cal_weight` converts logits with `.cpu().data.numpy()` → weights are **detached**; additionally only classes present both in the source labels and in the target **argmax** set enter the sum, averaged over that count [read] — [lmmd.py](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/DeepDA/loss_funcs/lmmd.py). MIL: direct on Ψ(B); batches must hold enough bags per class.
- **DCAN/CMMD**: L_CMMD = Tr(K_s L̃_s^{-1} L_s L̃_s^{-1}) + Tr(K_t L̃_t^{-1} L_t L̃_t^{-1}) − 2Tr(L_ts L̃_s^{-1} K_st L̃_t^{-1}), L̃ = L + λI, K on features, L on labels [read; first terms reconstructed from the fetched eq. (6) layout and the matching CS-DA Remark 2 form]. Targets: "If the maximum value of ŷ_i^t is larger than a threshold γ0, we consider this sample as high confidence and then assign this sample corresponding pseudo-label", γ0 = 0.95 [read]; plus mutual-information loss with a class-balance threshold γ1 = 1.5 on H(P(Ŷ_t)) [read] — [arXiv 2006.07776](https://arxiv.org/pdf/2006.07776). *Plain:* compares the conditional embedding operators Z|Y of the two domains (how features vary with label), only on confident targets. Hard, thresholded. MIL: needs (L+λI)^{-1} per batch of bags — fine with a bag kernel.
- **TPN**: prototypes μ_c^s, μ_c^t, μ_c^{st} (source-only, pseudo-labelled target-only, union); L_G = (1/C)Σ_c [‖μ̃_c^s − μ̃_c^t‖²_H + ‖μ̃_c^s − μ̃_c^{st}‖²_H + ‖μ̃_c^t − μ̃_c^{st}‖²_H] ("multi-granular"; first term read, the other two from the "pairwise across three prototypes" description) plus symmetric KL between the three prototype-classifier score distributions per sample [read]. Pseudo-labels: nearest source prototype, assigned **only if the maximised score > 0.6**, resampled every iteration [read] — [arXiv 1904.11227](https://arxiv.org/pdf/1904.11227). Hard, thresholded. MIL: prototypes become class means of Ψ(B) — direct.
- **CPDA**: D² = Σ_c π_t(c)‖μ̂_{s,c} − μ̂_{t,c}‖² with soft target weights [prior-read]; config table: "Detach target weights: Enabled", "Pseudo-label confidence threshold τ: 0.0" [read]; risk bound R_t(h) ≤ R_s^{π_t}(h) + B Σ_c π_t(c) MMD_CPDA(P_s^c, P_t^c) [read] — [arXiv 2608.09193](https://arxiv.org/pdf/2608.09193). MIL: swap the path-signature kernel for a bag kernel.

**C. Moment matching.**
- **CORAL / Deep CORAL**: (1/4d²)‖C_S − C_T‖²_F [from memory] — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10306). **CMD**: ‖E(X)−E(Y)‖/|b−a| + Σ_{k≥2} ‖c_k(X)−c_k(Y)‖/|b−a|^k [from memory] — [mlanthology](https://mlanthology.org/iclr/2017/zellinger2017iclr-central). Both need explicit finite features → on bags, use a finite bag feature (attention-pooled instance features) rather than Ψ in the RKHS.
- **HoMM**: L_d = (1/L^p)‖(1/n_s)Σ_i φ_θ(x_s^i)^{⊗p} − (1/n_t)Σ_i φ_θ(x_t^i)^{⊗p}‖²_F [read]; target-only clustering L_dc = (1/n_t)Σ_i ‖h_t^i − c_{ŷ_t^i}‖²₂ on targets whose predicted probability exceeds η, enabled after warm-up [read] — [arXiv 1912.11976](https://arxiv.org/pdf/1912.11976). *Plain:* matches p-th order tensor moments; the conditional part only tightens target clusters.
- **COT**: OT between a distribution over source class-conditionals Q_m^S and the data mixture; cost c(G(x), Q_m^S) = −log p_m^S(x) from an (M+1)-way discriminator; amortised plan via transport net T: min Σ_i Σ_m T_m(G(x_i)) c(G(x_i), Q_m^S); target entropy terms L_ent = (1/N_T)Σ H(T(G(x_i))) − H((1/N_T)Σ T(G(x_i))) [read] — [arXiv 2401.15952](https://arxiv.org/pdf/2401.15952). Soft (T), no confidence weight.

**D. Optimal transport.**
- **JDOT**: γ* = argmin_γ Σ_{ij} D(x_i^s, y_i^s; x_j^t, f(x_j^t)) γ_ij, D = α d(x_1,x_2) + L(y_1,y_2) [read] — [arXiv 1705.08848](https://arxiv.org/pdf/1705.08848). *Plain:* move source points to target points paying both feature distance and the classifier's loss at the target point against the source label.
- **DeepJDOT**: min (1/n_s)Σ L_s(y_i^s, f(g(x_i^s))) + Σ_{ij} γ_ij [α‖g(x_i^s) − g(x_j^t)‖² + λ_t L_t(y_i^s, f(g(x_j^t)))]; alternating: "With fixed CNN parameters (ĝ, f̂) … solve for γ", then "With fixed coupling γ̂ … update" [read] — [arXiv 1803.10081](https://arxiv.org/pdf/1803.10081). Plan γ is therefore not differentiated through. Soft predictions in the cost; no confidence weighting. MIL: cost on ‖Ψ(B_s) − Ψ(B_t)‖²_H is direct; O(b²) mini-batch plan with few bags per batch → high mini-batch-OT bias.
- **ETD**: C_ij = c(f_i^s, f_j^t); attention S_ij = σ((W_a^s η(f_i^s))ᵀ(W_a^t η(f_j^t))), softmax-normalised over the target batch; reweighed C = S ⊙ C; Kantorovich potential parameterised by a 3-layer network g, semi-dual sup E[g^{c,ε}(f^s)] + E[g(f^t)] − ε; plus target entropy loss [read] — [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Enhanced_Transport_Distance_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf). *Plain:* entropic OT whose pair costs are reweighted by a learned similarity of the two samples' predictions. Soft; no confidence weight.
- **RWOT**: D(i,k) = e^{−d(G_f(x_i^t), c_k^s)}/Σ_m e^{−d(G_f(x_i^t), c_m^s)} with multi-kernel d (MMD-like distance to source class centres); M(i,k) = P(y=k | Softmax(G_y(G_f(x_i^t))/τ)); **SSR** Q(i,k) = [d_A(k) D(i,k) + (2 − d_A(k)) M(i,k)] / Σ_m[…], d_A(k) = 2(1 − 2ε(h_k)) the class-wise 𝒜-distance; cost Z(i,j) = ‖G_f(x_i^s) − G_f(x_j^t)‖² · (1 − Q(j, y_i^s)); L_g = Σ γ*_ij(‖G_f(x_i^t) − G_f(x_j^s)‖² + F1(Softmax(G_y(G_f(x_i^t))), y_j^s)) [read] — [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf). *Plain:* a pair is cheap to transport only if the target sample is reliably of the source sample's class, where reliability blends prototype distance (trusted early) and classifier output (trusted once domains overlap). Soft reliability; explicit; detaching not stated in the paper.
- **PCT**: π_θ(μ_k|f_j^t) = p(μ_k)exp(μ_kᵀf_j^t)/Σ_{k'} p(μ_{k'})exp(μ_{k'}ᵀf_j^t); π_θ(f_j^t|μ_k) = exp(μ_kᵀf_j^t)/Σ_{j'} exp(μ_kᵀf_{j'}^t); L = L_cls + L_{t→μ} + L_{μ→t}, cost c = cosine dissimilarity; class proportions p(μ_k) estimated by an EM-like update; "we treat μ as fixed in both L_{t→μ} and L_{μ→t}" [read] — [arXiv 2110.12024](https://arxiv.org/pdf/2110.12024). *Plain:* bidirectional soft transport between target features and source class prototypes (the classifier weights). Soft; prototypes **detached**. MIL: prototypes vs Ψ(B) via inner product is direct.
- **MARS**: importance-weighted source loss + Wasserstein between class-proportion-reweighted marginals; target proportions from mixture estimation matched to classes by OT [read at abstract/algorithm level] — [arXiv 2006.08161](https://arxiv.org/abs/2006.08161v3).

**E. Divergence-based.**
- **CS-DA**: D̂_CS(p_s(z);p_t(z)) = log((1/M²)Σκ(z_i^s,z_j^s)) + log((1/N²)Σκ(z_i^t,z_j^t)) − 2 log((1/(MN))ΣΣκ(z_i^s,z_j^t)) [read]; conditional D̂_CCS(p_s(ŷ|z);p_t(ŷ|z)) built from feature Gram K and prediction Gram L (eq. 13) [read]; used inside a bi-classifier adversarial scheme — [arXiv 2405.19978](https://arxiv.org/pdf/2405.19978). *Plain:* the log of how much the two kernel density estimates overlap relative to their self-overlaps. Soft predictions, no weights. The paper notes the CCS aligns p(y|z), not p(z|y) [read].
- **BSP**: L_bsp = Σ_{i=1}^{k}(σ_{s,i}² + σ_{t,i}²), k = 1 default [read; squares lost in text extraction, squared form from memory] — [PMLR](https://proceedings.mlr.press/v97/chen19i.html).
- **CREDA**: class-conditional Rényi-entropy kernel alignment + entropy-weighted pseudo-labels [abstract only] — [IDEAS](https://ideas.repec.org/a/gam/jmathe/v13y2025i16p2602-d1724330.html).

### Inferences
- Hard vs soft: hard + filter (CDD, DCAN, TPN, JDA family, WMMD) vs soft (LMMD, CPDA, PCT, JAN, CS-DA, ETD, COT) vs soft reliability inside a cost (RWOT). MIL-CREDA is hard-routed with a continuous weight — the RWOT/CREDA "continuous reliability" style applied to a CDD/TPN-style hard partition.
- Detaching: the only verified detached weights are LMMD (code), CPDA (config) and PCT (prototypes); DeepJDOT's plan is fixed per step. If MIL-CREDA detaches w, LMMD/CPDA are the precedents to cite; if it does not, the gradient through w = 1 − H2(g)/ln C pushes predictions toward confidence (an entropy-minimisation side effect) — worth stating.
- MIL breakages, ranked: (1) explicit-moment methods (CORAL, CMD, HoMM, DICD scatter, MEDA's GFK) need finite features; (2) OT methods suffer from few bags per mini-batch; (3) clustering-based labels (CDD) need a meaningful cosine geometry on Ψ(B); (4) everything kernel-based (JDA, LMMD, TPN, DCAN, CPDA, CS-DA) transfers by swapping k(x,x') for ⟨Ψ(B),Ψ(B')⟩.

### Gaps
- Exact VDA, JGSA, ARTL, TCA, DAN, RTN, CORAL, CMD, MIDA formulas were not read this run (tagged [from memory]).
- Whether RWOT, TPN, DCAN, CAN official code detaches weights/labels was not checked (only DSAN's code was fetched).
- TPN's second and third multi-granular terms and DCAN's first two trace terms were reconstructed from a garbled two-column text extraction.

---

## Q3. How does MIL-CREDA sit relative to the survivors?

### Takeaway
MIL-CREDA lands in the F3-explicit cell next to CDD, DCAN, TPN, RWOT and CREDA, but differs from all of them on four verifiable axes: (i) the per-class score is a matrix-Rényi-2 Jensen gap that equals a function of a Cauchy–Schwarz divergence between **covariance operators** (second-order), whereas CDD/TPN/LMMD/CPDA compare **means** and Yin et al.'s CS compares **densities** (first-order Gram sums); (ii) hard routing **and** a continuous entropy weight in the same class term; (iii) bag-level units; (iv) a frozen source encoder (asymmetric two-encoder), whose closest discrepancy-family precedents are JGSA (two projections) and PCT (fixed prototypes).

### Cited Findings
- Yin et al.'s marginal CS estimator uses log means of κ over within- and cross-domain pairs — i.e. a cosine between kernel mean embeddings — [arXiv 2405.19978](https://arxiv.org/pdf/2405.19978).
- CDD, LMMD, TPN, CPDA are all differences of class-wise (weighted) kernel means — [arXiv 1901.00976](https://arxiv.org/pdf/1901.00976), [arXiv 2106.09388](https://arxiv.org/pdf/2106.09388), [arXiv 1904.11227](https://arxiv.org/pdf/1904.11227), [arXiv 2608.09193](https://arxiv.org/pdf/2608.09193).
- DCAN compares conditional embedding operators via regularised trace forms (an operator-level comparison, but of Z|Y operators, not covariance operators of Z within a class) — [arXiv 2006.07776](https://arxiv.org/pdf/2006.07776).
- RWOT blends prototype distance and sharpened classifier probability into a reliability Q, gated over training by 𝒜-distance — [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf).
- CDD drops classes with ≤ N0 confident targets — the analogue of MIL-CREDA's confidence gate on class aggregation — [arXiv 1901.00976](https://arxiv.org/pdf/1901.00976).
- JGSA learns separate source and target projections coupled by ‖A − B‖ — [arXiv 1705.05498](https://arxiv.org/pdf/1705.05498); PCT holds prototypes fixed during alignment — [arXiv 2110.12024](https://arxiv.org/pdf/2110.12024).

### Inferences
- **J_c ↔ CS (my derivation, not sourced):** with A = K_s/tr K_s, B = K_t/tr K_t and the ½–½ mix M, tr(M²) = ¼(‖A‖²_F + ‖B‖²_F + 2‖C‖²_F) with C the normalised cross block. Then J_c = −½log‖A‖² − ½log‖B‖² + log tr(M²) = log[½·r + ½·‖C‖²/(‖A‖‖B‖)], r = ½(ρ + 1/ρ), ρ = ‖A‖_F/‖B‖_F. Because ‖A‖²_F = ‖Σ_s‖²_HS and ‖C‖²_F = ⟨Σ_s, Σ_t⟩_HS for the empirical (uncentred) covariance operators, the second term is exp(−D_CS(Σ_s, Σ_t)). This agrees with the brief's log(½r + ½e^{−D_CS}). Equivalently it is a CS between mean embeddings under the squared kernel k². So MIL-CREDA's per-class divergence is second-order in k, whereas Yin et al.'s is first-order. That is the cleanest formal line against the CS-DA paper.
- The term r ≥ 1 rewards matching the spectral *spread* (purity) of the two class clouds, independent of their overlap. No surviving method has such a term. It is also a failure mode to discuss: J_c can be driven by ρ alone.
- Relative to CREDA (same group), the paper must spell out what is new, since both share the Rényi-H2 class-conditional score and the entropy weighting: bags + attention embeddings, the frozen-source two-encoder design, the gate, the soft personalised local term, and the CS-of-covariance-operators reading.
- Suggested citation set for "closest prior art": CDD (filtering + class-wise kernel), DCAN (threshold + operator-level), TPN (threshold + prototype RKHS distance; its μ^{st} mixed prototype parallels K_mix), RWOT (continuous reliability), LMMD/CPDA (soft weights, detached), CS-DA (CS divergence), CREDA (Rényi-H2).

### Gaps
- The J_c ↔ D_CS identity is my derivation; I found no paper stating CS divergence between kernel covariance operators for DA.
- No class-conditional discrepancy method on MIL bags was found in this run (absence after targeted searches in this and the prior run, not proof).
- Head-to-head numbers between F3-explicit and F3-soft methods under the same backbone were not collected.
