# Unsupervised class-conditional and joint-distribution kernel DA (MMD family and relatives), positioned against MIL-CREDA

Research run date: 2026-10-10. Every venue below was checked by a search or fetch in this run unless it is marked **[UNVERIFIED]**. Method *mechanics* that I could not read in a fetched source are marked **[mechanics from memory, unverified]**.

Reference point (MIL-CREDA, from the brief): source bags labelled, target bags unlabelled; bag embedding = attention-weighted kernel mean embedding (Gaussian instance kernel, one global σ); bag kernel = inner product of embeddings; two encoders (source frozen after phase 1); per-class global term J_c = ½[H2(K_s)+H2(K_t)] − H2(K_mix) on trace-normalised Gram matrices, target routed by **hard** argmax pseudo-labels with confidence weight w_j = 1 − H2(g_j)/ln C; local term = RKHS distance of each target bag to a personalised source reference built from the **full soft** prediction.

---

## Q1. Is there unsupervised class-conditional MMD in the state of the art? (catalogue)

### Takeaway
Yes, and it is a mature line: shallow class-conditional MMD with iterated hard pseudo-labels dates to JDA (ICCV 2013) and its descendants (BDA, VDA, DICD, ARTL, MEDA); deep versions include LMMD/DSAN (soft weights, TNNLS 2021), CDD/CAN (clustering + filtering, CVPR 2019), WMMD (class-prior reweighting, CVPR 2017) and CMMD/DCAN (conditional embedding operators). A direct precursor, CREDA (Pérez-Rosero, Álvarez-Meza, Castellanos-Dominguez, *Mathematics* 2025), already combines class-conditional kernel alignment with matrix-based Rényi quadratic entropy and entropy-weighted pseudo-labels. What I found **no** published instance of: class-conditional kernel alignment over **MIL bags** (bag-level mean embeddings) in unsupervised DA, nor any of the catalogued methods with **two domain-specific encoders where the source one is frozen** combined with a class-conditional kernel term.

### Cited Findings

**Shallow (projection / classifier learning, no deep encoder)**

- **TCA** — Pan, Tsang, Kwok, Yang, "Domain Adaptation via Transfer Component Analysis", IEEE Trans. Neural Networks 22(2), Feb 2011 (conference version IJCAI 2009). Learns transfer components in an RKHS by minimising MMD; aligns **marginal P(X)** only; semi-supervised variant SSTCA encodes label info — [UST PDF](https://cse.hkust.edu.hk/~jamesk/papers/tnn11.pdf), [IEEE Xplore](https://ieeexplore.ieee.org/document/5640675), [IJCAI 2009](https://www.ijcai.org/Abstract/09/200).
- **JDA** — Long et al., "Transfer Feature Learning with Joint Distribution Adaptation", ICCV 2013 — listed in [jindongwang/transferlearning awesome_paper.md](https://raw.githubusercontent.com/jindongwang/transferlearning/master/doc/awesome_paper.md). Learns a projection A minimising marginal MMD + **class-conditional MMD**, using pseudo-labels that are iteratively refined: train classifier on projected features, update target pseudo-labels, re-run JDA until convergence — [Zhuang et al. survey, arXiv 1911.02685](https://arxiv.org/pdf/1911.02685). Full author list **[UNVERIFIED in this run]** (only "Long et al.").
- **BDA** — Wang et al., "Balanced Distribution Adaptation for Transfer Learning" ([arXiv 1807.00516](https://arxiv.org/pdf/1807.00516)). Balance factor μ∈[0,1] weighting marginal vs conditional MMD (JDA is μ=0.5 special case); W-BDA adds class-adaptive weights for class imbalance. Venue **ICDM 2017 [UNVERIFIED — not stated in the pages fetched]**.
- **VDA** — Tahmoresnezhad & Hashemi, "Visual domain adaptation via transfer feature learning", Knowledge and Information Systems, 2016 (DOI 10.1007/s10115-016-0944-x). Reduces joint marginal + conditional distributions unsupervised — [datalearner index](https://datalearner.com/academic/journal-papers/0219-1377/volumes-and-issues/50/paper-detail/16750), [Unpaywall DOI](https://unpaywall.org/10.1007%2FS10115-016-0944-X). (Brief says VDA; year is 2016 online / vol. 50 is 2017 print.)
- **DICD** — Li, Song, Huang, Ding, Wu, "Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation", 2018; DOI 10.1109/TIP.2018.2839528 (IEEE TIP). Jointly matches marginal and **class-conditional** distributions while maximising inter-class dispersion and minimising intra-class scatter; generalised eigen-decomposition — [BUAA record](https://research.buaa.edu.cn/en/publications/domain-invariant-and-class-discriminative-feature-learning-for-vi/), [OpenAlex DOI](https://api.openalex.org/works/doi:10.1109%2FTIP.2018.2839528).
- **ARTL** — Long et al., "Adaptation Regularization: A General Framework for Transfer Learning": learns an adaptive classifier jointly optimising structural risk, joint (marginal + conditional) distribution matching, and manifold consistency — [Tsinghua PDF](https://ise.thss.tsinghua.edu.cn/mig/Adaptation%20Regularization%20A%20General%20Framework%20for%20Transfer%20Learning.pdf), [MEDA arXiv](https://arxiv.org/pdf/1807.07258). Venue **IEEE TKDE 2014 [UNVERIFIED]**.
- **MEDA** — Wang et al., "Visual Domain Adaptation with Manifold Embedded Distribution Alignment", ACM MM 2018 (index entry "ACMMM-18" in [transferlearning index](https://raw.githubusercontent.com/jindongwang/transferlearning/master/doc/awesome_paper.md); [arXiv 1807.07258](https://arxiv.org/pdf/1807.07258)). Grassmann-manifold features + **dynamic** distribution alignment (adaptive weight between marginal and conditional), structural-risk classifier (ARTL-style).
- **MIDA** — Ke Yan, Lu Kou, David Zhang, "Learning Domain-Invariant Subspace using Domain Features and Independence Maximization", accepted IEEE Trans. Cybernetics. Learns a subspace with **maximum independence (HSIC) from domain features** (device label, acquisition time); handles discrete and continuous drift; SMIDA semi-supervised — [arXiv 1603.04535](https://www.arxiv.org/pdf/1603.04535), [MATLAB code](https://www.mathworks.com/matlabcentral/fileexchange/56645-maximum-independence-domain-adaptation-mida). Note: MIDA's authors are **Ke Yan** et al., not the Hongliang Yan of WMMD. Publication year **[UNVERIFIED: 2017 vs 2018]**.
- **JDOT** — "Joint distribution optimal transportation for domain adaptation", NIPS 2017 — [transferlearning index](https://raw.githubusercontent.com/jindongwang/transferlearning/master/doc/awesome_paper.md). Authors (Courty, Flamary, Habrard, Rakotomamonjy) **[UNVERIFIED in this run]**.
- **CORAL** — Sun, Feng, Saenko, "Return of Frustratingly Easy Domain Adaptation", AAAI 2016. Aligns **second-order statistics** (covariances) of source and target, no target labels — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/10306), [arXiv 1511.05547](https://arxiv.org/pdf/1511.05547).
- **DJP-MMD** — Wen Zhang & Dongrui Wu, "Discriminative Joint Probability MMD for Domain Adaptation", IJCNN 2020. Measures the **joint** probability discrepancy directly instead of a weighted sum of marginal + conditional, and adds a discriminability (inter-class) term; uses pseudo-labels within a JDA-like framework — [arXiv 1912.00320](https://arxiv.org/abs/1912.00320).

**Deep (end-to-end feature learning)**

- **DAN / MK-MMD** — Long et al., "Learning transferable features with deep adaptation networks", ICML 2015 (citation verified via DSAN reference list, [ar5iv 2106.09388](https://ar5iv.arxiv.org/html/2106.09388)). Aligns **marginal** feature distributions of several task-specific layers with multi-kernel MMD **[mechanics from memory, unverified]**.
- **RTN** — Long, Zhu, Wang, Jordan, "Unsupervised Domain Adaptation with Residual Transfer Networks", NIPS 2016. Relaxes the shared-classifier assumption: source and target classifiers differ by a residual function; feature alignment by MMD — [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2016/hash/ac627ab1ccbdb62ec96e702f07f6425b-Abstract.html), [arXiv 1602.04433](https://arxiv.org/abs/1602.04433). Target entropy minimisation **[mechanics from memory, unverified]**.
- **JAN / JMMD** — Long, Zhu, Wang, Jordan, "Deep Transfer Learning with Joint Adaptation Networks", ICML 2017. JMMD = distance between kernel embeddings of the **joint** distribution of several layers' activations including the classifier layer fc8 (soft predictions); product kernels across layers; single shared backbone; JAN-A adds adversarial kernel training — [arXiv 1605.06636](https://arxiv.org/abs/1605.06636), [ar5iv](https://ar5iv.labs.arxiv.org/html/1605.06636).
- **WMMD** — Hongliang Yan, Yukang Ding, Peihua Li, Qilong Wang, Yong Xu, Wangmeng Zuo, "Mind the Class Weight Bias: Weighted MMD for Unsupervised Domain Adaptation", CVPR 2017. Class-specific auxiliary weights α_c = w_c^t / w_c^s reweight source samples so the source reference carries **target class priors** (label shift); classification-EM with hard pseudo-labels; shared CNN — [CVF open access](https://openaccess.thecvf.com/content_cvpr_2017/html/Yan_Mind_the_Class_CVPR_2017_paper.html), [ar5iv 1705.00609](https://ar5iv.arxiv.org/html/1705.00609).
- **Deep CORAL** — Sun & Saenko, "Deep CORAL: Correlation alignment for deep domain adaptation", ECCV 2016 (per DSAN references, [ar5iv](https://ar5iv.arxiv.org/html/2106.09388); it appeared in an ECCV 2016 **workshop** — the workshop qualifier is **[UNVERIFIED]**).
- **DANN** — Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand, Lempitsky, "Domain-Adversarial Training of Neural Networks", JMLR 17(59), 2016 (conference precursor: Ganin & Lempitsky, ICML 2015) — [JMLR](https://www.jmlr.org/papers/v17/15-239.html), [DSAN refs](https://ar5iv.arxiv.org/html/2106.09388). Marginal alignment via gradient-reversal domain classifier.
- **CDAN / CDAN+E** — Long, Cao, Wang, Jordan, "Conditional Adversarial Domain Adaptation", NeurIPS 2018. Discriminator conditioned on the **multilinear** map of features × soft classifier predictions (cross-covariance), i.e. aligns a joint of (features, predictions); CDAN+E weights samples by w = 1 + e^{−H(g)} — [arXiv 1705.10667](https://arxiv.org/abs/1705.10667).
- **DeepJDOT** — Damodaran, Kellenberger, Flamary, Tuia, Courty, ECCV 2018, pp. 447–463. OT discrepancy on **joint** (deep representation, label) space — [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.html), [code](https://github.com/bbdamodaran/deepJDOT).
- **CAN / CDD** — Kang, Jiang, Yang, Hauptmann, "Contrastive Adaptation Network for Unsupervised Domain Adaptation", CVPR 2019. CDD minimises intra-class and maximises inter-class cross-domain kernel discrepancy; target labels from spherical k-means initialised at source class centres, with sample- and class-level filtering; alternating optimisation; shared ResNet with domain-specific BN — [arXiv 1901.00976](https://arxiv.org/abs/1901.00976), [HTML](https://arxiv.org/html/1901.00976v2).
- **DSAN / LMMD** — Zhu, Zhuang, Wang, Ke, Chen, Bian, Xiong, He, "Deep Subdomain Adaptation Network for Image Classification", IEEE TNNLS ("published on TNNLS", arXiv submitted 17 Jun 2021). LMMD aligns class-conditional ("subdomain") distributions with **soft** target weights; no adversarial training; one shared network — [arXiv abs](https://arxiv.org/abs/2106.09388), [ar5iv](https://ar5iv.arxiv.org/html/2106.09388).
- **CMMD / DCAN** — Ge, Ren, Dai, Yan, "Domain Adaptation and Image Classification via Deep Conditional Adaptation Network" (Pattern Recognition, per [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320322005684) and [datalearner](https://datalearner.com/academic/journal-papers/0031-3203/volumes-and-issues/513/paper-detail/95389); year **2023 [UNVERIFIED]**). Minimises **Conditional MMD** between conditional distributions via **conditional embedding operators** in RKHS (i.e. aligns P(X|Y) through operators, not per-class means), plus mutual-information maximisation on target — [arXiv 2006.07776](https://arxiv.org/pdf/2006.07776). Follow-up KLN (Ren, Ge, Dai, Yan, arXiv 2020) learns the kernel for CMMD with an auto-encoder — [arXiv 2008.10165](https://arxiv.org/abs/2008.10165).
- **MCC** — Jin, Wang, Long, Wang, "Minimum Class Confusion for Versatile Domain Adaptation", ECCV 2020. Non-adversarial loss reducing pairwise class confusion of target predictions; no explicit domain alignment — [ECVA](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3769_ECCV_2020_paper.php), [arXiv 1912.03699](https://arxiv.org/pdf/1912.03699).
- **BNM** — Cui, Wang, Zhuo, Li, Huang, Tian, "Towards Discriminability and Diversity: Batch Nuclear-Norm Maximization under Label Insufficient Situations", CVPR 2020. Maximises the nuclear norm of the batch prediction matrix (discriminability + diversity) — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Cui_Towards_Discriminability_and_Diversity_Batch_Nuclear-Norm_Maximization_Under_Label_Insufficient_CVPR_2020_paper.html).
- **SHOT** — Liang, Hu, Feng, "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for UDA", ICML 2020. Source-free: freezes the source classifier (hypothesis) and learns a target-specific feature extractor with information maximisation + self-supervised pseudo-labelling — [ICML virtual](https://icml.cc/virtual/2020/poster/5793), [mlanthology](https://mlanthology.org/icml/2020/liang2020icml-we/). Pseudo-labels by weighted-centroid clustering **[mechanics from memory, unverified]**.
- **ATDOC** — Liang, Hu, Feng, "Domain Adaptation with Auxiliary Target Domain-Oriented Classifier", CVPR 2021. Auxiliary target-only classifier to reduce classifier bias in pseudo-labels; memory bank with non-parametric nearest-centroid or neighbourhood-aggregation classifiers, no extra parameters — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Liang_Domain_Adaptation_With_Auxiliary_Target_Domain-Oriented_Classifier_CVPR_2021_paper.pdf), [arXiv 2007.04171](https://arxiv.org/pdf/2007.04171).
- **HoMM** — "Higher-order Moment Matching for Unsupervised Domain Adaptation", AAAI 2020: matches p-level tensor powers of activations — [arXiv 1912.11976](https://arxiv.org/pdf/1912.11976). Authors **[UNVERIFIED]**.
- **CMD** — Zellinger et al., "Central Moment Discrepancy for Domain-Invariant Representation Learning", ICLR 2017: order-wise central-moment differences, no kernel matrices — [mlanthology](https://mlanthology.org/iclr/2017/zellinger2017iclr-central).

**Information-theoretic / recent relatives (closest to MIL-CREDA)**

- **CREDA** — Pérez-Rosero, Álvarez-Meza, Castellanos-Dominguez, "Conditional Domain Adaptation with α-Rényi Entropy Regularization and Noise-Aware Label Weighting", *Mathematics* (MDPI) 13(16):2602, 2025. Class-conditional alignment via a "Rényi-based entropy kernel estimator", matrix-based Rényi quadratic entropy, entropy-weighted pseudo-labels; Digits, ImageCLEF-DA, Office-31 — [RePEc/IDEAS](https://ideas.repec.org/a/gam/jmathe/v13y2025i16p2602-d1724330.html).
- **CS-divergence DA** — Yin, Yu, Lin, Liu, Sonke, Gavves, "Domain Adaptation with Cauchy-Schwarz Divergence", UAI 2024. Kernel (Parzen) CS divergence for marginal and a **conditional CS (CCS)** divergence for p(y|x), non-parametric estimators — [mlanthology](https://mlanthology.org/uai/2024/yin2024uai-domain/), [arXiv 2405.19978](https://arxiv.org/html/2405.19978v1).
- **Matrix-based Rényi entropy** foundations: Rényi entropy from the normalised eigenspectrum of a Gram matrix; multivariate extension — [arXiv 1808.07912](https://arxiv.org/pdf/1808.07912); representation Jensen-Rényi divergence — [ar5iv 2112.01583](https://ar5iv.labs.arxiv.org/html/2112.01583) (details of this paper not read).
- **CPDA** — Ott & Mutschler, "Class-Conditional Path Distribution Alignment for Unsupervised Time-Series DA", arXiv 2608.09193, 10 Aug 2026 (preprint). D² = Σ_c π_t(c)‖μ̂_{s,c} − μ̂_{t,c}‖²; source weights = indicator, target weights = soft posteriors; stabilised by information maximisation + VAT; shows marginal MMD, CORAL-type, HoMM and DAN-style as restricted cases — [arXiv HTML](https://arxiv.org/html/2608.09193v1).
- **Discriminative class-wise MMD** (deep kernel, pseudo-labels), AIMS Mathematics, Feb 2024 — [AIMS](http://www.aimspress.com/article/doi/10.3934/math.2024323). Authors not read.
- **Deep Subdomain Alignment for Cross-Domain Image Classification**, Zhao et al., WACV 2024 — [CVF PDF link](https://openaccess.thecvf.com/content/WACV2024/papers/Zhao_Deep_Subdomain_Alignment_for_Cross-Domain_Image_Classification_WACV_2024_paper.pdf) (fetch returned 403; content **[UNVERIFIED]**).
- **Intra-class-similarity pseudo-label refinement** — Jie Wang & Xiao-Lei Zhang, arXiv 2022: iteratively removes dissimilar/mislabelled samples per class via spanning trees — [arXiv 2207.12139](https://arxiv.org/abs/2207.12139).

**MIL + DA (for question f)**

- Domain-adaptive MIL for instance-level pathology prediction (source has instance+bag labels, target only bag labels — i.e. *not* unsupervised at bag level) — [arXiv 2304.03537](https://arxiv.org/pdf/2304.03537).
- Multi-scale domain-adversarial MIL CNN for cancer subtype classification (adversarial, not kernel) — [arXiv 2001.01599](https://arxiv.org/pdf/2001.01599).
- A comparative study of unsupervised adversarial DA strategies in MIL for digital pathology is reported (MIDL 2023) — [MIDL](https://2023.midl.io/papers/s028) (content not read).

### Catalogue table (a–f)

Legend — (a) distribution aligned; (b) target class info; (c) shallow/deep; (d) encoders; (e) fits MIL-CREDA assumptions (labelled source bags, unlabelled target bags, two encoders, source frozen); (f) MIL-bag extension and what breaks. Columns (e)/(f) are my inferences.

| Method | (a) | (b) | (c) | (d) | (e) | (f) |
|---|---|---|---|---|---|---|
| TCA | marginal | none | shallow (kernel projection) | one shared projection | partly: no class info, needs joint Gram on all samples | bag kernel plugs in directly (projection on bag Gram); breaks only scale (n_bags² Gram) |
| JDA | marginal + class-cond. (means) | hard pseudo-labels, iterated | shallow | shared projection A | yes in spirit; frozen-source two-encoder not native | per-class MMD on bag embeddings is direct; projection must act on bag kernel — kernelised JDA needed |
| BDA / W-BDA | weighted marginal + cond.; class weights | hard, iterated | shallow | shared | same as JDA | same; μ becomes another hyperparameter |
| VDA, DICD | marginal + cond. + discriminative scatter | hard, iterated | shallow | shared | same | scatter matrices need vector features → use bag embeddings in explicit or Nyström space |
| ARTL, MEDA | marginal + cond. (dynamic weight in MEDA) + manifold reg. | hard, iterated | shallow | shared classifier in RKHS | yes (kernel classifier) | MEDA's Grassmann (GFK) step needs vector features; ARTL's kernel classifier can take a bag kernel |
| MIDA | independence from domain features (HSIC) | none (domain features, not classes) | shallow | shared | marginal only | HSIC on bag kernel is direct |
| DJP-MMD | joint P(X,Y) directly + inter-class discrimination | hard pseudo-labels | shallow | shared | yes | as JDA |
| DAN | marginal, multi-layer | none | deep | shared | marginal only | MMD on bag embeddings fine |
| RTN | marginal + residual classifier | target entropy | deep | shared features, residual target classifier | partial | yes |
| JAN | joint over layers incl. softmax | soft predictions as a layer | deep | shared | soft joint, not class-routed | product kernel (bag kernel × prediction kernel) is natural for bags |
| WMMD | marginal with class-prior reweighting (label shift) | hard, CEM | deep | shared | yes; complementary (prior correction) | direct: reweight source bags by α_c |
| CDD/CAN | class-cond., intra minus inter | k-means clusters, filtered (D0, N0) | deep | shared + domain-specific BN | yes, closest deep analogue of a per-class contrastive J_c | class-aware sampling of bags; spherical k-means on bag embeddings needs a bag-embedding space with cosine geometry |
| DSAN/LMMD | class-cond. (soft subdomains) | soft predictions | deep | shared | yes | direct on bag embeddings; batch must contain enough bags per class |
| CMMD/DCAN | P(X|Y) via conditional embedding operators | predictions (soft) | deep | shared | yes | operator needs inverse of label Gram (regularised) — fine with bag kernel |
| CORAL / Deep CORAL | 2nd-order (covariance), marginal | none | shallow / deep | shared | marginal only | needs explicit features; bag embedding is infinite-dim → use finite features or kernel covariance operators |
| HoMM, CMD | higher-order moments, marginal | none (HoMM has a class-aware variant — unverified) | deep | shared | marginal | explicit features required |
| DANN | marginal (adversarial) | none | deep | shared | marginal only; frozen source encoder resembles ADDA (CVPR 2017, cited in DSAN refs) | discriminator on bag embeddings is easy |
| CDAN(+E) | joint (features ⊗ soft predictions) | soft + entropy weight 1+e^{−H} | deep | shared | yes; entropy weighting analogous to MIL-CREDA's w_j | multilinear map of a bag embedding needs finite features |
| DeepJDOT / JDOT | joint (feature, label) via OT | source labels + target predicted labels in cost | deep / shallow | shared | yes | OT between bags with bag-kernel cost is direct; O(n²) coupling per batch |
| MCC, BNM | prediction-space regularisers (no explicit alignment) | soft predictions | deep | shared | complementary only | trivially applicable to bag classifiers |
| SHOT | implicit alignment to frozen source hypothesis | pseudo-labels + info max | deep | separate target extractor, frozen source classifier | source-free; **closest on the "frozen source" axis** | applies to a bag classifier directly |
| ATDOC | auxiliary target classifier | memory bank, NC / neighbourhood aggregation | deep | shared | complementary pseudo-label engine | memory of bag embeddings is fine |
| CS-div (UAI 2024) | marginal + conditional p(y|x) | soft predictions | deep | shared | yes; information-theoretic kernel relative | Gram-based estimators accept bag kernels |
| CREDA (2025) | class-cond. via matrix Rényi H2 | entropy-weighted pseudo-labels | deep | per paper abstract unclear | direct precursor | MIL-CREDA *is* this extension |
| CPDA (2026 preprint) | class-cond. path distributions | soft posteriors | deep | shared | yes | swaps path signature kernel for bag kernel conceptually |

### Inferences
- The class-conditional MMD problem itself is not new; MIL-CREDA's novelty must rest on (i) bag-level kernel mean embeddings with attention as the unit of alignment, (ii) the Rényi-H2 Jensen-type score with domain-balanced K_mix rather than squared MMD, (iii) the dual global (hard-routed, entropy-weighted) + local (soft personalised reference) design, and (iv) the asymmetric two-encoder setup with a frozen source encoder.
- MIL-CREDA's global term is closest to CDD (per-class kernel discrepancy, hard assignments with confidence filtering) and the local soft term is closest to LMMD/CPDA (soft class weights on target means). CDD additionally uses an **inter-class** repulsive term that MIL-CREDA does not appear to have.
- The two-encoder, frozen-source design is atypical for the MMD family (all deep MMD methods found use one shared backbone); it resembles ADDA (unshared, source fixed — cited in DSAN refs, CVPR 2017) and SHOT (frozen source hypothesis). This is a positioning axis worth stating explicitly.
- Because CREDA (2025) is by the same research group as the brief's author context, the paper must clearly separate MIL-CREDA from CREDA.

### Gaps
- Full author lists for JDA, BDA, MEDA, ARTL, HoMM, JDOT were not read in a fetched page.
- Venues not confirmed: BDA (ICDM 2017), ARTL (TKDE 2014), MIDA year, DCAN year, Deep CORAL workshop status.
- WACV 2024 subdomain alignment paper: fetch blocked (403).
- I found no paper doing unsupervised class-conditional kernel DA on MIL bags; absence of evidence after ~3 targeted searches, not proof.
- Whether CREDA uses one or two encoders could not be read (abstract only).

---

## Q2. Hard vs soft pseudo-labels, and how wrong pseudo-labels are handled

### Takeaway
Shallow JDA-family methods and WMMD use **hard** argmax labels refreshed by iteration (JDA, BDA, WMMD via classification-EM); CDD uses **cluster** assignments with explicit distance and class-size filtering; LMMD, JAN, CDAN, CPDA use **soft** predictions; confidence weighting by entropy appears in CDAN+E (w = 1+e^{−H(g)}) and CREDA. MIL-CREDA's hybrid (hard routing + entropy weight for the global term, full soft vector for the local term) combines the CDD/WMMD and LMMD/CDAN traditions.

### Cited Findings
- JDA: pseudo-labels updated each iteration from a classifier trained on the projected features, repeated to convergence — [Zhuang survey](https://arxiv.org/pdf/1911.02685).
- WMMD: C-step assigns hard ŷ_j = argmax_c p(y_j^t=c|x_j^t), then estimates α_c = ŵ_c^t / w_c^s, M-step backprop — [ar5iv 1705.00609](https://ar5iv.arxiv.org/html/1705.00609).
- CDD: spherical k-means on target initialised at source class centres (cosine); samples farther than D0 from their centre are dropped; classes with fewer than N0 retained samples are excluded that round; class-aware sampling; alternate clustering ↔ feature update — [arXiv HTML 1901.00976v2](https://arxiv.org/html/1901.00976v2).
- LMMD: target weights use ŷ_i^t, the soft probability of x_i^t for each of the C classes — [ar5iv 2106.09388](https://ar5iv.arxiv.org/html/2106.09388).
- JAN: classifier-layer (fc8) activations enter the joint embedding, so soft predictions are used without any thresholding — [ar5iv 1605.06636](https://ar5iv.labs.arxiv.org/html/1605.06636).
- CDAN+E: entropy conditioning w = 1 + e^{−H(g)} down-weights uncertain target samples — [arXiv 1705.10667](https://arxiv.org/abs/1705.10667).
- CPDA: soft posteriors, stabilised by information maximisation (anti class-collapse) and VAT — [arXiv 2608.09193](https://arxiv.org/html/2608.09193v1).
- CREDA: entropy-weighted mechanism reduces the influence of uncertain predictions — [IDEAS](https://ideas.repec.org/a/gam/jmathe/v13y2025i16p2602-d1724330.html).
- ATDOC: target-only non-parametric classifier (memory bank, nearest centroid / neighbourhood aggregation) to reduce source-biased pseudo-labels — [arXiv 2007.04171](https://arxiv.org/pdf/2007.04171).
- SHOT: information maximisation + self-supervised pseudo-labelling — [mlanthology](https://mlanthology.org/icml/2020/liang2020icml-we/).
- Pruning mislabelled samples by intra-class similarity / spanning trees — [arXiv 2207.12139](https://arxiv.org/abs/2207.12139).
- DJP-MMD and BDA/VDA/DICD inherit JDA's iterative hard pseudo-labels — [DJP-MMD](https://arxiv.org/abs/1912.00320), [BDA](https://arxiv.org/pdf/1807.00516).

### Inferences
- MIL-CREDA's w_j = 1 − H2(g_j)/ln C is a normalised-entropy confidence in [0,1]; note H2 (collision entropy) ≤ Shannon H ≤ ln C, so the weight is ≥ the analogous Shannon-based weight — it is more permissive than a Shannon-entropy weight for the same prediction. CDAN+E uses Shannon entropy with an exponential, never reaching zero.
- None of the found methods combine hard routing **and** an entropy weight inside the same per-class kernel score; CDD drops samples (binary weight), CDAN weights but does not route.
- Hard routing has a known failure mode the catalogue handles explicitly: classes with few or no confident target samples (CDD's N0 class filter). MIL-CREDA should state what happens to J_c when a class receives no or very few target bags.

### Gaps
- Exact SHOT clustering procedure and BNM/MCC formulas not read in this run.
- No source found quantifying hard vs soft pseudo-label trade-offs head-to-head in a kernel-MMD setting.

---

## Q3. Which methods align covariance / higher-order structure rather than means

### Takeaway
Second-order: CORAL, Deep CORAL, CDAN's multilinear (cross-covariance) conditioning; higher-order: HoMM (tensor powers), CMD (central moments), JAN (joint embeddings over layers); HSIC/independence: MIDA; operator-level: CMMD/DCAN (conditional embedding operators). Note that MMD with a characteristic Gaussian kernel already compares all moments of the distribution through the mean embedding; Rényi-H2 on Gram matrices (CREDA, MIL-CREDA) and CS divergence also depend on full kernel spectra/Gram sums rather than only means.

### Cited Findings
- CORAL aligns second-order statistics (covariances) — [AAAI 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10306).
- HoMM matches p-level tensor powers of activations — [arXiv 1912.11976](https://arxiv.org/pdf/1912.11976).
- CMD matches order-wise central moments; it is a metric on distributions on a compact interval — [mlanthology](https://mlanthology.org/iclr/2017/zellinger2017iclr-central), [JKU PDF](https://www.flll.jku.at/sites/default/files/u34/cmd.pdf).
- CDAN multilinear conditioning captures the cross-covariance between features and predictions — [arXiv 1705.10667](https://arxiv.org/abs/1705.10667).
- MIDA maximises independence (HSIC) of the subspace from domain features — [arXiv 1603.04535](https://www.arxiv.org/pdf/1603.04535).
- CMMD aligns conditional distributions through conditional embeddings (operators) in RKHS — [arXiv 2006.07776](https://arxiv.org/pdf/2006.07776).
- CPDA: CORAL is a linear kernel on centred quadratic features; HoMM is a polynomial feature map of order p — both restricted cases of a kernel mean discrepancy — [arXiv 2608.09193](https://arxiv.org/html/2608.09193v1).
- Matrix-based Rényi entropy uses the normalised eigenspectrum of the Gram matrix — [arXiv 1808.07912](https://arxiv.org/pdf/1808.07912).

### Inferences
- For trace-normalised A, H2(A) = −log tr(A²) = −log ‖A‖_F²; so H2 of a Gram matrix is a log of a sum of squared kernel values, i.e. it measures second-order structure of the kernel matrix (spread of the embedded cloud), not just its mean. The J_c score (½[H2(K_s)+H2(K_t)] − H2(K_mix)) is a Jensen-type gap; with K_mix balanced to ½ mass per domain it relates to MMD-like cross terms (derivation not checked against a source — treat as my inference).
- Kernel covariance-operator alignment for bags is possible in principle (bag embeddings live in an RKHS), but every covariance method found works on explicit finite features.

### Gaps
- Did not find/verify a paper on "kernel covariance operator alignment" for DA specifically.
- HoMM class-aware variant (arXiv 2401.15952 seen in search) not read.

---

## Q4. Exact class-conditional MMD formulas in JDA, CDD and LMMD

### Takeaway
All three are per-class differences of (weighted) kernel mean embeddings; they differ in the weights: JDA uses uniform weights over hard pseudo-label sets (and a linear projection A); CDD uses cluster masks and subtracts inter-class terms; LMMD uses normalised soft probabilities and averages over classes.

### Cited Findings
- **JDA** (per Zhuang et al.'s restatement) — [arXiv 1911.02685](https://arxiv.org/pdf/1911.02685):
  - Marginal: ‖ (1/n_s) Σ_i Aᵀx_i^s − (1/n_t) Σ_j Aᵀx_j^t ‖²_H
  - Conditional: Σ_{c=1}^{C} ‖ (1/n_c) Σ_{x_i∈D_S^{(c)}} Aᵀx_i − (1/m_c) Σ_{x_j∈D_T^{(c)}} Aᵀx_j ‖²_H, with D_T^{(c)} the target samples whose **pseudo-label** is c, n_c, m_c their counts.
  - Matrix form (my rewrite, standard but not read in a source): Σ_{c=0}^{C} tr(AᵀX M_c XᵀA), with (M_c)_{ij} = 1/n_c² (both in source class c), 1/m_c² (both in target class c), −1/(n_c m_c) (one in each), 0 otherwise; c=0 is the marginal term. JDA solves min_A Σ_c tr(AᵀXM_cXᵀA) + λ‖A‖²_F s.t. AᵀXHXᵀA = I (generalised eigenproblem) — **constraint/regulariser [from memory, unverified]**.
- **CDD** — [arXiv HTML 1901.00976v2](https://arxiv.org/html/1901.00976v2):
  - D̂^{c1c2} = e1 + e2 − 2 e3, with mask μ_{cc'}(y,y') = 1 if y=c and y'=c', else 0.
  - e1 = Σ_{i,i'} μ_{c1c1}(y_i^s,y_{i'}^s) k(φ(x_i^s),φ(x_{i'}^s)) / Σ_{i,i'} μ_{c1c1}(y_i^s,y_{i'}^s); e2 = same over target with predicted labels ŷ^t and class c2; e3 = Σ_{i,j} μ_{c1c2}(y_i^s,ŷ_j^t) k(φ(x_i^s),φ(x_j^t)) / Σ_{i,j} μ_{c1c2}(y_i^s,ŷ_j^t). (The fetched page names e1, e2, e3 as intra-source, intra-target, cross terms and gives the mask; the explicit normalised-sum form above is my reconstruction from those definitions.)
  - D̂^{cdd} = (1/M) Σ_{c=1}^{M} D̂^{cc} − (1/(M(M−1))) Σ_{c=1}^{M} Σ_{c'≠c} D̂^{cc'}  (intra-class minimised, inter-class maximised), summed over adapted layers L; total loss ℓ = ℓ^{ce} + β D̂^{cdd}_L.
- **LMMD** — [ar5iv 2106.09388](https://ar5iv.arxiv.org/html/2106.09388):
  - d_H(p,q) ≜ E_c ‖ E_{p^{(c)}}[φ(x^s)] − E_{q^{(c)}}[φ(x^t)] ‖²_H
  - d̂_H(p,q) = (1/C) Σ_{c=1}^{C} ‖ Σ_{x_i^s∈D_s} w_i^{sc} φ(x_i^s) − Σ_{x_j^t∈D_t} w_j^{tc} φ(x_j^t) ‖²_H
  - w_i^c = y_{ic} / Σ_{(x_j,y_j)∈D} y_{jc}  (source: one-hot labels; target: soft ŷ^t)
  - Layer l kernel form: d̂_l = (1/C) Σ_c [ Σ_i Σ_j w_i^{sc} w_j^{sc} k(z_i^{sl},z_j^{sl}) + Σ_i Σ_j w_i^{tc} w_j^{tc} k(z_i^{tl},z_j^{tl}) − 2 Σ_i Σ_j w_i^{sc} w_j^{tc} k(z_i^{sl},z_j^{tl}) ]
  - Total: min_f (1/n_s) Σ_i J(f(x_i^s), y_i^s) + λ Σ_{l∈L} d̂_l(p,q).
- **Related forms**: WMMD — ‖ (1/Σ_i α_{y_i^s}) Σ_i α_{y_i^s} φ(x_i^s) − (1/N) Σ_j φ(x_j^t) ‖²_H with α_c = w_c^t/w_c^s — [ar5iv 1705.00609](https://ar5iv.arxiv.org/html/1705.00609). CPDA — Σ_c π_t(c) ‖μ̂_{s,c} − μ̂_{t,c}‖² — [arXiv 2608.09193](https://arxiv.org/html/2608.09193v1). JMMD — (1/n_s²)ΣΣ Π_{l∈L} k^l(z_i^{sl},z_j^{sl}) + (1/n_t²)ΣΣ Π_l k^l(z_i^{tl},z_j^{tl}) − (2/(n_s n_t))ΣΣ Π_l k^l(z_i^{sl},z_j^{tl}) — [ar5iv 1605.06636](https://ar5iv.labs.arxiv.org/html/1605.06636).

### Inferences
- With hard one-hot target weights normalised per class, LMMD reduces to JDA's per-class term (up to the 1/C average and the projection). CDD's D̂^{cc} is the same quantity with kernel-normalised masks plus inter-class repulsion. So all three are instances of ‖μ_{s,c} − μ_{t,c}‖² with different weightings; MIL-CREDA's J_c is **not** of this squared-difference form but a Rényi-H2 Jensen gap over Gram matrices — the key formal difference to state in the paper.
- Replacing φ(x) by the attention-weighted bag embedding Ψ(B) and k by ⟨Ψ,Ψ'⟩ turns each formula into a bag-level version without further change; the per-class kernel sums then cost O(n_bags² · instances²) unless embeddings are cached or approximated.

### Gaps
- JDA's exact optimisation constraints and CDD's explicit e1/e2/e3 normalisation were not read verbatim; flagged above.
- MIL-CREDA's J_c ↔ MMD relationship is my derivation sketch, not sourced.
