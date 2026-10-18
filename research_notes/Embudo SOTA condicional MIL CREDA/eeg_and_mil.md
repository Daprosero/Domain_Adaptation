# The conditional-DA funnel for cross-subject EEG and MIL/set-level data: positioning MIL-CREDA

Research run: 2026-10-10. About 35 search, fetch and OpenAlex-API calls. This file builds on `../SOTA adaptación condicional MIL CREDA/mil_domain_adaptation.md` and does not repeat it. Where a MIL item is already covered there, this file only places it in the funnel.

**Citation tags.** **[V]** = title, authors, venue and year were confirmed this run against OpenAlex (DOI-resolved metadata) or the primary page. **[S]** = only a search snippet or a title was seen. **[UNVERIFIED]** = no matching record was found this run.
**Mechanism tags** (whether the funnel decision itself was checked). **[mech: full text]** = read in the paper's text this run. **[mech: abstract]** = read in the abstract this run. **[mech: memory]** = from background knowledge, not checked.
**Math tags.** **[math: paper]** = equation read from the paper this run. **[math: memory]** = reconstructed from memory; check before reuse.

**Funnel.** S0 = strong or influential method (2018–2026). F1 = aligns class-conditional distributions, or conditions the alignment on class. F2 = needs no target labels (target calibration data counts as labels). F3 = weights or filters target pseudo-labels by confidence. "Out at Fk" means the method was discarded at that filter. A published conditional variant of a discarded method is listed separately and continues through the funnel.
**Columns used below.** MS = multi-source (each subject is its own domain). Bag = uses a subject-as-bag or set-level representation. MIL-ext = could extend to MIL.

---

## Q1. Cross-subject / cross-session EEG emotion DA: where does each method leave the funnel?

### Takeaway
Most influential SEED/DEAP methods leave at **F1**. They are either not DA at all (DGCNN, SOGNN, PGCN) or align only marginals adversarially (BiDANN, BiHDM, R2G-STNN, WGAN-DA, TANN). The conditional ones (MSJDA, JAGP, FSA-Net, BiHADA) mostly leave at **F3**, because they use soft predictions with no confidence gate. Survivors of all three filters appear only in 2025–2026: DANN-RPLI, DDSPR, the Zhang et al. 2026 subdomain method, and DAMSDAN. PR-PL survives only under a loose reading of F1. **Every survivor is instance-level (segment-level), and none uses a subject-as-bag representation.**

### Cited Findings

**Classic and influential methods (S0)**

| Method | Authors, venue, year [tag] | Data | Out at | Why | MS | Bag | MIL-ext |
|---|---|---|---|---|---|---|---|
| DGCNN | Song, Zheng, Song, Cui, *IEEE TAFFC* 2018 (vol. 11(3) 2020) [V] — [DOI](https://doi.org/10.1109/taffc.2018.2817622) | SEED, DREAMER | F1 | Not DA. A learned-adjacency GCN backbone [mech: abstract] | no | no | as backbone only |
| BiDANN / BiDANN-S | Li, Zheng, Zong, Cui, Zhang, *IEEE TAFFC* 2018 [V] — [DOI](https://doi.org/10.1109/taffc.2018.2885474); first version at IJCAI 2018 [V] — [IJCAI](https://www.ijcai.org/proceedings/2018/216) | SEED | F1 | One global and two per-hemisphere domain discriminators. Marginal only [mech: abstract] | no | no | no |
| BiHDM | Li, Wang, Zheng, Zong, Qi, *IEEE TCDS* 2020 (vol. 13(2) 2021) [V] — [DOI](https://doi.org/10.1109/tcds.2020.2999337) | SEED, SEED-IV, MPED [mech: memory] | F1 | Hemisphere-discrepancy RNN plus one marginal domain discriminator [mech: abstract] | no | no | no |
| R2G-STNN | Li, Zheng, Wang, Zong, Cui, *IEEE TAFFC* 2019 [V] — [DOI](https://doi.org/10.1109/taffc.2019.2922912) | SEED | F1 | BiLSTM with region attention and a marginal domain discriminator [mech: abstract] | no | no | Region attention resembles attention pooling, but over regions, not segments |
| WGAN-DA | Luo, Zhang, Zheng, Lu, *ICONIP* 2018, LNCS [V] — [DOI](https://doi.org/10.1007/978-3-030-04221-9_25) | SEED, DEAP [mech: memory] | F1 | Wasserstein critic on marginal features [mech: abstract] | no | no | no |
| DResNet | Ma, Li, Zheng, Lu, *ICONIP* 2019, LNCS [V] — [DOI](https://doi.org/10.1007/978-3-030-36708-4_3) | SEED, SEED-IV [mech: memory] | F1 | Domain *generalization*. Splits weights into shared (unbiased) and per-subject (biased) parts, and uses no target data [mech: abstract] | yes (per-subject weights) | no | Per-subject weights are a precedent for per-domain encoders |
| MS-MDA | Chen, Jin, Li, Fan, Li, He, *Front. Neurosci.* 2021 [V] — [DOI](https://doi.org/10.3389/fnins.2021.778488) | SEED, SEED-IV | F1 | One branch per source subject with marginal MMD [mech: abstract] | **yes** | no | The per-source branches are the closest EEG analogue of per-domain references |
| SOGNN | Li, Li, Pan, Wang, *Front. Neurosci.* 2021 [V] — [DOI](https://doi.org/10.3389/fnins.2021.611653) | SEED, SEED-IV | F1 | Not DA. Builds a per-sample graph [mech: abstract] | no | no | backbone |
| TANN | Li, Fu, Li, Shi, Zheng, *Neurocomputing* 2021 [V] — [DOI](https://doi.org/10.1016/j.neucom.2021.02.048) | 3 public sets (SEED, SEED-IV, MPED [mech: memory]) | F1 | Attention over source samples and regions, driven by domain discriminators, i.e. *transferability* weights. Not class-conditional [mech: abstract] | no | no | Weighting source samples resembles attention-MIL weights |
| PPDA (plug-and-play) | Zhao, Yan, Lu, *AAAI* 2021 [V] — [DOI](https://doi.org/10.1609/aaai.v35i1.16169) | SEED, SEED-IV [mech: memory] | F1 | Shared and private encoders. Not class-conditional. Uses a few target samples to calibrate the private part at test time [mech: memory] | yes (private per subject) | no | Private encoders are a precedent for "two encoders" |
| CLISA | Shen, Liu, Hu, Zhang, Song, *IEEE TAFFC* 2022 [V] — [DOI](https://doi.org/10.1109/taffc.2022.3164516) | THU-EP (80 subj.), SEED | F1 | Contrastive pairs are *same stimulus across subjects*, which is stimulus-conditioned, not emotion-class-conditioned. Needs stimulus-synchronised target data [mech: abstract] | — | no | no |
| MSMRA | Cao, He, Yang, Chen, Li, *Front. Psychol.* 2022 [V] — [DOI](https://doi.org/10.3389/fpsyg.2021.809459) | SEED, SEED-IV | F1 | Multi-representation marginal alignment per source domain [mech: abstract] | **yes** | no | no |
| PGCN | Jin, Du, He, Cai, Li, *IEEE Trans. Multimedia* 2024 [V] — [DOI](https://doi.org/10.1109/tmm.2024.3385676); arXiv 2302.02520 | 3 public sets (SEED, SEED-IV, SEED-V? [mech: memory]) | F1 | Not DA. Local/mesoscopic/global GCN [mech: abstract] | no | no | backbone |
| SSAA (attention alignment) | Yang, Wang, Tao, Liu, Jia, *IEEE TAFFC* 2024 [V] — [DOI](https://doi.org/10.1109/taffc.2024.3394436) | SEED, SEED-IV [mech: memory] | F1 (provisional) | Aligns spectral-spatial *attention* rather than features. The abstract does not mention class-conditioning [mech: abstract, truncated] | **yes** | no | — |
| **DCAN-EEG** | **[UNVERIFIED]**. No EEG method with this name was found on OpenAlex or the web. "DCAN" names vision methods (e.g. Domain Conditioned Adaptation Network [mech: memory]). The user may mean a different acronym. | — | — | — | — | — | — |

**Conditional variants and conditional methods (these continue past F1)**

| Method | Authors, venue, year [tag] | Data | Out at | Why | MS | Bag |
|---|---|---|---|---|---|---|
| MSJDA (conditional variant of the MS-MDA line) | Liang, Su, Fu, Wu, *Front. Hum. Neurosci.* 2022 [V] — [DOI](https://doi.org/10.3389/fnhum.2022.921346) | SEED | **F3** | Joint MMD (JMMD) over (private features ⊗ softmax predictions), so it is conditional through soft predictions. No pseudo-label gating [mech: full text, via the PMC page] | **yes**, K private branches | no |
| JAGP | Peng, Wang, Kong, Nie, Lu, *IEEE TAFFC* 2022 [V] — [DOI](https://doi.org/10.1109/taffc.2022.3189222) | SEED-IV, SEED-V | **F3** | Shared-subspace adaptation plus graph label propagation: the target labels are optimisation variables, so it is joint/conditional. No confidence weighting is stated [mech: abstract] | no | no |
| DIAGLP | Tao, Yan, He, *IEEE Access* 2024 [V] — [DOI](https://doi.org/10.1109/access.2024.3454082) | (SEED family [mech: memory]) | F3 | Soft-label joint-distribution measure plus label propagation [mech: abstract] | no | no |
| BiHADA (conditional variant of BiDANN) | Chen, Meng, *Brain Sciences* 2026 [V] — [DOI](https://doi.org/10.3390/brainsci16050507) | (not in the abstract excerpt) | F3 (borderline) | A "multimodal discriminator" that uses source label structure, i.e. CDAN-like conditioning. The perplexity weights select between *hemisphere classifiers* and do not gate pseudo-labels [mech: abstract] | no | no |
| FSA-Net | Ran, Zhong, Hu, Ye, Zhang, *Tsinghua Sci. & Tech.* 2026 [V] — [DOI](https://doi.org/10.26599/tst.2026.9010060) | SEED, SEED-IV | F3 (borderline) | Coarse MMD, then subdomain LMMD plus contrastive learning, so it is conditional. LMMD's soft class-membership weights are not a confidence gate. The ensemble is weighted by branch adaptation quality [mech: abstract] | aggregated sources | no |
| **DANN-RPLI** (conditional variant of DANN) | Ju, Su, Dai, Wu, Li, *Knowledge-Based Systems* 2025 [V] — [DOI](https://doi.org/10.1016/j.knosys.2025.113368) | SEED, SEED-IV, DEAP | **survives** | Adversarial confusion of *subdomains* (class-conditional). Pseudo-labels are kept only if close to their class centre and stable under perturbation [mech: abstract] | no | no |
| **DDSPR** | Du, Hai, He, Wang, *CogSci* 2025 (eScholarship) [V] — [DOI](https://doi.org/10.48448/9bzc-2b41) | SEED, SEED-IV | **survives F2/F3, F1 not confirmed** | Dynamic source selection with an "agent" domain strategy, plus confidence-based pseudo-label correction. Whether the alignment is class-conditional is not stated [mech: abstract] | **yes** | no |
| **DAMSDAN** | Hu, Wang, Zheng, Yang, Zhou, Li, Sun, Zhang, arXiv 2510.17475, 2025 [V] — [arXiv](https://arxiv.org/abs/2510.17475) | SEED, SEED-IV, FACED | **survives F1/F2, F3 likely** | Prototype-guided conditional alignment with "dual pseudo-label interaction" to improve reliability, and MMD-based source weighting. Explicit confidence weighting is not confirmed [mech: abstract] | **yes** | no |
| **Zhang et al., subdomain + confidence-adjudicated dual-decision pseudo-labelling** | Zhang, He, Liu, Ren, Zhan, *Multimedia Systems* 2026 [V] — [DOI](https://doi.org/10.1007/s00530-026-02247-4) | cross-subject and cross-session (datasets not seen) | **survives (title-level)** | The title states subdomain adaptation plus confidence-adjudicated pseudo-labels. No abstract was retrieved [mech: title only] | ? | no |
| **PR-PL** | Zhou, Zhang, Fu, Zhang, Li, Huang, Li, Yang, Dong, Zhang, Liang, *IEEE TAFFC* 2023 [V] — [DOI](https://doi.org/10.1109/taffc.2023.3288118); arXiv 2202.06509 | SEED, SEED-IV | **F1 (strict) / survives (loose)** | Domain alignment is a *marginal* DANN. Class enters through source-only prototypes and target pairwise pseudo-labels, which are filtered by similarity thresholds that relax over training [mech: full text] | no (pooled source) | no |
| PRISM | Zhou, Zhang, Deng, Yin et al., arXiv 2607.00358, 2026 [V] — [arXiv](https://doi.org/10.48550/arxiv.2607.00358) | — | F2 (likely) | *Semi-supervised* DA ("label-efficient"), with confidence-filtered pseudo-labels. Probably uses some labelled target data [mech: abstract] | — | no |
| SF-UDA (DLAR + LCL) | Imtiaz, Khan, *IEEE TAFFC* 2026 [V] — [DOI](https://doi.org/10.1109/taffc.2026.3713402) | DEAP, SEED, DREAMER (cross-dataset) | F1 (borderline) | Source-free. Prediction consistency on *confident* samples and neighbour consistency, with no class-conditional distribution alignment. It would pass F3 [mech: abstract] | no | no |
| Multi-source dynamic contrastive DA | Xiao, Zhang, Peng, Han, Zheng, *BSPC* 2024 [V] — [DOI](https://doi.org/10.1016/j.bspc.2024.107337) | — | not classified | No abstract retrieved | yes | no |
| CWMAN-EEG | Chen, Zhou, Nan, Cao, Yin, Research Square preprint 2026 [S] | — | not classified | Title only ("conditionally weighted multi-source adversarial") | yes | no |

**Core maths for the methods that matter for positioning**

- **BiDANN / R2G-STNN / BiHDM (DANN family)** [math: memory]: $\min_{f,c}\max_{d}\; \mathcal{L}_{cls}(c(f(x^s)),y^s) - \lambda\,\mathcal{L}_{dom}(d(f(x)),\text{domain})$. BiDANN adds two per-hemisphere discriminators $d_L, d_R$. *In plain words: make features from which no discriminator can tell source from target, ignoring class.*
- **WGAN-DA** [math: memory]: $\min_f \max_{\|h\|_L\le 1}\; \mathbb{E}_{s}[h(f(x))]-\mathbb{E}_{t}[h(f(x))]$. *A Lipschitz critic estimates the Wasserstein-1 distance between marginal feature distributions, and the encoder shrinks it.*
- **DResNet** [math: memory]: $W_k = W_0 + \Delta W_k$, where $W_0$ is the shared, unbiased part and $\Delta W_k$ is subject-specific; only $W_0$ is used at test time. *It keeps what is common across subjects and discards what is personal.*
- **MS-MDA** [math: memory]: $\mathcal{L}=\sum_{k}\big[\mathcal{L}_{cls}^{(k)} + \lambda\,\mathrm{MMD}^2(F_k(E(X^{s_k})),F_k(E(X^t)))\big] + \mathcal{L}_{disc}$. *Each source subject gets its own branch, aligned one-to-one with the target by marginal MMD.*
- **MSJDA** [math: paper, Eq. 4]: $\mathcal{L}_{JA}^{k}=\big\|\tfrac{1}{n_k}\sum_i \phi_1(f^{s}_{k,i})\otimes\phi_2(g^{s}_{k,i}) - \tfrac{1}{n_t}\sum_j \phi_1(f^{t}_{k,j})\otimes\phi_2(g^{t}_{k,j})\big\|^2_{\mathcal{H}_1\otimes\mathcal{H}_2}$. *It matches the joint embedding of (feature, softmax prediction) per source branch, which is conditional alignment by soft labels with no confidence gate.* — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9520599/)
- **JAGP** [math: memory]: $\min_{P,F,S}\ \mathrm{MMD}_{joint}(P^\top X^s,P^\top X^t; F) + \alpha\,\mathrm{tr}(F^\top L_S F) + \beta\|S\|^2$, s.t. $F$ = soft label matrix and $S$ = learned graph. *It learns a projection, a sample graph and propagated target labels together, so the alignment and the labelling improve each other.*
- **TANN** [math: memory]: the source loss is $\sum_i w_i\,\ell(x_i)$, with $w_i = 1 + H(d(f(x_i)))$-style transferability weights from the discriminators. *Source samples the discriminator finds target-like get more weight.*
- **PR-PL** [math: paper, Eqs. 4, 5, 14]: $\mu_c=\tfrac{1}{|X^c_s|}\sum_{x\in X^c_s} f(x)$; $h(f(x),\mu_c)=f(x)^\top S\mu_c$; $r^t_{ij}=1$ if $\hat l^t_i\cdot\hat l^t_j\ge\tau_u$, $0$ if $<\tau_l$, otherwise excluded. The thresholds $\tau_h^0=0.9$ and $\tau_l^0=0.5$ relax non-linearly with epoch. *Features are compared with source class prototypes, and only target pairs that are confidently same-class or confidently different-class train the target loss.* — [arXiv PDF](https://arxiv.org/pdf/2202.06509)
- **DANN-RPLI** [math: memory, reconstructed from the abstract only]: keep $\hat y_j$ only if $\|z_j-\mu_{\hat y_j}\|$ is small and $\hat y_j$ is unchanged under perturbation; then apply class-wise (subdomain) adversarial confusion $\sum_c \mathcal{L}_{adv}^{(c)}$ over the retained samples. *Only stable, central pseudo-labels define the target subdomains that are aligned.*
- **FSA-Net (LMMD part)** [math: memory]: $\mathrm{LMMD}=\tfrac1C\sum_c\big\|\sum_i w^{s,c}_i\phi(z^s_i)-\sum_j w^{t,c}_j\phi(z^t_j)\big\|^2$, where $w^{t,c}_j=\hat p_{jc}/\sum_{j'}\hat p_{j'c}$. *A per-class MMD in which target samples count in proportion to their soft class probability, a soft assignment rather than a confidence filter.*

### Inferences
- Within affect EEG, F1 removes almost all the canonical baselines (DGCNN, BiDANN, BiHDM, R2G-STNN, WGAN-DA, DResNet, MS-MDA, MSMRA, SOGNN, TANN, CLISA, PGCN). The strongest *conditional* baselines that a MIL-CREDA paper should beat are therefore MSJDA (multi-source, joint) and JAGP. For a full-funnel competitor, use DANN-RPLI and/or DAMSDAN (2025). Both are instance-level and published on SEED/SEED-IV.
- The multi-source line (MS-MDA, MSJDA, MSMRA, DDSPR, DAMSDAN, FSA-Net) treats each subject as a **domain over segments**. MIL-CREDA treats each subject as a **sample (bag) inside a domain**. No method in this table changes that unit.
- MIL-CREDA's "personalised source reference" (local term) has its nearest EEG relatives in per-source branches (MS-MDA, MSJDA) and source weighting or selection (DAMSDAN's MMD weights, DDSPR's dynamic selection, FSA-Net's affinity aggregation). All of these work at domain level on segment features, not as an RKHS distance between bag embeddings.

### Gaps
- **DCAN-EEG** could not be identified. Ask the user for the full title.
- For DAMSDAN, DDSPR and Zhang et al. 2026, whether F1 or F3 holds rests on abstracts or titles. Full texts were not read; the Zhang et al. text is paywalled (Springer).
- AdaTF-HyperMIL (Liu, Song, Research Square 2026 [S]) is titled "Hyperbolic Multiple-Instance Learning for **Subject-Independent EEG Emotion Recognition**". It is the only EEG-emotion MIL item found, but the page returned only its title. Whether it does DA, or treats subjects or trials as bags, is unknown. **This is the most important item to read before claiming novelty.** — [Research Square](https://www.researchsquare.com/article/rs-11043508/v1)
- Datasets marked [mech: memory] in the tables were not confirmed.

---

## Q2. BCI / motor-imagery DA (EA, RA, RPA, MEKT, DJDAN and others): where do they leave?

### Takeaway
The Riemannian/Euclidean alignment family (EA, RA, RPA, and SPDIM's re-centring) aligns **per-subject marginal second-order statistics**. It leaves at F1, and RPA's rotation step also needs target labels (F2). The conditional variants are **MEKT** (joint-distribution adaptation with pseudo-labels) and **DJDAN** (prediction-conditioned local discriminators). Both leave at F3, because neither gates by confidence. I found no BCI method that passes all three filters at the level of a well-cited paper.

### Cited Findings

| Method | Authors, venue, year [tag] | Data | Out at | Why | MS | Bag / set-level |
|---|---|---|---|---|---|---|
| RA (Riemannian re-centring) | Zanini, Congedo, Jutten, Said, Berthoumieu, *IEEE TBME* 65(5) 2018 [S, via search on HAL/OpenAlex link] — [HAL](https://hal.archives-ouvertes.fr/hal-01923278) | MI, ERP | F1 | Re-centres each session or subject's covariances at the identity [mech: memory] | per-subject | **yes, per-subject statistic** |
| RPA | Rodrigues, Jutten, Congedo, *IEEE TBME* 2018 (issue 2019) [V] — [DOI](https://doi.org/10.1109/tbme.2018.2889705) | 8 datasets, 243 subjects | F1, and F2 for the rotation step | Re-centring, stretching, then a rotation that matches class means, which needs some labelled target trials [mech: memory] | per-subject | **yes** |
| EA | He, Wu, *IEEE TBME* 2019 (vol. 67(2) 2020 [mech: memory]) [V] — [DOI](https://doi.org/10.1109/tbme.2019.2913914) | MI, ERP | F1 | Unsupervised whitening by each subject's mean covariance [mech: abstract] | per-subject | **yes** |
| DRDA | Zhao, Zheng, Ma, Li, Zheng, *IEEE TNNLS* 2020 [V] — [DOI](https://doi.org/10.1109/tnnls.2020.3010780) | BCI-IV-2a/2b [mech: memory] | F1 | Adversarial marginal alignment plus source center loss [mech: abstract] | pooled | no |
| WDA-MI (Wasserstein) | She et al., *IEEE TNSRE* 2023 [V] — [DOI](https://doi.org/10.1109/tnsre.2023.3241846) | MI | F1 | WGAN-style marginal alignment [mech: abstract] | pooled | no |
| T-TIME | Li, Wang, Luo, Ding, Wu, *IEEE TBME* 2023 [V] — [DOI](https://doi.org/10.1109/tbme.2023.3303289) | MI | F1 | Online test-time information maximisation over an ensemble, with no distribution alignment [mech: abstract] | — | EA per stream |
| SPDIM | Li, Kawanabe, Kobler, *ICLR* 2025 (arXiv 2411.07249) [V for arXiv; venue from the arXiv comment] — [arXiv](https://arxiv.org/abs/2411.07249) | BCI and sleep staging | F1 (borderline) / F3 | Source-free. One SPD bias parameter per target domain, learned by information maximisation. Corrects label shift that re-centring breaks. No pseudo-label weighting [mech: abstract] | per-domain | **yes, per-domain parameter** |
| **MEKT** (conditional variant of RA/EA) | Zhang, Wu, *IEEE TNSRE* 28(5) 2020 [V] — [DOI](https://doi.org/10.1109/tnsre.2020.2985996) | 4 datasets, MI and ERP | **F3** | Riemannian alignment, tangent space, then joint (marginal + conditional, via target pseudo-labels) distribution minimisation. Pseudo-labels are not weighted [mech: abstract; the no-weighting part is mech: memory] | yes (+ DTE source selection) | per-subject statistic |
| **DJDAN** (conditional variant of DANN-for-MI) | Hong, Zheng, Liu, Chen, Ma, Gao, Zheng, *IEEE TNSRE* 2021 [V] — [DOI](https://doi.org/10.1109/tnsre.2021.3059166) | BCI-IV-2a, 2b | **F3** | A global discriminator (marginal) plus local discriminators conditioned on predicted labels (conditional), with a dynamic factor balancing the two. Confidence gating is not stated [mech: abstract] | no (cross-session) | no |

**Core maths**
- **EA** [math: memory]: $\bar R=\tfrac1n\sum_i X_iX_i^\top$, $\tilde X_i=\bar R^{-1/2}X_i$. *Every subject's average spatial covariance becomes the identity, so subjects share a reference point.*
- **RA / RPA** [math: memory]: $\tilde C_i = M^{-1/2} C_i M^{-1/2}$, where $M$ is the Riemannian (Karcher) mean of subject $k$'s covariances. RPA then applies stretching $\tilde C_i^{\,p}$ and a rotation $U^\top \tilde C_i U$, with $U$ chosen to match class means. *Re-centre, rescale and rotate each subject's cloud of covariance matrices onto the source cloud; the rotation is the step that needs labels.*
- **MEKT** [math: memory]: $\min_{A,B}\ \mathrm{tr}\big(\begin{bmatrix}A\\B\end{bmatrix}^{\!\top}(\textstyle\sum_{c=0}^{C} M_c)\begin{bmatrix}A\\B\end{bmatrix}\big)+\text{(source discriminability, geometry terms)}$, where $M_0$ is the marginal MMD matrix and $M_{c\ge1}$ are class-conditional MMD matrices built from target pseudo-labels. *A JDA-style projection pair that matches both marginal and per-class means in tangent space.*
- **DJDAN** [math: memory]: $\mathcal{L}=\mathcal{L}_{cls}-\lambda\big[(1-\omega)\mathcal{L}_{glob}+\omega\sum_c \mathcal{L}^{(c)}_{loc}(\hat p_c\, f(x))\big]$, with $\omega$ updated from the two discriminators' errors. *Like MADA/DAAN: one discriminator per class, fed with features weighted by predicted class probability.*
- **SPDIM** [math: memory]: $\max_{\theta_t}\ H(\bar p_t) - \tfrac1n\sum_j H(p(\,\cdot\,|\log_{\theta_t}(C_j)))$, with $\theta_t\in\mathcal{S}_{++}$. *It picks the target re-centring point that makes predictions confident individually and diverse as a whole.*

### Inferences
- The EA/RA/RPA family is the strongest precedent for **subject-level (set-level) statistics** in EEG. Each subject is summarised by one mean covariance, which is a second-order set statistic, and aligned by it. That is structurally close to MIL-CREDA's "bag = subject" view, but it is marginal, unweighted and has no class conditioning. MIL-CREDA's per-class covariance-operator score (Cauchy–Schwarz / Rényi-2) can be presented as **a class-conditional, RKHS, attention-weighted generalisation of subject-wise covariance re-centring**. That framing is an inference, not a published claim.
- SPDIM shows that per-subject re-centring *fails under label shift*. That is a concrete published motivation for conditioning subject-level alignment on class, which is what MIL-CREDA does.

### Gaps
- I did not find a well-cited MI/BCI method that passes F3, i.e. one with confidence-gated pseudo-labels plus conditional alignment. Candidate search hits (FUSED consensus filtering, "Synchronized Self-training for cross-domain MI", Aberdeen) were seen only as snippets [S] and were not classified — [Aberdeen](https://aura.abdn.ac.uk/items/6fd12cf6-718e-464d-b4fc-a7c011addac7/full).
- Zanini et al. 2018 was confirmed through a HAL link and the OpenAlex ID in the search results, not by a direct OpenAlex query.

---

## Q3. MIL and set-level DA: where do they leave?

### Takeaway
No MIL or set-level DA method passes the funnel. Hashimoto and Wang & Zhao are marginal (F1). The class-aware MIL-DA methods (Yamaguchi 2025, Takahama 2023) use **target bag labels** (F2). DETA's "category-level" alignment has no stated pseudo-label gating (F3). Marginal transfer learning (Blanchard) uses a domain's KME as a set-level input, but it is DG and not conditional (F1).

### Cited Findings

| Method | Authors, venue, year [tag] | Data | Out at | Why | Bag / set | MIL-ext |
|---|---|---|---|---|---|---|
| Attention-MIL (reference, not DA) | Ilse, Tomczak, Welling, *ICML* 2018 (arXiv 1802.04712) [V] — [arXiv](https://doi.org/10.48550/arxiv.1802.04712) | MNIST-bags, histology | S0 reference | Bag = attention-weighted mean of instance embeddings | **yes** | — |
| Hashimoto MS-DA-MIL | Hashimoto, Fukushima, Koga, Takagi, Ko, et al., *CVPR* 2020 [V] — [DOI](https://doi.org/10.1109/cvpr42600.2020.00391) | malignant lymphoma WSIs, multi-hospital | F1 | Domain-adversarial training on instance features, marginal (see prior note) | yes (WSI = bag) | native |
| MIDA (Wang & Zhao) | Wang, Zhao, *WWW* 2018 [V] — [DOI](https://doi.org/10.1145/3178876.3186051) | VAERS → social media | F1 | Bag-level generalised MMD built from instance kernels. Marginal; no pseudo-labels | **yes (user = bag)** | native |
| Takahama DA-MIL | Takahama, Kurose, Mukuta, Abe, Yoshizawa, et al., *ISBI* 2023 [V] — [DOI](https://doi.org/10.1109/isbi53787.2023.10230546) | pathology | **F2** | Target has slide (bag) labels. Uses high-confidence instance pseudo-labels, so it would pass F3 | yes | native |
| Yamaguchi UC severity | Yamaguchi, Iwana, Bise, Harada, Okuo, et al., arXiv 2509.14573, 2025 (MLMI-MICCAI workshop per prior note) [V] — [arXiv](https://doi.org/10.48550/arxiv.2509.14573) | endoscopy, multi-hospital | **F2** | Class-wise alignment (shared aggregation tokens, max-severity triplet loss), driven by **target patient-level labels** | yes (patient = bag) | native |
| DETA | Shou, Yan, Yuan, Cao, Zhao, et al., arXiv 2411.14001, 2024 [V] — [arXiv](https://doi.org/10.48550/arxiv.2411.14001) | WSI survival, multi-hospital | **F3** (provisional) | Category-level coupling of two branches plus adversarial perturbation. No pseudo-label confidence gating in the abstract | yes (graph per WSI) | native |
| Marginal transfer learning / DG | Blanchard, Deshmukh, Dogan, Lee, Scott, arXiv 1711.07910 (JMLR 2021 [mech: memory]) [V for arXiv] — [arXiv](https://doi.org/10.48550/arxiv.1711.07910) | various | F1 (and DG, not DA) | Adds the KME of each domain's *marginal* as an input feature | **set-level (domain KME)** | Conceptual |
| BridgeMIL, subject-specific encoders | see prior note | — | not DA / F1 | — | subject = bag (BridgeMIL) | — |

**Core maths**
- **Attention-MIL** [math: memory]: $z_B=\sum_{k} a_k h_k$, $a_k=\dfrac{\exp\{w^\top\tanh(Vh_k)\}}{\sum_j \exp\{w^\top\tanh(Vh_j)\}}$. *The bag is a learned weighted average of its instances. With a kernel feature map this is MIL-CREDA's Ψ(B).*
- **MIDA (Wang & Zhao)** [math: memory]: $D(\mathcal{S},\mathcal{T})=\big\|\tfrac{1}{|\mathcal{S}|}\sum_{B\in\mathcal S}\mu_B-\tfrac{1}{|\mathcal T|}\sum_{B\in\mathcal T}\mu_B\big\|^2_{\mathcal H}$, where $\mu_B=\tfrac1{|B|}\sum_{x\in B}\phi(x)$ and the outer kernel is induced by "mixed instance kernels". *An MMD between domains whose samples are bags, each bag being an unweighted mean embedding.*
- **Hashimoto** [math: memory]: $\min_{f,g}\max_d\ \mathcal{L}_{MIL}(g(\mathrm{pool}_B f(x)),Y_B)-\lambda\sum_{x}\mathcal{L}_{dom}(d(f(x)))$. *Bag-level MIL loss plus instance-level gradient reversal.*
- **Marginal transfer** [math: memory]: $\hat f(x,P_X)$ with kernel $k((P,x),(P',x'))=K_P(\mu_P,\mu_{P'})\,k_X(x,x')$. *The classifier also sees an embedding of the whole domain, so a "set" conditions each prediction.*

### Inferences
- MIL-CREDA is the only design found that sits **inside** all three filters *and* uses bags. Its nearest single-property neighbours are: bag KME (Wang & Zhao, F1); class-wise bag alignment (Yamaguchi, F2); confidence pseudo-labels in MIL-DA (Takahama, F2); category-level alignment on WSI graphs (DETA, F3).

### Gaps
- A dedicated pass for 2025–2026 slide-level UDA with an unlabelled target (e.g. under the names "cross-site MIL" or "stain-shift MIL UDA") found nothing new on OpenAlex. The MIL-DA literature may still exist under other names.

---

## Q4. Survivors, and where MIL-CREDA sits

### Takeaway
Full-funnel survivors (F1 ✓, F2 ✓, F3 ✓): **DANN-RPLI (KBS 2025)**, **Zhang et al. 2026 (Multimedia Systems; title-level)**, **DAMSDAN (arXiv 2025; F3 likely)**, **DDSPR (CogSci 2025; F1 unconfirmed)**, PR-PL only under a loose F1, and CREDA (Mathematics 2025, from the prior note [S]). **All are instance-level, none uses a subject-as-bag or set-level representation, and none uses matrix-Rényi or Cauchy–Schwarz scores.** MIL-CREDA's distinctive combination is: survives the funnel + the bag (subject) is the unit + the per-class score uses covariance operators + a personalised RKHS reference. In this run that combination remains unoccupied.

### Cited Findings
- Survivors and their evidence: DANN-RPLI [V, mech: abstract] — [DOI](https://doi.org/10.1016/j.knosys.2025.113368); DAMSDAN [V, mech: abstract] — [arXiv](https://arxiv.org/abs/2510.17475); DDSPR [V, mech: abstract] — [DOI](https://doi.org/10.48448/9bzc-2b41); Zhang et al. 2026 [V, title] — [DOI](https://doi.org/10.1007/s00530-026-02247-4); PR-PL [V, mech: full text] — [arXiv](https://arxiv.org/pdf/2202.06509).
- Survivor CREDA (instance-level: kernel class-conditional alignment, matrix Rényi-2 entropy, down-weighting of low-confidence pseudo-labels; Digits, ImageCLEF, Office-31) — [DOI](https://doi.org/10.3390/math13162602) [S, from prior note].

**MIL-CREDA against the funnel, component by component** (descriptive, from the user's spec):

| MIL-CREDA component | Funnel role | Closest published precedent [tag] | Difference |
|---|---|---|---|
| Ψ(B) = attention-weighted RKHS mean embedding, bag = subject | none (representation) | Ilse 2018 [V]; Wang & Zhao 2018 [V]; EA/RA per-subject means [V] | Attention + kernel + bag = subject in EEG DA: no precedent found |
| Two encoders, frozen source | none | DResNet shared/private [V]; PPDA private encoders [V]; MS-MDA branches [V] | Frozen-source plus a target encoder is closer to source-free setups (SPDIM [V], Imtiaz 2026 [V]) |
| Hard per-class routing, $w = 1 - H_2(g)/\ln C$ | F3 | PR-PL thresholds [V]; DANN-RPLI reliability filter [V]; CREDA entropy weighting [S] | Rényi-2 (collision) entropy as the confidence weight, applied to **bags** |
| Per-class matrix-Rényi-2 / CS divergence between covariance operators | F1 | MSJDA JMMD (means only) [V]; MEKT class-conditional MMD [V]; LMMD (FSA-Net) [V]; CREDA [S] | Second-order (operator) and per class at bag level, where the precedents use first-order means at instance level |
| Confidence gate | F3 | PR-PL dynamic thresholds [V]; DDSPR confidence correction [V] | — |
| Local RKHS distance to a personalised source reference | (local term) | MS-MDA per-source branches [V]; DAMSDAN MMD source weights [V]; MEKT-DTE source selection [V] | Per-target-bag reference in RKHS, where the precedents work per domain on segments |

### Inferences
- Positioning sentence for the paper (inference): *"Existing conditional, pseudo-label-gated EEG DA methods (DANN-RPLI, DAMSDAN, DDSPR) align segment-level distributions and treat subjects as domains. Subject-level alignment in BCI (EA/RA/RPA) uses one marginal covariance per subject. MIL-DA (Hashimoto, Yamaguchi, Takahama) is either marginal or needs target bag labels. MIL-CREDA is the first, to our search, to perform class-conditional, confidence-gated alignment over subject-level bag embeddings, using an information-theoretic covariance-operator criterion."*
- For baselines, it is defensible to run one method per funnel stage: DANN or BiDANN (marginal, F1), MS-MDA (multi-source marginal, F1), MSJDA or JAGP (conditional, F3-out), DANN-RPLI or DAMSDAN (full survivor), and EA as a subject-level-statistic baseline.

### Gaps
- Before the novelty claim can be made, these must be read in full: AdaTF-HyperMIL (EEG emotion MIL, 2026), Zhang et al. 2026, DAMSDAN, and DDSPR.
- **DCAN-EEG** is [UNVERIFIED].
- Every [math: memory] equation needs checking against its paper before it goes into the manuscript. Only PR-PL (Eqs. 4, 5, 14) and MSJDA (Eq. 4) were read from source this run.
