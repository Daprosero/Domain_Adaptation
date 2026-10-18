# Adversarial domain adaptation through the conditional funnel: where MIL-CREDA stands

Run date: 2026-10-10. Conventions used throughout:
- **[V]** means the title, venue and year were confirmed by a search or fetch in this run. Author lists are confirmed only where the cited source names them.
- **[authors from memory]** means the venue and year were verified but the author list was not seen in a source during this run.
- **[math from memory]** means the equations were written from recall of the paper, not read from the PDF in this run. Read every equation as a sketch to be checked against the paper before it is quoted in a manuscript.
- **[UNVERIFIED]** means a claim had no source in this run.
- Notation: G or F is the feature extractor, C the classifier, D a domain discriminator, ŷ = softmax(C(G(x))), H(·) Shannon entropy, and GRL the gradient reversal layer.

Funnel: **F1** asks whether the method conditions on class. **F2** asks whether it works without target labels. **F3** asks whether it weights or filters target samples by confidence. "Discarded at Fk" means the method failed filter k.

---

## Key Question 1: Which adversarial methods survive the three-filter funnel?

### Takeaway
Only one canonical adversarial method clearly passes all three filters: **CDAN+E** (Long et al., NeurIPS 2018), with entropy weight w = 1 + e^{−H(ŷ)}. Two less canonical methods also pass. **SRADA** (IJCAI 2020) weights in the *opposite* direction, giving more adversarial force to high-entropy samples. **BACG** (arXiv 2023) uses evidential uncertainty, but the paper has been **withdrawn**. Most "class-aware" adversarial methods drop out at F3: MADA, DAAN, SymNets, DADA, MDD, MCD, SWD, DALN, ToAlign and NOUN all use soft routing or pseudo-labels but never weight by confidence. Wrapper methods (BSP, GVB, SDAT, ELS, GH/GH++, RADA) are not conditional on their own. They pass the funnel only when stacked on CDAN+E, and then they inherit its F3 status.

### Funnel table

| # | Method | Venue / year | Family | F1 conditional? | F2 no target labels? | F3 confidence weighting? | Stage reached |
|---|---|---|---|---|---|---|---|
| 1 | DANN / RevGrad | ICML 2015 [V] | Feature-level GRL | No (marginal P(Z)) | — | — | **Discarded F1** → variants: CDAN, MADA |
| 2 | ADDA | CVPR 2017 [V] | Untied encoders + GAN loss | No (unconditional D) | — | — | **Discarded F1** (no verified published conditional ADDA variant) |
| 3 | CoGAN | NIPS 2016 [V] | Weight-shared generative pair | No (joint image distribution over domains, not over (Z,Y)) | — | — | **Discarded F1** |
| 4 | PixelDA | CVPR 2017 [V] | Pixel-level GAN | No | — | — | **Discarded F1** |
| 5 | CyCADA | ICML 2018 [V] | Pixel + feature GAN, cycle consistency | No (its semantic consistency loss keeps labels fixed through translation; it does not align P(Z\|Y)) | — | — | **Discarded F1** (borderline) |
| 6 | MADA | AAAI 2018 [V] | One discriminator per class | Yes (K discriminators, soft routing by ŷ_k) | Yes | No (ŷ_k routes a sample to class k; it is not a confidence weight) | **Discarded F3** (borderline) |
| 7 | CDAN | NeurIPS 2018 [V] | Multilinear conditioning f⊗g | Yes (D sees f⊗ŷ) | Yes | No | **Discarded F3** → variant CDAN+E |
| 7b | **CDAN+E** | NeurIPS 2018 [V] | CDAN + entropy conditioning | Yes | Yes | **Yes**: w = 1+e^{−H(ŷ)} (code-verified) | **SURVIVOR** |
| 8 | MCD | CVPR 2018 [V] | Two classifiers act as critic | Implicit (the critic lives in prediction space) | Yes | No | **Discarded F3** |
| 9 | SWD | CVPR 2019 [V] | MCD with a sliced-Wasserstein discrepancy | Implicit | Yes | No | **Discarded F3** |
| 10 | BSP | ICML 2019 [V] | Spectral penalty on top of DANN/CDAN | No on its own | — | — | **Discarded F1** → BSP+CDAN(+E) inherits CDAN's status |
| 11 | DADA | AAAI 2020 [V] | Integrated (K+1)-way classifier | Yes (joint alignment through the shared classifier) | Yes | No [UNVERIFIED that no weighting exists] | **Discarded F3** |
| 12 | CAT | ICCV 2019 [V] | Teacher pseudo-labels + cluster alignment (+ RevGrad marginal) | Yes (aligns class-conditional clusters) | Yes | Not confirmed [UNVERIFIED] | **Discarded F3 (tentative)** |
| 13 | GVB (GVB-GD) | CVPR 2020 [V] | Bridge layers on G and D | No (base is DANN-style) | — | — | **Discarded F1** |
| 14 | DAAN | ICDM 2019 [V] | Global D + C local Ds, dynamic factor ω | Yes (local Ds routed by ŷ) | Yes | No (soft routing, like MADA) | **Discarded F3** |
| 15 | RADA | ICCV 2021 [V] | Relabels domain labels by D-entropy | No (the entropy is over *domain* predictions) | — | — | **Discarded F1** (it weights by confidence, but on the wrong variable) |
| 16 | SymNets | CVPR 2019 [V] | Symmetric source/target classifiers, category-level confusion | Yes | Yes | No [UNVERIFIED] | **Discarded F3** |
| 17 | TADA | AAAI 2019 [V] | Region and image attention from D-entropy | No (attention comes from domain entropy) | — | — | **Discarded F1** |
| 18 | AFN | ICCV 2019 [V] | Feature-norm adaptation (not adversarial) | No | — | — | **Discarded F1** (also outside the family) |
| 19 | SDAT | ICML 2022 [V] | SAM on the task loss inside DAT | No on its own (it is an optimizer) | — | — | **Discarded F1** → SDAT on CDAN+E inherits survivor status |
| 20 | f-DAL | ICML 2021 [V] | f-divergence through an auxiliary hypothesis | Implicit (prediction space) | Yes | No | **Discarded F3** |
| 21 | MDD | ICML 2019 [V] | Margin disparity, adversarial head h′ against argmax h | Implicit (the critic is conditioned on hard predictions) | Yes | No | **Discarded F3** |
| 22 | DALN | CVPR 2022 [V] | Classifier reused as discriminator, nuclear-norm Wasserstein discrepancy (NWD) | Implicit (works on the prediction matrix) | Yes | No | **Discarded F3** |
| 23 | ToAlign | NeurIPS 2021 [V] | Class-prior feature decomposition + DANN/HDA-style alignment | Partial (source side only, through the ground-truth class) | Yes | No | **Discarded F3** |
| 24 | CGDM | CVPR 2021 [V] | Bi-classifier adversarial training + gradient discrepancy with clustered pseudo-labels | Yes | Yes | Not confirmed [UNVERIFIED] | **Discarded F3 (tentative)** |
| 25 | SRADA | IJCAI 2020 [V] | Adversarial loss reweighted by conditional entropy + triplet loss | Yes | Yes | **Yes, but inverted**: high-entropy samples get more weight | **SURVIVOR (inverted weighting)** |
| 26 | NOUN / PRONOUN | IEEE TIP 2021 [V] | Normalized-output / prototype conditioner for D | Yes | Yes | No [UNVERIFIED] | **Discarded F3** |
| 27 | InAIt ("Infer, Align and Iterate") | ICCV 2021 [V] | Adversarial alignment of p(x\|y) with label-shift inference | Yes | Yes | Not confirmed [UNVERIFIED] | **Discarded F3 (tentative)** |
| 28 | BACG | arXiv 2023, **withdrawn** [V] | Two discriminators + class-level gradient alignment + evidential uncertainty | Yes | Yes | Yes (Dirichlet / evidential) | **SURVIVOR on paper, but withdrawn and unreviewed** |
| 29 | ELS | ICLR 2023 [V] | Smooths the domain labels inside DAT | No on its own | — | — | **Discarded F1** (a wrapper) |
| 30 | GH / GH++ | arXiv 2024 [V] (journal status [UNVERIFIED]) | Gradient-angle harmonization | No on its own | — | — | **Discarded F1** (a wrapper) |

### Cited Findings
- TLlib's benchmark averages (ResNet-50) for Office-31 / Office-Home / VisDA-2017: DANN 86.1 / 65.2 / 74.9; ADDA 87.3 / 65.6 / 74.3; CDAN 87.7 / 68.8 / 76.5; MCD 85.4 / 67.8 / 76.7; MDD 89.6 / 69.7 / 79.8; BSP 87.8 / 67.6 / 76.2; AFN 88.6 / 68.2 / 74.4; MCC (not adversarial) 89.6 / 72.4 / 80.4 — [TLlib image-classification README](https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/examples/domain_adaptation/image_classification/README.md)
- TLlib lists DANN (ICML 2015), ADDA (CVPR 2017), CDAN (NeurIPS 2018), MCD (CVPR 2018), MDD (ICML 2019), BSP (ICML 2019), AFN (ICCV 2019) and CyCADA (ICML 2018) — [TLlib repo](https://github.com/thuml/Transfer-Learning-Library)
- The official CDAN code reverses the gradient *through* the entropy weight: `softmax_output = input_list[1].detach()`, then `entropy.register_hook(grl_hook(coeff))`, then `entropy = 1.0+torch.exp(-entropy)`. Each domain's weights are normalized by their sum, and the sum is detached — [thuml/CDAN loss.py](https://raw.githubusercontent.com/thuml/CDAN/master/pytorch/loss.py)
- TLlib's reimplementation of CDAN instead **detaches** both the prediction and the weight: `g = F.softmax(g, dim=1).detach()`, `weight = 1.0 + torch.exp(-entropy(g))`, normalized as `weight / sum * batch_size` — [TLlib cdan.py](https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/tllib/alignment/cdan.py)
- SRADA "reduce[s] the weight of the adversarial loss for aligned features while increasing the adversarial force for those poorly aligned measured by the conditional entropy" (Wang & Zhang, IJCAI 2020) — [arXiv 2006.00223](https://arxiv.org/abs/2006.00223)
- BACG uses two discriminators, class-level gradient alignment, pseudo-labels from nearest neighbours and evidential (Dirichlet) uncertainty. The arXiv paper is withdrawn "due to issues including uncited figures" — [arXiv 2310.13959](https://arxiv.org/abs/2310.13959)

### Inferences
- The survivor set is very small. In adversarial DA, confidence weighting of *class-conditional* alignment comes down essentially to CDAN+E's single scalar weight. Most class-aware methods use soft predictions as routing coefficients (MADA, DAAN) or as conditioning inputs (CDAN, NOUN), not as reliability weights.
- Several methods weight by confidence on the *wrong variable*: RADA and TADA use the entropy of the domain discriminator. That is a different quantity from MIL-CREDA's class-posterior confidence and should not be cited as precedent for it.
- SRADA's inverted direction matters for positioning. It argues that hard samples should get *more* alignment force, while CDAN+E and MIL-CREDA give confident samples more weight. That disagreement is a real design axis, not noise.

### Gaps
- CAT, CGDM, InAIt, SymNets, DADA and NOUN: I did not read the PDFs in this run, so I could not confirm whether they threshold or weight pseudo-labels. Their F3 verdicts are tentative.
- Whether RADA, GVB and SDAT report numbers specifically on top of CDAN+E, which would create inheriting survivor variants, was not checked [UNVERIFIED].
- The journal venue of GH/GH++ (possibly TPAMI) was not confirmed. The ICCV 2023 GH conference version is likewise [UNVERIFIED].

---

## Key Question 2: Per-method records (core math, conditioning type, weighting, MIL extensibility)

### Takeaway
The adversarial family solves alignment as a min–max game with a learned critic. Class conditioning enters in three ways: (a) per-class critics routed softly (MADA, DAAN); (b) critics that take predictions as inputs (CDAN, NOUN, DADA); (c) critics built from classifier outputs, i.e. hypothesis discrepancy (MCD, SWD, MDD, f-DAL, DALN). Only (b) with entropy (CDAN+E) weights samples by confidence. ADDA is the closest *architectural* relative of MIL-CREDA: separate encoders with the source frozen. It is *unconditional*, however.

### Cited Findings (bibliographic verification)
- DANN: "Unsupervised Domain Adaptation by Backpropagation", ICML 2015 [authors from memory: Ganin & Lempitsky] — [TLlib](https://github.com/thuml/Transfer-Learning-Library)
- ADDA: Tzeng, Hoffman, Saenko, Darrell, CVPR 2017. It "combines discriminative modeling, untied weight sharing, and a GAN loss" — [CVF open access](https://openaccess.thecvf.com/content_cvpr_2017/html/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.html); [arXiv 1702.05464](https://ar5iv.labs.arxiv.org/html/1702.05464)
- CoGAN: Liu & Tuzel, NIPS 2016. Weight sharing learns "a joint distribution of images in the two domains from images drawn separately from the marginal distributions" — [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2016/hash/502e4a16930e414107ee22b6198c578f-Abstract.html)
- PixelDA: Bousmalis, Silberman, Dohan, Erhan, Krishnan, CVPR 2017 — [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.html)
- CyCADA: ICML 2018 [authors from memory: Hoffman, Tzeng, Park, Zhu, Isola, Saenko, Efros, Darrell] — [TLlib](https://github.com/thuml/Transfer-Learning-Library)
- MADA: Pei, Cao, Long, Wang, AAAI 2018. It uses "multiple domain discriminators" for "fine-grained alignment" — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11767)
- CDAN: "Conditional Adversarial Domain Adaptation", NeurIPS 2018 [authors from memory: Long, Cao, Wang, Jordan] — [TLlib](https://github.com/thuml/Transfer-Learning-Library); code — [thuml/CDAN](https://raw.githubusercontent.com/thuml/CDAN/master/pytorch/loss.py)
- MCD: CVPR 2018 [authors from memory: Saito, Watanabe, Ushiku, Harada] — [TLlib](https://github.com/thuml/Transfer-Learning-Library)
- SWD: Lee, Batra, Baig, Ulbricht (Apple), CVPR 2019 — [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Sliced_Wasserstein_Discrepancy_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html)
- BSP: "Transferability vs. Discriminability: Batch Spectral Penalization…", ICML 2019 [authors from memory: Chen, Wang, Long, Wang] — [TLlib](https://github.com/thuml/Transfer-Learning-Library)
- DADA: Tang & Jia, AAAI 2020, 34(04): 5940–5947. "Integrated category and domain classifier"; "mutually inhibitory relation between category and domain predictions" — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/6054)
- CAT: Deng, Luo, Zhu, ICCV 2019, pp. 9944–9953. A teacher "discover[s] the class-conditional structure"; aligns "corresponding clusters across domains" — [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Deng_Cluster_Alignment_With_a_Teacher_for_Unsupervised_Domain_Adaptation_ICCV_2019_paper.html)
- GVB: Cui, Wang, Zhuo, Su, Huang, Tian, CVPR 2020 — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Cui_Gradually_Vanishing_Bridge_for_Adversarial_Domain_Adaptation_CVPR_2020_paper.html)
- DAAN: Yu, Wang, Chen, Huang, ICDM 2019. It evaluates "the relative importance of global and local domain distributions" — [arXiv 1909.08184](https://arxiv.org/pdf/1909.08184); [PwC](https://paperswithcode.com/paper/transfer-learning-with-dynamic-adversarial)
- RADA: Jin et al., ICCV 2021. It relabels "well-aligned target domain samples as source domain samples", choosing them by the entropy of the domain classifier — [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Jin_Re-Energizing_Domain_Discriminator_With_Sample_Relabeling_for_Adversarial_Domain_Adaptation_ICCV_2021_paper.html)
- SymNets: Zhang, Tang, Jia, Tan, CVPR 2019. Uses a "category-level confusion loss" — [arXiv 1904.04663](https://arxiv.org/abs/1904.04663)
- TADA: Wang, Li, Ye, Long, Wang, AAAI 2019. Local attention from "multiple region-level domain discriminators" and global attention from "a single image-level domain discriminator" — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4472)
- AFN: "Larger Norm More Transferable…", ICCV 2019 [authors from memory: Xu, Li, Yang, Lin] — [TLlib README](https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/examples/domain_adaptation/image_classification/README.md)
- SDAT: Rangwani, Aithal, Mishra, Jain, Babu, ICML 2022. Smooth minima of the task loss stabilize DAT, while smooth minima of the adversarial loss hurt — [ICML](https://icml.cc/virtual/2022/poster/17769); code [val-iisc/SDAT](https://github.com/val-iisc/SDAT)
- f-DAL: Acuna, Zhang, Law, Fidler, ICML 2021. A discrepancy based on the "variational characterization of f-divergences", which recovers Ben-David et al. as a special case — [NVIDIA](https://research.nvidia.com/labs/toronto-ai/fDAL); [ICML](https://icml.cc/virtual/2021/poster/9515)
- MDD: "Bridging Theory and Algorithm for Domain Adaptation", ICML 2019 [authors from memory: Zhang, Liu, Long, Jordan] — [TLlib](https://github.com/thuml/Transfer-Learning-Library)
- DALN: Chen, Chen, Wei, Jin, Tan, Jin, Chen, CVPR 2022, pp. 7181–7190. The "category classifier is reused as a discriminator"; NWD satisfies the K-Lipschitz constraint without clipping or a gradient penalty — [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Reusing_the_Task-Specific_Classifier_as_a_Discriminator_Discriminator-Free_Adversarial_Domain_CVPR_2022_paper.html)
- ToAlign: Wei, Lan, Zeng, Zhang, Chen, NeurIPS 2021 (the search summary listed four authors; the full list is [UNVERIFIED]). Decomposes "a feature in the source domain into a task-related/discriminative feature that should be aligned" — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/731c83db8d2ff01bdc000083fd3c3740-Paper.pdf); code [microsoft/UDA](https://github.com/microsoft/UDA)
- CGDM: Du, Li, Su, Zhu, Lu, CVPR 2021. Pseudo-labels come "through clustering-based self-supervised learning" — [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Du_Cross-Domain_Gradient_Discrepancy_Minimization_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.html); code [lijin118/CGDM](https://github.com/lijin118/CGDM)
- InAIt: Liu, Guo, Li, Xing, You, Kuo, El Fakhri, Woo, ICCV 2021. Infers p(y) and aligns p(x|y) iteratively — [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Adversarial_Unsupervised_Domain_Adaptation_With_Conditional_and_Label_Shift_Infer_ICCV_2021_paper.html)
- NOUN/PRONOUN: Hu, Liang, Hou, Yan, Chen, IEEE TIP 2021, vol. 30, pp. 9359–9371 — [arXiv 2003.13274](https://arxiv.org/abs/2003.13274v4); code [tim-learn/NOUN](https://github.com/tim-learn/NOUN)
- ELS: Zhang, Wang, Liang, Zhang, Wang, Jin, Tan, ICLR 2023 — [ICLR](https://iclr.cc/virtual/2023/poster/11317)
- GH/GH++: Huang, Song, Zhang, arXiv 2408.00288 (2024) — [arXiv](https://arxiv.org/abs/2408.00288v1)

### Per-method records

All equations below are **[math from memory]** unless they are marked as code-verified.

**1. DANN / RevGrad (ICML 2015). Discarded at F1.**
min_{G,C} max_D  L_cls(C∘G; S) − λ [ E_s log D(G(x)) + E_t log(1−D(G(x))) ], implemented with a GRL.
*In plain words:* one network learns features that let the classifier work on the source while a discriminator fails to tell source from target.
Conditioning: none. Weighting: none. MIL: easy to apply to bag embeddings Ψ(B), but it would align only the marginal of the bag embeddings. Class-mixed bags make that worse. Compared with MIL-CREDA: it needs a discriminator, a GRL and a schedule for λ, and it is unstable. MIL-CREDA has no game. In return, DANN needs no pseudo-labels and so cannot suffer from confirmation bias.

**2. ADDA (CVPR 2017). Discarded at F1. Detailed comparison, because it is MIL-CREDA's architectural sibling.**
Phase 1: min_{M_s, C} L_cls(C(M_s(x_s)), y_s).
Phase 2: initialize M_t ← M_s, and **freeze M_s and C**. Then
- min_D −E_s log D(M_s(x_s)) − E_t log(1 − D(M_t(x_t)))
- min_{M_t} −E_t log D(M_t(x_t)) (inverted-label GAN loss, no GRL)

Test time: C(M_t(x_t)).
*In plain words:* train a source encoder and classifier, copy the encoder for the target, freeze everything on the source side, and push only the target encoder until a discriminator cannot tell its features from the frozen source features.
Key properties: **untied weights**, confirmed — [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.html). The source feature distribution is a **fixed target** for alignment, which is exactly the role of MIL-CREDA's frozen phase-1 encoder. Conditioning: none, because D sees only features. Weighting: none.
Similarities with MIL-CREDA: two encoders, source frozen after phase 1, a target encoder initialized from the source, and alignment that moves only the target side.
Differences:
- ADDA aligns the **marginal** with a learned discriminator. MIL-CREDA aligns per class, with routing by hard pseudo-labels, confidence weights and a closed-form matrix-Rényi-2 / Cauchy–Schwarz score on Gram matrices, with no discriminator.
- ADDA's GAN loss can collapse: M_t can map everything into one region of high source density, and nothing stops target classes from landing on the wrong source classes. That is the class-mismatch failure that conditioning fixes.
- ADDA has no local, per-sample anchor. MIL-CREDA's RKHS distance to a personalized source reference plays that role.

What would break in a MIL version: the discriminator would see Ψ(B) from two attention modules. If the target attention collapses onto a few instances, D can still be fooled while class information is lost. ADDA has no class-aware guard against this.
The reverse view: ADDA needs no confidence estimate. MIL-CREDA's routing depends on the frozen source classifier being calibrated enough on target bags at the start of phase 2.

**3. CoGAN (NIPS 2016). Discarded at F1.**
Two GANs (G_1, D_1), (G_2, D_2) share the weights of the first generator layers and the last discriminator layers. Each plays the standard min–max on its own domain.
*In plain words:* two image generators share their "high-level" layers, so the same noise produces corresponding images in both domains.
Conditioning: none on Y. The "joint" distribution it learns is over pairs of images from the two domains, not over (Z,Y). MIL: not applicable (it generates pixels).

**4. PixelDA (CVPR 2017). Discarded at F1.**
min_{G,T} max_D  α L_d(D,G) + β L_t(T,G), where x_f = G(x_s, z) and T is trained on (x_f, y_s).
*In plain words:* translate source images to look like the target, then train the classifier on the translated, labelled images.
Conditioning: none in D. MIL: would need a generator per instance. It is impractical for bags of non-image instances.

**5. CyCADA (ICML 2018). Discarded at F1 (borderline).**
L = L_GAN^pix + L_GAN^feat + L_cyc + L_sem, where L_sem = CE(f_s(G_{S→T}(x_s)), argmax f_s(x_s)).
*In plain words:* translate images with a CycleGAN, require the translation to keep the semantic label the frozen source classifier assigns, and also align features adversarially.
L_sem is a label-preservation constraint, not class-conditional alignment. MIL: pixel translation does not transfer to embedded instances.

**6. MADA (AAAI 2018). Discarded at F3.**
L_adv = Σ_{k=1}^K  E_{x∈S∪T}  L_d( D_k( ŷ_k(x) · G(x) ), d_x ).
*In plain words:* there is one discriminator per class, and each sample is fed to discriminator k scaled by how likely it is to belong to class k.
Conditioning: **soft** (ŷ_k). Weighting: ŷ_k is a per-class *membership*, not a reliability score: a uniform ŷ still spreads mass over all K discriminators. In the usual implementations ŷ is passed through the GRL path [UNVERIFIED for the official code]. MIL: substitute Ψ(B) and bag-level ŷ. This is closest to MIL-CREDA's per-class routing, but with K discriminators instead of one closed-form score. What breaks: K discriminators trained on few bags per class are unstable when bags are scarce.
Versus MIL-CREDA: MADA needs K critics and K min–max games. MIL-CREDA routes hard (argmax) and gates by w. MADA routes soft and has no gate.

**7. CDAN (NeurIPS 2018). Discarded at F3. 7b. CDAN+E: SURVIVOR.**
min_G max_D  −E_s log D(T(f,g)) − E_t log(1 − D(T(f,g))), with T(f,g) = f ⊗ g (multilinear map, or a randomized version R_f f ⊙ R_g g when the dimension is large) and g = ŷ.
CDAN+E: each sample's term is weighted by w(x) = 1 + e^{−H(g)}, normalized within each domain. **Code-verified.**
*In plain words:* the discriminator sees the outer product of features and predictions, so it must be fooled class by class. +E makes confident predictions count more.
Conditioning: **soft** (the full ŷ vector). Weighting formula: w = 1+e^{−H(ŷ)} ∈ (1, 2]. Note that a fully uncertain sample still gets weight 1 + 1/C, not 0. It is a soft emphasis, **not a gate**.
Detach vs GRL:
- In the official thuml/CDAN code, ŷ is detached as the conditioning input, but `entropy.register_hook(grl_hook(coeff))` sends a **reversed gradient through the entropy**. The weight is therefore trained adversarially. — [loss.py](https://raw.githubusercontent.com/thuml/CDAN/master/pytorch/loss.py)
- In TLlib, ŷ and w are both **detached**. — [cdan.py](https://raw.githubusercontent.com/thuml/Transfer-Learning-Library/master/tllib/alignment/cdan.py)
- This discrepancy between the two implementations is worth stating when comparing with MIL-CREDA, whose w is (presumably) detached.

MIL: T(Ψ(B), ŷ_B) is direct. What breaks: the multilinear dimension d·C grows with C, and the randomized map adds variance on small batches of bags.
Versus MIL-CREDA: CDAN+E is the closest *functional* precedent: class-conditional, works without target labels, confidence-weighted. The differences are soft conditioning against MIL-CREDA's hard routing, and weight floor 1 against MIL-CREDA's w ∈ [0,1], which can actually zero out a sample. CDAN+E also needs a discriminator and a GRL.

**8. MCD (CVPR 2018). Discarded at F3.**
Step A: min_{G,C1,C2} L_s. Step B: min_{C1,C2} L_s − E_t |p_1 − p_2|_1. Step C: min_G E_t |p_1 − p_2|_1.
*In plain words:* two classifiers are pushed to disagree on target samples, and the feature extractor is pushed to make them agree, which moves target features away from decision boundaries.
Conditioning: implicit (the critic acts in class-probability space, with no P(Z|Y=c) term). Weighting: none. MIL: apply to bag-level classifiers; this is direct. Versus MIL-CREDA: three-step alternating optimization against a closed-form score.

**9. SWD (CVPR 2019). Discarded at F3.**
MCD's L1 term is replaced by the sliced Wasserstein distance between p_1 and p_2 over M random projections: (1/M) Σ_m W_1(⟨θ_m, p_1⟩, ⟨θ_m, p_2⟩).
*In plain words:* MCD with a geometry-aware disagreement measure.
Same funnel outcome as MCD.

**10. BSP (ICML 2019). Discarded at F1 (a wrapper).**
L_bsp = Σ_{i=1}^k ( σ_{s,i}² + σ_{t,i}² ), the top-k singular values of the feature batch matrices, added to DANN or CDAN.
*In plain words:* penalize the dominant feature directions so adversarial alignment does not erase discriminability.
Not conditional on its own. Interesting for MIL-CREDA because it also works on spectral and covariance structure. Matrix-Rényi-2 is a function of the Gram matrix's eigenvalues, so the spectral-penalty idea is related.

**11. DADA (AAAI 2020). Discarded at F3.**
A single (K+1)-way classifier, whose last output is the domain. On source, the domain logit is pushed against the true class; on target, against the predicted class. This creates a "mutually inhibitory relation" — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/6054). Exact loss: [math not reproduced; UNVERIFIED].
*In plain words:* merge the classifier and the discriminator so that class and domain compete in the same softmax.
Conditioning: soft, through the shared head. Weighting: not confirmed.

**12. CAT (ICCV 2019). Discarded at F3 (tentative).**
L = L_s + L_RevGrad (marginal) + L_cluster (discriminative clustering on both domains) + L_align = Σ_k ‖μ_k^s − μ_k^t‖², with target class means computed from **teacher (mean-teacher) pseudo-labels**.
*In plain words:* an averaged "teacher" model labels target samples, then class means are matched across domains.
Conditioning: **hard** (pseudo-labels), and moment-based rather than adversarial in its conditional part. Weighting: not confirmed. MIL: class means of Ψ(B) are natural. This is a *first-moment* analogue of MIL-CREDA's second-order (covariance-operator) per-class score, so it is a strong comparison point.

**13. GVB (CVPR 2020). Discarded at F1.**
G(x) = f(x) − r(x), with a bridge r whose magnitude is penalized toward 0 over time. The discriminator gets a bridge too, so that D's output is reduced by a decaying residual.
*In plain words:* let a learned residual absorb what is domain-specific, then shrink it so the alignment becomes gradual.
Not conditional on its own.

**14. DAAN (ICDM 2019). Discarded at F3.**
L = L_cls − λ[ (1−ω) L_g + ω Σ_c L_l^c ]. Here L_l^c feeds ŷ_c · G(x) to local discriminator c, and ω is estimated dynamically from the A-distances of the global and local discriminators.
*In plain words:* MADA plus a global discriminator, mixed by a factor that measures whether the marginal or the conditional shift matters more.
Conditioning: soft. Weighting: none per sample. ω is a *global* class-versus-marginal balance, which is analogous to MIL-CREDA's balance between its local and per-class terms, but it is not a confidence gate.

**15. RADA (ICCV 2021). Discarded at F1.**
Target samples whose domain-discriminator entropy H(D(G(x))) is high are relabelled as "source" for D. The paper uses entropy of the *domain classifier's* predictions — [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Jin_Re-Energizing_Domain_Discriminator_With_Sample_Relabeling_for_Adversarial_Domain_Adaptation_ICCV_2021_paper.html).
*In plain words:* once a target sample already looks like source, stop asking the discriminator to separate it.
It is confidence filtering on the domain variable, not on class.

**16. SymNets (CVPR 2019). Discarded at F3.**
Source classifier C^s, target classifier C^t, and a joint 2K-way classifier C^{st}. The category-level confusion loss for a target sample is −½ log Σ_k p^{st}_k(x) − ½ log Σ_k p^{st}_{k+K}(x), with the source-side analogue using the true label.
*In plain words:* force each sample to be equally likely "source class k" and "target class k".
Conditioning: soft on target, hard (true label) on source. Weighting: none [UNVERIFIED].

**17. TADA (AAAI 2019). Discarded at F1.**
Local attention per region r is 1 + (1 − H(D_r(f_r))). Global attention on the image is 1 + H(D(G(x))) and multiplies a target entropy-minimization loss.
*In plain words:* look harder at regions and images the discriminator finds ambiguous.
Its weights come from domain entropy, not class confidence. *MIL relevance:* the closest analogue to attention over instances inside a bag. TADA's per-region discriminators correspond to instance-level critics. MIL-CREDA's attention Ψ(B) is learned for the task, not driven by a discriminator.

**18. AFN (ICCV 2019). Discarded at F1 (not adversarial).**
SAFN: L = Σ ( ‖h(x; θ_0)‖ + Δr − ‖h(x; θ)‖ )², which grows the feature norm step by step.
*In plain words:* features with larger norms transfer better, so enlarge them progressively.

**19. SDAT (ICML 2022). Discarded at F1 (a wrapper).**
Apply SAM only to the task loss: min_θ max_{‖ε‖≤ρ} L_cls(θ+ε) + L_adv(θ). The adversarial term stays as it is.
*In plain words:* look for flat minima of the classification loss but not of the adversarial loss.
Not conditional on its own. Lesson for MIL-CREDA: even with no game, sharpness of the task loss could matter in phase 1.

**20. f-DAL (ICML 2021). Discarded at F3.**
D_f(S‖T) ≥ sup_{h′} E_s[ ℓ(h′,h) ] − E_t[ φ*( ℓ(h′,h) ) ], where φ* is the Fenchel conjugate of the f-divergence. This is optimized as min_{g,h} max_{h′}.
*In plain words:* DANN generalized to any f-divergence, with a critic that compares an auxiliary classifier to the main one.
Conditioning: implicit (in hypothesis space). Notable because the paper argues that many extra regularizers are unnecessary — [NVIDIA](https://research.nvidia.com/labs/toronto-ai/fDAL).

**21. MDD (ICML 2019). Discarded at F3.**
min_{f,ψ} E_s L(f(ψ(x)), y) + η [ γ E_s log σ_{h_f}(f′(ψ(x))) + E_t log(1 − σ_{h_f}(f′(ψ(x)))) ], with max over f′. Here h_f = argmax f is a *hard* pseudo-label and γ is the margin.
*In plain words:* an adversarial copy of the classifier tries to disagree with the main classifier's predicted labels more on target than on source.
Conditioning: implicit, **hard** (argmax). Weighting: none. It is the strongest classic adversarial baseline in TLlib (89.6 / 69.7 / 79.8).

**22. DALN (CVPR 2022). Discarded at F3.**
NWD = (1/N_s) ‖C(G(X_s))‖_* − (1/N_t) ‖C(G(X_t))‖_*, maximized by C and minimized by G, where ‖·‖_* is the nuclear norm of the batch prediction matrix.
*In plain words:* reuse the classifier as the critic, and measure domain difference by how "diverse and confident" each domain's batch of predictions is.
Conditioning: implicit, soft (through the prediction matrix). Weighting: none explicit, though the nuclear norm implicitly rewards confident and diverse batches. *Positioning:* "discriminator-free" yet still min–max. MIL-CREDA is discriminator-free *and* game-free.

**23. ToAlign (NeurIPS 2021). Discarded at F3.**
Source feature split: f^p = w_{y} ⊙ f (task-related, weighted by gradients of the ground-truth class logit, Grad-CAM-like). The adversarial term then aligns f_s^p with f_t through a DANN- or HDA-style discriminator.
*In plain words:* align only the part of each source feature that matters for its true class.
Conditioning: source side only. Weighting: none.

**24. CGDM (CVPR 2021). Discarded at F3 (tentative).**
MCD-style bi-classifier adversarial training plus L_gd = 1 − cos( ∇_C L_s, ∇_C L_t(ŷ^{pl}) ), with target pseudo-labels from clustering — [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Du_Cross-Domain_Gradient_Discrepancy_Minimization_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.html).
*In plain words:* target pseudo-labels should produce classifier gradients pointing the same way as source gradients.
Conditioning: hard pseudo-labels. Weighting: [UNVERIFIED].

**25. SRADA (IJCAI 2020). SURVIVOR, with inverted weighting.**
L_adv = Σ_x w(x) ℓ_d(D(G(x)), d_x), with w increasing in the conditional entropy H(ŷ). Exact form: [UNVERIFIED]. A triplet loss over source and pseudo-labelled target samples is added — [arXiv 2006.00223](https://arxiv.org/abs/2006.00223).
*In plain words:* give badly aligned (uncertain) samples more adversarial pressure, and give pseudo-labelled samples a metric-learning push.
This is directly **opposite** to MIL-CREDA's w = 1 − H₂/ln C on the alignment term. Both choices are defensible, and the contrast is a useful ablation axis.

**26. NOUN / PRONOUN (TIP 2021). Discarded at F3.**
D sees G(x) ⊗ n(x), with a normalized output conditioner n. PRONOUN replaces ŷ with similarities to class prototypes [math UNVERIFIED].
*In plain words:* a CDAN-like conditioner that is less sensitive to overconfident softmax outputs.

**27. InAIt (ICCV 2021). Discarded at F3 (tentative).**
Alternates between estimating p_t(y) and adversarially aligning p(x|y), with a posterior correction at test time — [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Adversarial_Unsupervised_Domain_Adaptation_With_Conditional_and_Label_Shift_Infer_ICCV_2021_paper.html).
*In plain words:* correct for different class frequencies in the target before aligning class by class.
Relevance: MIL-CREDA's per-class aggregation assumes some prior over target classes, and label shift is a known failure mode of per-class gating.

**28. BACG (arXiv 2023, withdrawn). Passes the funnel on paper.**
Two discriminators, class-level gradient alignment, kNN pseudo-labels and evidential (Dirichlet) uncertainty — [arXiv](https://arxiv.org/abs/2310.13959).
Do not cite it as a peer-reviewed precedent.

**29–30. ELS (ICLR 2023) and GH / GH++ (arXiv 2024). Discarded at F1 (wrappers).**
ELS smooths the domain labels d ∈ {γ, 1−γ}. GH projects the alignment and task gradients so the angle between them is acute (GH) or orthogonal (GH++).
Both target the *instability of the game* or *conflicts between task and alignment gradients*, which is evidence that these are recognized pathologies of adversarial DA.

### Inferences
- In the standard implementations, the confidence weighting in CDAN+E never removes samples (w ≥ 1). Of the reviewed methods, MIL-CREDA's w ∈ [0,1] together with a separate confidence gate on class aggregation is the only design that can fully suppress unreliable classes. That claim rests on the verified CDAN code and the abstracts of the other methods, not on full-text checks for all of them.
- Every surviving or near-surviving method uses either a learned critic or a second classifier. None of the reviewed adversarial methods uses a closed-form, second-order (covariance-operator) per-class divergence.

### Gaps
- Full-text equations were not re-read in this run for any method except the CDAN code. Everything marked [math from memory] must be checked against the PDFs.
- Whether MADA's official code detaches ŷ_k, and whether DAAN or SymNets threshold by confidence, was not verified.

---

## Key Question 3: What does adversarial DA need that MIL-CREDA avoids, and the reverse?

### Takeaway
The adversarial family pays for its flexibility with a min–max game: a discriminator, a GRL or alternating steps, and schedules for λ. It also has documented instability, enough that ICML 2022 (SDAT), ICLR 2023 (ELS) and GH 2024 were written to address it. MIL-CREDA removes the game. It replaces the critic with a closed-form per-class matrix-Rényi-2 / Cauchy–Schwarz score on Gram matrices, and it borrows ADDA's frozen-source asymmetry. The price is complete dependence on the frozen source classifier's pseudo-labels and confidence. Adversarial marginal methods (DANN, ADDA) do not carry that risk.

### Cited Findings
- Domain adversarial training "suffers from training instability mostly due to over-confident domain discriminators and environment label noise" — [ELS, ICLR 2023](https://iclr.cc/virtual/2023/poster/11317)
- Converging to smooth minima of the adversarial loss "leads to sub-optimal generalization on the target domain" — [SDAT, ICML 2022](https://icml.cc/virtual/2022/poster/17769)
- The domain classifier's "discrimination capability … deteriorates as training progresses" because the features become aligned — [RADA, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Jin_Re-Energizing_Domain_Discriminator_With_Sample_Relabeling_for_Adversarial_Domain_Adaptation_ICCV_2021_paper.html)
- There is an inherent conflict between the gradients of domain alignment and classification — [GH/GH++, arXiv 2024](https://arxiv.org/abs/2408.00288v1)
- Existing domain-adversarial methods with separate task and domain classifiers have a mode-collapse issue that limits joint alignment — [DADA, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6054)
- DALN removes the separate discriminator but keeps the adversarial objective — [DALN, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Reusing_the_Task-Specific_Classifier_as_a_Discriminator_Discriminator-Free_Adversarial_Domain_CVPR_2022_paper.html)
- ADDA's untied weights with a GAN loss are presented as "simpler" than earlier tied-weight adversarial methods — [ADDA, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.html)

### Inferences (positioning matrix)

| Axis | Adversarial survivor (CDAN+E) | Closest architecture (ADDA) | MIL-CREDA |
|---|---|---|---|
| Critic | Learned D on f⊗ŷ | Learned D on features | **None**: closed-form per-class matrix-Rényi-2 (CS divergence between covariance operators) |
| Game | min–max with GRL | Alternating GAN | **Pure minimization** |
| Encoders | Shared | **Untied, source frozen** | **Untied, source frozen after phase 1** |
| Conditioning | Soft (ŷ vector) | None | **Hard** routing by argmax pseudo-label |
| Confidence weighting | w = 1+e^{−H}, floor 1, GRL through H in the official code / detached in TLlib | None | w = 1 − H₂(g)/ln C ∈ [0,1] + a separate gate on class aggregation |
| Statistic aligned | Implicit (whatever D can detect) | Implicit | Explicit second-order (Gram / covariance operator) |
| Local / per-sample anchor | None | None | RKHS distance to a personalized source reference |
| Bag / MIL ready | Yes, applied to Ψ(B); multilinear size d·C | Yes, applied to Ψ(B); attention collapse can fool D | Native (Ψ(B) as an RKHS mean embedding) |
| Main failure mode | D overconfidence, game instability, mode collapse | Mode collapse, class mismatch | Wrong confident pseudo-labels (confirmation bias); poor coverage of a class when the gate shuts it out |

- What adversarial methods need that MIL-CREDA avoids: a discriminator (or K of them for MADA/DAAN), GRL coefficient schedules, alternating optimization (MCD/SWD/CGDM), Lipschitz control (DALN's NWD is designed to avoid clipping), and stabilizers (SDAT, ELS, GH, RADA). Small-batch noise is also a factor, because discriminators on few bags per class (MADA, DAAN) are data-hungry. That last point is an inference, not measured in a source.
- What MIL-CREDA needs that adversarial methods avoid:
  - a frozen source classifier calibrated enough for hard routing at the start of phase 2;
  - enough confident target bags per class to estimate Gram matrices, since the matrix-Rényi-2 estimate is unstable for small per-class n;
  - a kernel bandwidth choice;
  - O(n²) Gram matrices per class;
  - robustness to label shift, which InAIt shows matters for conditional alignment.

  Marginal adversarial methods (DANN, ADDA) need none of these.
- Ablations suggested by the funnel:
  - CDAN+E-style soft conditioning against MIL-CREDA's hard routing;
  - SRADA-style inverse weighting against MIL-CREDA's direct weighting;
  - an ADDA-on-Ψ(B) baseline to isolate the effect of conditioning, holding the frozen-source architecture fixed;
  - CAT-style first-moment per-class alignment against the second-order Rényi-2 score.

### Gaps
- No published adversarial method designed natively for MIL bags was found in this run. The MIL-specific claims above are inferences about what would break, not reported results.
- No source in this run benchmarks a closed-form, discriminator-free per-class divergence directly against CDAN+E. The novelty claim is consistent with the methods covered here, but it was not checked against the kernel/MMD family (CAN, JAN, DSAN), which is outside this scope and presumably covered by other researchers.
