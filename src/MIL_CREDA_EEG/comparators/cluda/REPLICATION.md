# CLUDA: official-code loss under the common bag-level protocol

## Code search, pin and route

Historical code search (2026-11-05 UTC): exact paper title + github/code, Xu CLUDA
125452 research CUHK/code, and exact title + code authors resolved
https://github.com/xuchengjian632/CLUDA, not Ozyurt et al. ICLR 2023 CLUDA.
The CUHK publication record was fetched at
https://research.cuhk.edu.hk/en/publications/unsupervised-multi-source-domain-adaptation-via-contrastive-learn/
and confirmed the abstract, DOI 10.1016/j.eswa.2024.125452 and ESWA 261 (2025).
The venue page https://www.sciencedirect.com/science/article/pii/S0957417424023194
returned raw ScienceDirect HTML, not readable full text. Nevertheless the local
ingested `guidance/related-work-evidence/xu2024cluda/xu2024cluda.md` contains method
sections, Eqs. (1)–(8), Algorithm 1 and Table 1; the earlier claim that the method
text could not be read was false. Its data statement says the authors cannot
share data; no code link was found in that extraction.

The earlier scratch clone was removed outside this stretch. This rewrite freshly
cloned the official repository to
`/var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.BMfuyCBETP/CLUDA`, checked out
**9c655644e9f15eadddb51de236a1afecd5944468**, and reread its losses and weights.
The pinned tree contains README.md, models.py, main.py, utils.py and a figure,
without a license or tests/config files. Experiment/architecture defaults are in
main.py/models.py. `third_party/cluda/PIN.json` remains the existing pin, unchanged.
**Route: independent controlled-variant implementation; read-only upstream audit,
no upstream source vendored, no upstream patches.** No upstream training was run.

Bound protocol is `experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md`
(rebound from v54 on 2026-11-05, and to v56 on 2026-11-05 with no change of behaviour): v55 published the CLUDA replacements R1-R5, which
state the official-code loss and the main.py:118-119 selection implemented here.
Authority is operator L81, superseding L74–L75's paper-Eq. (3) choice and omission
of source-source loss, plus the parent's subsequent no-positive-anchor ruling.
L77 (separate projector batches), L78 (source queues/singleton skip), L79 (constant
weight), L59 (inductive evaluation), and L70–L71 (single phase-one checkpoint) stand.
Integration e5de342 is preserved; no shared registration files were edited here.

## Exact formula, code correspondence, and shapes

Let a contrastive batch B contain projected vectors e_a and labels y_a. Define
u_a=e_a/||e_a||_2, h_ba=u_b dot u_a / tau,
P(a)={b != a : y_b=y_a}, and V={a : |P(a)|>0}. The implemented loss is

    L(B,tau) = mean over a in V of
               [ log(sum_{b != a} exp(h_ba))
                 - log(max(sum_{b in P(a)} exp(h_ba), 1e-6)) ].

When V is empty the result is a differentiable zero. The exclusion of empty
positive sets is the **declared bag-level remedy**; all other terms follow
`models.py:257–282`: diagonal exclusion (:265), same-class mask (:268), negative
mask (:271), L2 normalization (:273), temperature-scaled dot products (:275),
exponentials/off-diagonal (:277–278), **unnormalized** numerator clamp (:280),
and symmetric mean (:282). Stable logsumexp evaluates the same formula without
large direct exponentials. Zero-norm embeddings fail explicitly rather than
silently substitute another cosine definition.

For sampled source subject-domains S_1,...,S_D and target batch T:

    ST = (1/D) sum_i L(concat(S_i,T), searched_temperature)
    SS = (1/choose(D,2)) sum_{i<j} L(concat(S_i,S_j), 0.03)
    total = frozen_source_CE + contrastiveWeight * (ST + 0.5 * SS).

`models.py:225–229` generates target pseudo-labels by argmax, no threshold;
:232–235 concatenates each projected source domain with the projected target
batch; :241 averages ST across source domains; :243–249 averages SS across
unordered source-domain pairs. Thus **target-target pairs participate** in every
ST domain's symmetric loss. Source source pairs are retained and computed, not
replaced by an unrelated global source SupCon loss. Queue repeats of one subject
are grouped back into its same source domain before either loss is evaluated.

The exact official coefficient formula at `main.py:88–91` is

    gamma = 2/(1 + exp(-10*iteration/total_iterations)) - 1
    beta1 = gamma
    beta2 = gamma/2
    official_total = cls_loss + beta1*ST + beta2*SS.

We retain **SS/ST weight ratio 1/2** but replace gamma with the single constant
searched `contrastiveWeight`, no ramp (L79). `models.py:208` and `main.py:193` fix
T=.07 and T2=.03; this variant searches only ST temperature using the existing
`search.CONTRASTIVE_TEMPERATURE` (.01–1 log). SS temperature stays .03.

- Common model input: list of `(windows, source/target channels, 512)` tensors.
  Existing EEGNet and common attention pool to `(S,256)` / `(T,256)`.
- `model.CLUDAModel`: same inherited predictor plus projector 256->1024->256;
  source and target pass through that projector separately, target first.
- Source representations are detached from the frozen source encoder. They are
  **not** detached from the projector: ST and SS both have projector gradients.
  SS has no route to the target encoder; ST does. Attention and head stay frozen.
- `losses.supervised_contrastive`: `(N,256),(N,) -> scalar, excluded_anchor_count`.
  `cross_domain_contrastive` performs the ST domain loop;
  `source_source_contrastive` performs the SS pair loop;
  `contrastive_objective` fixes the shared scale/ratio above.
- `training.adaptation_step` calculates the diagnostic frozen source CE and both
  contrastive losses. `train_phase_two` reuses existing bridge, schedule guards,
  ClassQueues, shuffled target batches, validation records with representations,
  max-norm, search records and model_store. It never reimplements bag preparation
  or phase-one training.

## No-positive-anchor remedy and the remaining singleton-domain limitation

The verbatim code clamp on an anchor without positives gives
`log(denominator)-log(1e-6)`, **not a constant**; it still has denominator gradients.
In this bag adaptation, many two-singleton SS pairs have different labels and no
positives. A preliminary smoke measured SS
`[3.701432,-4.371322,-8.335127,-9.730032,-10.253315]` and total
`[3.305864,-1.093510,-2.712546,-3.345008,-2.837553]`. This finite trajectory is not
proof of mathematical unboundedness, but exposed a source-only projector descent
caused by singleton-domain adaptation. The parent required the minimal remedy:
**exclude anchors with no positive from the anchor mean in BOTH ST and SS**.
Reference supplied by the parent: Khosla et al., Supervised Contrastive Learning,
NeurIPS 2020, https://arxiv.org/abs/2004.11362 (abstract fetched 2026-11-05).
This is an explicit implementation decision, not a claim that CLUDA's code already
handles missing positives this way. It does not replace CLUDA's log-of-positive-
sum with Khosla's mean-of-positive-log-probabilities objective.

An excluded anchor contributes no anchor loss or anchor gradient. Its embedding
may still be a negative candidate in OTHER valid anchors' denominators; excluding
that candidate would introduce another, unauthorized objective change. All-invalid
batches return a graph-connected zero. The official numerator clamp remains for
valid anchors. Consequently there is **no global nonnegativity claim**: even a
same-class pair of opposite unit vectors at .03 can have loss
`-1/.03-log(1e-6) = -19.5178...` when the clamp is active. Tests establish >=0 on
an explicitly non-saturated batch and separately pin this negative saturated case.

**SS is effectively inert for distinct singleton source domains**, for a different
reason from old L62's frozen-encoder argument:
- Different-class 2-bag pair: neither anchor has positives; both excluded, loss0.
- Same-class 2-bag pair: numerator equals denominator, loss0 and zero gradient
  unless the official numerator floor saturates.
- Repeated source draws can create multi-bag domains; SS then can train the
  projector through valid positives with negative candidates. A dedicated test
  proves this gradient route without inventing extra source views/bags.

Final synthetic smoke: 28/28 SS pairs have zero loss in every update (**100%**),
32/56 SS anchor evaluations are excluded per update (**57.1429%**), SS=0 throughout.
No new mechanism is added to make the SS term active. Neither this fraction nor
its frequency on real folds is inferred from the toy run.

Both losses expose per-step excluded counts, separately:
`step_anchors_without_positive` (ST), `step_source_anchors_without_positive` (SS),
with `step_anchors` / `step_source_anchors` as denominators. ST anchor counts sum
over domains; SS counts sum over pairs. `step_source_pairs`,
`step_source_zero_loss_pairs`, `source_zero_loss_pair_fraction` record measured
zero-loss pair counts (abs(value)<1e-12); they are not an unmeasured gradient norm.
History averages losses over executed updates; anchor counts sum over updates.

## Correspondence and chosen hyperparameters / architecture

`unspecified` means the paper does not fix the choice even where code or the
common protocol supplied a default. There are **6 unspecified rows**.
Paper/code loss differences are resolved **in favour of code** by L81, except
for the explicit no-positive-anchor remedy. Common-protocol departures remain
identified; no claim to reproduce the paper's numerical results is made.

| Element | Paper / pinned code | This variant / provenance | Verdict |
| --- | --- | --- | --- |
| Feature extractor | Paper Eq. (1)/Table 1 CNN+Transformer; models.py:38–162: 40-wide tokens, six layers, ten heads, expansion4, dropout .5, flatten1480->256 | Existing EEGNet-8,2: 8 temporal filters, multiplier2, 16 separable filters, kernels64/16, pooling4/8, BN/ELU, dropout .25, spatial maxnorm1; 256-wide bag representation, v56 §7.3 | deviation: common capacity |
| Classifier | Paper §3.2.3 two FC; code 256->32->4 with ELU/dropout .3 | Frozen common 256->2 linear bag head, maxnorm .25, v56 §7.3 | deviation: common capacity |
| MIL aggregation and domain unit | Trial classification; each source subject contains multiple trials | Common tanh/softmax attention, gamma0/no consensus, bag-level labels; subject-domain bag(s), v56 §§7.2–7.4 | deviation: common bag protocol |
| Projector | Paper §3.2.4/Table1; models.py:164–178, bias-free linear256->1024, affine BN, ReLU, nonaffine BN, bias-free linear1024->256, nonaffine BN | Same audited architecture; input width from existing model; no inference override | match to code; extra BN layers absent from simplified paper table |
| Projector initialization | Paper does not state initialization | Fresh PyTorch defaults seeded for each phase-2 run, no inherited phase-one projector | unspecified |
| BN defaults / precision | Paper silent, code framework defaults | eps1e-5, momentum.1, running stats, common model double precision | unspecified |
| Projector batching | Target forward models.py:225, each source separately :232 | Separate target forward then one pooled source forward; minimum2 per side; L77 | deviation: singleton-source BN pooling |
| Inference | Fig.2 / README:18 only extractor+classifier | Inherited target EEGNet+common attention+head, projector never called; state saved for audit | match mechanism under common MIL |
| Pseudo-labels | models.py:225–229 softmax argmax, no threshold | Detached frozen-head argmax on target bag logits; no target-training label access | match |
| ST positives, anchors and reduction | Paper Eq.(3) cross-only source-anchored -1/2 sum; code :233–241 symmetric mean on concat(source_i,target), then mean domains | `cross_domain_contrastive` follows code, includes target-target pairs, no paper factor1/2 | match code; paper/code disagreement resolved L81 |
| SS positives, anchors and reduction | Paper Eq.(4) cross-source formula; code :243–249 symmetric SupCon on each unordered domain pair, then pair mean | `source_source_contrastive`, trainable projector; computed even if singleton pairs yield zero | match code; paper/code disagreement resolved L81 |
| Cosine / self exclusion | Paper normalization implicit; models.py:265–278 excludes diagonal, normalizes L2, dot/tau | Same, logsumexp evaluation, explicit failure on undefined zero norm | match on defined inputs |
| Numerator floor | models.py:280 max(positive unnormalized mass,1e-6) | Same floor for valid anchors; possible negative saturated loss retained | match |
| Missing positives | Verbatim code retains clamped anchor; paper undefined at zero positive mass | Exclude from anchor mean; differentiable zero if none valid; per-step excluded counts, parent follow-up | deviation: minimal bag-level remedy |
| Coefficient ratio | Paper Eqs.(5)–(7) SS=alpha/10; main.py:88–91 SS=gamma/2, ST=gamma | total=source_ce+w*(ST+.5*SS), `contrastive_objective` | match code ratio; paper/code disagreement resolved L81 |
| Ramp | Native gamma=2/(1+exp(-10*t/D))-1 | Constant searched w, L79 common schedule | deviation: common fixed phase-2 schedule |
| Weight range | No published controlled-variant search | w=search.LOSS_WEIGHT [1e-3,10] log, v56 §7.7 | deviation: common search |
| Standalone weight default | No fixed paper value for this variant | 1 for smoke; actual search supplies value | unspecified |
| Temperatures | models.py:208 / main.py:193 T=.07, T2=.03; paper uses symbol tau | ST default.07, searched search.CONTRASTIVE_TEMPERATURE [.01,1] log; SS fixed.03, no second temperature axis | match code SS/default ST; deviation: common ST search |
| Optimizer | Paper §4.3 / main.py:62 Adam betas(.5,.999) lr.0002 | Adam betas(.5,.999), target trunk+projector; shared lr2[1e-5,1e-2] log, wd2[1e-7,1e-2] log | match betas; deviation common trainable set/search |
| Standalone optimizer defaults | No controlled-variant values | lr2=.001, wd2=0 for smoke, same repository convention | unspecified |
| Source sampling | Official separate shuffled trial batches per source | Shared ClassQueues.draw(bagsPerStep//classes); both source classes required before model construction; repeated subjects grouped, L78 | unspecified |
| Batch size / epochs | Paper batch36, main.py defaults batch20/epoch200 | Shared checked schedule, >=2 bags/side, exact epochs2, no early stopping, v56 §§4.5,7.7 | deviation: common budget |
| Singleton target remainder | Official projector BN cannot train one example | Skip entire update, no projector/optimizer/source draw, count in history; target set<2 refuses, L78 | unspecified |
| Source CE | Official mean source-domain CE, trainable source | Diagnostic sampled-source bag CE only, no gradient; source frozen v56 §7.3 | deviation: common frozen source |
| Phase-two bridge/freeze | Native shared trainable Conformer | Existing shared bridge copies compatible weights, keeps spatial initialization for29->14; source BN/attention/head frozen; target trunk trains | deviation: common two-phase protocol |
| Phase-one checkpoint seed | main.py setup_seed20 | Verify search.PHASE_ONE_SEED=1 despite trial seed0; transfer/rotation/partition/channels checks, L70–L71 | deviation: common checkpoint |
| Adaptation data | main.py:167–168 target_data=test_data | Target TRAIN bags only, disjoint subjects, no test input, L59 | deviation: inductive common protocol |
| Selection | main.py:118–119 retains maximum test accuracy | All inner validation epochs recorded, the search's centroid primary and EnsV secondary selection with no target label, no refit validation, v59 §4.5 | deviation: removes test-label selection |
| Preparation, Eq.(8) | Paper filtering/standardization, code main.py norm_type default none | Existing official prepared bag input only, no comparator preparation fork | deviation: common inputs |
| Persistence | No original controlled-variant checkpoint format | Existing shared model_store plus comparator digest; full projector+BN state, no bridge on reload; integration e5de342 | deviation: repository contract |
| Smoke | Not a paper experiment | Unit-id SHA256 seed639203496, 8 bags/domain, 2windows/bag,29/14channels,512samples;3/5epochs,8bags/step | deviation: synthetic execution check only |

Lineage limits inherited from the first audit: the cited Song2022 EEG Conformer,
Lee2022 and Shen2022 method references were not found by guidance-name search;
`song2024macs` and `lee2025lbm` are different papers. No lineage equation was
invented. This rewrite follows the pinned runnable loss rather than recovering
those references from memory. The abstract fetch for Khosla does not establish a
line-by-line equivalence to their implementation; the precise empty-anchor
extension is the parent's explicit decision above.

## Verification, RED/GREEN, and what was not tested

Scratch and all logs remain at
`/var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.BMfuyCBETP/`, undeleted.
The rewrite first changed tests while the old Eq.(3) implementation still stood:
`.venv/bin/python -m pytest -q tests/test_balladeer_comparator_cluda.py` exited1,
**6 failed,8 passed in5.20s** (target-target loss, missing new loss API, SS
projector gradient/ratio, domain averaging, stale provenance). Official-loss
implementation then passed14 tests. After the no-positive follow-up, new tests
were RED again: **5 failed,10 passed in5.37s**, exit1.

Final commands:
- `.venv/bin/python -m pytest -q tests/test_balladeer_comparator_cluda.py`:
  **15 passed in5.24s**, exit0; final rerun **15 passed in5.12s**, exit0.
- `.venv/bin/python -m pytest -q tests/test_balladeer_campaign.py -k cluda_runs`:
  **1 passed,31 deselected,2 warnings in82.58s**, exit0. Warnings are Optuna's
  experimental GPSampler notice. Campaign integration and full-state reload run.
- `PYTHONPATH=src .venv/bin/python -m MIL_CREDA_EEG.comparators.cluda.smoke`:
  exit0; output below.

Tests pin target-target pairs, symmetric/domain/pair means, exact coefficient
ratio, fixed SS temperature, normalized-vector scale invariance, empty-positive
exclusion and zero gradients, retained active numerator clamp, positive-set
non-saturated nonnegativity without a global claim, SS projector gradient with
multi-bag domains but no source/target encoder gradients, and singleton SS zero
pair fraction. Existing tests retain checkpoint/PHASE_ONE_SEED checks, common
state freezing, target gradients, label blindness, reproducibility, projector
BN minimum/separate forwards, singleton skips, final-epoch/refit semantics,
registration and model-store/digest round trips, provenance/test coverage.

A read-only AST extraction executed the pinned upstream loss alone on CPU in
scratch. Before the exclusion remedy, values AND feature gradients matched at
N2/tau.03, N5/tau.07, N8/tau.01 (float64 tolerance1e-11). After the remedy, exact
upstream equivalence holds only where all anchors have positives; final CPU
audit matched values and gradients at N4/tau.03, N6/tau.07, N8/tau.01 with all
anchors positive. No-positive cases intentionally differ and are tested
separately. No upstream source was copied to the target repository.

Final smoke output:

```text
seeded synthetic EEG CPU only; NOT paper numbers; seed=639203496
source=(8 bags,2 windows,29 channels,512); target=(8 bags,2 windows,14 channels,512)
phase1_loss [0.037831, 0.036197, 0.036509]
phase2_loss [1.455148, 1.282699, 1.5132, 1.142324, 2.480091]
contrastive [0.765644, 0.593196, 0.823696, 0.452821, 1.790588]
source_source [0.0, 0.0, 0.0, 0.0, 0.0]
step_anchors_without_positive [[0], [0], [0], [0], [0]]
step_source_anchors_without_positive [[32], [32], [32], [32], [32]]
source_zero_loss_pair_fraction [1.0, 1.0, 1.0, 1.0, 1.0]
skipped_singleton_target_steps [0, 0, 0, 0, 0]
step_target_pseudo_classes [[2], [2], [2], [2], [2]]
```

The existing lunas/rollo/toroide archives were previously inspected as benchmark
histories/latent arrays, not EEG window bags. This remains explicitly synthetic
EEG-shape execution, not the operator's hypothesis PoC or real-data evidence.
No accuracy or published numerical reproduction is claimed.

Not run: full suite, original data/training, GPU/remote jobs, real-fold SS inert
fractions or published-result comparison. No commits, shared registration edits,
AdaptMIL edits, pin changes, new source views, or GPU launches. The previous
remote-parallelism assessment refused missing measured unit/setup durations and
account limits; that remains a later stretch's prerequisite, not launch authority.
