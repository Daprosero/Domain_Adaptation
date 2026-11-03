# CS+CCS — replication record (controlled BALLADEER variant)

Paper: Yin, Yu, Principe et al., «Domain Adaptation with Cauchy-Schwarz Divergence», UAI 2024 (PMLR v244), arXiv 2405.19978v1. Ingested copy: `guidance/experimental-papers/cs-ccs-2405.19978/cs-ccs-2405.19978.md` (forge workspace). Binding plan: `experiments/experiments-plan-experimental-mil-creda-sobre-balladeer-v42.md`, §§4.5, 7.1-7.4, 7.7 (rebound from v40 on 2026-11-03; v42 §7.1 records the route below, reimplemented from the article, repository read only to audit at commit 79d010b5e125bf4558d9318bdcd633937e0c9274).

This is a **controlled variant** under plan §7.4, not a reproduction of the paper's results and not a claim of fidelity beyond the rows and tests below.

## Code search (2026-11-02)

| Where | What it answered |
|---|---|
| Ingested paper, Appendix C opening line | Points to the demo code of **CS-adv** at `https://anonymous.4open.science/r/CS-adv-58E5`; `curl` returned HTTP 401 `{"error":"not_connected"}`. |
| PMLR page `https://proceedings.mlr.press/v244/yin24a.html` | HTTP 200; links `https://github.com/ywzcode/CS-adv` as the code. |
| arXiv HTML `https://arxiv.org/html/2405.19978v1` | Only code link found: `github.com/ywzcode/CS-adv`. |
| `https://github.com/ywzcode/CS-adv` (GitHub API) | HEAD of `main` is `79d010b5e125bf4558d9318bdcd633937e0c9274` (2024-06-18), the commit the plan pins. `license: null`; `/license` endpoint HTTP 404. Tree: `CSutils.py` (CS and CCS estimators), `train_image_officehome_cs.py` (CS-adv on Office-Home), loaders, models. No CS+CCS (distance-metric) training script. |
| Web search for title + "github" | Not performed: this stretch had no search tool, only `curl`. |

**Route: reimplementation from the paper.** The repository has no license, so it is not vendored and no line is copied (operator's decision, 2026-11-02). It was read at the pinned commit, in a scratch directory outside every repository, only to audit this reading of the paper; every disagreement is a row below.

## Equation cross-checks

Each equation used was read in three places: the ingested Markdown, the arXiv HTML `alttext`, and the filed PDF's text layer (`pymupdf`, papersmith env).

| Equation | Ingested Markdown | arXiv HTML | PDF text layer | Used as |
|---|---|---|---|---|
| Eq. 2 (CS divergence) | correct, wrapped in a needless array with a literal (2) | `-log((∫pq)^2 / (∫p^2 ∫q^2))` | — | definition only |
| Eq. 3 (conditional CS) | correct, starts with prose | same four-term form | — | definition only |
| Eq. 11 (estimator of D_CS, Prop. 5) | correct, spliced with layout fragments ("Rema", "condit") | `log(1/M² ΣK^s) + log(1/N² ΣK^t) − 2 log(1/(MN) ΣK^st)`, Gaussian `exp(−‖z−z′‖²/2σ²)` | same, kernel stated under it | `divergences.cs_divergence` |
| Eq. 13 (estimator of D_CCS, Prop. 6) | **defect not listed in the brief:** the two self-term denominators are printed `Σ_i K_ji` instead of `(Σ_i K_ji)²` | squared: `(Σ_{i=1}^M K^s_ji)²`, `(Σ_{i=1}^N K^t_ji)²` | squared (`(PM i=1 Ks ji)2`) | `divergences.ccs_divergence` with the squares |
| Eqs. 84, 85 (one-sided CCS estimators) | `.json` equations 88 and 89; they are **not** the same estimator: 84 uses `−2 log(cross_st)`, 85 uses `−2 log(cross_ts)`; the `.json` 89 copy drops a closing bracket | as stated | — | test reference only: Eq. 86 is their mean |
| Eq. 86 (symmetric CCS, "consistent and symmetric") | `.json` 90, correct | identical to Eq. 13 | — | same function as Eq. 13 |
| Eq. 88 (CCS in the conditional test, App. C.1) | `.json` 92, starts with a stray `#` and prose | identical to Eq. 86 with x for z | — | not used beyond the identity |
| Eq. 90 (CS+CCS objective, App. C.2) | `.json` 94; its (90) label dropped | `L = L_CE + λ D_CS + β D_CCS` | `L = LCE + λDCS + βDCCS, (90)` | `training.cs_ccs_objective` |
| Sec. 4.2 implementation details | "we set λ and β as 1 … we normalize z and ŷ and set kernel size σ = 1" | same | same (lines quoted in the scratch extract) | defaults and normalisation |
| App. C.5 | "To have the same regularization strength with MMD, we keep λ and β as 1" | same | same | default λ = β = 1 |

## Audit of the unlicensed repository against the paper (read only)

| Item | Paper | Upstream code at 79d010b5 | Verdict for this reading |
|---|---|---|---|
| CS estimator | Eq. 11 | `CSutils.CS`: `−2 log(mean K12) + log(mean K1) + log(mean K2)` | match |
| CCS estimator | Eq. 13 / 86 | `CSutils.CondCSD`: `(cs1 + cs2)/2` with `cs1 = −2 log cross1 + log self1 + log self2`, `cs2` with `cross2`; denominators `(K1.sum(-1))**2`, `K1.sum(-1)*K12.sum(-1)`, `K2.sum(-1)*K21.sum(-1)` | match (it is Eq. 86 written as the mean of Eqs. 84 and 85) |
| Normalisation of z | Sec. 4.2: "normalize z" (form unstated) | `F.normalize(Z, p=2, dim=1)` | the L2 form is the code's, adopted; row below marks it |
| Normalisation of ŷ | Sec. 3.3: ŷ are "predicted class probabilities"; Sec. 4.2: "normalize ŷ" | L2-normalised **logits** (`F1` output fed to `nn.CrossEntropyLoss`), not probabilities | **deviation between paper and code**; this replication follows the paper (probabilities of the common head) and the plan ("predicciones de la cabeza común"), then L2-normalises them |
| Source side of CCS | Prop. 6 main text: source ŷ are predictions (appendix proof writes labels "for convenience") | source logits, never one-hot labels | match (predictions on both sides) |
| σ | Sec. 4.2: σ = 1 | `sigma=1` for CS and CCS, both kernels | match in the code; kept here for the prediction kernel, the representation kernel follows the plan's median rule (plan v42 §7.7, rows below) |
| λ, β | Sec. 4.2, C.5: 1 | `all_loss = … + CS_loss + CCS_loss`, weights 1 | match |
| Optimizer | Sec. 4.2: SGD, batch 32 | SGD, momentum 0.9 (`--momentum` default), `--lr 0.004` in `run_office_home.sh` | SGD match; momentum is code-only |
| CS+CCS training | App. C.2, Eq. 90 | no script; only CS-adv | nothing to audit |

## Correspondence table

Every hyperparameter and architecture choice of this module. A row cites what fixes it, or is `unspecified` with the default chosen and why.

| Choice | Value here | Source | Kind |
|---|---|---|---|
| Objective | `L_CE + λ D_CS + β D_CCS` | Eq. 90 | cited |
| Marginal term | Eq. 11 on bag representations z | Eq. 11; plan §7.4 ("la marginal sobre las representaciones de bolsa") | cited |
| Conditional term | Eq. 13 (= 86) on the common head's predictions | Eq. 13; plan §7.4 | cited |
| Kernel | Gaussian `exp(−‖a−b‖²/2σ²)`, unnormalised | Eq. 11 statement; plan §7.4 "núcleos gaussianos"; normalising constants cancel in both estimators | cited |
| ŷ | softmax probabilities of the common head | Sec. 3.3 "predicted class probabilities"; plan §7.4 | cited (deviates from upstream logits, see audit) |
| Normalisation | rows of z and ŷ to unit L2 norm before every kernel | Sec. 4.2 "we normalize z and ŷ" | cited for *that* both are normalised; the L2 form is `unspecified` in the paper — chosen because it is the authors' code |
| λ (`csWeight`) default | 1.0 | Sec. 4.2, App. C.5 | cited |
| β (`ccsWeight`) default | 1.0 | Sec. 4.2, App. C.5 | cited |
| λ, β search | `search.LOSS_WEIGHT`, log [1e-3, 10] | plan §7.7 generic rule | cited (plan) |
| Kernel widths | `σ_z = zKernelMultiplier × median`; `σ_y = 1` fixed (`training.PREDICTION_SIGMA`) on the L2-normalised softmax predictions | plan v42 §7.7 «el núcleo de las predicciones queda fijo en sigma igual a 1 sobre las predicciones softmax normalizadas en L2, como en su artículo … el núcleo de las representaciones sigue la regla del multiplicador de la mediana»; σ_y matches the paper's σ = 1, σ_z deviates from it | cited (plan) |
| Number of kernel widths | two: one for z (Eq. 11 and the K matrices of Eq. 13), one for ŷ (the L matrices); only σ_z is searched | plan v42 §7.7 «queda un solo eje de núcleo buscado, el de las representaciones» | cited (plan) |
| Multiplier search | one axis, `zKernelMultiplier` ∈ `search.KERNEL_MULTIPLIER` {0.5, 0.75, 1, 1.5}, default 1; `yKernelMultiplier` removed | plan v42 §7.7 | cited (plan) |
| Median: which points | L2-normalised z of every source and target **training** bag, each bag passed alone, in eval mode, after the bridge | plan §7.7 states the multiplier, not the scope | `unspecified` — same scope as the committed AdaptMIL comparator; target validation and test bags are never read |
| Median: cadence | once at phase-2 start, fixed for every epoch, stored in the checkpoint's `hyper` | not stated | `unspecified` — a per-batch median would make the kernel move with the trunk it is aligning |
| Median: definition | median of the positive pairwise Euclidean distances (`torch.median`, lower middle for even counts) | not stated | `unspecified` |
| Collapsed median | no positive distance -> 1.0 × multiplier | Sec. 4.2's σ = 1 used as the fallback | `unspecified` (the fallback value is the paper's, the rule is ours) |
| Numerical form | every kernel sum in the log domain (`logsumexp`) | not stated; upstream computes direct sums and its own comments record NaNs | `unspecified` — identical numbers where the direct form is finite (test), finite where it underflows |
| Squared distance | from the difference itself, not `cdist` | not stated | `unspecified` — zero, not NaN, gradient on coincident points |
| Source cross-entropy | `F.cross_entropy` on the frozen head's logits of frozen source bags, no graph | Eq. 90's `L_CE`; plan §7.3 freezes source trunk, attention and head | cited; it carries no gradient in phase 2 and is reported |
| Trained parameters | target EEGNet trunk only | plan §7.3 | cited (plan) |
| Encoder, attention, head | `CommonMILModel` (EEGNet-8,2 trunks, γ = 0 attention, common head) | plan §§7.2, 7.3 | cited (plan) |
| Phase-1 checkpoint | the common group's, same transfer/rotation/seed/partition/channels | plan §7.3 | cited (plan) |
| Optimizer | SGD | Sec. 4.2 | cited |
| Momentum | 0.9 | not stated in the paper | `unspecified` — the authors' script default |
| lr2, wd2 | searched, `search.SHARED_PHASE_TWO` | plan §7.7 | cited (plan) |
| Batch | `bagsPerStep` target bags per step, shuffled; source batch of `bagsPerStep // classes` bags per class from `two_phase.ClassQueues` | paper: batch 32 (Sec. 4.2), class balance unstated; plan leaves `bagsPerStep` to the pilot | `unspecified` — balanced source draws as MIL-CREDA's phase 2, so every class sits in the source kernel set |
| Max-norm after each step | `eegnet.apply_max_norm(target_trunk)` | plan §7.3 (EEGNet-8,2 config) | cited (plan) |
| Epoch selection | target-validation balanced accuracy, strict improvement (ties keep the earlier epoch); label-free score's best epoch kept as `secondary_checkpoint` | plan §4.5 | cited (plan); the tie rule is `unspecified`, same as the other comparators |
| Entropy term γ L_Ent | absent | belongs to CS-adv's L_cls (Eq. 16), not to Eq. 90 | cited |
| Second classifier, adversarial steps | absent | CS-adv only (Eqs. 15-18); plan §7.1 excludes CS-adv | cited |

`unspecified` rows: **10** (normalisation L2 form, median scope, median cadence, median definition, collapsed-median fallback, log-domain form, squared-distance form, momentum, batch composition, epoch tie rule).

## What was tested

Commands, from the worktree root:

1. `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_balladeer_comparator_cs_ccs.py -q` — 22 passed.
2. `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q -k balladeer` — 399 passed, 2 skipped (377 passed, 2 skipped before this unit).
3. (v40, before the v42 kernel change) `PYTHONPATH=$PWD/src .venv/bin/python -m MIL_CREDA_EEG.comparators.cs_ccs.smoke` — CPU, seeded synthetic bags (seed 407092304 from unit id `cs_ccs-synthetic-device-rotation0`), source 4 bags × 2 windows × 29 channels × 512, target the same at 14 channels, 3 phase-1 and 3 phase-2 epochs. Printed: `phase1_loss [0.038411, 0.035476, 0.033337]`, `phase2_loss [4.655045, 231.01322, 1183.992584]`, `phase2_source_ce [0.687582]×3`, `phase2_cs [0.551831, 0.487932, 0.592547]`, `phase2_ccs [3.415632, 229.837706, 1182.712454]`, `zBandwidth 1.3180721409123146`, `yBandwidth 0.0034478120961097675`, best epoch 1. Execution evidence only; it is not a BALLADEER result.

Invariants (each `test_<id>`): closed forms on single points (D_CS = ‖Δz‖²/σ², D_CCS = ‖Δŷ‖²/σ_y²), zero on identical samples, D_CS ≥ 0 on 50 random draws, Eq. 13 equal to a literal transcription and to the mean of Eqs. 84 and 85 and symmetric under swapping domains, finiteness where direct sums underflow, L2 normalisation, the median rule, the prediction kernel fixed at σ = 1 on unit-norm predictions whatever the representation multiplier, frozen common parts with finite non-zero target gradients for 29→14, 14→29 and 29→29, target training labels unread (poisoned and flipped labels give identical histories and weights), bandwidths fixed from training bags only and BatchNorm untouched by their pass, Eq. 90 composition with a gradient-free CE and each weight effective, best-epoch restore and secondary checkpoint, fail-closed inputs, explicit and restorable registration under the generic rule, foreign phase-1 records refused, save/load round trip through `search.save_phase_two`, reproducibility, the smoke.

Mutation check (each mutation applied alone, targeted tests run under a fresh bytecode cache, source restored and diffed after): first pass, 11 mutations, 8 killed; 3 survived — one was mathematically equivalent (swapping the two factors of a product), two exposed weak tests (the bandwidth test compared the function with itself; the selection test had no tie). The bandwidth test gained an independent median, the selection test a tie case; second pass, the two survivors plus a real replacement of the equivalent one: 3 of 3 killed.

## What was not tested, and what it means

- **Under plan v40 the smoke's D_CCS grew from 3.4 to 1183 in three epochs.** The phase-1 head is barely trained on synthetic data, so every starting prediction is within ~0.004 of (0.5, 0.5); the v40 median rule fixed `σ_y ≈ 0.0034`, and any later spread of target predictions made the L kernel almost an indicator. `/experimental-deliberation` resolved it (task log L56, plan v42 §7.7): `σ_y = 1` on the normalised predictions, as in the paper. The v42 smoke (2026-11-03, same seed and data) prints `phase2_loss [1.241453, 1.215438, 1.218374]`, `phase2_cs [0.551831, 0.527378, 0.530356]`, `phase2_ccs [0.00204, 0.000478, 0.000436]`, `zBandwidth 1.3180721409123146`, `yBandwidth 1.0`, best epoch 0: the conditional term no longer explodes. Execution evidence only.
- Nothing ran on BALLADEER data, on a GPU, or at the plan's scale; no number from the paper is reproduced or claimed.
- The paper's lineage (Jenssen et al. 2006 for Eq. 11; Yu et al. 2023 for Eq. 13) is not ingested; the estimators were checked against the paper's own derivation (App. B) and closed forms, not against those papers.
