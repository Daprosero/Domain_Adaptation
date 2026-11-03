# UDA-DDA: controlled BALLADEER variant

Tang, Li, Su, et al., «UDA-DDA: Unsupervised domain adaptation with dynamic
distribution alignment network for emotion recognition using EEG signals»,
Neurocomputing 658 (2025) 131715, DOI 10.1016/j.neucom.2025.131715.

This is a **controlled variant** under plan v40 §7.4, not a reproduction of the
paper's accuracy, features, backbone or schedule. No number in this file is a
result of the paper reproduced here.

## Code search and route (2026-11-02 UTC)

**Route: paper-based reimplementation.** Official code exists but carries no
license, so nothing was adopted, vendored or copied; the repository was read
only to audit this reading of the paper.

| Place searched | Answer |
| --- | --- |
| Ingested paper, abstract | «The source code is accessible at https://github.com/XuanSuTrum/UDA-DDA». |
| GitHub API `repos/XuanSuTrum/UDA-DDA/commits/main` | HEAD `6e82664a55a99d257daff24d9c92cf109a98213c`, 2026-06-28, «Add files via upload». |
| GitHub API `repos/XuanSuTrum/UDA-DDA/license` | HTTP 404. Repository metadata `license: null`. |
| Recursive tree at the pinned HEAD | 13 blobs: README.md, SDA_DDA.py, SEED_3394.m, SEED_3394_IV.m, backbone.py, cmmd.py, create_de4.py, get_dataset.py, load_data2.py, main.py, mmd.py, requirements.txt, utils.py. No LICENSE/COPYING. README grants no license. |
| arXiv 2502.16485 (same first author, SDA-DDA preprint) | Read as a secondary source for garbled equations only (PDF text layer). |
| Plan v40 §7.1 / task log L16, L31 | Paper paywalled; the operator supplied the PDF; ingested under `guidance/experimental-papers/uda-dda-neurocomputing-2025/`. |

Upstream files read for the audit (README.md, SDA_DDA.py, backbone.py, cmmd.py,
main.py, mmd.py, utils.py) live only in the scratch directory named in the
report, never in this repository. Consequence for plan §7.1: its baselines
table says the code column is the official repository; under the license rule
that row is a reimplementation from the article, as for CCA-LSC (task log L19).

## Sources and cross-checks

Primary source: the ingested Markdown (10 display equations), cross-checked
equation by equation against the filed PDF's own text layer (pypdfium2,
papersmith env), and against the SDA-DDA preprint where both are unclear.

| Item | Ingested Markdown | PDF text layer | SDA-DDA preprint | Used here |
| --- | --- | --- | --- | --- |
| Eq. 1 (differential entropy) | prose spliced in | clean `H(x) = -∫ p log p` | — | not used: plan replaces DE features by EEGNet bag representations |
| Eq. 2 (MMD) | clean | clean | same | `losses.mmd2` |
| Eq. 3 (trace form) | prose spliced in, block matrix malformed | split across columns, structure `tr(K M)` legible | — | identity tested numerically against Eq. 5 |
| Eq. 4 (M) | both diagonals 1/n_s², off-diagonal 1/n_s^t | `1/n_s²` (D_s,D_s), `1/n_t²` (D_t,D_t), `−1/(n_s n_t)` otherwise | — | text layer; `losses.mmd_matrix` |
| Eq. 5 (MMD² estimator) | prose spliced in, terms readable | clean, biased V-statistic | — | `losses.mmd2` |
| Sec. 3.2 kernel | `k(u,v)=e^{-‖u−v‖²/σ}` (exponent split) | same | — | `losses.gaussian_kernel` |
| Eq. 6 (CMMD) | `K_{t,t}` term garbled | clean: (1/C)Σ_c [K_ss/(n_s^c)² + K_tt/(n_t^c)² − 2K_st/(n_s^c n_t^c)] | — | `losses.cmmd2` |
| Eq. 7 (confidence filter) | clean; prose: 0, 0.5, 0.75, 1 at epochs [0,10), [10,40), [40,85], >85 | same | same schedule | `losses.confidence_threshold` |
| Eq. 8 (source CE) | clean | clean | — | `losses.source_cross_entropy` |
| Eq. 9 (β) | clean | clean | same (its Eq. 8), ρ0 = 0.1, ρ1 = 0.15 | `losses.cmmd_weight` |
| Eq. 10 | clean | clean | — | `training.adaptation_step` |
| Sec. 4.2 α, τ_h, τ_l | «configured as 1, 0.01 and 1» | same words | «α, τh and τl are 1, 0.01» (two values for three names) | contradictory; resolved below as `unspecified` |

**The delegation premise «pseudo-label confidence filtering does not belong to
UDA-DDA» is contradicted by the UDA-DDA paper itself:** its abstract, Sec. 3.3
(Eq. 7), Algorithm 1 step 4 and ablation EXP6 all include the filter, and plan
v40 §7.7 lists «umbrales de confianza de las pseudoetiquetas» in UDA-DDA's
phase-2 search. The filter is implemented. What is excluded from SDA-DDA is the
semi-supervised use of labelled target samples: no target training label is
read anywhere (test witness `uda_target_training_labels_are_unread`).

Table 1 (notation) is scrambled in the ingestion and was not needed: every
symbol used is defined in the prose next to its equation.

Lineage: [38] (MMD trace form) and [39] (conditional kernel mean embedding) are
not ingested. Eq. 3/4 was verified numerically instead; Eq. 6 is used as
printed.

## Module and shape map

Existing modules read before writing and reused, none edited: `eegnet.py`,
`mil_wrapper.py`, `mil_creda_model.py`, `two_phase.py`, `search.py`,
`model_store.py`, `assignment.py`, and the committed comparators `cca_lsc` and
`cot` as patterns.

- `losses.py`: Eqs. 4–9, the rescaled Eq. 7 stages, the α schedule, the median bandwidth.
- `training.py`: phase-2 adapter, common-checkpoint search hook, explicit `register()`.
- `smoke.py`: seeded synthetic CPU execution check.
- `__init__.py`: exports only; importing registers nothing.

Shapes: bag `(m, C_domain, 512)` → shared EEGNet-8,2 trunk `(m, 256)` →
common attention (Eq. 15, γ = 0) → bag representation `(B, 256)` float64 →
frozen common head `(B, 2)` probabilities. MMD and CMMD read `(B, 256)` bag
representations of both sides; pseudo-labels and the confidence mask are
`(B_t,)`. Source and target keep 29/14 (or 14/29) channels.

## Provenance table

Every row cites a source or is `unspecified` with the choice made and why.
For audited upstream code the column «code @6e82664a» says what it does and
the verdict compares the code with the paper (`match`, `deviation`,
`unspecified`).

| # | Choice | Value here | Source | Code @6e82664a | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Encoder | EEGNet-8,2 trunk, shared phase-1 checkpoint | Plan §7.2, §7.3 (paper: DE features + 2 FC layers, Sec. 3.1, 4.2) | `backbone.feature_extractor` 310→64→64; dropout layers declared but never applied in `forward` | deviation: code omits the paper's 0.25 dropout. Plan replaces both. |
| 2 | Bag aggregation and head | common attention γ = 0, common linear head, frozen | Plan §7.2, §7.3 | classifier 64→32→3 with ReLU and dropout 0.5, and the optimizer never includes it (Adam over `base_network` only) | deviation: classifier stays at initialization in the code; unspecified in the paper. Plan fixes ours. |
| 3 | Classes | 2 | Plan §7.4 «pasa de tres clases a dos» | 3, hard-coded in `cmmd.py` one-hots | — |
| 4 | Trainable in phase 2 | target trunk only | Plan §7.3 | whole feature extractor, both domains | plan-controlled deviation |
| 5 | Marginal term | biased MMD² V-statistic, Eq. 5 | Eq. 2–5 (Eq. 4 per the PDF text layer) | `mmd_rbf_accelerate`: linear-time paired estimator, assumes equal batch sizes | deviation: code estimator differs from Eq. 5 |
| 6 | Kernel | one Gaussian `exp(−‖u−v‖²/σ)` | Sec. 3.2 | sum of 5 Gaussians, bandwidths base·2^k, base = mean pairwise squared distance / 4 | deviation: code is multi-kernel, not the paper's single kernel |
| 7 | σ value | `kernelMultiplier` × median pairwise squared distance of the pooled batch | Plan §7.7 generic rule (paper fixes no σ) | data-dependent mean rule per batch | unspecified in paper |
| 8 | σ cadence | recomputed every step from that step's detached representations | **unspecified**: follows the code's per-batch cadence. Measured reason: a bandwidth fixed once in eval mode (σ = 0.083 in the smoke fixture) saturates against train-mode target distances (24–35), the MMD gradient vanished and MMD read 0.821641 at all 10 epochs | per batch, `L2_distance.data` (detached) | unspecified |
| 9 | Squared vs plain median | squared distances | **unspecified**: the paper's kernel divides ‖u−v‖², so σ is on the squared scale | squared (mean) | unspecified |
| 10 | Degenerate bandwidth | fallback 1.0 when no positive distance | **unspecified**: numerical guard | none (division by zero possible) | unspecified |
| 11 | Same σ for MMD and CMMD | yes | **unspecified**: the paper names one kernel; the code computes both with the same rule | same rule, separate calls | match (rule) |
| 12 | Conditional term | Eq. 6: (1/C) Σ_c MMD² of class c, per-class normalization | Eq. 6 (PDF text layer) | `cmmd`: one-hot masked kernel means over **all** pairs (normalized by n_s², n_t², n_s n_t, not by class counts), divided by 3 | deviation: code normalization differs from Eq. 6 |
| 13 | Class absent from source or from the selected target | contributes 0, still divided by C | **unspecified**: Eq. 6 undefined there | empty target: source–source term only (constant, no gradient for target) | unspecified; gradients agree |
| 14 | Pseudo-label | argmax of the frozen common head on the target bag, detached | Sec. 3.3 «ŷ = f(x)» | `target_clf.max(1)` on softmax | match |
| 15 | Confidence filter | keep target bags with max probability `>` τ (strict) | Eq. 7, Γ | `target_probabilities > confidence_threshold` | strictness unspecified in paper; matches code |
| 16 | Confidence schedule | 0 / `confidenceMiddle` / `confidenceLate` / 1 on paper epochs [0,10), [10,40), [40,85], >85 | Eq. 7 prose, Sec. 3.3 | same values, same boundaries (`e <= 85`) | match |
| 17 | Defaults of the two searched thresholds | 0.5, 0.75 | Eq. 7 prose | 0.5, 0.75 | match |
| 18 | Threshold 1 in the last stage | no bag passes; CMMD is 0 there | Eq. 7 prose («only 100 % confidence») with the strict filter | same consequence | match |
| 19 | Stage driver | epoch, not source loss | Eq. 7 prose and code; Sec. 3.3 also says «stages are defined based on the source domain classification loss» | epoch | paper self-contradictory; epoch chosen (code, and the plan's rescaling rule speaks of the epoch schedule) — counts as unspecified |
| 20 | Epoch rescaling | phase-2 epoch e of E sits at paper epoch 100·e/E | Plan §7.4 «su calendario por épocas se reescala al presupuesto de la fase 2»; paper 100 epochs, Sec. 4.2 | 100 epochs | plan |
| 21 | α schedule endpoints | τ_h = 1, τ_l = 0.01 | **unspecified**: Sec. 4.2 reads «α, τ_h and τ_l … 1, 0.01 and 1», which contradicts Sec. 3.4 «reduced … to τ_l»; the preprint gives two values for three names. Only τ_h > τ_l fits the prose | 1 until epoch 10, ≈ 0.99991 for (10,40] (a constant: `2/(1+exp(−10·n/n))−1`), `exp(−0.6 e)` ≈ 4e-11 for (40,85], 0 after | unspecified |
| 22 | α schedule shape | τ_h on [0,10), geometric decay to τ_l across [10,40), τ_l from 40 on (paper epochs) | **unspecified**: Sec. 3.4 «gradually reduced … then reduced to τ_l» states direction, not shape | step-like, effectively 1 then 0 | unspecified; code and prose disagree |
| 23 | α boundary convention | same stage boundaries as Eq. 7 (`< 10`) | **unspecified** | `epoch <= 10` | unspecified |
| 24 | MMD coefficient | `mmdWeight` × α(e), default 1 | Sec. 4.2 «α … 1»; plan §7.7 «pesos MMD» | `beta * transfer_loss` (no extra factor) | match at default |
| 25 | β | Eq. 9 step: 1 if L ≤ ρ0, 0.5 if ρ0 < L < ρ1, 0 otherwise | Eq. 9; ties as the code | `clf_loss <= 0.1 → 1; < 0.15 → 0.5; else 0` | match |
| 26 | ρ0, ρ1 | 0.1, 0.15, not searched | Sec. 4.2 | 0.1, 0.15 | match |
| 27 | L_Ds feeding β | CE of the **frozen** common model on the step's source bags | Eq. 8, 9; plan §7.3 freezes the source side | CE of the jointly trained model | plan-controlled deviation, consequence below |
| 28 | CMMD coefficient | `cmmdWeight` × β, default 1 | Eq. 10 (no coefficient beyond β); plan §7.7 «pesos … CMMD» | `cmmd_weight * cmmd_loss` | match at default |
| 29 | Source CE in the objective | kept as a frozen summand (no gradient), logged | Eq. 10 with plan §7.3 | trained | plan-controlled |
| 30 | Source mini-batch | class-balanced draws via `two_phase.ClassQueues`, `bagsPerStep // 2` per class | **unspecified**: Algorithm 1 «random a mini-batch»; balanced draws keep every class in Eq. 6's source side with few bags | random DataLoader | unspecified |
| 31 | Target mini-batch | `bagsPerStep` bags, reshuffled each epoch | Plan §7.7 schedule keys | DataLoader | plan |
| 32 | Optimizer | Adam over the target trunk, `lr2`, `wd2` searched | Plan §7.7 shared axes (paper: SGD with 5e-4 L2 decay and an unstated LR formula, Sec. 4.2) | Adam lr 0.05, no weight decay | deviation: code contradicts the paper's SGD and decay |
| 33 | Batch size | `bagsPerStep` (schedule) | Plan §7.6/§7.7 (paper 128, Sec. 4.2) | 32 in `main`, 256 in argparse | deviation inside the code itself |
| 34 | Max-norm | `eegnet.apply_max_norm` after each step | Plan §7.3 (EEGNet-8,2 config) | — | plan |
| 35 | Epoch selection | target-validation balanced accuracy, label-free score recorded as the secondary rule | Plan §4.5 | best accuracy on the target **test** loader across epochs | deviation: the code selects on test labels; not adopted |
| 36 | Search axes | `lr2`, `wd2`, `mmdWeight`, `cmmdWeight` (loss weight), `kernelMultiplier`, `confidenceMiddle`, `confidenceLate` (confidence threshold) | Plan §7.7 row and generic rule; `search.register_phase_two` | — | plan |
| 37 | Range of `confidenceMiddle` | [0.5, 0.95] | Plan §7.7 generic rule; the paper's own sensitivity grid (Sec. 5.4.2) used [0.4, 0.7] for it and [0.7, 0.95] for the late one | — | plan; the range [0.4, 0.5) of the paper is outside the rule |
| 38 | Ordering of the two thresholds | not enforced; a trial may sample middle > late | **unspecified**: the paper describes an increasing schedule; the generic rule samples axes independently and has no ordered pair | — | unspecified |
| 39 | Default `kernelMultiplier` | 1.0 | **unspecified**: the paper fixes no σ | — | unspecified |
| 40 | Seed | caller's seed (search and report seeds of plan §7.7) | Plan §7.7 (paper: 3, Sec. 4.2) | `torch.manual_seed(0)` per subject | plan |

`unspecified` rows: 8, 9, 10, 11, 13, 19, 21, 22, 23, 30, 38, 39 — **12 rows**.
(Row 7 is fixed by the plan's generic rule and row 15 by the code, so neither is counted.)

### Consequences the operator should read

- **β is static in this variant (row 27).** Eq. 9 makes CMMD's weight a step
  function of the source cross-entropy, which in the paper falls as the
  feature extractor trains. Plan §7.3 freezes the source trunk, attention and
  head in phase 2, so L_Ds is a constant of the phase-1 checkpoint and its
  balanced source draws: the smoke run reads `source_ce` 0.687702 at all ten
  epochs and β = 0 throughout, so CMMD never entered the gradient. On real
  data CMMD contributes only if the selected common checkpoint reaches source
  training CE ≤ 0.15 (β = 1) or < 0.15 (β = 0.5). ρ0 and ρ1 are paper values
  tuned on SEED differential-entropy features and are not in the search. This
  is a plan decision for `/experimental-deliberation`, not an implementation
  choice; the code applies Eq. 9 as printed.
- **α and the confidence stages are rescaled, not shortened (row 20).** With a
  small `epochs2` each paper stage spans few epochs; with `epochs2 < 10`
  stage 0 or 3 can vanish (e.g. `epochs2 = 4` gives paper epochs 0, 25, 50, 75:
  no stage 3).
- **AdaptMIL resemblance.** Plan §7.3 declares that AdaptMIL's mean-pooled MMD
  resembles UDA-DDA's marginal term. Here the marginal term reads the
  attention-pooled representation and adds CMMD, the α schedule and the
  confidence filter.

## Verification record

Tested (commands from the worktree root, `PYTHONPATH=$PWD/src .venv/bin/python`):

- `-m pytest tests/test_balladeer_comparator_uda_dda.py -q` → 26 passed.
- `-m pytest tests -q -k balladeer` → 403 passed, 2 skipped (377 passed,
  2 skipped without this file).
- Invariant witnesses (one test per declared invariant, same name with the
  `test_` prefix): Eq. 5 against a hand-computed value and Eq. 3/4's trace form
  with the corrected M; Eq. 6 against per-class Eq. 5 and its gradient; the
  kernel and the median rule in closed form; Eq. 7's thresholds at the
  paper's own boundaries (E = 100) and after rescaling (E = 10), strict `>`;
  α monotone from 1 to 0.01 with its geometric midpoint; Eq. 9 at five
  points; empty selection finite and connected; frozen common predictor in
  29→14, 14→29 and 29→29; poisoned target-training labels never read
  (identical history and weights); bandwidth computed once per step from
  detached tensors; search registration on the generic rule and refusal of a
  foreign checkpoint; best-epoch restore with first-best tie-break; invalid
  parameters refused; objective weights effective; β reads the frozen source
  CE; importing registers nothing; synthetic smoke.
- Mutation check: 18 single-line mutations of the code (Eq. 4 sign, Eq. 5
  cross coefficient, Eq. 6 1/C, Eq. 7 strictness, the 85 boundary, Eq. 9's
  half step, the α decay, squared median, bandwidth detach, β ignoring CE,
  dropped `cmmdWeight`, reading target labels, training the head, selection
  tie-break, leakage check, checkpoint check, search axis, import-time
  registration), each run with bytecode disabled; all 18 turned the focused
  suite red, then the sources were restored byte for byte.
- Smoke: `-m MIL_CREDA_EEG.comparators.uda_dda.smoke` — seeded synthetic
  batch, 4 source bags (2 windows, 29 channels, 512 samples) and 4 target bags
  (14 channels), phase 1 three epochs, phase 2 ten epochs. Printed phase-2
  loss 1.413759 → 0.695504 (falls mostly through α's decay), MMD
  0.726057…0.780234 (moving), CMMD nonzero in stages 0–1 and 0 from stage 2
  (no target bag above 0.75 or 1), β = 0 at all epochs. Execution evidence
  only; not a measurement of the method.

Not tested: any real BALLADEER bag, any GPU run, the paper's numbers, the
search study end to end for UDA-DDA (only one trial through
`search.train_phase_two`), and the timing of a phase-2 epoch on real data.
