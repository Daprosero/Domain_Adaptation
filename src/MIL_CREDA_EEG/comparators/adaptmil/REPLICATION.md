# AdaptMIL: BALLADEER controlled variant

## Code search and route (2026-11-02 UTC)

Route: reimplementation of the ABMIL mechanism with the LMMD objective (AdaptMIL-LMMD, plan v60 §7.1/§7.3; MMD until v50), **controlled variant**, not a replication of published results. No upstream AdaptMIL repository was adopted; URL/commit/license: not applicable. No third-party files were copied.

| Place searched | Answer |
|---|---|
| Ingested paper, abstract, introduction, Appendix A/B, supplementary material and links | Code and slide lists promised upon publication. TLlib is a dependency link, not AdaptMIL's implementation. |
| https://arxiv.org/html/2610.28007v1 and https://arxiv.org/abs/2610.28007 | Same promise; no method repository link found among HTML external links. No peer-reviewed venue declared. |
| Web: `"AdaptMIL" "github" Vardi`; full title plus `code`; `"AdaptMIL" site:github.com` | No usable AdaptMIL code located. DAMIL is a different paper and was excluded. |
| Web: `"AdaptMIL" "Ben Vardi" Shamir publication`; `"AdaptMIL" "MIL-Lab"` | arXiv paper and secondary indexes; MIL-Lab is the ABMIL dependency, not method code. |
| Author searches: `Ben Vardi personal website AdaptMIL`; `Ariel Shamir AdaptMIL publications` | No method code located; Shamir faculty site surfaced. |
| https://www.arielshamir.com/publications.html | DNS resolution failed, not evidence of absence. |
| https://faculty.runi.ac.il/arik/site/ | Readable extraction failed; not evidence of absence. |
| Venue search via full title/publication query and arXiv metadata | Only the preprint located; no venue code page identified. |

Source Markdown: `guidance/experimental-papers/adaptmil-2610.28007/adaptmil-2610.28007.md` in the forge (read whole, initially before its operator-directed move from reference-papers). Lineage read whole: `guidance/introduction/ilse2018/ilse2018.md`, especially §2.4 Eqs. 7–8. A filename search under guidance for MMD found no ingested Long et al. (2015) paper. That lineage paper and MIL-Lab's architectural defaults are not independently audited here: estimator and projection details absent from AdaptMIL are marked unspecified below.

## HTML cross-check of ingestion

Fetched raw HTML with the target interpreter and extracted every equation's LaTeX annotation and Appendix A. Scratch (retained outside the worktree): `/var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.UcTsx02T1m/paper.html`. The HTML is evidence, not an executable dependency. Checked 2026-11-02 against version v1.

| Item | HTML answer and implementation consequence |
|---|---|
| Eq. (1), §3.1 | Mean source task loss + lambda DA over representations. `uda_objective`; logits `(Ns,2)`, labels `(Ns,)`, domain vectors `(Ns,256)/(Nt,256)` -> scalar. |
| Eq. (2), §3.1 | Same loss on **attention-aggregated** MIL representations. `single_stage_objective` is an audit/reference function, not the BALLADEER adaptation path. |
| Eq. (3), §3.3 | HTML uses `\\begin{split}` and **L_stage1**, not the ingested L_stage. Source CE on mean(P(z)) plus lambda DA on source/target mean(P(z)); restored stage index 1. `stage_one_objective` is a reference function; the controlled phase-2 objective uses its mean-pooled DA summand only. |
| Eq. (4), §3.3 | L_stage2 is mean source CE on attention(P(z)); P frozen in the original. `stage_two_objective` is a reference loss, not an original-paper training scheduler. |
| Mean pooling, §3.3 | Exactly `1/m_i sum_k P(z_i,k)`, over instances of each bag, never across bags. |
| §4.3 | 80 epochs total, 40 per original stage, confirmed; not inherited by the standardized schedule. |
| Appendix A | AdamW, cosine from 1e-4 to zero with restart; classifier carried across stages; batch 16 = 8 source + 8 target; source balanced by class; lambda MMD 2.5, LMMD 1, CORAL 15, DANN 1; MK-MMD from TLlib; slide/attention widths equal PFM feature dimension. All confirmed in HTML. |
| Table 5 ingestion defects | No dataset-composition cell is used to configure BALLADEER or assert a reproduced result. No reliance on damaged cells. |

## Plan and equation analysis (before code)

Binding plan: `experiments/experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md`, §§4.5, 7.1–7.4, 7.7 (originally written against v40, task T6/S1–S5; the LMMD switch follows the operator decision of 2026-11-04, `odd/tasks/pipeline-phase-review.md` L60, L62).

- `model.py`: existing EEGNetTrunk instances encode ragged bags `(mi,C,512)` into `(mi,256)`; native projection P maps each instance to `(mi,256)`; ABMIL scores `(mi,)`, weights softmax over instances, weighted sum `(256,)`; existing EEGNetHead maps stacked bags `(B,256)` to `(B,2)`. One projection, attention and head shared across domains. No common MIL wrapper, no MIL-CREDA consensus or L1 attention constraint.
- `objectives.py`: mean pooling, Gaussian multi-kernel biased MMD, the four paper-equation reference functions (Eqs. 1–3 keep MK-MMD as the paper writes them), and the phase-2 LMMD. The controlled adaptation loss reads source labels and the frozen attention/head's target soft predictions as class weights, never target labels. Fixed scalar bandwidth derived only from training bags after the bridge.
- `training.py`: supervised source phase fits EEGNet source + native ABMIL/projection + common head, for exactly `epochs1` epochs, last epoch kept (no early stopping, no restore; corrected 2026-11-04 from "source-validation early stopping and restoration"). Shared `two_phase.bridge` copies compatible EEGNet layers, freezes source/projection/attention/head and keeps source buffers fixed. Adaptation fits target EEGNet only using LMMD between **mean-pooled projected** bag vectors, the frozen head's target soft predictions as class weights (corrected 2026-11-04 from "MMD", the v50 variant); per-epoch target-validation predictions feed the search's pooled selection only. Shared evaluate, represent (the aggregated bag vector the head reads, for the centroid rule of plan v60 §4.5), balanced_accuracy, ClassQueues, batching and BagTensor are reused. Test evaluation is a separate caller operation.
- `__init__.py`: explicit registration into existing model-store builder and phase-two search hook. No import-time registry mutation. The generic phase-one search trainer is hardcoded to the MIL-CREDA normalized CE; native CE pretraining is exposed separately, not silently routed through that trainer.

Original-paper ordering is deliberately **not** implemented as the experiment scheduler. Reference objective functions do not claim to reproduce its training runs. The probe reports `nextStep: convert` from repository-wide mixed numpy/torch import inference; the read stage reference requires changing existing packages, outside this delegation. No conversion, comparison, GPU or remote action is taken here.

## Provenance and configuration

`match` means the stated choice is implemented, not that pathology performance was reproduced. `deviation` includes deliberate plan-mandated differences. Every choice not fixed by the paper or plan is explicitly `unspecified`.

| Choice | Paper/plan source | Implementation | Verdict |
|---|---|---|---|
| Comparator ABMIL + LMMD | Plan v56 §§7.1,7.3; paper §4.3, Tab. 1 (AdaptMIL-LMMD) | Native projected ABMIL; mean-pool LMMD | match |
| Bag units and labels | Plan §§3.2,7.2 | Ragged subject-condition bags; CE once per bag, never per window | match |
| EEGNet encoder | Plan §7.3 | Reuse EEGNetTrunk, 8/2/16 filters, kernels64/16, pools4/8, dropout.25, maxnorm1; C=14 or29, T=512, d=256 | deviation: replaces frozen PFM with trainable EEGNet |
| Head | Plan §7.3 | Existing EEGNetHead linear256->2 + softmax, maxnorm.25 | deviation: standardized head |
| Projection and attention dimensions | Paper Appendix A | d=256, attention width256, equal encoder embedding dimension | match |
| Projection nonlinearity and bias | unspecified | Linear256->256 with bias then ReLU; simple trainable projection, no dropout | unspecified |
| Attention scoring | Ilse §2.4 Eq.8 | Ungated w^T tanh(Vh), no bias, softmax over instances; no L1 projection | match |
| Attention gating choice | unspecified | Ungated Ilse Eq.8 rather than Eq.9; simplest ABMIL instantiation | unspecified |
| Initialization of native layers | Ilse §4 | Xavier uniform weights, zero projection bias | match |
| Embedding normalization | unspecified | No L2 normalization; preserve EEGNet coordinates | unspecified |
| Native floating-point arithmetic | unspecified | EEGNet float32, projection/attention/head/MMD float64 for stable invariant checks | unspecified |
| Mean aggregation | Paper §3.3 / Eq.3; plan §7.3 | Mean after P, one vector per bag, no attention weights | match |
| MMD estimator | unspecified (paper names TLlib MK-MMD only) | Biased all-pairs V-statistic incl. diagonal; handles unequal batches and singleton domains | unspecified |
| Gaussian definition | unspecified | exp(-squared_distance/(2*sigma^2)) | unspecified |
| Kernel mixture | unspecified | Mean of three Gaussian kernels at sigma times (.5,1,2), not TLlib equivalence claimed | unspecified |
| Median bandwidth rule | Plan §7.7 specifies multiplier, not median scope/cadence | Initial joint source+target TRAINING mean-projected bag vectors; positive off-diagonal Euclidean median, fixed for phase2; collapsed median ->1 | unspecified |
| Bandwidth multiplier search | Plan v56 §7.7 "el ancho de sus núcleos" | kernelMultiplier in {.5,.75,1,1.5}, shared distribution reused | match |
| LMMD loss weight | Paper A: LMMD 1 (MMD 2.5); plan v60 §7.7 "el peso de la pérdida LMMD" | Key kept as `mmdWeight` (same search family and shared `search.LOSS_WEIGHT` distribution), now weighting LMMD; no production default | deviation: standardized search |
| Phase1 loss | Plan §7.3; paper Eq.4 supervised objective | Raw mean bag CE, no MMD; fits source trunk/P/attention/head | deviation: source-first instead of mean-align-first |
| Bridge | Plan §7.3 | Existing copy_compatible_layers via two_phase.bridge; mismatched spatial conv retains fresh initialization | match |
| Phase2 trainable state | Plan §§7.3–7.4 | Target trunk only; P/attention/head and source params+BN buffers frozen | deviation: no native attention-learning second stage |
| Phase2 objective | Plan v56 §7.3; paper Eq.3 DA summand with LMMD | `alignment_objective`: weight x `local_mmd` on mean-projected representations; source CE has no target gradient and is not optimized | deviation: controlled target-only optimization |
| LMMD definition | Zhu et al., TNNLS 2021 (DSAN), https://ieeexplore.ieee.org/document/9085896; paper Appendix A builds on the Transfer Learning Repo LMMD | Mean over kept classes c of the squared RKHS distance between the source-weighted and target-weighted kernel means; biased V-statistic, unequal batches allowed | match |
| LMMD class weights | Plan v56 §7.3; DSAN Eq. 4 | Source: one-hot label divided by its class count; target: soft prediction divided by its class total; each kept column sums to one; detached (no gradient through the weights) | match |
| Target soft predictions | Plan v56 §7.3 | Head softmax of the attention-aggregated bag from the training target encoder, frozen native attention and head; no threshold; same target forward pass as the alignment | match |
| Kept classes | unspecified by the paper; Transfer Learning Repo `cal_weight` (recalled, not re-fetched in this run) | A class enters only when present among the batch's source labels and the target argmax; the sum is divided by the classes kept; none kept gives a zero connected to the graph | unspecified |
| LMMD kernel | unspecified (repository uses its own multi-kernel Gaussian with batch bandwidth, recalled, not re-fetched) | The module's existing `mixture_kernel` (three Gaussians, sigma x (.5,1,2)) shared with `multi_kernel_mmd`, sigma the fixed training median below | unspecified |
| LMMD vs UDA-DDA CMMD | Plan v56 §7.3 | UDA-DDA (`comparators/uda_dda/losses.py`) aligns the common attention's bag representation, adds a marginal MMD, and its CMMD keeps only target bags whose pseudo-label passes two confidence thresholds, each class an unweighted MMD; AdaptMIL-LMMD aligns the mean-pooled P(z), has no marginal term and uses every target bag with soft weights. The similarity of the two is declared | declared |
| Choice of LMMD over MMD | Plan v56 §§7.3,9; paper §4.5 line ~180 | The authors report "MMD is the best-performing objective for two of the three PFMs, while LMMD and CORAL also yield notable gains"; the plan uses LMMD because its selection rule discards marginal-only alignment | deviation: not the paper's best objective |
| Optimizer | Paper Appendix A | AdamW in both controlled phases | match |
| Optimizer beta/epsilon | unspecified | PyTorch AdamW defaults (.9,.999), eps1e-8; explicit weight-decay input | unspecified |
| LR schedule | Paper Appendix A | No cosine in controlled schedule; fixed lr1/lr2 supplied from search | deviation: standardized controlled training |
| Learning rates/decay search | Plan §7.7 | Reuse shared lr1/lr2/wd1/wd2 spaces; no extra optimizer axes | match |
| Epoch caps | Plan v56 §§7.6–7.7; paper40+40 | Required epochs1/epochs2 inputs, no patience; no production cap default (row corrected 2026-11-04: it listed patience1) | deviation: pilot decides caps |
| Batch size | Paper A:8+8; plan does not fix size | Required bagsPerStep; shared class-balanced source queues and shuffled target pass, final partial target batch retained | unspecified |
| Source phase1 sampling | Plan does not fix sampling; paper A balances source | Shared class-interleaved pass, no oversampling in phase1; balanced queues in phase2 | deviation: phase1 may have unequal class counts |
| Primary selection | Plan v56 §4.5 | Both caps run whole and keep the last epoch; the search picks (trial, E1) by pooled source-validation BA and (trial, E2) by pooled target-validation BA over the three inner splits (row corrected 2026-11-04: it described a best-epoch selection inside the run) | deviation: target labels used for model selection only |
| Secondary selection | Plan v56 §4.5 | Shared H(mean p)-mean H(p), per epoch, pooled by the search; no secondary snapshot is kept in the run (row corrected 2026-11-04) | match |
| Epoch ties | unspecified | Earliest epoch on equal score; strict greater-than replacement, deterministic | unspecified |
| Seed | Plan §7.7 | Caller seed passed to torch and generator; shared search seeds reused by hook | match |
| Checkpoint storage | Plan §7.3 | Existing model_store; explicit registration; comparator source digest embedded and checked by package save/load helpers | match |
| Evaluation chunk | unspecified | Existing two_phase.EVAL_CHUNK=512; no chunking during training | unspecified |

## Integration surface

Import `MIL_CREDA_EEG.comparators.adaptmil` by its full path. Call `register()` explicitly in the parent runner to register its builder and the phase-two search callback; importing the package does not change global registries. `train_phase_one(data, rotation, seed, params, schedule, device)` provides native-CE source pretraining and returns the same trained-result shape as the shared search. `search.train_phase_two("AdaptMIL", ...)` works after registration, using stored phase-one state and metadata from the same group/transfer/rotation/seed/partition. Its record must include `adaptmil_source_digest`.

Use package `save_model` / `load_model` to delegate tensor, partition and shared-code checks to the existing model store and also enforce the comparator-source seal. The generic store does not hash comparator sources, and generic search save helpers do not add that seal. The parent must route AdaptMIL phase-one trials to the native trainer and saves to these helpers; the existing `search.train_phase_one` is hardcoded to MIL-CREDA's normalized CE and is **not** a native-CE AdaptMIL search entry point. Phase-one sidecars contain only `lr1`, `wd1` and the required schedule, never the shared schema's unread NaN placeholders. No monkeypatch of shared training or modification of an existing module was made. `controls.source_only` and `two_phase.evaluate` can consume this model's existing-compatible `predict` interface.

## Verification record

All commands below ran from `/Users/diego/Proyectos/papersmith-ai/implementations/Domain_Adaptation-worktrees/adaptmil`, detached at the parent's baseline `e4bf532`. Preflight `git status --porcelain` was empty. `.venv/bin/python` is the operator-arranged symlink to the target interpreter; **PYTHONPATH must point to this worktree's src**, because the editable install points at the main checkout. No main-checkout write, install, commit, branch operation, partition write, GPU execution or remote launch occurred.

1. `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_balladeer_comparator_adaptmil.py -q` — exit 0; **26 passed in 2.86s** (last run). Covers 18 declared invariants: native shapes and permutation symmetry, uniform-attention closed form, mean pooling per bag, all phase-one-used parameter gradients finite, all target-trunk adaptation gradients finite and nonzero, Gaussian singleton MMD closed form, identity/translation/domain-swap symmetries, gradcheck, invalid inputs, Eqs.1–4, no attention or head on alignment path, zero-weight gradient control, target-training-label poisoning, freeze including source BatchNorm buffers, both montage directions' compatible copies, primary/secondary checkpoint selection and restoration, registration/search spaces, record identity, storage round trip/code-seal rejection, role/hyperparameter guards. A deliberate attempt to unfreeze attention is refused.
2. `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q -k balladeer` — exit 0; **276 passed, 2 skipped, 821 deselected, 4 warnings in 22.14s**. Warnings concern existing experimental Optuna GPSampler. This is the isolated baseline worktree plus AdaptMIL; concurrent sibling comparator additions in other worktrees are not included.
3. `PYTHONPATH=$PWD/src .venv/bin/python /var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.UcTsx02T1m/smoke.py` — exit 0. CPU-only, source29 -> target14 channels, 512 samples/window, two bags per role with 68 and 69 windows each, 3+3 epochs. Seed 418312497 is derived from unit id `adaptmil-synthetic-device-rotation0` by `int(sha256(unit_id).hexdigest()[:8],16) % 2**31`. Printed phase1 losses `[0.6940715184219053, 0.6875595051073355, 0.6799133702097959]`; phase2 losses `[3.825061083721595, 3.8417206394576313, 3.827970963195038]`; sigma `0.14604611079884017`; prediction shape `[2,2]`; `finite: true`; stored and loaded phases `[1,2]`; last elapsed `4.341970167006366` seconds. The adaptation loss moves but is **not monotonically decreasing**. This is execution evidence, not a pathology or BALLADEER result. The scratch script creates a new output directory on each invocation so the model store's no-overwrite rule is preserved.
4. `PYTHONPATH=$PWD/src .venv/bin/python /Users/diego/Proyectos/papersmith-ai/skills/experimental-implementation/scripts/implementation_cli.py verify --target "$PWD" --name MIL-CREDA` — exit 0, read-only, **not a green fidelity verdict**: structure drift; priorWork modified; agreements open; position absent; search none; distribution none; remoteExecution absent; fidelity unknown (`latestRevision:null`); lfs pointers; report drift; audit none; validation incomplete. Prose reports 0 stale revisions and 1146 unresolved symbols; coupling reports existing Toyset notebook cell coupling. This repository-wide tool infers package `MIL_CREDA`, not comparator-specific coverage. Its report is retained as `scratch/verify.json`; no out-of-scope repair was attempted. It reports 18 pre-existing `.pth` LFS placeholders totaling 5,465,148,680 bytes (5.09 GiB); none was opened/fetched. Fetching them by any route would spend the repository owner's LFS bandwidth. Synthetic training requires none of them.
5. Direct AST inspection of the four new Python modules and this test file found **18 provenance invariant ids, zero missing test witnesses**, all bound to plan v40. `git diff --name-only` was empty (no tracked file modified); `git ls-files --others --exclude-standard` listed only the four comparator Python modules, this REPLICATION.md, and `tests/test_balladeer_comparator_adaptmil.py`. `test -d MIL-CREDA/Data/processed` returned exit 1, confirming that no processed-data directory is present here.

Retained scratch directory: `/var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.UcTsx02T1m` (not deleted). It holds HTML, probe/verify output, test logs, smoke script/output and synthetic checkpoint artifacts. **12** provenance-table rows have verdict `unspecified`; count by lines ending `| unspecified |`.

### Later parallel campaign: assessment attempted, measurements missing

The plan's four rotations times three report seeds imply 12 fold/seed units per transfer, not 12 arms. Ran with the target interpreter:

`.venv/bin/python /Users/diego/Proyectos/papersmith-ai/skills/remote-parallelism/scripts/parallelism_cli.py assess --axis fold --units 12`

Exit 2, exact refusal: `"error": "missing-input"`, `"message": "measured inputs are required, never guessed; missing: --unit-minutes, --owners, --sessions-per-account, --session-limit-hours, --quota-hours, --setup-minutes"`. No split verdict, speedup or quota was fabricated; the synthetic CPU smoke is not a remote pilot-unit measurement. The refusal's objective starts at measured inputs, proceeds through assessment and checklist satisfaction, and arrives at a recorded verdict before job generation. Measuring remote setup/limits requires the later owner/launch stretch, not this one. Captured refusal in `scratch/parallelism.json`; `checklist` exited 0 into `scratch/checklist.json`.

Checklist inherited by the launcher: split by declared fold/seed units, never comparator; derive seed from unit id; pin identical published commit and environment requirements digest; pin/request/stamp accelerator; stamp Python/torch/CUDA/cuDNN/accelerator/commit/requirements/data checksum/preparation version and require equality at merge; deterministic algorithms, manual seed, CUBLAS workspace `:4096:8`, cuDNN benchmark off; same official data/preparation; passing rehearsal per account; never pool timings/memory across environments; one distinct owner per account/service; one service per campaign unless environment stamps match; operator alone authorizes launch. Only the synthetic unit-id seeding and CPU deterministic flag were exercised here; remote checklist items remain unmeasured.

### Not tested / still owed

No original-paper schedule, pathology dataset, published metric, DSMIL/TITAN, CORAL/DANN, numerical parity with the Transfer Learning Repo LMMD, real BALLADEER training, GPU, remote quota, full 20-trial/four-rotation Optuna campaign, interrupted remote continuation, or downstream Results figures/statistics were exercised. The Long et al. (2015) MMD lineage paper was not ingested/read; the MMD estimator choice is therefore a declared unspecified default, not a claim of TLlib parity. Parent-owned native phase-one trainer dispatch and comparator-sealed save routing remain to be connected in the full experiment runner. The operator's fidelity decision remains open.
