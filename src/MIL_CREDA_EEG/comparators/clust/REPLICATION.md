# CLUST — source-free controlled BALLADEER variant

## Code search and route (2026-11-02 UTC)

Route: reimplementation; no usable CLUST implementation located in the searches below. No adopted repository, commit, license or vendored source.

- Ingested full paper: `guidance/experimental-papers/clust-s41598-025-21560-y/clust-s41598-025-21560-y.md`, all 657 lines. Data availability links datasets, not implementation.
- Venue and author information: https://www.nature.com/articles/s41598-025-21560-y (raw HTML fetched successfully). Its only GitHub link is `thuml/Transfer-Learning-Library`, a toolbox, not CLUST code. No CLUST code/supplement repository located in the article links. Author links lead to affiliations and correspondence email, not personal code pages.
- Web queries: `"Prototype-oriented classconditional clustering transport" code github`, `"s41598-025-21560-y" code`, `"Liangda Yan" "CLUST" github`. Results located Nature/PMC/PubMed and related PCT, not CLUST code.
- Author-page lead https://www.researchgate.net/scientific-contributions/Liangda-Yan-2291473879 returned HTTP 403. Not evidence of absence there.

## Plan and scope

Binding: experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md §§4.5, 7.1–7.4, 7.7 (rebound from v40 on 2026-11-03, rebound to v54 on 2026-11-04, to v55 on 2026-11-05 and to v56 on 2026-11-05 with no change of behaviour: v56 keeps its §7.1 row and the §§4.5, 7.2–7.4, 7.7 rules this record cites); task T6/S1–S5. This is NOT a reproduction of CLUST's reported results. Only the source-free variant is implemented: phase 2 accepts target training windows, never source bags or target training labels. Source classifier weights are the only source prototypes. No source CE, source sampling or replay exists in this adapter. No Eq.16 ensemble at inference.

Reuse common phase-1 model/trainer, bridge, EEGNet, evaluation, metrics, batching, assignment/search and model store. No existing module is changed. Phase-1 checkpoint is supplied by the shared common group rather than retrained here.

Modules and boundaries:
- `losses.py`: bag vectors Z [B,256], source/target/combined prototypes [2,256], assignment probabilities [B,2], concentration [2] -> scalar losses, active-class statistics and dynamic alpha.
- `training.py`: CommonMILModel and target training bags [(m,14 or 29,512)] -> target-only adaptation with auxiliary EMA target encoder/prototype bank; model inference remains CommonMILModel.predict.
- `__init__.py`: explicit search registration and shared-trainer adapter exports.
- `smoke.py`: CPU seeded synthetic BALLADEER-shape execution; never a scientific result.

Lineage gaps: searched guidance Markdown filenames; PCT (ref.17), SHOT (refs.23/33), and teacher papers (refs.60/62) were not located as ingested papers. Implementation uses CLUST's own explicit equations; no missing lineage mathematics is supplied from memory.

## Equation analysis and correspondence

Nature HTML was cross-checked for all numbered equations (1–24). Ingested Markdown has 25 display blocks, not 25 numbered method equations. HTML resolves the corrupted arrays/prose, including symmetric KL in Eq.13 (the Markdown block is badly corrupted) and the target-prototype argument in the third line of Eq.11 (Markdown incorrectly repeats source there).

| Paper | Function / treatment | Input -> output and assumption |
|---|---|---|
| 1 | Shared phase-1 trainer | Source bag predictions and bag labels -> CE; no phase-2 source CE |
| 2,3 | prototype_probabilities / confidence_assignments | [B,d],[K,d] -> [B,K] probabilities / boolean membership |
| 4 | target_centroids | [B,d],membership -> [K,d],counts; differentiable confident target means |
| 5 | ema_update | old,current,support -> EMA; also teacher encoder after optimizer step |
| 6 | conditional_probability | [B,d],[K,d],tau[K] -> [B,K], uniform prior |
| 7,8 | transport_cost | squared Euclidean cost, conditional class sums -> scalar |
| 9,10 | Not executable | Theorem/proof only; no claim of validating this inequality |
| 11 | clustering_terms | feature and same-class prototype transport; HTML corrected third argument |
| 12 | clustering_loss | feature scalar, semantic scalar,alpha -> weighted scalar |
| 13 | symmetric_kl | two [B,K] distributions -> mean symmetric KL |
| 14 | information_maximization | [B,K] -> entropy minus entropy of mean (gamma=1) |
| 15 | objective | no L_cls in source-free phase 2; separate searched weights for clustering,consistency,IM |
| 16 | Omitted by plan §7.4 | Diagnostic inference exclusively shared frozen head |
| 17 | Shared evaluate + argmax | [B,K] -> class per bag |
| 18–20 | Not executable | Generalization-error analysis, not training loss; no target labels enter adaptation |
| 21,22,23 | a_distance / dynamic_alpha | scalar feature cost, class semantic costs -> alpha; epoch boundary, detached |
| 24 | concentration | confident features/prototypes -> per-class concentration |

## Choices and provenance

Rows marked `unspecified` explicitly state choices the paper/plan do not settle; these are audit gaps, not fidelity claims.

| Choice | Chosen behavior | Source / verdict |
|---|---|---|
| Encoder | Reuse EEGNet-8,2,256 features,512 samples,dropout .25,14/29 electrodes | Plan §7.3 controlled deviation from image backbones |
| Aggregation and head | CommonMILModel frozen shared tanh attention and EEGNetHead | Plan §§7.2–7.4 controlled deviation |
| Shared attention width | Reuse ATTENTION_WIDTH=128; no new attention parameters or capacity search | unspecified in paper/plan: inherited common-wrapper choice, not CLUST's image architecture |
| Source access | Frozen head rows only; no source examples in phase 2 | Paper Source data privacy protection; plan §7.3 |
| Trainable parameters | Target trunk only; auxiliary teacher updated without gradient | Plan §7.3 |
| Initial target layers | Shared bridge/copy_compatible_layers, preserve spatial init on channel mismatch | Plan §7.3 |
| Feature normalization | L2 bag vectors for auxiliary transport, not for diagnostic head | Paper Methodology prototype classification paragraph |
| Metric sign | d=1-cosine; softmax(-d) favors closest prototype | unspecified: text says cosine similarity but Eq.2/6 negate d; use cosine distance to avoid reversed nearest-class ordering |
| Priors | Uniform over classes | Paper Experimental details, uniform prior default |
| Confidence | Caller supplies threshold; smoke .6; >=,class-wise Eq.4 | Paper Training procedure .6 (prose also says exceeding); Eq.4 uses >= |
| Source prototype initialization | Normalized frozen common-head weight rows | Paper Methodology; plan §7.3 |
| Target fallback | Previous EMA target prototype for absent confident class; exclude absent class from centroid/semantic costs | unspecified: Eq.4 has zero denominator for absent classes |
| Combined prototype | Midpoint of frozen source and current target centroid | unspecified: source-free combined prototype weighting not given; no source counts available and no source data read |
| Teacher | EMA copy of target trunk, frozen shared attention, source prototype assignments and EMA target bank for consistency | unspecified: paper mentions momentum encoder but Eq.5 specifies only prototypes; explicit mean-teacher interpretation |
| EMA buffers | Float buffers EMA, integer BatchNorm counters copied | unspecified: paper gives no BatchNorm teacher-buffer rule |
| EMA momentum | Parameter `emaMomentumComplement` = 1 − rho in (0,1]; the trainer uses rho = 1 − value (`training.ema_momentum`); smoke 1 − rho = .001 (rho = .999) | Paper Eq.5 rho=.999; plan v42 §7.7 searches rho, drawn as 1 − rho |
| EMA prototype destination | Update target-only bank from target centroids; source classifier prototypes remain fixed | unspecified: Eq.5 labels its bank source, while source-free section does not define that update without source data; plan freezes diagnostic head |
| Missing class EMA | Unchanged bank row | unspecified: Eq.5 omits empty-class behavior |
| KL branch | Teacher source-prototype distribution detached, student target-bank distribution differentiable | unspecified: Eq.13 symmetric KL clear, stop-gradient direction unspecified |
| Class conditioning | Multiply feature cost by confident class membership; semantic transport uses diagonal same-class probability | unspecified: Eq.8 class-set sums vs Eq.11 unqualified expectation; honor explicit class-conditional cost |
| Reductions | Feature sums divided by all bags; semantic sum divided by K; no-confidence yields differentiable zero clustering | unspecified: finite-batch reduction and empty-set handling omitted |
| Concentration smoothing | delta=exp(-n)/sqrt(2*pi), lower bound 1e-6, absent classes tau=1 | Paper Eq.24 following prose fixes delta; unspecified numerical/empty-cluster guards |
| Alpha | Initial .5; epoch update from costs via Eq.21–23 | Paper Dynamic clustering transport factor |
| A-distance guards | Negative distances clipped at zero; zero denominator returns .5; alpha in [0,1] | unspecified: squared clustering costs are not bounded classification errors; Eq.21–23 can be negative/undefined |
| Auxiliary weights | Caller-supplied clustering,consistency,IM weights; smoke 1 each, gamma=1 | Plan §7.7 separate searched families; paper Eq.15 lambda=100 and Eq.14 gamma=1 vs experimental gamma=100 conflict; controlled variant uses gamma=1 and exposes IM weight |
| Optimizer | Adam on target trunk, caller lr2/wd2, constraints after step | unspecified: standardized trainer uses Adam, paper uses SGD .9 and decaying lr .01 with beta .0002,exponent .75; no paper schedule retained |
| Schedule | Caller epochs2,bagsPerStep; full cap, one shuffled target pass/epoch | Plan §7.7; unspecified batch size controlled by caller, not paper target64/source32 |
| Selection | The search's centroid primary rule and EnsV secondary rule, neither reading a target label; the centroid rule's source pass runs in `search`, outside this trainer, so the adaptation still reads no source data; the run keeps its last epoch (no early stopping) and keeps no per-run secondary state | Plan v59 §§4.5, 7.3, 7.7 (row corrected 2026-11-04: it described independent primary/secondary epoch snapshots; 2026-11-06: v58 replaced the target-labelled and entropy rules and `secondary_checkpoint`) |
| Search ranges | Shared lr2/wd2; loss weights generic LOSS_WEIGHT, pseudo-label confidence generic CONFIDENCE_THRESHOLD | Plan v42 §7.7 CLUST row and generic rule |
| EMA search range | Own axis `search.EMA_MOMENTUM_COMPLEMENT`: 1 − rho log-uniform in [1e-3, 1e-1], i.e. rho in [0.9, 0.999], includes the paper's .999; admitted by `search.register_phase_two` for CLUST only | Plan v42 §7.7 «el momento rho de la EMA del maestro se busca en su propio eje, entre 0,9 y 0,999, con 1 − rho en escala logarítmica» (task log L59); replaces v40's reuse of CONFIDENCE_THRESHOLD [.5,.95] |
| State and history | Shared store; dashboard history keys; explicit seed,fold,transfer | Plan §7.3/S5 |
| Adapter code version | Additional sha256 of CLUST .py files in clust.save/load; shared store alone hashes only common code | unspecified: extra integrity check prevents accepting a changed adapter; separate roots for the two rules' models retain the store no-overwrite invariant |
| Seed | Caller seed; smoke 17, local torch generator | unspecified: paper reports three trials without seed values |

## Verification

Working directory for all commands: `/Users/diego/Proyectos/papersmith-ai/implementations/Domain_Adaptation-worktrees/clust`, detached worktree of e4bf532. Interpreter `.venv/bin/python`; `PYTHONPATH=$PWD/src` is required because its editable install otherwise points at the main checkout. Scratch retained at `/tmp/clust-replication.I2OA6l`.

- `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_balladeer_comparator_clust.py -q` — exit 0, **21 passed** (2.59 s).
- `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q -k balladeer` — exit 0, **271 passed, 2 skipped, 821 deselected**, 4 Optuna experimental warnings (29.16 s). No sibling comparator files are present in this isolated baseline worktree; this is not a claim about their concurrent branches.
- `PYTHONPATH=$PWD/src .venv/bin/python -m MIL_CREDA_EEG.comparators.clust.smoke` — exit 0, CPU, seed17, ragged windows `(2/3,29/14,512)`, shared supervised phase then three target-only epochs. Printed phase1 loss `[0.03769494443617027]`, phase2 losses `[0.0002472325979681305, -0.000496143341823255, -0.0001454566571337108]`, selected epoch0. Spatial depthwise initialization retained on 29→14 bridge. These are synthetic execution diagnostics, not paper results or evidence of convergence.
- AST provenance measurement: 17 declared invariant IDs; zero IDs missing their named test. Parametrization expands them to 21 tests.

Invariants cover closed-form transport, nearest-prototype softmax axis, confident centroids, missing classes, finite gradients through every target parameter, source/attention/head tensors and buffers bit-identical, no source forward call, unreadable target-training labels, teacher EMA, diagnostic independence from auxiliaries, independent primary/secondary epoch snapshots, explicit search registration with the EMA momentum on its own axis (v42) and rho = 1 − drawn complement, same-rotation/seed checkpoint guard, and a real shared-store save/load round trip for both selection rules including adapter-version refusal and no overwrite. Tests also execute shared supervised phase1 in the synthetic smoke.

For integration: call `clust.register()` explicitly (no import side effects), use the shared phase1 common checkpoint and the shared search engine. Use `clust.save/load` for final models, not bare `search.save_phase_two`: the latter does not record this adapter's extra code digest. A model of the EnsV rule is another refit, saved under a separate root. Auxiliary teacher/bank/alpha tensors round-trip, but optimizer state and mid-epoch resumption are not implemented or claimed.

Not tested: paper benchmarks/numbers, real BALLADEER data, full 20-trial CLUST search, GPU numerics, statistical efficacy, theoretical inequalities, absent lineage papers. Generic implementation probe reports `nextStep=convert` from existing NumPy tests; no existing backend/module is changed by this scoped comparator unit. The controlled deviations and unspecified choices above remain operator-review items.

## Later campaign parallelism (no launch)

Read `skills/remote-parallelism/SKILL.md`. Running its CLI with `assess --axis fold` exits 2, with exact refusal:

> measured inputs are required, never guessed; missing: --units or --units-file, --unit-minutes, --owners, --sessions-per-account, --session-limit-hours, --quota-hours, --setup-minutes

Therefore no speedup, makespan, quota or split verdict is asserted. The CPU synthetic smoke is not a remote full-unit timing measurement. `checklist` executed successfully and is retained in scratch; outstanding launch-stage checklist: declare fold/seed unit IDs (never split by comparator), derive each campaign seed as `int(sha256(unit_id).hexdigest()[:8],16) % 2**31` before BOTH phase1 and phase2 so checkpoints match, pin commit/requirements/accelerator, identical per-shard Python/torch/CUDA/cuDNN/accelerator/commit/requirements/data/preparation stamps, deterministic flags and CUBLAS workspace, identical official dataset, per-owner rehearsal, per-environment timing, one account/person/service and one service/campaign. No remote job was generated or launched.

## Plan v42 update — 2026-11-03

Run in `implementations/Domain_Adaptation-worktrees/v42cmp` (detached at 7db39f8). The teacher EMA momentum moved from the generic confidence axis to its own axis (rows above); the confidence threshold keeps the generic rule. Invariant witnesses added or renamed: `clust_registration_is_explicit_and_draws_the_ema_on_its_own_axis`, `clust_teacher_momentum_is_one_minus_the_drawn_complement`. The smoke (1 − rho = .001) prints the same losses as before, `[0.0002472325979681305, -0.000496143341823255, -0.0001454566571337108]`, because rho = .999 is unchanged.
