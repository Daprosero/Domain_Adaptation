# CCA-LSC: controlled BALLADEER variant

## Code search and route (2026-11-02 UTC)

**Route: paper-based reimplementation, not adopted code.** Official repository:
https://github.com/ysxcj-hub/CCA-LSC, observed HEAD
`4c78f079026cfce7b55b0bf36d45251f3d134ed7`. No upstream implementation bytes
are copied into this target; `third_party/cca_lsc/` is not populated.

Checked 2026-11-03 against plan v42 §7.1: its CCA-LSC row and canonical table
record the same route («Reimplementado desde el artículo»; repository without a
license, read only to audit, at commit
`4c78f079026cfce7b55b0bf36d45251f3d134ed7`), so this route row stands. The
modules of this package were not changed by v42 and stay bound to v40.

| Place searched | Answer |
| --- | --- |
| Ingested paper abstract/code statement and arXiv abstract https://arxiv.org/abs/2412.20337 | Authors link ysxcj-hub/CCA-LSC. |
| Web search `"CCA-LSC" github code` | Same official repository; no alternative authorized distribution found in returned results. |
| Web search `"CCA-LSC" ICPR 2024 authors` | Sun, Wu, Zhan, Ji; arXiv and author-index results point to the same work. |
| Author repository page https://github.com/ysxcj-hub/CCA-LSC and pinned README | Functional image training entrypoint; OfficeHome command has 20 epochs, DomainNet command 30. |
| Venue bibliography https://dblp.org/rec/conf/icpr/SunWZJ24 | Bot challenge, no code/license evidence obtained. |
| GitHub `/repos/ysxcj-hub/CCA-LSC/license` | HTTP body status 404, `Not Found`. |
| GitHub recursive tree at pinned HEAD (complete listing) | No LICENSE/COPYING or other license file; entrypoint, data lists, datasets, setup.py, sampling.py, utils.py, requirements.txt and README present. |
| Pinned README and setup.py | README contains install/run commands, not a license grant. setup.py has `License :: OSI Approved :: MIT License`, but describes **TLlib**, author THUML and URL thuml/Transfer-Learning-Library; it does not grant reuse of the authors' CCA-LSC implementation. GitHub repo metadata has `license: null`. |

License status is **unresolved**, not MIT asserted for this method. The missing
reuse grant was reported to the orchestrator before implementation. Per the
replicator contract, uncertain/unlicensed source is not vendored; the fallback
below is independently implemented from the ingested equations. Upstream tests
and configs were not executed/audited as an adopted implementation; that route
was not taken. The README discrepancy is recorded, not silently resolved.

Search evidence retained outside the worktree at
`/var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.3lVxbpl52g/`
(`upstream-tree.json`, `repo.json`, `README.md`, `setup.py`, `paper.html`,
`paper-text.txt`, `probe.json`, `parallelism.json`, test log). Scratch is not
removed and is not a runtime dependency.

## Sources and scope

Primary source (read whole, 12 displays):
`guidance/experimental-papers/cca-lsc-2412.20337/cca-lsc-2412.20337.md`
in the forge. Hyperparameters were cross-checked against
https://arxiv.org/html/2412.20337v1, Sections 3.3, 3.4, 4.1, 4.5, 4.6 and
Tables 5/6. The HTML Table 6 has lambda rows 1/3/5 and mu columns 0.4/0.6/0.8;
these are sensitivity settings, not new defaults or reproduced results.
The ingested Table 6 image could not be rendered by this session's image reader;
the HTML table was read instead. Garbled result tables are not used as config.

Binding experiment source:
`experiments/experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md`,
Sections 4.5, 7.1–7.4 and 7.7. Task T6, S1–S5 in
`odd/tasks/balladeer-implementacion-experimental.md` (read through `## Log`).
This is explicitly a **controlled variant**, not reproduction of the paper's
accuracy, schedule, image backbone or training freedoms.

Lineage gaps: the HTML bibliography identifies [5] Ganin et al., *Domain-adversarial
training of neural networks*; [41] Xie et al., *Learning semantic representations
for unsupervised domain adaptation*; [43] Hjelm et al., *Learning deep representations
by mutual information estimation and maximization*. A title search over `guidance/**/*.md`
returned no ingested matches. These lineage texts were not available to read here.
The unspecified centroid decay is not represented as if read from [41].

## Module and shape map

Existing code was read before writing: `eegnet.py`, `mil_wrapper.py`,
`mil_creda_model.py`, `two_phase.py`, `bag_data.py`, `assignment.py`,
`controls.py`, `search.py`, `model_store.py`. None was edited.

- `losses.py`: calibration, weighted centroid memory, contrastive ratios,
  adversarial saddle objective. No EEG encoder or aggregator implementation.
- `training.py`: phase-2 adapter, explicit search registration, common-checkpoint
  hook. Reuses `two_phase.bridge`, `ClassQueues`, `shuffled_batches`, `evaluate`,
  `balanced_accuracy`, `label_free_score`, `check_hyper`; reuses `search.check_schedule`
  and `model_store.build_model`. Assignment, bag loading, controls, search budget
  and persistence remain owned by their existing modules.
- `smoke.py`: seeded synthetic CPU check, shared source trainer then adapter.
- `__init__.py`: exports only; importing does **not** alter global search registries.
- Tests import the package by its full path. No parent `comparators/__init__.py`.

Shapes: each input bag `(m, C_domain, 512)`; shared EEGNetTrunk gives `(m,256)`;
CommonMILModel's single frozen attention gives `(B,256)` bag representations;
shared frozen EEGNetHead gives `(B,2)` probabilities. Source/target channel counts
remain 29/14 (or reverse), never reduced to their intersection. Pseudo-labels and
confidence are `(B,)`; two centroid memories are `(2,256)`; domain logits `(B,1)`;
each objective is scalar. Only bag labels enter the objective, never window labels.

Common phase 1 is external/shared, not repeated for every comparator. The search
hook requires its matching transfer/rotation/seed/partition checkpoint, rebuilds
via the existing store, copies compatible source layers and freezes the source,
attention and predictive head via the shared bridge. The source trunk remains
in eval, including BatchNorm statistics. The discriminator is an auxiliary, not
an inference head. Only the target trunk and discriminator parameters are optimized.

## Equation analysis and correspondence

Equation numbers 4–6/12 follow their order/HTML where ingestion omitted tags.
All feature equations operate on bag representations, not EEG windows.

| Paper | Function or treatment | Assumptions and consequences |
| --- | --- | --- |
| Eq. 1, log D domain game | `losses.domain_adversarial` | Source domain=0, target=1, sum of domain-mean BCE losses. D minimizes BCE; target features receive its negative gradient through GRL. Frozen source features detached. This is the saddle sign, not cooperative D/G minimization. |
| Eq. 2, original centroid attraction | Numerator of `losses.centroid_alignment` | This is the precursor replaced by Eq. 3, not a second loss added to the objective. |
| Eq. 3, weighted centroid attraction/repulsion | `weighted_centroids`, `CentroidMemory`, `centroid_alignment` | Weighted sum divided by total confidence per class; Euclidean distances; ratio of sums, not squared distance or average of ratios. Detached history with current-step gradient. |
| Eq. 4, discriminative feature alignment | `discriminative_alignment` | Mean over the count of same/different cross-domain pairs, not over total weight. Geometric mean of raw confidences. |
| Eq. 5, label-shift ratio | `label_shift` | Target estimated prior divided by original source-training bag prior, not balanced minibatch prior; positive source class mass required. |
| Eq. 6, bounded calibration matrix | `calibration_weights` | Exactly `1/(hm+exp(-sqrt(shift)))`, no normalization. |
| Eq. 7, calibrated argmax | `calibrated_pseudo` | Multiply original probability by class weights; deterministic first-index argmax ties. No calibrated softmax. |
| Eq. 8, confidence | `calibrated_pseudo` | Gather **uncalibrated** probability at the calibrated label. This can be less than the raw maximum. No gradient through pseudo-labels/confidence. |
| Eq. 9, uncalibrated warmup objective | `training.adaptation_step` with all-one weights | Source CE + lambda DSM + mu DFA + gamma domain saddle surrogate. Source CE is computed but carries no gradient under the protocol freeze. |
| Eq. 10, calibrated objective | Same function after one full target-training prior estimate | Same terms with calibrated pseudo-labels; fixed prior after warmup. |
| Eq. 11, target-error bound | Analysis only, no training operation | Not an empirical quantity reproduced here. |
| Eq. 12, triangle-inequality bound | Analysis only, no training operation | No synthetic numeric implementation presented as a proof. |

## Hyperparameter and architecture provenance

Each `unspecified` row is one unresolved paper choice with the concrete policy
used here. `deviation` is intentional protocol standardization; `match` names
an equation/default, not a claim about published performance.

| Choice | Paper / protocol origin | Implementation and verdict |
| --- | --- | --- |
| Encoder: EEGNet-8,2, 256 features, 512-sample windows | Paper 4.1 uses ResNet50 + 256 bottleneck; plan 7.3 fixes EEGNet | Reuses EEGNetTrunk via CommonMILModel; deviation: no ResNet/bottleneck. |
| EEGNet filters 8,2,16; kernels 64/16; pools 4/8; dropout .25; BN/ELU/no conv bias; spatial/head max norms 1/.25 | Plan 7.3; existing `eegnet.py` | Unmodified shared implementation, no architecture search; deviation from image paper. |
| Shared attention width 128, gamma=0, temperature=1; float32 trunk, float64 attention/head | Existing common wrapper/model; plan 7.2 | Reuse; deviation from image sample prediction. Existing shared defaults, no comparator-owned capacity. |
| Two classes, bag-level pseudo-labels/centroids/pairs | Plan 7.2–7.4 | All objectives use `(B,256)`, head once per bag; deviation from image classification. |
| Independent source/target trunks; freeze source/attention/head; copy all compatible layers, keep spatial init for 29↔14 | Plan 7.3 | Shared bridge; deviation from jointly trained image backbone/classifier. |
| Calibration hm=1.5 | Paper 3.3/4.5 and HTML Table 5 | `DEFAULTS`; match. Search axis under plan generic positive loss-weight rule. |
| Lambda=3, mu=.6, gamma=1 | Paper 4.1; HTML Table 6 confirms sensitivity context | `DEFAULTS` centroid/discriminative/adversarial weights; match as defaults, searchable under plan 7.7. |
| Initial uncalibrated stage 3 epochs | Paper 3.4/4.1 | `WARMUP_EPOCHS=3` within phase 2 **after** the common supervised phase 1; match to warmup length, deviation in overall schedule. Phase-2 budget must exceed 3; otherwise refuse. |
| Prior-estimation confidence strictly >.5 | Paper 3.4/4.6 | `estimate_prior`; match. `priorThreshold` search .5–.95 under plan 7.7. |
| Source prior from training bag-label counts | Paper 3.3 derives PS from source labels | Original unresampled training counts; match. |
| Target prior once after warmup from high-confidence argmax counts | Paper 3.4/4.6 | Full target-training bags via shared evaluate, no target labels/validation/test; match. No continual prior refit. |
| Class-balanced source sampling, no target class balancing | Paper 3.2/4.4, plan 7.4 | Reuses ClassQueues equal class draws, target shuffled pass; match. Odd batch sizes floor source per-class count to equal sizes as shared trainer does. |
| SGD momentum .9; classifier lr .005 OfficeHome/.01 DomainNet; DANN-style adjustment | Paper 4.1 | Not used unchanged: target adapter uses Adam with constant caller lr2/wd2; deviation for common training conventions. |
| 20 total image epochs; first 3 warmup; batch size 50; seed 100 | Paper 4.1 | Caller phase-2 cap/bagsPerStep/seed, not these image settings; deviation. Pinned upstream README lists DomainNet 30, not paper's 20; not adopted. |
| Search lr2 [1e-5,1e-2], wd2 [1e-7,1e-2] log; three loss weights [1e-3,10] log | Plan 7.7 | Shared search + explicit `register()` extras; match to plan, not a reproduced paper HPO. |
| Search calibrationHm [1e-3,10] log | Plan 7.7 calibration-parameter family and generic comparator rule | Positive calibration parameter uses generic loss-weight distribution; deviation from fixed paper hm. |
| Search 20 trials, GPSampler, four rotations, no pruning, distinct search/report seeds, mean-minus-SD | Plan 4.5/7.7; `search.py` | Reuse existing engine; not run here. No second sampler or split. |
| Primary selection target-validation BA, secondary H(mean p)-mean H(p), first best tie, full epoch cap | Plan 4.5/7.3 | Shared metrics/evaluate; best target/aux snapshot restored; deviation from paper evaluation. Warmup epochs are eligible for selection. |
| Centroid EMA decay .9, first observation initializes | unspecified: paper calls it moving average without decay/initialization formula | `CENTROID_DECAY`; first seen class=current, subsequent=.9 history+.1 current; detached memory. |
| Confidence-normalized centroid denominator sum(w) | unspecified: paper says weighted centroid without exact denominator | Chosen normalized weighted mean, invariant to class weight scaling. |
| Missing centroid class retained; unseen class excluded from both sums | unspecified: absent-class minibatch policy | Memory presence masks prevent zero placeholders from acting as observed centroids. |
| Epsilon 1e-8 on centroid mass and ratio denominators | unspecified: singularity handling | Finite floor, no numerator epsilon; identity loss remains zero. |
| No same or no different feature pairs yields graph-connected zero | unspecified: Eq. 4 undefined empty means | Explicit skip policy prevents NaN on collapsed pseudo-label batches. |
| No confident target bags yields uniform target prior | unspecified: empty prior estimate | Defined neutral fallback with explicit test; does not read labels. |
| Domain discriminator single affine 256→1, torch default initialization | unspecified: discriminator architecture/initialization not stated | Minimal trainable auxiliary, double precision; not used at inference. |
| Adam target/discriminator optimizer defaults beta=(.9,.999), eps=1e-8, weight decay caller; no scheduler | unspecified: controlled protocol does not fix optimizer internals | Existing shared-training Adam convention; deviation from paper SGD recorded above. |
| Confidence and pseudo-label weights detached | unspecified: paper does not state stop-gradient | Avoids optimizing confidence to escape alignment; gradients flow via target features. |
| Argmax first-index ties | unspecified: tie policy | Torch argmax deterministic tie break. |
| CPU smoke unit/seed=11, two windows per bag, four bags per side, phase1=1/phase2=5 | unspecified: synthetic verification fixture, not an experimental budget | No real data copied; fixed unit id supplies seed. |

## Verification

Run **inside** `implementations/Domain_Adaptation-worktrees/cca_lsc`:

```sh
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_balladeer_comparator_cca_lsc.py -q
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q -k balladeer
PYTHONPATH=$PWD/src .venv/bin/python -m MIL_CREDA_EEG.comparators.cca_lsc.smoke
```

The venv is the operator-provided excluded symlink; its executable resolves to
the target main venv. `training.__file__` was measured and points into this
**worktree**, because PYTHONPATH overrides the main checkout's editable install.

Invariants test: raw-confidence calibration (including a flipped argmax), strict
confidence filtering, weighted mean closed form, EMA initialization/update and
missing-class retention, finite gradients through memory, exact 1/4 centroid
and pair ratios, zero identity distance, GRL sign versus ordinary BCE for both
features and D, no source gradient, degenerate denominators, all target-trunk
and discriminator gradients finite, frozen tensors/BN statistics, both device
directions and equal-channel copy, poisoned target-training labels, one prior
estimate after three epochs, balanced source draw, real search hook execution
from a matching common checkpoint and foreign-rotation refusal, best-epoch
restore against snapshots at the validation seam, bad parameter/split/schedule
refusals, synthetic end-to-end source training plus adaptation. Every declared
invariant has a `test_<id>` in the one allowed test file.

Initial targeted run: 21 passed (exit 0), before adding both montage directions
and equal-channel parametrization. Final targeted rerun: **23 passed in 3.20s**
(exit 0). Final required BALLADEER suite: **273 passed, 2 skipped, 821 deselected**
in 28.10s (exit 0). This is the isolated e4bf532 worktree's suite, not a run over
other replicators' uncollected changes. No sibling comparator files are present
in this worktree. AST/import provenance check: four modules, 15 invariant IDs,
zero missing matching test functions. **Unspecified table rows: 11** (count rows
starting with `|` and containing `| unspecified:`).

CPU smoke (exit 0) printed:

```text
synthetic CPU only; seed=11; source=(4 bags,2 windows,29 channels,512); target=(4 bags,2 windows,14 channels,512)
phase1_loss [0.039896]
phase2_loss [5.721558, 5.666182, 5.620655, 5.626721, 5.640451]
calibrated [False, False, False, True, True] target_prior [1.0, 0.0]
```

The loss moves; it is not monotonic and is not an accuracy reproduction.
Synthetic bags were used because gitignored BALLADEER data are not present in
this isolated worktree (operator declaration). No real-data training, full Optuna
study, GPU work, remote launch, upstream test execution, or paper-number
reproduction was performed. Final models can use existing `search.save_phase_two`
/ `model_store.save`; model state is inference-complete. Resuming an interrupted
optimizer mid-epoch is not implemented by this comparator; auxiliary selected
state is returned in its checkpoint. Existing model-store `CODE_FILES` does not
include comparator bytes: parent must extend digest coverage before relying on
its version check for these new modules; existing modules were not edited here.

## Skill and later-run state

Read-only proposal `probe` on this worktree/MIL-CREDA reported `nextStep=convert`;
its next-step reference concerns the whole proposal package's inferred backend,
not permission to rewrite existing modules here. This scoped comparator already
runs torch through the existing common EEG code. No broad conversion was made.

`parallelism_cli.py assess --axis fold` was invoked without inventing unavailable
measurements; exit 2, `missing-input`:

> measured inputs are required, never guessed; missing: --units or --units-file, --unit-minutes, --owners, --sessions-per-account, --session-limit-hours, --quota-hours, --setup-minutes

No split verdict is available. Before assessment/launch: measure per-unit remote
pilot time and rehearsal setup, record distinct owners and dated capacity/quota
limits, declare fold/seed units (never split by comparator), seed from unit id,
pin identical code/data/partition/hyperparameters/environment and accelerator
stamps, separate unit output paths, rehearse every account, and merge without
changing the statistical unit. Assessment is not launch authorization.
