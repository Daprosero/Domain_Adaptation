# CLUDA: controlled bag-level variant, not a reproduction of published results

## Code search and route (2026-11-05 UTC)

| Place searched | Observed answer |
| --- | --- |
| Ingested `guidance/related-work-evidence/xu2024cluda/xu2024cluda.md`, entire available extraction, pages 1–9 and page-10 marker | Contrary to the delegation's paywall premise, the method text, Eqs. (1)–(8), Algorithm 1 and Table 1 are available locally. Data availability says authors lack permission to share data; no code-availability link found in this extraction. |
| Web search: exact title + github code; Xu CLUDA 125452 research cuhk code; exact title + code authors | Resolved official `https://github.com/xuchengjian632/CLUDA`, not the ICLR 2023 time-series namesake. |
| Author/institution publication record: `https://research.cuhk.edu.hk/en/publications/unsupervised-multi-source-domain-adaptation-via-contrastive-learn/` | Fetched abstract identifies conditional source-target and inter-source contrastive learning; DOI and journal 261, article 125452, 1 February 2025. |
| Venue: `https://www.sciencedirect.com/science/article/pii/S0957417424023194` | Raw fetch returned ScienceDirect HTML, not readable full paper; local ingested method was used instead. |
| Official repository README, complete `main.py`, `models.py`, pinned tracked-file enumeration | Commit `9c655644e9f15eadddb51de236a1afecd5944468`. Tracked tree contains README, three Python files and a figure, no license or test/config files. Experiment defaults read from main.py, architecture defaults from models.py. No upstream tests found in that enumeration; upstream training was not run. |

**Route:** independent implementation of the operator-approved controlled variant.
No upstream source copied into this repository: only `third_party/cluda/PIN.json`.
An unlicensed public repository is not a reusable dependency. Read-only audit clone
and logs remain at `/var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.zyYss9s8Sn/`;
nothing there was deleted. No API/model calls are part of the implementation.

Bound protocol: `experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md`
§§4.5, 7.1–7.5, 7.7. Operator decisions: task log L59, L62, L70–L71.
This file records correspondence, not a claim of original-result fidelity.

## Plan, equation analysis and boundaries

- Existing `CommonMILModel` supplies EEGNet, common attention and linear head.
  Lists of bags hold `(windows, source/target channels, 512)` tensors; embeddings
  and attention pooling yield `(source bags, 256)` and `(target bags, 256)`.
- `model.CLUDAModel` adds a fresh training-only projector, mapping `(B,256)` to
  `(B,256)` via width 1024. It inherits inference unchanged.
- `training.adaptation_step`: source EEGNet/attention/head in no-grad/eval;
  target EEGNet trains, common attention/head frozen but differentiable in its
  input. Head logits `(T,2)` produce detached argmax pseudo-labels `(T,)`.
  Project target and source **separately**, in that order. Source embeddings are
  detached from the source encoder but not from the trainable projector.
- Eq. (3) -> `losses.cross_domain_contrastive`: projected `(S,256),(T,256)` plus
  labels `(S,),(T,)` -> scalar and missing-positive count. Each source bag is a
  singleton subject-domain. Cosine logits `(S,T)` are normalized over TARGET
  candidates only. Each anchor contributes `-1/2 log(positive probability)`;
  average over sampled source domains. No target-target or source-source pairs.
- Eq. (2) -> `F.cross_entropy` in `adaptation_step`: source CE is diagnostic and
  constant with respect to optimized parameters. Eq. (5) -> `source_ce +
  contrastiveWeight * contrastive`. Eq. (4) is deliberately not computed.
  Eqs. (6)–(7) schedules are deliberately not used. Eqs. (1),(8) are replaced by
  the common protocol's feature extractor/preparation, not rewritten here.
- `training.train_phase_two` reuses shared checkpoint, bridge, schedule guards,
  source `ClassQueues`, target shuffled batches, evaluation, validation records,
  label-free keeper, max-norm and store. It never trains phase one itself.
  `load_phase_two` uses `model_store.load` and verifies comparator source digest.

Lineage gaps: the method cites Song et al. (2022) EEG Conformer, Lee et al.
(2022) and Shen et al. (2022) for its feature/loss lineage. A guidance-name search
found `song2024macs` and `lee2025lbm`, not those cited works; they were not ingested
or read here. We do not infer additional equations from them. The common EEGNet
replaces Conformer; Eq. (3) and audited projector suffice for this bounded variant.

## Recorded parent rulings during this implementation

1. Follow **paper Eq. (3)**, source-anchor/target-candidate only, not official
   `models.py:230–234,264–294` concatenated symmetric SupCon. Keep the projector
   from §3.2.4 / Algorithm 1 line 3; initialize each phase-2 run independently.
2. Do **not compute** source-source loss. L62's premise that a frozen source
   encoder makes it inert needs qualification: the original term would still
   train a shared projector. Dropping that source-only projector objective is a
   declared deviation, not a claim that its gradient vanishes mathematically.
3. Floor relative positive probability at `1e-6`, absent-positive anchors included
   in the source-domain mean with constant `-0.5*log(1e-6)` and zero gradient.
   Official `models.py:287` instead floors the unnormalized numerator: related
   numerical precedent, NOT an identical operation. Common pooled balanced
   accuracy selects models, never this loss or its constant offset.
4. Project target separately, then pooled sampled source bags. Official
   `models.py:224,230` forwards target and each source **separately**; :231 only
   concatenates already projected features. Source domains sharing BN is the
   remaining singleton-bag deviation; target statistics remain separate.
5. Use existing class-balanced `ClassQueues.draw(bagsPerStep // classes)` just
   like UDA-DDA; missing source classes refuse before model construction.
   Each source forward thus has >=2 bags under the shared schedule guard.
   A target singleton remainder skips its entire update: no projector call,
   optimizer update, or source draw; count it explicitly. Target training length
   <2 refuses. No target bags are merged, padded or pseudo-label-filtered.
6. The searched ST weight is **constant**, without the native ramp. v54 §7.7
   generic weight rule and its single CLUDA weight axis, the repository CASUAL
   controlled-variant precedent, and the common fixed phase-2 schedule motivate
   this ruling. Native paper/code ramp departure is stated below.

## Correspondence and every chosen parameter/architecture

Verdicts compare the controlled implementation to its stated source. `unspecified`
means the paper fixes no such choice; it remains so even if code/protocol provides
a usable default. There are **8 unspecified rows**.

| Element | Paper / official pinned code | Our implementation / source of choice | Verdict |
| --- | --- | --- | --- |
| Feature extractor, Eq. (1), Table 1 | CNN + Transformer; models.py:38–165: 40 tokens, six layers, ten heads, expansion four, dropout .5, flatten 1480->256 for PhysioNet | Inherited EEGNet-8,2 (8 temporal filters, spatial multiplier 2, 16 separable filters, kernels 64/16, pooling 4/8, BN/ELU, dropout .25, spatial max norm 1), 256-dimensional bag representation, v54 §7.3; no Conformer implementation | deviation: common capacity |
| Classifier, §3.2.3 | Paper two FC layers; code 256->32->4, ELU/dropout .3, models.py:183–200 | Frozen common 256->2 linear head, bag softmax, max norm .25 inherited from phase one, v54 §7.3 | deviation: common capacity |
| MIL attention, bag labels | Native method trial labels, no MIL | Inherited single learned tanh/softmax attention, gamma zero, no consensus/local term; only bag labels, v54 §§7.2–7.4 | deviation: bag adaptation |
| Source-domain unit | §3.1: each subject's multiple trials | Each source subject represented by its bag, v54 §7.4 / L62; each anchor equally weighted | deviation: singleton domains |
| Projector architecture | §3.2.4, Table 1; models.py:167–181: bias-free 256->1024, affine BN, ReLU, nonaffine BN, bias-free 1024->256, nonaffine BN | `model.CLUDAModel.projector`, same widths (input follows bag width), BN placement/affinity, no dropout | match to audited code; extra BN differs from paper's simplified table |
| Projector lifecycle | Trained together with extractor; discarded at test, Fig. 2 / README:18 | Fresh phase-2 auxiliary, trained only by ST; inherited inference never calls it, saved for reproducibility | deviation: two-phase common protocol and parent ruling |
| Projector initialization | No initialization rule in method text | PyTorch defaults, seeded per run; fresh module each time | unspecified |
| Projector BN epsilon/momentum | Code uses framework defaults; paper silent | PyTorch eps=1e-5, momentum=.1, track running statistics; double precision common model | unspecified |
| Separate projector BN statistics | Code target :224, each source :230 | Target-only forward then one pooled source-only forward; singleton source domains cannot carry BN, parent ruling | deviation: source-domain BN pooling |
| Cosine normalization | Eq. (3) uses embedding dot product, normalization implicit; code models.py:280 L2-normalizes | `F.normalize` before dot product in `cross_domain_contrastive` | match to code; paper implicit |
| Zero-vector norm safeguard | Paper/code no zero-vector safeguard | F.normalize eps=1e-12, producing finite zero embeddings | unspecified |
| Eq. (3) candidate axis | Paper source-anchor to target candidates; code concatenates source+target and symmetric within/cross positives, models.py:231–234,264–294 | Cross-domain only `(S,T)`; no target-target pairs; parent ruling follows paper and v54 §7.4 | match to paper, deviation from code |
| Eq. (3) reduction | Paper `-1/2 sum_l`; code mean symmetric anchors without 1/2 | Singleton source domains, retain factor 1/2 then mean over domains | match to paper specialized to bags; deviation from code |
| No-positive finite extension | Exact Eq. (3) infinite if class absent; code :287 clamps unnormalized numerator 1e-6 | Relative mass floor 1e-6; constant absent-class anchor, zero gradient, counted per step; parent ruling | unspecified |
| Numerical evaluation | Paper exponential ratio; code direct exp may overflow | Stable logsumexp and safe all-masked handling in losses.py | match mathematically above floor |
| Pseudo-label | Algorithm 1; models.py:224–228 argmax softmax, no threshold | Frozen common-head detached logits argmax; all target bags eligible, training labels unread | match |
| Source-source loss, Eq. (4) | Code :241–248 trains encoder AND projector across source pairs | NOT computed; history source_source=0 diagnostic placeholder, not a measured original loss; parent ruling | deviation: no source-only projector training |
| Objective Eq. (5) | Source CE + alpha ST + gamma SS | Diagnostic frozen source CE + constant searched ST weight; no SS, training.py | deviation: controlled phase 2 |
| Alpha/gamma Eqs. (6)–(7) | Alpha logistic `2/(1+exp(-10*t/D))-1`; paper gamma=alpha/10, code main.py:88–91 SS=alpha/2 | No ramps; constant contrastiveWeight, SS moot; parent ruling, v54 §7.7 | deviation from paper and code |
| ST weight range | Paper dynamic alpha; no search range | `contrastiveWeight=search.LOSS_WEIGHT`, [1e-3,10] log, v54 §7.7 | deviation: approved common search |
| Default ST weight | No fixed paper default for this variant | 1 for standalone smoke; searches supply their value | unspecified |
| ST temperature | Code models.py:210 and main.py:191: .07; paper temperature symbol | Standalone .07; `TEMPERATURE=FloatDistribution(.01,1,log=True)`, v54 §7.7 | match default; deviation search range explicitly protocol assumption |
| SS temperature | Code .03; paper symbol tau shared in Eqs. (3),(4) | Not used or searched, SS dropped | deviation, moot |
| Adam optimizer | §4.3 / main.py:62: Adam, beta1=.5, beta2=.999 | Same betas, target trunk + projector only | match betas, deviation trainable set |
| Learning rate / weight decay ranges | Paper lr=.0002; no variant search | Shared search lr2 [1e-5,1e-2] log, wd2 [1e-7,1e-2] log, v54 §7.7 | deviation: approved common search |
| Standalone lr2/wd2 | No paper values for controlled variant | .001 / 0; supplied to standalone smoke only, same convention as other comparators | unspecified |
| Source batch selection | Code one shuffled trial batch from every source | Shared class-balanced cyclic source queues; count classes * floor(bagsPerStep/classes), parent ruling, UDA-DDA template | unspecified |
| Batch size / epochs | Paper batch36; main.py defaults batch20, epoch200; iterations ceil(epoch*90/batch), :274–283 | Shared checked schedule, bagsPerStep>=2; epochs2 exact, no early stopping; v54 §§4.5,7.7 | deviation: common budget |
| Target singleton handling | Eq. (3) one candidate has zero gradient; code projector BN cannot train one sample | Refuse target set <2; otherwise skip singleton remainder update and record count plus contributing bag count, parent ruling | unspecified |
| Source freeze / target bridge | Native one shared trainable extractor | Shared bridge copies compatible source weights; keeps new spatial convolution for 29 vs14 channels; frozen source BN, attention, head; target trunk trainable, v54 §7.3 | deviation: common two-phase protocol |
| Checkpoint seed | Native setup_seed(20), main.py:25 | Verify record seed `search.PHASE_ONE_SEED=1` even when trial seed=0; transfer, rotation, partition, channels checked; L70–L71 / v54 §4.5 | deviation: common checkpoint |
| Training/evaluation split | main.py:167–168 target_data=test_data; LOSO PhysioNet 103 subjects | Adapt target TRAIN only; no test argument, disjoint subject role guards, v54 §4.5 / L59 | deviation: removes transductive test exposure |
| Selection | main.py:112–114 maximizes test accuracy over iterations | Record all inner validation epochs, pooled balanced accuracy chosen by shared search; refit has no validation; optional shared label-free secondary; v54 §4.5 | deviation: removes test-label selection |
| Preprocessing, Eq. (8) | §4.2 filtering/standardization; main.py norm_type default none, utils loader external files | Existing prepared bag data only, no comparator preprocessing, v54 common input | deviation: common data |
| Persistence | No released checkpoint format for our variant | Shared save/load manifest, partition/code checks, own source digest; include projector BN buffers; no bridge when loading trained target | deviation: repository contract |
| Smoke seed and shapes | Not a paper experiment | Unit id `CLUDA:synthetic:rotation0`, SHA256-derived seed 639203496; (8 bags,2 windows,29/14 channels,512), 3/5 epochs, schedule8 | deviation: explicitly synthetic execution only |

**Degenerate cases:** a single target candidate, or a target batch with only one
pseudo-class, gives zero contrastive gradient for every anchor (present positives
have probability one, absent positives are constant). No confidence workaround is
introduced. Target singletons skip updates explicitly as above. For other batches,
`step_anchors_without_positive`, `step_anchors`, `step_target_pseudo_classes`,
`skipped_singleton_target_steps`, `target_bags_contributing` expose the issue.
Loss histories average only the executed updates; missing-anchor counts sum over
updates. The finite floor also saturates extremely low positive mass.

## Registration the parent must add (not performed here)

Phase-1 group **common**, strategy name **CLUDA**; no new phase-1 group.

```python
# model_store.STRATEGIES: add "CLUDA" (parent removes HASD separately).
# search.PHASE_ONE_GROUP_OF (or derive by the existing comprehension):
PHASE_ONE_GROUP_OF["CLUDA"] = "common"
# search.OWN_AXES: admit the sole non-generic temperature axis:
OWN_AXES["CLUDA"] = (FloatDistribution(.01, 1., log=True),)
# campaign.register imports / registrars map:
from MIL_CREDA_EEG.comparators.cluda import training as cluda
# Add "CLUDA": cluda.register to the comparator registration map.
# cluda.register() supplies:
search.register_phase_two("CLUDA", {
    "contrastiveWeight": search.LOSS_WEIGHT,
    "temperature": cluda.TEMPERATURE,
}, cluda.train_phase_two)
```

Shared lr2/wd2 are supplied by `search.phase_two_space`, not extra axes.
`remote_units` derives groups/strategies from shared registries; parent owns its
remaining declarations/configs. Existing `mechanisms.load_phase_two` can load the
common inference state and return projector keys as unused auxiliaries. For full
auxiliary restoration AND comparator digest enforcement, use this package's
`training.load_phase_two(path, partition_path=...)`, returning `(model, meta)`;
do not silently re-bridge a trained target. No campaign or remote files edited.

## Verification and bounds (2026-11-05)

Test-first RED, before package existed:
`.venv/bin/python -m pytest -q tests/test_balladeer_comparator_cluda.py`:
exit 2, collection `ModuleNotFoundError: No module named 'MIL_CREDA_EEG.comparators.cluda'`.
Initial implementation: 11 passed in 4.85s. After separate-BN/source-queue/singleton
rulings: same focused command **12 passed in 4.85s**, exit 0; final rerun
**12 passed in 4.74s**, exit 0. `git diff --check` on the allowed surfaces exited 0.

Tests cover Eq. (3)'s closed form, factor 1/2/domain mean, scale invariance,
finite/nonzero gradients, no-positive finite zero gradients, degenerate classes,
projector shape/all trainable parameters, inference never calling projector,
minimum BN size, source/attention/head parameter+buffer freezing, compatible
bridge, no target-label reads, exact reproducibility, PHASE_ONE_SEED metadata,
invalid params/roles, final-epoch retention, validation per epoch and refit
non-evaluation, registration with temporary parent-owned registries, shared-store
round-trip and own-code digest rejection, separate projector forwards, singleton
skip/count and missing-source-class refusal before model construction, and every
provenance invariant having a test.

Smoke command:
`PYTHONPATH=src .venv/bin/python -m MIL_CREDA_EEG.comparators.cluda.smoke`
(exit 0). Output after final separate-BN ruling:

```text
seeded synthetic EEG CPU only; NOT paper numbers; seed=639203496
source=(8 bags,2 windows,29 channels,512); target=(8 bags,2 windows,14 channels,512)
phase1_loss [0.037831, 0.036197, 0.036509]
phase2_loss [0.866523, 0.870275, 0.708391, 0.947383, 0.877775]
contrastive [0.17702, 0.180771, 0.018888, 0.257879, 0.188272]
step_anchors_without_positive [[0], [0], [0], [0], [0]]
step_target_pseudo_classes [[2], [2], [2], [2], [2]]
```

Existing `MIL-CREDA/Results/Toyset/full/{lunas,rollo,toroide}.npz` were inspected:
they contain old benchmark histories/latent arrays such as `(81,64)`, not EEG
window bags. Thus a seeded synthetic EEG-shape fixture was used explicitly,
not a claim to have run the dataset-curator toy or the approved hypothesis PoC.
Loss moved; no monotone-convergence or accuracy claim follows.

Out-of-scope diagnostics: proposal-implementation probe exited 0, `nextStep=convert`
(global inferred mixed NumPy/Torch imports, not a CLUDA failure); read its
`next-step-convert.md` reference, changed no global backend declaration.
Remote-parallelism `assess --units 4 --axis fold` exited 2:
`measured inputs are required, never guessed; missing: --unit-minutes, --owners,
--sessions-per-account, --session-limit-hours, --quota-hours, --setup-minutes`.
No verdict or launch authorization. Later stretch must measure those inputs;
checklist: split fold/seed/subject axis, never arm; unit-id SHA256 seeds; same
commit/requirements/accelerator/data prep; identical environment stamps; deterministic
flags; per-owner rehearsals and one account per person/service; no cross-host timing
pooling. This comparator smoke seed is already unit-id-derived.

Not tested: upstream execution, full repository suite (not requested), real EEG
training, held-out metrics, paper performance, GPU/remote runs, multi-account
readiness, or parent registration end-to-end. No commits, launches, upstream
patches, or edits to `comparators/adaptmil` were made by this stretch.
