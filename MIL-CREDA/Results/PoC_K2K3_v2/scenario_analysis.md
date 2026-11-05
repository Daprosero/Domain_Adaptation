## 0. Scenario analysis: what the toy must imitate from BALLADEER EEG

All EEG numbers are read from disk. Sources: `MIL-CREDA/Data/processed/manifest.csv` and the per-record `signals/*.npz` of the implementation repository (non-excluded records only, i.e. no undetermined subjects, no EPOC+), `MIL-CREDA/Results/separability/{summary.json,tables/*.csv}`, and the protocol `experiments/experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md` §2-§6.

The window features for (b) are the log of the channel-mean band power in 0.5-4, 4-8, 8-13, 13-30 and 30-45 Hz (Welch, 2 s segments), with at most 150 windows sampled per bag. That makes them device-agnostic: 5 features per window. Scripts: `eeg.py`, `pct.py`, `toy.py`, `cal.py` (scratch, not committed).

### 0.1 EEG circumstances

**(a) Bag size and subjects.** A bag is one subject × condition (§3.2).

| condition | bags | ADHD / control | windows per bag (median, min-max) |
|---|---|---|---|
| Slackline-L1/CGX | 123 | 80 / 43 | 75 (75-76) |
| Slackline-L6/CGX | 123 | 79 / 44 | 75 (75-76) |
| Slackline-L11/CGX | 123 | 79 / 44 | 75 (75-282) |
| Robots/EMOTIV (EPOC X) | 60 | 29 / 31 | 72 (68-132) |
| Cognifit/EMOTIV (EPOC X) | 61 | 30 / 31 | 529 (406-809) |

Per transfer and rotation (`assignment_sizes.csv`): level arm ≈ 63 subjects per side, of which ≈ 47 are non-test; task arm 33 per side (≈ 25 non-test); device arm 75 / 52.

**(b) Subject identity against class signal.** Columns:
- η²: the fraction of window-level variance explained by subject or by class (median over the 5 bands).
- subject ID: a bag's first half of windows matched to its own second half, nearest centroid.
- class AUC: 5-fold logistic regression on bag means.

| condition | η² subject | η² class | subject ID (chance) | class AUC |
|---|---|---|---|---|
| L1/CGX | 0.62 | 0.0001 | 0.28 (0.008) | 0.45 |
| L6/CGX | 0.63 | 0.017 | 0.34 (0.008) | 0.58 |
| L11/CGX | 0.65 | 0.020 | 0.28 (0.008) | 0.57 |
| Robots/EMOTIV | 0.43 | 0.030 | 0.50 (0.017) | 0.73 |
| Cognifit/EMOTIV | 0.54 | 0.004 | 0.48 (0.016) | 0.64 |

- Subject identity explains 20 to over 1000 times more window variance than the class does, and the class signal in band power is weak (AUC 0.45-0.73). This is measured, not assumed.
- What K2 needs, though, is the fingerprint ACROSS conditions. The relevant reading is the own-partner percentile: for each common subject, the fraction of other subjects whose other-condition bag lies farther away than the subject's own (bag means, per-domain standardized; 0.5 is chance, 1 is always nearest).

| pair (arm) | common subjects | percentile, all | percentile, within class | own bag nearest (chance) |
|---|---|---|---|---|
| L1 / L11 (level, core) | 119 | 0.795 | 0.789 | 0.126 (0.008) |
| Cognifit / Robots (task) | 55 | 0.734 | 0.723 | 0.036 (0.018) |
| L6 / Robots (device) | 56 | 0.644 | 0.647 | 0.036 (0.018) |

So the EEG cross-condition fingerprint is real but moderate in the level arm (the own bag is nearest for 1 subject in 8), weak in the task arm, and close to chance in the device arm.

**(c) Domain shift.** The protocol's domain classifier works on band power with subject-disjoint sides and is grouped by subject (§5, `separability.csv`): level L1-L11 AUC 0.636, task 0.665, device 0.978. The other level pairs (0.508, 0.445) are at chance.

**(d) Marked windows (§6.1: integrity per §2.3 or bad contact per §2.5).** Columns are the per-bag fraction of marked windows.

| condition | median | Q25-Q75 | bags with any marked | mean over bags with any |
|---|---|---|---|---|
| L1, L6, L11/CGX | 0 | 0-0 | 0 of 369 | n/a |
| Robots/EMOTIV | 0.043 | 0-1.0 | 65 % | 0.63 |
| Cognifit/EMOTIV | 0.319 | 0.002-1.0 | 77 % | 0.59 |

- CGX has no contact indicator, and its integrity marks are 0 in every non-excluded bag. The marked-window contrast of §6.1 therefore has no subject to compute on in the level arm, which is where K1's success criterion is evaluated. This is a protocol issue for the parent; the toy does not change it.
- On EMOTIV the fraction is bimodal: many bags are clean and many are marked in every window. A marked window means bad contact on at least one of 14 electrodes, not a corrupted window.

**(e) Class balance.** CGX is 65 % ADHD (80 / 43); EPOC X is balanced (30 / 31).

### 0.2 The toy measured with the same instruments (rollo, CONDITIONS_SCALE, RUN3, seeds 10-11)

| reading | toy | EEG target |
|---|---|---|
| bag size | 50 | 72-75 (CGX, Robots); 529 (Cognifit) |
| bags per class per domain (train) | 39 (16) | ≈ 40-80 ADHD / 22-44 control per condition |
| classes | 3, balanced | 2; 65 % ADHD on CGX |
| class AUC, bag means, linear | 0.90-0.91 | 0.45-0.73 |
| domain AUC, raw bag means | 1.00 (any rotation; the 3σ translation of `perturbar` dominates) | level 0.64, task 0.67, device 0.98 |
| domain AUC, per-domain standardized | 0.39-0.45 (the rotation preserves Σ) | n/a |
| own-partner percentile within class, no signature, rotation 25 | 0.725 | level 0.789, task 0.723, device 0.647 |

### 0.3 Calibration

- **Subject signature.** The knob is expressed as a multiple of the source's between-class separation. Scan at rotation 25, within-class own-partner percentile over the 16 train bags (mean of seeds 10 and 11):

  | multiple | 0 | 0.05 | 0.10 | 0.13 | 0.15 | 0.20 | 0.5 | 1.0 |
  |---|---|---|---|---|---|---|---|---|
  | percentile | 0.725 | 0.733 | 0.758 | 0.776 | **0.789** | 0.817 | 0.929 | 0.975 |

  - The level arm (0.789) is matched at **0.15 × separation**, an amplitude of about 0.59, which is below the within-bag spread of about 3.45.
  - The task arm (0.723) is matched with no signature at all.
  - The device arm (0.647) lies below anything the toy reaches without adding noise.
  - The run-5 first try (amplitude 3.7-3.99, about 0.93-0.99 × separation) made the toy far more identifiable than EEG.
- **Shift.** The only shift knob is `rotation_degrees`; the 3σ translation is fixed inside `CREDA.toy_experiment.perturbar`. In raw terms the toy shift is therefore device-like (domain AUC 1.0 against 0.978) at every rotation, while its cross-domain subject identity is level-like. Level-like shift size (AUC about 0.64) is not reachable with the declared knobs. Rotation stays at run 3's 25.
- **Artifact rate.** 0.2, 0.3 and 0.4 all lie inside the observed EMOTIV range, so none is excluded by EEG. The EEG distribution is bimodal per bag, which a fixed per-bag rate does not imitate, and the CGX level arm has no marked windows at all. The toy artifact (noise of 1 esc on every coordinate) is stronger than a single bad electrode out of 14.
- **Bag size, class balance, class count.** Kept at 50 points per bag, 3 balanced classes. Bag size enters only the consensus estimate, and imbalance enters none of the K2/K3 mechanisms. A 75-point scale would cost about 1.5× compute and break comparability with runs 3-4.
- **Class signal.** The toy class is much stronger than EEG's (0.90 against 0.45-0.73). Weakening it is outside the declared knobs. Consequently the routing error at the start of phase 2 is driven by the shift (0.667 with the target collapsed onto one class), not by a weak class.

### 0.4 What the calibrated toy can and cannot settle

- **Can settle:**
  - K3a as an aggregation question on artifact bags at a declared rate.
  - K3b and K3c accuracy contrasts under a device-sized raw shift with level-like subject identity.
  - Whether the local term raises the kernel-space partner rank during phase 2 (rank at the end against rank at the start, TGL against TG).
- **Cannot settle:**
  - K2b and K3b under the pre-registered gates. In input space the toy already carries EEG-level subject identity (0.79), yet the frozen phase-1 kernel puts the partner at chance (start rank 0.02-0.08 against a 0.5 gate) and pi stays near uniform (0.98-1.0). The gates fail for a reason EEG input-space evidence does not rule out for EEG either: no EEG kernel-space reading exists on disk.
  - K1 / K3a on the level arm, which has no marked windows.
  - Anything about the level arm's shift size.
  - The absolute class difficulty of ADHD.
