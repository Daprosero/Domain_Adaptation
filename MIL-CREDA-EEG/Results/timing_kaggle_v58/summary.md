# v58 timing probes and placement (2026-11-06)

Short probes, warm-up epoch excluded, inner split of L1->L11. Source: `timing_kaggle_v58.json`, `probes/`, `../timing_probe_v58/`.

## Per-epoch seconds

| Unit | Local CPU (4 threads) | Kaggle CPU | Kaggle T4 |
|---|---|---|---|
| Phase 1: common / AdaptMIL / MIL-CREDA | 14.5 / 14.5 / 12.8 | 9.2 / 8.6 / 9.2 | 0.40 / 0.42 / 0.42 |
| Phase 2: AdaptMIL / UDA-DDA / CCA-LSC | 23.0 / 24.8 / 31.3 | 15.1 / 14.9 / 14.8 | 0.66 / 0.65 / 0.68 |
| Phase 2: CASUAL | 71.4 | 37.6 | 1.49 |
| Phase 2: COT / CS+CCS / CLUST | 32.8 / 37.0 / 29.3 | 14.2 / 14.4 / 14.7 | 0.68 / 0.66 / 0.65 |
| Phase 2: CLUDA / MIL-CREDA | 33.4 / 27.2 | 14.4 / 14.8 | 0.95 / 1.30 |
| Data load per unit | 0.6 | 38.6 | 42.9 |

Local MPS cannot run the models (float64 attention and head).

## Serial cost (seconds; C1, C2 = epoch caps, pending in plan v58 §7.6)

- Local CPU: 136534·C1 + 890214·C2 + 4847
- Kaggle CPU: 88568·C1 + 444953·C2 + 20222
- Kaggle T4: 4065·C1 + 22292·C2 + 21092

## Makespan with 15 accounts (GPU 2 notebooks per account, CPU 5 per account)

C1 = C2 = C. The C columns are illustrative.

| Option | C = 10 | C = 20 | C = 30 | Largest C in 1 week | Largest C in 2 weeks |
|---|---|---|---|---|---|
| Local CPU (serial) | 2853 h | 5706 h | 8558 h | none | none |
| Kaggle CPU only | 27.8 h | refused | refused | 4 | 4 |
| Kaggle T4 only | 3.24 h | 6.10 h | 8.95 h | 59 | 105 |
| Mixed T4 + CPU | 3.24 h | 5.67 h | 8.34 h | not computed | not computed |

Kaggle CPU is bounded by the 12 h session (assumed): the longest CASUAL unit fits only up to C = 4. The T4 two-week cap is bounded by the longest CASUAL unit fitting one session (C = 105). Above C ≈ 59 the campaign exceeds the team's 450 weekly GPU hours (30 h per account, read for daprosero only and assumed for the others).

Recommendation: Kaggle T4 on all 15 accounts. Unmeasured: queue time before a kernel starts; per-session setup taken from the kernel log; one probe per unit on one inner split.
