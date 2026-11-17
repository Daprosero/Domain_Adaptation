# Local timing probes of the v58 campaign (2026-11-06)

Host: this Mac (macOS 26.5.2 arm64, 8 cores), torch 2.13.0, CPU with 4 torch threads.
Probe: `tools/timing_v58.py`, transfer L1->L11, outer fold 0, inner split 0, real
processed data, bagsPerStep 8 (not declared by the plan; Sec. 7.6), epochs 2 (CCA-LSC 4:
3 warm-up + 1 calibrated). Epoch time = steady epoch (epoch 1 of 0..1 and later), warm-up
excluded; it includes the epoch's validation pass.

## Steady epoch seconds (CPU, inner split of L1->L11)

| Unit | s/epoch |
|---|---|
| phase 1 common | 14.5 |
| phase 1 AdaptMIL | 14.5 |
| phase 1 MIL-CREDA | 12.8 |
| phase 2 AdaptMIL | 23.0 |
| phase 2 UDA-DDA | 24.8 |
| phase 2 CCA-LSC | 31.3 (warm-up 27.2/29.7, calibrated 37.2) |
| phase 2 CASUAL | 71.4 |
| phase 2 COT | 32.8 |
| phase 2 CS+CCS | 37.0 |
| phase 2 CLUST (source-free) | 29.3 |
| phase 2 CLUDA | 33.4 |
| phase 2 MIL-CREDA | 27.2 |
| centroid source pass (3 inner splits, once per phase-2 study) | 9.7-10.1 s |
| setup (load bags + search data) | 0.57 s |

Peak RSS about 4.8 GB.

## MPS

Not usable: every model fails at `.to("mps")` with "Cannot convert a MPS Tensor to float64
dtype as the MPS framework doesn't support float64" (the attention and head are float64 by
design, `mil_creda_model.py`). `PYTORCH_ENABLE_MPS_FALLBACK=1` does not help (it covers
unsupported ops, not dtypes). Fallback: CPU. No `mps` row is written to `timing.json`.

## Extrapolation (`tools/timing_v58_record.py`, `breakdown.json`)

The epoch caps C1 and C2 are still pending (Sec. 7.6), so the campaign is
`seconds = 136 534 * C1 + 890 214 * C2 + 4 847` on this CPU, serial: 37.9 h per epoch of
C1 and 247.3 h per epoch of C2. `timing.json` (schema 1) is normalized to C1 = C2 = 1;
`estimate` over it gives 1 031 594 s (286.6 h; range 229-344 h) for the full campaign and
placement `cloud`.
