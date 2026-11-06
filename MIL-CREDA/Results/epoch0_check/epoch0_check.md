# Epoch-0 toy-set check of the phase-2 centroid rule

Protocol: experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md, Sec. 4.5. Arm `TGL`, scale `full`, device `cpu`, torch 2.13.0, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. Measured 2026-11-06T08:13:32.783506+00:00.

Declared before measuring: one row per (transfer, seed); a transfer passes when the epoch-0 centroid score is strictly lower than the score at the epoch the rule chooses (argmax over phase-2 epochs 1..E, earliest on ties) in the majority of its seeds; the rule is 'adecuada' on the toy set when the majority of the transfers pass (plan v58, Sec. 4.5).

## Per (transfer, seed)

| Transfer | Seed | Score epoch 0 | Chosen epoch | Score at chosen epoch | Epoch 0 < chosen? |
|---|---|---|---|---|---|
| lunas source -> lunas target | 0 | 0.8000 | 10 | 0.8667 | yes |
| lunas source -> lunas target | 1 | 0.6000 | 1 | 0.6000 | no |
| lunas source -> lunas target | 2 | 0.4000 | 1 | 0.7333 | yes |
| lunas source -> lunas target | 3 | 0.3333 | 1 | 0.6667 | yes |
| lunas source -> lunas target | 4 | 0.7333 | 15 | 0.8000 | yes |
| lunas source -> lunas target | 5 | 0.4667 | 1 | 0.8000 | yes |
| lunas source -> lunas target | 6 | 0.6667 | 3 | 0.7333 | yes |
| lunas source -> lunas target | 7 | 0.4000 | 1 | 0.6667 | yes |
| lunas source -> lunas target | 8 | 0.6667 | 12 | 0.6000 | no |
| lunas source -> lunas target | 9 | 0.6667 | 1 | 0.7333 | yes |
| toroide source -> toroide target | 0 | 0.3333 | 1 | 1.0000 | yes |
| toroide source -> toroide target | 1 | 0.5333 | 6 | 1.0000 | yes |
| toroide source -> toroide target | 2 | 0.6000 | 3 | 1.0000 | yes |
| toroide source -> toroide target | 3 | 0.3333 | 1 | 1.0000 | yes |
| toroide source -> toroide target | 4 | 0.4000 | 1 | 1.0000 | yes |
| toroide source -> toroide target | 5 | 0.3333 | 2 | 1.0000 | yes |
| toroide source -> toroide target | 6 | 0.6000 | 1 | 1.0000 | yes |
| toroide source -> toroide target | 7 | 0.3333 | 3 | 1.0000 | yes |
| toroide source -> toroide target | 8 | 0.6667 | 1 | 1.0000 | yes |
| toroide source -> toroide target | 9 | 0.4000 | 1 | 1.0000 | yes |
| rollo source -> rollo target | 0 | 0.6000 | 1 | 1.0000 | yes |
| rollo source -> rollo target | 1 | 0.4000 | 2 | 0.9333 | yes |
| rollo source -> rollo target | 2 | 0.5333 | 1 | 0.8667 | yes |
| rollo source -> rollo target | 3 | 0.4667 | 1 | 1.0000 | yes |
| rollo source -> rollo target | 4 | 0.3333 | 2 | 1.0000 | yes |
| rollo source -> rollo target | 5 | 0.6000 | 2 | 1.0000 | yes |
| rollo source -> rollo target | 6 | 0.3333 | 1 | 1.0000 | yes |
| rollo source -> rollo target | 7 | 0.5333 | 1 | 1.0000 | yes |
| rollo source -> rollo target | 8 | 0.4667 | 6 | 1.0000 | yes |
| rollo source -> rollo target | 9 | 0.5333 | 1 | 1.0000 | yes |

## Per transfer

| Transfer | Seeds with epoch 0 < chosen | Passes |
|---|---|---|
| lunas | 8/10 | yes |
| toroide | 10/10 | yes |
| rollo | 10/10 | yes |

## Verdict

adecuada: the epoch-0 score is lower than the chosen epoch's in the majority of the transfers (3/3).
