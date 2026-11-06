"""The pure arithmetic of the proof-of-concept K2 readings, shared with the EEG.

`poc` re-exports both functions, so the toy battery reads them where it always
did; `MIL_CREDA_EEG.mechanisms` reads them for the Sec. 6.2 mechanism
diagnostics of the experiments plan (v56). Neither touches a model: they take
the correspondence or the distances a caller already measured.

No `__provenance__`: like the rest of this package, it implements no equation.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def entropy_ratio(pi: torch.Tensor) -> float:
    """H(pi) / log(k) for one distribution over k bags: 1 uniform, 0 one-hot.

    Undefined (nan) for k = 1, where there is no choice to spread over.
    """
    p = pi.detach().double().reshape(-1)
    if p.numel() < 2:
        return float("nan")
    nonzero = p[p > 0]
    return float(-(nonzero * nonzero.log()).sum() / math.log(p.numel()))


def partner_push(start: dict, end: dict) -> dict:
    """K2's push from two `partner_geometry` readings of one arm:

    - `pushMean`: mean over bags of d_start(own) - d_end(own), positive when
      the target bag moved toward its own partner;
    - `marginStart`/`marginEnd`: mean of own distance minus the median
      distance to the other same-class source bags (negative: own is closer);
    - `roomRatioMedian`: median over bags of d_start(own) / the class's
      same-class source spread, the room the local term starts with.
    """
    room = start["own"] / start["spread"]
    return {"pushMean": float((start["own"] - end["own"]).mean()),
            "ownDistanceStart": float(start["own"].mean()),
            "ownDistanceEnd": float(end["own"].mean()),
            "marginStart": float((start["own"] - start["othersMedian"]).mean()),
            "marginEnd": float((end["own"] - end["othersMedian"]).mean()),
            "roomRatioMedian": float(np.median(room.numpy())),
            "sourceSpreadMedian": float(np.median(start["spread"].numpy())),
            "bags": int(len(start["own"]))}
