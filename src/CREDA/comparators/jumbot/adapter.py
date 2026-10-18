"""JUMBOT adapter (Fatras et al., ICML 2021) over the vendored kilianFatras/JUMBOT code.

Calls the verbatim digits/jumbot.py L82-108 lines (`Jumbot.da_loss`): ground cost
eta1 * ||f_s - f_t||^2 + eta2 * (-y_s . log softmax(g_t)), an unbalanced entropic
OT plan from POT (`sinkhorn_knopp_unbalanced`, plan detached), and
sum(pi * cost). Upstream names features `g_x` and logits `f_g_x`; here f is the
feature and g the logit, mapped accordingly.

Hyperparameters come from per-dataset presets taken from the paper, appendix C.1
(PDF p.21), cross-checked against the official repo at af9a22d:

* ``office_home``: eta1=0.01, eta2=0.5, tau=0.5, epsilon=0.01
  (Domain_Adaptation/train_jumbot.py L419 comment and L423-425; epsilon is the
  literal 0.01 passed to sinkhorn_knopp_unbalanced at L271-272).
* ``visda``: eta1=0.005, eta2=1.0, tau=0.3, epsilon=0.01
  (train_jumbot.py L420 comment; same L271-272 epsilon).
* ``digits``: eta1=0.1, eta2=0.1, tau=1.0, epsilon=0.1 (digits/train.py L84-87).

The default preset is ``office_home``. This is an assumption: the paper reports
no Office-31 setting, and Office-Home is its closest ResNet-50 benchmark. The
`Jumbot.__init__` signature defaults (digits/jumbot.py L30: eta1=0.001,
eta2=0.0001) match no experiment in the paper and are not used. Any explicit
eta1/eta2/tau/epsilon kwarg overrides the preset value; an unknown preset raises
ValueError.

The paper's Eq. (10) weights this term by eta3, which is 1 in every reported
experiment (appendix C.1; train_jumbot.py L279 adds it unweighted). This
comparator returns the bare term (eta3 = 1); any other weight is the caller's.

The paper uses class-stratified source minibatches for every dataset (appendix
C.1, PDF p.20: "each class has the same number of samples in the minibatches";
train_jumbot.py L120 `BalancedBatchSampler`). This adapter does not sample:
building stratified source batches is the caller's responsibility.

Not zero for identical batches: the entropic plan spreads mass off the diagonal
and the label cost is non-zero for every non-degenerate prediction.
"""
from __future__ import annotations

import contextlib
import io

from .upstream_jumbot import Jumbot

PRESETS: dict[str, dict[str, float]] = {
    "office_home": {"eta1": 0.01, "eta2": 0.5, "tau": 0.5, "epsilon": 0.01},
    "visda": {"eta1": 0.005, "eta2": 1.0, "tau": 0.3, "epsilon": 0.01},
    "digits": {"eta1": 0.1, "eta2": 0.1, "tau": 1.0, "epsilon": 0.1},
}
DEFAULT_PRESET = "office_home"


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        'third-party baseline: JUMBOT (Fatras et al., ICML 2021), Eq. 9 '
        'through the vendored kilianFatras/JUMBOT@af9a22d code, appendix C.1 '
        'presets; it implements no equation of the CREDA research concept '
        '(see CITATIONS.md)'
    ),
    "invariants": [
        "default_build_uses_office_home_values",
        "each_preset_sets_the_four_values",
        "loss_with_explicit_values_matches_upstream_direct_call",
    ],
}


class JUMBOT:
    def __init__(self, feature_dim: int, num_classes: int, preset: str = DEFAULT_PRESET,
                 eta1: float | None = None, eta2: float | None = None,
                 tau: float | None = None, epsilon: float | None = None) -> None:
        if preset not in PRESETS:
            raise ValueError(
                f"unknown JUMBOT preset {preset!r}; expected one of {sorted(PRESETS)}"
            )
        hp = dict(PRESETS[preset])
        for key, value in (("eta1", eta1), ("eta2", eta2), ("tau", tau), ("epsilon", epsilon)):
            if value is not None:
                hp[key] = value
        self.preset = preset
        with contextlib.redirect_stdout(io.StringIO()):  # upstream __init__ prints its hyperparameters
            self.jumbot = Jumbot(None, None, num_classes, **hp)

    def __call__(self, f_s, f_t, g_s, g_t, y_s, **extra):
        return self.jumbot.da_loss(g_t, f_s, f_t, y_s)
