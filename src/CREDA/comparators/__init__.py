"""Official comparator losses, vendored from each paper's own repository.

Every method is exposed through one contract: `build(name, feature_dim,
num_classes, **upstream_kwargs)` returns a callable

    method(f_s, f_t, g_s, g_t, y_s, **extra) -> 0-d tensor

with f_* features (n, d), g_* logits (n, C), y_s source labels (n_s,) long. The
value is the ALIGNMENT / ADAPTATION term only, never the source cross-entropy and
never multiplied by the method's trade-off schedule.

Methods that own modules are nn.Module. CDAN+E: add its parameters to the main
optimiser (its discriminator sits behind a gradient-reversal layer). PFAN: its
adversarial term is G = -D, not a gradient-reversal layer, so add only
`generator_parameters()` to the main optimiser and train the discriminator with
its own optimiser via `discriminator_param_groups()` and `discriminator_step()`;
never pass `PFAN.parameters()` to the main optimiser. DALN owns no parameters:
its critic is the shared head, passed as `head=h` so it can be re-run behind
the gradient-reversal layer.

Per-epoch procedures (SHOT `obtain_label`, PFAN `ehts_select`) are separate
methods; SHOT's whole-dataset labels go back in with the batch's dataset
indices as `pseudo_labels=..., tar_idx=...`. See each adapter's docstring and
CITATIONS.md.
"""
from __future__ import annotations

from .cdan.adapter import CDANE
from .cs_da.adapter import CSDA
from .daln.adapter import DALN
from .dsan.adapter import LMMD
from .jumbot.adapter import JUMBOT
from .pfan.adapter import PFAN
from .shot.adapter import SHOT

REGISTRY = {
    "shot": SHOT,
    "cdan_e": CDANE,
    "dsan_lmmd": LMMD,
    "cs_da": CSDA,
    "pfan": PFAN,
    "jumbot": JUMBOT,
    "daln": DALN,
}


def build(name: str, feature_dim: int, num_classes: int, **kwargs):
    """Instantiate the comparator `name` with its documented defaults unless overridden.

    Defaults follow the official code except where the adapter docstring says the
    paper wins: CDAN+E picks the multilinear map by Eq. 8, JUMBOT uses the
    paper's Office-Home preset.
    """
    return REGISTRY[name](feature_dim, num_classes, **kwargs)


__all__ = ["REGISTRY", "build", "SHOT", "CDANE", "LMMD", "CSDA", "PFAN", "JUMBOT", "DALN"]
