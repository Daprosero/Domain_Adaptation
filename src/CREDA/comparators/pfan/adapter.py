"""PFAN adapter (Chen et al., CVPR 2019) over the PyTorch translation in pfan_torch.py.

Per-batch term (`forward`): G_loss + semantic (APA) loss, the two adaptation
terms of model.py L187 (`F_loss = loss + Gregloss + adlamb*G_loss +
adlamb*Semanticloss`); the caller applies `adaptation_factor` (train.py L26-31)
as the trade-off.

THE DISCRIMINATOR HAS ITS OWN OPTIMISER. PFAN's adversarial term is G = -D
(model.py L151-152), not a gradient-reversal layer: putting D in the main
optimiser would make D ascend its own loss, and a single backward of
`out + D_loss` cancels the adversarial gradient on the features exactly.
Upstream (`adoptimize`, model.py L74-87) minimises D_loss + Dregloss over D's
variables only, with momentum 0.9, biases at 2x lr and Dregloss = 0.0005 *
mean l2_loss of the weights. Caller contract, per training step:

    main_opt = SGD(list(backbone/head params) + list(pfan.generator_parameters()), ...)
    d_opt    = SGD(pfan.discriminator_param_groups(lr, momentum=0.9))
    loss = ce + lam * pfan(f_s, f_t, g_s, g_t, y_s); main_opt.zero_grad(); loss.backward(); main_opt.step()
    pfan.discriminator_step(d_opt)                 # D on DETACHED features of that batch
    pfan.update_centroids(*pfan.last_centroids)

`forward` runs D with detached parameters, so the generator backward never
writes a gradient into D; `discriminator_loss()` runs D on detached features,
so the D update never touches the features. `generator_parameters()` is empty
(PFAN owns no generator weights, only centroid buffers); it exists so the
caller never has to filter `.parameters()`, which still includes D.

UPDATE THE CENTROIDS. Skipping `update_centroids` leaves the moving centroids
at zero, so the semantic term only aligns per-batch prototypes: that is the
paper's woA ablation (no accumulation), not PFAN.

NOT a per-batch loss: `ehts_select` (EHTS pseudo-labelling, train.py + pseudo.py)
is a per-round procedure over the whole target set. It returns the selected
target indices; following Eq. 11 means feeding ONLY those targets to the
adaptation term. The official code ignores the selection and feeds all targets.

`temperature` (T=1.8, model.py L60) has no effect under this contract: upstream
uses it only inside the softmax whose argmax gives the target pseudo-class, and
argmax is invariant to T; the classification CE is computed by the caller.
"""
from __future__ import annotations

from .pfan_torch import PFAN as _PFAN
from .pfan_torch import adaptation_factor, ehts_select  # noqa: F401  (re-exported)


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        'third-party baseline: PFAN (Chen et al., CVPR 2019), Eq. 9 and 11 '
        'through a PyTorch translation of Xiewp/PFAN@328c4c5; it implements '
        'no equation of the CREDA research concept (see CITATIONS.md)'
    ),
    "invariants": [
        "generator_and_discriminator_parameter_sets_are_disjoint_and_cover_all",
        "forward_value_is_unchanged",
        "generator_backward_reaches_features_and_never_d",
        "discriminator_step_trains_d_only_and_leaves_phi_grads_untouched",
    ],
}


class PFAN(_PFAN):
    def __init__(self, feature_dim: int, num_classes: int, temperature: float = 1.8, decay: float = 0.3) -> None:
        super().__init__(num_classes=num_classes, featurelen=feature_dim, temperature=temperature, decay=decay)
        self._last_features = None

    def forward(self, f_s, f_t, g_s, g_t, y_s):
        G_loss, D_loss, sc, tc = self.adloss(f_s, f_t, g_t, y_s, detach_discriminator=True)
        self.last_g_loss = G_loss
        self.last_d_loss = D_loss.detach()  # for logging; train D with discriminator_step
        self.last_centroids = (sc, tc)
        self._last_features = (f_s.detach(), f_t.detach())
        return G_loss + self.semantic_loss

    def generator_parameters(self):
        """Parameters for the main optimiser: none (centroids are buffers)."""
        d_ids = {id(p) for p in self.D.parameters()}
        return [p for p in self.parameters() if id(p) not in d_ids]

    def discriminator_parameters(self):
        """Parameters for D's own optimiser only; never the main one."""
        return list(self.D.parameters())

    def discriminator_loss(self, f_s=None, f_t=None):
        """D's own objective, model.py L151+L157 plus Dregloss (L83): 0.1*BCE(target=1,
        source=0) + regulariser, on DETACHED features (the last forward's by default)."""
        if f_s is None or f_t is None:
            if self._last_features is None:
                raise RuntimeError("discriminator_loss needs features: call forward first or pass f_s, f_t")
            f_s, f_t = self._last_features
        f_s, f_t = f_s.detach(), f_t.detach()
        source_logits, _ = self.D(f_s)
        target_logits, _ = self.D(f_t)
        bce = self._bce(target_logits, 1.0) + self._bce(source_logits, 0.0)
        return 0.1 * bce + self.discriminator_regularizer()

    @staticmethod
    def _bce(logits, label):
        import torch.nn.functional as F
        return F.binary_cross_entropy_with_logits(logits, logits.new_full(logits.shape, label))

    def discriminator_step(self, optimizer, f_s=None, f_t=None):
        """One upstream D update: zero D's grads, backward discriminator_loss, step.
        `optimizer` must hold only discriminator_parameters() (see
        discriminator_param_groups). Returns the detached loss."""
        optimizer.zero_grad(set_to_none=True)
        loss = self.discriminator_loss(f_s, f_t)
        loss.backward()
        optimizer.step()
        return loss.detach()
