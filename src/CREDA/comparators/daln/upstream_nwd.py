# VENDORED UPSTREAM CODE -- copied verbatim from the paper's official repository.
# Paper: Chen, Chen, Wei, Jin, Tan, Jin, Chen. Reusing the Task-specific Classifier as a Discriminator:
#   Discriminator-free Adversarial Domain Adaptation. CVPR 2022. arXiv:2204.03838
# Repository: https://github.com/xiaoachen98/DALN
# Commit: e40ca55612b7d6a973cd40e50cea32e350966a88
# Upstream path: nwd.py (whole file)
# License as found: none (the repository ships no LICENSE file)
# Copied with the operator's explicit authorisation, cited here and in ../CITATIONS.md.
# Every deviation from the upstream bytes is marked with a `# LOCAL EDIT:` comment;
# all other lines (including their indentation) are unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .upstream_grl import WarmStartGradientReverseLayer  # LOCAL EDIT: was `from grl import ...` (package-relative import)


class NuclearWassersteinDiscrepancy(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / y_t.shape[0]
        return loss

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f_grl = self.grl(f)
        y = self.classifier(f_grl)
        y_s, y_t = y.chunk(2, dim=0)

        loss = self.n_discrepancy(y_s, y_t)
        return loss
