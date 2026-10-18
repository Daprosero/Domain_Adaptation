# VENDORED UPSTREAM CODE -- copied verbatim from the paper's official repository.
# Paper: Fatras, Sejourne, Courty, Flamary. Unbalanced minibatch Optimal Transport; applications to Domain Adaptation. ICML 2021 (PMLR 139)
# Repository: https://github.com/kilianFatras/JUMBOT
# Commit: af9a22d800cf9b9c1243ab125314eefaf06fdda8
# Upstream path: digits/jumbot.py (excerpts: L14-22, L24-50, L82, L87-102, L108)
# License as found: MIT (LICENSE at repo root)
# Copied with the operator's explicit authorisation, cited here and in ../CITATIONS.md.
# Every deviation from the upstream bytes is marked with a `# LOCAL EDIT:` comment;
# all other lines (including their indentation) are unchanged.

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import itertools
import torch.nn.functional as F

import ot

# LOCAL EDIT: upstream L23 `from utils import model_eval` dropped (training-loop helper, not the loss).

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Jumbot(object):
    """Jumbot class"""
    def __init__(self, model_g, model_f, n_class, eta1=0.001, eta2=0.0001, tau=1., epsilon=0.1):
        """
        Initialize jumbot method.
        args :
        - model_g : feature exctrator (torch.nn)
        - model_f : classification layer (torch.nn)
        - n_class : number of classes (int)
        - eta_1 : feature comparison coefficient (float)
        - eta_2 : label comparison coefficient (float)
        - tau : marginal coeffidient (float)
        - epsilon : entropic regularization (float)
        """
        self.model_g = model_g   # target model
        self.model_f = model_f
        self.n_class = n_class
        self.eta1 = eta1  # weight for the alpha term
        self.eta2 = eta2 # weight for target classification
        self.tau = tau
        self.epsilon = epsilon
        print('eta1, eta2, tau, epsilon: ', self.eta1, self.eta2, self.tau, self.epsilon)
    
    # LOCAL EDIT: wrapper def; body is upstream L82 and L87-102, L108 (inside fit()), indentation kept.
    def da_loss(self, f_g_xt_mb, g_xs_mb, g_xt_mb, ys):
                pred_xt = F.softmax(f_g_xt_mb, 1)
                ###  Ground cost
                embed_cost = torch.cdist(g_xs_mb, g_xt_mb)**2
                
                ys = F.one_hot(ys, num_classes=self.n_class).float()
                t_cost = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
                
                total_cost = self.eta1 * embed_cost + self.eta2 * t_cost

                #OT computation
                a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(),
                                                             self.epsilon, self.tau)
                # To get DeepJDOT (https://arxiv.org/abs/1803.10081) comment the line above 
                # and uncomment the following line:
                #pi = ot.emd(a, b, total_cost.detach().cpu().numpy())
                # LOCAL EDIT: device-agnostic, no CUDA (upstream L102 was: pi = torch.from_numpy(pi).float().cuda())
                pi = torch.from_numpy(pi).float().to(total_cost.device)
                da_loss = torch.sum(pi * total_cost)
                return da_loss  # LOCAL EDIT: return added
