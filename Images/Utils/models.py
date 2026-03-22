import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.serialization
from torchvision import models, datasets, transforms
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import random
import math
import os
import zipfile
from itertools import permutations
from tqdm import tqdm
import timm
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Gradient Reversal Layer (usada en DANN y opcionalmente en CDAN)
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class FeatureExtractor(nn.Module):
    """
    backbone : str
        Opciones:
        - 'resnet18'
        - 'resnet50'
        - 'vit_tiny_patch16_224'
    pretrained : bool
        Si carga pesos preentrenados.
    """
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()

        self.backbone = backbone

        if backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.flatten = nn.Flatten()
            self.output_dim = 512

        elif backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.flatten = nn.Flatten()
            self.output_dim = 2048

        elif backbone == 'vit_tiny_patch16_224':
            self.features = timm.create_model(
                backbone,
                pretrained=False,
                img_size=32
            )
            self.output_dim = self.features.head.in_features
            self.features.head = nn.Identity()
            self.flatten = nn.Identity()

            if pretrained:
                pretrained_model = timm.create_model(backbone, pretrained=True)
                state_dict = pretrained_model.state_dict()
                state_dict = self._interpolate_pos_embed(state_dict, self.features)
                self.features.load_state_dict(state_dict, strict=False)

        else:
            raise ValueError(
                f"Backbone no soportado: {backbone}. "
                f"Usa 'resnet18', 'resnet50' o 'vit_tiny_patch16_224'."
            )

    def _interpolate_pos_embed(self, state_dict, model):
        if 'pos_embed' not in state_dict:
            return state_dict

        posemb = state_dict['pos_embed']
        posemb_new = model.pos_embed

        num_extra_tokens = posemb.shape[1] - int(math.sqrt(posemb.shape[1] - 1)) ** 2

        cls_token = posemb[:, :num_extra_tokens]
        posemb_grid = posemb[:, num_extra_tokens:]

        gs_old = int(math.sqrt(posemb_grid.shape[1]))
        gs_new = int(math.sqrt(posemb_new.shape[1] - num_extra_tokens))

        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(
            posemb_grid,
            size=(gs_new, gs_new),
            mode='bicubic',
            align_corners=False
        )
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

        posemb = torch.cat([cls_token, posemb_grid], dim=1)
        state_dict['pos_embed'] = posemb
        return state_dict

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

class Classifier(nn.Module):
    def __init__(self, feature_dim=256, num_classes=10):
        super().__init__()
        set_seed(42)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.BatchNorm1d(feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim//4),
            nn.BatchNorm1d(feature_dim//4),
            nn.ReLU(),
            nn.Linear(feature_dim//4, num_classes)
        )

    def forward(self, x):
        return self.net(x)
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        set_seed(42)
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.BatchNorm1d(input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, 1)
        )
    def forward(self, x):
        return self.net(x)
class DANN_ResNet(nn.Module):
    def __init__(self, backbone,num_classes=10, lambda_grl=1.0):
        super().__init__()
        set_seed(42)
        self.feature = FeatureExtractor(backbone=backbone, pretrained=True)
        self.classifier = Classifier(feature_dim=self.feature.output_dim, num_classes=num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim=self.feature.output_dim)
        self.grl = GradientReversalLayer(lambda_grl)

    def forward(self, x, mode='class'):
        f = self.feature(x)
        if mode == 'class':
            return self.classifier(f)
        elif mode == 'domain':
            return self.domain_discriminator(self.grl(f))
class ADDA_ResNet(nn.Module):
    def __init__(self,backbone, num_classes=10):
        super().__init__()
        set_seed(42)
        self.Fs = FeatureExtractor(backbone=backbone, pretrained=True)
        self.Ft = FeatureExtractor(backbone=backbone, pretrained=True)
        self.classifier = Classifier(feature_dim=self.Fs.output_dim, num_classes=num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim=self.Fs.output_dim)

    def forward(self, x, domain='source', mode='class'):
        if domain == 'source':
            f = self.Fs(x)
        elif domain == 'target':
            f = self.Ft(x)
        else:
            raise ValueError("domain must be 'source' or 'target'")
        if mode == 'class':
            return self.classifier(f)
        elif mode == 'domain':
            return self.domain_discriminator(f)
        else:
            raise ValueError("mode must be 'class' or 'domain'")
class CDAN_ResNet(nn.Module):
    def __init__(self,backbone, num_classes=10):
        super().__init__()
        set_seed(42)
        self.feature = FeatureExtractor(backbone=backbone, pretrained=True)
        self.classifier = nn.Linear(self.feature.output_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim=self.feature.output_dim * num_classes)

    def forward(self, x, mode='class'):
        f = self.feature(x)
        y = self.classifier(f)
        if mode == 'class':
            return y
        elif mode == 'domain':
            y_soft = F.softmax(y, dim=1)
            outer = torch.bmm(y_soft.unsqueeze(2), f.unsqueeze(1)).view(x.size(0), -1)
            return self.domain_discriminator(outer)

class CREDA_ResNet(nn.Module):
    def __init__(self,backbone, num_classes=10):
        super().__init__()
        set_seed(42)
        self.feature = FeatureExtractor(backbone=backbone, pretrained=True)
        self.classifier = nn.Linear(self.feature.output_dim, num_classes)

    def forward(self, x, mode='class'):
        f = self.feature(x)
        y = self.classifier(f)
        if mode == 'class':
            return y  # softmax se aplica externamente si es necesario
        elif mode == 'feature':
            return f
class CREDALoss(nn.Module):
    def __init__(self, sigma='auto', lambda_creda=1.0, use_entropy_weighting=True):
        super(CREDALoss, self).__init__()
        self.lambda_creda = lambda_creda
        self.sigma = sigma
        self.use_entropy_weighting = use_entropy_weighting
    def _squared_euclidean_dist(self, x, y):
        # x: (N, D), y: (M, D)
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)  # (N, 1)
        y_norm = (y ** 2).sum(dim=1).view(1, -1)  # (1, M)
        dist_sq = x_norm + y_norm - 2.0 * (x @ y.T)
        return torch.clamp(dist_sq, min=0.0)

    def _compute_sigma(self, x, y):
        combined = torch.cat([x, y], dim=0)
        dist_sq = self._squared_euclidean_dist(combined, combined)
        # Elimina diagonal (distancia 0) para evitar sesgo
        non_diag = dist_sq[~torch.eye(dist_sq.shape[0], dtype=bool, device=dist_sq.device)]
        return torch.sqrt(torch.median(non_diag) + 1e-6)

    def _gaussian_kernel(self, x, y, sigma_val):
        dist_sq = self._squared_euclidean_dist(x, y)
        return torch.exp(-dist_sq / (2 * sigma_val ** 2))


    def _renyi_entropy_order_2(self, K):
        tr_K = torch.trace(K) + 1e-6
        K_norm = K / tr_K
        log2 = torch.log(torch.tensor(2.0, device=K.device))
        return -torch.log(torch.trace(K_norm @ K_norm) + 1e-6) / log2

    def _mix_kernel_concat(self, K_s, K_t, K_st):
        K_mix = torch.cat([
            torch.cat([K_s, K_st], dim=1),
            torch.cat([K_st.T, K_t], dim=1)
        ], dim=0)
        return K_mix / (torch.trace(K_mix) + 1e-6)

    def forward(self, f_s, f_t, y_s, g_t, reduction='mean'):
        y_t_pseudo = torch.argmax(g_t, dim=1)
        unique_classes = torch.unique(y_s, sorted=True)
        losses_per_class = []
        valid_class_count = 0

        if self.use_entropy_weighting:
            squared_sum = torch.sum(g_t ** 2, dim=1)
            log2 = torch.log(torch.tensor(2.0, device=g_t.device))
            entropy = -torch.log(squared_sum + 1e-6) / log2
            target_weights = 1.0 - entropy / torch.log(torch.tensor(float(g_t.shape[1]), device=g_t.device))
        else:
            target_weights = None

        for c in unique_classes:
            f_s_c = f_s[y_s == c]
            f_t_c = f_t[y_t_pseudo == c]

            if f_s_c.shape[0] == 0 or f_t_c.shape[0] == 0:
                continue

            sigma_val = self._compute_sigma(f_s_c, f_t_c) if self.sigma == 'auto' else self.sigma

            K_s_c = self._gaussian_kernel(f_s_c, f_s_c, sigma_val)
            K_t_c = self._gaussian_kernel(f_t_c, f_t_c, sigma_val)
            K_st_c = self._gaussian_kernel(f_s_c, f_t_c, sigma_val)

            if self.use_entropy_weighting:
                weights_c = target_weights[y_t_pseudo == c]
                W_c = torch.outer(weights_c, weights_c)
                K_t_c = K_t_c * W_c

            K_mix_c = self._mix_kernel_concat(K_s_c, K_t_c, K_st_c)

            h_s = self._renyi_entropy_order_2(K_s_c)
            h_t = self._renyi_entropy_order_2(K_t_c)
            h_mix = self._renyi_entropy_order_2(K_mix_c)
            creda_c = h_mix - 0.5 * (h_s + h_t)
            losses_per_class.append(creda_c)
            valid_class_count += 1

        if valid_class_count == 0:
            return torch.tensor(0.0, device=f_s.device)

        losses = torch.stack(losses_per_class)

        if reduction == 'none':
            return self.lambda_creda * losses
        else:
            return self.lambda_creda * losses.mean()
