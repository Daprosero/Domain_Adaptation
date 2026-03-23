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
import matplotlib.pyplot as plt
from skimage.transform import resize
from torchvision.transforms.functional import to_pil_image
from Domain_Adaptation.Images.Utils.models import (
    FeatureExtractor,
    Classifier,
    DomainDiscriminator,
    DANN_ResNet,
    ADDA_ResNet,
    CDAN_ResNet,
    CREDA_ResNet,
    GradientReversalFunction,
    GradientReversalLayer,
    CREDALoss
)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- División estratificada común ---
def split_stratified(dataset, val_ratio=0.2, test_ratio=None, seed=42):
    set_seed(42)
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("El dataset no tiene 'targets' ni 'labels'.")

    indices = np.arange(len(dataset))
    if test_ratio is None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(indices, labels))
        return Subset(dataset, train_idx), Subset(dataset, val_idx)
    else:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=seed)
        train_idx, temp_idx = next(sss1.split(indices, labels))
        temp_labels = labels[temp_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
        val_idx, test_idx = next(sss2.split(temp_idx, temp_labels))
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]
        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

# --- Batch automático ---
def util_auto_batch_size(dataset_len, max_bs=64, min_bs=16, fraction=0.1):
    bs = int(dataset_len * fraction)
    return max(min_bs, min(max_bs, bs))

def eval_model(feature_extractor, classifier, dataloader, device):
    set_seed(42)
    feature_extractor.eval()
    classifier.eval()

    total_correct = 0
    total_samples = 0
    batch_accuracies = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            feats = feature_extractor(x)
            preds = classifier(feats)
            loss = loss_fn(preds, y)
            total_loss += loss.item()

            correct = (preds.argmax(1) == y).sum().item()
            batch_acc = 100.0 * correct / y.size(0)
            batch_accuracies.append(batch_acc)

            total_correct += correct
            total_samples += y.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_samples
    std_acc = torch.std(torch.tensor(batch_accuracies)).item()

    return avg_loss, avg_acc, std_acc

# --- Evaluación final común ---
def eval_accuracy_only(feature_extractor, classifier, dataloader, device,num_classes):
    _, acc,std = eval_model(feature_extractor, classifier, dataloader, device)
    return acc,std

# --- Entrenamiento de DANN ---
def get_eta(epoch, total_epochs,alpha, eta_0=0.01, beta=0.75):
    p = epoch / total_epochs
    return eta_0 * (1 + alpha * p) ** (-beta)

def get_lambda(epoch, total_epochs, delta):
    p = epoch / total_epochs
    return (1 - np.exp(-delta * p)) / (1 + np.exp(-delta * p))

def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-5), dim=1)
def grl_hook(coeff):
    def hook(grad):
        return -coeff * grad.clone()
    return hook
def get_num_classes(dataset):
    targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
    targets = targets.tolist() if hasattr(targets, 'tolist') else list(targets)
    return len(set(targets))
def get_batch_size(domain, dataset_len, cfg):
    if domain in cfg["large_batch_domains"]:
        return cfg["large_batch_size"]
    return util_auto_batch_size(dataset_len)
def get_da_batch_size(src, tgt, src_len, tgt_len, cfg):
    if (src in cfg["large_batch_domains"]) or (tgt in cfg["large_batch_domains"]):
        return cfg["large_batch_size"], cfg["large_batch_size"]
    return util_auto_batch_size(src_len), util_auto_batch_size(tgt_len)
# --- Entrenamiento supervisado ---
def train_baseline(feature_extractor, classifier, train_loader, val_loader, device,num_classes, epochs=5, lr=1e-3):
    set_seed(42)
    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        feature_extractor.train()
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            feats = feature_extractor(x)
            preds = classifier(feats)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = 100.0 * correct / total
        train_loss = total_loss / len(train_loader)
        val_loss, val_acc,_ = eval_model(feature_extractor, classifier, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}% | Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

def train_dann(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, num_classes,alpha,delta, epochs=5, eta_0=1e-3):
    set_seed(42)
    optimizer = optim.Adam(model.parameters(), lr=eta_0)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()

        # Ajuste dinámico de tasa de aprendizaje y lambda
        lambda_val = get_lambda(epoch, epochs,delta=delta)
        if alpha!=None:
            new_lr = get_eta(epoch, epochs,alpha, eta_0)
            for param_group in optimizer.param_groups:
               param_group['lr'] = new_lr
        else:
            new_lr=0

        total_cls_loss, total_dom_loss = 0, 0
        cls_correct, dom_correct, total_dom = 0, 0, 0
        tgt_iter = iter(tgt_train_loader)

        for xs, ys in tqdm(src_train_loader, desc=f"[DANN] Epoch {epoch+1}/{epochs}"):
            try:
                xt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_train_loader)
                xt, _ = next(tgt_iter)

            # Enviar a dispositivo
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            # Predicción de clase
            preds_class = model(xs, mode='class')
            loss_class = criterion_class(preds_class, ys)
            cls_correct += (preds_class.argmax(1) == ys).sum().item()

            # Discriminación de dominio
            domain_src = model(xs, mode='domain')
            domain_tgt = model(xt, mode='domain')

            bs_src = domain_src.size(0)
            bs_tgt = domain_tgt.size(0)

            domain_preds = torch.cat([domain_src, domain_tgt], dim=0)
            domain_labels = torch.cat([
                torch.ones(bs_src, 1),
                torch.zeros(bs_tgt, 1)
            ], dim=0).to(device)

            loss_domain = criterion_domain(domain_preds, domain_labels)
            dom_correct += ((domain_preds > 0.5).float() == domain_labels).sum().item()
            total_dom += domain_labels.size(0)

            # Pérdida total
            loss = loss_class + lambda_val* loss_domain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cls_loss += loss_class.item()
            total_dom_loss += loss_domain.item()

        val_loss_src, val_acc_src,_ = eval_model(model.feature, model.classifier, src_val_loader, device)
        val_loss_tgt, val_acc_tgt,_ = eval_model(model.feature, model.classifier, tgt_val_loader, device)

        print(f"Epoch {epoch+1}: LR={new_lr:.6f}, Lambda={lambda_val:.4f}")
        print(f"  Train: Cls Loss={total_cls_loss:.4f}, Dom Loss={total_dom_loss:.4f}, "
              f"Cls Acc={100. * cls_correct / len(src_train_loader.dataset):.2f}%, "
              f"Dom Acc={100. * dom_correct / total_dom:.2f}%")
        print(f"  Val Src: Loss={val_loss_src:.4f}, Acc={val_acc_src:.2f}%")
        print(f"  Val Tgt: Loss={val_loss_tgt:.4f}, Acc={val_acc_tgt:.2f}%")

# --- ADDA Entrenamiento Fase 1 ---
def train_adda_phase1(model, loader_train, loader_val, device, num_classes,epochs=5, lr=1e-4):
    set_seed(42)
    optimizer = optim.Adam(list(model.Fs.parameters()) + list(model.classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(loader_train, desc=f"[ADDA Phase 1] Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            preds = model(x, domain='source', mode='class')
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = 100.0 * correct / total
        val_loss, val_acc,_ = eval_model(model.Fs, model.classifier, loader_val, device)
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Acc = {train_acc:.2f}%, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
# --- ADDA Entrenamiento Fase 2 ---
def train_adda_phase2(model, src_loader, tgt_loader, val_loader_src, val_loader_tgt, device,num_classes,alpha,delta, epochs=5, eta_0=1e-3,lr=1e-4):
    set_seed(42)
    for p in model.Fs.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = False

    optimizer_Ft = optim.Adam(model.Ft.parameters(), lr=eta_0)
    optimizer_D = optim.Adam(model.domain_discriminator.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()

        # Schedulers dinámicos
        lambda_val = get_lambda(epoch, epochs,delta=delta)
        if alpha!=None:
            new_lr = get_eta(epoch, epochs,alpha, eta_0)
            for param_group in optimizer_Ft.param_groups:
                param_group['lr'] = new_lr
        else:
            new_lr=0

        #for param_group in optimizer_D.param_groups:
        #    param_group['lr'] = new_lr

        total_domain_loss, total_target_loss = 0, 0
        dom_correct, total_dom = 0, 0
        tgt_iter = iter(tgt_loader)

        for xs, _ in tqdm(src_loader, desc=f"[ADDA Phase 2] Epoch {epoch+1}"):
            try:
                xt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt, _ = next(tgt_iter)

            xs, xt = xs.to(device), xt.to(device)

            # Discriminador
            feats_src = model.Fs(xs).detach()
            feats_tgt = model.Ft(xt).detach()

            bs_src = feats_src.size(0)
            bs_tgt = feats_tgt.size(0)

            d_inputs = torch.cat([feats_src, feats_tgt], dim=0)
            d_labels = torch.cat([
                torch.ones(bs_src, 1),
                torch.zeros(bs_tgt, 1)
            ], dim=0).to(device)

            preds_d = model.domain_discriminator(d_inputs)
            loss_d = criterion(preds_d, d_labels)
            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()

            # Ft: Generador adversarial
            f_tgt = model.Ft(xt)
            preds_t = model.domain_discriminator(f_tgt)
            loss_t = criterion(preds_t, torch.ones_like(preds_t))
            loss_t = loss_t

            optimizer_Ft.zero_grad()
            loss_t.backward()
            optimizer_Ft.step()

            total_domain_loss += loss_d.item()
            total_target_loss += loss_t.item()
            dom_correct += ((preds_d > 0.5).float() == d_labels).sum().item()
            total_dom += d_labels.size(0)

        val_loss_src, val_acc_src,_ = eval_model(model.Fs, model.classifier, val_loader_src, device)
        val_loss_tgt, val_acc_tgt,_ = eval_model(model.Ft, model.classifier, val_loader_tgt, device)

        print(f"Epoch {epoch+1}: LR = {new_lr:.6f}, Lambda = {lambda_val:.4f}")
        print(f"  Domain Loss = {total_domain_loss:.4f}, Ft Loss = {total_target_loss:.4f}, "
              f"Domain Acc = {100. * dom_correct / total_dom:.2f}%")
        print(f"  Val Src: Loss = {val_loss_src:.4f}, Acc = {val_acc_src:.2f}%")
        print(f"  Val Tgt: Loss = {val_loss_tgt:.4f}, Acc = {val_acc_tgt:.2f}%")
# --- CDAN Entrenamiento ---
def train_cdan(model, src_loader, tgt_loader, val_loader_src, val_loader_tgt, device,num_classes,alpha,delta, epochs=5, eta_0=1e-3):
    set_seed(42)
    optimizer = optim.Adam(model.parameters(), lr=eta_0)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        print(f"[CDAN+E] => Epoch {epoch+1}/{epochs} - INICIO")

        model.train()
        lambda_val = get_lambda(epoch, epochs,delta=delta)
        if alpha!=None:
            new_lr = get_eta(epoch, epochs,alpha, eta_0)
            for param_group in optimizer.param_groups:
               param_group['lr'] = new_lr
        else:
            new_lr=0

        total_cls_loss, total_dom_loss, total_ent_loss = 0, 0, 0
        cls_correct, dom_correct, total_dom = 0, 0, 0
        tgt_iter = iter(tgt_loader)

        for xs, ys in tqdm(src_loader, desc=f"[CDAN+E] Epoch {epoch+1}"):
            try:
                xt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt, _ = next(tgt_iter)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            feats_src = model.feature(xs)
            preds_src = model.classifier(feats_src)
            loss_class = criterion_class(preds_src, ys)
            cls_correct += (preds_src.argmax(1) == ys).sum().item()

            feats_tgt = model.feature(xt)
            preds_tgt = model.classifier(feats_tgt)

            # Forward
            feats = torch.cat([feats_src, feats_tgt], dim=0)
            softmax_out = nn.Softmax(dim=1)(torch.cat([preds_src, preds_tgt], dim=0))
            bs_src = feats_src.size(0)
            bs_tgt = feats_tgt.size(0)
            joint = torch.bmm(
                softmax_out.unsqueeze(2),
                feats.unsqueeze(1)
            ).view(-1, feats.size(1) * softmax_out.size(1))

            domain_preds = model.domain_discriminator(joint)
            domain_labels = torch.cat([
                torch.ones(bs_src, 1),
                torch.zeros(bs_tgt, 1)
            ], dim=0).to(device)

            # Entropy por muestra y pesos
            ent_all = entropy(softmax_out)
            ent_all.register_hook(grl_hook(lambda_val))

            weights = 1.0 + torch.exp(-ent_all)
            weights = weights / weights.sum().detach()

            # Pérdida de dominio ponderada
            domain_loss_raw = nn.BCEWithLogitsLoss(reduction='none')(domain_preds, domain_labels)
            loss_domain = torch.sum(weights * domain_loss_raw)

            dom_correct += ((domain_preds > 0.5).float() == domain_labels).sum().item()
            total_dom += domain_labels.size(0)

            # --- Entropy loss (only on target predictions) ---
            softmax_tgt = nn.Softmax(dim=1)(preds_tgt)
            loss_entropy = torch.mean(entropy(softmax_tgt))

            # --- Total loss (with entropy regularization) --
            loss = loss_class + lambda_val*loss_domain

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cls_loss += loss_class.item()
            total_dom_loss += loss_domain.item()
            total_ent_loss += loss_entropy.item()


        val_loss_src, val_acc_src,_ = eval_model(model.feature, model.classifier, val_loader_src, device)
        val_loss_tgt, val_acc_tgt,_ = eval_model(model.feature, model.classifier, val_loader_tgt, device)

        print(f"Epoch {epoch+1}: LR={new_lr:.6f}, Lambda={lambda_val:.4f}")
        print(f"  Cls Loss={total_cls_loss:.4f}, Dom Loss={total_dom_loss:.4f}, Ent Loss={total_ent_loss:.4f}, "
              f"Cls Acc={100. * cls_correct / len(src_loader.dataset):.2f}%, "
              f"Dom Acc={100. * dom_correct / total_dom:.2f}%")
        print(f"  Val Src: Loss={val_loss_src:.4f}, Acc={val_acc_src:.2f}%, "
              f"Val Tgt: Loss={val_loss_tgt:.4f}, Acc={val_acc_tgt:.2f}%")
def train_creda(model, src_loader, tgt_loader, val_loader_src, val_loader_tgt, device,lambda_, sigma,alpha,delta, epochs=5, eta_0=1e-3):
    import torch.nn.functional as Fu
    set_seed(42)
    optimizer = optim.Adam(model.parameters(), lr=eta_0)
    criterion_class = nn.CrossEntropyLoss()
    creda_loss_fn = CREDALoss(sigma=sigma,lambda_creda=lambda_, use_entropy_weighting=True)

    for epoch in range(epochs):
        print(f"[CREDA] => Epoch {epoch+1}/{epochs} - INICIO")

        model.train()
        lambda_val = get_lambda(epoch, epochs,delta=delta)
        if alpha!=None:
            new_lr = get_eta(epoch, epochs,alpha, eta_0)
            for param_group in optimizer.param_groups:
               param_group['lr'] = new_lr
        else:
            new_lr=0

        total_cls_loss, total_creda_loss = 0.0, 0.0
        cls_correct = 0
        tgt_iter = iter(tgt_loader)

        for xs, ys in tqdm(src_loader, desc=f"[CREDA] Epoch {epoch+1}"):
            try:
                xt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt, _ = next(tgt_iter)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            # Forward pass
            feats_src = model.forward(xs, mode='feature')
            preds_src = model.forward(xs, mode='class')
            feats_tgt = model.forward(xt, mode='feature')
            preds_tgt = model.forward(xt, mode='class')

            # Pérdida de clasificación
            loss_class = criterion_class(preds_src, ys)
            cls_correct += (preds_src.argmax(1) == ys).sum().item()

            # Pérdida de alineación CREDA
            softmax_tgt = Fu.softmax(preds_tgt, dim=1)
            loss_creda = creda_loss_fn(feats_src, feats_tgt, ys, softmax_tgt).to(device)

            # Pérdida total
            loss = loss_class + lambda_val * loss_creda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cls_loss += loss_class.item()
            total_creda_loss += loss_creda.item()

        val_loss_src, val_acc_src, _ = eval_model(model.feature, model.classifier, val_loader_src, device)
        val_loss_tgt, val_acc_tgt, _ = eval_model(model.feature, model.classifier, val_loader_tgt, device)

        print(f"Epoch {epoch+1}: LR={new_lr:.6f}, Lambda={lambda_val:.4f}")
        print(f"  Cls Loss={total_cls_loss:.4f}, CREDA Loss={total_creda_loss:.4f}, "
              f"Cls Acc={100. * cls_correct / len(src_loader.dataset):.2f}%")
        print(f"  Val Src: Loss={val_loss_src:.4f}, Acc={val_acc_src:.2f}%, "
              f"Val Tgt: Loss={val_loss_tgt:.4f}, Acc={val_acc_tgt:.2f}%")

def run_baseline(dataset_key, sets_dict, cfg, epochs=5, save=False, specific_pair=None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    models = {}

    pairs = [specific_pair] if specific_pair else permutations(sets_dict.keys(), 2)

    for src, tgt in pairs:
        print(f"\n[{dataset_key}] Baseline {src} → {tgt}")

        src_data = sets_dict[src]
        tgt_data = sets_dict[tgt]
        num_classes = get_num_classes(src_data)

        src_train, src_val = split_stratified(src_data, val_ratio=cfg["val_ratio_baseline"])

        train_bs = get_batch_size(src, len(src_train), cfg)
        train_loader = DataLoader(src_train, batch_size=train_bs, shuffle=True)
        val_loader = DataLoader(src_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_loader = DataLoader(tgt_data, batch_size=cfg["eval_batch_size"], shuffle=False)

        F = FeatureExtractor(backbone=cfg["backbone"], pretrained=True).to(device)
        C = Classifier(feature_dim=F.output_dim, num_classes=num_classes).to(device)

        train_baseline(
            F, C, train_loader, val_loader, device,
            epochs=epochs,
            num_classes=num_classes,
            lr=cfg["baseline_lr"]
        )

        acc, std = eval_accuracy_only(F, C, tgt_loader, device, num_classes)

        results.append({
            "Dataset": dataset_key + "_Baseline",
            "Source": src,
            "Target": tgt,
            "Test Accuracy": acc,
            "std": std
        })

        if save:
            models[(src, tgt)] = (F, C)
    return (pd.DataFrame(results), models) if save else pd.DataFrame(results)

def run_dann(dataset_key, sets_dict, cfg, epochs=5, save=False, specific_pair=None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    models = {}

    pairs = [specific_pair] if specific_pair else permutations(sets_dict.keys(), 2)

    for src, tgt in pairs:
        print(f"\n[{dataset_key}] DANN {src} → {tgt}")

        src_data = sets_dict[src]
        tgt_data = sets_dict[tgt]
        num_classes = get_num_classes(src_data)

        src_train, src_val, src_test = split_stratified(
            src_data,
            val_ratio=cfg["val_ratio_da"],
            test_ratio=cfg["test_ratio_da"]
        )
        tgt_train, tgt_val, tgt_test = split_stratified(
            tgt_data,
            val_ratio=cfg["val_ratio_da"],
            test_ratio=cfg["test_ratio_da"]
        )

        src_bs, tgt_bs = get_da_batch_size(src, tgt, len(src_train), len(tgt_train), cfg)

        src_train_loader = DataLoader(src_train, batch_size=src_bs, shuffle=True)
        tgt_train_loader = DataLoader(tgt_train, batch_size=tgt_bs, shuffle=True)
        src_val_loader = DataLoader(src_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_val_loader = DataLoader(tgt_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_test_loader = DataLoader(tgt_test, batch_size=cfg["eval_batch_size"], shuffle=False)

        model = DANN_ResNet(cfg["backbone"], num_classes=num_classes, lambda_grl=1.0).to(device)

        eta_0 = cfg["dann_lr_special"] if src in cfg["special_domains"] else cfg["dann_lr_default"]

        train_dann(
            model,
            src_train_loader,
            tgt_train_loader,
            src_val_loader,
            tgt_val_loader,
            device,
            num_classes,
            epochs=epochs,
            delta=cfg["delta"],
            alpha=cfg["alpha"],
            eta_0=eta_0
        )

        acc, std = eval_accuracy_only(model.feature, model.classifier, tgt_test_loader, device, num_classes)

        results.append({
            "Dataset": dataset_key + "_DANN",
            "Source": src,
            "Target": tgt,
            "Test Accuracy": acc,
            "std": std
        })

        if save:
            models[(src, tgt)] = model
    return (pd.DataFrame(results), models) if save else pd.DataFrame(results)
def run_adda(dataset_key, sets_dict, cfg, epochs_cl=5, epochs_dc=5, save=False, specific_pair=None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    pretrained_sources = {}
    models = {}

    pairs = [specific_pair] if specific_pair else permutations(sets_dict.keys(), 2)

    for src, tgt in pairs:
        print(f"\n[{dataset_key}] ADDA {src} → {tgt}")

        src_data = sets_dict[src]
        tgt_data = sets_dict[tgt]
        num_classes = get_num_classes(src_data)

        src_train, src_val, src_test = split_stratified(
            src_data,
            val_ratio=cfg["val_ratio_da"],
            test_ratio=cfg["test_ratio_da"]
        )
        tgt_train, tgt_val, tgt_test = split_stratified(
            tgt_data,
            val_ratio=cfg["val_ratio_da"],
            test_ratio=cfg["test_ratio_da"]
        )

        src_bs, tgt_bs = get_da_batch_size(src, tgt, len(src_train), len(tgt_train), cfg)

        src_train_loader = DataLoader(src_train, batch_size=src_bs, shuffle=True)
        tgt_train_loader = DataLoader(tgt_train, batch_size=tgt_bs, shuffle=True)
        src_val_loader = DataLoader(src_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_val_loader = DataLoader(tgt_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_test_loader = DataLoader(tgt_test, batch_size=cfg["eval_batch_size"], shuffle=False)

        model = ADDA_ResNet(cfg["backbone"], num_classes=num_classes).to(device)

        if src not in pretrained_sources:
            print("\nPhase 1: Entrenamiento supervisado en dominio fuente")
            train_adda_phase1(
                model, src_train_loader, src_val_loader, device,
                epochs=epochs_cl, num_classes=num_classes
            )
            pretrained_sources[src] = {
                "Fs": model.Fs.state_dict(),
                "C": model.classifier.state_dict()
            }
        else:
            print(f"\nPhase 1: Reutilizando pesos entrenados para {src}")
            model.Fs.load_state_dict(pretrained_sources[src]["Fs"])
            model.classifier.load_state_dict(pretrained_sources[src]["C"])

        print("\nPhase 2: Adaptación adversarial con Ft")
        model.Ft.load_state_dict(model.Fs.state_dict())

        eta_0 = cfg["adda_phase2_lr_special"] if src in cfg["special_domains"] else cfg["adda_phase2_lr_default"]

        train_adda_phase2(
            model,
            src_train_loader,
            tgt_train_loader,
            src_val_loader,
            tgt_val_loader,
            device,
            num_classes,
            epochs=epochs_dc,
            delta=cfg["delta"],
            alpha=cfg["alpha"],
            eta_0=eta_0
        )

        test_loss, test_acc, std = eval_model(model.Ft, model.classifier, tgt_test_loader, device)

        results.append({
            "Dataset": dataset_key + "_ADDA",
            "Source": src,
            "Target": tgt,
            "Test Accuracy": test_acc,
            "std": std
        })

        if save:
            models[(src, tgt)] = model
    return (pd.DataFrame(results), models) if save else pd.DataFrame(results)

def run_cdan(dataset_key, sets_dict, cfg, epochs=5, save=False, specific_pair=None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    models = {}

    pairs = [specific_pair] if specific_pair else permutations(sets_dict.keys(), 2)

    for src, tgt in pairs:
        print(f"\n[{dataset_key}] CDAN {src} → {tgt}")

        src_data = sets_dict[src]
        tgt_data = sets_dict[tgt]
        num_classes = get_num_classes(src_data)

        src_train, src_val, src_test = split_stratified(
            src_data,
            val_ratio=cfg["val_ratio_da"],
            test_ratio=cfg["test_ratio_da"]
        )
        tgt_train, tgt_val, tgt_test = split_stratified(
            tgt_data,
            val_ratio=cfg["val_ratio_da"],
            test_ratio=cfg["test_ratio_da"]
        )

        src_bs, tgt_bs = get_da_batch_size(src, tgt, len(src_train), len(tgt_train), cfg)

        src_train_loader = DataLoader(src_train, batch_size=src_bs, shuffle=True)
        tgt_train_loader = DataLoader(tgt_train, batch_size=tgt_bs, shuffle=True)
        src_val_loader = DataLoader(src_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_val_loader = DataLoader(tgt_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_test_loader = DataLoader(tgt_test, batch_size=cfg["eval_batch_size"], shuffle=False)

        model = CDAN_ResNet(cfg["backbone"], num_classes=num_classes).to(device)

        train_cdan(
            model,
            src_train_loader,
            tgt_train_loader,
            src_val_loader,
            tgt_val_loader,
            device,
            num_classes,
            epochs=epochs,
            delta=cfg["delta"],
            alpha=cfg["alpha"],
            eta_0=cfg["cdan_lr"]
        )

        test_loss, test_acc, std = eval_model(model.feature, model.classifier, tgt_test_loader, device)

        results.append({
            "Dataset": dataset_key + "_CDAN",
            "Source": src,
            "Target": tgt,
            "Test Accuracy": test_acc,
            "std": std
        })

        if save:
            models[(src, tgt)] = model
    return (pd.DataFrame(results), models) if save else pd.DataFrame(results)

def run_creda(dataset_key, sets_dict, cfg, epochs=5, save=False, specific_pair=None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    models = {}

    pairs = [specific_pair] if specific_pair else permutations(sets_dict.keys(), 2)

    for src, tgt in pairs:
        print(f"\n[{dataset_key}] CREDA {src} → {tgt}")

        src_data = sets_dict[src]
        tgt_data = sets_dict[tgt]
        num_classes = get_num_classes(src_data)

        src_train, src_val, src_test = split_stratified(
            src_data,
            val_ratio=cfg["val_ratio_da"],
            test_ratio=cfg["test_ratio_da"]
        )
        tgt_train, tgt_val, tgt_test = split_stratified(
            tgt_data,
            val_ratio=cfg["val_ratio_da"],
            test_ratio=cfg["test_ratio_da"]
        )

        src_bs, tgt_bs = get_da_batch_size(src, tgt, len(src_train), len(tgt_train), cfg)

        src_train_loader = DataLoader(src_train, batch_size=src_bs, shuffle=True)
        tgt_train_loader = DataLoader(tgt_train, batch_size=tgt_bs, shuffle=True)
        src_val_loader = DataLoader(src_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_val_loader = DataLoader(tgt_val, batch_size=cfg["val_batch_size"], shuffle=False)
        tgt_test_loader = DataLoader(tgt_test, batch_size=cfg["eval_batch_size"], shuffle=False)

        model = CREDA_ResNet(cfg["backbone"], num_classes=num_classes).to(device)

        eta_0 = cfg["creda_lr_special"] if src in cfg["special_domains"] else cfg["creda_lr_default"]
        lambda_ = cfg["creda_lambda_special"] if src in cfg["special_domains"] else cfg["creda_lambda_default"]

        train_creda(
            model,
            src_train_loader,
            tgt_train_loader,
            src_val_loader,
            tgt_val_loader,
            device,
            epochs=epochs,
            lambda_=lambda_,
            sigma='auto',
            delta=cfg["delta"],
            alpha=cfg["alpha"],
            eta_0=eta_0
        )

        test_loss, test_acc, std = eval_model(model.feature, model.classifier, tgt_test_loader, device)

        results.append({
            "Dataset": dataset_key + "_CREDA",
            "Source": src,
            "Target": tgt,
            "Test Accuracy": test_acc,
            "std": std
        })

        if save:
            models[(src, tgt)] = model
    return (pd.DataFrame(results), models) if save else pd.DataFrame(results)

def run_all_models(combinations_dict, sets_all, cfg, output_dir="saved_models", epochs=5):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for dataset_key, pair in combinations_dict.items():
        print(f"\n📦 Ejecutando modelos para: {dataset_key}")

        sets_dict = sets_all[dataset_key]

        if pair == (None, None):
            subset = sets_dict
            specific_pair = None
            print(" • Usando TODO el dataset.")
        else:
            src, tgt = pair
            subset = {src: sets_dict[src], tgt: sets_dict[tgt]}
            specific_pair = (src, tgt)
            print(f" • Subconjunto: {src} → {tgt}")

        df_baseline, models_baseline = run_baseline(
            dataset_key=dataset_key,
            sets_dict=subset,
            cfg=cfg,
            epochs=epochs,
            save=True,
            specific_pair=specific_pair
        )
        all_results.append(df_baseline)
        for (s, t), (F, C) in models_baseline.items():
            torch.save(F.state_dict(), os.path.join(output_dir, f"{cfg['backbone']}_Baseline_{s}_{t}_F.pth"))
            torch.save(C.state_dict(), os.path.join(output_dir, f"{cfg['backbone']}_Baseline_{s}_{t}_C.pth"))

        df_dann, models_dann = run_dann(
            dataset_key=dataset_key,
            sets_dict=subset,
            cfg=cfg,
            epochs=epochs,
            save=True,
            specific_pair=specific_pair
        )
        all_results.append(df_dann)
        for (s, t), model in models_dann.items():
            torch.save(model.state_dict(), os.path.join(output_dir, f"{cfg['backbone']}_DANN_{s}_{t}_weights.pth"))

        df_adda, models_adda = run_adda(
            dataset_key=dataset_key,
            sets_dict=subset,
            cfg=cfg,
            epochs_cl=epochs,
            epochs_dc=epochs,
            save=True,
            specific_pair=specific_pair
        )
        all_results.append(df_adda)
        for (s, t), model in models_adda.items():
            torch.save(model.state_dict(), os.path.join(output_dir, f"{cfg['backbone']}_ADDA_{s}_{t}_weights.pth"))

        df_cdan, models_cdan = run_cdan(
            dataset_key=dataset_key,
            sets_dict=subset,
            cfg=cfg,
            epochs=epochs,
            save=True,
            specific_pair=specific_pair
        )
        all_results.append(df_cdan)
        for (s, t), model in models_cdan.items():
            torch.save(model.state_dict(), os.path.join(output_dir, f"{cfg['backbone']}_CDAN_{s}_{t}_weights.pth"))

        df_creda, models_creda = run_creda(
            dataset_key=dataset_key,
            sets_dict=subset,
            cfg=cfg,
            epochs=epochs,
            save=True,
            specific_pair=specific_pair
        )
        all_results.append(df_creda)
        for (s, t), model in models_creda.items():
            torch.save(model.state_dict(), os.path.join(output_dir, f"{cfg['backbone']}_CREDA_{s}_{t}_weights.pth"))

        torch.cuda.empty_cache()

    return pd.concat(all_results, ignore_index=True)

def load_model(model_type, src, tgt, path="saved_models", num_classes=10):
    model_type = model_type.lower()

    if model_type == "baseline":
        F_model = FeatureExtractor(backbone='resnet50')
        C = Classifier(feature_dim=2048, num_classes=num_classes)

        F_model.load_state_dict(torch.load(f"{path}/Baseline_{src}_{tgt}_F.pth"))
        C.load_state_dict(torch.load(f"{path}/Baseline_{src}_{tgt}_C.pth"))

        F_model.eval()
        C.eval()
        return F_model, C

    elif model_type == "dann":
        model = DANN_ResNet(backbone='resnet50',num_classes=num_classes)

    elif model_type == "adda":
        model = ADDA_ResNet(backbone='resnet50',num_classes=num_classes)

    elif model_type == "cdan":
        model = CDAN_ResNet(backbone='resnet50',num_classes=num_classes)

    elif model_type == "creda":
        model = CREDA_ResNet(backbone='resnet50',num_classes=num_classes)

    else:
        raise ValueError(f"Modelo no reconocido: {model_type}")

    model.load_state_dict(torch.load(f"{path}/{model_type.upper()}_{src}_{tgt}_weights.pth"))
    model.eval()
    return model

def extract_features_model(model, dataloader, domain="source", device="cuda"):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            if isinstance(model, ADDA_ResNet):
                feats = model.Fs(x) if domain == "source" else model.Ft(x)
            else:
                feats = model.feature(x)

            features.append(feats.cpu())
            labels.append(y)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def extract_features_baseline(F_model, dataloader, device="cuda"):
    F_model.eval()
    features, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            feats = F_model(x)
            features.append(feats.cpu())
            labels.append(y)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def extract_flattened_features(dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    X, y = [], []

    for imgs, labels in loader:
        imgs_flat = imgs.view(imgs.size(0), -1).numpy()
        X.append(imgs_flat)
        y.append(labels.numpy())

    return np.vstack(X), np.hstack(y)

def resize_image_pil(img, target_shape=(28, 28)):
    img = np.array(img.convert("RGB"))
    resized_img = resize(img, target_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return resized_img
def tensor_to_imgarray(image, target_shape=(32, 32)):
    if not torch.is_tensor(image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        tensor = image.clone().cpu().clamp(0, 1)
        img_pil = to_pil_image(tensor)
        img_rgb = np.array(img_pil.convert("RGB"))

    resized = resize(
        img_rgb,
        target_shape,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.uint8)

    return resized


def pil_to_imgarray(pil_image, target_shape=(64, 64)):
    img_rgb = np.array(pil_image.convert("RGB"))
    resized = resize(
        img_rgb,
        target_shape,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.uint8)

    return resized


def get_one_sample_per_selected_classes(dataset, class_list):
    samples = []
    found = {cls: False for cls in class_list}

    for i in range(len(dataset)):
        x, y = dataset[i]
        y_int = int(y)

        if y_int in class_list and not found[y_int]:
            samples.append((class_list.index(y_int), x))
            found[y_int] = True

        if all(found.values()):
            break

    return [x for _, x in sorted(samples, key=lambda x: x[0])]


def get_representative_per_class(dataset):
    class_to_img = {}

    for path, label in dataset.samples:
        if label not in class_to_img:
            img = dataset.loader(path)
            class_to_img[label] = img

        if len(class_to_img) == len(dataset.classes):
            break

    return [class_to_img[i] for i in range(len(dataset.classes))]


def show_digit_domains_grid(
    domains,
    datasets,
    class_list,
    target_shape=(32, 32),
    title="",
    only_class=None,
):
    n_domains = len(domains)

    if only_class is not None:
        if only_class not in class_list:
            raise ValueError(f"La clase {only_class} no está en class_list={class_list}")

        class_pos = class_list.index(only_class)

        fig, axs = plt.subplots(1, n_domains, figsize=(n_domains * 2.5, 2.5))
        if n_domains == 1:
            axs = [axs]

        for col_idx, (domain, dataset) in enumerate(zip(domains, datasets)):
            samples = get_one_sample_per_selected_classes(dataset, class_list)
            img = samples[class_pos]

            axs[col_idx].imshow(tensor_to_imgarray(img, target_shape))
            axs[col_idx].set_title(domain)
            axs[col_idx].axis("off")

        plt.suptitle(f"{title} — Clase: {only_class}")
    else:
        n_rows, n_cols = len(domains), len(class_list)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

        for row_idx, (domain, dataset) in enumerate(zip(domains, datasets)):
            samples = get_one_sample_per_selected_classes(dataset, class_list)

            for col_idx, img in enumerate(samples):
                ax = axs[row_idx, col_idx] if n_rows > 1 else axs[col_idx]
                ax.imshow(tensor_to_imgarray(img, target_shape))
                ax.axis("off")

                if col_idx == 0:
                    ax.set_ylabel(domain)

                if row_idx == n_rows - 1:
                    ax.set_xlabel(class_list[col_idx])

    plt.tight_layout()
    plt.savefig("digits_grid.pdf", bbox_inches="tight")
    plt.show()


def show_multi_domain_class_grid(
    domains_dict,
    target_shape=(224, 224),
    title="",
    only_class=None,
):
    domain_names = list(domains_dict.keys())
    example_dataset = list(domains_dict.values())[0]

    class_names = example_dataset.classes
    n_domains = len(domain_names)
    n_classes = len(class_names)

    if only_class is not None:
        if isinstance(only_class, str):
            class_idx = class_names.index(only_class)
        else:
            class_idx = only_class

        fig, axs = plt.subplots(1, n_domains, figsize=(n_domains * 3, 3))

        if n_domains == 1:
            axs = [axs]

        for i, domain in enumerate(domain_names):
            dataset = domains_dict[domain]
            samples = get_representative_per_class(dataset)

            axs[i].imshow(pil_to_imgarray(samples[class_idx], target_shape))
            axs[i].set_title(domain)
            axs[i].axis("off")

        plt.suptitle(f"{title} — Clase: {class_names[class_idx]}")
    else:
        fig, axs = plt.subplots(n_domains, n_classes, figsize=(n_classes * 2, n_domains * 2.5))

        for i, domain in enumerate(domain_names):
            dataset = domains_dict[domain]
            samples = get_representative_per_class(dataset)

            for j in range(n_classes):
                ax = axs[i, j]
                ax.imshow(pil_to_imgarray(samples[j], target_shape))
                ax.axis("off")

                if i == n_domains - 1:
                    ax.set_xlabel(class_names[j])

                if j == 0:
                    ax.set_ylabel(domain)

    plt.tight_layout()
    plt.savefig("multi_domain_grid.pdf", bbox_inches="tight")
    plt.show()

def extract_features_adaptive(model, dataloader, domain="source", device="cuda"):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            if isinstance(model, ADDA_ResNet):
                f = model.Fs(x) if domain == "source" else model.Ft(x)
            else:
                f = model.feature(x)
            features.append(f.cpu())
            labels.append(y)

    return torch.cat(features), torch.cat(labels)

def get_last_conv_layer(model):
    return [m for m in model.feature.modules() if isinstance(m, torch.nn.Conv2d)][-1]

def get_denormalizer(dataset_key):
    if dataset_key == "MNIST-USPS-SVHN":
        mean, std = [0.5]*3, [0.5]*3
    else:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    def denorm(t):
        for ch, m, s in zip(t, mean, std):
            ch.mul_(s).add_(m)
        return t.clamp(0, 1)
    return denorm

def norm_cam(cam):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam

def get_nth_image_for_class(dataloader, class_idx, index, device):
    """Devuelve la imagen N-ésima (index) de clase `class_idx` encontrada en el dataloader."""
    imgs = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        mask = (y == class_idx)
        if mask.any():
            imgs.extend(x[mask].detach().cpu())
        if len(imgs) > index:
            return imgs[index]
    return None  # No hay suficientes imágenes de esa clase



