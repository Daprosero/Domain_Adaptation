# VENDORED UPSTREAM CODE -- copied verbatim from the paper's official repository.
# Paper: Liang, Hu, Feng. Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. ICML 2020. arXiv:2002.08546
# Repository: https://github.com/tim-learn/SHOT
# Commit: f7d555a0d53b525b885e5ef2a887a267a5be3c36
# Upstream path: object/image_target.py (excerpts: L5-7, L15, L190-197, L199-206, L242-297)
# License as found: MIT (LICENSE at repo root)
# Copied with the operator's explicit authorisation, cited here and in ../CITATIONS.md.
# Every deviation from the upstream bytes is marked with a `# LOCAL EDIT:` comment;
# all other lines (including their indentation) are unchanged.

# LOCAL EDIT: import block reduced to what the excerpts below use (upstream L5-7, L15, L10).
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from . import upstream_loss as loss  # LOCAL EDIT: upstream L10 was `import network, loss`


# LOCAL EDIT: wrapper def; body is upstream L190-197 (inside train_target's loop), indentation kept.
def pseudo_label_loss(outputs_test, mem_label, tar_idx, iter_num, interval_iter, args):
        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            # LOCAL EDIT: device-agnostic, no CUDA (upstream L197 was: classifier_loss = torch.tensor(0.0).cuda())
            classifier_loss = torch.tensor(0.0, device=outputs_test.device)
        return classifier_loss  # LOCAL EDIT: return added


# LOCAL EDIT: wrapper def; body is upstream L199-206 (inside train_target's loop), indentation kept.
def im_loss_fn(outputs_test, args):
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
        return im_loss  # LOCAL EDIT: return added


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            # LOCAL EDIT: Python 3 iterator protocol (upstream L247 was: data = iter_test.next())
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            # LOCAL EDIT: device-agnostic, no CUDA (upstream L250 was: inputs = inputs.cuda())
            inputs = inputs
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')
