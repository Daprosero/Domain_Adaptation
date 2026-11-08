"""Adapt the decision, not the trunk: label-free adaptation of the head on the target, scored by
balanced accuracy (the plan's primary metric) on unseen target test subjects.

After phase 1 (spectral recipe), the bag representations of the unlabelled target TRAINING
subjects are computed with the frozen source trunk (same channels: the predict path) and the head
is trained on them with information maximization (SHOT, Liang et al. 2020): low conditional
entropy per subject plus the KL of the mean prediction to the source class prior (known from the
source labels). Variants: the bias only (a shift of the decision) or the whole head. The trunk,
attention and sigma never move. Target labels only score; the best threshold line is diagnostic.
"""
import copy, json, sys, warnings
sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, spectral_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
torch.set_num_threads(4)
run = json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != 'remoteInputs'}, 'device': 'cpu'})
campaign.register(cfg['strategies'])
bags = bag_data.load_bags(None, None); dev = torch.device('cpu')
hyper = two_phase.check_hyper({**sc.HYPER, 'bagsPerStep': cfg['schedule']['bagsPerStep']})
STEPS, LR = 200, 1e-2


def reps(m, bs):
    with torch.no_grad():
        m.eval()
        return torch.cat([m.represent(m.bag_pairs(m.embed([b.windows], 'source'))) for b in bs])


def im_adapt(head, Z, prior, bias_only, lam=1.0):
    head = copy.deepcopy(head)
    params = [head.linear.bias] if bias_only else list(head.parameters())
    for p in head.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=LR)
    q = torch.tensor([1 - prior, prior], dtype=Z.dtype)
    for _ in range(STEPS):
        p = head(Z).clamp_min(1e-8)
        ent = -(p * p.log()).sum(1).mean()
        mean = p.mean(0)
        kl = (mean * (mean / q).log()).sum()
        loss = ent + lam * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if not bias_only:
            head.constrain()
    return head


seed = int(sys.argv[1])
for name in sys.argv[2:]:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    Y, P = [], {}
    for r in range(4):
        side = lambda s, role: sc.prepared(two_phase.bag_tensors(A.side_bags(a, bags, s, r, role)))
        src, tgt, test = side('source', A.REFIT_ROLE), side('target', A.REFIT_ROLE), side('target', 'test')
        prior = float(np.mean([b.label for b in src])); Y += [b.label for b in test]
        with spectral_trunk.installed():
            torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
            c = int(src[0].windows.shape[1])
            m = model_store.build_model('MIL-CREDA', c, c, hyper)
            m.source_trunk.norm = sc.standard_of(m.source_trunk, src)
            with explicit_l2(sc.L2):
                two_phase.train_source_phase(m, src, [], hyper, g, dev, seed=seed, fold=r, transfer=name)
            Zt, Zs = reps(m, tgt), reps(m, test)
        with torch.no_grad():
            P.setdefault('predict', []).extend(m.head(Zs)[:, 1].tolist())
        for label, bias_only in (('IM, bias only', True), ('IM, whole head', False)):
            h = im_adapt(m.head, Zt, prior, bias_only)
            with torch.no_grad():
                P.setdefault(label, []).extend(h(Zs)[:, 1].tolist())
    out = {}
    for k, p in P.items():
        p = np.nan_to_num(np.asarray(p), nan=.5)
        out[k] = {'BA': round(BA(Y, p > .5), 3), 'AUC': round(AUC(Y, p), 3),
                  'best threshold BA (test labels, diagnostic)': round(max(BA(Y, p > t) for t in np.unique(p)), 3)}
    print('RESULT seed', seed, name, json.dumps(out), flush=True)
