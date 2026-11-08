"""Label-free decision thresholds on the target, scored by balanced accuracy (the plan's primary
metric) on unseen target test subjects. Every threshold is fitted on the unlabelled target
TRAINING subjects and applied to the target test subjects; target labels only score.

Rules: 0.5 (as is); source prior (the share of the source's positive training subjects, from
source labels, is the share called positive on the target); EM prior adjustment (Saerens et al.
2002) on the target training posteriors; median split. Models: predict (source model, same
channels only) and MIL-CREDA phase 2 (target keeps the source standardization when channels
match, global term only, lr2 1e-4, 20 epochs, blended .5 with the start).
"""
import json, sys, warnings
sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, spectral_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
from MIL_CREDA_EEG.phase_two_terms import balanced_pseudolabels, blended
torch.set_num_threads(4)
run = json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != 'remoteInputs'}, 'device': 'cpu'})
campaign.register(cfg['strategies'])
bags = bag_data.load_bags(None, None); dev = torch.device('cpu')
hyper = two_phase.check_hyper({**sc.HYPER, 'bagsPerStep': cfg['schedule']['bagsPerStep']})


def em_prior(p, prior_s, iters=100):
    """Saerens et al. 2002: the target prior that makes the adjusted posteriors self-consistent."""
    pi = prior_s
    for _ in range(iters):
        a = p * pi / prior_s; b = (1 - p) * (1 - pi) / (1 - prior_s)
        pi = float(np.mean(a / (a + b)))
    return pi


def adjust(p, prior_s, pi):
    a = p * pi / prior_s; b = (1 - p) * (1 - pi) / (1 - prior_s)
    return a / (a + b)


def decisions(train_p, test_p, prior_s):
    train_p, test_p = np.asarray(train_p), np.asarray(test_p)
    pi = em_prior(train_p, prior_s)
    return {'0.5': test_p > 0.5,
            'source prior': test_p > np.quantile(train_p, 1 - prior_s),
            'EM prior': adjust(test_p, prior_s, pi) > 0.5,
            'median': test_p > np.median(train_p)}


prob = lambda m, b, side: np.nan_to_num(two_phase.evaluate(m, b, side, dev)[:, 1].detach().numpy(), nan=.5)
seed = int(sys.argv[1])
for name in sys.argv[2:]:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    Y, D, S = [], {}, {}
    for r in range(4):
        side = lambda s, role: sc.prepared(two_phase.bag_tensors(A.side_bags(a, bags, s, r, role)))
        src, tgt, test = side('source', A.REFIT_ROLE), side('target', A.REFIT_ROLE), side('target', 'test')
        prior_s = float(np.mean([b.label for b in src]))
        Y += [b.label for b in test]
        same = src[0].windows.shape[1] == tgt[0].windows.shape[1]
        with spectral_trunk.installed():
            torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
            m = model_store.build_model('MIL-CREDA', int(src[0].windows.shape[1]), int(tgt[0].windows.shape[1]), hyper)
            m.source_trunk.norm = sc.standard_of(m.source_trunk, src)
            m.target_trunk.norm = m.source_trunk.norm if same else sc.standard_of(m.target_trunk, tgt)
            with explicit_l2(sc.L2):
                ph1 = two_phase.train_source_phase(m, src, [], hyper, g, dev, seed=seed, fold=r, transfer=name)
            med = two_phase.within_bag_median_distance(m, src, 'source', dev)
            m._sigma = hyper['sigmaMultiplier'] * med if med > 0 else m._sigma; ph1['checkpoint']['sigma'] = m._sigma
            outs = {}
            if same:
                outs['predict'] = (prob(m, tgt, 'source'), prob(m, test, 'source'))
            j = two_phase.bridge(m); start = {k: v.detach().clone() for k, v in m.target_trunk.state_dict().items()}
            h2 = two_phase.check_hyper({**hyper, 'lr2': 1e-4, 'epochs2': 20})
            with balanced_pseudolabels(m):
                two_phase.adapt_target_phase(m, ph1, j['frozen_state'], src, tgt, [], h2, torch.Generator().manual_seed(seed),
                                             dev, seed=seed, fold=r, transfer=name, term_selection='global term only')
            adapted = {k: v.detach().clone() for k, v in m.target_trunk.state_dict().items()}
            m.target_trunk.load_state_dict(blended(start, adapted, 0.5))
            outs['MIL-CREDA'] = (prob(m, tgt, 'target'), prob(m, test, 'target'))
        for model, (tp, sp) in outs.items():
            S.setdefault(model, []).extend(sp.tolist())
            for rule, d in decisions(tp, sp, prior_s).items():
                D.setdefault((model, rule), []).extend(d.tolist())
    out = {f'{mdl} | {rule}': round(BA(Y, d), 3) for (mdl, rule), d in D.items()}
    out.update({f'{mdl} AUC': round(AUC(Y, s), 3) for mdl, s in S.items()})
    # Diagnostic only: the best balanced accuracy any threshold could reach, chosen on the test labels.
    out.update({f'{mdl} | best threshold (test labels, diagnostic)': round(max(BA(Y, np.asarray(s) > t) for t in np.unique(s)), 3)
                for mdl, s in S.items()})
    print('RESULT seed', seed, name, json.dumps(out), flush=True)
