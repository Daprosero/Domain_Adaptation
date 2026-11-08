"""MIL-CREDA phase 2 on a frozen REVE instance encoder, scored by balanced accuracy (the plan's
primary metric) and AUC on unseen target test subjects. REVE embeds any montage in one 512-value
space, so predict exists on the device pair too: the source-trained projection, attention and head
applied to the target windows (read through the target trunk right after the bridge, which copies
the channel-independent projection). Predict is read with the source standardization and with the
target's own (label-free). Phase 2: the default (both terms, lr2 1e-3, 20 epochs, balanced
pseudo-labels) and the gentle one (global term only, lr2 1e-4, blended .5 with the start). Every
target-side statistic comes from the unlabelled target training subjects; target labels only score.
"""
import copy, json, sys, warnings
sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, reve_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
from MIL_CREDA_EEG.phase_two_terms import balanced_pseudolabels, blended
torch.set_num_threads(4)
run = json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != 'remoteInputs'}, 'device': 'cpu'})
campaign.register(cfg['strategies'])
bags = bag_data.load_bags(None, None); dev = torch.device('cpu')
hyper = two_phase.check_hyper({**sc.HYPER, 'bagsPerStep': cfg['schedule']['bagsPerStep']})
PHASE2 = {'MIL-CREDA default': (1e-3, 'both terms', None), 'MIL-CREDA gentle': (1e-4, 'global term only', 0.5)}
prob = lambda m, b: np.nan_to_num(two_phase.evaluate(m, b, 'target', dev)[:, 1].detach().numpy(), nan=.5)

seeds = [int(s) for s in sys.argv[1].split(',')]
for name in sys.argv[2:]:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    for seed in seeds:
        Y, P = [], {}
        for r in range(4):
            side = lambda s, role: sc.prepared(two_phase.bag_tensors(A.side_bags(a, bags, s, r, role)))
            src, tgt, test = side('source', A.REFIT_ROLE), side('target', A.REFIT_ROLE), side('target', 'test')
            Y += [b.label for b in test]
            with reve_trunk.installed():
                torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
                m = model_store.build_model('MIL-CREDA', int(src[0].windows.shape[1]), int(tgt[0].windows.shape[1]), hyper)
                m.source_trunk.norm = sc.standard_of(m.source_trunk, src)
                with explicit_l2(sc.L2):
                    ph1 = two_phase.train_source_phase(m, src, [], hyper, g, dev, seed=seed, fold=r, transfer=name)
                med = two_phase.within_bag_median_distance(m, src, 'source', dev)
                m._sigma = hyper['sigmaMultiplier'] * med if med > 0 else m._sigma
                ph1['checkpoint']['sigma'] = m._sigma
                j = two_phase.bridge(m)
                m.target_trunk.norm = m.source_trunk.norm
                P.setdefault('predict, source standardization', []).extend(prob(m, test))
                target_norm = sc.standard_of(m.target_trunk, tgt)
                m.target_trunk.norm = target_norm
                P.setdefault('predict, target standardization', []).extend(prob(m, test))
                for label, (lr2, terms, alpha) in PHASE2.items():
                    mm = copy.deepcopy(m)
                    start = {k: v.detach().clone() for k, v in mm.target_trunk.state_dict().items()}
                    h2 = two_phase.check_hyper({**hyper, 'lr2': lr2, 'epochs2': 20})
                    with balanced_pseudolabels(mm):
                        two_phase.adapt_target_phase(mm, ph1, j['frozen_state'], src, tgt, [], h2,
                                                     torch.Generator().manual_seed(seed), dev, seed=seed,
                                                     fold=r, transfer=name, term_selection=terms)
                    if alpha is not None:
                        adapted = {k: v.detach().clone() for k, v in mm.target_trunk.state_dict().items()}
                        mm.target_trunk.load_state_dict(blended(start, adapted, alpha))
                    P.setdefault(label, []).extend(prob(mm, test))
        out = {k: {'BA': round(BA(Y, np.asarray(p) > .5), 3), 'AUC': round(AUC(Y, p), 3)} for k, p in P.items()}
        print('RESULT', name, 'seed', seed, json.dumps(out), flush=True)
