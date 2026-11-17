"""Phase 1 neutral, MIL-CREDA whole in phase 2 (spectral encoder).

Phase 1 uses none of MIL-CREDA's aggregation: no consensus (gamma 0) and a flat softmax
(temperature 10), so predict owes nothing to the proposal. Phase 2 switches the proposal's
aggregation on (consensus gamma .5, temperature 1, Eqs. 15-16) for source and target alike, and
then trains the adaptation terms. Read on the unseen target test (BA, the primary metric, and
AUC): predict (phase 1 as it is); predict with the proposal's aggregation switched on and nothing
trained (what the aggregation alone changes); MIL-CREDA default (target standardization, both
terms, lr2 1e-3, 20 epochs); MIL-CREDA gentle (global term only, lr2 1e-4, blended .5).
Target labels only score.
"""
import copy, json, sys, warnings
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
GAMMA, TEMPERATURE = 0.5, 1.0
prob = lambda m, b, side: np.nan_to_num(two_phase.evaluate(m, b, side, dev)[:, 1].detach().numpy(), nan=.5)

seeds = [int(s) for s in sys.argv[1].split(',')]
for name in sys.argv[2:]:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    for seed in seeds:
        Y, P = [], {}
        for r in range(4):
            side = lambda s, role: sc.prepared(two_phase.bag_tensors(A.side_bags(a, bags, s, r, role)))
            src, tgt, test = side('source', A.REFIT_ROLE), side('target', A.REFIT_ROLE), side('target', 'test')
            Y += [b.label for b in test]
            same = src[0].windows.shape[1] == tgt[0].windows.shape[1]
            with spectral_trunk.installed():
                torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
                m = model_store.build_model('MIL-CREDA', int(src[0].windows.shape[1]), int(tgt[0].windows.shape[1]), hyper)
                m.source_trunk.norm = sc.standard_of(m.source_trunk, src)
                m.target_trunk.norm = m.source_trunk.norm if same else sc.standard_of(m.target_trunk, tgt)
                with explicit_l2(sc.L2):
                    ph1 = two_phase.train_source_phase(m, src, [], hyper, g, dev, seed=seed, fold=r, transfer=name)
                if same:
                    P.setdefault('predict (phase 1 as it is)', []).extend(prob(m, test, 'source'))
                m.attention_gamma, m.attention_temperature = GAMMA, TEMPERATURE      # MIL-CREDA on
                if same:
                    P.setdefault('predict + proposal aggregation, nothing trained', []).extend(prob(m, test, 'source'))
                med = two_phase.within_bag_median_distance(m, src, 'source', dev)
                m._sigma = hyper['sigmaMultiplier'] * med if med > 0 else m._sigma
                ph1['checkpoint']['sigma'] = m._sigma
                j = two_phase.bridge(m)
                for label, (lr2, terms, alpha, target_std) in {
                        'MIL-CREDA default': (1e-3, 'both terms', None, True),
                        'MIL-CREDA gentle': (1e-4, 'global term only', 0.5, False)}.items():
                    mm = copy.deepcopy(m)
                    if target_std or not same:
                        mm.target_trunk.norm = sc.standard_of(mm.target_trunk, tgt)
                    start = {k: v.detach().clone() for k, v in mm.target_trunk.state_dict().items()}
                    h2 = two_phase.check_hyper({**hyper, 'lr2': lr2, 'epochs2': 20})
                    with balanced_pseudolabels(mm):
                        two_phase.adapt_target_phase(mm, ph1, j['frozen_state'], src, tgt, [], h2,
                                                     torch.Generator().manual_seed(seed), dev, seed=seed,
                                                     fold=r, transfer=name, term_selection=terms)
                    if alpha is not None:
                        adapted = {k: v.detach().clone() for k, v in mm.target_trunk.state_dict().items()}
                        mm.target_trunk.load_state_dict(blended(start, adapted, alpha))
                    P.setdefault(label, []).extend(prob(mm, test, 'target'))
        out = {k: {'BA': round(BA(Y, np.asarray(p) > .5), 3), 'AUC': round(AUC(Y, p), 3)} for k, p in P.items()}
        print('RESULT', name, 'seed', seed, json.dumps(out), flush=True)
