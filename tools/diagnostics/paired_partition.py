"""A paired partition, as a candidate protocol change (it is not plan v60): in rotation r the test
subjects are every subject of fold r that has a target recording, never seen in training; the
training subjects (every other fold) contribute their source recording with its label AND their
target recording without it. Each domain then trains on about twice the subjects of the
subject-disjoint plan. Spectral encoder, the plan's folds. Scored on the unseen target test:
predict (the phase-1 model as it is) and MIL-CREDA phase 2 (default: target standardization, both
terms, lr2 1e-3; gentle: source standardization kept, global term only, lr2 1e-4, blend .5).
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


def recordings(subjects, condition):
    out = []
    for s in sorted(subjects):
        try:
            out.append(bags.get(s, condition))
        except Exception:
            pass
    return sc.prepared(two_phase.bag_tensors(out))


def run_transfer(name, seed):
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    cs = a.conditions[next(iter(a.subjects('source')))]
    ct = a.conditions[next(iter(a.subjects('target')))]
    everyone = set(a.folds)
    Y, P, sizes = [], {}, []
    for r in range(4):
        train_ids = {s for s in everyone if a.folds[s] != r}
        test_ids = {s for s in everyone if a.folds[s] == r}
        src, tgt, test = recordings(train_ids, cs), recordings(train_ids, ct), recordings(test_ids, ct)
        sizes.append((len(src), len(tgt), len(test)))
        Y += [b.label for b in test]
        with spectral_trunk.installed():
            torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
            m = model_store.build_model('MIL-CREDA', int(src[0].windows.shape[1]), int(tgt[0].windows.shape[1]), hyper)
            m.source_trunk.norm = sc.standard_of(m.source_trunk, src)
            same = src[0].windows.shape[1] == tgt[0].windows.shape[1]
            m.target_trunk.norm = m.source_trunk.norm if same else sc.standard_of(m.target_trunk, tgt)
            with explicit_l2(sc.L2):
                ph1 = two_phase.train_source_phase(m, src, [], hyper, g, dev, seed=seed, fold=r, transfer=name)
            med = two_phase.within_bag_median_distance(m, src, 'source', dev)
            m._sigma = hyper['sigmaMultiplier'] * med if med > 0 else m._sigma
            ph1['checkpoint']['sigma'] = m._sigma
            if same:
                P.setdefault('predict', []).extend(
                    np.nan_to_num(two_phase.evaluate(m, test, 'source', dev)[:, 1].detach().numpy(), nan=.5))
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
                P.setdefault(label, []).extend(
                    np.nan_to_num(two_phase.evaluate(mm, test, 'target', dev)[:, 1].detach().numpy(), nan=.5))
    out = {k: {'BA': round(BA(Y, np.asarray(p) > .5), 3), 'AUC': round(AUC(Y, p), 3)} for k, p in P.items()}
    return out, sizes


seeds = [int(s) for s in sys.argv[1].split(',')]
for name in sys.argv[2:]:
    for seed in seeds:
        out, sizes = run_transfer(name, seed)
        print('RESULT', name, 'seed', seed, 'sizes(src,tgt,test)', sizes, json.dumps(out), flush=True)
