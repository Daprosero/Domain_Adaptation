"""Phase 1 under the same conditions with two instance encoders: the spectral one and frozen REVE.
Same windows (1-45 Hz, CAR, at most 60 per bag), per-domain standardization, explicit L2 .01,
lr1 3e-3, 200 epochs. Unseen source test AUC and BA, pooled over the 4 rotations, per seed.
Usage: encoder_phase1.py <seeds comma-separated> <transfer> ... (the source of each transfer)."""
import json, sys, warnings
sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, reve_trunk, spectral_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
torch.set_num_threads(4)
run = json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != 'remoteInputs'}, 'device': 'cpu'})
campaign.register(cfg['strategies'])
bags = bag_data.load_bags(None, None); dev = torch.device('cpu')
hyper = two_phase.check_hyper({**sc.HYPER, 'bagsPerStep': cfg['schedule']['bagsPerStep']})
ENCODERS = {'spectral': spectral_trunk.installed, 'REVE frozen': reve_trunk.installed}
seeds = [int(s) for s in sys.argv[1].split(',')]
for name in sys.argv[2:]:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    for seed in seeds:
        out = {}
        for enc, install in ENCODERS.items():
            y, p = [], []
            for r in range(4):
                side = lambda role: sc.prepared(two_phase.bag_tensors(A.side_bags(a, bags, 'source', r, role)))
                train, test = side(A.REFIT_ROLE), side('test')
                with install():
                    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
                    c = int(train[0].windows.shape[1])
                    m = model_store.build_model('MIL-CREDA', c, c, hyper)
                    m.source_trunk.norm = sc.standard_of(m.source_trunk, train)
                    with explicit_l2(sc.L2):
                        two_phase.train_source_phase(m, train, [], hyper, g, dev, seed=seed, fold=r, transfer=name)
                    y += [b.label for b in test]
                    p += np.nan_to_num(two_phase.evaluate(m, test, 'source', dev)[:, 1].detach().numpy(), nan=.5).tolist()
            out[enc] = {'AUC': round(AUC(y, p), 3), 'BA': round(BA(y, np.asarray(p) > .5), 3)}
        print('RESULT', name.split('->')[0], 'seed', seed, json.dumps(out), flush=True)
