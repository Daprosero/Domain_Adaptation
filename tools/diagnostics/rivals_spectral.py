"""The 8 rival UDA strategies of plan v60 with the spectral instance encoder. Diagnostic only.

Same conditions as `full_flow.py` (MIL-CREDA): the v59-k3 run config, 1-45 Hz, CAR, at most 60
windows per bag (`spectral_check.prepared`), the spectral trunk (`spectral_trunk.installed`), both
trunks standardized by the SOURCE training windows (no stage 0), explicit L2 .01 and lr1 3e-3 in
phase 1, the device pair restricted to the 12 shared channels. Phase 1 per group: the common
wrapper (UDA-DDA, CCA-LSC, CASUAL, COT, CS+CCS, CLUST, CLUDA) and AdaptMIL's native ABMIL, each
trained once per (transfer, seed, rotation) through `search.train_phase_one`; phase 2 through each
rival's own registered trainer (`search.train_phase_two`) with no target validation bags, so no
target label is read. Neutral aggregation: attention temperature 10 (spectral_check.HYPER) for the
common wrapper and AdaptMIL's ABMIL, in every phase. Every change is a patch inside this script.

Usage: rivals_spectral.py <seeds,> <epochs1> <epochs2> <dump.jsonl> <transfer> [<transfer> ...]
Environment: RIVALS=<comma-separated strategy names> restricts the strategies.
"""
import json, os, sys, time, traceback, warnings
sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, eegnet, mil_wrapper, search, spectral_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
torch.set_num_threads(3)
run = json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != 'remoteInputs'}, 'device': 'cpu'})
campaign.register(cfg['strategies'])
from MIL_CREDA_EEG.comparators.adaptmil import model as adaptmil_model, training as adaptmil_training
bags = bag_data.load_bags(None, None); dev = torch.device('cpu')
SHARED = ('F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8')
NEUTRAL_T = sc.HYPER['attentionTemperature']                          # 10: near-flat attention

# ---- patches (diagnostic only) ------------------------------------------------------------
# 1. Every spectral trunk is built with the SOURCE standardization of the current rotation, so a
#    rival's phase-2 trainer, which rebuilds the model and loads the phase-1 state, gets the same
#    norm (a buffer-only module, loaded from the state) and the target keeps the source's.
NORM = {'std': None}
_init = spectral_trunk.SpectralTrunk.__init__


def _trunk_init(self, channels, samples=eegnet.SAMPLES):
    _init(self, channels, samples)
    if NORM['std'] is not None:
        self.norm = sc.Standard(NORM['std'].mean.clone(), NORM['std'].std.clone())


def _trunk_reset(self):
    if hasattr(self.norm, 'reset_parameters'):
        self.norm.reset_parameters()
    self.project.reset_parameters()


spectral_trunk.SpectralTrunk.__init__ = _trunk_init
spectral_trunk.SpectralTrunk.reset_parameters = _trunk_reset
# 2. Neutral aggregation: the common wrapper's softmax temperature, and AdaptMIL's ABMIL scores
#    divided by the same temperature.
mil_wrapper.COMMON_TEMPERATURE = NEUTRAL_T
mil_wrapper.COMMON_PHASE_ONE_FIXED = {**mil_wrapper.COMMON_PHASE_ONE_FIXED, 'attentionTemperature': NEUTRAL_T}


def _abmil_weights(self, projected):
    if projected.ndim != 2 or projected.shape[0] == 0:
        raise ValueError("ABMIL requires a nonempty (instances, features) bag")
    return (self.w(torch.tanh(self.V(projected))).squeeze(-1) / NEUTRAL_T).softmax(dim=0)


adaptmil_model.NativeABMIL.weights = _abmil_weights
# 3. Explicit L2 for AdaptMIL's own phase-1 objective (explicit_l2 covers two_phase.source_term).
_stage_two = adaptmil_training.stage_two_objective


def _l2_stage_two(model, bags_, labels):
    w = [p for n, p in model.source_trunk.named_parameters() if 'weight' in n]
    return _stage_two(model, bags_, labels) + sc.L2 * sum((x ** 2).sum() for x in (*w, model.head.linear.weight))


# ---- hyperparameters ------------------------------------------------------------------------
SHARED2 = {'lr2': sc.HYPER['lr2'], 'wd2': sc.HYPER['wd2']}            # MIL-CREDA's, 1e-3 / 1e-5
PARAMS = {
    'AdaptMIL': {'mmdWeight': 1.0, 'kernelMultiplier': 1.0},
    'UDA-DDA': {'mmdWeight': 1.0, 'cmmdWeight': 1.0, 'kernelMultiplier': 1.0, 'confidenceA': 0.5, 'confidenceB': 0.75},
    'CCA-LSC': {'centroidWeight': 3.0, 'discriminativeWeight': 0.6, 'adversarialWeight': 1.0,
                'calibrationHm': 1.5, 'priorThreshold': 0.5},
    'CASUAL': {'lambda_align': 1.0, 'lambda_entropy': 1.0, 'lambda_vat': 1.0, 'vat_radius_fraction': 0.1},
    'COT': {'alphaCluster': .2, 'alphaOT': .2, 'dualLR': .1, 'dualMultiplier': 1.0},
    'CS+CCS': {'csWeight': 1.0, 'ccsWeight': 1.0, 'zKernelMultiplier': 1.0},
    'CLUST — variante source-free': {'clusteringWeight': 1.0, 'consistencyWeight': 1.0, 'informationWeight': 1.0,
                                     'confidence': 0.6, 'emaMomentumComplement': 1e-3},
    'CLUDA': {'contrastiveWeight': 1.0, 'temperature': .07},
}
RIVALS = [s for s in PARAMS if not os.environ.get('RIVALS') or s in os.environ['RIVALS'].split(',')]


def shared_only(raw, prepared):
    out = []
    for b, p in zip(raw, prepared):
        idx = [list(b.channels).index(c) for c in SHARED]
        out.append(two_phase.BagTensor(p.subject, p.condition, p.label, p.windows[:, idx]))
    return out


prob = lambda m, b, side: np.nan_to_num(two_phase.evaluate(m, b, side, dev)[:, 1].detach().numpy(), nan=.5)
seeds = [int(s) for s in sys.argv[1].split(',')]
E1, E2, DUMP = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
schedule = {'epochs1': E1, 'epochs2': E2, 'bagsPerStep': cfg['schedule']['bagsPerStep']}
TRANSFERS = {t.name: t for t in A.TRANSFERS}
for name in sys.argv[5:]:
    tr = TRANSFERS[name]; a = A.assign_from_bags(tr, bags)
    for seed in seeds:
        search.PHASE_ONE_SEED = seed                  # the trainers' record guard reads it
        Y, SUB, P, T, ERR = [], [], {}, {}, {}
        for r in range(4):
            def side(s, role):
                raw = A.side_bags(a, bags, s, r, role)
                prep = sc.prepared(two_phase.bag_tensors(raw))
                return shared_only(raw, prep) if name in ('L6->Robots', 'Robots->L6') else prep
            src, tgt, test = side('source', A.REFIT_ROLE), side('target', A.REFIT_ROLE), side('target', 'test')
            assert src[0].windows.shape[1] == tgt[0].windows.shape[1]
            Y += [b.label for b in test]; SUB += [b.subject for b in test]
            data = search.SearchData(transfer=tr, partition_sha256=a.partition_sha256,
                                     rotations={r: {'source_train': src, 'source_valid': [],
                                                    'target_train': tgt, 'target_valid': []}})
            with spectral_trunk.installed():
                NORM['std'] = None
                NORM['std'] = sc.standard_of(spectral_trunk.SpectralTrunk(int(src[0].windows.shape[1])), src)
                phase1 = {}
                for group in ('common', 'AdaptMIL'):
                    if group == 'AdaptMIL' and 'AdaptMIL' not in RIVALS:
                        continue
                    if group == 'common' and not [s for s in RIVALS if s != 'AdaptMIL']:
                        continue
                    t0 = time.time()
                    fixed = mil_wrapper.COMMON_PHASE_ONE_FIXED
                    hyper = two_phase.check_hyper({**sc.HYPER, **fixed, **schedule,
                                                   'lr2': float('nan'), 'wd2': float('nan'), 'tauLocal': float('nan')})
                    adaptmil_training.stage_two_objective = _l2_stage_two
                    try:
                        with explicit_l2(sc.L2):
                            tr1 = search.train_phase_one(group, data, r, seed, hyper, dev)
                    finally:
                        adaptmil_training.stage_two_objective = _stage_two
                    key = 'predict (phase 1 as it is)' + ('' if group == 'common' else ' [AdaptMIL]')
                    P.setdefault(key, []).extend(prob(tr1['model'], test, 'source'))
                    T[key] = T.get(key, 0) + time.time() - t0
                    meta = {'phase': 1, 'name': group, 'transfer': name, 'rotation': r, 'seed': seed,
                            'partition_sha256': a.partition_sha256, 'channels': data.channels(), 'hyper': hyper,
                            'sigma': tr1['result']['checkpoint'].get('sigma')}
                    phase1[group] = (tr1['result']['checkpoint']['state'], meta)
                for s in RIVALS:
                    if s in ERR:
                        continue
                    t0 = time.time()
                    try:
                        rec = phase1[search.PHASE_ONE_GROUP_OF[s]]
                        torch.manual_seed(seed)
                        out = search.train_phase_two(s, data, r, seed, rec, {**SHARED2, **PARAMS[s]}, schedule, dev)
                        P.setdefault(s, []).extend(prob(out['model'], test, 'target'))
                    except Exception as e:                                  # recorded, not hidden
                        ERR[s] = f'{type(e).__name__}: {e}'
                        traceback.print_exc()
                    T[s] = T.get(s, 0) + time.time() - t0
                NORM['std'] = None
        P = {k: v for k, v in P.items() if k not in ERR and len(v) == len(Y)}
        res = {k: {'BA': round(BA(Y, np.asarray(p) > .5), 3), 'AUC': round(AUC(Y, p), 3)} for k, p in P.items()}
        print('RESULT', name, 'seed', seed, json.dumps(res), 'seconds', json.dumps({k: round(v) for k, v in T.items()}),
              'errors', json.dumps(ERR), flush=True)
        with open(DUMP, 'a') as fh:
            fh.write(json.dumps({'deconf': False, 'transfer': name, 'seed': seed, 'subjects': SUB,
                                 'labels': [int(v) for v in Y], 'scores': {k: [float(v) for v in p] for k, p in P.items()},
                                 'epochs1': E1, 'epochs2': E2, 'seconds': {k: round(v, 1) for k, v in T.items()},
                                 'errors': ERR}) + '\n')
