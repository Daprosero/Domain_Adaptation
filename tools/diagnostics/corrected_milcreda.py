"""MIL-CREDA corrected for the located causes, against predict (the phase-1 model as it is).
Diagnostic only: every change is a patch active inside this script, the plan's code is untouched.

Stage 0 (label-free, closed form): per-domain feature standardization of the target (the global
mean/scale offset is the whole linear domain shift, data audit). Phase 2 corrections (audit of the
proposal against the code):
  - Eq. 26: target confidences enter as weights without gradient (the channel r44 Sec. 5.1 already
    removed from the local term), so a misaligned bag cannot be withdrawn by lowering confidence;
  - Eq. 35: the prediction g enters the correspondence without gradient (no self-confirming pull),
    and the correspondence temperature is the spread of that bag's kernel row (not 1);
  - pseudo-labels: a fixed cut, the median of the bridged model's scores on the unlabelled target
    training subjects, computed once (no drift of the whole batch across 0.5);
  - sigma: the median distance between instances of different source bags (between-bag scale),
    so the bag kernel can separate subjects and classes;
  - optional proximal anchor mu * ||theta_t - theta_t0||^2 toward the bridged target trunk.
Device pair: the 12 channels Emotiv and CGX share, so predict exists and phase 2 starts from it.
Scored on unseen target test subjects: BA (primary), AUC, and AUC over age-matched ADHD/control
pairs (|age difference| <= 1 year), so a gain cannot come from age alone. Labels and ages only
score; nothing is chosen with them.
"""
import copy, json, sys, warnings
sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, spectral_trunk, two_phase
from MIL_CREDA_EEG import mil_creda_model as M
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
from MIL_CREDA_EEG.phase_two_terms import blended
torch.set_num_threads(4)
run = json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg = campaign.check_config({**{k: v for k, v in run.items() if k != 'remoteInputs'}, 'device': 'cpu'})
campaign.register(cfg['strategies'])
bags = bag_data.load_bags(None, None); dev = torch.device('cpu')
hyper = two_phase.check_hyper({**sc.HYPER, 'bagsPerStep': cfg['schedule']['bagsPerStep']})
manifest = pd.read_csv('MIL-CREDA-EEG/Data/processed/manifest.csv', low_memory=False)
AGE = manifest.groupby('subject').age.first().to_dict()
SHARED = ('F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8')

_conf, _corr, _pl, _obj = M.confidences, M.total_correspondence, M.pseudolabel, two_phase.phase_two_objective
STATE = {'thr': None, 'mu': 0.0, 'model': None, 'theta0': None}


def detached_confidences(G):
    return _conf(G).detach()


def detached_correspondence(cross, labels, g, tau):
    return _corr(cross, labels, g.detach(), float(cross.detach().std()) + 1e-6)


def fixed_pseudolabel(g):
    return int(float(g[1]) >= STATE['thr'])


def anchored_objective(terms, term_selection='both terms'):
    out = _obj(terms, term_selection)
    if STATE['mu'] > 0:
        m = STATE['model']
        prox = sum(((p - STATE['theta0'][n]) ** 2).sum() for n, p in m.target_trunk.named_parameters())
        out = out + STATE['mu'] * prox.to(out.dtype)
    return out


def corrected(on):
    M.confidences = detached_confidences if on else _conf
    M.total_correspondence = detached_correspondence if on else _corr
    M.pseudolabel = fixed_pseudolabel if on else _pl
    two_phase.phase_two_objective = anchored_objective if on else _obj


def between_bag_sigma(m, bs, domain, n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        H = [m.trunk(domain)(b.windows).double() for b in bs]
    d = []
    for _ in range(n):
        i, j = torch.randint(len(H), (2,), generator=g).tolist()
        if i == j:
            continue
        a = H[i][torch.randint(len(H[i]), (1,), generator=g)]; b = H[j][torch.randint(len(H[j]), (1,), generator=g)]
        d.append(float(torch.cdist(a, b)))
    return float(np.median(d))


def shared_only(raw, prepared):
    out = []
    for b, p in zip(raw, prepared):
        idx = [list(b.channels).index(c) for c in SHARED]
        out.append(two_phase.BagTensor(p.subject, p.condition, p.label, p.windows[:, idx]))
    return out


def age_matched_auc(y, p, subjects):
    pos = [(AGE[s], q) for s, q, l in zip(subjects, p, y) if l == 1]
    neg = [(AGE[s], q) for s, q, l in zip(subjects, p, y) if l == 0]
    pairs = [(qp > qn) + 0.5 * (qp == qn) for ap, qp in pos for an, qn in neg if abs(ap - an) <= 1]
    return (round(float(np.mean(pairs)), 3), len(pairs)) if pairs else (None, 0)


SEX = (manifest.groupby('subject').sex.first() == 'F').astype(float).to_dict()
_features = spectral_trunk.SpectralTrunk.features
OFFSET = {}                                    # window hash -> feature offset of its subject


def _key(row):
    import hashlib
    return hashlib.blake2b(row.tobytes(), digest_size=16).digest()


def deconfounded_features(self, x):
    f = _features(self, x)
    if not OFFSET:
        return f
    rows = x.detach().cpu().numpy()
    off = torch.stack([OFFSET.get(_key(r), torch.zeros(f.shape[1], dtype=f.dtype)) for r in rows]).to(f.device, f.dtype)
    return f - off


def fit_deconfounding(src, groups):
    """Age and sex effects on each spectral feature, fitted on the source training subjects only;
    every subject's own effect is then removed from its windows, in every domain."""
    OFFSET.clear()
    probe = spectral_trunk.SpectralTrunk(int(src[0].windows.shape[1]))
    with torch.no_grad():
        mean_f = {b.subject: torch.nan_to_num(_features(probe, b.windows)).double().mean(0) for b in src}
    C = np.array([[AGE[s], SEX[s]] for s in mean_f]); c0 = C.mean(0)
    X = np.c_[np.ones(len(C)), C - c0]; Yf = torch.stack(list(mean_f.values())).numpy()
    B = np.linalg.lstsq(X, Yf, rcond=None)[0][1:]                       # (2, features)
    for bs in groups:
        for b in bs:
            o = torch.from_numpy((np.array([AGE[b.subject], SEX[b.subject]]) - c0) @ B).float()
            for r in b.windows.numpy():
                OFFSET[_key(r)] = o


spectral_trunk.SpectralTrunk.features = deconfounded_features
DECONF = sys.argv[2] == 'deconf'
DUMP = sys.argv[3]
prob = lambda m, b, side: np.nan_to_num(two_phase.evaluate(m, b, side, dev)[:, 1].detach().numpy(), nan=.5)
VARIANTS = {'corrected': (1e-3, 'both terms', 0.0), 'corrected + anchor': (1e-3, 'both terms', 10.0),
            'corrected, global only': (1e-3, 'global term only', 0.0)}

seeds = [int(s) for s in sys.argv[1].split(',')]
for name in sys.argv[4:]:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    for seed in seeds:
        Y, SUB, P = [], [], {}
        for r in range(4):
            def side(s, role):
                raw = A.side_bags(a, bags, s, r, role)
                prep = sc.prepared(two_phase.bag_tensors(raw))
                return shared_only(raw, prep) if name in ('L6->Robots', 'Robots->L6') else prep
            src, tgt, test = side('source', A.REFIT_ROLE), side('target', A.REFIT_ROLE), side('target', 'test')
            Y += [b.label for b in test]; SUB += [b.subject for b in test]
            if DECONF:
                fit_deconfounding(src, (src, tgt, test))
            else:
                OFFSET.clear()
            with spectral_trunk.installed():
                torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
                c = int(src[0].windows.shape[1])
                m = model_store.build_model('MIL-CREDA', c, c, hyper)
                m.source_trunk.norm = sc.standard_of(m.source_trunk, src)
                m.target_trunk.norm = m.source_trunk.norm
                with explicit_l2(sc.L2):
                    ph1 = two_phase.train_source_phase(m, src, [], hyper, g, dev, seed=seed, fold=r, transfer=name)
                P.setdefault('predict (phase 1 as it is)', []).extend(prob(m, test, 'source'))
                m._sigma = between_bag_sigma(m, src, 'source', seed=seed); ph1['checkpoint']['sigma'] = m._sigma
                j = two_phase.bridge(m)
                m.target_trunk.norm = sc.standard_of(m.target_trunk, tgt)                 # stage 0
                P.setdefault('stage 0 only (per-domain standardization)', []).extend(prob(m, test, 'target'))
                for label, (lr2, terms, mu) in VARIANTS.items():
                    mm = copy.deepcopy(m)
                    STATE.update(thr=float(np.median(prob(mm, tgt, 'target'))), mu=mu, model=mm,
                                 theta0={n: p.detach().clone() for n, p in mm.target_trunk.named_parameters()})
                    h2 = two_phase.check_hyper({**hyper, 'lr2': lr2, 'epochs2': 20})
                    corrected(True)
                    try:
                        two_phase.adapt_target_phase(mm, ph1, j['frozen_state'], src, tgt, [], h2,
                                                     torch.Generator().manual_seed(seed), dev, seed=seed,
                                                     fold=r, transfer=name, term_selection=terms)
                    finally:
                        corrected(False)
                    P.setdefault(label, []).extend(prob(mm, test, 'target'))
        out = {}
        for k, p in P.items():
            am, n = age_matched_auc(Y, p, SUB)
            out[k] = {'BA': round(BA(Y, np.asarray(p) > .5), 3), 'AUC': round(AUC(Y, p), 3), 'ageMatchedAUC': am}
        print('RESULT', 'deconf' if DECONF else 'plain', name, 'seed', seed, 'agePairs', n, json.dumps(out), flush=True)
        with open(DUMP, 'a') as fh:
            fh.write(json.dumps({'deconf': DECONF, 'transfer': name, 'seed': seed, 'subjects': SUB,
                                 'labels': [int(v) for v in Y], 'scores': {k: [float(v) for v in p] for k, p in P.items()}}) + '\n')
