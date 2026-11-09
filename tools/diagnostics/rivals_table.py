"""Table of the rival strategies' dump (`rivals_spectral.py`): per transfer and strategy, the mean
over seeds of BA (threshold .5), AUC, and the AUC left after linearly removing age from the score
(the score regressed on age over that run's target test subjects, label-free; AUC of the residual).
Ages: manifest.csv, first age per subject (subject ids are unique across datasets, so CogniFit's
'Cognifit/EMOTIV' condition name does not enter the join).

Usage: rivals_table.py <dump.jsonl> [<dump.jsonl> ...]
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA

manifest = pd.read_csv('MIL-CREDA-EEG/Data/processed/manifest.csv', low_memory=False)
AGE = manifest.groupby('subject').age.first().to_dict()
ORDER = ('CogniFit->Robots', 'Robots->CogniFit', 'L1->L11', 'L11->L1', 'L6->Robots', 'Robots->L6')


def age_removed_auc(y, p, subjects):
    a = np.array([AGE[s] for s in subjects], float)
    X = np.c_[np.ones_like(a), a]
    resid = np.asarray(p) - X @ np.linalg.lstsq(X, np.asarray(p), rcond=None)[0]
    return AUC(y, resid)


rows = []
for path in sys.argv[1:]:
    for line in open(path):
        d = json.loads(line)
        y = np.array(d['labels'])
        for k, p in d['scores'].items():
            rows.append({'transfer': d['transfer'], 'strategy': k, 'seed': d['seed'],
                         'BA': BA(y, np.asarray(p) > .5), 'AUC': AUC(y, p),
                         'AUC age removed': age_removed_auc(y, p, d['subjects'])})
df = pd.DataFrame(rows)
order = {t: i for i, t in enumerate(ORDER)}
strategies = list(dict.fromkeys(df.strategy))
tab = (df.groupby(['transfer', 'strategy'])
         .agg(seeds=('seed', 'nunique'), BA=('BA', 'mean'), AUC=('AUC', 'mean'),
              AUC_age_removed=('AUC age removed', 'mean'), AUC_sd=('AUC', 'std'))
         .reset_index())
tab['t'] = tab.transfer.map(order); tab['s'] = tab.strategy.map(strategies.index)
tab = tab.sort_values(['t', 's']).drop(columns=['t', 's'])
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 500)
print(tab.round(3).to_string(index=False))
