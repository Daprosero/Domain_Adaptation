"""Ceiling for a transfer: predict (source labels) vs a model trained with target labels
vs source+target labels, all scored on the same unseen target test subjects. Spectral, local."""
import json,sys,warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, spectral_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
torch.set_num_threads(4)
run=json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg=campaign.check_config({**{k:v for k,v in run.items() if k!='remoteInputs'},'device':'cpu'}); campaign.register(cfg['strategies'])
bags=bag_data.load_bags(None,None); dev=torch.device('cpu')
hyper=two_phase.check_hyper({**sc.HYPER,'bagsPerStep':cfg['schedule']['bagsPerStep']})
def fit(train,seed,r,name):
    with spectral_trunk.installed():
        torch.manual_seed(seed); g=torch.Generator().manual_seed(seed); c=int(train[0].windows.shape[1])
        m=model_store.build_model('MIL-CREDA',c,c,hyper); m.source_trunk.norm=sc.standard_of(m.source_trunk,train)
        with explicit_l2(sc.L2): two_phase.train_source_phase(m,train,[],hyper,g,dev,seed=seed,fold=r,transfer=name)
    return m
score=lambda m,b: list(np.nan_to_num(two_phase.evaluate(m,b,'source',dev)[:,1].detach().numpy(),nan=.5))
seed=int(sys.argv[1])
for name in sys.argv[2:]:
    a=A.assign_from_bags({t.name:t for t in A.TRANSFERS}[name],bags); y=[]; P={'predict (source labels)':[],'ceiling (target labels)':[],'ideal (source+target labels)':[]}
    for r in range(4):
        side=lambda s,role: sc.prepared(two_phase.bag_tensors(A.side_bags(a,bags,s,r,role)))
        src,tgt,test=side('source',A.REFIT_ROLE),side('target',A.REFIT_ROLE),side('target','test')
        y+=[b.label for b in test]
        P['predict (source labels)']+=score(fit(src,seed,r,name),test)
        P['ceiling (target labels)']+=score(fit(tgt,seed,r,name),test)
        P['ideal (source+target labels)']+=score(fit(list(src)+list(tgt),seed,r,name),test)
    print('RESULT seed',seed,name,{k:round(AUC(y,v),3) for k,v in P.items()},flush=True)
