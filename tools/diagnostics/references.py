"""Reference ceilings per transfer, scored on unseen subjects (BA and AUC):
the source-domain model on source test, the target-domain model on target test (two independent
phase-1 models, the ideal per-domain ceilings), predict (source model on target test, same channels
only), and the task ceiling: phase 2 trained with the target labels. Diagnostic: chooses nothing."""
import copy,json,sys,warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, spectral_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
from MIL_CREDA_EEG.phase_two_terms import blended
torch.set_num_threads(4)
run=json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg=campaign.check_config({**{k:v for k,v in run.items() if k!='remoteInputs'},'device':'cpu'}); campaign.register(cfg['strategies'])
bags=bag_data.load_bags(None,None); dev=torch.device('cpu')
hyper=two_phase.check_hyper({**sc.HYPER,'bagsPerStep':cfg['schedule']['bagsPerStep']})
_plain=two_phase.phase_two_step
def labelled_step(model,source,target,tau_local,device,term_selection='both terms'):
    out=_plain(model,source,target,tau_local,device,term_selection=term_selection)
    y=torch.tensor([b.label for b in target],device=device)
    G=model.classify(model.represent(model.bag_pairs(model.embed([b.windows.to(device) for b in target],'target'))))
    out['objective']=two_phase.source_term(model,G,y); return out
def fit(src,tgt,seed,r,name):
    with spectral_trunk.installed():
        torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
        m=model_store.build_model('MIL-CREDA',int(src[0].windows.shape[1]),int(tgt[0].windows.shape[1]),hyper)
        m.source_trunk.norm=sc.standard_of(m.source_trunk,src)
        same=src[0].windows.shape[1]==tgt[0].windows.shape[1]
        m.target_trunk.norm=m.source_trunk.norm if same else sc.standard_of(m.target_trunk,tgt)
        with explicit_l2(sc.L2): ph1=two_phase.train_source_phase(m,src,[],hyper,g,dev,seed=seed,fold=r,transfer=name)
    return m,ph1,g
prob=lambda m,b,side: list(np.nan_to_num(two_phase.evaluate(m,b,side,dev)[:,1].detach().numpy(),nan=.5))
seed=int(sys.argv[1])
for name in sys.argv[2:]:
    a=A.assign_from_bags({t.name:t for t in A.TRANSFERS}[name],bags)
    Y={'source':[],'target':[]}; P={k:[] for k in ['source model on source test','target model on target test','predict','task ceiling: phase 2 with target labels','task ceiling, blend .5']}
    for r in range(4):
        side=lambda s,role: sc.prepared(two_phase.bag_tensors(A.side_bags(a,bags,s,r,role)))
        src,stest,tgt,ttest=side('source',A.REFIT_ROLE),side('source','test'),side('target',A.REFIT_ROLE),side('target','test')
        Y['source']+=[b.label for b in stest]; Y['target']+=[b.label for b in ttest]
        m,ph1,g=fit(src,tgt,seed,r,name)
        P['source model on source test']+=prob(m,stest,'source')
        same=src[0].windows.shape[1]==tgt[0].windows.shape[1]
        if same: P['predict']+=prob(m,ttest,'source')
        mt,_,_=fit(tgt,tgt,seed,r,name); P['target model on target test']+=prob(mt,ttest,'source')
        med=two_phase.within_bag_median_distance(m,src,'source',dev); m._sigma=hyper['sigmaMultiplier']*med if med>0 else m._sigma; ph1['checkpoint']['sigma']=m._sigma
        with spectral_trunk.installed(): j=two_phase.bridge(m)
        start={k:v.detach().clone() for k,v in m.target_trunk.state_dict().items()}
        two_phase.phase_two_step=labelled_step
        try:
            with spectral_trunk.installed():
                two_phase.adapt_target_phase(m,ph1,j['frozen_state'],src,tgt,[],hyper,torch.Generator().manual_seed(seed),dev,seed=seed,fold=r,transfer=name)
        finally: two_phase.phase_two_step=_plain
        P['task ceiling: phase 2 with target labels']+=prob(m,ttest,'target')
        adapted={k:v.detach().clone() for k,v in m.target_trunk.state_dict().items()}
        m.target_trunk.load_state_dict(blended(start,adapted,0.5)); P['task ceiling, blend .5']+=prob(m,ttest,'target')
    out={}
    for k,v in P.items():
        if not v: continue
        y=Y['source'] if k.startswith('source model') else Y['target']
        out[k]={'AUC':round(AUC(y,v),3),'BA':round(BA(y,[p>.5 for p in v]),3)}
    print('RESULT seed',seed,name,json.dumps(out),flush=True)
