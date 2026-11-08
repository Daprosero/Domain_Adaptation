"""Phase 2 on the solved spectral phase 1: which phase-2 setting keeps or beats predict? Local."""
import copy,json,sys,warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, spectral_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
from MIL_CREDA_EEG.phase_two_terms import balanced_pseudolabels, blended
torch.set_num_threads(4)
run=json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg=campaign.check_config({**{k:v for k,v in run.items() if k!='remoteInputs'},'device':'cpu'}); campaign.register(cfg['strategies'])
bags=bag_data.load_bags(None,None); dev=torch.device('cpu')
VARIANTS={'src-norm lr2 1e-3 x20':(1e-3,20,'both terms'),'src-norm lr2 1e-4 x20':(1e-4,20,'both terms'),'src-norm global only':(1e-3,20,'global term only'),'src-norm global lr2 1e-4':(1e-4,20,'global term only')}
seed=int(sys.argv[1])
for name in sys.argv[2:]:
    a=A.assign_from_bags({t.name:t for t in A.TRANSFERS}[name],bags); y=[]; P={'predict':[],'target norm only':[]}; P.update({k:[] for k in VARIANTS}); P.update({k+' blend.5':[] for k in VARIANTS})
    for r in range(4):
        side=lambda s,role: sc.prepared(two_phase.bag_tensors(A.side_bags(a,bags,s,r,role)))
        src,tgt,test=side('source',A.REFIT_ROLE),side('target',A.REFIT_ROLE),side('target','test')
        hyper=two_phase.check_hyper({**sc.HYPER,'bagsPerStep':cfg['schedule']['bagsPerStep']})
        with spectral_trunk.installed():
            torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
            m=model_store.build_model('MIL-CREDA',int(src[0].windows.shape[1]),int(tgt[0].windows.shape[1]),hyper)
            m.source_trunk.norm=sc.standard_of(m.source_trunk,src); m.target_trunk.norm=m.source_trunk.norm
            with explicit_l2(sc.L2): ph1=two_phase.train_source_phase(m,src,[],hyper,g,dev,seed=seed,fold=r,transfer=name)
            med=two_phase.within_bag_median_distance(m,src,'source',dev); m._sigma=hyper['sigmaMultiplier']*med if med>0 else m._sigma; ph1['checkpoint']['sigma']=m._sigma
            y+=[b.label for b in test]; P['predict']+=list(two_phase.evaluate(m,test,'source',dev)[:,1].detach().numpy())
            nm=copy.deepcopy(m); two_phase.bridge(nm); nm.target_trunk.norm=sc.standard_of(nm.target_trunk,tgt)
            P['target norm only']+=list(two_phase.evaluate(nm,test,'target',dev)[:,1].detach().numpy())
            for k,(lr2,ep,terms) in VARIANTS.items():
                mm=copy.deepcopy(m); j=two_phase.bridge(mm)  # target keeps the source standardization
                start={kk:v.detach().clone() for kk,v in mm.target_trunk.state_dict().items()}
                h2=two_phase.check_hyper({**hyper,'lr2':lr2,'epochs2':ep}); g2=torch.Generator().manual_seed(seed)
                with balanced_pseudolabels(mm):
                    two_phase.adapt_target_phase(mm,ph1,j['frozen_state'],src,tgt,[],h2,g2,dev,seed=seed,fold=r,transfer=name,term_selection=terms)
                P[k]+=list(np.nan_to_num(two_phase.evaluate(mm,test,'target',dev)[:,1].detach().numpy(),nan=.5))
                adapted={kk:v.detach().clone() for kk,v in mm.target_trunk.state_dict().items()}
                mm.target_trunk.load_state_dict(blended(start,adapted,0.5))
                P[k+' blend.5']+=list(np.nan_to_num(two_phase.evaluate(mm,test,'target',dev)[:,1].detach().numpy(),nan=.5))
    print('RESULT seed',seed,name,{k:round(AUC(y,v),3) for k,v in P.items()},flush=True)
