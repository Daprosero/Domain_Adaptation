"""MIL-CREDA phase 1 with a spectral encoder fed by CAR + 1-45 Hz windows, every source, local."""
import json,sys,warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from torch import nn
from scipy.signal import butter,sosfiltfilt
from sklearn.metrics import roc_auc_score as AUC
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, spectral_trunk, two_phase
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
torch.set_num_threads(4)
run=json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg=campaign.check_config({**{k:v for k,v in run.items() if k!='remoteInputs'},'device':'cpu'}); campaign.register(cfg['strategies'])
bags=bag_data.load_bags(None,None); dev=torch.device('cpu'); SOS=butter(4,[1,45],'bandpass',fs=128,output='sos')
class Std(nn.Module):
    def __init__(s,m,sd): super().__init__(); s.register_buffer('m',m); s.register_buffer('sd',sd)
    def forward(s,x): return (x-s.m)/s.sd
def prep(g):
    out=[]
    for b in g:
        w=b.windows.double().numpy(); w=w[::max(1,len(w)//60)]; w=sosfiltfilt(SOS,w-w.mean(-1,keepdims=True),axis=-1); w=w-w.mean(1,keepdims=True)
        out.append(two_phase.BagTensor(b.subject,b.condition,b.label,torch.from_numpy(w.copy()).float()))
    return out
lam=float(sys.argv[1]); SEED=int(sys.argv[2])
for name in sys.argv[3:]:
    a=A.assign_from_bags({t.name:t for t in A.TRANSFERS}[name],bags); y_,p_,ptr=[],[],[]
    for r in range(4):
        tr=prep(two_phase.bag_tensors(A.side_bags(a,bags,'source',r,A.REFIT_ROLE))); te=prep(two_phase.bag_tensors(A.side_bags(a,bags,'source',r,'test')))
        C=int(tr[0].windows.shape[1])
        hyper=two_phase.check_hyper({'lr1':float(__import__('os').environ.get('LR1','1e-2')),'wd1':0.0,'attentionGamma':0.0,'attentionTemperature':10.0,'sigmaMultiplier':1.0,'lr2':1e-3,'wd2':1e-5,'tauLocal':1.0,'epochs1':int(__import__('os').environ.get('EP1','100')),'epochs2':1,'bagsPerStep':8})
        with spectral_trunk.installed():
            torch.manual_seed(SEED); g=torch.Generator().manual_seed(SEED)
            m=model_store.build_model('MIL-CREDA',C,C,hyper)
            with torch.no_grad(): F=torch.cat([m.source_trunk.features(b.windows) for b in tr]); F=torch.nan_to_num(F)
            std=Std(F.mean(0),F.std(0)+1e-6); m.source_trunk.norm=std; m.target_trunk.norm=std
            with explicit_l2(lam): two_phase.train_source_phase(m,tr,[],hyper,g,dev,seed=SEED,fold=r,transfer=name)
            p=two_phase.evaluate(m,te,'source',dev)[:,1].detach().numpy(); pt=two_phase.evaluate(m,tr,'source',dev)[:,1].detach().numpy()
        y_+=[b.label for b in te]; p_+=list(np.nan_to_num(p,nan=.5)); ptr.append(AUC([b.label for b in tr],pt))
    print('RESULT lr',__import__('os').environ.get('LR1'),'ep',__import__('os').environ.get('EP1'),'lam',lam,'seed',SEED,name.split('->')[0],'train AUC %.3f'%np.mean(ptr),'| SOURCE unseen AUC %.3f'%AUC(y_,p_),flush=True)
