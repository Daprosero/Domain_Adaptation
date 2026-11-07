import sys,json,time,tempfile,warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter,sosfiltfilt
from sklearn.metrics import roc_auc_score as AUC
from MIL_CREDA_EEG import bag_data, assignment as A, campaign, search, controls, two_phase
torch.set_num_threads(4)
cfg=json.load(open('tools/kaggle/v59-final/run-config.json'))['run']['kwargs']['config']; cfg={k:v for k,v in cfg.items() if k!='remoteInputs'}; cfg['device']='cpu'; cfg=campaign.check_config(cfg)
P=Path('MIL-CREDA-EEG'); b0=bag_data.load_bags(None,None); T={t.name:t for t in A.TRANSFERS}
campaign.register(cfg['strategies'])
sel={n:campaign._selections(P,search.search_data(T[n],b0),1,'MIL-CREDA',cfg) for n in ('Robots->CogniFit','CogniFit->Robots')}
raw=bag_data.Bag.windows; sos=butter(4,[1,45],'bandpass',fs=128,output='sos')
def filt(self):
    w=raw(self).astype(np.float64); w=w-w.mean(-1,keepdims=True); w=sosfiltfilt(sos,w,axis=-1); w=w-w.mean(1,keepdims=True)
    return w.astype(np.float32)
bag_data.Bag.windows=filt
b=bag_data.load_bags(None,None)
def ba(y,p):
    y=np.array(y);h=(np.array(p)>=.5).astype(int); return float(np.mean([(h[y==c]==c).mean() for c in (0,1)]))
root=Path(tempfile.mkdtemp())
for n,s in sel.items():
    t=T[n]; data=search.search_data(t,b); a=A.assign_from_bags(t,b); S=([],[]); G=([],[])
    for r in range(4):
        t0=time.time(); e=int(min(60,max(20,s[r]['epochs'])))
        hy=search.phase_one_hyper('MIL-CREDA',s[r]['params'],search.refit_schedule(cfg['schedule'],1,e))
        tr=search.train_phase_one('MIL-CREDA',data,r,search.PHASE_ONE_SEED,hy,'cpu')
        path=search.save_phase_one(root,'MIL-CREDA',data,r,search.PHASE_ONE_SEED,tr)
        for side,acc in (('source',S),('target',G)):
            c=controls.source_only_from_store(path,t,two_phase.bag_tensors(A.side_bags(a,b,side,r,'test')),device=torch.device('cpu'),allow_code_mismatch=True)
            for q in c['predictions']: acc[0].append(q['label']); acc[1].append(q['probability'])
        print(n,'rot',r,'E',e,'E1 stored',s[r]['epochs'],round(time.time()-t0),'s',flush=True)
    print('RESULT',n,'heldout source BA',round(ba(*S),3),'AUC',round(AUC(*S),3),'| target BA',round(ba(*G),3),'AUC',round(AUC(*G),3),flush=True)
