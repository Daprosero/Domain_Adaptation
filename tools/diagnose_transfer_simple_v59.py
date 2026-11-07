import sys,warnings,collections; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from scipy.signal import welch,butter,sosfiltfilt
from MIL_CREDA_EEG import bag_data, assignment as A
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score as BA, roc_auc_score as AUC
b=bag_data.load_bags(None,None); bands=[(1,4),(4,8),(8,13),(13,30),(30,45)]; C={}; T={t.name:t for t in A.TRANSFERS}
def ft(x):
    k=(x.subject,x.condition)
    if k in C: return C[k]
    w=x.windows().astype(np.float64); w=w[::max(1,len(w)//60)]; w=w-w.mean(-1,keepdims=True)
    w=sosfiltfilt(butter(4,[1,45],'bandpass',fs=128,output='sos'),w,axis=-1); w=w-w.mean(1,keepdims=True)
    f,p=welch(w,fs=128,axis=-1,nperseg=256); p=np.median(p,0)
    a=np.stack([p[...,(f>=l)&(f<h)].mean(-1) for l,h in bands],-1); C[k]=np.r_[np.log(a+1e-12).ravel(),np.log(a/a.sum(-1,keepdims=True)+1e-12).ravel()]; return C[k]
yn=lambda x:int(x.label=='yes')
M={'LR':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=800,class_weight='balanced')),'RF':lambda:RandomForestClassifier(400,class_weight='balanced',random_state=0)}
for name in ('L1->L11','L11->L1','Robots->CogniFit','CogniFit->Robots'):
    t=T[name]; a=A.assign_from_bags(t,b); r_=collections.defaultdict(lambda:([],[]))
    for r in range(4):
        tr=A.side_bags(a,b,'source',r,'nontest'); te=A.side_bags(a,b,'target',r,'test')
        Xa=np.stack([ft(x) for x in tr]); Xb=np.stack([ft(x) for x in te]); ya=[yn(x) for x in tr]; yb=[yn(x) for x in te]
        for k,m in M.items():
            q=m().fit(Xa,ya); r_[k][0].extend(yb); r_[k][1].extend(q.predict_proba(Xb)[:,1])
            # unsupervised target alignment of feature statistics, no target label
            mu,sd=Xb.mean(0),Xb.std(0)+1e-9; ma,sa=Xa.mean(0),Xa.std(0)+1e-9
            r_[k+'+align'][0].extend(yb); r_[k+'+align'][1].extend(q.predict_proba((Xb-mu)/sd*sa+ma)[:,1])
    print(name,len(r_['LR'][0]),' '.join(f'{k} BA={BA(v[0],np.array(v[1])>=.5):.3f}/AUC={AUC(*v):.3f}' for k,v in r_.items()),flush=True)
