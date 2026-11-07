import sys,warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from scipy.signal import welch,butter,sosfiltfilt
from MIL_CREDA_EEG import bag_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold,cross_val_predict
from sklearn.metrics import balanced_accuracy_score as BA, roc_auc_score as AUC
b=bag_data.load_bags(None,None); bands=[(1,4),(4,8),(8,13),(13,30),(30,45)]; C={}
FS={'CGX':128,'EPOCX':128}
def ft(x):
    k=(x.subject,x.condition)
    if k in C: return C[k]
    fs=FS.get(x.device,128); w=x.windows().astype(np.float64); w=w[::max(1,len(w)//60)]
    w=w-w.mean(-1,keepdims=True); w=sosfiltfilt(butter(4,[1,45],'bandpass',fs=fs,output='sos'),w,axis=-1)
    w=w-w.mean(1,keepdims=True)
    f,p=welch(w,fs=fs,axis=-1,nperseg=min(256,w.shape[-1])); p=np.median(p,0)
    a=np.stack([p[...,(f>=l)&(f<h)].mean(-1) for l,h in bands],-1); rel=a/a.sum(-1,keepdims=True)
    C[k]=np.r_[np.log(a+1e-12).ravel(),np.log(rel+1e-12).ravel()]; return C[k]
def cv(xs,y,name):
    X=np.stack([ft(x) for x in xs]); g=[x.subject for x in xs]; y=np.array(y,int); o=[]
    for nm,m in (('LR',make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=800,class_weight='balanced'))),('RF',RandomForestClassifier(300,class_weight='balanced',random_state=0))):
        r=[]
        for s in range(3):
            p=cross_val_predict(m,X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=s),groups=g,method='predict_proba')[:,1]; r.append((BA(y,p>=.5),AUC(y,p)))
        o.append(f'{nm} BA={np.mean([a for a,_ in r]):.3f} AUC={np.mean([a for _,a in r]):.3f}')
    print(f'{name:28s} n={len(xs)}',*o,flush=True)
by=lambda *c:[x for x in b.bags if x.condition in c]
print({c:by(c)[0].device for c in ('L1','L6','L11','Robots','CogniFit')})
cv(by('L1','L11'),[x.condition=='L11' for x in by('L1','L11')],'CONTROL L1 vs L11')
cv(by('Robots','CogniFit'),[x.condition=='Robots' for x in by('Robots','CogniFit')],'CONTROL Robots vs CogniFit')
for c in ('L1','L6','L11','Robots','CogniFit'): cv(by(c),[x.label=='yes' for x in by(c)],'ADHD in '+c)
