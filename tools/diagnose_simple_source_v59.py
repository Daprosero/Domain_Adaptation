import sys,warnings,collections,pickle,os; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
from scipy.signal import welch
from MIL_CREDA_EEG import bag_data, assignment as A
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AUC
torch.set_num_threads(3)
b=bag_data.load_bags(None,None); T={t.name:t for t in A.TRANSFERS}
bands=[(1,4),(4,8),(8,13),(13,30),(30,45)]; C={}
def wf(x):
    k=(x.subject,x.condition)
    if k not in C:
        w=x.windows().astype(np.float32); w=w[::max(1,len(w)//100)]
        f,p=welch(w,fs=128,axis=-1,nperseg=128)
        C[k]=np.log(np.stack([p[...,(f>=a)&(f<c)].mean(-1) for a,c in bands],-1)+1e-12).reshape(len(w),-1).astype(np.float32)
    return C[k]
y_=lambda x:int(x.label=='yes')
def ba(y,p):
    y=np.array(y);h=(np.array(p)>=.5).astype(int); return float(np.mean([(h[y==c]==c).mean() for c in (0,1)]))
sub=lambda xs:np.stack([np.r_[wf(x).mean(0),wf(x).std(0)] for x in xs])
def mil(tr,te,seed):
    torch.manual_seed(seed); d=wf(tr[0]).shape[1]
    allw=np.concatenate([wf(x) for x in tr]); mu,sd=allw.mean(0),allw.std(0)+1e-6
    n=lambda x:torch.tensor((wf(x)-mu)/sd)
    f=nn.Sequential(nn.Linear(d,16),nn.ReLU(),nn.Dropout(.5)); a=nn.Linear(16,1); h=nn.Linear(16,1)
    o=torch.optim.Adam([*f.parameters(),*a.parameters(),*h.parameters()],lr=1e-3,weight_decay=1e-2)
    yy=np.array([y_(x) for x in tr]); pw=torch.tensor((1-yy).sum()/max(1,yy.sum()),dtype=torch.float32)
    def fw(x):
        z=f(n(x)); w=torch.softmax(a(z),0); return h((w*z).sum(0))
    for e in range(40):
        for i in np.random.default_rng(seed*100+e).permutation(len(tr)):
            f.train(); l=nn.functional.binary_cross_entropy_with_logits(fw(tr[i]),torch.tensor([float(yy[i])]),pos_weight=pw); o.zero_grad(); l.backward(); o.step()
    f.eval()
    with torch.no_grad(): return [torch.sigmoid(fw(x)).item() for x in te]
M={'LR':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=500,class_weight='balanced')),'SVM':lambda:make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced',probability=True,random_state=0)),'RF':lambda:RandomForestClassifier(400,class_weight='balanced',random_state=0)}
rows=[]
for name in ('L1->L11','L11->L1','Robots->CogniFit','CogniFit->Robots','L6->Robots'):
    t=T[name]; a=A.assign_from_bags(t,b); res=collections.defaultdict(lambda:([],[]))
    for r in range(4):
        tr=A.side_bags(a,b,'source',r,'train') if False else None
        te=A.side_bags(a,b,'source',r,'test'); ts={x.subject for x in te}
        tr=[b.get(s,t.source) for s,v in a.sides.items() if v=='source' and s not in ts]
        ytr=[y_(x) for x in tr]; yte=[y_(x) for x in te]
        for k,m in M.items():
            q=m(); q.fit(sub(tr),ytr); res[k][0].extend(yte); res[k][1].extend(q.predict_proba(sub(te))[:,1])
        p=np.mean([mil(tr,te,s) for s in (0,1)],0); res['MIL16'][0].extend(yte); res['MIL16'][1].extend(p)
    line=' '.join(f'{k} BA={ba(*v):.3f}/AUC={AUC(*v):.3f}' for k,v in res.items()); print(f'{t.source:9s} n={len(res["LR"][0])} {line}',flush=True)
