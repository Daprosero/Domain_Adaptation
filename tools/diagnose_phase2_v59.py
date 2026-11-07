import sqlite3,json,glob,sys,os,collections,re
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,'src')
from MIL_CREDA_EEG import bag_data
lab={k:int(v=='yes') for k,v in bag_data.load_bags(None,None).labels().items()}
def ba(y,p):
    y=np.asarray(y);h=(p>=.5).astype(int); a=[(h[y==c]==c).mean() for c in (0,1) if (y==c).any()]
    return float(np.mean(a))
rng=np.random.default_rng(0); out=[]
core=('L1-to-L11','L11-to-L1','Robots-to-CogniFit','CogniFit-to-Robots','L6-to-Robots','Robots-to-L6')
for f in sorted(glob.glob('MIL-CREDA-EEG/Results/hpo/*__phase2__*.db')):
    t,_,s,o=os.path.basename(f)[:-3].split('__')
    if t not in core: continue
    c=sqlite3.connect(f'file:{f}?mode=ro',uri=True)
    A=collections.defaultdict(dict)
    for tid,k,v in c.execute("select trial_id,key,value_json from trial_user_attributes where key in ('ensvProbabilities','ensvSubjects','pooledCentroidBalancedAccuracy')"): A[tid][k]=json.loads(v)
    sc=[];P=[];subs=None
    for tid,d in A.items():
        if 'ensvProbabilities' not in d: continue
        subs=d['ensvSubjects']; pr=np.asarray(d['ensvProbabilities'],float)[...,1]
        for e in range(min(len(pr),len(d['pooledCentroidBalancedAccuracy']))):
            sc.append(d['pooledCentroidBalancedAccuracy'][e]); P.append(pr[e])
    if not P: continue
    y=np.array([lab[x] for x in subs]); sc=np.array(sc); P=np.stack(P)
    tb=np.array([ba(y,p) for p in P])
    sel=tb[np.flatnonzero(sc==sc.max())].mean()
    nul=[]
    for _ in range(30):
        yy=rng.permutation(y); nul.append(max(ba(yy,p) for p in P[::max(1,len(P)//120)]))
    rho=spearmanr(sc,tb).statistic if len(set(sc))>1 and len(set(tb))>1 else np.nan
    out.append(dict(t=t,s=s,o=o,n=len(y),cands=len(P),sel=sel,best=tb.max(),med=float(np.median(tb)),nul=float(np.mean(nul)),rho=float(rho) if rho==rho else None,e0=float(np.mean([tb[0]]))))
json.dump(out,open(sys.argv[1],'w'))
import pandas as pd
d=pd.DataFrame(out); print(len(d))
g=d.groupby('s')[['sel','med','best','nul','rho']].mean().round(3); g['n']=d.groupby('s').size(); print(g.sort_values('sel').to_string())
print(d[d.s=='MIL-CREDA'].groupby('t')[['sel','med','best','nul','rho']].mean().round(3).to_string())
print('ALL',d[['sel','med','best','nul','rho']].mean().round(3).to_dict())
