import sqlite3,json,glob,sys,os,collections
import numpy as np, pandas as pd
sys.path.insert(0,'src')
from MIL_CREDA_EEG import bag_data
from sklearn.metrics import roc_auc_score as AUC
lab={k:int(v=='yes') for k,v in bag_data.load_bags(None,None).labels().items()}
def ba(y,p):
    h=(p>=.5).astype(int); return float(np.mean([(h[y==c]==c).mean() for c in (0,1)]))
core=('L1-to-L11','L11-to-L1','Robots-to-CogniFit','CogniFit-to-Robots','L6-to-Robots','Robots-to-L6')
R=[];T=[];rng=np.random.default_rng(1)
for f in sorted(glob.glob('MIL-CREDA-EEG/Results/hpo/*__phase2__*.db')):
    t,_,s,o=os.path.basename(f)[:-3].split('__')
    if t not in core: continue
    c=sqlite3.connect(f'file:{f}?mode=ro',uri=True); A=collections.defaultdict(dict)
    for tid,k,v in c.execute("select trial_id,key,value_json from trial_user_attributes where key in ('ensvProbabilities','ensvSubjects','pooledCentroidBalancedAccuracy','trainedEpochs','bestEpoch','reachedCap')"): A[tid][k]=json.loads(v)
    for tid,d in A.items():
        if 'ensvProbabilities' not in d: continue
        y=np.array([lab[x] for x in d['ensvSubjects']]); pr=np.asarray(d['ensvProbabilities'],float)[...,1]
        b=[ba(y,p) for p in pr]; a=[AUC(y,p) if len(set(np.round(p,9)))>1 else .5 for p in pr]; pos=[(p>=.5).mean() for p in pr]
        n=len(pr); best=int(np.argmax(b))
        T.append(dict(t=t,s=s,trained=d.get('trainedEpochs',n),n=n,sel=d.get('bestEpoch'),cap=bool(d.get('reachedCap')),bestBA=max(b),selBA=b[min(int(d.get('bestEpoch') or 0),n-1)],firstBA=b[0],lastBA=b[-1],bestAt=best,firstAUC=a[0],lastAUC=a[-1],maxAUC=max(a),medAUC=float(np.median(a)),firstPos=pos[0],lastPos=pos[-1],lastGain=b[-1]-max(b[:-1]) if n>1 else 0, atEnd=best>=n-2))
        for e in range(n): R.append((s,e,b[e],a[e],pos[e]))
d=pd.DataFrame(T); print('trials',len(d),'trained median',d.trained.median(),'p90',d.trained.quantile(.9),'at cap',round(d.cap.mean(),3),'selected epoch median',d.sel.median(),'selected at 0',round((d.sel==0).mean(),3))
print('BA first/sel/last/best',d[['firstBA','selBA','lastBA','bestBA']].mean().round(3).to_dict())
print('AUC first/median/last/max',d[['firstAUC','medAUC','lastAUC','maxAUC']].mean().round(3).to_dict())
print('best epoch at the stop edge',round(d.atEnd.mean(),3),'mean gain best over selected',round((d.bestBA-d.selBA).mean(),3),'pred-yes first/last',d[['firstPos','lastPos']].mean().round(3).to_dict())
r=pd.DataFrame(R,columns='s e ba auc pos'.split())
r['bin']=pd.cut(r.e,[-1,0,4,9,19,39,99]); print(r.groupby('bin',observed=True)[['ba','auc','pos']].mean().round(3).assign(n=r.groupby('bin',observed=True).size()).to_string())
print(d.groupby('s')[['trained','firstBA','selBA','lastBA','bestBA','firstAUC','lastAUC']].mean().round(3).to_string())
print(d[d.s=='MIL-CREDA'].groupby('t')[['trained','firstBA','selBA','lastBA','bestBA','firstAUC','lastAUC','maxAUC']].mean().round(3).to_string())
