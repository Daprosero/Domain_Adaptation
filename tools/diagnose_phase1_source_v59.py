import sys,warnings,collections; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, torch
from MIL_CREDA_EEG import bag_data, assignment as A, model_store, controls, two_phase, search
from sklearn.metrics import roc_auc_score as AUC
b=bag_data.load_bags(None,None); T={t.name:t for t in A.TRANSFERS}; root=Path('MIL-CREDA-EEG/Models/Balladeer')
torch.set_num_threads(3)
def ba(y,p):
    y=np.array(y);h=(np.array(p)>=.5).astype(int); return float(np.mean([(h[y==c]==c).mean() for c in (0,1)]))
for name in ('L1->L11','L11->L1','Robots->CogniFit','CogniFit->Robots','L6->Robots','Robots->L6'):
    t=T[name]; a=A.assign_from_bags(t,b)
    for g in ('common','MIL-CREDA'):
        ys=[];ps=[]; err=None
        for r in range(4):
            path=model_store.model_path(root,t,r,search.PHASE_ONE_SEED,1,g)
            test=two_phase.bag_tensors(A.side_bags(a,b,'source',r,'test'))
            try:
                c=controls.source_only_from_store(path,t,test,device=torch.device('cpu'))
            except Exception as e: err=repr(e)[:200]; break
            if not c.get('applicable'): err='not applicable'; break
            for q in c['predictions']: ys.append(q['label']); ps.append(q['probability'])
        print(name,g,err or f'n={len(ys)} BA={ba(ys,ps):.3f} AUC={AUC(ys,ps):.3f} predYes={np.mean(np.array(ps)>=.5):.2f}',flush=True)
