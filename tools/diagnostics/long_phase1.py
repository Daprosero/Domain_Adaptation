"""Phase 1 for 500 epochs with the current recipe: per epoch, the pooled inner source validation
(3 inner splits; label-free, could choose the epoch) and the unseen source test of the refit
(diagnostic only). Prints AUC and BA at 100, 200, 500, the best epoch of each curve, and the test
score at the epoch the inner validation would choose. Curves go to <out>.json."""
import json,sys,warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, torch
from sklearn.metrics import roc_auc_score as AUC, balanced_accuracy_score as BA
from MIL_CREDA_EEG import assignment as A, bag_data, campaign, model_store, search, spectral_trunk, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2
torch.set_num_threads(4)
EPOCHS=500
run=json.load(open('tools/kaggle/v59-k3/run-config.json'))['run']['kwargs']['config']
cfg=campaign.check_config({**{k:v for k,v in run.items() if k!='remoteInputs'},'device':'cpu'}); campaign.register(cfg['strategies'])
bags=bag_data.load_bags(None,None); dev=torch.device('cpu')
hyper=two_phase.check_hyper({**sc.HYPER,'epochs1':EPOCHS,'bagsPerStep':cfg['schedule']['bagsPerStep']})
def curve(train,valid,seed,r,name):
    with spectral_trunk.installed():
        torch.manual_seed(seed); g=torch.Generator().manual_seed(seed); c=int(train[0].windows.shape[1])
        m=model_store.build_model('MIL-CREDA',c,c,hyper); m.source_trunk.norm=sc.standard_of(m.source_trunk,train)
        with explicit_l2(sc.L2): ph1=two_phase.train_source_phase(m,train,valid,hyper,g,dev,seed=seed,fold=r,transfer=name)
    return [(rec['labels'],[p[1] for p in rec['probabilities']]) for rec in ph1['validation']]
def pooled(per_rot):          # per_rot: list over parts of list over epochs of (labels, probs)
    out=[]
    for e in range(EPOCHS):
        y=sum((p[e][0] for p in per_rot),[]); q=sum((p[e][1] for p in per_rot),[])
        q=list(np.nan_to_num(q,nan=.5)); out.append((AUC(y,q),BA(y,[v>.5 for v in q])))
    return out
seed,outp=int(sys.argv[1]),sys.argv[2]
res={}
for name in sys.argv[3:]:
    a=A.assign_from_bags({t.name:t for t in A.TRANSFERS}[name],bags); test,inner=[],[]
    for r in range(4):
        side=lambda role,**k: sc.prepared(two_phase.bag_tensors(A.side_bags(a,bags,'source',r,role,**k)))
        test.append(curve(side(A.REFIT_ROLE),side('test'),seed,r,name))
        for s in range(search.N_INNER):
            inner.append(curve(side('inner_train',split=s),side('inner_validation',split=s),seed,r,name))
    T,V=pooled(test),pooled(inner)
    pick=int(np.argmax([v[0] for v in V]))
    at=lambda C,e: {'AUC':round(C[e][0],3),'BA':round(C[e][1],3)}
    summary={'test at 100':at(T,99),'test at 200':at(T,199),'test at 500':at(T,EPOCHS-1),
             'test best epoch':int(np.argmax([t[0] for t in T]))+1,'test best':at(T,int(np.argmax([t[0] for t in T]))),
             'inner valid at 100':at(V,99),'inner valid at 200':at(V,199),'inner valid best epoch':pick+1,'inner valid best':at(V,pick),
             'test at the inner-chosen epoch':at(T,pick)}
    res[name]={'summary':summary,'test':T,'inner':V}
    print('RESULT seed',seed,name.split('->')[0],json.dumps(summary),flush=True)
json.dump(res,open(outp,'w'))
