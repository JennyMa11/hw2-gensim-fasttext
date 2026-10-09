"""Post-hoc optimization diagnostic; does NOT replace or tune the main results."""
import csv
import gc
import time
import fasttext
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from common import DATA, RESULTS, MODELS, CLASSES, SUPERVISED, read_json, save_json

test=[]
for line in (DATA/'test_8.txt').read_text(encoding='utf-8').splitlines():
    label,text=line.split(' ',1)
    test.append((label.removeprefix('__label__'),text))
rows=[];predictions=[]
for r in read_json(RESULTS/'classification_runs.json'):
    if r['classes']==8 and r['train_per_class']==48 and r['method'] in ['scratch','fasttext'] and not r['repeat']:
        rows.append(dict(epoch=25,method=r['method'],seed=r['seed'],accuracy=r['accuracy'],macro_f1=r['macro_f1'],seconds=r['seconds']))
for epoch in [100,200]:
    for seed in [42,43,44]:
        for method in ['scratch','fasttext']:
            params=dict(SUPERVISED,input=str(DATA/'train_8_48.txt'),epoch=epoch,seed=seed)
            if method=='fasttext':params['pretrainedVectors']=str(MODELS/'fasttext.vec')
            start=time.perf_counter(); model=fasttext.train_supervised(**params)
            elapsed=time.perf_counter()-start
            pred=[]; true=[]
            for i,(label,text) in enumerate(test):
                labels,probs=model.predict(text,k=1)
                p=labels[0].removeprefix('__label__');pred.append(p);true.append(label)
                predictions.append(dict(epoch=epoch,seed=seed,method=method,test_row=i,true=label,predicted=p,probability=float(probs[0])))
            r=dict(epoch=epoch,seed=seed,method=method,accuracy=float(accuracy_score(true,pred)),
                   macro_f1=float(f1_score(true,pred,labels=CLASSES,average='macro',zero_division=0)),seconds=elapsed)
            rows.append(r);print(r,flush=True)
            del model;gc.collect()
summary=[]
for epoch in [25,100,200]:
    for method in ['scratch','fasttext']:
        part=[r for r in rows if r['epoch']==epoch and r['method']==method]
        summary.append(dict(epoch=epoch,method=method,
            accuracy_mean=float(np.mean([r['accuracy'] for r in part])),accuracy_std=float(np.std([r['accuracy'] for r in part],ddof=1)),
            macro_f1_mean=float(np.mean([r['macro_f1'] for r in part])),macro_f1_std=float(np.std([r['macro_f1'] for r in part],ddof=1))))
save_json(RESULTS/'epoch_diagnostic.json',dict(protocol='Post-hoc diagnostic after inspecting primary results; no hyperparameter selection; main epoch=25 unchanged',runs=rows,summary=summary,predictions=predictions))
for epoch in [100,200]:
    for seed in [42,43,44]:
        for method in ['scratch','fasttext']:
            part=[p for p in predictions if p['epoch']==epoch and p['seed']==seed and p['method']==method]
            r=next(r for r in rows if r['epoch']==epoch and r['seed']==seed and r['method']==method)
            assert len(part)==96
            assert abs(accuracy_score([p['true'] for p in part],[p['predicted'] for p in part])-r['accuracy'])<1e-12
            assert abs(f1_score([p['true'] for p in part],[p['predicted'] for p in part],average='macro',labels=CLASSES,zero_division=0)-r['macro_f1'])<1e-12
print('Post-hoc predictions verified.',flush=True)
