"""Nested 2/4/8-class comparisons, training-size ablation, and repeat check."""
import csv
import gc
import hashlib
import json
import time
from collections import Counter
import fasttext
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from common import DATA, RESULTS, MODELS, CLASSES, SUPERVISED, read_json, save_json

articles = {a['id']:a for a in (json.loads(s) for s in (DATA/'articles.jsonl').read_text(encoding='utf-8').splitlines())}
selection = read_json(DATA / 'selection.json')
predictions, runs = [], []
def rows_for(k, split, ntrain=48):
    rows = []
    for label in CLASSES[:k]:
        part = [r for r in selection if r['label']==label and r['split']==split]
        part.sort(key=lambda r: hashlib.sha256(('learning42:'+r['id']).encode()).hexdigest())
        rows.extend(part[:ntrain] if split == 'train' else part)
    return rows

def text_for(row):
    words = articles[row['id']]['body'][:500]
    assert all(not w.startswith('__label__') for w in words)
    return ' '.join(words)

def train_eval(k, method, seed, ntrain=48, repeat=False):
    train = rows_for(k,'train',ntrain)
    test = rows_for(k,'test')
    assert len(train)==k*ntrain and len(test)==k*12
    assert not {r['id'] for r in train} & {r['id'] for r in test}
    train_path = DATA/f'train_{k}_{ntrain}.txt'
    train_path.write_text(''.join(f"__label__{r['label']} {text_for(r)}\n" for r in train), encoding='utf-8')
    (DATA/f'test_{k}.txt').write_text(''.join(f"__label__{r['label']} {text_for(r)}\n" for r in test), encoding='utf-8')
    params = dict(SUPERVISED, input=str(train_path), seed=seed)
    if method != 'scratch':
        params['pretrainedVectors'] = str(MODELS/f'{method}.vec')
    start = time.perf_counter()
    model = fasttext.train_supervised(**params)
    seconds = time.perf_counter()-start
    true, pred = [], []
    for row in test:
        labels, probs = model.predict(text_for(row),k=1)
        label = labels[0].removeprefix('__label__')
        true.append(row['label']); pred.append(label)
        predictions.append(dict(classes=k, train_per_class=ntrain, method=method, seed=seed,
            repeat=repeat, id=row['id'], true=row['label'], predicted=label, probability=float(probs[0]), title=row['title']))
    run = dict(classes=k, train_per_class=ntrain, method=method, seed=seed, repeat=repeat,
        train_count=len(train), test_count=len(test), correct=sum(a==b for a,b in zip(true,pred)),
        accuracy=float(accuracy_score(true,pred)), macro_f1=float(f1_score(true,pred,labels=CLASSES[:k],average='macro',zero_division=0)),
        seconds=seconds, confusion_matrix=confusion_matrix(true,pred,labels=CLASSES[:k]).tolist(),
        classification_report=classification_report(true,pred,labels=CLASSES[:k],output_dict=True,zero_division=0),
        dictionary_size=len(model.get_words()),
        parameters={key:val for key,val in params.items() if key not in ['input','pretrainedVectors']})
    if k==8 and method=='fasttext' and seed==42 and ntrain==48 and not repeat:
        model.save_model(str(MODELS/'classifier_8_fasttext.bin'))
    runs.append(run)
    save_json(RESULTS/'classification_runs.json',runs)
    save_json(RESULTS/'predictions.json',predictions)
    print(f"k={k}, n={ntrain}, {method}, seed={seed}, repeat={repeat}: {run['correct']}/{len(test)}, F1={run['macro_f1']:.4f}, {seconds:.2f}s",flush=True)
    del model
    gc.collect()
    return pred

for k in [2,4,8]:
    for seed in [42,43,44]:
        for method in ['scratch','fasttext']:
            train_eval(k,method,seed)
# Extension: keep all other hyperparameters fixed; compare W2V initialization too.
for seed in [42,43,44]:
    train_eval(8,'word2vec',seed)
# Extension: nested subsets of the SAME 48 training articles per class.
for n in [12,24]:
    for seed in [42,43,44]:
        for method in ['scratch','fasttext']:
            train_eval(8,method,seed,ntrain=n)
repeat_pred = train_eval(8,'fasttext',42,repeat=True)
original = [p['predicted'] for p in predictions if p['classes']==8 and p['method']=='fasttext'
            and p['seed']==42 and p['train_per_class']==48 and not p['repeat']]
assert repeat_pred == original

summary = []
for k in [2,4,8]:
    for n in ([12,24,48] if k==8 else [48]):
        for method in (['scratch','fasttext','word2vec'] if k==8 and n==48 else ['scratch','fasttext']):
            part = [r for r in runs if r['classes']==k and r['train_per_class']==n and r['method']==method and not r['repeat']]
            summary.append(dict(classes=k,train_per_class=n,method=method,seeds=[r['seed'] for r in part],
                accuracy_mean=float(np.mean([r['accuracy'] for r in part])), accuracy_std=float(np.std([r['accuracy'] for r in part],ddof=1)),
                macro_f1_mean=float(np.mean([r['macro_f1'] for r in part])), macro_f1_std=float(np.std([r['macro_f1'] for r in part],ddof=1))))
save_json(RESULTS/'classification_summary.json',summary)
# Matched comparison on the fixed 96 test items, seed 42. This is not label-quality uncertainty.
a = {p['id']:p for p in predictions if p['classes']==8 and p['method']=='fasttext' and p['seed']==42 and p['train_per_class']==48 and not p['repeat']}
b = {p['id']:p for p in predictions if p['classes']==8 and p['method']=='scratch' and p['seed']==42 and p['train_per_class']==48}
diff = np.array([int(a[i]['true']==a[i]['predicted'])-int(b[i]['true']==b[i]['predicted']) for i in sorted(a)])
rng = np.random.default_rng(42)
boot = rng.choice(diff, size=(10000,len(diff)),replace=True).mean(axis=1)
wins, losses = int((diff==1).sum()), int((diff==-1).sum())
save_json(RESULTS/'paired_comparison.json',dict(seed=42, sample_size=len(diff), pretraining_only_correct=wins,
    scratch_only_correct=losses, accuracy_difference=float(diff.mean()),
    paired_bootstrap_accuracy_difference_ci95=np.quantile(boot,[0.025,0.975]).tolist(),
    exact_mcnemar_pvalue=float(binomtest(wins,wins+losses,0.5).pvalue) if wins+losses else 1.0,
    repeat_predictions_identical=repeat_pred==original, bootstrap_replicates=10000))
with (RESULTS/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
# Coverage is measured on the actual truncated body inputs.
vocab = {line.split(' ',1)[0] for line in (MODELS/'fasttext.vec').read_text().splitlines()[1:]}
coverage = {}
for split in ['train','test']:
    words = [w for r in selection if r['split']==split for w in articles[r['id']]['body'][:500]]
    coverage[split] = dict(tokens=len(words), covered_tokens=sum(w in vocab for w in words),
        token_coverage=sum(w in vocab for w in words)/len(words), types=len(set(words)),
        covered_types=len(set(words)&vocab), type_coverage=len(set(words)&vocab)/len(set(words)))
save_json(RESULTS/'pretrained_coverage.json',coverage)
