"""Audit saved results against per-article predictions; render all figures."""
import hashlib
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from common import DATA, RESULTS, FIGURES, MODELS, CLASSES, read_json, save_json

font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'
from pathlib import Path
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
else:
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':220})
stats = read_json(RESULTS/'corpus_stats.json')
assert hashlib.sha256((DATA/'sentences.txt').read_bytes()).hexdigest()==stats['sentences_sha256']
selection = read_json(DATA/'selection.json')
articles = {a['id']:a for a in (json.loads(s) for s in (DATA/'articles.jsonl').read_text(encoding='utf-8').splitlines())}
assert len({r['id'] for r in selection})==480
for c in CLASSES:
    assert Counter(r['split'] for r in selection if r['label']==c)=={'train':48,'test':12}
test_ids = {r['id'] for r in selection if r['split']=='test'}
assert len({articles[r['id']]['body_sha256'] for r in selection})==480
reconstructed = hashlib.sha256()
for a in articles.values():
    if a['id'] not in test_ids:
        for s in a['sentences']:
            reconstructed.update((' '.join(s)+'\n').encode('utf-8'))
assert reconstructed.hexdigest()==stats['sentences_sha256']
preds = read_json(RESULTS/'predictions.json')
runs = read_json(RESULTS/'classification_runs.json')
for run in runs:
    part = [p for p in preds if all(p[k]==run[k] for k in ['classes','train_per_class','method','seed','repeat'])]
    assert len(part)==run['test_count'] and len({p['id'] for p in part})==run['test_count']
    y=[p['true'] for p in part]; yh=[p['predicted'] for p in part]
    assert abs(accuracy_score(y,yh)-run['accuracy'])<1e-12
    assert abs(f1_score(y,yh,labels=CLASSES[:run['classes']],average='macro',zero_division=0)-run['macro_f1'])<1e-12
    assert confusion_matrix(y,yh,labels=CLASSES[:run['classes']]).tolist()==run['confusion_matrix']
for k in [2,4,8]:
    train_lines = (DATA/f'train_{k}_48.txt').read_text().splitlines()
    test_lines = (DATA/f'test_{k}.txt').read_text().splitlines()
    assert len(train_lines)==k*48 and len(test_lines)==k*12
    if k>2:
        prev=k//2
        assert train_lines[:prev*48]==(DATA/f'train_{prev}_48.txt').read_text().splitlines()
        assert test_lines[:prev*12]==(DATA/f'test_{prev}.txt').read_text().splitlines()
for n in [12,24]:
    small=(DATA/f'train_8_{n}.txt').read_text().splitlines()
    large=(DATA/'train_8_48.txt').read_text().splitlines()
    assert set(small)<=set(large)
vector_hashes={}
for name in ['word2vec','fasttext']:
    path=MODELS/f'{name}.vec'
    with path.open(encoding='utf-8') as f:
        assert list(map(int,next(f).split()))==[stats['vocabulary_min_count_3'],100]
        assert sum(1 for _ in f)==stats['vocabulary_min_count_3']
    vector_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
assert vector_hashes['word2vec']!=vector_hashes['fasttext']
import fasttext
saved_model=fasttext.load_model(str(MODELS/'classifier_8_fasttext.bin'))
for p in preds:
    if p['classes']==8 and p['train_per_class']==48 and p['method']=='fasttext' and p['seed']==42 and not p['repeat']:
        labels,_=saved_model.predict(' '.join(articles[p['id']]['body'][:500]))
        assert labels[0].removeprefix('__label__')==p['predicted']
del saved_model
save_json(RESULTS/'verification.json',dict(status='PASS', checked_runs=len(runs), checked_predictions=len(preds),
    exported_vec_sha256=vector_hashes,
    checks=['archive/corpus SHA256','480 unique article IDs and body hashes','balanced article splits',
            'test articles completely absent from representation corpus','nested 2/4/8 splits and 12/24/48 subsets',
            'all saved accuracy, macro F1, and confusion matrices recomputed from predictions',
            'exported vector dimensions/counts/checksums','saved classifier reload reproduces all 96 test predictions']))

summary=read_json(RESULTS/'classification_summary.json')
colors={'scratch':'#D48A50','fasttext':'#176B75','word2vec':'#7280B4'}
names={'scratch':'从零训练','fasttext':'FastText 词向量','word2vec':'Word2Vec 词向量'}
fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for ax,metric,title in zip(axs,['accuracy','macro_f1'],['准确率','宏平均 F1']):
    x=np.arange(3)
    for offset,method in [(-.18,'scratch'),(.18,'fasttext')]:
        rows=[next(s for s in summary if s['classes']==k and s['train_per_class']==48 and s['method']==method) for k in [2,4,8]]
        vals=[s[metric+'_mean'] for s in rows]
        ax.bar(x+offset,vals,.34,yerr=[s[metric+'_std'] for s in rows],capsize=3,color=colors[method],label=names[method])
        for pos,val in zip(x+offset,vals):
            ax.text(pos,val+.025,f'{val:.3f}',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['2 类','4 类','8 类'],ylim=(0,1.12),title=title,ylabel='分数')
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
axs[0].legend(loc='lower left',fontsize=8)
fig.suptitle('每类 48 篇训练；3 个分类器种子的均值 ± 样本标准差',fontsize=11)
fig.savefig(FIGURES/'classification_scores.png');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
for ax,method in zip(axs,['scratch','fasttext']):
    run=next(r for r in runs if r['classes']==8 and r['train_per_class']==48 and r['method']==method and r['seed']==42 and not r['repeat'])
    cm=np.array(run['confusion_matrix'])
    im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=12)
    for i in range(8):
        for j in range(8): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>6 else '#243144',fontsize=9)
    ax.set(xticks=range(8),yticks=range(8),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='预测类别',ylabel='规则标签',title=f"{names[method]}：{run['correct']}/96")
    ax.tick_params(axis='x',rotation=35)
fig.suptitle('8 类混淆矩阵（seed=42；每类测试 12 篇）',fontsize=12)
fig.savefig(FIGURES/'classification_confusion_8.png');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(8.8,3.5),layout='constrained')
emb=read_json(RESULTS/'embedding_analysis.json')
for ax,name in zip(axs,['word2vec','fasttext']):
    e=emb[name]; matrix=np.array(e['similarity_matrix']);labels=e['group_order']
    ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.65 else '#162639')
    ax.set(xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,title=name)
fig.suptitle('预先指定词组的平均余弦相似度；对角线排除自相似度',fontsize=11)
fig.savefig(FIGURES/'word_group_similarity.png');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for ax,metric,title in zip(axs,['accuracy','macro_f1'],['准确率','宏平均 F1']):
    for method in ['scratch','fasttext']:
        rows=[next(s for s in summary if s['classes']==8 and s['train_per_class']==n and s['method']==method) for n in [12,24,48]]
        ax.errorbar([12,24,48],[s[metric+'_mean'] for s in rows],yerr=[s[metric+'_std'] for s in rows],color=colors[method],marker='o',capsize=4,label=names[method])
    ax.set(xticks=[12,24,48],ylim=(0,1.04),xlabel='每类训练文章数',ylabel=title)
    ax.grid(alpha=.16)
axs[0].legend(fontsize=8)
fig.suptitle('8 类学习曲线：固定 96 篇测试文章；3 个种子均值 ± 标准差',fontsize=11)
fig.savefig(FIGURES/'learning_curve.png');plt.close(fig)
print('Verified all results and generated four figures.',flush=True)
