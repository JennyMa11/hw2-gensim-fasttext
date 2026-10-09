"""IV/OOV neighbors, predeclared word groups, and direct subword reconstruction."""
from itertools import combinations, product
import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models.fasttext import ft_ngram_hashes
from common import RESULTS, MODELS, GROUPS, read_json, save_json

freq = read_json(RESULTS / 'word_frequencies.json')
queries = ['银行', '学校', '电脑', '经济', '足球', '计算机']
oov = ['足球运动员', '量子计算芯片', '经济全球化']
results = {}
for name, cls in [('word2vec', Word2Vec), ('fasttext', FastText)]:
    model = cls.load(str(MODELS / f'{name}.model'))
    wv = model.wv
    iv = {w:dict(frequency=freq.get(w, 0), neighbors=wv.most_similar(w, topn=5)) for w in queries if w in wv.key_to_index}
    out = {}
    for word in oov:
        in_vocab = word in wv.key_to_index
        entry = dict(frequency=freq.get(word,0), in_vocabulary=in_vocab)
        if in_vocab or name == 'fasttext':
            v = wv[word]
            entry.update(norm=float(np.linalg.norm(v)), finite=bool(np.isfinite(v).all()), neighbors=wv.most_similar(word,topn=5))
        else:
            try:
                wv[word]
            except KeyError:
                entry['error'] = 'KeyError'
        out[word] = entry
    # Only IV words in both models, preventing OOV availability from confounding this comparison.
    assert all(w in wv.key_to_index for words in GROUPS.values() for w in words)
    labels = list(GROUPS)
    matrix = []
    within, between = [], []
    for i, a in enumerate(labels):
        row = []
        for j, b in enumerate(labels):
            pairs = list(combinations(GROUPS[a],2)) if a == b else list(product(GROUPS[a],GROUPS[b]))
            sims = [float(wv.similarity(x,y)) for x,y in pairs]
            row.append(float(np.mean(sims)))
            if i == j:
                within.extend(sims)
            elif i < j:
                between.extend(sims)
        matrix.append(row)
    results[name] = dict(queries=iv, oov=out, groups=GROUPS, group_order=labels, similarity_matrix=matrix,
                         within_mean=float(np.mean(within)), between_mean=float(np.mean(between)),
                         gap=float(np.mean(within)-np.mean(between)), within_pairs=len(within), between_pairs=len(between))
    if name == 'fasttext':
        reconstruction = {}
        for word in ['足球', '计算机']+oov:
            hashes = ft_ngram_hashes(word, wv.min_n, wv.max_n, wv.bucket)
            if word in wv.key_to_index:
                v = wv.vectors_vocab[wv.key_to_index[word]].copy()
                for h in hashes:
                    v += wv.vectors_ngrams[h]
                v /= len(hashes)+1
            else:
                v = np.zeros(wv.vector_size, dtype=np.float32)
                for h in hashes:
                    v += wv.vectors_ngrams[h]
                v /= len(hashes)
            err = float(np.max(np.abs(v-wv[word])))
            assert err < 1e-5
            reconstruction[word] = dict(ngrams=len(hashes), max_absolute_error=err)
        results[name]['subword_reconstruction'] = reconstruction
    del model
save_json(RESULTS / 'embedding_analysis.json', results)
print({k: {p: v[p] for p in ['within_mean','between_mean','gap']} for k,v in results.items()}, flush=True)
