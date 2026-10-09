"""Train the two embeddings on identical sentences and export composed vectors."""
import os
assert os.environ.get('PYTHONHASHSEED') == '0', 'Run with PYTHONHASHSEED=0'
import logging
import platform
import sys
import time
import importlib.metadata
import gensim
from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence
from common import DATA, RESULTS, MODELS, EMBEDDING, save_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
stats = {}
for name, cls in [('word2vec', Word2Vec), ('fasttext', FastText)]:
    start = time.perf_counter()
    params = dict(EMBEDDING)
    if name == 'fasttext':
        params.update(min_n=2, max_n=4, bucket=200000)
    model = cls(sentences=LineSentence(str(DATA / 'sentences.txt')), **params)
    elapsed = time.perf_counter() - start
    model.save(str(MODELS / f'{name}.model'))
    model.wv.save_word2vec_format(str(MODELS / f'{name}.vec'), binary=False)
    stats[name] = dict(seconds=elapsed, vocabulary=len(model.wv), params=params)
    save_json(RESULTS / 'embedding_training.json', stats)
    print(f'FINISHED {name}: {elapsed:.2f}s, {len(model.wv)} words', flush=True)
    del model
environment = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                   pythonhashseed=os.environ['PYTHONHASHSEED'],
                   packages={p:importlib.metadata.version(p) for p in
                       ['numpy','scipy','gensim','fasttext-wheel','scikit-learn','matplotlib']})
save_json(RESULTS / 'environment.json', environment)
