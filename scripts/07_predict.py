"""Predict a space-tokenized Chinese body with the saved 8-class model."""
import argparse
import fasttext
from common import MODELS

p=argparse.ArgumentParser(description='输入已分词正文（空格分隔），不含标题或标签。')
p.add_argument('--text', required=True)
p.add_argument('--model', default=str(MODELS/'classifier_8_fasttext.bin'))
args=p.parse_args()
tokens=args.text.split()[:500]
if not tokens:
    p.error('正文不能为空')
model=fasttext.load_model(args.model)
labels,probs=model.predict(' '.join(tokens),k=3)
for label,prob in zip(labels,probs):
    # fastText adds a numerical epsilon, so its returned scores can exceed 1 slightly.
    print(f'{label.removeprefix("__label__")}\t{min(1.0, max(0.0, float(prob))):.6f}')
