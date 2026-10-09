"""Download, verify, parse, weak-label, split, then exclude held-out articles."""
import csv
import hashlib
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from common import DATA, RESULTS, CLASSES, KEYWORDS, GROUPS, save_json

COMMIT = '4020ecc558c554c6f3c9bd81a606d2197e28e628'
URL = f'https://raw.githubusercontent.com/chenhui-bupt/PeopleDaily1998/{COMMIT}/199801.zip'
SHA = '17474bbf2b360921f2eae0089fd83e22601e5a0649943c3f59403650997b4ed2'
archive = DATA / '199801.zip'
if not archive.exists():
    urllib.request.urlretrieve(URL, archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == SHA, 'Archive checksum mismatch'
raw = defaultdict(list)
counts = Counter()

def parse_token(token):
    # Inner token annotations precede an optional ]entity suffix. Keep component words.
    token = token.lstrip('[').split(']', 1)[0]
    match = re.fullmatch(r'(.+?)/([A-Za-z]+)(?:/.*)?', token)
    if not match:
        counts['unparsed_tokens'] += 1
        return None
    word, pos = match.groups()
    if token.count('/') > 1:
        counts['multiple_slashes'] += 1
    assert '/' not in word or word == '/', token
    return word, pos

with zipfile.ZipFile(archive) as z:
    members = sorted(n for n in z.namelist() if re.fullmatch(r'199801/19980[1-6]\.txt', n))
    assert len(members) == 6
    for member in members:
        for line in z.read(member).decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            m = re.match(r'^(\d{8}-\d{2}-\d{3})-(\d{3})/m\s+(.*)', line)
            if not m:
                counts['unparsed_lines'] += 1
                continue
            aid, sequence, text = m.groups()
            tokens = [x for x in (parse_token(t) for t in text.split()) if x]
            raw[aid].append((int(sequence), tokens))
            counts['parsed_lines'] += 1

articles = []
seen = set()
for aid, lines in sorted(raw.items()):
    lines.sort(key=lambda x: x[0])
    title_tokens = lines[0][1] if lines[0][0] == 1 else []
    if not title_tokens:
        counts['missing_title'] += 1
    body_lines = lines[1:] if title_tokens else lines
    body = [w for _, ts in body_lines for w, p in ts if p.lower() != 'w']
    # Drop punctuation and use original punctuation to recover sentence boundaries.
    sentences = []
    for _, ts in lines:
        sentence = []
        for w, p in ts:
            if p.lower() == 'w':
                if any(c in w for c in '。！？；!?;'):
                    if len(sentence) >= 3:
                        sentences.append(sentence)
                    sentence = []
                continue
            sentence.append(w)
        if len(sentence) >= 3:
            sentences.append(sentence)
    if len(body) < 30:
        counts['short_body_articles'] += 1
        continue
    body_hash = hashlib.sha256(' '.join(body).encode()).hexdigest()
    if body_hash in seen:
        counts['duplicate_body_articles'] += 1
        continue
    seen.add(body_hash)
    articles.append(dict(id=aid, title=''.join(w for w, _ in title_tokens),
                         body=body, sentences=sentences, body_sha256=body_hash))

candidates = defaultdict(list)
for article in articles:
    title = article['title']
    matches = {c: [k for k in KEYWORDS[c] if k in title] for c in CLASSES}
    hit = [c for c in CLASSES if matches[c]]
    if len(hit) > 1:
        counts['cross_topic_candidates'] += 1
    if len(hit) == 1 and 4 <= len(title) <= 80 and len(article['body']) >= 80:
        label = hit[0]
        candidates[label].append(dict(id=article['id'], label=label, title=title,
                                      hits=matches[label], body_sha256=article['body_sha256']))

selection = []
for label in CLASSES:
    rows = sorted(candidates[label], key=lambda x: hashlib.sha256(('sample42:'+x['id']).encode()).hexdigest())
    assert len(rows) >= 60, (label, len(rows))
    selected = rows[:60]
    selected.sort(key=lambda x: hashlib.sha256(('split42:'+x['id']).encode()).hexdigest())
    for i, row in enumerate(selected):
        row['split'] = 'test' if i < 12 else 'train'
        selection.append(row)

test_ids = {a['id'] for a in selection if a['split'] == 'test'}
assert len(test_ids) == 96
with (DATA / 'articles.jsonl').open('w', encoding='utf-8') as f:
    for article in articles:
        import json
        f.write(json.dumps(article, ensure_ascii=False)+'\n')
with (DATA / 'sentences.txt').open('w', encoding='utf-8') as f:
    sentence_count = token_count = 0
    freq = Counter()
    for article in articles:
        if article['id'] in test_ids:
            continue
        for sentence in article['sentences']:
            f.write(' '.join(sentence)+'\n')
            sentence_count += 1
            token_count += len(sentence)
            freq.update(sentence)
with (RESULTS / 'sample_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id','label','split','title','hits','body_sha256'])
    w.writeheader()
    for row in selection:
        w.writerow(dict(row, hits='|'.join(row['hits'])))
save_json(DATA / 'selection.json', selection)
save_json(RESULTS / 'word_frequencies.json', dict(freq))
save_json(RESULTS / 'protocol.json', dict(classes=CLASSES, keywords=KEYWORDS, word_groups=GROUPS,
    sample_per_class=60, train_per_class=48, test_per_class=12,
    selection='SHA256(sample42:ID), first 60; SHA256(split42:ID), first 12 test', max_body_tokens=500))
stats = dict(source=URL, archive_sha256=SHA, months=members, raw_articles=len(raw),
             eligible_unique_articles=len(articles), parsing=dict(counts),
             candidates={c:len(candidates[c]) for c in CLASSES}, selected_articles=len(selection),
             excluded_pretraining_test_articles=len(test_ids), representation_articles=len(articles)-len(test_ids),
             representation_sentences=sentence_count, representation_tokens=token_count,
             raw_vocabulary=len(freq), vocabulary_min_count_3=sum(n>=3 for n in freq.values()),
             sentences_sha256=hashlib.sha256((DATA/'sentences.txt').read_bytes()).hexdigest())
save_json(RESULTS / 'corpus_stats.json', stats)
print(stats, flush=True)
