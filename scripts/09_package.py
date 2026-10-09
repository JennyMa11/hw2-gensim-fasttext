"""Build a compact, checksum-indexed submission excluding raw corpus/models."""
import hashlib
import zipfile
from common import ROOT

files=[]
for path in ROOT.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts:
        continue
    relative=path.relative_to(ROOT)
    if relative.parts[0]=='models':
        continue
    if relative.parts[0]=='data' and not (path.name=='selection.json' or path.name.startswith(('train_','test_'))):
        continue
    if path.name=='MANIFEST.sha256':
        continue
    files.append(path)
files.sort()
manifest=ROOT/'MANIFEST.sha256'
manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf-8')
target=ROOT.parent/'hw2_submission.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[manifest]:
        z.write(p,'hw2/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for p in files:
        actual=z.read('hw2/'+p.relative_to(ROOT).as_posix())
        assert hashlib.sha256(actual).digest()==hashlib.sha256(p.read_bytes()).digest()
print(f'Created {target}: {target.stat().st_size:,} bytes, {len(files)+1} files; all checksums verified.')
