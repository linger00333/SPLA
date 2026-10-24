"""Verify portable source text, compressed predictions, tensor headers and hashes."""
from pathlib import Path
import argparse,ast,gzip,hashlib,json,re,struct

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'artifacts/records/publication_manifest.json'
def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for k,v in value.items():yield from strings(k);yield from strings(v)
    elif isinstance(value,list):
        for v in value:yield from strings(v)

def check(write=False):
    files=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if set(rel.parts)&{'.git','__pycache__','.pytest_cache'} or p.suffix=='.pyc':continue
        if not p.is_file():continue
        assert str(rel).isascii(),f'Invalid filename: {rel}'
        raw=p.read_bytes()
        if p.suffix=='.safetensors':
            n=struct.unpack('<Q',raw[:8])[0]
            text=raw[8:8+n].decode('ascii');obj=json.loads(text)
        elif p.suffix=='.gz':
            text=gzip.decompress(raw).decode('ascii');obj=None
            if p.name.endswith('.json.gz'):obj=json.loads(text)
        else:
            text=raw.decode('ascii');obj=None
            if p.suffix=='.json':obj=json.loads(text)
            if p.suffix=='.py':
                tree=ast.parse(text,filename=str(rel))
                for node in ast.walk(tree):
                    if isinstance(node,ast.Constant) and isinstance(node.value,str):assert node.value.isascii(),f'Invalid decoded source text: {rel}'
        if obj is not None:assert all(s.isascii() for s in strings(obj)),f'Invalid decoded metadata: {rel}'
        assert not re.search(r'/(?:home|Users)/(?:lixl|13790)',text),f'Local path found: {rel}'
        assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',text),f'Credential-like content: {rel}'
        if p!=MANIFEST:files.append({'path':rel.as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    if write:
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        MANIFEST.write_text(json.dumps({'files':files,'text_policy':'ASCII filenames, source, configuration, decompressed predictions and model headers; tensor payloads are binary numerical data'},indent=2)+'\n',encoding='ascii',newline='\n')
    else:
        assert json.loads(MANIFEST.read_text())['files']==files,'Publication content differs from manifest'
    registry=json.loads((ROOT/'models/registry.json').read_text())
    inputs=json.loads((ROOT/'artifacts/records/input_manifest.json').read_text())
    for name,row in registry.items():
        directory=ROOT/'models/legacy'/name
        assert hashlib.sha256((directory/'model.safetensors').read_bytes()).hexdigest()==row['tensor_sha256'],name
        assert hashlib.sha256((directory/'config.json').read_bytes()).hexdigest()==row['config_sha256'],name
        assert row['source_checkpoint_sha256']==inputs['old_weights'][name],name
    print(f'Checked {len(files)} files; source, metadata, compressed content and hashes passed.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--write-manifest',action='store_true');args=ap.parse_args();check(args.write_manifest)
