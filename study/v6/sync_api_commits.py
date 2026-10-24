"""Import verified API-created commits while retaining the identical working tree."""
from pathlib import Path
import datetime,hashlib,itertools,json,subprocess
V=Path(__file__).resolve().parent;repo=V.parents[1]/'github/SPLA'
receipt=json.loads((V/'api_publication_receipt.json').read_text(encoding='utf-8-sig'))
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
assert subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip()==receipt['tree']
for obj in receipt['commits']:
    times={k:int(datetime.datetime.fromisoformat(obj[k]['date'].replace('Z','+00:00')).timestamp()) for k in ('author','committer')}
    found=None
    for at,ct,tail in itertools.product(('+0000','+0800'),('+0000','+0800'),('','\n')):
        lines=['tree '+obj['tree']['sha'],*['parent '+p['sha'] for p in obj['parents']]]
        for role,tz in [('author',at),('committer',ct)]:
            person=obj[role];lines.append(f"{role} {person['name']} <{person['email']}> {times[role]} {tz}")
        raw=('\n'.join(lines)+'\n\n'+obj['message']+tail).encode('ascii')
        digest=hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if digest==obj['sha']:found=raw;break
    assert found is not None,'Could not reconstruct the published commit exactly'
    written=subprocess.check_output(['git','hash-object','-t','commit','-w','--stdin'],cwd=repo,input=found).decode().strip()
    assert written==obj['sha']
subprocess.run(['git','update-ref','refs/heads/main',receipt['published_head'],receipt['local_previous_head']],cwd=repo,check=True)
subprocess.run(['git','update-ref','refs/remotes/origin/main',receipt['published_head']],cwd=repo,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
print('Synchronized local branch to published commits; tree and working files unchanged:',receipt['published_head'])
