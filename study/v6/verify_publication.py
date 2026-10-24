from pathlib import Path
import datetime,hashlib,json,subprocess,sys
V=Path(__file__).resolve().parent;repo=V.parents[1]/'github/SPLA'
api=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
assert not api['truncated']
remote={row['path']:row['sha'] for row in api['tree'] if row['type']=='blob'}
local={}
for row in subprocess.check_output(['git','ls-tree','-r','-z','HEAD'],cwd=repo).split(b'\0'):
    if not row:continue
    meta,path=row.split(b'\t',1);path=path.decode('ascii');local[path]=meta.split()[2].decode('ascii')
assert local==remote
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
manifest=json.loads((repo/'artifacts/records/publication_manifest.json').read_text())
reader=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for row in manifest['files']:
    reader.stdin.write(('HEAD:'+row['path']+'\n').encode('ascii'));reader.stdin.flush()
    header=reader.stdout.readline().split();assert len(header)==3
    raw=reader.stdout.read(int(header[2]));assert reader.stdout.read(1)==b'\n'
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
reader.stdin.close();assert reader.wait()==0
proof={'verified_at':datetime.datetime.now().isoformat(),'repository':'https://github.com/linger00333/SPLA','commit':head,'tree':api['sha'],'verified_files':len(local),'all_blob_hashes_match':True,'all_filenames_ascii':True,'full_content_ascii_and_manifest_check':'check_release.py passed before commit','working_tree_clean':True}
proof['all_manifest_hashes_match_Git_stored_bytes']=True
(V/('github_upload_verified_'+head[:7]+'.json')).write_text(json.dumps(proof,indent=2)+'\n',encoding='ascii')
print(json.dumps(proof,indent=2))
