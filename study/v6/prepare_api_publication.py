"""Prepare reviewed commits as text Git Data API operations (no credentials)."""
from pathlib import Path
import datetime
import base64
import json
import re
import subprocess
import sys
V = Path(__file__).resolve().parent
repo = V.parents[1] / 'github/SPLA'
git = lambda *a: subprocess.check_output(['git', *a], cwd=repo)
base = sys.argv[1]
commits = []
for ref in git('rev-list', '--reverse', base + '..HEAD').decode().splitlines():
    raw = git('cat-file', 'commit', ref).decode('ascii')
    header, message = raw.split('\n\n', 1)
    fields = {line.split(' ', 1)[0]: line.split(' ', 1)[1] for line in header.splitlines()}
    def identity(value):
        match = re.fullmatch(r'(.*?) <([^>]+)> (\d+) ([+-]\d{4})', value)
        name, email, seconds, tz = match.groups()
        offset = (int(tz[1:3])*60 + int(tz[3:])) * (1 if tz[0] == '+' else -1)
        date = datetime.datetime.fromtimestamp(int(seconds), datetime.timezone(datetime.timedelta(minutes=offset))).isoformat()
        return {'name': name, 'email': email, 'date': date}
    entries = []
    for line in git('diff', '--name-status', fields['parent'], ref).decode().splitlines():
        status, path = line.split('\t', 1)
        assert status in ('A', 'M')
        raw_blob = git('show', ref + ':' + path)
        entry = {'path': path, 'mode': '100644', 'type': 'blob'}
        try:
            entry['content'] = raw_blob.decode('ascii')
        except UnicodeDecodeError:
            entry['base64'] = base64.b64encode(raw_blob).decode('ascii')
        entries.append(entry)
    commits.append({'local_sha': ref, 'tree': fields['tree'], 'parent': fields['parent'], 'entries': entries,
                    'commit': {'message': message.rstrip('\n'), 'author': identity(fields['author']), 'committer': identity(fields['committer'])}})
assert commits
plan = {'repository': 'linger00333/SPLA', 'base': base, 'expected_final_tree': commits[-1]['tree'],
        'local_head': commits[-1]['local_sha'], 'commits': commits}
(V / 'api_publication_plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='ascii')
print('Prepared', len(commits), 'reviewed commits with explicit binary blob encoding where needed.')
