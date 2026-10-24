"""One local task continuation: wait for this queue, export, validate and publish once.

No scheduler, recurring automation, retries of training or remote credential copying.
All GitHub credentials remain in the existing local credential manager.
"""
import argparse
import datetime
import json
import msvcrt
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument('--remote-study', required=True)
args = ap.parse_args()
V = Path(__file__).resolve().parent
workspace = V.parents[1]
repo = workspace / 'github/SPLA'
lock = (V / 'watch_once.lock').open('a+b')
lock.seek(0); lock.write(b'1'); lock.flush(); lock.seek(0)
msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
assert not (V / 'watch_once_status.json').exists(), 'One-shot watcher already launched; inspect before resuming'
remote = args.remote_study
assert remote.endswith('/experiments/v6_fusion_diagnostics_20261024')
q = shlex.quote
state = {'phase': 'waiting_for_experiment', 'pid': __import__('os').getpid(), 'recurring': False,
         'remote_study': remote, 'last_observed': None}
def save():
    state['updated_at'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    tmp = V / 'watch_once_status.tmp'; tmp.write_text(json.dumps(state, indent=2)); tmp.replace(V / 'watch_once_status.json')
def call(cmd, timeout=120, capture=True):
    r = subprocess.run([str(x) for x in cmd], cwd=workspace, capture_output=capture, text=True,
                       encoding='utf-8', errors='replace', timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'Command failed ({r.returncode}): {cmd[0]}\n{r.stderr[-4000:] if capture else "See log"}')
    return r.stdout if capture else ''
def remote_python(script):
    return ['ssh', 'A100', q(remote + '/venv/bin/python') + ' ' + q(remote + '/' + script)]
def update_private(text):
    date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    p = workspace / 'handoff.md'
    old = p.read_text(encoding='utf-8')
    prefix = '# Handoff\n'
    entry = f'\n## {date} V6 one-shot continuation\n\n{text}\n'
    p.write_text(prefix + entry + old[len(prefix):] if old.startswith(prefix) else entry + old, encoding='utf-8')
save()
try:
    failures = 0
    dead_controller_polls = 0
    while True:
        try:
            snap = json.loads(call(remote_python('snapshot.py')))
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired):
            failures += 1
            if failures >= 10:
                raise
            time.sleep(60); continue
        state['last_observed'] = snap; save()
        assert snap['step1']['phase'] != 'failed' and snap['step2']['phase'] != 'failed', 'Experiment gate failed; no automatic restart'
        dead = any(snap[s]['phase'] not in ('complete', 'complete_report_ready') and not snap[s].get('controller_process') for s in ('step1', 'step2'))
        dead_controller_polls = dead_controller_polls + 1 if dead else 0
        assert dead_controller_polls < 3, 'An unfinished remote controller disappeared; inspect without restarting jobs'
        if snap['step2']['phase'] == 'complete_report_ready':
            assert snap['step1']['phase'] == 'complete'
            break
        time.sleep(60)
    assert not call(['git', '-C', repo, 'status', '--porcelain']).strip(), 'Repository has edits; preserve them and stop publication'
    base = call(['git', '-C', repo, 'rev-parse', 'HEAD']).strip()
    state['phase'] = 'exporting_complete_results'; save()
    call(remote_python('export_study.py'), timeout=900)
    call(['scp', 'A100:' + remote + '/publication_export.zip', V / 'publication_export.zip'], timeout=900)
    call([sys.executable, V / 'finish_publication.py'], timeout=180)
    call([sys.executable, repo / 'scripts/check_release.py', '--write-manifest'], timeout=180)
    call(['git', '-C', repo, 'diff', '--check'])
    call(['git', '-C', repo, 'add', 'artifacts/v6', 'artifacts/predictions/v6', 'models/v6',
          'README.md', 'artifacts/records/publication_manifest.json'])
    call(['git', '-C', repo, 'commit', '-m', 'results: publish complete fusion diagnostics and RGB baseline'])
    state['phase'] = 'publishing_verified_results'; save()
    call([sys.executable, V / 'prepare_api_publication.py', base], timeout=180)
    call(['pwsh', '-NoProfile', '-File', V / 'publish_via_api.ps1'], timeout=1800)
    call([sys.executable, V / 'sync_api_commits.py'])
    receipt = json.loads((V / 'api_publication_receipt.json').read_text(encoding='utf-8-sig'))
    url = 'https://api.github.com/repos/linger00333/SPLA/git/trees/' + receipt['tree'] + '?recursive=1'
    command = "Invoke-RestMethod -Uri '" + url + "' | ConvertTo-Json -Depth 80 | Set-Content -LiteralPath '" + str(V / 'github_tree_final.json').replace("'", "''") + "' -Encoding utf8"
    call(['pwsh', '-NoProfile', '-Command', command])
    call([sys.executable, V / 'verify_publication.py', V / 'github_tree_final.json'], timeout=180)
    try:
        call(remote_python('cleanup_study.py'), timeout=180)
    except Exception as exc:
        state['cleanup_warning'] = str(exc)
    (V / 'source_bundle.zip').unlink(missing_ok=True)
    (V / 'publication_export.zip').unlink(missing_ok=True)
    state['phase'] = 'complete_and_published'; state['published_commit'] = receipt['published_head']; save()
    summary = json.loads((V / 'final_summary.json').read_text())
    pooled = summary['complementarity']['pooled']
    update_private('V6 completed and published to https://github.com/linger00333/SPLA at ' + receipt['published_head'] +
                   '. Both blank-input runs and RGB inference cover 67 sequences / 61999 frames each. RGB scratch training completed 100 epochs. '+
                   'Fixed-detector oracle_gain=' + str(pooled['oracle_gain']) + ', rescue_share=' + str(pooled['rescue_share']) +
                   '. Report: github/SPLA/artifacts/v6/REPORT.md. Full Git blob/manifest verification passed. Test unevaluated; recurring automation remains off. No follow-up training launched. This one-shot local continuation has finished.')
    result_path = workspace / 'experiment/results.md'
    old = result_path.read_text(encoding='utf-8')
    result_path.write_text(old + '\nV6 complete: see [English final report](../github/SPLA/artifacts/v6/REPORT.md). '+
                           'Fixed-detector oracle_gain=' + str(pooled['oracle_gain']) + '; rescue_share=' + str(pooled['rescue_share']) +
                           '. Single 100-epoch RGB scratch run; all three prediction sets 67/61999; no test or follow-up training. Published ' + receipt['published_head'] + '.\n', encoding='utf-8')
except BaseException:
    state['phase'] = 'failed_needs_inspection'; state['error'] = traceback.format_exc(); save()
    update_private('V6 one-shot continuation stopped and requires inspection. No training was restarted. Read experiment/v6_fusion_diagnostics_20261024/watch_once_status.json and logs; do not duplicate the remote controllers or local watcher. Completed experiment evidence is retained.')
    raise
finally:
    lock.close()
