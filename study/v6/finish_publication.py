"""Integrate a checked completed export and write an evidence-bounded English report."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import zipfile

V = Path(__file__).resolve().parent
repo = V.parents[1] / 'github/SPLA'
with zipfile.ZipFile(V / 'publication_export.zip') as z:
    assert z.testzip() is None
    for member in z.infolist():
        assert member.filename.startswith(('artifacts/v6/', 'artifacts/predictions/v6/', 'models/v6/'))
        dst = (repo / member.filename).resolve()
        assert dst.is_relative_to(repo.resolve())
        raw = z.read(member)
        if not member.filename.endswith('.safetensors'):
            (gzip.decompress(raw) if member.filename.endswith('.gz') else raw).decode('ascii')
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(raw)
        if member.filename.startswith('artifacts/v6/'):
            private = V / 'evidence' / Path(member.filename).relative_to('artifacts/v6')
            private.parent.mkdir(parents=True, exist_ok=True); private.write_bytes(raw)
A = repo / 'artifacts/v6'
R = A / 'reports'
snapshot = json.loads((A / 'export_snapshot.json').read_text())
assert snapshot['step1_phase'] == 'complete' and snapshot['step2_phase'] == 'complete_report_ready'
assert snapshot['training_completed'] and not snapshot['test_evaluated']
for s in ('step1', 'step2'):
    status = json.loads((A / f'{s}_status_snapshot.json').read_text())
    assert all(row['state'] == 'complete' for row in status['jobs'].values())
for file in ('coverage_ablation.json', 'coverage_rgb.json'):
    coverage = json.loads((R / file).read_text())
    assert all(x['n_sequences'] == 67 and x['frames'] == 61999 and not x['length_mismatches'] for x in coverage.values())
scores = json.loads((R / 'ablation_scores.json').read_text())
assert scores['n_sequences'] == 67
cp = json.loads((R / 'complementarity_val.json').read_text())
assert cp['videos'] == 67 and cp['coverage']['total_frames'] == 61999
assert sum(v['n'] for v in cp['per_video'].values()) == cp['pooled']['frames']
assert all(v['n'] == sum(v[k] for k in ('both', 'ir_only', 'rgb_only', 'neither')) for v in cp['per_video'].values())
recipe = json.loads((A / 'records/training_recipe_audit.json').read_text())
assert recipe['all_selected_recipe_fields_match_old_A0']
training = (A / 'training/A0rgb_s0/results.csv').read_text().strip().splitlines()
assert len(training) == 101
model = json.loads((A / 'records/rgb_model_export.json').read_text())
directory = repo / 'models/v6/A0rgb_s0'
assert hashlib.sha256((directory / 'model.safetensors').read_bytes()).hexdigest() == model['tensor_sha256']
assert hashlib.sha256((directory / 'config.json').read_bytes()).hexdigest() == model['config_sha256']
p = cp['pooled']
gain, ci = p['oracle_gain'], cp['oracle_gain_ci95']
clean_gain = cp['excluding_known_label_issues']['oracle_gain']
if gain is not None and gain >= .02 and ci and ci[0] > .005 and clean_gain is not None and clean_gain >= .02:
    decision = 'The fixed detector pair shows potentially useful complementarity under the predeclared exploratory bands. Propose one controlled fusion follow-up only after checking the known-issue sensitivity and RGB-to-IR box mapping; no follow-up training is launched.'
elif gain is not None and gain < .005 and ci and ci[1] < .02:
    decision = 'Complementarity between these fixed detectors is small under this pipeline. Prioritize frame-level error, size and resolution analysis before costly fusion training. This does not prove RGB is uninformative or that learned fusion cannot improve.'
else:
    decision = 'The diagnostic is inconclusive under the predeclared exploratory bands. Review sequence variation, known label issues and error cases before proposing a single next experiment; no training search is launched.'
lines = ['# V6 completed fusion diagnostics', '',
         'Two A1 blank-input validation runs, one independent 100-epoch scratch RGB-only training run, RGB validation inference and fixed-detector complementarity are complete. Every full inference run covers 67 videos and 61,999 frames without length gaps. Test was not evaluated. Only physical GPU 1 was used; no recurring automation or failed-pilot three-seed queue was started.', '',
         '## A1 blank-input intervention', '',
         'The original A1 weights, four-parameter prior tables, zero prior, thresholds and IR-coordinate scorer are retained. The unchanged scorer uses 10,000 paired sequence bootstrap draws. Flat 114 inputs are out of distribution; these differences do not identify a strict causal contribution or reliance upper bound.', '',
         '| Condition | SA | Penalized SA | AP50 |', '|---|---:|---:|---:|']
for name, row in scores['runs'].items():
    cells = ['null (undefined)' if row[k] is None else f'{row[k]:.6f}' for k in ('SA', 'SA_pen', 'AP50')]
    lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
lines += ['', '| Difference from A1 | SA point delta [95% CI] | Penalized SA point delta [95% CI] | AP50 point delta [95% CI] |', '|---|---|---|---|']
for key, comparison in scores['vs_baseline'].items():
    name = key.rsplit('-A1', 1)[0]
    cells = []
    for metric in ('SA', 'SA_pen', 'AP50'):
        if scores['runs'][name][metric] is None or scores['runs']['A1'][metric] is None:
            cells.append('null (undefined)')
            continue
        delta = scores['runs'][name][metric] - scores['runs']['A1'][metric]
        interval = comparison.get(metric, {}).get('ci95')
        if interval is None:
            cells.append(f'{delta:+.6f} [CI undefined]')
        else:
            lo, hi = interval
            cells.append(f'{delta:+.6f} [{lo:+.6f}, {hi:+.6f}]')
    lines.append('| ' + key + ' | ' + ' | '.join(cells) + ' |')
lines += ['', 'Undefined values from the unchanged legacy scorer, including AP when no candidate exists, are retained as null rather than converted to zero.', '', '## RGB-only baseline and fixed-detector selection', '',
          'A0rgb_s0 completed exactly 100 epochs with the old A0 scratch recipe: YOLO11n, seed 0, 640, batch 16 and workers 4. Its selected checkpoint uses validation performance in RGB labels. Training and validation use exactly the old 29,708/12,158 sampled images. The original sampling depends on IR existence, and the visible images are stretched to the IR dimensions. This is not a native-resolution RGB benchmark.', '',
          'The actual framework loader was checked against every RGB-normalized label; all images were hard-linked, source labels remained unchanged, and four real validation overlays were inspected. A decimal-format audit mismatch was corrected without changing labels; its failure evidence was retained. Source images and overlays are not published.', '',
          'Complementarity compares old A0 and new A0rgb_s0, each against its own modality labels. Saved RGB boxes use original RGB pixels despite the historical filename suffix. On the intersection of valid positive annotations, the highest-confidence saved box succeeds at IoU >= 0.5. Missing files and unequal lengths are errors. These conditional success rates are not official-style SA and do not directly quantify IR-coordinate improvements.', '',
          f"Both-positive frames: {p['frames']}; IR failures: {p['ir_failures']}. All-frame coverage: 61,999. Each bootstrap draw samples 67 whole sequences with replacement and recomputes pooled ratios; 10,000 draws, seed 0.", '',
          '| Statistic | Value |', '|---|---:|']
for key in ('ir_success', 'rgb_success', 'oracle_success', 'oracle_gain', 'rescue_share'):
    value = p[key]
    text = 'null (undefined)' if value is None else f'{value:.6f}'
    interval = cp.get(key + '_ci95')
    if interval:
        text += f' [{interval[0]:.6f}, {interval[1]:.6f}]'
    lines.append(f'| {key} | {text} |')
lines += ['', 'oracle_gain is RGB-only successes divided by all both-positive frames; rescue_share divides those successes by IR failures. The oracle selects between two fixed outputs. It is not a mathematical upper bound on fusion models, retrained detectors or native RGB resolution.', '',
          'Known release annotation issues remain unchanged and are listed in the JSON report. The primary analysis retains all videos; the sensitivity summary excludes flagged sequences:', '',
          '```json', json.dumps(cp['excluding_known_label_issues'], indent=2), '```', '',
          '## Next step', '', decision, '',
          'The 0.5% and 2% bands were frozen before observing these results and are task-specific exploratory heuristics, not accepted standards. Confidence intervals are unadjusted for multiple comparisons and quantify sequence sampling, not training-seed robustness. Blanking and shift tolerance do not prove RGB is ignored; optimization, EMA and normalization explanations remain unverified.', '',
          'Full numeric evidence: [ablation scores](reports/ablation_scores.json), [per-video SA](reports/ablation_per_sequence.csv), [complementarity](reports/complementarity_val.json), [dataset audit](reports/dataset_audit.json), and [frozen protocol](PROTOCOL.md). All three complete prediction sets and the portable RGB model are included.']
(A / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='ascii', newline='\n')
summary = {'snapshot': snapshot, 'ablation': scores, 'complementarity': cp, 'recommendation': decision}
(R / 'final_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='ascii', newline='\n')
(V / 'final_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='ascii')
readme = repo / 'README.md'
text = readme.read_text()
note = '\nThe [v6 fusion diagnostics](artifacts/v6/REPORT.md) are complete: two blank-input ablations, one 100-epoch scratch RGB baseline, full RGB validation inference and fixed-detector complementarity. Complete predictions and the portable RGB model are published. Test remains unevaluated.\n'
if '<!-- V6_STATUS -->' in text:
    text = re.sub(r'<!-- V6_STATUS -->.*?<!-- /V6_STATUS -->', '<!-- V6_STATUS -->\n' + note.strip() + '\n<!-- /V6_STATUS -->', text, flags=re.S)
else:
    text = text.replace('## Contents\n', note + '\n## Contents\n')
readme.write_text(text, encoding='ascii', newline='\n')
print('Validated completed coverage, training budget, model hashes and fixed-detector counts; final report written.')
