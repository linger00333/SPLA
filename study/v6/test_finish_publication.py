"""Exercise the final integration gate and report on a complete synthetic export."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import runpy
import zipfile

V = Path(__file__).resolve().parent

class FinalReport(unittest.TestCase):
    def test_watcher_stops_at_failed_remote_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = root / 'experiment/v6'; study.mkdir(parents=True)
            (root / 'handoff.md').write_text('# Handoff\n')
            script = study / 'watch_once.py'
            shutil.copy2(V / 'watch_once.py', script)
            snapshot = {'step1': {'phase': 'failed'}, 'step2': {'phase': 'waiting_for_step1'}}
            observed = []
            def fake_run(cmd, **kwargs):
                observed.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, json.dumps(snapshot), '')
            with patch.object(sys, 'argv', [str(script), '--remote-study', '/private/experiments/v6_fusion_diagnostics_20261024']), patch('subprocess.run', fake_run):
                with self.assertRaises(AssertionError):
                    runpy.run_path(str(script), run_name='__main__')
            state = json.loads((study / 'watch_once_status.json').read_text())
            self.assertEqual(state['phase'], 'failed_needs_inspection')
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][:2], ['ssh', 'A100'])
            self.assertIn('snapshot.py', observed[0][2])

    def test_complete_export_is_validated_and_null_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = root / 'experiment/v6'; study.mkdir(parents=True)
            repo = root / 'github/SPLA'; repo.mkdir(parents=True)
            (repo / 'README.md').write_text('# Study\n\n<!-- V6_STATUS -->\nOld progress snapshot\n<!-- /V6_STATUS -->\n\n## Contents\n')
            shutil.copy2(V / 'finish_publication.py', study / 'finish_publication.py')
            files = {}
            def js(rel, obj):
                files['artifacts/v6/' + rel] = json.dumps(obj).encode()
            js('export_snapshot.json', {'step1_phase': 'complete', 'step2_phase': 'complete_report_ready',
                                       'training_completed': True, 'test_evaluated': False})
            for name in ('step1', 'step2'):
                js(name + '_status_snapshot.json', {'jobs': {'job': {'state': 'complete'}}})
            for name in ('coverage_ablation', 'coverage_rgb'):
                js('reports/' + name + '.json', {'run': {'n_sequences': 67, 'frames': 61999, 'length_mismatches': []}})
            row = {'SA': .8, 'SA_pen': .7, 'AP50': .99}
            comparison = {k: {'ci95': [-.01, .01]} for k in row}
            js('reports/ablation_scores.json', {'n_sequences': 67, 'runs': {'A1': row, 'A1_blankrgb': row, 'A1_blankir': dict(row, AP50=None)},
                                               'vs_baseline': {'A1_blankrgb-A1': comparison,
                                                               'A1_blankir-A1': {k: v for k, v in comparison.items() if k != 'AP50'}}})
            per = {str(i): {'n': 0, 'both': 0, 'ir_only': 0, 'rgb_only': 0, 'neither': 0} for i in range(67)}
            pooled = {'frames': 0, 'ir_failures': 0, **{k: None for k in ('ir_success', 'rgb_success', 'oracle_success', 'oracle_gain', 'rescue_share')}}
            js('reports/complementarity_val.json', {'videos': 67, 'coverage': {'total_frames': 61999},
                  'per_video': per, 'pooled': pooled, 'oracle_gain_ci95': None, 'excluding_known_label_issues': pooled})
            js('records/training_recipe_audit.json', {'all_selected_recipe_fields_match_old_A0': True})
            files['artifacts/v6/training/A0rgb_s0/results.csv'] = ('epoch,time\n' + '\n'.join(f'{i},1' for i in range(1, 101)) + '\n').encode()
            tensor, config = b'synthetic export placeholder', b'{"ch":3}'
            files['models/v6/A0rgb_s0/model.safetensors'] = tensor
            files['models/v6/A0rgb_s0/config.json'] = config
            js('records/rgb_model_export.json', {'tensor_sha256': hashlib.sha256(tensor).hexdigest(),
                                                'config_sha256': hashlib.sha256(config).hexdigest()})
            with zipfile.ZipFile(study / 'publication_export.zip', 'w') as z:
                for name, raw in files.items():
                    z.writestr(name, raw)
            result = subprocess.run([sys.executable, str(study / 'finish_publication.py')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = (repo / 'artifacts/v6/REPORT.md').read_text()
            self.assertIn('null (undefined)', report)
            self.assertIn('inconclusive', report)
            self.assertNotIn('NaN', (study / 'final_summary.json').read_text())
            self.assertNotIn('Old progress snapshot', (repo / 'README.md').read_text())

if __name__ == '__main__':
    unittest.main()
