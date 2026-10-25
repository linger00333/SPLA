"""Validate the completed export and produce a bounded English v7 report."""
from pathlib import Path
import csv
import gzip
import json
import math
import zipfile

V = Path(__file__).resolve().parent
REPO = V.parents[1] / 'github/SPLA'
if not REPO.exists(): REPO = V.parents[1]

def primary_gate(scores):
    point = scores['runs']['P']['SA'] - scores['runs']['A0']['SA']
    ci = scores['vs_baseline']['P-A0']['SA']['ci95']
    return {'point_delta': point, 'ci95': ci, 'passed': bool(point >= .003 and ci[0] > 0),
            'rule': 'P only: macro SA delta >= +0.003 and 95% lower bound > 0'}

def interval_text(point, interval):
    if point is None: return 'null (undefined)'
    if not interval: return f'{point:+.6f} [CI undefined]'
    return f'{point:+.6f} [{interval[0]:+.6f}, {interval[1]:+.6f}]'

def integrate():
    with zipfile.ZipFile(V / 'publication_export.zip') as z:
        assert z.testzip() is None
        for member in z.infolist():
            assert member.filename.startswith(('artifacts/v7/', 'artifacts/predictions/v7/'))
            dst = (REPO / member.filename).resolve()
            assert dst.is_relative_to(REPO.resolve())
            raw = z.read(member)
            (gzip.decompress(raw) if member.filename.endswith('.gz') else raw).decode('ascii')
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(raw)
            if member.filename.startswith('artifacts/v7/'):
                p = V / 'evidence' / Path(member.filename).relative_to('artifacts/v7')
                p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)

def finish():
    A = REPO / 'artifacts/v7'; R = A / 'reports'
    read = lambda p: json.loads(p.read_text())
    state = read(A / 'status_snapshot.json')
    assert state['phase'] == 'complete_report_ready' and all(x['state'] == 'complete' for x in state['jobs'].values())
    assert not state['training'] and not state['test'] and not state['GPU_used']
    manifest = read(A / 'records/frozen_manifest.json')
    assert manifest['n_sequences'] == 67 and manifest['frames'] == 61999
    assert read(A / 'records/v6_SA_bootstrap_reproduction.json')['exact_first_comparison_reproduction']
    scores = read(R / 'fusion_val.json'); assert scores['n_sequences'] == 67
    assert scores['bootstrap_mode'] == 'per_comparison' and scores['bootstrap_resamples'] == 10000
    assert abs(scores['runs']['A0']['SA'] - read(REPO / 'artifacts/v3/val_scores_all.json')['runs']['A0']['SA']) < 1e-12
    with (R / 'fusion_val_per_seq.csv').open() as f: csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 67 and len({r['sequence'] for r in csv_rows}) == 67
    coverage = {}; conditions = {}
    for name in ('P', 'S', 'R'):
        directory = REPO / 'artifacts/predictions/v7' / name
        cfg = json.loads(gzip.decompress((directory / 'run_config.json.gz').read_bytes()))
        assert cfg['split'] == 'val' and cfg['output_coordinates'] == 'IR_pixels'
        assert cfg['uses_evaluation_labels'] == (name == 'R')
        if name != 'R': assert cfg['seq_theta'] is None
        files = list(directory.glob('*_IR.txt.gz')); assert len(files) == 67
        frames = 0
        for p in files:
            pred = json.loads(gzip.decompress(p.read_bytes()))['res']
            detp = p.with_name(p.name.replace('_IR.txt.gz', '_dets.json.gz'))
            dets = json.loads(gzip.decompress(detp.read_bytes()))
            assert len(pred) == len(dets) > 0; frames += len(pred)
        assert frames == 61999
        coverage[name] = {'videos': 67, 'frames': frames, 'length_gaps': 0, 'coordinates': 'IR_pixels'}
        summ = json.loads(gzip.decompress((directory / 'fusion_summary.json.gz').read_bytes()))
        detail = read(R / ('breakdown_' + name + '.json'))
        difference = scores['runs'][name]['SA'] - scores['runs']['A0']['SA']
        reconstruction = math.fsum(r['sa_change_in_video'] for r in detail['per_video'].values()) / 67
        assert abs(reconstruction-difference) < 1e-12
        bound = math.fsum(r['actions'].get('replace', 0)/r['frames'] for r in summ['per_video'].values()) / 67
        assert abs(difference) <= bound+1e-12
        conditions[name] = {'summary': summ, 'breakdown': detail, 'macro_SA_delta': difference,
                            'reconstructed_macro_delta': reconstruction, 'absolute_macro_SA_bound': bound}
    gate = primary_gate(scores)
    sync = {}
    for source in ('predictions', 'labels'):
        for mode in ('whole', 'windows200', 'common'):
            d = read(R / f'sync_{source}_{mode}.json')['sources'][source]
            sync[source + '_' + mode] = {'candidates': len(d['flagged']), 'total_videos': d['videos'],
                'estimable_videos': sum(any(w['best_lag'] is not None for w in wins) for wins in d['per_video'].values())}
    heldout = read(R / 'sync_heldout.json')
    for source, d in heldout['sources'].items():
        sync[source + '_heldout'] = {'candidates': len(d['candidates']), 'total_videos': len(d['per_video']),
                                    'estimable_videos': sum(v['status'] == 'estimated' for v in d['per_video'].values())}
    geometry = read(R / 'geometry_review.json')
    lines = ['# V7 decision-level fusion and data audit', '',
             'All registered CPU-only validation jobs are complete. No training, GPU inference or test evaluation was run. P, S and R each cover all 67 videos and 61,999 frames. Original data, labels, old experiments and weights were not changed.', '',
             '## Main result', '',
             'The primary gate is ' + ('met' if gate['passed'] else 'not met') + '. ' + gate['rule'] + '. No follow-up training is launched. S is a registered exploratory secondary and R is a label-assisted reference; neither can independently open the primary gate.', '',
             '| Condition | SA | Penalized SA | AP50 |', '|---|---:|---:|---:|']
    for name in ('A0', 'P', 'S', 'R'):
        r = scores['runs'][name]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{r[k]:.6f}' if r[k] is not None else 'null' for k in ('SA', 'SA_pen', 'AP50')) + ' |')
    lines += ['', 'Point differences from A0 and 95% paired whole-sequence intervals, 10,000 draws. Seed 0 restarts independently for each comparison. Only P/SA is primary; other intervals are unadjusted descriptive diagnostics.', '',
              '| Condition minus A0 | SA delta [CI] | Penalized SA delta [CI] | AP50 delta [CI] |', '|---|---|---|---|']
    for name in ('P', 'S', 'R'):
        cells = []
        for k in ('SA', 'SA_pen', 'AP50'):
            a, b = scores['runs'][name][k], scores['runs']['A0'][k]
            cells.append(interval_text(a-b if a is not None and b is not None else None, scores['vs_baseline'][name+'-A0'].get(k, {}).get('ci95')))
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'The penalized metric and SA can move in opposite directions: the missing-output penalty rewards filling a positive frame even when localization remains poor. AP also has an added-candidate and uncalibrated cross-detector confidence limitation. These secondary changes do not override the declared SA gate.', '',
              '## Replacement costs and mapping', '',
              '| Condition | Replacements | IR-positive replacements | IR-negative replacements | Actual negative-frame score change | Absolute macro-SA bound |', '|---|---:|---:|---:|---:|---:|']
    for name, d in conditions.items():
        rows = d['breakdown']['replaced_frames']; p = rows['ir_positive']; n = rows['ir_negative']
        lines.append(f"| {name} | {d['summary']['pooled']['actions'].get('replace',0)} | {int(p.get('frames',0))} | {int(n.get('frames',0))} | {n.get('sum_delta',0):.0f} | {d['absolute_macro_SA_bound']:.6f} |")
    lines += ['', 'A negative-label replacement costs -1 only if the original output was empty. If it already contained a false positive, the added SA cost is 0. The full per-video score change is reconstructed exactly from replacement contributions for all three conditions. Bounds are macro-video bounds for this fixed rule, not the pooled RGB-coordinate oracle from v6.', '',
              'The requested 20190925_213001_1_1 precheck inspected nine selected pairs. Early IR samples are visibly blurred, while RGB shows target lights. Some modalities have different official presence labels. This is a visibility/annotation concern, not permission to repair IR labels using RGB. Images and frame-level label states remain private.', '',
              '| Condition | Replacements in requested video | Its SA change |', '|---|---:|---:|']
    for name, d in conditions.items():
        r = d['breakdown']['per_video'].get('20190925_213001_1_1', {})
        lines.append(f"| {name} | {r.get('replaced',0)} | {r.get('sa_change_in_video',0):+.6f} |")
    lines += ['', 'Prediction-only proposal agreement with a confident IR box was replayed within 1e-12 of the supplied material. It is not annotation IoU. S agreement is measured in a regime with confident paired history, whereas many actual replacement frames fall back to the train mean.', '',
              '| Condition | Confident-IR comparison frames | Proposal agreement IoU >= 0.5 |', '|---|---:|---:|']
    for name, d in conditions.items():
        f = d['summary']['pooled']['transfer_fidelity']
        lines.append(f"| {name} | {f['frames']} | {f['share_iou_ge_0.5']:.4%} |")
    online = conditions['S']['summary']['pooled']
    lines += ['', f"S uses online mapping for {online['replaced_map_sources']['online']} actual replacements and train-mean fallback for {online['replaced_map_sources']['online_fallback']}; recent IR size is available for {online['replaced_size_sources']['ir_recent']} and mapped-size fallback is used for {online['replaced_size_sources']['mapped_fallback']}. Conditional agreement on confident IR frames therefore overstates transfer quality at the replacement operating point.", '',
              '## Predeclared sensitivities', '', '| Exclusion | Videos retained | P SA delta [CI] | S SA delta [CI] | R SA delta [CI] |', '|---|---:|---|---|---|']
    for label in ('without_213001', 'without_known_issues', 'without_prediction_sync_candidates'):
        rs = [read(R / (name+'_'+label+'.json')) for name in ('P','S','R')]
        assert len({r['n_seq'] for r in rs}) == 1
        lines.append('| '+label+' | '+str(rs[0]['n_seq'])+' | '+' | '.join(interval_text(r['point'], r['ci95']) for r in rs)+' |')
    lines += ['', 'Exclusions and their lists were fixed before scoring; these subset intervals cannot replace the primary all-video comparison. The synchronization subset uses the original 34 prediction-screen candidates, not a later favorable label-defined subset.', '',
              '## Temporal audit: candidates, not confirmed acquisition defects', '', '| Trajectory source and support | Candidates | Estimable / all videos |', '|---|---:|---:|']
    for name, row in sync.items(): lines.append(f"| {name} | {row['candidates']} | {row['estimable_videos']} / {row['total_videos']} |")
    lines += ['', 'The supplied prediction screen (34 candidates) is reproduced. Label confirmation, common-support comparison and held-out evaluation do not support simply reporting half of the dataset as confirmed desynchronized. Held-out selection uses the first two thirds, a 15-frame boundary gap, and evaluates fixed transforms/lags on the final third.', '',
              'For 20190926_183941_1_5, the whole-label fit selects +4 frames; for 20190925_131530_1_3 it selects +7. The detailed JSON files report zero-lag and selected-lag residuals and window estimates. Constant-velocity motion can be absorbed by translation; low excitation, annotation delay, detector errors and spatial-model mismatch remain rival explanations.', '',
              'Encoded timing was checked separately in the two examples and 20190926_141816_1_6. The media audit reports presentation timestamps, not synchronized sensor capture clocks. Regular encoded PTS cannot establish or exclude source-level frame loss. No temporal shifts were fed into the fusion comparison, and the training split was not audited here.', '',
              '## Geometry and A1 interpretation', '']
    g = geometry['identity_center_error_IR_px']; shape = geometry['label_derived_sequence_mapping_reference']
    lines += [f"The additional post-registration explanatory label review covers {geometry['paired_positive_frames']} both-positive frames. Identity center mismatch has pooled median {g['median']:.3f} IR px, IQR [{g['q25']:.3f}, {g['q75']:.3f}]. A1 receives resized, unregistered inputs; it is inaccurate to describe its blanking or shift tests as a direct measurement of perfectly aligned RGB utility.", '',
              f"Even a label-derived per-video mapping of RGB GT boxes gives median IR GT IoU {shape['median_GT_IoU']:.6f}, with {shape['share_GT_IoU_ge_0_5']:.4%} at IoU >= 0.5. Median IR height divided by mapped RGB height is {shape['median_IR_height_over_mapped_RGB_height']:.4f}, while the width ratio is {shape['median_IR_width_over_mapped_RGB_width']:.4f}. These are label-geometry diagnostics with additional supervision, not deployable fusion performance.", '',
              '## Repository fixes and evidence limits', '',
              'The scorer now refuses missing/truncated/invalid predictions and RGB-coordinate inputs, reads gzip artifacts, and supports order-invariant bootstrap draws. Legacy shared-stream draws remain available. Subset CSV omissions/duplicates and malformed fusion lengths are errors. The previously published v6 first SA interval was reproduced exactly. Old reports are preserved; their separate full-coverage audits protected them from the newly identified silent-input failure mode.', '',
              'The supplied author is preserved. Only the private external session URL was removed from the applied commit message; original and application hashes are recorded. Full source/input hashes and commands, 32 actual-runtime tests and 12 supplemental local tests (three held-out, four I/O, five reporting-gate checks) are retained. Test logs include an expected invalid-video fixture error; scoring warnings refer to undefined mechanism diagnostics, not missing SA or AP results. No raw labels, images or videos are published.', '',
              '## Concrete next step (proposed, not executed)', '',
              'First verify timing/visibility and spatial identifiability on a small fixed set: the +4-frame example, the +7-frame example, the window-drift example and the requested blurred-IR video. Retain original labels, distinguish acquisition timing from annotation or motion effects, and use a calibration/held-out split for any corrected mapping. Only after that audit should a separately versioned train-derived calibration or confidence/box-transfer experiment be specified. This run does not justify automatically launching modality dropout, attention, cross-scale fusion or the old failed-pilot queue.', '',
              'A failure of this fixed confidence-gated rule does not establish that RGB is uninformative or that all fusion methods are redundant. Single detector seeds, selected validation checkpoints, conditional proposal diagnostics, possible timing mismatch and label visibility differences limit generalization.', '',
              'Evidence: [registered protocol](PROTOCOL.md), [full scores](reports/fusion_val.json), [per-video scores](reports/fusion_val_per_seq.csv), [replacement breakdowns](reports/breakdown_P.json), [held-out audit](reports/sync_heldout.json), [geometry](reports/geometry_review.json), [media timing](reports/media_timing_audit.json), [frozen sources and inputs](records/frozen_manifest.json).']
    (A / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='ascii', newline='\n')
    summary = {'primary_gate': gate, 'coverage': coverage, 'scores': scores, 'sync_summary': sync,
               'replacement_summary': {k: {'macro_SA_delta': v['macro_SA_delta'], 'bound': v['absolute_macro_SA_bound'],
                  'counts': v['summary']['pooled']['actions'], 'replaced_frame_breakdown': v['breakdown']['replaced_frames']} for k, v in conditions.items()},
               'training': False, 'test': False, 'GPU_used': False}
    (R / 'final_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='ascii', newline='\n')
    (V / 'final_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='ascii')
    print(json.dumps({'primary_gate': gate, 'sync_summary': sync}, indent=2))

if __name__ == '__main__':
    integrate(); finish()
