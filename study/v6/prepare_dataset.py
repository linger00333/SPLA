"""Build and audit isolated RGB labels, including the actual framework loader."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from runtime import V, P, R, PY, ENV

os.environ.update(ENV)
sys.path[:0] = [str(V / 'kit'), str(V / 'framework')]
import cv2
import numpy as np
from spla.antiuav_io import find_files, read_labels
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = P / 'antiuav_yolo'
data = V / 'data'
assert not (V / 'reports/dataset_audit.json').exists()
original = {}
for split in ('train', 'val'):
    for p in (source / 'labels/visible' / split).glob('*.txt'):
        original[p] = sha(p)
    for p in (source / 'images/visible' / split).glob('*.jpg'):
        with p.open('rb') as f:
            f.seek(-2, 2)
            assert f.read() == b'\xff\xd9', 'Refuse loader JPEG repair through a hard link'
cmd = [PY, str(V / 'kit/tools/make_rgb_only.py'), '--root', str(R), '--source', str(source),
       '--out', str(data), '--splits', 'train', 'val']
if '--audit-only' not in sys.argv:
    subprocess.run(cmd, env=ENV, check=True)
else:
    assert (data / 'rgbonly/summary.json').exists(), 'Builder must have completed before audit-only mode'
cfg = get_cfg(overrides={'imgsz': 640, 'batch': 16, 'workers': 4, 'task': 'detect',
                         'use_simotm': 'RGB', 'channels': 3})
ds_cfg = check_det_dataset(str(data / 'antiuav_rgbonly.yaml'), autodownload=False)
assert ds_cfg.get('test') is None
report = {'builder_command': cmd, 'splits': {}, 'test_processed': False,
          'sampling': 'Exactly the old IR-existence-based sampled train/val image lists',
          'shared_source_labels_unchanged': False, 'private_visualizations': []}
for split in ('train', 'val'):
    meta = json.loads((source / f'spla_meta_{split}.json').read_text())
    ds = build_yolo_dataset(cfg, ds_cfg[split], 16, ds_cfg, mode='val', rect=False, use_simotm='RGB')
    assert len(ds) == len(meta) and {Path(p).stem for p in ds.im_files} == set(meta)
    expected_root = (data / 'rgbonly/labels' / split).resolve()
    cache = {}; npos = 0; samples = []; manifest = []
    for imfile, labfile, loaded in zip(ds.im_files, ds.label_files, ds.labels):
        im, lab = Path(imfile), Path(labfile)
        assert lab.parent.resolve() == expected_root and not im.is_symlink()
        m = meta[im.stem]; seq = m['seq']; t = m['frame']
        srcim = source / 'images/visible' / split / im.name
        assert os.path.samefile(im, srcim), 'Expected hard-linked images on this filesystem'
        if seq not in cache:
            fs = find_files(R / split / seq)
            e, b = read_labels(fs['rgb_label'])
            cap = cv2.VideoCapture(str(fs['rgb_video']))
            wh = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)])
            cap.release(); assert min(wh) > 0
            cache[seq] = e, b, wh
        e, b, wh = cache[seq]
        expected = np.empty((0, 4))
        if e[t]:
            box = b[t]
            xy = np.clip((box[:2] + box[2:] / 2) / wh, 0, 1)
            # Match decimal serialization; np.round can differ at exact half-way values.
            expected = np.array([float(f'{v:.6f}') for v in np.r_[xy, np.minimum(box[2:] / wh, 1)]])[None]
            npos += 1
            if split == 'val' and (not samples or seq not in [p[1]['seq'] for p in samples]) and len(samples) < 4:
                samples.append((im, m, expected[0]))
        np.testing.assert_allclose(loaded['bboxes'], expected, rtol=0, atol=6e-8)
        assert len(loaded['cls']) == len(expected)
        manifest.append({'image': im.name, 'frame': t, 'sequence': seq, 'label_sha256': sha(lab),
                         'image_sha256': sha(im)})
    # Exercise decoding and tensor formatting through the actual dataset, not only its path mapper.
    sample = ds[0]
    assert sample['img'].shape[0] == 3
    report['splits'][split] = {'images': len(ds), 'positive_rgb_labels': npos, 'sequences': len(cache),
                              'all_actual_loader_label_paths_correct': True,
                              'all_loader_boxes_match_original_rgb_normalization': True,
                              'all_images_hardlinked': True, 'sample_tensor_shape': list(sample['img'].shape),
                              'metadata_sha256': sha(source / f'spla_meta_{split}.json')}
    (V / 'records' / f'rgb_dataset_manifest_{split}.json').write_text(json.dumps(manifest, indent=1))
    for im, m, box in samples:
        image = cv2.imread(str(im)); h, w = image.shape[:2]
        cx, cy, bw, bh = box * [w, h, w, h]
        x1, y1, x2, y2 = map(int, [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        marked = image.copy(); cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked, f"{m['seq']} frame {m['frame']} RGB label", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        radius = int(max(bw, bh, 30) * 2)
        crop = marked[max(0, int(cy)-radius):min(h, int(cy)+radius), max(0, int(cx)-radius):min(w, int(cx)+radius)]
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
        path = V / 'private_visuals' / (im.stem + '_rgb_boxes.jpg')
        cv2.imwrite(str(path), np.hstack([marked, crop]))
        report['private_visualizations'].append(path.name)
assert all(sha(p) == value for p, value in original.items())
report['shared_source_labels_unchanged'] = True
report['hardlinked_images_policy'] = 'Images are read-only inputs; JPEG EOF precheck prevents the loader repair path'
report['dataset_yaml_sha256'] = sha(data / 'antiuav_rgbonly.yaml')
(V / 'reports/dataset_audit.json').write_text(json.dumps(report, indent=2))
print('Dataset audit passed; inspect private_visuals before enabling the training gate.')
