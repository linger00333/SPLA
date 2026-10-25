"""Strict JSON prediction reading, including published gzip artifacts."""
import gzip
import json
from pathlib import Path
import numpy as np

def read_json(path):
    path = Path(path)
    compressed = path.with_name(path.name + '.gz')
    found = [p for p in (path, compressed) if p.is_file()]
    if not found:
        raise FileNotFoundError(f'Missing prediction artifact: {path}')
    values = []
    for p in found:
        raw = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
        values.append(json.loads(raw))
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError(f'Conflicting plain and gzip artifacts: {path}')
    return values[0]

def validate_frames(frames, expected=None, detections=False):
    if not isinstance(frames, list) or not frames:
        raise ValueError('Predictions must contain a nonempty frame list')
    if expected is not None and len(frames) != expected:
        raise ValueError(f'Prediction length {len(frames)} differs from expected {expected}')
    for frame in frames:
        boxes = frame if detections else ([frame] if frame else [])
        if not isinstance(frame, list):
            raise ValueError('Each prediction frame must be a list')
        for box in boxes:
            if not isinstance(box, list) or len(box) != (5 if detections else 4):
                raise ValueError('Malformed prediction box')
            arr = np.asarray(box, dtype=float)
            if not np.isfinite(arr).all() or min(arr[2:4]) <= 0:
                raise ValueError('Prediction boxes must be finite with positive sizes')
            if detections and not 0 <= arr[4] <= 1:
                raise ValueError('Invalid confidence')
        if detections and any(frame[i][4] < frame[i+1][4] for i in range(len(frame)-1)):
            raise ValueError('Detection candidates must be sorted by descending confidence')
