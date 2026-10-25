"""Inspect encoded presentation timestamps in three preselected lag examples."""
import json
import shutil
import subprocess
import sys
import numpy as np
from runtime import V, R
sys.path.insert(0, str(V / 'kit'))
from spla.antiuav_io import find_files
exe = shutil.which('ffprobe')
report = {'scope': 'Post-registration encoded media timing diagnostic; not used to change any lag or fusion input. Encoded timestamps are not synchronized acquisition-clock ground truth.', 'available': exe is not None, 'videos': {}}
if exe:
    for seq in ('20190926_183941_1_5', '20190925_131530_1_3', '20190926_141816_1_6'):
        fs = find_files(R / 'val' / seq); per = {}
        for role in ('ir_video', 'rgb_video'):
            p = fs[role]
            meta = json.loads(subprocess.check_output([exe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=avg_frame_rate,r_frame_rate,nb_frames,duration,start_time,time_base', '-of', 'json', str(p)], text=True))
            packet = json.loads(subprocess.check_output([exe, '-v', 'error', '-select_streams', 'v:0', '-show_packets',
                '-show_entries', 'packet=pts_time', '-of', 'json', str(p)], text=True))
            pts = np.sort([float(x['pts_time']) for x in packet['packets'] if 'pts_time' in x])
            delta = np.diff(pts); positive = delta[delta > 1e-9]
            nominal = float(np.median(positive)) if len(positive) else None
            per[role] = {'stream_metadata': meta['streams'], 'packets_with_PTS': len(pts),
                         'duplicate_PTS': int(np.sum(delta <= 1e-9)), 'median_PTS_step_seconds': nominal,
                         'max_PTS_step_seconds': float(delta.max()) if len(delta) else None,
                         'steps_above_1_5x_median': int(np.sum(delta > nominal*1.5)) if nominal else None}
        report['videos'][seq] = per
(V / 'reports/media_timing_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
