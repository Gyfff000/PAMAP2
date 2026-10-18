from pathlib import Path
import csv
import hashlib
import json
import os
import random
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
LABEL_NAMES = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling',
               'Nordic walking', 'ascending stairs', 'descending stairs',
               'vacuum cleaning', 'ironing', 'rope jumping']
LABEL_ZH = ['躺卧', '坐', '站立', '步行', '跑步', '骑车', '北欧式行走',
            '上楼', '下楼', '吸尘', '熨烫', '跳绳']
SUBJECTS = list(range(101, 109))
# Zero-based columns: +/-16g acceleration followed by angular velocity.
CHANNELS = [4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29,
            38, 39, 40, 44, 45, 46]
SENSOR_NAMES = ['wrist', 'chest', 'ankle']

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'Cannot serialize {type(value)}')
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=encode)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)

def load_data():
    p = ROOT / 'data/processed'
    return (np.load(p / 'windows.npy', mmap_mode='r'),
            np.load(p / 'labels.npy'), np.load(p / 'subjects.npy'),
            np.load(p / 'features.npy'), np.load(p / 'window_metadata.npy'))

def ensure_predictions():
    """Restore the public outer-fold predictions supplied with a clone."""
    import zipfile
    archive=ROOT/'artifacts/outer_fold_predictions.zip'
    if not archive.exists():
        return
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            relative=Path(info.filename)
            if relative.is_absolute() or '..' in relative.parts or not info.filename.startswith('outputs/folds/') or relative.name!='predictions.npz':
                raise ValueError('Unexpected prediction archive entry')
            target=ROOT/relative
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(info))
