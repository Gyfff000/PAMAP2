"""Offline, pure-activity windows; never bridge subjects, transitions or long gaps."""
import numpy as np
from scipy.signal import resample_poly
from .common import ROOT, CHANNELS, LABEL_IDS, LABEL_NAMES, SUBJECTS, save_json, sha256, write_csv

WINDOW = 256
STEP = 128

def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2], edges[1::2]))

def long_missing_rows(x, max_gap=50):
    bad = np.zeros(len(x), dtype=bool)
    for c in range(x.shape[1]):
        for start, end in runs(~np.isfinite(x[:, c])):
            if end - start > max_gap:
                bad[start:end] = True
    return bad

def interpolate(x):
    result = x.copy()
    for c in range(x.shape[1]):
        valid = np.flatnonzero(np.isfinite(x[:, c]))
        if not len(valid):
            raise ValueError('All-missing channel must not reach interpolation')
        # Offline interpolation within one uninterrupted activity segment only.
        result[:, c] = np.interp(np.arange(len(x)), valid, x[valid, c])
    return result

def features(windows):
    """Seven channel statistics + four acceleration/gyro magnitude statistics per sensor."""
    x = np.asarray(windows, dtype=np.float64)
    centered = x - x.mean(axis=2, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=2)) ** 2
    spectrum[:, :, 0] = 0
    p = spectrum / np.maximum(spectrum.sum(axis=2, keepdims=True), 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=2)
    freq = np.fft.rfftfreq(x.shape[2], 1 / 50.)[spectrum.argmax(axis=2)]
    stats = [x.mean(2), x.std(2), x.min(2), x.max(2),
             np.sqrt(np.mean(x*x, axis=2)), entropy, freq]
    # Keep each sensor's features contiguous for sensor removal comparisons.
    result = []
    for s in range(3):
        blocks = [a[:, s*6:(s+1)*6] for a in stats]
        for offset in (s*6, s*6+3):
            mag = np.linalg.norm(x[:, offset:offset+3], axis=1)
            blocks.extend([mag.mean(1, keepdims=True), mag.std(1, keepdims=True),
                           mag.min(1, keepdims=True), mag.max(1, keepdims=True)])
        result.append(np.concatenate(blocks, axis=1))
    return np.concatenate(result, axis=1).astype(np.float32)

def main():
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    windows, labels, subjects, metadata = [], [], [], []
    audits = []
    class_counts = []
    segment_number = 0
    for path in sorted((ROOT / 'data/raw/Protocol').glob('subject*.dat')):
        subject = int(path.stem.replace('subject', ''))
        print(f'Reading {path.name}', flush=True)
        raw = np.loadtxt(path, dtype=np.float64)
        times, activity, signals = raw[:, 0], raw[:, 1].astype(int), raw[:, CHANNELS]
        audit = {'subject': subject, 'raw_rows': len(raw),
                 'activity_rows': int(np.isin(activity, LABEL_IDS).sum()),
                 'missing_sensor_values': int((~np.isfinite(signals)).sum()),
                 'missing_sensor_fraction': float((~np.isfinite(signals)).mean()),
                 'raw_activity_ids': sorted(int(v) for v in np.unique(activity)),
                 'primary_included': subject in SUBJECTS, 'windows': 0,
                 'short_segment_rows': 0, 'long_gap_rows': 0,
                 'interpolated_sensor_values': 0, 'sha256': sha256(path)}
        if subject not in SUBJECTS:
            audits.append(audit)
            continue
        # Split on every true activity change and timestamp discontinuity first.
        breaks = np.r_[0, np.flatnonzero((np.diff(activity) != 0) |
                         (np.diff(times) > 0.03) | (np.diff(times) <= 0)) + 1, len(raw)]
        for a, b in zip(breaks[:-1], breaks[1:]):
            if activity[a] not in LABEL_IDS:
                continue
            sig = signals[a:b]
            invalid = long_missing_rows(sig)
            audit['long_gap_rows'] += int(invalid.sum())
            for c, d in runs(~invalid):
                if d-c < 2*WINDOW:
                    audit['short_segment_rows'] += d-c
                    continue
                part = sig[c:d]
                audit['interpolated_sensor_values'] += int((~np.isfinite(part)).sum())
                sampled = resample_poly(interpolate(part), 1, 2, axis=0).astype(np.float32)
                for start in range(0, len(sampled)-WINDOW+1, STEP):
                    window = sampled[start:start+WINDOW].T.copy()
                    windows.append(window)
                    labels.append(LABEL_IDS.index(int(activity[a])))
                    subjects.append(subject)
                    metadata.append([segment_number, times[a+c+2*start],
                                     times[a+c+min(2*(start+WINDOW)-1, d-c-1)]])
                    audit['windows'] += 1
                segment_number += 1
        audits.append(audit)
        print(f'Subject {subject}: {audit["windows"]} windows', flush=True)
    if not windows:
        raise RuntimeError('No windows. Run 00_download_data.py first.')
    x = np.stack(windows)
    y = np.array(labels, dtype=np.int64)
    g = np.array(subjects, dtype=np.int64)
    assert set(np.unique(g)) == set(SUBJECTS)
    assert x.shape[1:] == (18, WINDOW) and np.isfinite(x).all()
    np.save(out / 'windows.npy', x)
    np.save(out / 'labels.npy', y)
    np.save(out / 'subjects.npy', g)
    np.save(out / 'window_metadata.npy', np.array(metadata, dtype=np.float64))
    print('Extracting deterministic statistical features', flush=True)
    f = np.concatenate([features(x[i:i+512]) for i in range(0, len(x), 512)])
    assert f.shape[1] == 150 and np.isfinite(f).all()
    np.save(out / 'features.npy', f)
    for s in SUBJECTS:
        for k, name in enumerate(LABEL_NAMES):
            class_counts.append({'subject': s, 'activity_id': LABEL_IDS[k],
                                 'activity': name, 'windows': int(((g == s) & (y == k)).sum())})
    write_csv(ROOT / 'outputs/class_subject_counts.csv', class_counts)
    save_json(ROOT / 'outputs/data_audit.json', {'subjects': audits,
        'window_shape': list(x.shape), 'feature_shape': list(f.shape),
        'label_ids': LABEL_IDS, 'label_names': LABEL_NAMES,
        'sampling_hz_original': 100, 'sampling_hz_model': 50,
        'window_seconds': 5.12, 'stride_seconds': 2.56,
        'channels_zero_based': CHANNELS,
        'long_missing_threshold_original_samples': 50,
        'subject109_exclusion': 'Partial Protocol recording; not a complete 12-activity participant.',
        'evaluation_scope': 'Offline classification of windows entirely within annotated activities; transitions excluded.',
        'array_hashes': {p.name: sha256(p) for p in out.glob('*.npy')}})
    print(f'Prepared {len(x)} windows, {len(SUBJECTS)} subjects, 12 classes.', flush=True)

if __name__ == '__main__':
    main()
