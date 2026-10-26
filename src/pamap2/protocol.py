"""Frozen 70/30 recording-segment split and three training-only group CV folds."""
import csv
import json
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.model_selection import PredefinedSplit
from .common import ROOT, LABEL_NAMES, save_json, write_csv, sha256

OUT = ROOT / 'outputs/holdout'


def solve_assignment(y, segments, seed=42):
    """Stratify complete segments, using counts/labels only, never model outcomes.

    Part 0 is test; parts 1..3 are the training validation folds. The committed
    assignment is authoritative: a time-limited solver need not regenerate the
    identical feasible solution on another machine.
    """
    groups, inverse, counts = np.unique(segments, return_inverse=True, return_counts=True)
    labels = np.array([y[np.flatnonzero(inverse == i)[0]] for i in range(len(groups))])
    for i, label in enumerate(labels):
        if not np.all(y[inverse == i] == label):
            raise ValueError('A recording segment contains multiple activities.')
    if len(y) != 7410:
        raise ValueError('This frozen course protocol expects the audited 7,410 windows.')
    n, parts = len(groups), 4
    size = parts*n + parts*12
    rows, lower, upper = [], [], []
    def add(row, lo, hi):
        rows.append(row); lower.append(lo); upper.append(hi)
    for i in range(n):
        row = np.zeros(size); row[np.arange(parts)*n+i] = 1
        add(row, 1, 1)
    totals = np.bincount(y, minlength=12)
    for p, target in enumerate([2223, 1729, 1729, 1729]):
        row = np.zeros(size); row[p*n:(p+1)*n] = counts
        add(row, target, target)
        for k in range(12):
            ids = np.flatnonzero(labels == k)
            row = np.zeros(size); row[p*n+ids] = 1
            add(row, 1, np.inf)
            for sign in [1, -1]:
                row = np.zeros(size); row[p*n+ids] = sign*counts[ids]
                row[parts*n+p*12+k] = -1
                add(row, -np.inf, sign*target/len(y)*totals[k])
    objective = np.r_[np.random.default_rng(seed).random(parts*n)*1e-7,
                      np.tile(1/totals, parts)]
    result = milp(objective, integrality=np.r_[np.ones(parts*n), np.zeros(parts*12)],
                  bounds=Bounds(np.zeros(size), np.r_[np.ones(parts*n), np.full(parts*12, np.inf)]),
                  constraints=LinearConstraint(np.array(rows), lower, upper),
                  options={'time_limit':30, 'mip_rel_gap':.05})
    if result.x is None:
        raise RuntimeError('No feasible grouped 70/30 split; do not fall back to random windows.')
    assignment = result.x[:parts*n].reshape(parts,n).argmax(0)[inverse]
    return assignment, {'method':'Count-based mixed-integer stratified group allocation',
        'solver_message':result.message, 'objective':float(result.fun),
        'mip_gap':float(result.mip_gap), 'optimality_claimed':bool(result.success),
        'note':'Frozen feasible partition; exact sizes and isolation are independently asserted. No prediction or signal values used.'}


def validate_split(y, segments, assignment):
    if assignment.shape != y.shape or set(np.unique(assignment)) != {0,1,2,3}:
        raise AssertionError('Invalid split labels.')
    sizes = np.bincount(assignment, minlength=4)
    np.testing.assert_array_equal(sizes, [2223,1729,1729,1729])
    for group in np.unique(segments):
        if len(np.unique(assignment[segments == group])) != 1:
            raise AssertionError('Overlapping windows from one segment cross partitions.')
    for part in range(4):
        if len(np.unique(y[assignment == part])) != 12:
            raise AssertionError('Each test/validation part must cover all 12 classes.')
    train, test = np.flatnonzero(assignment != 0), np.flatnonzero(assignment == 0)
    assert 3*len(train) == 7*len(test)
    folds = [(train[fit],train[val]) for fit,val in
             PredefinedSplit(assignment[train]-1).split()]
    np.testing.assert_array_equal(np.sort(np.concatenate([v for _,v in folds])), train)
    for fit, val in folds:
        assert not set(segments[fit]) & set(segments[val])
        assert not set(segments[fit]) & set(segments[test])
    return train, test, folds


def load_split(y, subjects, metadata):
    path = OUT/'split_assignments.csv'
    segments = metadata[:,0].astype(np.int64)
    if not path.exists():
        assignment, solver = solve_assignment(y, segments)
        validate_split(y, segments, assignment)
        write_csv(path, [{'window_index':i, 'subject':int(subjects[i]),
            'activity':int(y[i]), 'segment':int(segments[i]), 'partition':int(assignment[i])}
            for i in range(len(y))])
        save_json(OUT/'split_manifest.json', {'protocol_version':2, 'seed':42,
            'train_windows':5187, 'test_windows':2223, 'train_fraction':.7,
            'test_fraction':.3, 'cv_folds':3, 'cv_validation_windows':[1729]*3,
            'group_unit':'Whole uninterrupted annotated activity segment after long-gap removal',
            'subject_disjoint':False, 'split_selection_uses_predictions':False,
            'historical_data_use':'These data were previously assessed under LOSO; this is not a new external dataset.',
            'assignment_sha256':sha256(path), 'solver':solver,
            'processed_hashes':{n:sha256(ROOT/'data/processed'/n) for n in
                ['labels.npy','subjects.npy','window_metadata.npy']}})
    manifest = json.loads((OUT/'split_manifest.json').read_text(encoding='utf-8'))
    if manifest['assignment_sha256'] != sha256(path):
        raise RuntimeError('Frozen split was modified.')
    for name, digest in manifest['processed_hashes'].items():
        if sha256(ROOT/'data/processed'/name) != digest:
            raise RuntimeError(f'Processed dataset changed: {name}')
    with path.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    np.testing.assert_array_equal([int(r['window_index']) for r in rows], np.arange(len(y)))
    np.testing.assert_array_equal([int(r['activity']) for r in rows], y)
    np.testing.assert_array_equal([int(r['subject']) for r in rows], subjects)
    np.testing.assert_array_equal([int(r['segment']) for r in rows], segments)
    assignment = np.array([int(r['partition']) for r in rows])
    train, test, folds = validate_split(y, segments, assignment)
    write_csv(OUT/'split_class_counts.csv', [{'activity':LABEL_NAMES[k],
        'total':int((y==k).sum()), 'train':int((y[train]==k).sum()),
        'test':int((y[test]==k).sum()), **{f'cv_validation_{f}':int((y[v]==k).sum())
        for f,(_,v) in enumerate(folds,1)}} for k in range(12)])
    return train, test, folds
