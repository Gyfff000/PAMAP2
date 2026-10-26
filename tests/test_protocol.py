"""Guard the meaningful failure modes: channel mapping, subject leakage and sensor removal."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from pamap2.common import CHANNELS
from pamap2.prepare import runs, long_missing_rows, interpolate, features
from pamap2.experiment import feature_indices, normalization
from pamap2.models import make_model
from pamap2.protocol import validate_split, OUT
import csv

class ProtocolTests(unittest.TestCase):
    def frozen_split(self):
        with (OUT/'split_assignments.csv').open(encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        return tuple(np.array([int(r[k]) for r in rows]) for k in ['activity','segment','partition'])

    def test_exact_ratio_and_training_only_cv(self):
        y,segments,assignment=self.frozen_split()
        train,test,folds=validate_split(y,segments,assignment)
        self.assertEqual((len(train),len(test)),(5187,2223))
        for fit,val in folds:
            self.assertEqual((len(fit),len(val)),(3458,1729))
            self.assertFalse(set(test)&(set(fit)|set(val)))
            self.assertFalse(set(segments[fit])&set(segments[val]))
            self.assertFalse(set(segments[test])&set(segments[val]))

    def test_rejects_window_leakage(self):
        y,segments,assignment=self.frozen_split()
        # Swap two same-label windows across partitions: sizes/classes stay valid,
        # but each entire recording segment must still belong to only one side.
        a=np.flatnonzero((y==0)&(assignment==0))[0]
        b=np.flatnonzero((y==0)&(assignment==1))[0]
        assignment[a],assignment[b]=assignment[b],assignment[a]
        with self.assertRaisesRegex(AssertionError,'cross partitions'):
            validate_split(y,segments,assignment)

    def test_original_column_mapping(self):
        # Official columns are 1-based: hand acceleration 5-7, gyro 11-13.
        self.assertEqual(CHANNELS[:6],[4,5,6,10,11,12])
        self.assertEqual(CHANNELS[6:12],[v+17 for v in CHANNELS[:6]])
        self.assertEqual(CHANNELS[12:],[v+34 for v in CHANNELS[:6]])

    def test_rejects_missing_validation_class(self):
        y,segments,assignment=self.frozen_split()
        y[(assignment==1)&(y==11)]=10
        with self.assertRaisesRegex(AssertionError,'cover all 12 classes'):
            validate_split(y,segments,assignment)

    def test_long_gap_is_excluded(self):
        x=np.ones((180,2))
        x[20:71,0]=np.nan
        x[100:102,1]=np.nan
        bad=long_missing_rows(x,50)
        self.assertEqual(int(bad.sum()),51)
        self.assertTrue(np.isfinite(interpolate(x[72:])).all())
        self.assertEqual(runs(np.array([0,1,1,0,1],bool)),[(1,3),(4,5)])

    def test_features_are_sensor_local(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(3,18,256)).astype(np.float32)
        a=features(x)
        x[:,6:12] *= 3
        b=features(x)
        self.assertEqual(a.shape,(3,150))
        np.testing.assert_array_equal(a[:,:50],b[:,:50])
        np.testing.assert_array_equal(a[:,100:],b[:,100:])
        self.assertFalse(np.allclose(a[:,50:100],b[:,50:100]))
        self.assertEqual(len(feature_indices('rf_no_chest')),100)
        self.assertFalse(set(feature_indices('rf_chest')) & set(feature_indices('rf_no_chest')))

    def test_normalization_uses_training_only(self):
        x=np.zeros((3,18,256),dtype=np.float32)
        f=np.zeros((3,150),dtype=np.float32)
        x[0]=1;x[1]=3;x[2]=999
        f[0]=2;f[1]=4;f[2]=999
        n=normalization(x,f,np.array([0,1]))
        self.assertTrue(np.all(n['mean']==2))
        self.assertTrue(np.all(n['fmean']==3))

    def test_network_shapes_and_gradients(self):
        import torch
        for name in ['cnn','fusion']:
            model=make_model(name)
            out=model(torch.randn(3,18,256),torch.randn(3,150))
            self.assertEqual(tuple(out.shape),(3,12))
            out.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':
    unittest.main()
