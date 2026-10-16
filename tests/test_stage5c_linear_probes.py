import numpy as np
import pytest
from scripts.stage5c_linear_probes import fit_probe, PROBES


def test_probe_scaler_and_fit_exclude_validation_labels():
    features=np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.],[100.,100.],[-100.,-100.]])
    labels=np.array([0,1,0,1,0,1]); train=np.array([True]*4+[False]*2)
    parameters=dict(C=1.,solver='lbfgs',max_iter=2000,tol=1e-4)
    first,_,scaler=fit_probe(features,labels,train,parameters)
    changed=labels.copy();changed[~train]=1-changed[~train]
    second,_,_=fit_probe(features,changed,train,parameters)
    assert np.array_equal(first,second)
    assert np.array_equal(scaler.mean_,features[train].mean(axis=0))
    assert scaler.n_samples_seen_==4
    assert len(PROBES)==6


def test_nonfinite_representation_stops_without_repair():
    with pytest.raises(RuntimeError):
        fit_probe(np.array([[0.],[np.nan]]),np.array([0,1]),np.array([True,False]),{})
