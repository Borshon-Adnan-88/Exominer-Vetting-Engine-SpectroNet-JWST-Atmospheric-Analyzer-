import numpy as np
import pandas as pd
import pytest
from scripts.stage5a_finalize_report import group_labels,table,aggregate,OUT,TABLES


def test_disjoint_groups_preserve_independent_flags():
    r=pd.DataFrame(dict(probability_range=[1e-8,.4,.4],unique_probability_count=[2,3,4],range_below_1e7=[True,False,False],few_values=[True,True,False]))
    assert group_labels(r).tolist()==['severe','low_diversity_only','non_collapsed']
    r.loc[0,'range_below_1e7']=False
    with pytest.raises(RuntimeError): group_labels(r)


def test_table_preserves_tiny_values():
    text=table(pd.DataFrame({'range':[1e-8],'notes':['a|b']}))
    assert '1e-08' in text and 'a/b' in text


def test_measured_group_findings():
    r=pd.read_csv(OUT/'run_diagnostics.csv'); a=pd.read_csv(OUT/'activation_diagnostics.csv')
    r['group']=group_labels(r)
    assert r.group.value_counts().to_dict()==dict(non_collapsed=13,severe=4,low_diversity_only=3)
    joined=a.merge(r[['strategy','repeat_id','few_values']],on=['strategy','repeat_id'],validate='many_to_one')
    hidden=joined[joined.few_values & joined.layer.eq('hidden')]
    assert len(hidden)==7 and hidden.zero_fraction.eq(1).all() and hidden.between_object_variance.eq(0).all()
    n=pd.read_csv(OUT/'numerical_sensitivity.csv').merge(r[['strategy','repeat_id','few_values']],on=['strategy','repeat_id'])
    assert n[n.few_values].maximum_probability_difference.eq(0).all()
    assert n.changed_labels.eq(0).all()


def test_group_summaries_keep_kernel_and_bias_separate():
    frames={name:pd.read_csv(OUT/(name+'.csv')) for name in TABLES}
    priors=frames['run_diagnostics'][['strategy','repeat_id']].copy()
    runs,out=aggregate(frames,priors)
    assert out['group_comparison'].runs.sum()==20
    w=out['weight_group_comparison']
    assert set(w.role)=={'weight','bias'}
    assert not w.duplicated(['group','layer','role']).any()
    assert len(out['activation_group_comparison'])==18
