"""Checks for incorrect pairing, metric drift and incomplete final evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.stage5b_finalize import equal_table, summarize_pairs, primary, table


def test_table_comparison_rejects_identity_and_numeric_drift():
    expected = pd.DataFrame({'kepoi_name': ['K1', 'K2'], 'probability': [.2, .7]})
    equal_table(expected, expected.copy())
    for changed in (expected.iloc[::-1].reset_index(drop=True), expected.assign(probability=[.2, .71])):
        with pytest.raises(RuntimeError):
            equal_table(expected, changed)
    with pytest.raises(RuntimeError):
        equal_table(expected, expected.iloc[:1])


def test_pair_summary_retains_negative_brier_and_literal_ties():
    pairs = pd.DataFrame({'metric': ['brier']*4, 'difference': [-.1, 0., .2, -.3]})
    result = summarize_pairs(pairs, ['metric']).iloc[0]
    assert result['count'] == 4
    assert (result.higher, result.lower, result.tied) == (1, 2, 1)
    assert result['mean'] == pytest.approx(-.05)
    assert result['std'] == pytest.approx(np.std(pairs.difference, ddof=1))


def test_primary_filter_does_not_mix_qc_or_validation():
    frame = pd.DataFrame({'partition': ['test', 'test', 'validation'], 'track': ['primary', 'qc_pass', 'primary']})
    assert primary(frame).index.tolist() == [0]
    assert '1e-10' in table(pd.DataFrame({'variance': [1e-10]}))


def test_completed_matrix_and_gate_precede_test():
    root = Path('data/stage5b')
    integrity = json.loads((root/'run_integrity.json').read_text())
    assert integrity['models'] == 20 and integrity['protected_initial_fits'] == 9
    assert integrity['gate_time'] < integrity['first_test_time']
    runs = pd.read_csv(root/'run_outcomes.csv')
    assert len(runs) == 20 and not runs.duplicated(['strategy', 'repeat_id']).any()
    assert set(runs.strategy) == {'object_stratified', 'host_group_stratified'}
    assert runs.groupby('strategy').repeat_id.apply(set).apply(lambda x: x == set(range(10))).all()


def test_final_replay_and_all_matched_comparisons():
    root = Path('data/stage5b')
    replay = json.loads((root/'prediction_verification.json').read_text())
    assert replay['status'] == 'passed' and replay['models'] == 20 and replay['rows'] == 132740
    assert replay['max_probability_difference'] <= 1e-7
    for name, references in [('stage5_comparisons', 1), ('baseline_comparisons', 2)]:
        frame = primary(pd.read_csv(root/(name+'.csv')))
        assert len(frame) == references*20*8
        assert not frame.duplicated(['reference','strategy','repeat_id','metric']).any()
        assert np.allclose(frame.variant-frame.reference_value, frame.difference, rtol=0, atol=1e-12)
    acts = pd.read_csv(root/'activation_diagnostics.csv')
    assert len(acts) == 80 and acts.groupby(['strategy','repeat_id']).size().eq(4).all()
    assert acts.nonfinite_count.eq(0).all()
