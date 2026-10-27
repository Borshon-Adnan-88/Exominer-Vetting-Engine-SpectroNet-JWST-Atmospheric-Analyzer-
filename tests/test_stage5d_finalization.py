"""Reporting uses paired sample summaries and rejects corrupt inventories."""
import pytest
from scripts import stage5d_finalize as f


def test_sample_summary_keeps_negative_and_tied_pairs():
    result = f.stats([-.1, 0, .1])
    assert result['mean'] == 0
    assert result['median'] == 0
    assert result['sd'] == pytest.approx(.1)
    assert (result['improved'],result['worsened'],result['tied']) == (1,1,1)


def test_inventory_rejects_corruption_and_duplicate_entries(tmp_path):
    artifact = tmp_path/'value.csv'; artifact.write_text('frozen')
    line = f.r.sha256(artifact)+'  value.csv\n'
    manifest = tmp_path/'checksums.sha256'; manifest.write_text(line)
    assert f.inventory(manifest,tmp_path) == {'value.csv':f.r.sha256(artifact)}
    artifact.write_text('changed')
    with pytest.raises(RuntimeError,match='Pinned artifact changed'):
        f.inventory(manifest,tmp_path)
    manifest.write_text(line+line)
    with pytest.raises(RuntimeError,match='Duplicate'):
        f.inventory(manifest,tmp_path)


def test_reporting_has_no_training_or_model_inference_entrypoint():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(f.__file__).read_text(encoding='utf-8'))
    names = {n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
             for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not names.intersection({'fit','fit_probe','predict','load_model','execute','load_scope'})
