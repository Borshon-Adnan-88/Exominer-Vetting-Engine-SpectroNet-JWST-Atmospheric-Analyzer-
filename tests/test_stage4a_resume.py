from kepler_scale.checkpoints import read_checkpoint,write_checkpoint
def test_atomic_checkpoint_resume(tmp_path):
 assert read_checkpoint(tmp_path,42) is None; value={"kepid":42,"status":"products_found"}; write_checkpoint(tmp_path,42,value); assert read_checkpoint(tmp_path,42)==value; assert not list(tmp_path.rglob("*.part"))
