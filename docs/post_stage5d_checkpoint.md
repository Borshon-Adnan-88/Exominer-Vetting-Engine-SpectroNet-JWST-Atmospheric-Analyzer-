# Post-Stage-5D project checkpoint

The three strongest findings are:

1. The certified Stage 4B flux-only Random Forest remains the principal benchmark; Stage 5D minus RF mean test ROC-AUC is -0.222503.
2. Stage 5C located a strong linear-probe loss at fusion; width128 reduced the concat-to-hidden AUC gap by 97.9%, improving 19/20 matched pairs without observed collapse.
3. Frozen Stage 5D test ROC-AUC is 0.646314 (object) and 0.668214 (host-group), with overall paired change +0.016475 versus Stage 5B.

The strongest negative result is that preserving linearly accessible information does not by itself make this CNN competitive with the tree baseline. The fusion bottleneck explains an important representation loss, but is not established as the sole limitation of end-to-end optimization or classification.

Object-minus-host-group test ROC-AUC averages -0.021900 (sample SD 0.056905). These split comparisons are descriptive, not evidence of equivalence or a pure causal leakage estimate; strategy-specific validation gains and reused test partitions require caution.

Recommended next step: **publication/manuscript consolidation**, focusing on the fixed benchmark, failure diagnosis, and the distinction between probe preservation and held-out prediction. Do not start another architecture experiment now. Kepler-to-TESS observational transfer is a possible separately designed later direction, with domain and label-shift controls; it is not executed here.
