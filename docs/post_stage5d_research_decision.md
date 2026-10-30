# Research decision after Stage 5D

**Recommend B: strengthen and write the current Kepler paper.** This is a research-direction recommendation, not authorization to begin a new experiment. The manuscript outline, Results draft, and evidence checkpoint consolidate the completed work; none changes certified artifacts.

## Decision basis

The completed sequence already answers a useful bounded question. RF achieves approximately 0.880 test ROC-AUC under both split protocols. The original CNN exhibits an activation-related endpoint failure; LeakyReLU removes the observed collapse. Widening subsequently reduces the validation fusion-probe gap by approximately 97.9%, but the actual CNN reaches only 0.646314 object and 0.668214 host-group test AUC. Its overall paired test gain over width32 is 0.016475, while its RF deficit remains 0.222503. These outcomes justify careful writing of a benchmark and diagnostic/negative result before further model development.

The ten repeats overlap, and the same held-out partitions have been reused across model variants. Another architecture scored on those partitions would add development evidence, not restore independent confirmation. A literature review may establish that the current empirical scope needs a different framing; that review should precede additional computation.

## Comparison of the three options

| Option | Likely scientific value | Novelty potential, not yet verified | Workload | Main risk | Data requirements |
| --- | --- | --- | --- | --- | --- |
| **A. More CNN architecture work** | Could investigate why accessible branch/hidden discrimination is not realized by the output; may improve this particular classifier | Uncertain. Another width/architecture increment alone is unlikely to constitute a clear contribution without a mechanism-driven question and literature comparison | High: a new frozen hypothesis, implementation, controlled fits, diagnostics and evaluation plan | Further adaptation to reused partitions; architecture search may obscure the existing diagnosis and still fail to beat RF | Existing Kepler training/validation data can support development, but a credible new confirmation claim needs a separately defensible untouched evaluation source; current data are not a new independent test set |
| **B. Strengthen and write the current Kepler paper** | Clarifies the scientific scope, preserves a complete controlled evidence chain, and distinguishes stability, probe accessibility and prediction | Plausible as a combined benchmark/diagnostic negative result; publication value depends on verified related work rather than a claimed new method | Moderate: literature comparison, tighter task framing, results editing, figure curation and reproducibility documentation | Review may show limited novelty; careless framing could overstate host leakage, causality, or generality | No new observations or model fits required for the current paper-ready checkpoint; use certified tables/reports only |
| **C. Observational-domain robustness, especially Kepler → TESS** | Potentially high: tests whether a Kepler-derived classifier survives changes in cadence, time baseline, passband, noise, selection and label provenance | Potentially stronger but unverified; transfer across these missions is not assumed novel | Very high: target/label design, product inventory, preprocessing comparability, selection accounting, domain controls and independent evaluation | Label mismatch, target overlap, insufficient transit coverage, mission-specific artifacts and selection confounding can mimic domain failure or success | A separately frozen TESS cohort with explicit labels/conflicts, official light curves, ephemerides, QC accounting, host identity/overlap controls, and a training/adaptation/test separation appropriate to the proposed transfer question |

## Why B is the next axis

The strongest current contribution is the chain of evidence, not an architecture claimed to outperform established methods. The manuscript can show that a classical nonlinear baseline is strong on the common frozen representation, that an identifiable CNN failure can be repaired, and that nearly eliminating a probe-level fusion gap is insufficient to close the held-out prediction deficit. The small classical split differences and variable CNN split effects add a careful boundary on what host-aware evaluation means in this cohort.

The immediate intellectual work is to verify the literature claims listed in [the evidence checkpoint](manuscript_evidence_after_stage5d.md#what-may-be-publishable-or-novel), distinguish this confirmation-selected KOI task from other vetting benchmarks, and make the result tables and limitations publication-ready. A negative or diagnostic paper should be evaluated on the validity and usefulness of that bounded contribution, not on retrospective significance thresholds or selected favorable repeats.

## Is another CNN experiment justified now?

**No.** The evidence leaves an unresolved end-to-end learning limitation, but it does not uniquely identify a next architecture change. A further architecture experiment would need a literature-informed, discriminating hypothesis and a fresh evaluation justification. Neither follows automatically from the width128 result. This is not a claim that CNN improvement is impossible; it is a judgment about the next most informative use of the completed evidence.

Kepler-to-TESS remains an interesting possible later axis after consolidation and literature assessment. It would require a separate scientific design and authorization; it is not a continuation to execute here. No new dataset, download, model, split, probe or held-out evaluation is initiated by this decision.

Evidence: [master results and Results draft](manuscript_results_stage1_to_stage5d.md), [Stage 4B report](kepler_stage4b_report.md), [Stage 5C diagnostics](kepler_stage5c_linear_probes.md), and [Stage 5D frozen test evaluation](kepler_stage5d_test_evaluation.md).
