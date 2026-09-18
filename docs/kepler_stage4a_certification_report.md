# Stage 4A certification report ? 2026-09-18

**Decision: certified with documented scientific warnings. Stage 4B has not started.**

The systematic FITS warning is an Astropy 7.2.0 CHECKSUM-without-DATASUM interpretation defect. The sampled embedded CHECKSUM values are valid under an independent checksum over the original header and data records. No source-file corruption or archive-metadata invalidity was found in the audited sample. Historical false flags remain preserved, with their interpretation documented.

See the [validation policy](kepler_stage4a_validation_policy.md) and [machine-readable integrity evidence](../data/stage4a/kepler_stage4a_integrity_audit_v1.json). Certification of purged FITS uses the frozen inventory and retrieval-time hash/identity chain plus bounded fresh comparisons; it does not claim a new byte-level inspection of all 88,642 original sources.

## Final counts

| Measure | Result |
|---|---:|
| Batches accounted for | 581 |
| Hosts accounted for | 5,804 |
| KOIs assigned/accounted for | 6,637 |
| Constructed QC pass | 6,354 |
| Constructed QC fail | 283 |
| Retrieval-failure KOIs | 0 |
| No-product hosts / no-product KOIs | 0 / 0 |
| Invalid-source-product KOIs | 0 |
| Hard preprocessing failures | 0 |
| Other technical failures | 0 |
| model_data_available true / false | 6,637 / 0 |
| Canonical NPZs independently verified | 6,637 |
| Logical array hashes independently verified | 39,822 |
| Missing / duplicate-identity / orphan NPZs | 0 / 0 / 0 |
| Duplicate whole-NPZ SHA-256 values | 0 |
| Canonical array bytes | 141,102,620 |
| Canonical array size (MiB) | 134.565945 |
| Unique FITS products processed | 88,642 |
| Unique raw FITS bytes processed | 38,803,838,400 |
| Downloaded bytes in surviving batch reports | 35,954,003,520 |
| Additional bounded audit download bytes | 1,794,240 |

The canonical-array size is 141,102,620 bytes. The 15 finalized scientific metadata files add 40,073,607 bytes, giving **181,176,227 bytes** for the arrays plus finalized metadata. This bundle accounting excludes unchanged upstream inputs, documentation, runtime caches and retained raw evidence. Exact lifetime network-transfer bytes are unavailable because resumed reports replace earlier transfer counters and retries are not fully recoverable. The surviving download counter is not the total raw-data denominator.

Every indexed NPZ exists and matches its original SHA-256. All six logical hashes per KOI match. Global arrays have 2,000 elements and local arrays 200; flux/count/mask dtypes are float32/int32/bool in the frozen little-endian representation. Finite-value, count/mask and coverage-metadata checks pass. Reconstructing the canonical ZIP bytes from the existing arrays reproduced every NPZ exactly. No photometry was reprocessed.

## Fresh source comparison

Sample selection: first, median-index and last host in the sorted inventory, each with its earliest and latest quarter, plus the earliest and latest retained files for host 3530668. Four hosts, quarters 1/7/17, and three file sizes are represented. Only eight exact frozen MAST URIs were downloaded.

All rows have matching advertised/local sizes and SHA-256 against the original retrieval record. Rows for host 3530668 also match retained original bytes and HDU results. The other originals were already purged. Each fresh file has three readable HDUs: Astropy reports pass/fail/fail; the independent full-HDU calculation reports pass/pass/pass. DATASUM is absent in every sampled HDU. Target, quarter, DR25, science columns and cadence validate.

| Host | Quarter | Advertised = fresh local bytes | SHA-256 |
|---|---:|---:|---|
| 757450 | 1 | 192,960 | `3d6c3f00d0f720dce9eead4a0a1d89bfe4e22242bf910b088dec0a54396344cf` |
| 757450 | 17 | 187,200 | `88526473b426da3e2f8f1ea6029041faab3049fe39896f773fc3989dbb60fd82` |
| 3530668 | 7 | 466,560 | `05a9c99b6df4ba195f2375546b9734066b1525d6eea43c7bff533105fdda0e62` |
| 3530668 | 17 | 187,200 | `466ccbddd0def4a23ee31e5bf2292319dc5cfb6b0f9e83373b448f15c927cdcb` |
| 7841986 | 1 | 192,960 | `56c9db0d84a8721baf6cb3e4d4567978d0cdc8b1333fbb457db6d11596d5f1df` |
| 7841986 | 17 | 187,200 | `fae957ff7a765753dbb401ec07bdbaed7b29f5dc309b3a8764ffe92ff0eeb280` |
| 12935144 | 1 | 192,960 | `3b2ae32a6bd6d99667ff2dee50dde0201ee5c495a077b4bfc492c8b118707796` |
| 12935144 | 17 | 187,200 | `f27d87e618a0ecd60e356fbad1b708cc8cd331949bf936eb4c9e1de2b501bf87` |

Frozen URIs, product filenames, per-HDU header CHECKSUM cards, DATASUM absence/status, target/release metadata, original/local hashes and retained-file timestamps are listed for every file in the integrity JSON. All 11 retained FITS also pass: 17 distinct products and 51 distinct HDUs were independently checked overall (the fresh and retained samples overlap by two files).

The [registered FITS convention](https://fits.gsfc.nasa.gov/registry/checksum.html) defines verification over the entire HDU. Its [specification](https://fits.gsfc.nasa.gov/registry/checksum/checksum.pdf) permits CHECKSUM and DATASUM independently. Inspection of the installed Astropy verifier demonstrates its zero-data-sum branch when DATASUM is absent. The independent implementation is tested against a valid CHECKSUM-only file and a one-byte payload corruption. No checksum gate was silently disabled.

## Attrition and frozen assignments

All 132,740 original assignment rows remain: two strategies ? ten repeats ? 6,637 KOIs. Every strategy/repeat retains 6,637 assignments. No KOI was replaced, excluded because of QC, or moved between partitions. Model-data availability describes constructible arrays, not model inclusion.

| Stage 1 class | Assigned | QC pass | QC fail | Unavailable |
|---|---:|---:|---:|---:|
| False positive (0) | 3,963 | 3,716 | 247 | 0 |
| Confirmed (1) | 2,674 | 2,638 | 36 | 0 |

The [attrition report](../data/stage4a/kepler_stage4a_attrition_report_v1.csv) contains the joint strategy, repeat, train/validation/test partition, class, single/multiple-host and period-stratum counts with original assigned denominators. [Marginal summaries](../data/stage4a/kepler_stage4a_attrition_summary_v1.json) include every requested dimension. Period bins (0?10, 10?100, and ?100 days, with invalid/missing separate) are descriptive audit strata and do not change scientific selection.

## Aggregate diagnostics and unresolved scientific warnings

| Flag | KOIs | Fraction of 6,637 |
|---|---:|---:|
| phase_inversion_flag | 145 | 2.18% |
| sap_pdcsap_material_difference_flag | 3,648 | 54.96% |
| strong_quarter_discontinuity_flag | 1,127 | 16.98% |
| unusually_high_interpolation_flag | 344 | 5.18% |
| visibly_miscentered_flag | 605 | 9.12% |

The SAP/PDCSAP material-difference flag affects more than half the cohort and merits scientific review. The frozen fractional depth metric is sensitive to shallow depths; this audit does not establish the cause of each flag. The maximum reported morphology difference fraction is 569.68, and the maximum quarter-boundary jump is 1.45444 in normalized flux. No array or threshold was changed.

Median observed-bin coverage is 1.0 for both views. Global minimum coverage is 0.338; local minimum is 0.185 and its fifth percentile is 0.795. Maximum global/local interpolation fractions are 0.662/0.815. There are 4,252 KOIs with 17 valid quarters and 231 with fewer than eight. The frozen baseline fraction measures retained usable span relative to available source files, not completeness against the entire mission.

QC reasons can overlap: `global_coverage_below_0.80`: 36, `global_missing_run_above_0.10`: 10, `local_missing_run_above_0.25`: 27, `observed_transit_epochs_below_3`: 2, `valid_quarters_below_8`: 231.

Quarter availability for every quarter, numerical distributions and failure counts are in the [aggregate diagnostics](../data/stage4a/kepler_stage4a_aggregate_diagnostics_v1.json). No per-target plots were generated.

## Raw retention and operational repair

Eleven original FITS for host 3530668 remain in local scratch, totaling 4,881,600 bytes. All were written after the recorded purge-complete state; five were also created after it. The original batch log already reported this residue. This supports post-purge writes from overlapping processing, not deterministic-rebuild residue. Exact historical process attribution is unavailable because process IDs were not retained.

The minimal operational fix adds an OS-released per-host writer lock and makes resume reconcile a stale purge claim when files remain, without deleting or downloading. Focused regressions pass. Only host 3530668?s operational state was updated (`raw_purged=false` and an explicit evidence-retention reason); its scientific outcomes and source hashes remain unchanged.

The research-data root contains no raw FITS or `.part` files. The separately named cloud scratch folder is empty. Local scratch contains the 11 intentionally retained FITS and no `.part` files. Eight fresh comparison FITS (1,794,240 bytes) remain in the separate temporary audit directory, with no `.part` files. Nothing was deleted. These checks cover the project dataset and its identified scratch locations, not unrelated personal Drive folders.

## Validation results

- Final Stage 4A acceptance audit: passed.
- All 6,637 NPZ SHA-256 checks, 39,822 logical hashes, shapes/dtypes and deterministic ZIP reconstruction: passed.
- Full source-record membership, advertised sizes and per-host source-set hashes: passed for 88,642 products.
- All nine existing frozen metadata checksums and all 14 entries in the final metadata checksum manifest: passed.
- Deterministic metadata built twice from the verified inputs: byte-identical; written bytes and hashes verified.
- Frozen source hashes, audit implementation/policy hashes and provenance references: passed.
- Absolute-path audit across Stage 4A metadata: passed.
- `git diff --check`: passed.

The complete offline suite was run **once**: **116 passed, 3 setup errors, 3 network tests deselected** in 185.88 seconds. The three errors were `FileNotFoundError` from a missing parent for the requested pytest temporary directory; there were no assertion failures. After creating that parent, only those three affected tests were retried: **3 passed** in 1.10 seconds. Thus all **119 selected offline tests passed**, without rerunning the full suite. An earlier focused ten-test check passed its assertions but encountered a permission error during cleanup of pytest?s default external temporary-directory link; the workspace-local test directory avoided that cleanup path.

Full-suite command:

```text
python -m pytest -m "not network" --basetemp=data/tmp/stage4a-certification-pytest-20260918 --junitxml=data/tmp/stage4a-certification-tests-20260918.xml
```

Targeted retry: `test_missing_live_field_aborts_before_catalogue_request`, `test_output_is_byte_reproducible_and_input_order_independent`, and `test_file_hash_and_sorted_json_are_deterministic`. No network tests ran; the only archive requests were the eight authorized integrity comparisons.

## Final hashes

Paths below are relative to `data/stage4a/`. The final provenance is separate from the unchanged pre-download provenance. The checksum manifest covers the scientific metadata and integrity evidence; this human-readable report records the subsequent test results.

| Artifact | SHA-256 |
|---|---|
| `kepler_stage4a_aggregate_diagnostics_v1.json` | `68a1b7fd37825231a5ce67a1d2f98ca3bfc34a39d2f22a3a5ec4ff9bfadc2ffa` |
| `kepler_stage4a_array_checksums_v1.sha256` | `ae2e667e4768c4a1a5d047c95c29f607ed00f3ea47ef910cf2fc8480cc5c157b` |
| `kepler_stage4a_array_index_v1.csv` | `a647925a26b869519b5dce47c17e40e8e256e0d7cb9eb2e5e3c3b71bfd77ed09` |
| `kepler_stage4a_attrition_report_v1.csv` | `770d088e1248004fbd1a9648e235a799731f3eed936291209a32da58de9f93f2` |
| `kepler_stage4a_attrition_summary_v1.json` | `ce82091a0622d95e507352b4c623fa340785ed322c5f73fe2241777b7642f5ae` |
| `kepler_stage4a_final_provenance_v1.json` | `30d938c3b1e36126764c74c8f44381025304c031bfdee0db295e7818915aacbd` |
| `kepler_stage4a_fits_checksums_v1.sha256` | `a832ea9c39f6a11cc85b49efe1eb20a20ddd19fe834026509e86f7e41cc3590d` |
| `kepler_stage4a_fits_manifest_v1.csv` | `514a3a2c7a3b311b5ec07811cb1478fca07604d00a1238441f10f2b646eaa5fa` |
| `kepler_stage4a_host_outcomes_v1.csv` | `79bdc3f58fb1598e192328304abaf9676576d0a6e1890a14ee394ae1b84d04d9` |
| `kepler_stage4a_integrity_audit_v1.json` | `a9ac8869efe2ba679df6c323f54aba13fc1ea3673aacfcfd3c762d214773deb7` |
| `kepler_stage4a_processing_manifest_v1.csv` | `0290763f6c9855df96aa7576ef0da37cdd28eb19d9585e0ee9479064874c8dc6` |
| `kepler_stage4a_raw_cleanup_audit_v1.json` | `9e14d90e4c1a9349798c4380c0dec767cc3f776ac3d4433ecd556dfcf62f9d67` |
| `kepler_stage4a_runtime_metadata_checksums_v1.sha256` | `b25f231033c04627a06180f131293171ea3edbd77b791d2498f904d283a31cde` |
| `kepler_stage4a_summary_v1.json` | `42ae973fd8b772777b21785fe48c82fa012630eb6f6a9c0961eb3af3e4a7b5bb` |
| `kepler_stage4a_final_metadata_v1.sha256` | `6c4bcb6ba41176c057267ccbe6965e07b2be657a952c6848392284cad79475d5` |

The raw-FITS checksum manifest records original retrieval hashes and logical source paths. Purged originals are not claimed to exist or to have been freshly rehashed. The canonical array checksum manifest was verified against every present NPZ.

## Exact deliverable changes

Modified existing repository files:

- `README.md`
- `kepler_scale/runner.py`
- `tests/test_stage4a_runner.py`

Created repository files:

- `data/stage4a/kepler_stage4a_aggregate_diagnostics_v1.json`
- `data/stage4a/kepler_stage4a_array_checksums_v1.sha256`
- `data/stage4a/kepler_stage4a_array_index_v1.csv`
- `data/stage4a/kepler_stage4a_attrition_report_v1.csv`
- `data/stage4a/kepler_stage4a_attrition_summary_v1.json`
- `data/stage4a/kepler_stage4a_final_metadata_v1.sha256`
- `data/stage4a/kepler_stage4a_final_provenance_v1.json`
- `data/stage4a/kepler_stage4a_fits_checksums_v1.sha256`
- `data/stage4a/kepler_stage4a_fits_manifest_v1.csv`
- `data/stage4a/kepler_stage4a_host_outcomes_v1.csv`
- `data/stage4a/kepler_stage4a_integrity_audit_v1.json`
- `data/stage4a/kepler_stage4a_processing_manifest_v1.csv`
- `data/stage4a/kepler_stage4a_raw_cleanup_audit_v1.json`
- `data/stage4a/kepler_stage4a_runtime_metadata_checksums_v1.sha256`
- `data/stage4a/kepler_stage4a_summary_v1.json`
- `docs/kepler_stage4a_certification_report.md`
- `docs/kepler_stage4a_validation_policy.md`
- `kepler_scale/finalize.py`
- `kepler_scale/integrity.py`
- `scripts/stage4a_audit_integrity.py`
- `scripts/stage4a_certify.py`
- `tests/test_stage4a_certification.py`
- `tests/test_stage4a_integrity.py`

External runtime changes (paths relative to the dataset root):

- Modified `cache/host_state/003530668.json`: operational purge flag and retention explanation only.
- Created `cache/host_locks/003530668.lock`: persistent lock identity; no active lock remains.

Created temporary audit files in `%TEMP%/exominer-stage4a-integrity-20260918/`:

- `kplr000757450-2009166043257_llc.fits`
- `kplr000757450-2013131215648_llc.fits`
- `kplr003530668-2010355172524_llc.fits`
- `kplr003530668-2013131215648_llc.fits`
- `kplr007841986-2009166043257_llc.fits`
- `kplr007841986-2013131215648_llc.fits`
- `kplr012935144-2009166043257_llc.fits`
- `kplr012935144-2013131215648_llc.fits`

Ignored operational evidence: `data/tmp/stage4a-final-acceptance-checks.json`, `data/tmp/stage4a-certification-tests-20260918.xml`, and `data/tmp/stage4a-certification-retry-20260918.xml`; pytest fixture directories are under the corresponding `data/tmp/stage4a-certification-*` roots. Normal Python/pytest cache files are not scientific deliverables.

## Scope confirmation

Stage 1 artifacts, Stage 2 scientific preprocessing code/configuration, Stage 3 assignments, Stage 4A scientific configuration and frozen MAST inventory are unchanged. No full-cohort reprocessing or bulk archive download occurred. No model code was introduced or trained. Stage 4B was not started. No commit was made. Work stops here for review.
