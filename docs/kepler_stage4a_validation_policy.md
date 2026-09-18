# Stage 4A integrity policy and certification recovery

This policy applies only to Stage 4A certification. It does not change the Stage 1
cohort, Stage 2 preprocessing or QC behavior, Stage 3 assignments, frozen MAST
inventory, or Stage 4A scientific configuration. No model or Stage 4B work is
included. Evidence is recorded in
`data/stage4a/kepler_stage4a_integrity_audit_v1.json`.

## Finding: CHECKSUM without DATASUM

The installed Astropy version is 7.2.0. Its `_ValidHDU.verify_checksum` computes
the data checksum only when a `DATASUM` card is present; otherwise it passes zero
to the checksum calculation. The sampled Kepler files have `CHECKSUM` cards and
no `DATASUM` cards. As a result the empty primary HDU passes, while extensions
containing data return failure. The original Stage 2 wrapper faithfully recorded
those return values. Its code and the historical records remain unchanged.

An independent calculation sums the untouched on-disk header **and padded data**
records as big-endian unsigned 32-bit words, with end-around carry. A valid HDU
with CHECKSUM sums to `0xffffffff`. This is the registered FITS checksum
convention, and does not require substituting a zero data contribution when a
DATASUM keyword is absent. When DATASUM is supplied, the audit also checks the
padded data records against that value.

All 24 HDUs in eight fresh MAST downloads and all 33 HDUs in the 11 retained
FITS pass this independent checksum calculation. The two sets overlap in two
files, giving 17 unique independently checked products / 51 unique HDUs. All
sampled primary HDUs pass Astropy's original verifier; all sampled LIGHTCURVE
and APERTURE extensions reproduce its failure. Required science columns, target
identity, quarter, DR25 metadata, cadence, parsing and structural validation pass.

The eight exact frozen URIs produce the same byte sizes and SHA-256 values as
the original retrieval records. Two can also be compared directly against
retained originals; their complete HDU audit results agree. The other six
original files were purged, so their comparison is against the historical
retrieval hash, not a retained copy. Only 1,794,240 bytes were freshly downloaded.

This supports an **Astropy CHECKSUM-without-DATASUM interpretation defect**.
There is no evidence in this sample for damaged payloads, stale embedded
CHECKSUM values, or a download transformation. In particular, the evidence does
not justify saying that MAST supplied invalid embedded checksums. Nor does it
establish the behavior of every Kepler archive product outside this sample.

Primary references:

- [Registered FITS checksum convention](https://fits.gsfc.nasa.gov/registry/checksum.html)
- [Convention specification](https://fits.gsfc.nasa.gov/registry/checksum/checksum.pdf)
- [Astropy 7.2.0 verifier source](https://github.com/astropy/astropy/blob/v7.2.0/astropy/io/fits/hdu/base.py)
- [Astropy checksum verification documentation](https://docs.astropy.org/en/stable/io/fits/usage/verification.html)
- [MAST Kepler data release notes](https://archive.stsci.edu/missions-and-data/kepler/documents/data-release-notes)

The audit records a SHA-256 of the installed verifier's source. The implemented
independent sum is in `kepler_scale/integrity.py`; regression tests include a
valid CHECKSUM-only file and a one-byte payload mutation that must fail.

## Evidence and acceptance rules

Three distinct evidence layers must remain separate:

1. **External byte identity:** frozen MAST URI and advertised size, retrieval-time
   SHA-256, and recorded source-set hash. The eight fresh comparisons validate
   the historical chain on a bounded sample. Retrieval hashes were computed by
   this pipeline; they are not archive-published cryptographic checksums.
2. **FITS structure and identity:** original per-product validation of target,
   quarter, DR25, cadence and required science columns; revalidated directly on
   all retained and fresh audit files.
3. **Embedded checksum evidence:** preserve the historical Astropy result as
   `legacy_astropy_embedded_checksums_valid`. Its systematic false value is a
   documented interpretation warning. Record independent full-HDU checksum
   results separately, only for files actually checked. A failing independent
   checksum, size, SHA-256, structure or identity remains a certification blocker.

The evidence supports interpreting the common historical warning; it does not
turn uninspected, purged files into individually reverified sources. The final
FITS manifest explicitly marks them `not_rechecked_source_purged`. No header is
repaired, no source hash is replaced, and no failed scientific QC is overridden.

Every canonical NPZ is separately required to pass file SHA-256, all six logical
array hashes, frozen shapes/dtypes, finite values, nonnegative counts, mask/count
agreement and observed-fraction consistency. Its identity and path must match
the unchanged cohort, without missing, duplicate or orphan arrays. Deterministic
ZIP reconstruction must reproduce the original NPZ bytes. Metadata is built
twice and compared byte for byte. This is an output determinism check, not a new
full preprocessing run from purged FITS.

Constructed QC-fail arrays remain `model_data_available=true`, as required by
the frozen Stage 2 preserve-constructible-arrays policy. Availability is not an
endorsement for model inclusion. Attrition retains every Stage 3 denominator,
partition and KOI without substitution. Period strata are descriptive audit
bins only; they introduce no selection rule.

## Host 3530668 retained sources

The original host state recorded `raw_purged=true` at
`2026-09-09T08:23:13.046000+00:00`. All 11 surviving Q7-Q17 FITS have last-write
times after that record, ending at `2026-09-09T08:23:39.381266+00:00`.
Six have creation times before the state, while the five Q13-Q17 files were
created after it. Their size and SHA-256 match the state's source records.
The original batch-00041 log reports 4,881,600 bytes of scratch residue despite
ten purged hosts. The surviving batch report was subsequently overwritten by
a fully resumed run. The integrity audit retains the original state/log hashes
and timestamps as evidence.

These are post-purge writes, strongly consistent with overlapping host runs:
one writer records completion/purge while another continues its sequential
downloads. A simple interruption of the first purge would not explain later
file writes. The host has `deterministic_rebuild_verified=false`, and the rebuild
code writes temporary NPZ, not FITS, so these are not rebuild residue. The logs
do not retain process IDs; attribution to particular concurrent processes cannot
be proven retrospectively from the available records.

Two operational defects are reproducible in the code: there was no per-host
writer lock, and resume returned complete state without inspecting raw residue.
The fix holds an OS-released exclusive host lock across state inspection,
download, processing and purge. Resume now marks a stale purge claim false and
records a review reason without deleting or downloading anything. Locks release
on process exit; their persistent lock-file identity is not unlinked, avoiding a
second writer locking a replacement file. Tests cover overlapping writers and
resume with post-purge residue. These locks coordinate processes accessing the
same filesystem; they are not distributed locks across cloud-synced machines.

For the real host, only the operational purge flag and retention explanation
were reconciled after verifying its existing NPZ hash. The 11 FITS (4,881,600
bytes) remain unchanged as integrity evidence. The eight fresh audit files are
also retained in a separate temporary audit directory. No raw cleanup was
performed. The frozen artifacts and original per-product hashes were untouched.

## Limits and scientific warnings

Certification concerns integrity, reproducibility and cohort accounting. It does
not certify labels as astrophysical truth, validate a model, or remove pilot QC
warnings. Aggregate phase-centering, inversion, SAP/PDCSAP morphology,
interpolation and quarter-discontinuity flags must be reported for review.
Broad flags can reflect real variability as well as preprocessing limitations;
the audit does not infer a cause or silently correct arrays.

The total unique FITS bytes processed can be reconstructed from the source
records. Exact lifetime network transfer cannot: retries and resumed batch
reports may overwrite earlier transfer counters. Report the surviving counters
with that limitation and the bounded audit download separately.
