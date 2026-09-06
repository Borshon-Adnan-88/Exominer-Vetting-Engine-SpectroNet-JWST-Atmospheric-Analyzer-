# Stage 1 Kepler catalogue data

`manifests/` contains the deterministic metadata manifest, summary, and label
conflict audit. `provenance/` records the exact TAP query context and SHA-256
checksums. Untouched TAP responses are stored in ignored `raw/` files and can be
recreated with `python -m scripts.fetch_kepler_catalog`.

This is a KOI-centred dataset. It is not the complete DR25 TCE population.

Stage 2 adds a committed deterministic pilot selection, frozen MAST product
inventory, retrieval provenance, FITS SHA-256 manifest, and processing/QC
metadata. The downloaded FITS under `lightcurves/raw/` and constructed NPZ files
under `lightcurves/processed/` are intentionally ignored because they are
reproducible binary artifacts.

`splits/` contains the deterministic Stage 3 assignment manifest, partition and
distribution audits, summary, provenance, and checksums for paired object-level
and host-group split strategies.

`stage4a/` contains compact full-cohort host rosters, discovery inventory,
outcomes, preflight reports, and checksums. Raw FITS, canonical NPZ arrays,
checkpoints, caches, and temporary files live outside Git under the runtime data
root.
