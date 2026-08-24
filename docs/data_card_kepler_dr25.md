# Data card: Kepler Q1-Q17 DR25 KOIs

## Source and scope

The Stage 1 catalogue is retrieved from the NASA Exoplanet Archive TAP table
`q1_q17_dr25_koi`, the final Q1-Q17 DR25 KOI delivery produced with Kepler SOC
pipeline 9.3. Archive DOI: **10.26133/NEA4**.

The query deliberately retrieves every row for the selected fields. Filtering,
label construction, and conflict exclusion happen locally under version control.

## Unit of observation

Each row is a Kepler Object of Interest identified by `kepoi_name`. A KOI is a
vetted subset of threshold-crossing events, not a synonym for a confirmed planet.
`kepid` identifies the Kepler target and is the future host-grouping key. Multiple
KOIs can therefore belong to one `kepid`.

This KOI-centred cohort does not represent all DR25 transit detections. Because
the research question concerns transit-event vetting, a future sensitivity
dataset based on `q1_q17_dr25_tce` may be required. That work is outside Stage 1.

## Labels and limitations

The binary policy is documented in `docs/label_policy.md`. Confirmation depends
on follow-up, signal quality, scientific interest, and historical information.
Consequently, confirmed-versus-false-positive classification is not identical to
uniform Robovetter planet-candidate vetting and should not be described as such.

Candidates and unlabelled KOIs remain in the manifest for accounting but are
excluded from the binary cohort. Contradictory source labels are retained in a
separate conflict audit and excluded by default.

## Feature status

No model features are approved in Stage 1. SNR, ephemeris, transit, positional,
and stellar metadata are `candidate_feature_pending_review`. Archive disposition,
Robovetter score and flags, comments, confirmation names, and derived label fields
are forbidden future model features because they directly encode or explain the
target.

## Reproducibility

The provenance record includes the endpoint, exact ADQL, retrieval time, schema
metadata hash, raw response hash, row count, environment, and derived artifact
hashes. Raw TAP responses are ignored by Git; their content-addressed filename and
SHA-256 allow integrity verification.
