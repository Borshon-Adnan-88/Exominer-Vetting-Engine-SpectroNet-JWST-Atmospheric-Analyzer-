# Stage 2 Kepler light-curve preprocessing

Stage 2 consumes only the frozen Stage 1 KOI manifest. It selects a deterministic
16-KOI pilot, freezes official MAST Q1–Q17 DR25 long-cadence light-curve product
identifiers, and constructs independent per-KOI global and local views. It does
not query live catalogue labels or parameters during processing.

PDCSAP_FLUX is the primary representation because Kepler's pipeline systematic
correction is appropriate for this transit-view pilot. SAP_FLUX is retained only
for diagnostic morphology comparison. Every valid quarter is independently
median-normalized; there is no learned detrending, sigma clipping, imputation
model, or cross-object normalization. Future population-level scaling or learned
detrending must be fit using training data only after dataset splitting.

Cadences with any bit in the explicit DR25 quality mask integer `1130799` are
removed. Long-cadence identity is established from the official `_llc.fits`
product type, Kepler identity/release/quarter headers, and a plausible measured
time spacing. The descriptive nominal exposure is approximately 1800 seconds;
exact equality is never required, and each product's measured cadence is saved.

Phase is `((time - koi_time0bk + period/2) mod period) / period - 1/2`, so transit
centre is phase zero. The global view has 2000 equal bins on `[-0.5, 0.5)`. The
local coordinate is phase times period divided by duration (with duration
converted from hours to days), and has 200 equal bins on `[-2.5, 2.5)`. Observed
bins contain medians; deterministic linear interpolation constructs finite array
values while observed masks and counts preserve missingness.

The temporal-baseline fraction is deterministic: the numerator is the maximum
minus minimum BKJD among all usable, finite, quality-masked PDCSAP cadences; the
denominator is the maximum minus minimum finite raw BKJD across all
identity-valid downloaded Q1–Q17 files for that target. A zero denominator gives
zero. Threshold failures are recorded as `qc_pass` and `qc_reasons`; constructible
objects are retained. Hard failures are restricted to invalid identity or
ephemeris, absent usable photometry, impossible folding, or unconstructible
outputs.

For diagnostic review, an object is separately flagged when more than 20% of
either view is interpolated. This review flag does not change `qc_pass` or remove
the object.

Labels and dispositions affect pilot stratification only. They are not inputs to
masking, normalization, folding, binning, or diagnostic calculations.
