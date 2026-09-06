# Stage 4A full-cohort data pipeline

Stage 4A validates the frozen Stage 1 cohort, Stage 2 preprocessing configuration,
and immutable Stage 3 assignments. Hosts are sorted by `kepid` and placed into
label-blind batches. Discovery is performed once per unique host and checkpointed
atomically outside Git. A partial discovery is never frozen as the final inventory.

Discovery and download are separate commands. Preprocessing reads only a frozen
inventory and never queries MAST. Each host's FITS products are reused for every
KOI on that host. Phase folding and view construction remain KOI-specific.

The canonical representation is one deterministic NPZ per KOI. Stage 2's writer
fixes member order, ZIP timestamps and compression metadata; Stage 4A additionally
requires explicitly little-endian inputs and records logical-array and whole-file
hashes. No arrays are duplicated for Stage 3 strategies or repeats.

`laptop_safe` requires free space only for the largest active host, processed
outputs, and its safety margin. `cloud_archive` assesses all remaining retained
raw products. Neither profile uses cloud-sync state as validity or provenance.

The separate `public_wifi` operational profile schedules one host at a time with
one CPU worker and one download worker. Its default run ceiling is 100 MiB. A
host is never started when its complete advertised product size would cross the
ceiling, and resume starts at the first host without a final verified processing,
array-checksum, and provenance state. Temporary FITS use the scratch root and are
never copied automatically into cloud-backed data storage. Purging remains
subject to every existing host safety gate.
