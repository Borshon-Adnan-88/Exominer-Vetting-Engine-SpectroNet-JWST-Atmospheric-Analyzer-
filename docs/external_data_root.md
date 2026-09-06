# External data root

Stage 4A has one implementation. Storage is selected at runtime by `--data-root`
or `EXOMINER_DATA_ROOT`; Google Drive and OneDrive are ordinary filesystems, not
scientific dependencies. Deterministic artifacts contain paths relative to the
data root only.

Windows with a Google Drive-synced directory:
`python -m scripts.stage4a_preflight --data-root "G:\My Drive\exominer-data" --profile cloud_archive`

Windows with an ordinary disk:
`python -m scripts.stage4a_preflight --data-root "D:\exominer-data" --profile laptop_safe`

Linux or HPC:
`EXOMINER_DATA_ROOT=/scratch/$USER/exominer-data python -m scripts.stage4a_preflight --profile cloud_archive`

Google Colab:
`EXOMINER_DATA_ROOT=/content/drive/MyDrive/exominer-data python -m scripts.stage4a_preflight --profile cloud_archive`

These examples invoke identical code and scientific configuration. Absolute
paths are runtime inputs and never enter committed scientific artifacts.

For temporary downloads, `--scratch-root` takes precedence over
`EXOMINER_SCRATCH_ROOT`; if neither is supplied, Stage 4A uses the existing data
root's `temporary/` area. Scratch location and the `public_wifi` transfer ceiling
are operational only and never enter deterministic artifacts.

Example public-Wi-Fi transfer plan (no download):
`python -m scripts.stage4a_plan_transfer --data-root "G:\My Drive\exominer-data" --scratch-root "C:\exominer-scratch" --profile public_wifi`
