param([int]$TrainingProcessId = 0)
$ErrorActionPreference = 'Stop'
if ($TrainingProcessId -gt 0) {
    Wait-Process -Id $TrainingProcessId
}
$manifest = Get-Content -LiteralPath 'data/stage5/run_manifest.json' -Raw | ConvertFrom-Json
if ($manifest.status -ne 'complete' -or $manifest.completed_fits -ne 20) {
    throw 'Stage 5 fitting did not finish successfully. Stop for review.'
}
& '.venv/stage5/Scripts/python.exe' -u -m scripts.stage5_run_cnn --replay
if ($LASTEXITCODE -ne 0) { throw 'Fresh-process replay failed. Stop for review.' }
& '.venv/stage5/Scripts/python.exe' -m pytest -m 'not network' --basetemp=data/tmp/stage5-full --junitxml=data/tmp/stage5-full.xml
if ($LASTEXITCODE -ne 0) { throw 'Offline tests failed. Stop for review.' }
& '.venv/stage5/Scripts/python.exe' -m scripts.stage5_report
if ($LASTEXITCODE -ne 0) { throw 'Final reporting failed. Stop for review.' }
Write-Output 'Stage 5 terminal sequence complete; ready for review.'
