# Waits for the main API job to finish, re-cleans everything, then fetches missing-answer files.
# Safe to Ctrl+C at any point; re-run this script to continue.
$log = Join-Path $PSScriptRoot "data\logs\api.log"
Write-Host "Waiting for the API job to finish (checks every 60 s)..."
while (-not (Select-String -Path $log -Pattern "FINISHED state=complete" -Quiet)) { Start-Sleep -Seconds 60 }
Write-Host "API job complete. Re-cleaning all data..."
python -I -X utf8 (Join-Path $PSScriptRoot "pipeline\clean.py")
Write-Host "Fetching fallback files for answers with no text..."
python -I -X utf8 (Join-Path $PSScriptRoot "fetch_fallback.py")
Write-Host "Done. Re-run pipeline\clean.py once the fallback files are parsed by the cleaner."
