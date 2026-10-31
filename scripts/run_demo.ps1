# Start the simulator (native viewer), backend and dashboard in separate windows, then open the dashboard.
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -NoViewer
param([switch]$NoViewer)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "No .venv found. Follow the setup steps in README.md first."; exit 1 }

$simArgs = if ($NoViewer) { "-m saltsim.run --no-viewer" } else { "-m saltsim.run" }
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$py' $simArgs"
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$py' -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -WorkingDirectory (Join-Path $root "frontend") -ArgumentList "-NoExit", "-Command", "npm run dev"

Write-Host "Waiting for the dashboard..."
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173 -TimeoutSec 1 | Out-Null; break } catch { Start-Sleep 1 }
}
Start-Process "http://127.0.0.1:5173"
Write-Host "Dashboard: http://127.0.0.1:5173  (close the three windows to stop)"
