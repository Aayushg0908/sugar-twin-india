# Start the Sugar Twin dashboard on Windows: API (port 8000) + web app (port 5173), each in its own window.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File start_dashboard.ps1
# Stop: close the two windows (or press Ctrl+C in each).

$root = $PSScriptRoot
$frontend = Join-Path $root "dashboard\frontend"

if (-not (Test-Path (Join-Path $root "data\model\features.parquet"))) {
    Write-Host "Building the training table first (data/model/features.parquet, 2-5 min)..." -ForegroundColor Yellow
    python (Join-Path $root "src\build_features.py")
}
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing web app packages (first run only)..." -ForegroundColor Yellow
    Push-Location $frontend; npm install; Pop-Location
}

Write-Host "Starting API on http://localhost:8000 (takes about 20 s to load)..." -ForegroundColor Cyan
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command",
    "python -m uvicorn app:app --app-dir dashboard/backend --port 8000"

Write-Host "Starting web app on http://localhost:5173 ..." -ForegroundColor Cyan
Start-Process powershell -WorkingDirectory $frontend -ArgumentList "-NoExit", "-Command", "npm run dev"

Start-Sleep -Seconds 6
Start-Process "http://localhost:5173"
Write-Host "Dashboard opening in your browser. It shows 'Starting Sugar Twin...' until the API is ready." -ForegroundColor Green
