# Setup and start script for the project (backend + frontend)
# Usage: powershell -ExecutionPolicy Bypass -File .\start.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host "=== Project setup ===" -ForegroundColor Cyan

# --- Backend: Python venv + dependencies ---
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvPath
}

Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

# --- Frontend: npm install ---
$ViewPath = Join-Path $ProjectRoot "view"
if (-not (Test-Path (Join-Path $ViewPath "node_modules"))) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location $ViewPath
    npm install
    Pop-Location
}

# --- Start frontend (in a new window, once backend is ready) ---
Write-Host "=== Starting backend and frontend ===" -ForegroundColor Cyan

$FrontendLauncher = {
    param($ViewPath)
    while (-not (Test-NetConnection -ComputerName 127.0.0.1 -Port 8000 -InformationLevel Quiet -WarningAction SilentlyContinue)) {
        Start-Sleep -Seconds 1
    }
    $FrontendCmd = "Set-Location '$ViewPath'; npm run dev"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCmd
}
Start-Job -ScriptBlock $FrontendLauncher -ArgumentList $ViewPath | Out-Null

Write-Host "Backend:  http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173 (will start once backend is ready)" -ForegroundColor Green
Write-Host ""

# --- Start backend in the current window (foreground) ---
Set-Location $ProjectRoot
& $VenvPython -m uvicorn controller.routes:app --host 127.0.0.1 --port 8000 --reload
