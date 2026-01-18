# Installation and Setup Script for Windows PowerShell
# Run this script to set up the entire project

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Regime-Switching Options Project Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check Python installation
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  $pythonVersion found" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Please install Python 3.8+`n" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "  Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n[3/6] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "  Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host "`n[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "  pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host "`n[5/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
pip install -r requirements.txt --quiet
Write-Host "  Dependencies installed" -ForegroundColor Green

# Install project
Write-Host "`n[6/6] Installing project in development mode..." -ForegroundColor Yellow
pip install -e . --quiet
Write-Host "  Project installed" -ForegroundColor Green

# Verify installation
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Verifying Installation" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

try {
    python -c "import numpy; import scipy; import matplotlib; print('  Core packages: OK')"
    python -c "from src.models.regime_switching import RegimeSwitchingModel; print('  Project modules: OK')"
    Write-Host ""
    Write-Host "  Installation successful!" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Some packages may not be installed correctly" -ForegroundColor Yellow
}

# Display next steps
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Next Steps" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Run example scripts:" -ForegroundColor White
Write-Host "  python examples/basic_pricing.py" -ForegroundColor Gray
Write-Host "  python examples/advanced_hedging.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Run comprehensive analysis:" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Launch Jupyter notebooks:" -ForegroundColor White
Write-Host "  jupyter notebook notebooks/" -ForegroundColor Gray
Write-Host ""
Write-Host "Run tests:" -ForegroundColor White
Write-Host "  pytest tests/ -v" -ForegroundColor Gray
Write-Host ""
Write-Host "See QUICKSTART.md for more information`n" -ForegroundColor White

Write-Host "========================================`n" -ForegroundColor Cyan
