# Build base images for ReazonSpeech Docker (Windows PowerShell)
# Usage: .\scripts\build-base-images.ps1 [-Target gpu|cpu|all]

param(
    [ValidateSet("gpu", "cpu", "all")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Enable BuildKit
$env:DOCKER_BUILDKIT = "1"

function Build-GPU {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Building GPU base images..." -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    Write-Host "[1/3] Building common-base:gpu..." -ForegroundColor Yellow
    docker build `
        -f "$ProjectRoot/services/base/Dockerfile.common-gpu" `
        -t reazonspeech-common-base:gpu `
        "$ProjectRoot"
    if ($LASTEXITCODE -ne 0) { throw "Failed to build common-base:gpu" }

    Write-Host "[2/3] Building k2-v2-base:gpu..." -ForegroundColor Yellow
    docker build `
        -f "$ProjectRoot/services/base/Dockerfile.k2-v2-gpu" `
        -t reazonspeech-k2-v2-base:gpu `
        "$ProjectRoot"
    if ($LASTEXITCODE -ne 0) { throw "Failed to build k2-v2-base:gpu" }

    Write-Host "[3/3] Building espnet-v2-base:gpu..." -ForegroundColor Yellow
    docker build `
        -f "$ProjectRoot/services/base/Dockerfile.espnet-v2-gpu" `
        -t reazonspeech-espnet-v2-base:gpu `
        "$ProjectRoot"
    if ($LASTEXITCODE -ne 0) { throw "Failed to build espnet-v2-base:gpu" }

    Write-Host "GPU base images built successfully!" -ForegroundColor Green
}

function Build-CPU {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Building CPU base images..." -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    Write-Host "[1/3] Building common-base:cpu..." -ForegroundColor Yellow
    docker build `
        -f "$ProjectRoot/services/base/Dockerfile.common-cpu" `
        -t reazonspeech-common-base:cpu `
        "$ProjectRoot"
    if ($LASTEXITCODE -ne 0) { throw "Failed to build common-base:cpu" }

    Write-Host "[2/3] Building k2-v2-base:cpu..." -ForegroundColor Yellow
    docker build `
        -f "$ProjectRoot/services/base/Dockerfile.k2-v2-cpu" `
        -t reazonspeech-k2-v2-base:cpu `
        "$ProjectRoot"
    if ($LASTEXITCODE -ne 0) { throw "Failed to build k2-v2-base:cpu" }

    Write-Host "[3/3] Building espnet-v2-base:cpu..." -ForegroundColor Yellow
    docker build `
        -f "$ProjectRoot/services/base/Dockerfile.espnet-v2-cpu" `
        -t reazonspeech-espnet-v2-base:cpu `
        "$ProjectRoot"
    if ($LASTEXITCODE -ne 0) { throw "Failed to build espnet-v2-base:cpu" }

    Write-Host "CPU base images built successfully!" -ForegroundColor Green
}

switch ($Target) {
    "gpu" { Build-GPU }
    "cpu" { Build-CPU }
    "all" { Build-GPU; Build-CPU }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Base images ready. You can now run:" -ForegroundColor Cyan
Write-Host "  docker compose build" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
