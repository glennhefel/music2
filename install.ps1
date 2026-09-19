$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Environment = Join-Path $Root ".venv-tf"
$Python = Join-Path $Environment "Scripts\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/"
}

if (-not (Test-Path $Python)) {
    uv venv --python 3.11 $Environment
}

uv pip install --python $Python -r requirements-gpu-cu128.txt
uv pip install --reinstall --python $Python torch --index-url https://download.pytorch.org/whl/cu128

& $Python -c "import torch, tensorflow, librosa; print('torch:', torch.__version__); print('tensorflow:', tensorflow.__version__); print('librosa:', librosa.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
if ($LASTEXITCODE -ne 0) {
    throw "Dependency verification failed."
}

Write-Host "Environment ready: $Python"
