$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUTF8 = '1'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPath = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$vendorPath = Join-Path $projectRoot 'runtime\vendor\Real-ESRGAN'

function Find-Python {
    foreach ($candidate in @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
    )) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($version -eq '3.12') { return $candidate }
        }
    }
    throw 'Python 3.12 x64 is required.'
}

$python = Find-Python
if (-not (Test-Path -LiteralPath $venvPython)) { & $python -m venv $venvPath }
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
& $venvPython -m pip install --editable $projectRoot

if (-not (Test-Path -LiteralPath (Join-Path $vendorPath '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $vendorPath) | Out-Null
    & git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git $vendorPath
}
& git -C $vendorPath fetch --depth 1 origin a4abfb2979a7bbff3f69f58f58ae324608821e27
& git -C $vendorPath checkout --detach a4abfb2979a7bbff3f69f58f58ae324608821e27
& $venvPython -m pip install --editable $vendorPath --no-deps

# BasicSR 1.4.2 imports a torchvision module moved in newer torchvision.
$basicSrPath = Join-Path $venvPath 'Lib\site-packages\basicsr'
if (Test-Path -LiteralPath $basicSrPath) {
    Get-ChildItem -LiteralPath $basicSrPath -Recurse -Filter '*.py' | ForEach-Object {
        $before = Get-Content -LiteralPath $_.FullName -Raw
        if ($null -ne $before) {
            $after = $before.Replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale', 'from torchvision.transforms.functional import rgb_to_grayscale')
            if ($after -ne $before) { Set-Content -LiteralPath $_.FullName -Value $after -Encoding utf8 }
        }
    }
}

& $venvPython (Join-Path $PSScriptRoot 'download_models.py') $projectRoot
& $venvPython -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
# Keep the published dependency snapshot stable.  The installer must not
# overwrite tracked requirements.lock with machine-specific pip-freeze output.
Write-Host 'Portrait Local installation completed.' -ForegroundColor Green
