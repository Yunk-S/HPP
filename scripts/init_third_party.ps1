# UTF-8 BOM: required for Windows PowerShell 5.1 to parse Chinese comments/strings correctly.
# Run from repo root: .\scripts\init_third_party.ps1
# Empty submodule dirs: run git submodule update --init first.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> git submodule update --init --recursive"
try {
    git submodule update --init --recursive third_party/torkit3d third_party/apex
} catch {
    try {
        git submodule update --init --recursive torkit3d apex
    } catch {
        git submodule update --init --recursive
    }
}

$TorkitPath = $null
if (Test-Path (Join-Path $Root "third_party/torkit3d/setup.py")) {
    $TorkitPath = Join-Path $Root "third_party/torkit3d"
} elseif (Test-Path (Join-Path $Root "torkit3d/setup.py")) {
    $TorkitPath = Join-Path $Root "torkit3d"
}

if ($TorkitPath) {
    Write-Host "==> pip install torkit3d from $TorkitPath"
    if (-not $env:FORCE_CUDA) {
        $env:FORCE_CUDA = "1"
    }
    pip install -v $TorkitPath
} else {
    Write-Warning "torkit3d/setup.py not found. Run: git submodule update --init --recursive"
}

$ApexPath = $null
if (Test-Path (Join-Path $Root "third_party/apex/setup.py")) {
    $ApexPath = Join-Path $Root "third_party/apex"
} elseif (Test-Path (Join-Path $Root "apex/setup.py")) {
    $ApexPath = Join-Path $Root "apex"
}

if ($ApexPath) {
    Write-Host "Optional: apex (FusedLayerNorm). This repo uses native LayerNorm; see torch_utils.replace_with_fused_layernorm."
    $r = Read-Host "Build and install apex? (y/N)"
    if ($r -eq "y" -or $r -eq "Y") {
        pip install -v --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" $ApexPath
    }
}

Write-Host "Done."
