# run script by @bdsqlsz, A-N-I-K

# Find where the latest MSVC is installed dynamically
$vswherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

if (Test-Path $vswherePath) {
    $vcvarsPath = & $vswherePath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vcvarsPath) {
        $vcvarsPath = Join-Path $vcvarsPath "VC\Auxiliary\Build\vcvars64.bat"
    }
} else {
    Write-Host "ERROR: vswhere.exe not found! Please ensure Visual Studio is installed."
    exit 1
}

# Ensure the path is valid
if (Test-Path $vcvarsPath) {
    Write-Host "Setting up MSVC environment..."
    cmd.exe /c "`"$vcvarsPath`" && set" | foreach {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Content -Path "env:\$($matches[1])" -Value "$($matches[2])"
        }
    }
} else {
    Write-Host "ERROR: vcvars64.bat could not be found! Please make sure MSVC is installed properly."
    exit 1
}

# Activate python venv
Set-Location $PSScriptRoot

if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = $PSScriptRoot + "\huggingface"
$Env:TORCH_HOME = $PSScriptRoot + "\torch"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:CUDA_HOME = "${env:CUDA_PATH}"
$Env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
$Env:GRADIO_HOST = "127.0.0.1"

python -m gradio_interface

Read-Host | Out-Null ;
