Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT="https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
#$Env:PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
#$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = 1
$Env:UV_NO_CACHE = 0
$Env:UV_LINK_MODE = "symlink"
$Env:GIT_LFS_SKIP_SMUDGE = 1
$Env:CUDA_HOME = "${env:CUDA_PATH}"

function InstallFail {
    Write-Output "Install failed|安装失败。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

try {
    ~/.local/bin/uv --version
    Write-Output "uv installed|UV模块已安装."
}
catch {
    Write-Output "Installing uv|安装uv模块中..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        Check "uv install failed|安装uv模块失败。"
    }
    else {
        curl -LsSf https://astral.sh/uv/install.sh | sh
        Check "uv install failed|安装uv模块失败。"
    }
}

if ($env:OS -ilike "*windows*") {
    chcp 65001
    # First check if UV cache directory already exists
    if (Test-Path -Path "${env:LOCALAPPDATA}/uv/cache") {
        Write-Host "UV cache directory already exists, skipping disk space check"
    }
    else {
        # Check C drive free space with error handling
        try {
            $CDrive = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction Stop
            if ($CDrive) {
                $FreeSpaceGB = [math]::Round($CDrive.FreeSpace / 1GB, 2)
                Write-Host "C: drive free space: ${FreeSpaceGB}GB"
                
                # $Env:UV cache directory based on available space
                if ($FreeSpaceGB -lt 10) {
                    Write-Host "Low disk space detected. Using local .cache directory"
                    $Env:UV_CACHE_DIR = ".cache"
                } 
            }
            else {
                Write-Warning "C: drive not found. Using local .cache directory"
                $Env:UV_CACHE_DIR = ".cache"
            }
        }
        catch {
            Write-Warning "Failed to check disk space: $_. Using local .cache directory"
            $Env:UV_CACHE_DIR = ".cache"
        }
    }
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Windows venv"
        . ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Windows .venv"
        . ./.venv/Scripts/activate
    }
    else {
        Write-Output "Create .venv"
        ~/.local/bin/uv venv -p 3.10
        . ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Linux venv"
    . ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Linux .venv"
    . ./.venv/bin/activate.ps1
}
else {
    Write-Output "Create .venv"
    ~/.local/bin/uv venv -p 3.10
    . ./.venv/bin/activate.ps1
}

Write-Output "Installing main requirements"

~/.local/bin/uv pip install --upgrade setuptools wheel

~/.local/bin/uv pip sync requirements-uv.txt --index-strategy unsafe-best-match
Check "Install main requirements failed"

try {
    espeak-ng --version
    Write-Output "espeak-ng installed|espeak-ng模块已安装."
}
catch {
    try {
        # 设置变量
        $githubURL = "https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi"  # 替换为你的 GitHub MSI 文件的 URL
        $downloadPath = "$env:USERPROFILE\Downloads\espeak-ng.msi"  # 替换为你想要保存 MSI 文件的路径（这里使用用户的“下载”文件夹）
        $logFile = "./install.log"
        Write-Output "Downlaoding espeak-ng|下载espeak-ng模块中..."
        Invoke-WebRequest -Uri $githubURL -OutFile $downloadPath -UseBasicParsing
        Write-Host "Downlaod finished|下载完成！" -ForegroundColor Green
        Write-Host "Installing espeak-ng msi|安装espeak-ng模块中..." -ForegroundColor Green
        # 使用 Start-Process 安装 MSI 文件（静默安装，记录日志）
        Start-Process -FilePath "msiexec.exe" -ArgumentList "/i `"$downloadPath`" /qn /l*v `"$logFile`"" -Wait -PassThru -Verb RunAs
        try {
            espeak-ng --version
            Write-Output "espeak-ng Install finished| espeak-ng安装成功" -ForegroundColor Green
        }
        catch {
            Write-Output "espeak-ng install failed, check log $logFile|安装espeak-ng模块失败。请查看日志文件:$logFile"  -ForegroundColor Red
        }
    }
    catch {
        Write-Error "Doownload error:$($_.Exception.Message)"
        if (Test-Path $downloadPath) {
            Remove-Item $downloadPath -Force # 如果下载失败，删除未完成的下载文件
            Write-Host "Delete download file" -ForegroundColor Yellow
        }
    }
}

Write-Output "Install finished"
Read-Host | Out-Null ;
