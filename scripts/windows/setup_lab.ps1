# ==========================================================================================
#  Instalasi OSCE-STT di komputer server (Windows + GPU NVIDIA). Aman dijalankan ulang.
#
#  Jalankan dari folder repo:
#    powershell -ExecutionPolicy Bypass -File scripts\windows\setup_lab.ps1
#  Opsi:
#    -Cpu          tanpa GPU (Whisper di CPU int8; jauh lebih lambat)
#    -SkipOllama   lewati unduhan model LLM
#    -SkipTests    lewati pytest
#  Penjelasan tiap langkah: docs\INSTALASI_LAB.md
# ==========================================================================================
param([switch]$Cpu, [switch]$SkipOllama, [switch]$SkipTests)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$Py = ".\.venv\Scripts\python.exe"

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "[GAGAL] $msg" -ForegroundColor Red; exit 1 }
function Run($exe, [string[]]$argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { Fail "$exe $($argList -join ' ') (exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------- 1. Python
Step "1/8 Cek Python 3.10-3.12"
$base = $null
foreach ($cand in @(@("py", "-3.11"), @("py", "-3.12"), @("py", "-3.10"), @("python"))) {
    try {
        $v = & $cand[0] @($cand[1..9] + @("-c", "import sys;print('%d.%d' % sys.version_info[:2])")) 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(10|11|12)$') { $base = $cand; break }
    } catch { }
}
if (-not $base) { Fail "Python 3.10-3.12 tidak ditemukan. Pasang: winget install --id Python.Python.3.11 -e" }
Write-Host "Python $v ($($base -join ' '))"

# ---------------------------------------------------------------- 2. venv
Step "2/8 Virtual environment .venv"
if (-not (Test-Path $Py)) { Run $base[0] (@($base[1..9]) + @("-m", "venv", ".venv")) }
Run $Py @("-m", "pip", "install", "--upgrade", "pip", "--quiet")

# ---------------------------------------------------------------- 3. paket Python
Step "3/8 Paket Python (requirements.txt + requirements-optional.txt)"
Run $Py @("-m", "pip", "install", "-r", "requirements.txt", "-r", "requirements-optional.txt")

# ---------------------------------------------------------------- 4. CUDA libs
$useGpu = (-not $Cpu) -and [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($useGpu) {
    Step "4/8 cuBLAS + cuDNN untuk Whisper GPU (pip, +-2 GB)"
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    Run $Py @("-m", "pip", "install", "nvidia-cublas-cu12>=12,<13", "nvidia-cudnn-cu12>=9,<10")
} else {
    Step "4/8 Mode CPU (GPU NVIDIA tidak dipakai)"
}

# ---------------------------------------------------------------- 5. .env
Step "5/8 File .env + token akses"
if (-not (Test-Path ".env")) {
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = -join ($bytes | ForEach-Object { $_.ToString("x2") })
    $envText = (Get-Content ".env.example") -replace '^APP_ACCESS_TOKEN=.*$', "APP_ACCESS_TOKEN=$token"
    if (-not $useGpu) {
        $envText = $envText -replace '^APP_WHISPER_DEVICE=.*$', 'APP_WHISPER_DEVICE=cpu' `
                            -replace '^APP_WHISPER_COMPUTE_TYPE=.*$', 'APP_WHISPER_COMPUTE_TYPE=int8'
    }
    # ASCII tanpa BOM: parser .env sederhana akan salah membaca BOM di baris pertama
    Set-Content -Path ".env" -Value $envText -Encoding ASCII
    Write-Host ".env dibuat. Token akses: $token" -ForegroundColor Yellow
} else {
    Write-Host ".env sudah ada, tidak diubah."
}

# ---------------------------------------------------------------- 6. Ollama
Step "6/8 Model LLM Ollama (qwen2.5:3b, +-1.9 GB)"
if ($SkipOllama) {
    Write-Host "Dilewati (-SkipOllama)."
} elseif (Get-Command ollama -ErrorAction SilentlyContinue) {
    Run "ollama" @("pull", "qwen2.5:3b")
} else {
    Write-Host "[PERINGATAN] Ollama belum terpasang: winget install --id Ollama.Ollama -e" -ForegroundColor Yellow
    Write-Host "Tanpa Ollama, rangkuman memakai fallback rule-based." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 7. Whisper + model ML
Step "7/8 Unduh Whisper small + uji muat di perangkat target, lalu latih model intent"
Run $Py @("-c", "from app.config import get_settings; from app.core.services import _model_factories; m=_model_factories(get_settings())['asr'](); print('Whisper OK:', m.model_name); m.close()")
Run $Py @("scripts/train_text_intent.py", "--data", "data/synthetic/text_intent.jsonl", "data/synthetic/text_intent_llm.jsonl", "--out", "models/text_intent")

# ---------------------------------------------------------------- 8. test
if ($SkipTests) {
    Step "8/8 pytest dilewati (-SkipTests)"
} else {
    Step "8/8 pytest"
    Run $Py @("-m", "pytest", "-q")
}

Write-Host ""
Write-Host "SELESAI. Langkah berikutnya:" -ForegroundColor Green
Write-Host "  - Akses publik (Cloudflare): klik dua kali start_osce.bat"
Write-Host "  - Hanya lokal/LAN          : start_server.bat  atau  start_server.bat lan"
Write-Host "  - Token akses ada di .env (APP_ACCESS_TOKEN). Panduan: docs\INSTALASI_LAB.md"
