@echo off
REM ==========================================================================================
REM  Mode UJI COBA lokal: server di http://127.0.0.1:8000, rangkuman memakai DeepSeek API (cloud).
REM  Whisper, ML-1, ML-2, PII scrub tetap berjalan lokal. Hanya teks transkrip (sudah di-scrub) yang
REM  dikirim ke DeepSeek -> sistem TIDAK offline. Pakai data fiktif saja.
REM  Butuh APP_DEEPSEEK_API_KEY di .env. Setting .env lain (Ollama untuk lab) tidak diubah.
REM ==========================================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Folder .venv belum ada. Jalankan: powershell -ExecutionPolicy Bypass -File scripts\windows\setup_lab.ps1 -SkipOllama
  pause >nul
  exit /b 1
)
findstr /b /c:"APP_DEEPSEEK_API_KEY=sk-" .env >nul 2>nul
if errorlevel 1 (
  echo [ERROR] APP_DEEPSEEK_API_KEY belum ada di .env. Tambahkan baris: APP_DEEPSEEK_API_KEY=sk-...
  pause >nul
  exit /b 1
)

REM Variabel proses menimpa .env (app/config.py tidak menimpa env yang sudah di-set).
set "APP_LLM_BACKEND=deepseek"

echo [OSCE-STT] Mode uji coba DeepSeek: http://127.0.0.1:8000
echo            Browser akan terbuka otomatis. Tutup jendela ini untuk mematikan server.
start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{Invoke-RestMethod http://127.0.0.1:8000/api/health -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:8000'; break}catch{Start-Sleep 2}}"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo Server berhenti. Tekan tombol apa saja untuk menutup jendela ini.
pause >nul
endlocal
