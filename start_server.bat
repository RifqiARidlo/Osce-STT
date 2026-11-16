@echo off
REM ==========================================================================================
REM  Menyalakan server OSCE-STT (satu worker uvicorn).
REM    start_server.bat          mode "tunnel": HTTP di 127.0.0.1:8000, dibuka publik oleh Cloudflare
REM    start_server.bat lan      mode "lan"   : HTTPS mkcert di 0.0.0.0:8000 untuk laptop satu jaringan
REM  Detail: docs\INSTALASI_LAB.md
REM ==========================================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :no_venv
if not exist ".env" goto :no_env

set "MODE=%~1"
if "%MODE%"=="" set "MODE=tunnel"
if /i "%MODE%"=="lan" goto :lan

echo [OSCE-STT] Mode tunnel: http://127.0.0.1:8000  (hanya lokal; akses publik lewat start_osce.bat)
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
goto :end

:lan
if not exist "key.pem" goto :no_cert
if not exist "cert.pem" goto :no_cert
echo [OSCE-STT] Mode LAN: https://<IP-komputer-ini>:8000
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
goto :end

:no_venv
echo [ERROR] Folder .venv belum ada. Jalankan instalasi dulu: scripts\windows\setup_lab.ps1
goto :end
:no_env
echo [ERROR] File .env belum ada. Jalankan scripts\windows\setup_lab.ps1 atau salin .env.example ke .env
goto :end
:no_cert
echo [ERROR] key.pem / cert.pem belum ada. Buat dengan mkcert, lihat README bagian 4.
goto :end

:end
echo.
echo Server berhenti. Tekan tombol apa saja untuk menutup jendela ini.
pause >nul
endlocal
