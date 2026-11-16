@echo off
REM ==========================================================================================
REM  Satu klik: nyalakan server (jendela 1) + Cloudflare Quick Tunnel (jendela 2), lalu tampilkan
REM  URL publik https://....trycloudflare.com. URL juga disimpan ke runtime\tunnel_url.txt.
REM  URL BERUBAH setiap kali tunnel dinyalakan ulang. Detail: docs\INSTALASI_LAB.md
REM ==========================================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

where cloudflared >nul 2>nul
if errorlevel 1 (
  echo [ERROR] cloudflared tidak ditemukan. Pasang: winget install --id Cloudflare.cloudflared
  pause >nul
  exit /b 1
)
if not exist runtime mkdir runtime

echo [1/3] Menyalakan server di jendela baru...
start "OSCE-STT server" cmd /c start_server.bat

echo [2/3] Menunggu server siap (maks. 120 detik)...
set /a tries=0
:wait_server
curl.exe -s -o nul http://127.0.0.1:8000/api/health
if not errorlevel 1 goto :server_ok
set /a tries+=1
if !tries! geq 60 (
  echo [ERROR] Server tidak menjawab. Lihat pesan error di jendela "OSCE-STT server".
  pause >nul
  exit /b 1
)
timeout /t 2 /nobreak >nul
goto :wait_server
:server_ok
echo       Server siap.

echo [3/3] Menyalakan Cloudflare Quick Tunnel di jendela baru...
if exist runtime\tunnel.log del /q runtime\tunnel.log
start "OSCE-STT tunnel (JANGAN DITUTUP)" cloudflared tunnel --no-autoupdate --logfile runtime\tunnel.log --url http://127.0.0.1:8000

set "URL="
set /a tries=0
:wait_url
timeout /t 2 /nobreak >nul
if exist runtime\tunnel.log (
  for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "(Select-String -Path 'runtime\tunnel.log' -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1).Matches.Value"`) do set "URL=%%L"
)
if defined URL goto :url_ok
set /a tries+=1
if !tries! geq 30 (
  echo [ERROR] URL tunnel tidak muncul. Cek koneksi internet dan jendela "OSCE-STT tunnel".
  pause >nul
  exit /b 1
)
goto :wait_url

:url_ok
> runtime\tunnel_url.txt echo !URL!
echo.
echo ==========================================================================
echo   URL publik : !URL!
echo   Bagikan URL ini + token akses (APP_ACCESS_TOKEN di .env) ke anggota.
echo   Tersimpan  : runtime\tunnel_url.txt
echo   Untuk mematikan: tutup jendela "OSCE-STT tunnel" dan "OSCE-STT server".
echo ==========================================================================
echo.
start "" "!URL!"
pause
endlocal
