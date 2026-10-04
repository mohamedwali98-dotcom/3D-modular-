@echo off
rem Sketch-to-CAD: installs what is missing, builds the web app, then starts everything.
rem   Web app + API  -> http://localhost:8000
rem The server runs in its own window; close it (or press Ctrl+C in it) to stop it.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul || (echo uv is not installed: https://docs.astral.sh/uv/ & pause & exit /b 1)
where npm >nul 2>nul || (echo Node.js / npm is not installed: https://nodejs.org/ & pause & exit /b 1)

echo [1/3] Python packages...
call uv sync --extra ai --extra trocr || (echo uv sync failed. & pause & exit /b 1)

echo [2/3] Web app...
pushd web
if not exist node_modules (call npm install || (popd & echo npm install failed. & pause & exit /b 1))
call npm run build || (popd & echo The web build failed. & pause & exit /b 1)
popd

echo [3/3] Starting the server...
start "Sketch-to-CAD API (:8000)" cmd /k uv run uvicorn s2c.web.server:app --port 8000

rem give the server a moment (the models load at startup), then open the web app
timeout /t 15 /nobreak >nul
start "" http://localhost:8000

echo.
echo Web app: http://localhost:8000
echo From a phone on the same network, set S2C_ACCESS_TOKEN in .env first (see README).
endlocal
