@echo off
chcp 65001 > nul
echo ========================================================
echo   AI株式投資分析 Webアプリを起動しています...
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/2] Ollama サービスを確認中...
curl -s http://localhost:11434 > nul
if %errorlevel% neq 0 (
    echo [警告] Ollama が起動していません。バックグラウンドで起動を試みます...
    start /b ollama serve
    timeout /t 3 > nul
) else (
    echo [OK] Ollama は起動中です。
)

echo.
echo [2/2] Streamlit アプリを起動中...
python -m streamlit run app.py

pause
