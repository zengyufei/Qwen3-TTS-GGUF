@echo off
chcp 65001 >nul
echo [1/3] 正在创建 Python 虚拟环境 (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 并添加到环境变量。
    pause
    exit /b
)
echo [成功] 虚拟环境已创建在 venv 文件夹。
pause