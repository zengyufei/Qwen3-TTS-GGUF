@echo off
chcp 65001 >nul
echo [2/3] 正在安装项目依赖...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败，请检查网络连接。
    pause
    exit /b
)
echo [成功] 所有依赖已安装完毕。
pause