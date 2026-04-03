@echo off
chcp 65001 >nul

echo 正在创建环境...
call python -m venv venv
echo 正在安装依赖...
call venv\Scripts\activate

REM set http_proxy=http://127.0.0.1:10808

REM set https_proxy=http://127.0.0.1:10808

pip install -r requirements.txt
echo 部署完成！
pause