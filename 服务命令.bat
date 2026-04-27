@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: 初始化上一次操作描述
set "last_action=无"
set "kill_exe_name=qwen.exe"

:menu
cls
echo.
call echo =============== 服务管理脚本 ===============
call echo 上一次执行的操作：%last_action%
echo.
call echo a) 状态
call echo b) 重启
call echo c) 停止
call echo d) 运行
call echo e) 重装
call echo f) 卸载
call echo g) 安装
call echo x) 退出
echo ===========================================
echo.

set /p "choice=请选择操作 (a/b/c/d/e/f/g/x): "

if /i "!choice!"=="x" (
    echo 脚本已退出。
    exit /b
)

:: 根据选择设置动作描述和执行命令（占位）
set "action_desc="
set "exec_cmd="
set "exec_cmd2=echo."
set "waite_time=1"

if /i "!choice!"=="a" (
    set "action_desc=查看状态"
    set "exec_cmd=service_tts.exe status"
    set "waite_time=5"
)
if /i "!choice!"=="b" (
    set "action_desc=重启服务"
    set "exec_cmd=taskkill /im %kill_exe_name% /f"
    set "exec_cmd2=service_tts.exe start"
)
if /i "!choice!"=="c" (
    set "action_desc=停止服务"
    set "exec_cmd=taskkill /im %kill_exe_name% /f"
)
if /i "!choice!"=="d" (
    set "action_desc=启动服务"
    set "exec_cmd=service_tts.exe start"
)
if /i "!choice!"=="e" (
    set "action_desc=重装服务"
    set "exec_cmd=service_tts.exe uninstall"
    set "exec_cmd2=service_tts.exe install"
)
if /i "!choice!"=="f" (
    set "action_desc=卸载服务"
    set "exec_cmd=service_tts.exe uninstall"
)
if /i "!choice!"=="g" (
    set "action_desc=安装服务"
    set "exec_cmd=service_tts.exe install"
)

:: 如果是无效选项
if "!action_desc!"=="" (
    set "last_action=无效选项：!choice!"
    timeout /t 1 /nobreak >nul
    goto menu
)

:: 更新上一次操作
set "last_action=!action_desc!"

:: 执行命令（你可在此处替换为真实命令，如 sc、npm、自定义脚本等）
echo.
echo 正在执行：!action_desc!
echo ----------------------------------------
%exec_cmd%
%exec_cmd2%
echo ----------------------------------------

:: 等待 1 秒后自动返回菜单
timeout /t %waite_time% /nobreak >nul

goto menu