@echo off
:: 设置控制台编码为 UTF-8，防止中文输入乱码
chcp 65001 >nul

:: 提示输入提交备注
set /p commit_msg=请输入提交备注（按 Enter 提交）:

:: 添加所有更改
git add .

:: 提交更改
git commit -m "%commit_msg%"

:: 推送到 GitHub
git push origin main

pause