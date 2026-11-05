@echo off
py -3.13 "%~dp0shared\code\run_nearby_pairs.py" run --workers 12 %*
exit /b %errorlevel%
