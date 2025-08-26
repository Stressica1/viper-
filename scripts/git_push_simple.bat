@echo off
set GIT_PAGER=cat
git remote set-url origin https://github.com/Stressica1/viper-.git
git push -u origin main
echo.
echo Press any key to continue...
pause >nul
