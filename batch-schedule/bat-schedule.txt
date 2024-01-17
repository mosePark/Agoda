@echo off
cd C:\Users\UOS\proj_0\selenium

FOR /L %%i IN (1,1,45) DO (
    set BATCH_NUMBER=%%i
    python c:/Users/UOS/proj_0/selenium/batch-process.py %%i
    timeout /t 60
)

echo All batches completed.
pause
