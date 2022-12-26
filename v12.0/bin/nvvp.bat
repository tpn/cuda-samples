@echo off 
setlocal
    setx PATH "%~dp0\..\extras\CUPTI\lib64;%PATH%"
    start "" "%~dp0\..\libnvvp\nvvp.exe" %*
endlocal
