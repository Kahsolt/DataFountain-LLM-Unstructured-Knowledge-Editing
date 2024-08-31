@ECHO OFF

PUSHD %~dp0

REM EasyEdit
git clone https://github.com/zjunlp/EasyEdit

REM UnKE
git clone https://github.com/TrustedLLM/UnKE

POPD

ECHO Done!
ECHO.

PAUSE
