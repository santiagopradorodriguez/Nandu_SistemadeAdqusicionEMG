@echo off
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%EMG_desarrollo"

echo ====================================================
echo NANDU LSD - Compilacion Multiplataforma (WINDOWS)
echo ====================================================

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python no se encuentra en el PATH.
    exit /b 1
)

echo [1/4] Creando entorno de compilacion temporal...
python herramientas_build\crear_entorno_ejecutable.py

echo [2/4] Aplicando parches de compatibilidad...
python herramientas_build\aplicar_parches_ejecutable.py

echo [3/4] Generando archivo .spec...
python herramientas_build\crear_spec_ejecutable.py

echo [4/4] Ejecutando PyInstaller...
cd EMG_Ejecutable_Build
pyinstaller EMG_Studio.spec --noconfirm --clean
cd ..

set CSC_PATH=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC_PATH%" set CSC_PATH=C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe
if exist "%CSC_PATH%" (
    echo Compilando Launcher C# nativo...
    "%CSC_PATH%" /nologo /target:winexe /out:EMG_Ejecutable_Build\dist\NanduLsd_Core\NanduLsd.exe /win32icon:icono.ico herramientas_build\launcher.cs
) else (
    echo [Aviso] csc.exe no encontrado. Copiando ejecutable principal...
    copy /Y "EMG_Ejecutable_Build\dist\NanduLsd_Core\NanduLsd_Core.exe" "EMG_Ejecutable_Build\dist\NanduLsd_Core\NanduLsd.exe"
)

echo Finalizando estructura de distribucion...
if exist EMG_Ejecutable_Build\dist\NanduLsd rmdir /s /q EMG_Ejecutable_Build\dist\NanduLsd
if exist EMG_Ejecutable_Build\dist\NanduLsd_Core rename EMG_Ejecutable_Build\dist\NanduLsd_Core NanduLsd

if not exist "%ROOT_DIR%build_windows" mkdir "%ROOT_DIR%build_windows"
if exist "%ROOT_DIR%build_windows\NanduLsd" rmdir /s /q "%ROOT_DIR%build_windows\NanduLsd"
xcopy /E /I /Y "EMG_Ejecutable_Build\dist\NanduLsd" "%ROOT_DIR%build_windows\NanduLsd"

echo ====================================================
echo BUILD COMPLETADO EXITOSAMENTE.
echo Ejecutable listo en: %ROOT_DIR%build_windows\NanduLsd\NanduLsd.exe
echo ====================================================
