@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set ROOT_DIR=%~dp0..\
cd /d "%SCRIPT_DIR%"

echo ====================================================
echo NANDU LSD - Compilacion Multiplataforma (WINDOWS)
echo ====================================================

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python no se encuentra en el PATH.
    exit /b 1
)

:: 1. Crear el entorno base
echo [1/4] Creando entorno de compilacion...
python herramientas_build\crear_entorno_ejecutable.py

:: 2. Aplicar los parches para PyInstaller
echo [2/4] Aplicando parches...
python herramientas_build\aplicar_parches_ejecutable.py

:: 3. Generar el SPEC
echo [3/4] Generando SPEC...
python herramientas_build\crear_spec_ejecutable.py

:: 4. Compilar con PyInstaller
echo ====================================================
echo Iniciando PyInstaller...
cd EMG_Ejecutable_Build
pyinstaller EMG_Studio.spec --noconfirm --clean
cd ..

:: 5. Compilar el Launcher en C#
set CSC_PATH=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if exist "%CSC_PATH%" (
    echo Compilando Launcher C# nativo...
    "%CSC_PATH%" /nologo /target:winexe /out:EMG_Ejecutable_Build\dist\NanduLsd_Core\NanduLsd.exe /win32icon:icono.ico herramientas_build\launcher.cs
)

:: 6. Renombrar carpeta para el usuario
echo Finalizando estructura de distribucion...
if exist EMG_Ejecutable_Build\dist\NanduLsd rmdir /s /q EMG_Ejecutable_Build\dist\NanduLsd
if exist EMG_Ejecutable_Build\dist\NanduLsd_Core rename EMG_Ejecutable_Build\dist\NanduLsd_Core NanduLsd

if not exist "%ROOT_DIR%build_windows" mkdir "%ROOT_DIR%build_windows"
if exist "%ROOT_DIR%build_windows\NanduLsd" rmdir /s /q "%ROOT_DIR%build_windows\NanduLsd"
xcopy /E /I /Y "EMG_Ejecutable_Build\dist\NanduLsd" "%ROOT_DIR%build_windows\NanduLsd"

echo ====================================================
echo BUILD COMPLETADO. El ejecutable principal esta en %ROOT_DIR%build_windows\NanduLsd\NanduLsd.exe
echo ====================================================
