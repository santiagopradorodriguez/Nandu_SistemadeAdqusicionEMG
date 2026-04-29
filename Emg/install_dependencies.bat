@echo off
echo ====================================================================
echo  Configurador del Entorno Virtual para el Sistema de Adquisicion EMG
echo ====================================================================

REM Comprobar si Python está instalado
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python no esta instalado o no se encuentra en el PATH.
    echo  Por favor, instala Python 3.8 o superior y asegurate de que este
    echo  agregado al PATH del sistema.
    echo.
    pause
    exit /b 1
)

REM Comprobar si el entorno virtual ya existe
if exist "venv" (
    echo.
    echo  El entorno virtual 'venv' ya existe.
) else (
    echo.
    echo  Creando entorno virtual 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo.
        echo  ERROR: No se pudo crear el entorno virtual.
        echo.
        pause
        exit /b 1
    )
    echo  Entorno virtual creado exitosamente.
)

echo.
echo  Activando el entorno virtual...
call "venv\Scripts\activate.bat"

echo.
echo  Instalando dependencias desde requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Hubo un problema al instalar las dependencias.
    echo  Revisa el archivo requirements.txt y tu conexion a internet.
    echo.
    pause
    exit /b 1
)

echo.
echo ===============================================================
echo   Instalacion completada!
echo ===============================================================
echo.
echo  Para empezar a usar los programas, activa el entorno virtual
echo  ejecutando el siguiente comando en tu terminal:
echo.
echo    .\venv\Scripts\activate
echo.
pause
