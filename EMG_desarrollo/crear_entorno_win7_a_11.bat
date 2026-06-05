@echo off
echo ==============================================================================
echo  CREADOR DE ENTORNO VIRTUAL COMPATIBLE (WIN 7 - WIN 11)
echo  Proyecto: NANDU LSD - Sistema de Adquisicion EMG
echo ==============================================================================
echo.
echo NOTA: Para que el ejecutable resultante funcione en Windows 7, DEBES
echo ejecutar este script utilizando Python 3.8. Python 3.9 o superior
echo generara un ejecutable que crasheara en Windows 7 con errores de DLL.
echo.
echo Buscando Python 3.8 en el sistema...

:: Intentar encontrar python 3.8 a traves de py launcher
py -3.8 -V >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Python 3.8 detectado via 'py -3.8'. Creando entorno...
    py -3.8 -m venv venv_win_legacy
) ELSE (
    echo [ADVERTENCIA] No se detecto 'py -3.8'.
    echo Se intentara usar la version por defecto de 'python'.
    python -V
    echo Si ves una version mayor a 3.8, el ejecutable NO funcionara en Windows 7.
    pause
    python -m venv venv_win_legacy
)

echo.
echo Activando entorno virtual e instalando dependencias (puede tardar varios minutos)...
call venv_win_legacy\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ==============================================================================
echo ¡ENTORNO VIRTUAL LISTO!
echo ==============================================================================
echo Para activar este entorno en el futuro, ejecuta:
echo     venv_win_legacy\Scripts\activate
echo.
echo Para generar el ejecutable compatible, ejecuta (con el entorno activado):
echo     python herramientas_build\crear_spec_ejecutable.py
echo     pyinstaller EMG_Studio.spec
echo.
pause
