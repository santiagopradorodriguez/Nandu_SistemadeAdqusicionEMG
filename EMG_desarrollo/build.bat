@echo off
echo ====================================================
echo Construyendo Ejecutable para EMG Studio (v4.x)
echo ====================================================

:: 1. Crear el entorno base
echo [1/3] Creando entorno de compilacion...
python herramientas_build/crear_entorno_ejecutable.py

:: 2. Aplicar los parches para PyInstaller
echo [2/3] Aplicando parches...
python herramientas_build/aplicar_parches_ejecutable.py

:: 3. Generar el SPEC
echo [3/3] Generando SPEC...
python herramientas_build/crear_spec_ejecutable.py

:: 4. Compilar con PyInstaller
echo ====================================================
echo Iniciando PyInstaller...
cd EMG_Ejecutable_Build
pyinstaller EMG_Studio.spec --noconfirm
cd ..

echo ====================================================
echo BUILD COMPLETADO. El ejecutable esta en EMG_Ejecutable_Build/dist/EMG_Studio_App/
