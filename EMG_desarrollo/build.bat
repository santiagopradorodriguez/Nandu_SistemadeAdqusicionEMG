@echo off
echo ====================================================
echo Construyendo Ejecutable para EMG Studio (v4.x)
echo ====================================================

:: 1. Crear el entorno base
echo [1/4] Creando entorno de compilacion...
python herramientas_build/crear_entorno_ejecutable.py

:: 2. Aplicar los parches para PyInstaller
echo [2/4] Aplicando parches...
python herramientas_build/aplicar_parches_ejecutable.py

:: 3. Generar el SPEC
echo [3/4] Generando SPEC...
python herramientas_build/crear_spec_ejecutable.py

:: 4. Compilar con PyInstaller
echo ====================================================
echo Iniciando PyInstaller...
cd EMG_Ejecutable_Build
pyinstaller EMG_Studio.spec --noconfirm --clean
cd ..

:: 5. Compilar el Launcher en C#
echo ====================================================
echo Compilando Launcher C# nativo...
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /nologo /target:winexe /out:EMG_Ejecutable_Build\dist\NanduLsd_Core\NanduLsd.exe /win32icon:icono.ico herramientas_build\launcher.cs

:: 6. Renombrar carpeta para el usuario
echo Renombrando directorio de distribucion...
if exist EMG_Ejecutable_Build\dist\NanduLsd rmdir /s /q EMG_Ejecutable_Build\dist\NanduLsd
rename EMG_Ejecutable_Build\dist\NanduLsd_Core NanduLsd

echo ====================================================
echo BUILD COMPLETADO. El ejecutable principal esta en EMG_Ejecutable_Build\dist\NanduLsd\NanduLsd.exe
