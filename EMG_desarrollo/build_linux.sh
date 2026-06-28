#!/bin/bash
# Cambiar al directorio donde está el script para que las rutas relativas funcionen
cd "$(dirname "$0")" || exit

echo "===================================================="
echo "Construyendo Ejecutable para EMG Studio (v4.x) - LINUX"
echo "===================================================="

# 1. Crear el entorno base
echo "[1/4] Creando entorno de compilacion..."
../venv/bin/python herramientas_build/crear_entorno_ejecutable.py

# 2. Aplicar los parches para PyInstaller
echo "[2/4] Aplicando parches..."
../venv/bin/python herramientas_build/aplicar_parches_ejecutable.py

# 3. Generar el SPEC
echo "[3/4] Generando SPEC..."
../venv/bin/python herramientas_build/crear_spec_ejecutable.py

# 4. Compilar con PyInstaller
echo "===================================================="
echo "Iniciando PyInstaller..."
cd EMG_Ejecutable_Build || exit
../../venv/bin/pyinstaller EMG_Studio.spec --noconfirm --clean
cd ..

# 5. Renombrar carpeta para el usuario
echo "Renombrando directorio de distribucion..."
if [ -d "EMG_Ejecutable_Build/dist/NanduLsd" ]; then
    rm -rf "EMG_Ejecutable_Build/dist/NanduLsd"
fi

if [ -d "EMG_Ejecutable_Build/dist/NanduLsd_Core" ]; then
    mv "EMG_Ejecutable_Build/dist/NanduLsd_Core" "EMG_Ejecutable_Build/dist/NanduLsd"
fi

# 6. Crear un script lanzador bash
echo "Creando lanzador bash..."
cat << 'EOF' > EMG_Ejecutable_Build/dist/NanduLsd/run_nandu.sh
#!/bin/bash
# Lanzador de Nandu LSD para Linux
cd "$(dirname "$0")"
./NanduLsd_Core
EOF
chmod +x EMG_Ejecutable_Build/dist/NanduLsd/run_nandu.sh

echo "===================================================="
echo "BUILD COMPLETADO. El ejecutable principal esta en EMG_Ejecutable_Build/dist/NanduLsd/run_nandu.sh"
