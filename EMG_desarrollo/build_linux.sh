#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$(dirname "$0")"

echo "===================================================="
echo "NANDU LSD - Compilacion Multiplataforma (LINUX)"
echo "===================================================="

# Determinar interprete de Python y PyInstaller
if [ -f "$ROOT_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$ROOT_DIR/venv/bin/python"
    PYINSTALLER_EXEC="$ROOT_DIR/venv/bin/pyinstaller"
elif [ -f "../venv/bin/python" ]; then
    PYTHON_EXEC="../venv/bin/python"
    PYINSTALLER_EXEC="../venv/bin/pyinstaller"
else
    PYTHON_EXEC="python3"
    PYINSTALLER_EXEC="pyinstaller"
fi

# 1. Crear el entorno base
echo "[1/4] Creando entorno de compilacion..."
"$PYTHON_EXEC" herramientas_build/crear_entorno_ejecutable.py

# 2. Aplicar los parches para PyInstaller
echo "[2/4] Aplicando parches..."
"$PYTHON_EXEC" herramientas_build/aplicar_parches_ejecutable.py

# 3. Generar el SPEC
echo "[3/4] Generando SPEC..."
"$PYTHON_EXEC" herramientas_build/crear_spec_ejecutable.py

# 4. Compilar con PyInstaller
echo "===================================================="
echo "Iniciando PyInstaller..."
cd EMG_Ejecutable_Build
"$PYINSTALLER_EXEC" EMG_Studio.spec --noconfirm --clean
cd ..

# 5. Renombrar carpeta para el usuario
echo "Finalizando estructura de distribucion..."
DIST_DIR="EMG_Ejecutable_Build/dist"
if [ -d "$DIST_DIR/NanduLsd" ]; then
    rm -rf "$DIST_DIR/NanduLsd"
fi

if [ -d "$DIST_DIR/NanduLsd_Core" ]; then
    mv "$DIST_DIR/NanduLsd_Core" "$DIST_DIR/NanduLsd"
fi

# 6. Crear un script lanzador bash
cat << 'EOF' > "$DIST_DIR/NanduLsd/run_nandu.sh"
#!/bin/bash
cd "$(dirname "$0")"
./NanduLsd_Core "$@"
EOF
chmod +x "$DIST_DIR/NanduLsd/run_nandu.sh"

# Sincronizar con la carpeta root build_linux si existe
if [ -d "$ROOT_DIR/build_linux" ] || [ -d "$ROOT_DIR" ]; then
    mkdir -p "$ROOT_DIR/build_linux"
    rm -rf "$ROOT_DIR/build_linux/NanduLsd"
    cp -r "$DIST_DIR/NanduLsd" "$ROOT_DIR/build_linux/"
fi

echo "===================================================="
echo "BUILD COMPLETADO. El ejecutable principal esta en $ROOT_DIR/build_linux/NanduLsd/run_nandu.sh"
echo "===================================================="
