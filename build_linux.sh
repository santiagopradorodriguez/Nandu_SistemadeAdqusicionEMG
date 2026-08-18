#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/EMG_desarrollo"

echo "===================================================="
echo "NANDU LSD - Compilacion Multiplataforma (LINUX)"
echo "===================================================="

# Determinar interprete de Python y PyInstaller
if [ -f "$ROOT_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$ROOT_DIR/venv/bin/python"
    PYINSTALLER_EXEC="$ROOT_DIR/venv/bin/pyinstaller"
else
    PYTHON_EXEC="python3"
    PYINSTALLER_EXEC="pyinstaller"
fi

echo "[1/4] Creando entorno de compilacion temporal..."
"$PYTHON_EXEC" herramientas_build/crear_entorno_ejecutable.py

echo "[2/4] Aplicando parches de compatibilidad..."
"$PYTHON_EXEC" herramientas_build/aplicar_parches_ejecutable.py

echo "[3/4] Generando archivo .spec..."
"$PYTHON_EXEC" herramientas_build/crear_spec_ejecutable.py

echo "[4/4] Ejecutando PyInstaller..."
cd EMG_Ejecutable_Build
"$PYINSTALLER_EXEC" EMG_Studio.spec --noconfirm --clean
cd ..

echo "Finalizando estructura de distribucion..."
DIST_DIR="EMG_Ejecutable_Build/dist"
if [ -d "$DIST_DIR/NanduLsd" ]; then
    rm -rf "$DIST_DIR/NanduLsd"
fi
if [ -d "$DIST_DIR/NanduLsd_Core" ]; then
    mv "$DIST_DIR/NanduLsd_Core" "$DIST_DIR/NanduLsd"
fi

# Crear script lanzador con soporte para reenviar argumentos
cat << 'EOF' > "$DIST_DIR/NanduLsd/run_nandu.sh"
#!/bin/bash
cd "$(dirname "$0")"
./NanduLsd_Core "$@"
EOF
chmod +x "$DIST_DIR/NanduLsd/run_nandu.sh"

# Sincronizar con la carpeta root build_linux
mkdir -p "$ROOT_DIR/build_linux"
if [ -d "$ROOT_DIR/build_linux/NanduLsd" ]; then
    rm -rf "$ROOT_DIR/build_linux/NanduLsd"
fi
cp -r "$DIST_DIR/NanduLsd" "$ROOT_DIR/build_linux/"

echo "===================================================="
echo "BUILD COMPLETADO EXITOSAMENTE."
echo "Ejecutable listo en: $ROOT_DIR/build_linux/NanduLsd/run_nandu.sh"
echo "===================================================="
