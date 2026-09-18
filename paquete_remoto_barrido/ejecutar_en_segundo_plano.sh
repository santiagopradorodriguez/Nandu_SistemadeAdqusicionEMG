#!/usr/bin/env bash
# ==============================================================================
# Script de ejecucion remota en segundo plano
# ==============================================================================

DIR_ACTUAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR_ACTUAL"

echo "======================================================================"
echo "INICIANDO BARRIDO DE HIPERPARAMETROS EN MAQUINA REMOTA"
echo "======================================================================"

# Verificar o crear entorno virtual si no existe
if [ ! -d "venv_remoto" ]; then
    echo "Creando entorno virtual Python (venv_remoto)..."
    python3 -m venv venv_remoto
    echo "Instalando dependencias basicas..."
    ./venv_remoto/bin/pip install --upgrade pip
    echo "Instalando PyTorch CPU (version ligera)..."
    ./venv_remoto/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo "Instalando librerias cientificas..."
    ./venv_remoto/bin/pip install numpy scipy scikit-learn pandas matplotlib
fi

echo "Lanzando barrido sistematico en segundo plano..."
export MPLCONFIGDIR=/tmp/matplotlib
nohup ./venv_remoto/bin/python barrido_hiperparametros_separabilidad.py --modo estandar --epochs 250 > barrido.log 2>&1 &
PID=$!

echo "======================================================================"
echo "PROCESO INICIADO CORRECTAMENTE CON PID: $PID"
echo "El barrido seguira corriendo aunque cierres la sesion SSH."
echo ""
echo "Para ver el avance en tiempo real:"
echo "  tail -f barrido.log"
echo ""
echo "Para verificar si sigue corriendo:"
echo "  ps aux | grep barrido_hiperparametros"
echo "======================================================================"
