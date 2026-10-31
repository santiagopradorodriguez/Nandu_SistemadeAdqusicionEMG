#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: utils/limpiar_repo_para_github.py
# Descripción: Sanitiza el repositorio antes de hacer git push a GitHub.
#              Separa configuraciones personales en config_local.json (ignorado),
#              limpia nombres/IPs locales en config_general.json y metronome_config.json,
#              y genera un .gitignore blindado contra datos biomédicos crudos.
# ==============================================================================

import os
import json
import argparse


GITIGNORE_TEMPLATE = """# ==============================================================================
# ÑANDÚ LSD — .gitignore blindado (Datos Biomédicos, Modelos Pesados y Config Local)
# ==============================================================================

# 1. Configuraciones locales de cada PC (IPs de VM, últimos sujetos, rutas personales)
config_local.json
metronome_config.json
cache_datos/
_sujeto_explorer_cache.json
*.log
multiplexer_error.log
argv_log.txt

# 2. Bases de datos crudas y grabaciones de sujetos (Privacidad Biomédica)
base_de_datos_electrodos/
base_de_datos_letras/
analisis_comparativos/
analisis_de_sesiones/
resultados/
EMG_desarrollo/base_de_datos_electrodos/
EMG_desarrollo/resultados/
fotos/

# 3. Entornos virtuales, compilaciones PyInstaller y carpetas temporales
venv/
.venv/
env/
__pycache__/
*.py[cod]
*$py.class
build/
dist/
build_linux/
build_windows/
EMG_Ejecutable_Build/
carpeta_para_vm_windows/
temp_*.py
temp_*.json
.pytest_cache/
"""


CLEAN_CONFIG_GENERAL = {
    "hardware": {
        "dispositivo_daq": "Dev1",
        "sample_rate": 2000,
        "modo_terminal": "DIFF",
        "zmq_bridge_activo": False,
        "zmq_bridge_endpoint": "auto:5555",
    },
    "adquisicion": {
        "bpm": 25,
        "subdivisiones": 4,
        "ruido_segundos": 3.0,
        "notch_50hz_activo": False,
        "pasabanda_low_hz": 20.0,
        "pasabanda_high_hz": 500.0,
        "limitar_rango_500uv": True,
    },
    "protocolo_autoforge": {
        "modo_secuencia": "debruijn",
        "calibrar_cvm_por_defecto": False,
        "cvm_modo": "guiado_sin_metronomo",
        "cvm_tomas_por_vocal": 2,
        "cvm_duracion_contraccion_s": 2.5,
        "cvm_duracion_descanso_s": 2.5,
        "repeticiones_continua_default": 25,
        "repeticiones_discreta_default": 5,
    },
    "canales": {
        "Canal 0": {"musculo": "Depresor Anguli Oris", "ganancia": 1004.0},
        "Canal 1": {"musculo": "Mylohyoid", "ganancia": 996.0},
        "Canal 2": {"musculo": "Orbicularis Oris", "ganancia": 1020.0},
        "Canal 3": {"musculo": "Micrófono", "ganancia": 1.0},
        "Canal 4": {"musculo": "Sin Asignar", "ganancia": 1000.0},
        "Canal 5": {"musculo": "Sin Asignar", "ganancia": 1000.0},
        "Canal 6": {"musculo": "Sin Asignar", "ganancia": 1000.0},
        "Canal 7": {"musculo": "Sin Asignar", "ganancia": 1000.0},
    },
}


def sanitizar_repositorio(repo_dir: str):
    repo_dir = os.path.abspath(repo_dir)
    print(f"==> Sanitizando repositorio en: {repo_dir}")

    # 1. Crear o actualizar .gitignore
    gitignore_path = os.path.join(repo_dir, ".gitignore")
    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write(GITIGNORE_TEMPLATE)
    print(f"  [OK] .gitignore actualizado ({gitignore_path})")

    # 2. Respaldar preferencias personales actuales en config_local.json (ignorado por Git)
    cfg_general_path = os.path.join(repo_dir, "config_general.json")
    cfg_local_path = os.path.join(repo_dir, "config_local.json")
    if os.path.exists(cfg_general_path) and not os.path.exists(cfg_local_path):
        try:
            with open(cfg_general_path, "r", encoding="utf-8") as f_in:
                current_cfg = json.load(f_in)
            with open(cfg_local_path, "w", encoding="utf-8") as f_local:
                json.dump(current_cfg, f_local, indent=4, ensure_ascii=False)
            print(f"  [OK] Preferencias personales respaldadas en config_local.json (ignorado por Git)")
        except Exception as e:
            print(f"  [Aviso] No se pudo respaldar config_general.json: {e}")

    # 3. Escribir config_general.json neutro y portable para cualquier PC
    with open(cfg_general_path, "w", encoding="utf-8") as f_clean:
        json.dump(CLEAN_CONFIG_GENERAL, f_clean, indent=4, ensure_ascii=False)
    print("  [OK] config_general.json neutralizado con valores por defecto independientes de la PC.")

    # 4. Limpiar temporales sueltos (temp_*.py, temp_*.json)
    eliminados = 0
    for root, _, files in os.walk(repo_dir):
        if any(skip in root for skip in (".git", "venv", "base_de_datos_electrodos")):
            continue
        for fname in files:
            if fname.startswith(("temp_run_", "temp_comparativo", "temp_procesar", "temp_sesion")) or fname in ("multiplexer_error.log", "argv_log.txt"):
                try:
                    os.remove(os.path.join(root, fname))
                    eliminados += 1
                except Exception:
                    pass
    print(f"  [OK] Archivos temporales de ejecución eliminados: {eliminados}")
    print("\n==> ¡Repositorio listo para subir a GitHub sin exponer datos personales ni sesiones crudas!")


def main():
    parser = argparse.ArgumentParser(description="Sanitiza el repositorio Ñandú LSD para GitHub.")
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--repo", default=default_root, help="Ruta raíz del repositorio")
    args = parser.parse_args()
    sanitizar_repositorio(args.repo)


if __name__ == "__main__":
    main()
