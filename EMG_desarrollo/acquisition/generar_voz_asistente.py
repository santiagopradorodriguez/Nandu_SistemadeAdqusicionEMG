#!/usr/bin/env python3
"""Generador de los WAV de la guia por voz del asistente lineal.

Los sintetiza con la misma voz Piper del asistente TARS del usuario y los
escribe en acquisition/voz_asistente. Ejecutar una sola vez con el
interprete del venv del proyecto:

    /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/venv/bin/python generar_voz_asistente.py

Las claves y textos salen de FRASES_VOZ_ASISTENTE en fotometria_webcam.py;
los archivos ya existentes y validos se conservan.
"""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("LC_ALL", "C.utf8")
os.environ.setdefault("LANG", "C.utf8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

RUTA_RAIZ = Path(__file__).resolve().parent.parent
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from acquisition.fotometria_webcam import (
    DIRECTORIO_VOZ_ASISTENTE,
    FRASES_VOZ_ASISTENTE,
    MODELO_PIPER_RUTA,
    PIPER_BINARIO_RUTA,
)


def main() -> int:
    if not PIPER_BINARIO_RUTA.is_file() or not MODELO_PIPER_RUTA.is_file():
        print("No se encontro el binario o el modelo de Piper:")
        print(f"  binario: {PIPER_BINARIO_RUTA}")
        print(f"  modelo : {MODELO_PIPER_RUTA}")
        return 1
    DIRECTORIO_VOZ_ASISTENTE.mkdir(parents=True, exist_ok=True)
    total = len(FRASES_VOZ_ASISTENTE)
    generados = 0
    fallidos: list = []
    for clave, texto in FRASES_VOZ_ASISTENTE.items():
        destino = DIRECTORIO_VOZ_ASISTENTE / f"{clave}.wav"
        if destino.is_file() and destino.stat().st_size > 0:
            print(f"[OK existente] {clave}")
            generados += 1
            continue
        try:
            resultado = subprocess.run(
                [
                    str(PIPER_BINARIO_RUTA),
                    "--model", str(MODELO_PIPER_RUTA),
                    "--output_file", str(destino),
                ],
                input=texto.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=90,
            )
            if resultado.returncode == 0 and destino.is_file() and destino.stat().st_size > 0:
                print(f"[OK] {clave}")
                generados += 1
            else:
                print(f"[FALLO] {clave}")
                fallidos.append(clave)
        except Exception as exc:
            print(f"[FALLO] {clave}: {exc}")
            fallidos.append(clave)
    print(f"Listo: {generados} de {total} wavs en {DIRECTORIO_VOZ_ASISTENTE}")
    return 1 if fallidos else 0


if __name__ == "__main__":
    raise SystemExit(main())
