# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de purga y liberación de espacio para archivos de análisis JSON.
# ==============================================================================

import os
import sys
import argparse
from pathlib import Path

def purgar_analisis_results(directorio_base: Path, solo_env: bool = False, callback_progreso = None) -> dict:
    """
    Recorre recursivamente directorio_base y elimina los archivos JSON de análisis
    (analisis_results*.json, results*.json, *_env*.json), preservando intactos
    los archivos de bioseñales (CSV, WAV), metadatos (metadata.json) y fotos.

    Args:
        directorio_base: Ruta de la carpeta a escanear y limpiar.
        solo_env: Si es True, solo elimina archivos con '_env' (archivos acumulativos).
        callback_progreso: Función opcional callback(actual, total, bytes_liberados).

    Returns:
        dict con estadísticas de archivos eliminados y espacio liberado.
    """
    directorio_base = Path(directorio_base)
    if not directorio_base.exists():
        return {"error": f"El directorio {directorio_base} no existe", "eliminados": 0, "bytes": 0}

    # 1. Escaneo previo de archivos a eliminar
    archivos_a_eliminar = []
    for raiz, _, archivos in os.walk(directorio_base):
        for f in archivos:
            f_lower = f.lower()
            if 'results' in f_lower and f_lower.endswith('.json'):
                if solo_env:
                    if '_env' in f_lower:
                        archivos_a_eliminar.append(Path(raiz) / f)
                else:
                    archivos_a_eliminar.append(Path(raiz) / f)

    total_archivos = len(archivos_a_eliminar)
    bytes_liberados = 0
    eliminados = 0

    print(f"[Purga] Se detectaron {total_archivos} archivos de resultados para eliminar en {directorio_base.name}")

    for idx, fp in enumerate(archivos_a_eliminar, start=1):
        try:
            sz = fp.stat().st_size
            fp.unlink()
            bytes_liberados += sz
            eliminados += 1
        except Exception as e:
            print(f"[Aviso] No se pudo eliminar {fp}: {e}")

        if callback_progreso and (idx % 50 == 0 or idx == total_archivos):
            callback_progreso(idx, total_archivos, bytes_liberados)
        elif idx % 200 == 0 or idx == total_archivos:
            mb = bytes_liberados / (1024 * 1024)
            pct = (idx / total_archivos) * 100.0 if total_archivos > 0 else 100.0
            print(f"[Purga] {idx}/{total_archivos} ({pct:.1f}%) - {mb:.2f} MB liberados...")

    gb_liberados = bytes_liberados / (1024 * 1024 * 1024)
    mb_liberados = bytes_liberados / (1024 * 1024)

    return {
        "eliminados": eliminados,
        "bytes": bytes_liberados,
        "mb": mb_liberados,
        "gb": gb_liberados
    }

def main():
    parser = argparse.ArgumentParser(description="Purgador de archivos JSON de análisis para liberar espacio.")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).resolve().parent.parent / "base_de_datos_electrodos"),
        help="Directorio a purgar (por defecto base_de_datos_electrodos)"
    )
    parser.add_argument(
        "--solo-env",
        action="store_true",
        help="Solo eliminar archivos acumulativos de envolventes (*_env*.json)"
    )

    args = parser.parse_args()
    target_dir = Path(args.dir)

    print("=" * 70)
    print(f"INICIANDO PURGA DE RESULTADOS EN: {target_dir}")
    print(f"Modo: {'Solo archivos acumulativos _env' if args.solo_env else 'Todos los archivos results JSON'}")
    print("=" * 70)

    res = purgar_analisis_results(target_dir, solo_env=args.solo_env)

    print("=" * 70)
    print("PURGA COMPLETADA")
    print(f"- Archivos eliminados: {res['eliminados']}")
    print(f"- Espacio liberado:    {res['gb']:.2f} GB ({res['mb']:.2f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
