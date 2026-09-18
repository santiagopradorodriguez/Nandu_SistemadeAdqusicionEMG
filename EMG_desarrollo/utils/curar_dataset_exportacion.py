# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Curaduría y exportación limpia de datasets de mediciones sEMG.
# ==============================================================================

import os
import sys
import shutil
import argparse
import time
from pathlib import Path

EXTENSIONES_IMAGEN = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

def obtener_tamano_directorio(directorio: Path) -> int:
    """Calcula el tamaño total en bytes de un directorio de forma recursiva."""
    total_bytes = 0
    for ruta_raiz, _, archivos in os.walk(directorio):
        for f in archivos:
            fp = os.path.join(ruta_raiz, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes

def curar_sesion_mediciones(ruta_origen: Path, ruta_destino: Path):
    """
    Clona una carpeta de mediciones hacia un directorio seguro en Descargas,
    preservando intactas bioseñales (CSV, WAV), metadatos (JSON, logs) y
    exclusivamente la fotografía photo.png, eliminando el resto de imágenes.
    """
    if not ruta_origen.exists():
        print(f"[Error] La ruta de origen no existe: {ruta_origen}")
        return None

    print("=" * 70)
    print(f"[Inicio Curaduría] Origen:  {ruta_origen}")
    print(f"[Inicio Curaduría] Destino: {ruta_destino}")
    print("=" * 70)

    # Preparar directorio destino limpio
    if ruta_destino.exists():
        print(f"[Aviso] Limpiando destino previo existente: {ruta_destino}")
        shutil.rmtree(ruta_destino)
    ruta_destino.mkdir(parents=True, exist_ok=True)

    # Listar todos los archivos a procesar
    todos_los_archivos = []
    for raiz, _, archivos in os.walk(ruta_origen):
        for arch in archivos:
            todos_los_archivos.append(Path(raiz) / arch)

    total_archivos = len(todos_los_archivos)
    print(f"[Auditoría Previa] Total de archivos detectados en origen: {total_archivos}")

    imagenes_eliminadas = []
    fotos_conservadas = []
    archivos_conservados = []

    t_inicio = time.time()
    for idx, archivo_src in enumerate(todos_los_archivos, start=1):
        rel_path = archivo_src.relative_to(ruta_origen)
        archivo_dst = ruta_destino / rel_path

        ext = archivo_src.suffix.lower()
        es_imagen = ext in EXTENSIONES_IMAGEN
        es_foto_permitida = (archivo_src.name == "photo.png")

        if es_imagen and not es_foto_permitida:
            # Descartar imagen que no sea photo.png
            imagenes_eliminadas.append(rel_path)
        else:
            # Crear directorio padre si no existe
            archivo_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archivo_src, archivo_dst)

            if es_foto_permitida:
                fotos_conservadas.append(rel_path)
            else:
                archivos_conservados.append(rel_path)

        # Monitoreo de progreso en tiempo real
        if idx % 25 == 0 or idx == total_archivos:
            porcentaje = (idx / total_archivos) * 100.0
            print(f"[Progreso] {idx}/{total_archivos} archivos ({porcentaje:.1f}%) procesados...")

    t_total = time.time() - t_inicio
    tamano_final_bytes = obtener_tamano_directorio(ruta_destino)
    tamano_final_mb = tamano_final_bytes / (1024 * 1024)

    # Resumen en consola
    print("\n" + "=" * 70)
    print(f"RESUMEN DE CURADURÍA: {ruta_origen.name}")
    print("=" * 70)
    print(f"- Archivos de imagen eliminados/omitidos: {len(imagenes_eliminadas)}")
    print(f"- Archivos de bioseñales/metadatos conservados: {len(archivos_conservados)}")
    print(f"- Archivos 'photo.png' conservados: {len(fotos_conservadas)}")
    print(f"- Peso final de la carpeta: {tamano_final_mb:.2f} MB ({tamano_final_bytes:,} bytes)")
    print(f"- Tiempo de procesamiento: {t_total:.2f} s")
    print("\nUbicación de 'photo.png' conservados:")
    for f in sorted(fotos_conservadas):
        print(f"  * {ruta_destino.name} / {f}")
    print("=" * 70 + "\n")

    return {
        "origen": str(ruta_origen),
        "destino": str(ruta_destino),
        "eliminadas": len(imagenes_eliminadas),
        "conservados": len(archivos_conservados),
        "fotos": len(fotos_conservadas),
        "tamano_mb": tamano_final_mb,
        "tamano_bytes": tamano_final_bytes
    }

def main():
    parser = argparse.ArgumentParser(description="Curador de datasets de bioseñales sEMG.")
    parser.add_argument(
        "--fechas",
        nargs="+",
        default=None,
        help="Fechas a procesar (ej. 2026-09-15 2026-09-01)"
    )
    parser.add_argument(
        "--mapeo",
        nargs="+",
        default=None,
        help="Mapeos en formato origen:nombre_destino (ej. 2026-09-16:2026-09-15_clean 2026-09-01:2026-09-01_clean)"
    )
    parser.add_argument(
        "--incluir-16",
        action="store_true",
        help="Incluir también la carpeta formal 2026-09-16 grabada el 15/09"
    )
    parser.add_argument(
        "--destino-dir",
        default=str(Path.home() / "Descargas"),
        help="Directorio base de salida (por defecto ~/Descargas)"
    )

    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent
    base_datos_dir = repo_dir / "base_de_datos_electrodos"
    destino_base = Path(args.destino_dir)

    tareas = []
    if args.mapeo:
        for m in args.mapeo:
            if ":" in m:
                orig, dest_name = m.split(":", 1)
            else:
                orig = m
                dest_name = f"{m}_clean"
            tareas.append((orig, dest_name))
    else:
        fechas_a_procesar = list(args.fechas) if args.fechas else ["2026-09-15", "2026-09-01"]
        if args.incluir_16 and "2026-09-16" not in fechas_a_procesar:
            fechas_a_procesar.append("2026-09-16")
        for f in fechas_a_procesar:
            tareas.append((f, f"{f}_clean"))

    print(f"Directorio base de mediciones: {base_datos_dir}")
    print(f"Directorio de exportación:    {destino_base}")
    print(f"Tareas a procesar:            {tareas}\n")

    resumen_global = []
    for orig, dest_name in tareas:
        ruta_origen = base_datos_dir / orig
        ruta_destino = destino_base / dest_name
        res = curar_sesion_mediciones(ruta_origen, ruta_destino)
        if res:
            resumen_global.append(res)

    print("\n" + "#" * 70)
    print("REPORTE GLOBAL CONSOLIDADO")
    print("#" * 70)
    for r in resumen_global:
        print(f"Carpeta: {Path(r['destino']).name} | Eliminados: {r['eliminadas']} imgs | Conservados: {r['conservados']} + {r['fotos']} fotos | Peso: {r['tamano_mb']:.2f} MB")
    print("#" * 70)

if __name__ == "__main__":
    main()
