#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institucion: Laboratorio de Sistemas Dinamicos (LSD) - FCEyN, UBA
# Descripcion: Interfaz CLI para ejecucion desatendida de colas de reportes.
# ==============================================================================

import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from analysis.report_queue import ReportTask, ReportQueueManager


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutor por lotes de reportes tecnicos EMG (CLI)."
    )
    parser.add_argument(
        "--cola",
        type=str,
        default=None,
        help="Ruta a un archivo JSON con la cola de tareas configuradas."
    )
    parser.add_argument(
        "--sesion",
        type=str,
        default=None,
        help="Carpeta de sesion (ej. 2026-10-23) o ruta directa para tarea rapida."
    )
    parser.add_argument(
        "--filtro",
        type=str,
        default="",
        help="Filtro de subcadena para nombres de tomas (ej. Pruebatricanal)."
    )
    parser.add_argument(
        "--tipo",
        type=str,
        default="pca",
        choices=["base", "pca", "snr", "espectral", "multimodal", "todos"],
        help="Tipo de reporte a generar."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Etiqueta sufijo para diferenciar archivos de salida."
    )
    parser.add_argument(
        "--salida",
        type=str,
        default=None,
        help="Directorio de salida de los reportes PDF/LaTeX."
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=500.0,
        help="Frecuencia de corte pasabajos en Hz (por defecto 500.0)."
    )
    parser.add_argument(
        "--pesos",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 1.0],
        help="Pesos multiplicadores de canales C0 C1 C2 para PCA."
    )
    args = parser.parse_args()

    manager = ReportQueueManager(output_dir=args.salida)

    if args.cola:
        manager.load_from_json(args.cola)
    elif args.sesion:
        opciones = {
            "filtro_prueba": args.filtro,
            "lowpass_cutoff_hz": args.lowpass,
            "correccion_impedancia": False,
            "aplicar_correccion_intersesion": False,
            "pesos_canales": args.pesos,
            "ejecutar_grid": True,
            "report_tag": args.tag
        }
        task = ReportTask(
            nombre=f"Tarea CLI - {args.sesion} ({args.filtro or 'Todas'})",
            sesion=args.sesion,
            tipo_reporte=args.tipo,
            opciones_analisis=opciones
        )
        manager.add_task(task)
    else:
        default_json = os.path.abspath(
            os.path.join(script_dir, "..", "reportes_experimentos", "cola_tareas_2026_10_23.json")
        )
        if os.path.exists(default_json):
            print(f"Cargando cola por defecto: {default_json}")
            manager.load_from_json(default_json)
        else:
            parser.print_help()
            sys.exit(1)

    resumen = manager.execute_queue(logger=print, continuar_en_error=True)
    if resumen.get("fallidas", 0) > 0 and resumen.get("exitosas", 0) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
