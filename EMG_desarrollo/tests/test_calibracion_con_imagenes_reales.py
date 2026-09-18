# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Script de prueba y validación de calibración geométrica 3D sobre imágenes reales de electrodos sEMG faciales.
# ==============================================================================

import os
import sys
import json
import cv2
import numpy as np

# Agregar directorio de adquisición al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "acquisition")))
from calibracion_espacial_electrodos import AntigravityAgent


def procesar_imagen_montaje(
    ruta_imagen: str,
    carpeta_salida: str,
    nombre_base: str,
) -> None:
    """
    Procesa una imagen de montaje conteniendo tomas faciales y de electrodos.
    """
    print(f"\n=======================================================")
    print(f"Procesando: {nombre_base} ({ruta_imagen})")
    print(f"=======================================================")

    if not os.path.exists(ruta_imagen):
        print(f"Error: No existe el archivo {ruta_imagen}")
        return

    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return

    alto, ancho, _ = img.shape
    print(f"Dimensiones de la imagen completa: {ancho} x {alto} px")

    agente = AntigravityAgent(refine_landmarks=True)

    # 1. Probar detección en la imagen completa
    resultado_global = agente.triangular_electrodos(img, usar_broches_metalicos=True)
    if resultado_global["exito"]:
        meta = resultado_global["metadatos_calibracion_espacial"]
        print(f"Rostro detectado en imagen global.")
        print(f"  Distancia interpupilar 3D: {meta['distancia_interpupilar_3d']:.6f}")
        print(f"  Distancia interpupilar en pixeles: {meta['distancia_interpupilar_px']:.1f} px")
        print(f"  Punta de la nariz (origen 3D): {meta['origen_coordenadas']['coordenadas_3d_absolutas']}")
        print(f"  Electrodos triangulados: {len(meta['electrodos'])}")
        for el in meta["electrodos"]:
            print(f"    - {el['canal']} ({el['musculo_estimado']}): dist_3d_norm={el['distancia_3d_normalizada']:.4f}, error_px={el['error_reproyeccion_2d_px']}")

        # Renderizar
        vis_global = agente.renderizar_visualizacion(img, resultado_global, dibujar_malla_completa=False)
        ruta_vis_global = os.path.join(carpeta_salida, f"{nombre_base}_global_triangulado.png")
        cv2.imwrite(ruta_vis_global, vis_global)
        print(f"  Visualizacion guardada en: {ruta_vis_global}")

        # Guardar JSON
        ruta_json = os.path.join(carpeta_salida, f"{nombre_base}_metadatos.json")
        agente.exportar_a_json(resultado_global, ruta_json)
        print(f"  Metadatos guardados en: {ruta_json}")
    else:
        print(f"Aviso en imagen global: {resultado_global.get('mensaje', 'Sin rostro')}")

    # 2. Segmentar los paneles individuales si la imagen es un mosaico horizontal (ancho >> alto)
    aspect_ratio = ancho / float(alto)
    if aspect_ratio > 2.0:
        # Estimar cantidad de paneles horizontales (típicamente 4 o 5)
        num_paneles = int(round(aspect_ratio / 0.8)) # paneles aprox 0.8 aspect ratio cada uno
        num_paneles = max(3, min(num_paneles, 6))
        ancho_panel = ancho // num_paneles
        print(f"Mosaico detectado ({num_paneles} paneles estimados, ~{ancho_panel} px c/u). Evaluando panel por panel...")

        for p_idx in range(num_paneles):
            x_ini = p_idx * ancho_panel
            x_fin = min((p_idx + 1) * ancho_panel, ancho)
            panel = img[:, x_ini:x_fin].copy()

            res_panel = agente.triangular_electrodos(panel, usar_broches_metalicos=True)
            if res_panel["exito"]:
                meta_p = res_panel["metadatos_calibracion_espacial"]
                print(f"\n--- PANEL {p_idx + 1}/{num_paneles} ---")
                print(f"  Distancia interpupilar 3D: {meta_p['distancia_interpupilar_3d']:.6f} ({meta_p['distancia_interpupilar_px']:.1f} px)")
                print(f"  Electrodos detectados: {len(meta_p['electrodos'])}")
                for el in meta_p["electrodos"]:
                    print(f"    * {el['canal']}: dist_norm={el['distancia_3d_normalizada']:.4f}, v_3d={el['vertice_malla_3d']}")

                vis_p = agente.renderizar_visualizacion(panel, res_panel, dibujar_malla_completa=False)
                ruta_vis_p = os.path.join(carpeta_salida, f"{nombre_base}_panel_{p_idx + 1}_triangulado.png")
                cv2.imwrite(ruta_vis_p, vis_p)
                print(f"  Visualizacion guardada en: {ruta_vis_p}")


def probar_inyeccion_csv(carpeta_salida: str) -> None:
    """
    Prueba la inyección de metadatos de geometría sEMG en un archivo CSV de adquisición sintético.
    """
    print("\n=======================================================")
    print("Probando inyeccion de metadatos en archivo CSV sEMG")
    print("=======================================================")

    ruta_csv_prueba = os.path.join(carpeta_salida, "grabacion_prueba_calibracion.csv")

    # Crear archivo CSV sintético similar al de adquisición
    tiempo = np.linspace(0, 1.0, 500)
    canal0 = np.sin(2 * np.pi * 50 * tiempo)
    canal1 = np.cos(2 * np.pi * 75 * tiempo)
    canal2 = np.sin(2 * np.pi * 100 * tiempo)
    datos = np.column_stack([tiempo, canal0, canal1, canal2])

    np.savetxt(
        ruta_csv_prueba,
        datos,
        delimiter=",",
        header="Tiempo (s),Canal 0,Canal 1,Canal 2",
        comments="",
    )
    print(f"CSV de prueba generado: {ruta_csv_prueba}")

    # Estructura de metadatos de triangulación
    metadatos_demo = {
        "exito": True,
        "metadatos_calibracion_espacial": {
            "version_modulo": "1.0.0",
            "sujeto": "Lucas",
            "distancia_interpupilar_3d": 0.145201,
            "distancia_interpupilar_px": 142.5,
            "origen_coordenadas": {
                "descripcion": "Punta de la nariz (pronasale)",
                "indice_mediapipe": 1,
            },
            "electrodos": [
                {
                    "canal": "Canal 0",
                    "musculo": "Mylohyoid",
                    "vertice_malla_3d": 152,
                    "distancia_3d_normalizada": 0.5842,
                },
                {
                    "canal": "Canal 1",
                    "musculo": "Depresor Anguli Oris",
                    "vertice_malla_3d": 377,
                    "distancia_3d_normalizada": 0.4215,
                },
                {
                    "canal": "Canal 2",
                    "musculo": "Orbicularis Oris",
                    "vertice_malla_3d": 17,
                    "distancia_3d_normalizada": 0.2831,
                },
            ],
        },
    }

    exito_inyec = AntigravityAgent.inyectar_metadatos_en_csv(ruta_csv_prueba, metadatos_demo)
    print(f"Resultado inyeccion en CSV: {'EXITO' if exito_inyec else 'FALLO'}")

    # Verificar que pandas o numpy puedan leer el archivo sin romperse con la cabecera comentada
    try:
        import pandas as pd
        df = pd.read_csv(ruta_csv_prueba, comment="#")
        print(f"Validacion con pandas (comment='#'): {df.shape[0]} filas, {df.shape[1]} columnas leidas correctamente.")
        print(f"Columnas: {list(df.columns)}")
    except Exception as e:
        print(f"Error al validar lectura con pandas: {e}")


def main() -> None:
    carpeta_salida = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "resultados", "calibracion_espacial_pruebas")
    )
    os.makedirs(carpeta_salida, exist_ok=True)

    imagenes_prueba = [
        (
            "/home/santiago/.gemini/antigravity/brain/fd13144b-12f0-469f-a648-c112c1446237/.user_uploaded/media_1789364103426.png",
            "sujeto_lucas_toma_1",
        ),
        (
            "/home/santiago/.gemini/antigravity/brain/fd13144b-12f0-469f-a648-c112c1446237/.user_uploaded/media_1789364146266.png",
            "sujeto_lucas_toma_2",
        ),
        (
            "/home/santiago/.gemini/antigravity/brain/fd13144b-12f0-469f-a648-c112c1446237/.user_uploaded/media_1789364207033.png",
            "sujeto_lucas_toma_3",
        ),
    ]

    for ruta, nombre in imagenes_prueba:
        procesar_imagen_montaje(ruta, carpeta_salida, nombre)

    probar_inyeccion_csv(carpeta_salida)


if __name__ == "__main__":
    main()
