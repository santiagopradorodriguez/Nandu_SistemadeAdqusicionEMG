
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'/tmp/tmpuwv_k5j9.json', 'r') as f:
    kwargs = json.load(f)

mediciones = ['2026-09-01/A_Prueba1_Candela', '2026-09-01/A_Prueba2_Candela', '2026-09-01/A_Prueba3_Candela', '2026-09-01/A_Prueba4_Candela', '2026-09-01/E_Prueba1_Candela', '2026-09-01/E_Prueba2_Candela', '2026-09-01/E_Prueba3_Candela', '2026-09-01/E_Prueba4_Candela', '2026-09-01/E_Prueba5_Candela', '2026-09-01/I_Prueba1_Candela', '2026-09-01/I_Prueba2_Candela', '2026-09-01/I_Prueba3_Candela', '2026-09-01/I_Prueba4_Candela', '2026-09-01/O_Prueba1_Candela', '2026-09-01/O_Prueba2_Candela', '2026-09-01/O_Prueba3_Candela', '2026-09-01/O_Prueba4_Candela', '2026-09-01/O_Prueba5_Candela', '2026-09-01/U_Prueba1_Candela', '2026-09-01/U_Prueba2_Candela', '2026-09-01/U_Prueba3_Candela', '2026-09-01/U_Prueba4_Candela']
base_dir = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos'

import deep_learning.pca_analysis as pca_ana

print("=========================================================")
print("     INICIANDO BÚSQUEDA DE PARÁMETROS ÓPTIMOS (PCA)     ")
print("=========================================================")

res = pca_ana.buscar_mejor_configuracion_pca(
    mediciones=mediciones,
    base_dir=base_dir,
    params_base=kwargs.get("params_3d", {}),
    aplicar_trevisan=kwargs.get("aplicar_trevisan", False),
    modo_alineacion=kwargs.get("modo_alineacion", "Pico Volumen Micrófono"),
    pre_pct=kwargs.get("pre_pct", 0.4),
    post_pct=kwargs.get("post_pct", 0.6),
    canales_features=kwargs.get("canales_features", ["canal_0", "canal_1", "canal_2"]),
    ignorar_ventana_cero=kwargs.get("ignorar_ventana_cero", False),
    algoritmo_clustering=kwargs.get("algoritmo_clustering_pca", "GMM"),
    logger=print,
    n_components=3,
    aplicar_correccion_intersesion=kwargs.get("aplicar_correccion_intersesion", True),
    tag_nombre=kwargs.get("tag_nombre", None),
    ejecutar_ganador_con_graficos=True
)

best_config = None
best_acc = 0.0
best_sil = 0.0
best_vocal_acc = {}
carpeta_salida = ""

if isinstance(res, tuple) and len(res) >= 3:
    best_config = res[0]
    best_acc = res[1]
    best_sil = res[2]
    best_vocal_acc = res[3] if len(res) >= 4 and isinstance(res[3], dict) else {}
    carpeta_salida = res[5] if len(res) >= 6 else ""

if best_config:
    if len(best_config) == 4:
        best_smooth, best_pts, best_alpha, best_notch = best_config
    else:
        best_smooth, best_pts, best_alpha = best_config[:3]
        best_notch = 2.0
    out_file = os.path.join(project_root, "deep_learning", "parametros_optimos_pca.json")
    with open(out_file, "w") as f:
        json.dump({
            "smooth_ms": best_smooth,
            "target_length": best_pts,
            "alpha_ruido": best_alpha,
            "notch_q": best_notch,
            "accuracy_clasificacion": best_acc,
            "porcentaje_por_vocal": best_vocal_acc,
            "silhouette_score": best_sil
        }, f, indent=4)
    print("")
    print("---------------------------------------------------------")
    print("¡CONFIGURACIÓN ÓPTIMA HALLADA (MAX CLASIFICACIÓN)! ")
    print("  - Smooth (Envolvente): " + str(best_smooth) + " ms")
    print("  - Remuestreo (Pts):    " + str(best_pts))
    print("  - Alfa Ruido:          " + str(best_alpha))
    print("  - Notch Q:             " + str(best_notch))
    print("  - Clasificación (%):   " + str(best_acc) + " %")
    if best_vocal_acc:
        print("  - Desglose por Vocal:")
        for v_name, v_pct in best_vocal_acc.items():
            print(f"      * Vocal {v_name}: {v_pct:.1f}%")
    print("  - Silhouette Score:    " + str(best_sil))
    if carpeta_salida:
        print("  - Carpeta Versionada:  " + str(carpeta_salida))
        print("  - Gráficos y Distribución generados exitosamente.")
    print("---------------------------------------------------------")
    print("Se cargaron los resultados automáticamente.")
