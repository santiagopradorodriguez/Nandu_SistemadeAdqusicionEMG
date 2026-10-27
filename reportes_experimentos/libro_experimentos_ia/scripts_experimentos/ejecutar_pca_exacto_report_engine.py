import os
import sys
import json

repo_root = os.path.abspath(os.curdir)
sys.path.append(os.path.join(repo_root, "EMG_desarrollo", "deep_learning", "pca_umap_clustering"))
import generador_pca_umap as pca_ana

base_dir = os.path.join(repo_root, "EMG_desarrollo", "base_de_datos_electrodos")

# Las mediciones exactas relativas a base_dir
mediciones_rel = [
    os.path.join("2026-10-23", f"{v}_Pruebatricanal_{suf}")
    for v in ['A', 'E', 'I', 'O', 'U']
    for suf in ['Sujeto1', '2_Sujeto1', '3_Sujeto1', '4_Sujeto1']
    if os.path.isdir(os.path.join(base_dir, "2026-10-23", f"{v}_Pruebatricanal_{suf}"))
]

opt_2d_json = "reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/parametros_optimos_pca_2d.json"
opt_3d_json = "reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/parametros_optimos_pca_3d.json"

with open(opt_2d_json) as f: best_params_2d = json.load(f)
with open(opt_3d_json) as f: best_params_3d = json.load(f)

out_dir = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/salida_pca_report_engine"
os.makedirs(out_dir, exist_ok=True)

print("Llamando a pca_ana.ejecutar_procesamiento con exactamente los mismos argumentos de report_engine...")
pca_ana.ejecutar_procesamiento(
    mediciones=mediciones_rel,
    base_dir=base_dir,
    params_2d=best_params_2d,
    params_3d=best_params_3d,
    proc_pca_2d=True,
    proc_pca_3d=True,
    canales_features=["canal_0", "canal_1", "canal_2"],
    modo_alineacion="Pico Volumen Micrófono",
    algoritmo_clustering_pca="GMM",
    estilo_visual="Fronteras",
    out_dir=out_dir,
    aplicar_correccion_intersesion=False,
    highpass_cutoff_hz=20.0,
    lowpass_cutoff_hz=500.0,
    abrir_visor=False
)
print("Ejecucion terminada exitosamente!")
