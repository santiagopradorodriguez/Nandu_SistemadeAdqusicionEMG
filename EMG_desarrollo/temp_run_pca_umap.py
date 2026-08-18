
import json
import sys
with open(r'/tmp/tmp8_mo3td3_pca_umap.json', 'r') as f:
    kwargs = json.load(f)

# Llamamos al original generador_pca_umap de forma nativa para respetar todos los prints de consola
import deep_learning.pca_umap_clustering.generador_pca_umap as generador
try:
    generador.ejecutar_procesamiento(
        mediciones=['2026-07-10/A_T1_Lucas', '2026-07-10/A_T2_Lucas', '2026-07-10/A_T3_Lucas', '2026-07-10/A_T4_Lucas', '2026-07-10/A_T5_Lucas', '2026-07-10/A_T6_Lucas', '2026-07-10/A_T7_Lucas', '2026-07-10/E_T1_Lucas', '2026-07-10/E_T2_Lucas', '2026-07-10/E_T3_Lucas', '2026-07-10/E_T4_Lucas', '2026-07-10/E_T5_Lucas', '2026-07-10/E_T6_Lucas', '2026-07-10/E_T7_Lucas', '2026-07-10/I_T1_Lucas', '2026-07-10/I_T2_Lucas', '2026-07-10/I_T3_Lucas', '2026-07-10/I_T4_Lucas', '2026-07-10/I_T5_Lucas', '2026-07-10/I_T6_Lucas', '2026-07-10/I_T7_Lucas', '2026-07-10/O_T1_Lucas', '2026-07-10/O_T2_Lucas', '2026-07-10/O_T3_Lucas', '2026-07-10/O_T4_Lucas', '2026-07-10/O_T5_Lucas', '2026-07-10/O_T6_Lucas', '2026-07-10/O_T7_Lucas', '2026-07-10/U_T1_Lucas', '2026-07-10/U_T2_Lucas', '2026-07-10/U_T3_Lucas', '2026-07-10/U_T4_Lucas', '2026-07-10/U_T5_Lucas', '2026-07-10/U_T6_Lucas', '2026-07-10/U_T7_Lucas'],
        base_dir=r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos',
        params_2d=kwargs.get('params_2d'),
        params_3d=kwargs.get('params_3d'),
        params_umap=kwargs.get('params_umap'),
        proc_pca_2d=kwargs.get('proc_pca_2d'),
        proc_pca_3d=kwargs.get('proc_pca_3d'),
        proc_umap_2d=kwargs.get('proc_umap_2d'),
        proc_umap_3d=kwargs.get('proc_umap_3d'),
        umap_n_neighbors=kwargs.get('umap_n_neighbors'),
        umap_min_dist=kwargs.get('umap_min_dist'),
        umap_metric=kwargs.get('umap_metric'),
        aplicar_trevisan=kwargs.get('aplicar_trevisan'),
        algoritmo_clustering_pca=kwargs.get('algoritmo_clustering_pca'),
        algoritmo_clustering_umap=kwargs.get('algoritmo_clustering_umap'),
        modo_alineacion=kwargs.get('modo_alineacion'),
        pre_pct=kwargs.get('pre_pct'),
        post_pct=kwargs.get('post_pct'),
        canales_features=kwargs.get('canales_features'),
        ocultar_leyenda=kwargs.get('ocultar_leyenda'),
        estilo_visual=kwargs.get('estilo_visual'),
        ignorar_ventana_cero=kwargs.get('ignorar_ventana_cero')
    )
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
