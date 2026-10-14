# Paquete de Scripts Experimentales: Libro de Experimentos IA

Este paquete agrupa de manera sistemática los 61 scripts de prueba, barrido de hiperparámetros y verificación empírica desarrollados durante la investigación y formalización del proyecto Ñandú EMG.

Cada script se corresponde directamente con los capítulos del documento maestro [`libro_de_experimentos_ia.tex`](../libro_de_experimentos_ia.tex):

## Capítulo 1: Decodificación Vocálica No Supervisada y Variedades Motoras
- `experimento_autoencoder_trisujeto_3d.py`: Modelo trisujeto latente 3D canónico SO(3).
- `experimento_autoencoder_4d.py`: Ensayo comparativo del espacio latente 4D frente a 3D.
- `evaluar_petra_4d.py`: Evaluación específica de transferencia en Petra bajo 4D.
- `experimento_test_extremo_petra_3d.py`: Prueba de estrés con electrodos de silicona en dos sesiones temporales.
- `experimento_1_petra_interdia_cca.py` & `experimento_1_petra_sobre_espacio_lucas.py`: Análisis de CCA y alineación inter-día.
- `experimento_calibracion_ganancia_relativa.py` & `experimento_ganancia_trisujeto_holdout.py`: Calibración dinámica por ganancia escalar.
- `experimento_evaluar_lda_y_clasificadores.py`: Evaluación externa con LDA y k-NN sobre la variedad latente.
- `experimento_test_holdout_multisujeto.py`: Validación ciega sobre particiones de test holdout.
- `generar_matrices_confusion_petra_dos_dias.py`: Generación de matrices de confusión inter-día.

## Capítulo 2: Arquitecturas Convolucionales y Gate Doble en Tiempo Real
- `grid_search_conv_ortogonal.py` & `grid_search_conv_ortogonal_3d.py`: Búsqueda sistemática de regularización ortogonal y topología convolucional.
- `experimento_candela_conv_ortogonal.py`: Entrenamiento y verificación del modelo convolucional ortogonal.
- `experimento_candela_perdida_compuesta.py` & `experimento_perdida_compuesta_tkeo.py`: Función de pérdida compuesta (reconstrucción + energía Teager-Kaiser).
- `experimento_resolucion_temporal.py`: Análisis de sensibilidad temporal y longitud de ventana.
- `experimento_rms_tkeo_lucas.py` & `experimento_rms_vs_tkeo_riguroso.py`: Comparativa rigurosa entre descriptores RMS y TKEO.
- `experimento_readin_campeon_87.py`, `experimento_readin_campeon_87_reales.py`, `experimento_readin_semisupervisado_campeon.py`, `experimento_readin_tronco_congelado.py`: Arquitecturas Read-in y transferencia con capas adaptativas.

## Capítulo 3: Variedad Motora Tridimensional Pseudo-HD de 12 Canales y Codo de Tenenbaum
- `experimento_pseudo_hd_candela_multicanal_20pts.py` & `experimento_pseudo_hd_candela_20pts_impedancia.py`: Ensamblado del súper-vector 240D.
- `experimento_montecarlo_1000_candela_12ch.py`: Ensamble estocástico Monte Carlo N=1000 con perturbación gaussiana.
- `evaluar_gmm_y_curva_varianza_1000.py`: Cuantificación de varianza residual y codo geodésico en d=3.
- `barrido_k_isomap_3d_lucas.py` & `barrido_k_isomap_lucas.py`: Sensibilidad de vecindad k en Isomap.
- `curva_varianza_pca_isomap_candela.py`: Comparación de varianza explicada PCA vs Isomap.
- `graficar_isomap_2d_k35_candela.py` & `graficar_pca_isomap_3d_candela_13ch.py`: Generación de figuras latentes de alta dimensión.

## Capítulo 4: Formalización Matemática de la Variedad Unificada
- `experimento_pca_sara_solla_tri_sujeto.py` & `experimento_pca_sara_solla_inter_sesiones_por_sujeto.py`: Formalización de variedades neurales compartidas (Sara Solla).
- `experimento_fisica_teorica_multisujeto.py`: Demostración analítica de invariantes de Lie SO(3) y Procrustes generalizado.
- `compilar_dataset_multisujeto_canonica.py` & `comparativa_canonica_tri_sujeto_sin_clustering.py`: Estandarización de base canónica multisujeto.
