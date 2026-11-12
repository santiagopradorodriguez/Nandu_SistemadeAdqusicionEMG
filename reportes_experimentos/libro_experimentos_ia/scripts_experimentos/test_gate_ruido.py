import sys
import numpy as np

# Probar diferentes anchos de ventana y umbrales de gate de ruido sobre las series
# para ver cómo maximizan la precisión LDA sin cortar la rampa motora prefonatoria
from generador_multimodal_cinetico_formantes import procesar_sesion_multimodal, evaluar_pca

s_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-11-09/SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1"

print(f"{'Pre (s)':<8} | {'Post (s)':<8} | {'Gate (mm)':<10} | {'Var 3D':<8} | {'LDA 6ch':<8} | {'LDA 3ch':<8} | {'/a/':<5} | {'/e/':<5} | {'/i/':<5} | {'/o/':<5} | {'/u/':<5}")
print("-" * 85)

for pre_s, post_s in [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.50, 0.60)]:
    for gate_mm in [0.0, 0.15, 0.25]:
        X3, X6, y, pulsos, df_f, _, _ = procesar_sesion_multimodal(
            s_dir, pre_s=pre_s, post_s=post_s, outlier_contamination=0.08
        )
        
        # Aplicar gate de ruido: si el pico del canal en la ventana no supera el gate, se apaga a cero
        # X6 tiene 6 canales x 25 puntos = 150 features
        if gate_mm > 0:
            X6_g = X6.copy()
            for i in range(X6_g.shape[0]):
                for ch_i in range(6):
                    seg_ch = X6_g[i, ch_i*25 : (ch_i+1)*25]
                    if np.max(seg_ch) < gate_mm:
                        X6_g[i, ch_i*25 : (ch_i+1)*25] = 0.0
        else:
            X6_g = X6
            
        _, var6, lda6, acc6, _ = evaluar_pca(X6_g, y)
        _, var3, lda3, _, _ = evaluar_pca(X3, y)
        
        print(f"{pre_s:<8.2f} | {post_s:<8.2f} | {gate_mm:<10.2f} | {np.sum(var6):<8.1f} | {lda6:<8.1f} | {lda3:<8.1f} | {acc6['A']:<5.1f} | {acc6['E']:<5.1f} | {acc6['I']:<5.1f} | {acc6['O']:<5.1f} | {acc6['U']:<5.1f}")
