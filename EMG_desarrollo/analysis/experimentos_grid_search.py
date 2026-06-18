import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import warnings
warnings.filterwarnings("ignore")

# Añadimos el directorio base al path temporal para importar fácilmente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generador_pca_umap import procesar_mediciones, extraer_features_concatenadas, evaluar_clustering_no_supervisado

# ----------------- CONFIGURACIÓN DEL EXPERIMENTO -----------------
# Poner TEST_MODE = False para la ejecución real masiva
TEST_MODE = False

if TEST_MODE:
    print("!!! MODO PRUEBA ACTIVADO !!! (Valores reducidos para testear el script)")
    SWEEP_SMOOTH = [100, 250]
    SWEEP_TARGET_LEN = [50, 100]
    SWEEP_UMAP_NN = [5, 15]
    SWEEP_UMAP_MD = [0.1, 0.5]
    SWEEP_UMAP_METRIC = ['euclidean', 'cosine']
else:
    print("=== MODO PRODUCCIÓN ACTIVADO ===")
    SWEEP_SMOOTH = [30, 60, 90, 120, 150, 180, 210, 240]
    SWEEP_TARGET_LEN = [20, 50, 100, 150, 200]
    SWEEP_UMAP_NN = [2, 5, 10, 15, 20]
    SWEEP_UMAP_MD = [0.1, 0.3, 0.5, 0.8]
    SWEEP_UMAP_METRIC = ['euclidean', 'cosine', 'manhattan', 'correlation']

# Parámetros fijos
ALPHA_RUIDO = 1.0
NOTCH_Q = 2.0  # Elegido por el usuario
BEST_SMOOTH = 150 # Default seguro
BEST_TARGET_LEN = 100

def print_progress_bar(iteration, total, start_time, prefix='', length=40):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    
    elapsed = time.time() - start_time
    if iteration > 0:
        eta = elapsed * (total / iteration) - elapsed
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
    else:
        eta_str = "--:--:--"
        
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% Completado | ETA: {eta_str}')
    sys.stdout.flush()
    if iteration == total:
        print()

def main():
    global BEST_SMOOTH, BEST_TARGET_LEN
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(script_dir), "base_de_datos_electrodos")
    out_dir = os.path.join(script_dir, "resultados_experimentos")
    os.makedirs(out_dir, exist_ok=True)
    
    mediciones = procesar_mediciones(base_dir)
    if not mediciones:
        print("No se encontraron mediciones.")
        return
        
    if TEST_MODE:
        # Tomar solo 4 mediciones para test rápido
        mediciones = mediciones[:4]

    print(f"\nSe procesarán {len(mediciones)} mediciones en total.")
    
    # Redirigir stdout para que 'extraer_features_concatenadas' no ensucie la barra
    import io
    old_stdout = sys.stdout
    
    # =================================================================
    # FASE 1A: Sweep de Smooth MS (Envolvente)
    # =================================================================
    print("\n" + "="*50)
    print("FASE 1A: Barrido de Envolvente (smooth_ms)")
    print(f" -> Variando: smooth_ms en {SWEEP_SMOOTH}")
    print(f" -> Fijos (Default): target_len={BEST_TARGET_LEN}, notch_q={NOTCH_Q}, alpha={ALPHA_RUIDO}")
    print("="*50)
    
    res_smooth_ms = []
    res_smooth_acc = []
    res_smooth_sil = []
    res_smooth_acc_vocales = []
    vocs_unicas = []
    
    start_time = time.time()
    for i, s_ms in enumerate(SWEEP_SMOOTH):
        print_progress_bar(i, len(SWEEP_SMOOTH), start_time, prefix=f'Evaluando smooth={s_ms}ms')
        
        # 1. Extracción DSP (Se mostrarán los prints de cada toma)
        X, Y, _, _ = extraer_features_concatenadas(
            base_dir, mediciones, alpha_ruido=ALPHA_RUIDO, 
            smooth_ms=s_ms, notch_q=NOTCH_Q, target_len=BEST_TARGET_LEN
        )
        
        if len(X) < 5: continue
        
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X)
        
        try: sil_3d = silhouette_score(X_pca_3d, Y, metric='euclidean')
        except: sil_3d = 0.0
            
        sys.stdout = io.StringIO()
        acc_3d, acc_voc, vocs_unicas = evaluar_clustering_no_supervisado(X_pca_3d, Y, f"PCA_smooth_{s_ms}")
        sys.stdout = old_stdout
        
        res_smooth_ms.append(s_ms)
        res_smooth_acc.append(acc_3d)
        res_smooth_sil.append(sil_3d)
        res_smooth_acc_vocales.append(acc_voc)
        
    print_progress_bar(len(SWEEP_SMOOTH), len(SWEEP_SMOOTH), start_time, prefix='Fase 1A Finalizada')
    
    if res_smooth_ms:
        best_idx = np.argmax(res_smooth_acc)
        BEST_SMOOTH = res_smooth_ms[best_idx]
        print(f"\n=> MEJOR smooth_ms ENCONTRADO: {BEST_SMOOTH}ms (Acc: {res_smooth_acc[best_idx]:.2f}%)")
        
        # Graficar Fase 1A
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        ax1.plot(res_smooth_ms, res_smooth_acc, 'g-o', label='Accuracy (%)')
        ax2.plot(res_smooth_ms, res_smooth_sil, 'b-s', label='Silhouette Score')
        
        ax1.set_xlabel('Envolvente (smooth_ms)')
        ax1.set_ylabel('Accuracy (%)', color='g')
        ax2.set_ylabel('Silhouette Score', color='b')
        plt.title('Impacto de la Envolvente en el Clustering (PCA 3D)')
        
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "exp1a_smooth_vs_metrics.png"))
        plt.close()
        
        # Graficar Accuracy por Vocal
        fig, ax = plt.subplots(figsize=(10, 6))
        acc_voc_array = np.array(res_smooth_acc_vocales)
        for i, v in enumerate(vocs_unicas):
            ax.plot(res_smooth_ms, acc_voc_array[:, i], marker='o', label=f'Vocal {v}')
        ax.set_xlabel('Envolvente (smooth_ms)')
        ax.set_ylabel('Accuracy por Vocal (%)')
        plt.title('Accuracy por Vocal vs Envolvente (PCA 3D)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "exp1a_smooth_vs_vowels.png"))
        plt.close()

    # =================================================================
    # FASE 1B: Sweep de Target Length (Remuestreo)
    # =================================================================
    print("\n" + "="*50)
    print("FASE 1B: Barrido de Remuestreo (target_len)")
    print(f" -> Variando: target_len en {SWEEP_TARGET_LEN}")
    print(f" -> Fijos (Ganador Fase 1A): smooth_ms={BEST_SMOOTH}ms")
    print(f" -> Fijos (Default): notch_q={NOTCH_Q}, alpha={ALPHA_RUIDO}")
    print("="*50)
    
    res_len_val = []
    res_len_acc = []
    res_len_sil = []
    res_len_acc_vocales = []
    
    start_time = time.time()
    for i, t_len in enumerate(SWEEP_TARGET_LEN):
        print_progress_bar(i, len(SWEEP_TARGET_LEN), start_time, prefix=f'Evaluando target_len={t_len}')
        
        # Se mostrarán los prints de cada toma
        X, Y, _, _ = extraer_features_concatenadas(
            base_dir, mediciones, alpha_ruido=ALPHA_RUIDO, 
            smooth_ms=BEST_SMOOTH, notch_q=NOTCH_Q, target_len=t_len
        )
        
        if len(X) < 5: continue
        
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X)
        
        try: sil_3d = silhouette_score(X_pca_3d, Y, metric='euclidean')
        except: sil_3d = 0.0
            
        sys.stdout = io.StringIO()
        acc_3d, acc_voc, vocs_unicas = evaluar_clustering_no_supervisado(X_pca_3d, Y, f"PCA_targetlen_{t_len}")
        sys.stdout = old_stdout
        
        res_len_val.append(t_len)
        res_len_acc.append(acc_3d)
        res_len_sil.append(sil_3d)
        res_len_acc_vocales.append(acc_voc)
        
    print_progress_bar(len(SWEEP_TARGET_LEN), len(SWEEP_TARGET_LEN), start_time, prefix='Fase 1B Finalizada')

    if res_len_val:
        best_idx = np.argmax(res_len_acc)
        BEST_TARGET_LEN = res_len_val[best_idx]
        print(f"\n=> MEJOR target_len ENCONTRADO: {BEST_TARGET_LEN} pts (Acc: {res_len_acc[best_idx]:.2f}%)")
        
        # Graficar Fase 1B
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        ax1.plot(res_len_val, res_len_acc, 'g-o', label='Accuracy (%)')
        ax2.plot(res_len_val, res_len_sil, 'b-s', label='Silhouette Score')
        
        ax1.set_xlabel('Puntos de Remuestreo (target_len)')
        ax1.set_ylabel('Accuracy (%)', color='g')
        ax2.set_ylabel('Silhouette Score', color='b')
        plt.title('Impacto del Remuestreo en el Clustering (PCA 3D)')
        
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "exp1b_targetlen_vs_metrics.png"))
        plt.close()

        # Graficar Accuracy por Vocal
        fig, ax = plt.subplots(figsize=(10, 6))
        acc_voc_array = np.array(res_len_acc_vocales)
        for i, v in enumerate(vocs_unicas):
            ax.plot(res_len_val, acc_voc_array[:, i], marker='o', label=f'Vocal {v}')
        ax.set_xlabel('Puntos de Remuestreo (target_len)')
        ax.set_ylabel('Accuracy por Vocal (%)')
        plt.title('Accuracy por Vocal vs Remuestreo (PCA 3D)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "exp1b_targetlen_vs_vowels.png"))
        plt.close()

    # =================================================================
    # FASE 2: Sweep UMAP (Matemática pura en Memoria)
    # =================================================================
    print("\n" + "="*50)
    print("FASE 2: Barrido UMAP en Memoria")
    print(f" -> Variando: metric, n_neighbors, min_dist")
    print(f" -> Fijos (Ganadores Fase 1): smooth_ms={BEST_SMOOTH}ms, target_len={BEST_TARGET_LEN}")
    print(f" -> Fijos (Default): notch_q={NOTCH_Q}, alpha={ALPHA_RUIDO}")
    print("="*50)
    
    X_best, Y_best, _, _ = extraer_features_concatenadas(
        base_dir, mediciones, alpha_ruido=ALPHA_RUIDO, 
        smooth_ms=BEST_SMOOTH, notch_q=NOTCH_Q, target_len=BEST_TARGET_LEN
    )
    
    if len(X_best) < 5:
        print("Error: No se obtuvieron datos suficientes con la mejor configuración.")
        return
        
    umap_results = []
    total_umap_tests = len(SWEEP_UMAP_NN) * len(SWEEP_UMAP_MD) * len(SWEEP_UMAP_METRIC)
    count = 0
    start_time = time.time()
    
    for metric in SWEEP_UMAP_METRIC:
        for nn in SWEEP_UMAP_NN:
            for md in SWEEP_UMAP_MD:
                print_progress_bar(count, total_umap_tests, start_time, prefix=f'UMAP m:{metric} nn:{nn} md:{md}')
                
                real_nn = min(nn, len(X_best) - 1) if len(X_best) > 1 else 2
                reducer = umap.UMAP(n_neighbors=real_nn, min_dist=md, metric=metric, n_components=3, random_state=42)
                X_umap = reducer.fit_transform(X_best)
                
                try: sil = silhouette_score(X_umap, Y_best, metric='euclidean')
                except: sil = 0.0
                
                sys.stdout = io.StringIO()
                acc, acc_voc, vocs_unicas = evaluar_clustering_no_supervisado(X_umap, Y_best, f"UMAP_{metric}_{nn}_{md}")
                sys.stdout = old_stdout
                
                row = {
                    'metric': metric,
                    'n_neighbors': nn,
                    'min_dist': md,
                    'accuracy': acc,
                    'silhouette': sil
                }
                for v_idx, v in enumerate(vocs_unicas):
                    row[f'acc_vocal_{v}'] = acc_voc[v_idx]
                    
                umap_results.append(row)
                count += 1
                
    print_progress_bar(total_umap_tests, total_umap_tests, start_time, prefix='Fase 2 Finalizada')
    
    df_umap = pd.DataFrame(umap_results)
    df_umap.to_csv(os.path.join(out_dir, "exp2_umap_raw_results.csv"), index=False)
    
    best_umap = df_umap.loc[df_umap['accuracy'].idxmax()]
    print(f"\n=> EL MEJOR UMAP ESTÁ EN: Métrica={best_umap['metric']}, n_neighbors={best_umap['n_neighbors']}, min_dist={best_umap['min_dist']}")
    print(f"=> UMAP ACCURACY MÁXIMO LOGRADO: {best_umap['accuracy']:.2f}% (Silhouette: {best_umap['silhouette']:.4f})")
    
    for metric in SWEEP_UMAP_METRIC:
        df_metric = df_umap[df_umap['metric'] == metric]
        if df_metric.empty: continue
        
        pivot = df_metric.pivot(index='n_neighbors', columns='min_dist', values='accuracy')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
        plt.title(f"UMAP Accuracy Heatmap (Metric: {metric})")
        plt.xlabel("min_dist")
        plt.ylabel("n_neighbors")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"exp2_umap_heatmap_{metric}.png"))
        plt.close()

    print(f"\n=======================================================")
    print(f"EXPERIMENTO COMPLETADO CON ÉXITO.")
    print(f"Todos los gráficos guardados en: {out_dir}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
