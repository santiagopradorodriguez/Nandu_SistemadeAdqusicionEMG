import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.signal import resample
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
    SWEEP_SMOOTH = [30, 60, 90, 120, 150, 180, 210, 240, 250]
    SWEEP_TARGET_LEN = [20, 50, 100, 150, 200]
    SWEEP_UMAP_NN = [2, 5, 10, 15, 20]
    SWEEP_UMAP_MD = [0.1, 0.3, 0.5, 0.8]
    SWEEP_UMAP_METRIC = ['euclidean', 'cosine', 'manhattan', 'correlation']

# Parámetros fijos (idénticos a los defaults de la GUI)
ALPHA_RUIDO = 1.0
NOTCH_Q = 2.0  # Elegido por el usuario
SNR_THRESHOLD = 0.5
OUTLIER_CONTAMINATION = 0.05
BEST_SMOOTH = 150 # Default seguro
BEST_TARGET_LEN = 100

def limpiar_datos(X, Y, Tomas, SNRs):
    """Aplica filtro SNR + IsolationForest, idéntico a ejecutar_procesamiento de la GUI."""
    from sklearn.ensemble import IsolationForest
    X_clean, Y_clean, Tomas_clean = [], [], []
    for vocal in np.unique(Y):
        mask = Y == vocal
        X_v = X[mask]
        Tomas_v = Tomas[mask]
        SNRs_v = SNRs[mask]
        # 1. Filtro SNR
        valid = SNRs_v >= SNR_THRESHOLD
        X_v = X_v[valid]
        Tomas_v = Tomas_v[valid]
        # 2. IsolationForest
        if len(X_v) > 5 and OUTLIER_CONTAMINATION > 0:
            iso = IsolationForest(contamination=OUTLIER_CONTAMINATION, random_state=42)
            preds = iso.fit_predict(X_v)
            for j, p in enumerate(preds):
                if p == 1:
                    X_clean.append(X_v[j])
                    Y_clean.append(vocal)
                    Tomas_clean.append(Tomas_v[j])
        else:
            for j in range(len(X_v)):
                X_clean.append(X_v[j])
                Y_clean.append(vocal)
                Tomas_clean.append(Tomas_v[j])
    return np.array(X_clean), np.array(Y_clean), np.array(Tomas_clean)

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
    # FASE 1: Barrido Fino 2D (smooth_ms vs target_len)
    # =================================================================
    print("\n" + "="*50)
    print("FASE 1: Barrido Fino 2D (smooth_ms vs target_len)")
    print(f" -> Variando: smooth_ms en {SWEEP_SMOOTH}")
    print(f" -> Variando: target_len en {SWEEP_TARGET_LEN}")
    print(f" -> Fijos (Default): notch_q={NOTCH_Q}, alpha={ALPHA_RUIDO}")
    print("="*50)
    
    resultados_2d_acc = np.zeros((len(SWEEP_SMOOTH), len(SWEEP_TARGET_LEN)))
    resultados_2d_sil = np.zeros((len(SWEEP_SMOOTH), len(SWEEP_TARGET_LEN)))
    
    total_fase1 = len(SWEEP_SMOOTH) * len(SWEEP_TARGET_LEN)
    count_fase1 = 0
    start_time = time.time()
    
    # IMPORTANTE: UMAP es estocástico, PCA es determinista. 
    # Usaremos UMAP para esta evaluación fina 2D ya que es nuestro clasificador final.
    from sklearn.preprocessing import StandardScaler
    
    for i, s_ms in enumerate(SWEEP_SMOOTH):
        for j, t_len in enumerate(SWEEP_TARGET_LEN):
            print_progress_bar(count_fase1, total_fase1, start_time, prefix=f'Fase 1: {s_ms}ms | {t_len}pts')
            
            # 1. Extracción DSP
            sys.stdout = io.StringIO() # Silenciar prints de la extracción
            X, Y, Tomas_tmp, SNRs_tmp = extraer_features_concatenadas(
                base_dir, mediciones, alpha_ruido=ALPHA_RUIDO, 
                smooth_ms=s_ms, notch_q=NOTCH_Q, target_len=t_len
            )
            sys.stdout = old_stdout
            
            if len(X) < 5:
                count_fase1 += 1
                continue
                
            X_clean, Y_clean, _ = limpiar_datos(X, Y, Tomas_tmp, SNRs_tmp)
            if len(X_clean) < 10:
                count_fase1 += 1
                continue
            
            # 2. Reducir con PCA (a pedido del usuario)
            pca_3d = PCA(n_components=3)
            X_pca_3d = pca_3d.fit_transform(X_clean)
            
            # 3. Métricas
            try: sil_3d = silhouette_score(X_pca_3d, Y_clean, metric='euclidean')
            except: sil_3d = 0.0
                
            sys.stdout = io.StringIO()
            acc_3d, _, _ = evaluar_clustering_no_supervisado(X_pca_3d, Y_clean, f"PCA_2D_{s_ms}_{t_len}")
            sys.stdout = old_stdout
            
            resultados_2d_acc[i, j] = acc_3d
            resultados_2d_sil[i, j] = sil_3d
            count_fase1 += 1
            
    print_progress_bar(total_fase1, total_fase1, start_time, prefix='Fase 1 Finalizada')
    
    # Encontrar el mejor valor global
    best_idx = np.unravel_index(np.argmax(resultados_2d_acc), resultados_2d_acc.shape)
    BEST_SMOOTH = SWEEP_SMOOTH[best_idx[0]]
    BEST_TARGET_LEN = SWEEP_TARGET_LEN[best_idx[1]]
    
    print(f"\n=> MEJOR COMBINACIÓN ENCONTRADA: smooth_ms={BEST_SMOOTH}ms | target_len={BEST_TARGET_LEN}pts (Acc: {resultados_2d_acc[best_idx]:.2f}%)")
    
    # Guardar CSV y Plot
    df_acc_2d = pd.DataFrame(resultados_2d_acc, index=[f"{ms}ms" for ms in SWEEP_SMOOTH], columns=[f"{pt}pts" for pt in SWEEP_TARGET_LEN])
    df_acc_2d.to_csv(os.path.join(out_dir, "exp1_grid_2d_accuracy.csv"))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_acc_2d, annot=True, fmt=".1f", cmap="magma", cbar_kws={'label': 'Accuracy PCA (%)'})
    plt.title("Grid Search 2D Acoplado: Accuracy Topológica (PCA 3D)\nEnvolvente (Y) vs Puntos de Remuestreo (X)")
    plt.ylabel("Envolvente (smooth_ms)")
    plt.xlabel("Puntos de Remuestreo (target_length)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp1_heatmap_smooth_vs_target.png"))
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
    
    X_best, Y_best, Tomas_best, SNRs_best = extraer_features_concatenadas(
        base_dir, mediciones, alpha_ruido=ALPHA_RUIDO, 
        smooth_ms=BEST_SMOOTH, notch_q=NOTCH_Q, target_len=BEST_TARGET_LEN
    )
    
    if len(X_best) < 5:
        print("Error: No se obtuvieron datos suficientes con la mejor configuración.")
        return
    
    X_best, Y_best, _ = limpiar_datos(X_best, Y_best, Tomas_best, SNRs_best)
    
    if len(X_best) < 5:
        print("Error: No quedaron datos suficientes tras filtrar outliers.")
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
