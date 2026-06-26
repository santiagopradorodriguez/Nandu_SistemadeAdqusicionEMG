import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import umap
import warnings
warnings.filterwarnings('ignore')

# Configurar rutas para importar modulos hermanos
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) == "deep_learning":
    sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
    sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
    sys.path.append(os.path.dirname(deep_learning_dir))

# Importar funciones base del generador supervisado
from generador_umap_supervisado import extraer_features_concatenadas, procesar_mediciones

def get_session_id(toma_str):
    parts = toma_str.split('_')
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}" 
    return toma_str.split('_Win')[0]

def filtrar_datos(X, Y, Tomas, SNRs, snr_threshold=0.5, outlier_contamination=0.05):
    from sklearn.ensemble import IsolationForest
    X_clean, Y_clean, Tomas_clean = [], [], []
    for vocal in np.unique(Y):
        mask = Y == vocal
        X_vocal = X[mask]
        Tomas_vocal = np.array(Tomas)[mask]
        SNRs_vocal = np.array(SNRs)[mask]
        
        # 1. Filtro SNR
        valid_snr_mask = SNRs_vocal >= snr_threshold
        X_vocal_snr = X_vocal[valid_snr_mask]
        Tomas_vocal_snr = Tomas_vocal[valid_snr_mask]
        
        # 2. Filtro Isolation Forest
        if len(X_vocal_snr) > 5 and outlier_contamination > 0:
            iso = IsolationForest(contamination=outlier_contamination, random_state=42)
            preds = iso.fit_predict(X_vocal_snr)
            
            for i, is_inlier in enumerate(preds):
                if is_inlier == 1:
                    X_clean.append(X_vocal_snr[i])
                    Y_clean.append(vocal)
                    Tomas_clean.append(Tomas_vocal_snr[i])
        else:
            # Si no hay suficientes datos para Isolation Forest, se guardan los que pasaron el SNR
            X_clean.extend(X_vocal_snr)
            Y_clean.extend([vocal]*len(X_vocal_snr))
            Tomas_clean.extend(Tomas_vocal_snr)
            
    return np.array(X_clean), np.array(Y_clean), np.array(Tomas_clean)

def train_test_split_by_session(X, Y, Tomas):
    sesiones_base = [get_session_id(toma) for toma in Tomas]
    sesiones_unicas = list(set(sesiones_base))
    sesiones_unicas.sort()
    
    np.random.seed(42)
    np.random.shuffle(sesiones_unicas)
    
    train_size = int(0.8 * len(sesiones_unicas))
    train_sesiones = set(sesiones_unicas[:train_size])
    val_sesiones = set(sesiones_unicas[train_size:])
    
    train_indices = [i for i, s in enumerate(sesiones_base) if s in train_sesiones]
    test_indices = [i for i, s in enumerate(sesiones_base) if s in val_sesiones]
    
    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]

def evaluate_umap_knn(X_train, Y_train, X_test, Y_test, n_neighbors, min_dist, metric):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    Y_train_encoded = le.fit_transform(Y_train)
    
    # Ajuste de seguridad si hay muy pocas muestras
    n_neigh = min(n_neighbors, max(2, len(X_train) - 1))
    
    umap_model = umap.UMAP(n_neighbors=n_neigh, min_dist=min_dist, metric=metric, n_components=2, random_state=42)
    X_train_umap = umap_model.fit_transform(X_train, y=Y_train_encoded)
    X_test_umap = umap_model.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_umap, Y_train)
    Y_pred = knn.predict(X_test_umap)
    
    acc_global = accuracy_score(Y_test, Y_pred) * 100
    cm = confusion_matrix(Y_test, Y_pred, labels=['A','E','I','O','U'], normalize='true') * 100
    acc_a, acc_e, acc_i, acc_o, acc_u = np.diag(cm)
    min_acc = min(acc_a, acc_e, acc_i, acc_o, acc_u)
    
    return acc_global, acc_a, acc_e, acc_i, acc_o, acc_u, min_acc

def procesar_dsp_worker(smooth, tlen, base_dir, mediciones, combinaciones_umap):
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        X_raw, Y_raw, Tomas_raw, SNRs_raw = extraer_features_concatenadas(
            base_dir, mediciones, alpha_ruido=1.0, smooth_ms=smooth, notch_q=2.0, target_len=tlen
        )
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
    if len(X_raw) == 0:
        return []
        
    X_clean, Y_clean, Tomas_clean = filtrar_datos(np.array(X_raw), np.array(Y_raw), Tomas_raw, SNRs_raw, snr_threshold=0.5, outlier_contamination=0.05)
    X_train, X_test, Y_train, Y_test = train_test_split_by_session(X_clean, Y_clean, Tomas_clean)
    
    resultados = []
    for metric, nn, md in combinaciones_umap:
        acc_global, acc_a, acc_e, acc_i, acc_o, acc_u, min_acc = evaluate_umap_knn(X_train, Y_train, X_test, Y_test, n_neighbors=nn, min_dist=md, metric=metric)
        resultados.append({
            'smooth_ms': smooth,
            'target_len': tlen,
            'metric': metric,
            'n_neighbors': nn,
            'min_dist': md,
            'Accuracy_Global': acc_global,
            'Acc_A': acc_a, 'Acc_E': acc_e, 'Acc_I': acc_i, 'Acc_O': acc_o, 'Acc_U': acc_u,
            'Min_Acc_Vocal': min_acc,
            'Score_Ponderado': (min_acc * 0.7) + (acc_global * 0.3)
        })
    return resultados

def main():
    import itertools
    from tqdm import tqdm
    
    base_dir = r"C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\base_de_datos_electrodos"
    out_dir = r"C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\deep_learning\pca_umap_clustering\resultados_grid_search_2"
    os.makedirs(out_dir, exist_ok=True)
    
    mediciones = procesar_mediciones(base_dir)
    if not mediciones:
        print("No se encontraron mediciones.")
        return
        
    print("=========================================")
    print("  GRID SEARCH EXHAUSTIVO (Modo Dios)")
    print("=========================================")
    smooth_list = [50, 75, 90, 100, 125, 150]
    target_len_list = [20, 30, 40, 50, 80]
    
    n_neighbors_list = [5, 10, 15, 20, 30, 50]
    min_dist_list = [0.01, 0.1, 0.25, 0.5]
    metric_list = ['euclidean', 'correlation', 'cosine']
    
    combinaciones_dsp = list(itertools.product(smooth_list, target_len_list))
    combinaciones_umap = list(itertools.product(metric_list, n_neighbors_list, min_dist_list))
    
    total_combinaciones = len(combinaciones_dsp) * len(combinaciones_umap)
    print(f"Total de DSPs a extraer: {len(combinaciones_dsp)}")
    print(f"Total de UMAPs por DSP: {len(combinaciones_umap)}")
    print(f"Total absoluto de combinaciones a evaluar: {total_combinaciones}")
    print("=========================================\n")
    
    import concurrent.futures
    import multiprocessing
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    resultados_totales = []
    
    print(f"Lanzando {len(combinaciones_dsp)} DSPs en paralelo ({n_workers} cores)...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futuros = [executor.submit(procesar_dsp_worker, smooth, tlen, base_dir, mediciones, combinaciones_umap) for smooth, tlen in combinaciones_dsp]
        for f in tqdm(concurrent.futures.as_completed(futuros), total=len(futuros), desc="DSPs Procesados", unit="dsp"):
            res = f.result()
            if res:
                resultados_totales.extend(res)
            
    # Guardar resultados
    df_resultados = pd.DataFrame(resultados_totales)
    df_resultados.to_csv(os.path.join(out_dir, "resultados_exhaustivos_mododios.csv"), index=False)
    
    print("\n=========================================")
    print("  RESULTADOS FINALES DEL MODO DIOS")
    print("=========================================")
    if len(df_resultados) > 0:
        # Ordenar de mejor a peor según el score ponderado
        df_ranking = df_resultados.sort_values(by='Score_Ponderado', ascending=False).reset_index(drop=True)
        
        print("\n--- TOP 5 MEJORES CONFIGURACIONES ---")
        for i in range(min(5, len(df_ranking))):
            mejor = df_ranking.iloc[i]
            print(f"\n[{i+1}] Score Ponderado: {mejor['Score_Ponderado']:.2f}")
            print(f"    DSP  -> Envolvente: {int(mejor['smooth_ms'])}ms | Remuestreo: {int(mejor['target_len'])}")
            print(f"    UMAP -> {mejor['metric']} | vecinos: {mejor['n_neighbors']} | min_dist: {mejor['min_dist']}")
            print(f"    [+] Accuracy Global: {mejor['Accuracy_Global']:.2f}% | Peor Vocal: {mejor['Min_Acc_Vocal']:.2f}%")
            print(f"    [+] Detalle: A={mejor['Acc_A']:.1f}%, E={mejor['Acc_E']:.1f}%, I={mejor['Acc_I']:.1f}%, O={mejor['Acc_O']:.1f}%, U={mejor['Acc_U']:.1f}%")
            
        print("\nTodos los resultados fueron guardados en 'resultados_exhaustivos_mododios.csv'.")
    else:
        print("Error: No se generaron resultados válidos.")

if __name__ == "__main__":
    main()
