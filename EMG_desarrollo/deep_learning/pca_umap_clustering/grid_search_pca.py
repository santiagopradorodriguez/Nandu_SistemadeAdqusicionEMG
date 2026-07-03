import os
import sys
import itertools
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import time

# Importamos las funciones del generador actual
from generador_pca_umap import procesar_mediciones, extraer_features_concatenadas

def evaluate_triplet(X, Y, vocales_unicas):
    """
    Evalúa una matriz X usando K-Means y devuelve el Accuracy 
    (usando asignación húngara) de forma completamente silenciosa.
    """
    n_clases = len(vocales_unicas)
    kmeans = KMeans(n_clusters=n_clases, random_state=42, n_init=10)
    y_pred_kmeans = kmeans.fit_predict(X)
    
    y_true_int = np.array([vocales_unicas.index(v) for v in Y])
    cm = confusion_matrix(y_true_int, y_pred_kmeans)
    
    row_ind, col_ind = linear_sum_assignment(-cm)
    total_correctos = cm[row_ind, col_ind].sum()
    accuracy = (total_correctos / len(Y)) * 100
    
    return accuracy

def main():
    print("=" * 60)
    print("GRID SEARCH DE COMPONENTES PCA (Búsqueda Exhaustiva 3D)")
    print("=" * 60)
    
    # Configuramos el directorio base asumiendo la estructura del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "base_de_datos_electrodos"))
    
    if not os.path.exists(base_dir):
        print(f"[ERROR] Directorio base no encontrado: {base_dir}")
        return
        
    print("[*] Buscando mediciones en la base de datos...")
    mediciones = procesar_mediciones(base_dir)
    
    if not mediciones:
        print("[ERROR] No se encontraron mediciones válidas.")
        return
        
    print(f"[*] Se encontraron {len(mediciones)} mediciones. Extrayendo features con hiperparámetros óptimos (smooth=90ms, target_len=20)...")
    
    # Extraemos las características con los hiperparámetros ganadores
    X, Y, Tomas, SNRs = extraer_features_concatenadas(
        base_dir=base_dir, 
        mediciones=mediciones, 
        alpha_ruido=1.0, 
        smooth_ms=90, 
        notch_q=2.0, 
        target_len=20
    )
    
    if len(X) == 0:
        print("[ERROR] Falló la extracción de características.")
        return
        
    X = np.array(X)
    Y = np.array(Y)
    Tomas = np.array(Tomas)
    SNRs = np.array(SNRs)
    
    # ------------------ FILTRADO DE OUTLIERS ------------------
    # Replicamos el filtro IsolationForest de generador_pca_umap para que los datos sean idénticos
    from sklearn.ensemble import IsolationForest
    
    print("\n[*] Aplicando filtro de Outliers (Isolation Forest, contaminación=0.05)...")
    X_clean, Y_clean, Tomas_clean = [], [], []
    outlier_contamination = 0.05
    snr_threshold = 0.5
    
    for vocal in np.unique(Y):
        mask = Y == vocal
        X_vocal = X[mask]
        Tomas_vocal = Tomas[mask]
        SNRs_vocal = SNRs[mask]
        
        valid_snr_mask = SNRs_vocal >= snr_threshold
        X_vocal_snr = X_vocal[valid_snr_mask]
        Tomas_vocal_snr = Tomas_vocal[valid_snr_mask]
        
        if len(X_vocal_snr) > 5 and outlier_contamination > 0:
            iso = IsolationForest(contamination=outlier_contamination, random_state=42)
            preds = iso.fit_predict(X_vocal_snr)
            for i, is_inlier in enumerate(preds):
                if is_inlier == 1:
                    X_clean.append(X_vocal_snr[i])
                    Y_clean.append(vocal)
                    Tomas_clean.append(Tomas_vocal_snr[i])
        else:
            for i in range(len(X_vocal_snr)):
                X_clean.append(X_vocal_snr[i])
                Y_clean.append(vocal)
                Tomas_clean.append(Tomas_vocal_snr[i])
                
    X_scaled = np.array(X_clean)
    Y_clean = np.array(Y_clean)
    vocales_unicas = sorted(list(set(Y_clean)))
    
    print(f"  -> Quedaron {len(X_scaled)} repeticiones válidas de las {len(X)} originales.")
    print("\n[*] Extracción y limpieza finalizada. Calculando PCA base...")
    
    # Calcular PCA con las 15 primeras componentes
    top_n_components = 15
    # Asegurarnos de no pedir más componentes que el mínimo entre muestras y dimensiones
    n_samples, n_features = X_scaled.shape
    max_components = min(n_samples, n_features, top_n_components)
    
    pca_base = PCA(n_components=max_components)
    X_pca_base = pca_base.fit_transform(X_scaled)
    varianzas = pca_base.explained_variance_ratio_
    
    # Generar todas las combinaciones de a 3 (índices 0-based)
    indices = list(range(max_components))
    tripletas = list(itertools.combinations(indices, 3))
    
    print(f"[*] Evaluando {len(tripletas)} tripletas diferentes de componentes PCA usando K-Means...")
    print("    (Esto puede tomar unos segundos. Por favor espera...)\n")
    
    resultados = []
    
    start_time = time.time()
    
    for idx, (c1, c2, c3) in enumerate(tripletas):
        # Extraemos las columnas de la matriz base para armar nuestra proyección 3D de prueba
        X_test = X_pca_base[:, [c1, c2, c3]]
        
        # Evaluamos
        acc = evaluate_triplet(X_test, Y_clean, vocales_unicas)
        
        # Calculamos varianza explicada total de esta tripleta
        var_expl = (varianzas[c1] + varianzas[c2] + varianzas[c3]) * 100
        
        # Guardamos usando índices 1-based (índice humano) para el reporte
        resultados.append({
            "comps": (c1+1, c2+1, c3+1),
            "acc": acc,
            "var_expl": var_expl
        })
        
        # Simple barra de progreso visual en consola
        if (idx + 1) % 50 == 0 or (idx + 1) == len(tripletas):
            progress = ((idx + 1) / len(tripletas)) * 100
            print(f"  -> Progreso: {progress:.1f}% ({idx+1}/{len(tripletas)})", end='\r')
            
    elapsed = time.time() - start_time
    print(f"\n\n[*] Búsqueda finalizada en {elapsed:.2f} segundos.")
    
    # Ordenar de mayor a menor accuracy
    resultados.sort(key=lambda x: x["acc"], reverse=True)
    
    print("\n" + "=" * 50)
    print(" TOP 10 MEJORES COMBINACIONES PCA (3D)")
    print("=" * 50)
    print(f"{'Puesto':<8} | {'Componentes':<15} | {'Accuracy K-Means':<18} | {'Varianza Explicada'}")
    print("-" * 50)
    
    for i in range(min(10, len(resultados))):
        res = resultados[i]
        comps_str = f"[{res['comps'][0]}, {res['comps'][1]}, {res['comps'][2]}]"
        print(f"#{i+1:<7} | {comps_str:<15} | {res['acc']:.2f}%{'':<11} | {res['var_expl']:.2f}%")
        
    print("=" * 50)
    
    print("\n[INFO] Nota: Las componentes estándar [1, 2, 3] priorizan capturar la mayor varianza posible,")
    print("       mientras que componentes menores podrían discriminar mejor ruido o firmas sutiles,")
    print("       pero con un costo en la varianza explicada del modelo general.")

if __name__ == "__main__":
    main()
