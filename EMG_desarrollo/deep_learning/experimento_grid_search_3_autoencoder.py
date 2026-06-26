import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

# Agregar rutas relativas para importar módulos hermanos
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir_abs) == "deep_learning":
    sys.path.append(os.path.join(script_dir_abs, "dataset_tools"))
    base_repo_dir = os.path.dirname(script_dir_abs)
else:
    base_repo_dir = script_dir_abs

import generador_pca_tensorial as gpt
import train_autoencoder as ta

# Configuraciones base
EPOCHS = 140
BATCH_SIZE = 32
LATENT_DIM = 16
FORCE_EPOCHS = False  # Utilizamos Checkpointing (ahora basado en Accuracy)

# Grillas de búsqueda ampliadas
smooth_ms_grid = [50, 80, 120, 150, 180, 220, 250]
target_length_grid = [40, 60, 80, 100, 120, 150]
alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

out_dir = os.path.join(base_repo_dir, "resultados")
os.makedirs(out_dir, exist_ok=True)
csv_out_fase1 = os.path.join(out_dir, "resultados_grid_search_3_fase1.csv")
csv_out_fase2 = os.path.join(out_dir, "resultados_grid_search_3_fase2.csv")

def worker_fase1(s_ms, t_len, mediciones_dia7):
    # Proceso aislado para Fase 1
    try:
        X, Y, Tomas, _ = gpt.extraer_features_concatenadas(
            base_dir=os.path.join(base_repo_dir, "base_de_datos_electrodos"),
            mediciones=mediciones_dia7,
            alpha_ruido=1.0,
            smooth_ms=s_ms,
            notch_q=2.0,
            target_length=t_len,
            use_manual_exclusions=True,
            verbose=False
        )
        
        # Archivo temporal UNICO por hilo para evitar colisiones
        temp_csv = os.path.join(out_dir, f"temp_grid_fase1_{s_ms}_{t_len}.csv")
        X_flat = X.reshape(X.shape[0], -1)
        df = pd.DataFrame(X_flat)
        df.insert(0, "Vocal", Y)
        df.insert(1, "Toma", Tomas)
        df.to_csv(temp_csv, index=False)
        
        best_val_loss, val_acc = ta.train_autoencoder(
            csv_path=temp_csv,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            latent_dim=LATENT_DIM,
            force_epochs=FORCE_EPOCHS,
            alpha=0.5,
            verbose=False,
            save_model=False
        )
        
        # Limpiar archivo temporal
        if os.path.exists(temp_csv): os.remove(temp_csv)
            
        return {"smooth_ms": s_ms, "target_length": t_len, "best_val_loss": best_val_loss, "val_accuracy": val_acc}
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}", "smooth_ms": s_ms, "target_length": t_len}

def worker_fase2(alpha_val, temp_csv):
    # Proceso aislado para Fase 2
    try:
        best_val_loss, val_acc = ta.train_autoencoder(
            csv_path=temp_csv,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            latent_dim=LATENT_DIM,
            force_epochs=FORCE_EPOCHS,
            alpha=alpha_val,
            verbose=False,
            save_model=False
        )
        return {"alpha": alpha_val, "best_val_loss": best_val_loss, "val_accuracy": val_acc}
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}", "alpha": alpha_val}

if __name__ == '__main__':
    # Mediciones a usar (las del día 7)
    todas_mediciones = gpt.procesar_mediciones(os.path.join(base_repo_dir, "base_de_datos_electrodos"))
    mediciones_dia7 = [m for m in todas_mediciones if "2026-07-10" in m]
    print(f"Total de mediciones encontradas para el día 7: {len(mediciones_dia7)}")

    # ==============================================================
    # FASE 1: Grid Search sobre Parámetros de DSP (Smooth y Target)
    # ==============================================================
    print("\n" + "="*50)
    print("INICIANDO FASE 1: Búsqueda de Parámetros DSP y Remuestreo (Multihilo)")
    print("="*50)

    resultados_fase1 = []
    mejor_acc_fase1 = 0
    mejores_params_fase1 = {"smooth_ms": 150, "target_length": 100}

    combinaciones_fase1 = [(s, t) for s in smooth_ms_grid for t in target_length_grid]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_fase1, s, t, mediciones_dia7): (s, t) for s, t in combinaciones_fase1}
        
        with tqdm(total=len(combinaciones_fase1), desc="Fase 1 (DSP)", unit="exp") as pbar:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if "error" not in res:
                    resultados_fase1.append(res)
                    if res["val_accuracy"] > mejor_acc_fase1:
                        mejor_acc_fase1 = res["val_accuracy"]
                        mejores_params_fase1 = {"smooth_ms": res["smooth_ms"], "target_length": res["target_length"]}
                else:
                    tqdm.write(f"Error en hilo (smooth={res['smooth_ms']}, target={res['target_length']}):\n{res['error']}")
                pbar.update(1)

    if not resultados_fase1:
        print("\n[!] FATAL: Todos los hilos de la Fase 1 fallaron. Abortando experimento.")
        sys.exit(1)
        
    # Guardar Resultados Fase 1
    df_fase1 = pd.DataFrame(resultados_fase1)
    df_fase1.to_csv(csv_out_fase1, index=False)
    print(f"\nMejores parámetros Fase 1: {mejores_params_fase1} con Accuracy {mejor_acc_fase1:.2f}%")

    # Plot Heatmap Fase 1
    pivot_table = df_fase1.pivot(index='smooth_ms', columns='target_length', values='val_accuracy')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Validation Accuracy (%)'})
    plt.title("Fase 1: Smooth MS vs Target Length (Autoencoder Accuracy)")
    plt.savefig(os.path.join(out_dir, "grid_search_3_heatmap_fase1.png"))
    plt.close()


    # ==============================================================
    # FASE 2: Grid Search sobre Alpha (Usando los mejores params)
    # ==============================================================
    print("\n" + "="*50)
    print("INICIANDO FASE 2: Búsqueda del Factor Alpha (Multihilo)")
    print("="*50)

    best_smooth = mejores_params_fase1["smooth_ms"]
    best_target = mejores_params_fase1["target_length"]

    print(f"Extrayendo dataset óptimo de referencia (Smooth={best_smooth}, Target={best_target})...")
    X, Y, Tomas, _ = gpt.extraer_features_concatenadas(
        base_dir=os.path.join(base_repo_dir, "base_de_datos_electrodos"),
        mediciones=mediciones_dia7,
        alpha_ruido=1.0,
        smooth_ms=best_smooth,
        notch_q=2.0,
        target_length=best_target,
        use_manual_exclusions=True,
        verbose=False
    )
    temp_csv_fase2 = os.path.join(out_dir, "temp_grid_fase2_base.csv")
    X_flat_fase2 = X.reshape(X.shape[0], -1)
    df = pd.DataFrame(X_flat_fase2)
    df.insert(0, "Vocal", Y)
    df.insert(1, "Toma", Tomas)
    df.to_csv(temp_csv_fase2, index=False)

    resultados_fase2 = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_fase2, alpha_val, temp_csv_fase2): alpha_val for alpha_val in alpha_grid}
        
        with tqdm(total=len(alpha_grid), desc="Fase 2 (Alpha)", unit="exp") as pbar:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if "error" not in res:
                    resultados_fase2.append(res)
                pbar.update(1)

    # Limpiar archivo temporal
    if os.path.exists(temp_csv_fase2): os.remove(temp_csv_fase2)

    # Guardar Resultados Fase 2
    df_fase2 = pd.DataFrame(resultados_fase2)
    # Ordenar para el plot
    df_fase2 = df_fase2.sort_values(by="alpha")
    df_fase2.to_csv(csv_out_fase2, index=False)

    # Plot Linechart Fase 2
    plt.figure(figsize=(8, 6))
    plt.plot(df_fase2['alpha'], df_fase2['val_accuracy'], marker='o', linestyle='-', color='b')
    plt.title(f"Fase 2: Impacto de Alpha en el Accuracy (Autoencoder)")
    plt.xlabel("Alpha (Peso de la Clasificación)")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "grid_search_3_lineplot_fase2.png"))
    plt.close()

    print("\n!!! EXPERIMENTO GRID SEARCH 3 COMPLETADO EXITOSAMENTE !!!")
