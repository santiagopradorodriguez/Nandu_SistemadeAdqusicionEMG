import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def main():
    print("--- INICIANDO OPCION NUCLEAR: CLASIFICADOR XGBOOST ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_repo_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Argumentos
    method = "pca_tensorial"
    if len(sys.argv) > 1:
        method = sys.argv[1]
        
    out_dir = os.path.join(base_repo_dir, "resultados", f"resultados_{method}")
    csv_path = os.path.join(out_dir, "caracteristicas_exportadas.csv")
        
    if not os.path.exists(csv_path):
        print(f"❌ Error: No se encontró {csv_path}. ¡Extrae las features primero!")
        return
        
    print(f"Cargando dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    features = [col for col in df.columns if col not in ['Toma', 'Vocal']]
    df[features] = df[features].fillna(0.0)
    
    X = df[features].values
    y_str = df['Vocal'].values
    
    # Codificar etiquetas
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    
    print(f"\nEntrenando XGBoost con {X.shape[1]} features para {X.shape[0]} muestras...")
    
    # Validación Cruzada Estratificada (5 Folds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    y_pred_all = np.zeros_like(y)
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred_all[test_index] = model.predict(X_test)
        
    acc = accuracy_score(y, y_pred_all)
    print(f"\n[+] Precision (Accuracy) Global: {acc*100:.2f}%")
    
    print("\nReporte de Clasificacion:")
    print(classification_report(y, y_pred_all, target_names=le.classes_))
    
    # Matriz de confusión
    cm = confusion_matrix(y, y_pred_all)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matriz de Confusión XGBoost (Dataset: {method.upper()})\nAccuracy: {acc*100:.2f}%')
    plt.ylabel('Vocal Real')
    plt.xlabel('Vocal Predicha')
    plt.tight_layout()
    
    out_img = os.path.join(out_dir, f"xgboost_confusion_{method}.png")
    plt.savefig(out_img, dpi=300)
    print(f"Grafico guardado en: {out_img}")
    plt.show()

if __name__ == "__main__":
    main()
