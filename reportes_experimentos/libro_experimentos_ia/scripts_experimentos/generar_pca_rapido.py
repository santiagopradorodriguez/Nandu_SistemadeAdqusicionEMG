import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
mediciones = [
    f"{v}_Pruebatricanal_{suf}"
    for v in ['A', 'E', 'I', 'O', 'U']
    for suf in ['Sujeto1', '2_Sujeto1', '3_Sujeto1', '4_Sujeto1']
    if os.path.isdir(os.path.join(base_dir, f"{v}_Pruebatricanal_{suf}"))
]

# Leemos params
with open("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/parametros_optimos_pca_2d.json") as f:
    params = json.load(f)

pesos = params.get('pesos_canales', [2.5, 1.0, 1.0])
target_len = params.get('target_length', 60)
alpha_ruido = params.get('alpha_ruido', 0.1)

# Procesamos directamente leyendo results.json (instantáneo)
X_list = []
Y_list = []
Tomas_list = []

for med in mediciones:
    vocal = med.split('_')[0].upper()
    med_p = os.path.join(base_dir, med)
    
    # Cargar results.json de los 4 canales
    res = {}
    for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
        rp = os.path.join(med_p, ch, "results.json")
        if not os.path.exists(rp): break
        with open(rp) as f: res[ch] = json.load(f)
        
    if len(res) < 4: continue
    
    meta_p = os.path.join(med_p, "canal_0", "metadata.json")
    with open(meta_p) as f: meta = json.load(f)
    sr = meta.get('sample_rate', 6000)
    bpm = meta.get('bpm', 30)
    muestras_pulso = int(sr * (60.0 / bpm))
    
    # Alineacion por pico de micrófono (canal_3)
    env_mic = np.array(res["canal_3"]['env_recortada'])
    dist_samples = int(0.8 * muestras_pulso)
    min_height = np.max(env_mic) * 0.20
    picos_mic, _ = find_peaks(env_mic, distance=dist_samples, height=min_height)
    
    pre_samp = int(muestras_pulso * 0.4)
    post_samp = int(muestras_pulso * 0.6)
    
    for win_idx, pico in enumerate(picos_mic):
        start = pico - pre_samp
        end = pico + post_samp
        
        if start < 0 or end > len(env_mic): continue
        
        # Canales sEMG
        segs = []
        max_supremo = 1e-9
        for ch in ["canal_0", "canal_1", "canal_2"]:
            e_ch = np.array(res[ch]['env_recortada'])
            if end > len(e_ch): break
            seg = e_ch[start:end].copy()
            # Ruido interpulso
            ruido = res[ch].get('noise_levels', [0])[0] if len(res[ch].get('noise_levels', [])) > 0 else 0
            seg = np.maximum(seg - ruido * alpha_ruido, 0.0)
            m = np.max(seg)
            if m > max_supremo: max_supremo = m
            segs.append(seg)
            
        if len(segs) < 3 or max_supremo <= 1e-6: continue
        
        vec = []
        for c_idx, s in enumerate(segs):
            s_norm = resample(s / max_supremo, target_len)
            s_norm[s_norm < 0] = 0.0
            s_norm *= pesos[c_idx] # aplicar pesos [2.5, 1.0, 1.0]
            vec.append(s_norm)
            
        X_list.append(np.concatenate(vec))
        Y_list.append(vocal)
        Tomas_list.append(f"{med}_Win{win_idx}")

# Filtrado por Isolation Forest por vocal (contamination=0.10)
X_clean = []
Y_clean = []
Tomas_clean = []

vocales_unicas = ['A', 'E', 'I', 'O', 'U']
for v in vocales_unicas:
    idxs = [i for i, y in enumerate(Y_list) if y == v]
    X_v = np.array([X_list[i] for i in idxs])
    iso = IsolationForest(contamination=0.10, random_state=42)
    preds = iso.fit_predict(X_v)
    for k, is_in in enumerate(preds):
        if is_in == 1:
            X_clean.append(X_v[k])
            Y_clean.append(v)
            Tomas_clean.append(Tomas_list[idxs[k]])

X_clean = np.array(X_clean)
Y_clean = np.array(Y_clean)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean)

# Replicación visual
COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
fig, ax = plt.subplots(figsize=(9, 8))
for v in vocales_unicas:
    m = (Y_clean == v)
    ax.scatter(X_pca[m, 0], X_pca[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/ (N={np.sum(m)})", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)

var = pca.explained_variance_ratio_ * 100.0
ax.set_title(f"PCA 2D Replicado con generador_pca_umap\nVar: PC1={var[0]:.1f}%, PC2={var[1]:.1f}% (Total: {np.sum(var):.1f}%)", fontweight='bold')
ax.set_xlabel(f"PC 1 ({var[0]:.1f}%)", fontweight='bold')
ax.set_ylabel(f"PC 2 ({var[1]:.1f}%)", fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', frameon=True)
plt.tight_layout()

out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_2d_replicado_generador.png"
plt.savefig(out_f, dpi=300)
plt.close()

# Comparar con tu archivo guardado
df_orig = pd.read_csv("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/triada/proyecciones_pca_2d.csv")
print(f"Pulsos replicados: {len(X_pca)} vs Guardados en tu sesion: {len(df_orig)}")
print(f"Figura guardada en: {out_f}")
