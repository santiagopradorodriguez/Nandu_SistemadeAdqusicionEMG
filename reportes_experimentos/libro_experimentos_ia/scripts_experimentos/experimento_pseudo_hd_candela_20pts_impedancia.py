# ==============================================================================
# PROYECTO NANDU LSD - DECODIFICACION BIOELECTRICA ORAL
# Experimento: Pseudo HD-sEMG Candela (9 Canales) con 20 Puntos Temporales
# y Acondicionamiento Reposo Basal e Impedancia (Cadena generador_pca_umap)
# ==============================================================================

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ------------------------------------------------------------------------------
# 1. Semillas Universales
# ------------------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

project_root = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG'
base_dir = os.path.join(project_root, 'EMG_desarrollo/base_de_datos_electrodos')
out_dir = os.path.join(project_root, 'EMG_desarrollo/resultados/grid_search_conv_ortogonal')
os.makedirs(out_dir, exist_ok=True)

VOCALES = ['A', 'E', 'I', 'O', 'U']
COLOR_MAP_HEX = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

# ------------------------------------------------------------------------------
# 2. Configuración de 9 Canales en Estricto Orden Anatómico-Funcional
# ------------------------------------------------------------------------------
CANALES_CONFIG = [
    # --- Grupo I: Depresores Mandibulares (Apertura) ---
    {
        'id': 'Ch0',
        'grupo': 'Depresor Mandibular',
        'musculo': 'Digastrico Anterior',
        'fecha': '2026-09-16',
        'filtro': 'cande',
        'canal_idx': 0,
        'descripcion': 'Apertura mandibular vertical'
    },
    {
        'id': 'Ch1',
        'grupo': 'Depresor Mandibular',
        'musculo': 'Milohioideo',
        'fecha': '2026-08-29',
        'filtro': 'cande',
        'canal_idx': 1,
        'descripcion': 'Suelo bucal / soporte hioideo'
    },
    # --- Grupo II: Elevador Mandibular (Cierre) ---
    {
        'id': 'Ch2',
        'grupo': 'Elevador Mandibular',
        'musculo': 'Masetero',
        'fecha': '2026-09-27',
        'filtro': 'mentoniano',
        'canal_idx': 0,
        'descripcion': 'Cierre mandibular y oclusion'
    },
    # --- Grupo III: Retraccion Comisural y Sonrisa ---
    {
        'id': 'Ch3',
        'grupo': 'Comisural / Sonrisa',
        'musculo': 'Cigomatico Mayor',
        'fecha': '2026-09-16',
        'filtro': 'cande',
        'canal_idx': 1,
        'descripcion': 'Sonrisa superolateral'
    },
    {
        'id': 'Ch4',
        'grupo': 'Comisural / Sonrisa',
        'musculo': 'Risorio',
        'fecha': '2026-09-01',
        'filtro': 'candela',
        'canal_idx': 1,
        'descripcion': 'Retraccion comisural horizontal'
    },
    {
        'id': 'Ch5',
        'grupo': 'Comisural / Sonrisa',
        'musculo': 'Modiolo',
        'fecha': '2026-09-18',
        'filtro': 'cande',
        'canal_idx': 1,
        'descripcion': 'Centro de confluencia muscular lateral'
    },
    {
        'id': 'Ch6',
        'grupo': 'Comisural / Sonrisa',
        'musculo': 'Nudo Sonrisa',
        'fecha': '2026-08-30',
        'filtro': 'cande',
        'canal_idx': 1,
        'descripcion': 'Anclaje dinamico de comisura'
    },
    # --- Grupo IV: Constriccion y Esfinter Labial ---
    {
        'id': 'Ch7',
        'grupo': 'Labial / Esfinter',
        'musculo': 'Orbicular Labios',
        'fecha': '2026-09-16',
        'filtro': 'cande',
        'canal_idx': 2,
        'descripcion': 'Constriccion y cierre labial'
    },
    {
        'id': 'Ch8',
        'grupo': 'Labial / Esfinter',
        'musculo': 'Mentoniano',
        'fecha': '2026-09-27',
        'filtro': 'mentoniano',
        'canal_idx': 1,
        'descripcion': 'Elevacion labio inferior y menton'
    }
]

# ------------------------------------------------------------------------------
# 3. DSP: Filtrado, Ruido Interpulso IQR y Supremo Tricanal
# ------------------------------------------------------------------------------
fs = 2000
TARGET_LEN = 20  # 20 puntos por canal requeridos por el usuario
b_notch, a_notch = signal.iirnotch(w0=50.0, Q=2.0, fs=fs)
b_band, a_band = signal.butter(N=4, Wn=[20.0, 450.0], btype='bandpass', fs=fs)
smooth_len = int(0.090 * fs)
h_rect = np.ones(smooth_len) / smooth_len

def get_interpulse_noise(segment):
    """Estima el piso de ruido dinamico depurando espigas con IQR."""
    if len(segment) < 10:
        return 0.0
    abs_s = np.abs(segment)
    q1 = np.percentile(abs_s, 25)
    q3 = np.percentile(abs_s, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    valid = abs_s[abs_s <= upper_bound]
    if len(valid) < 5:
        valid = abs_s
    return float(np.mean(valid))

def procesar_toma_tricanale_riguroso(toma_path):
    """
    Carga los 3 canales sincronizados de la toma, filtra, estima el piso
    de ruido interpulso (IQR), resta el ruido y normaliza por supremo tricanal fisico.
    Retorna diccionario {c_idx: [pulsos_20pts]}
    """
    sigs_rect = []
    sigs_env = []
    
    for c in range(3):
        wav_path = os.path.join(toma_path, f'canal_{c}', 'grabacion.wav')
        if not os.path.exists(wav_path):
            return None
        try:
            sr, raw = wavfile.read(wav_path)
        except Exception:
            return None
        if raw.ndim > 1:
            raw = raw[:, 0]
        raw = raw.astype(np.float64)
        if raw.dtype == np.int16:
            raw /= 32768.0
        elif raw.dtype == np.int32:
            raw /= 2147483648.0
            
        f1 = signal.filtfilt(b_notch, a_notch, raw)
        f2 = signal.filtfilt(b_band, a_band, f1)
        r = np.abs(f2)
        e = np.convolve(r, h_rect, mode='same')
        sigs_rect.append(r)
        sigs_env.append(e)
        
    sigs_rect = np.array(sigs_rect)  # (3, T)
    sigs_env = np.array(sigs_env)    # (3, T)
    T = sigs_rect.shape[1]
    
    picos = []
    meta_json = os.path.join(toma_path, 'canal_0', 'analisis_results.json')
    if os.path.exists(meta_json):
        try:
            with open(meta_json, 'r', encoding='utf-8') as f:
                ar = json.load(f)
            picos = ar.get('maxima_per_cut', [])
        except Exception:
            picos = []
            
    if len(picos) == 0:
        env_sum = np.sum(sigs_env, axis=0)
        p20 = np.percentile(env_sum, 20)
        p95 = np.percentile(env_sum, 95)
        umbral = p20 + 0.30 * (p95 - p20)
        active = env_sum > umbral
        labeled, num_features = label(active)
        min_len = int(0.12 * fs)
        for feat_idx in range(1, num_features + 1):
            idx_seg = np.where(labeled == feat_idx)[0]
            if len(idx_seg) >= min_len:
                picos.append(idx_seg[np.argmax(env_sum[idx_seg])])
                
    if len(picos) == 0:
        return None
        
    w_half = int(0.20 * fs)   # 400 ms ventana de pulso
    w_noise = int(0.10 * fs)  # 100 ms para piso de ruido
    
    pulsos_por_canal = {0: [], 1: [], 2: []}
    
    for p_idx in picos:
        ini = max(0, p_idx - w_half)
        fin = min(T, p_idx + w_half)
        if (fin - ini) < int(0.15 * fs):
            continue
            
        n_pre_ini = max(0, ini - w_noise)
        n_pre_fin = ini
        n_post_ini = fin
        n_post_fin = min(T, fin + w_noise)
        
        rect_limpios = []
        for c in range(3):
            ruido_pre = get_interpulse_noise(sigs_rect[c, n_pre_ini:n_pre_fin])
            ruido_post = get_interpulse_noise(sigs_rect[c, n_post_ini:n_post_fin])
            ruido_local = (ruido_pre + ruido_post) / 2.0
            
            seg_c = sigs_rect[c, ini:fin]
            seg_limpio = np.maximum(0.0, seg_c - ruido_local)
            rect_limpios.append(seg_limpio)
            
        M_supremo = max([np.max(seg) for seg in rect_limpios])
        if M_supremo <= 1e-7:
            continue
            
        for c in range(3):
            p_norm = rect_limpios[c] / M_supremo
            p_resamp = signal.resample(p_norm, TARGET_LEN)
            p_resamp = np.maximum(0.0, p_resamp)
            pulsos_por_canal[c].append(p_resamp)
            
    return pulsos_por_canal

# ------------------------------------------------------------------------------
# 4. Acondicionamiento Reposo e Impedancia Oficial (de generador_pca_umap)
# ------------------------------------------------------------------------------
def acondicionar_reposo_impedancia(X_array, sesiones, n_canales=9, n_pts_reposo=5):
    """
    Acondicionamiento por reposo basal pre-contraccion y rango dinamico P95 por sesion y canal.
    Normaliza cada canal muscular para que el silencio basal sea 0.0 y el pico de activacion sea ~1.0.
    Implementacion exacta extraida de generador_pca_umap.py.
    """
    orig_shape = X_array.shape
    if X_array.ndim == 2:
        N, D = X_array.shape
        n_pts = D // n_canales
        X_reshaped = X_array.reshape(N, n_canales, n_pts).copy()
    else:
        N, n_canales, n_pts = X_array.shape
        X_reshaped = X_array.copy()

    X_filt = np.zeros_like(X_reshaped)
    if n_pts >= 12:
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(N=3, Wn=0.3, btype='low')
            for i in range(N):
                for c in range(n_canales):
                    X_filt[i, c, :] = filtfilt(b, a, X_reshaped[i, c, :])
        except Exception:
            X_filt = X_reshaped.copy()
    else:
        X_filt = X_reshaped.copy()

    unique_ses = np.unique(sesiones)
    X_norm = np.zeros_like(X_filt)
    pts_base = max(1, min(n_pts_reposo, n_pts // 4))

    for s in unique_ses:
        mask = (sesiones == s)
        for c in range(n_canales):
            base_mean = np.mean(X_filt[mask, c, :pts_base])
            base_max = np.percentile(X_filt[mask, c, :], 95) - base_mean + 1e-6
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / base_max

    if len(orig_shape) == 2:
        return X_norm.reshape(N, -1)
    return X_norm

# ------------------------------------------------------------------------------
# 5. Extracción y Ensamble del Súper-Vector Multicanal
# ------------------------------------------------------------------------------
def extraer_dataset_pseudo_hd(target_len=20):
    print("="*95)
    print(f"EXTRACCION PSEUDO HD-sEMG FACIAL CANDELA (9 CANALES, {target_len} PUNTOS POR CANAL)")
    print("Piso de Ruido Interpulso IQR + Supremo Tricanal Fisico")
    print("="*95)

    datos_musculos = {cfg['musculo']: {v: [] for v in VOCALES} for cfg in CANALES_CONFIG}

    for cfg in CANALES_CONFIG:
        m_nom = cfg['musculo']
        f_dir = os.path.join(base_dir, cfg['fecha'])
        if not os.path.exists(f_dir):
            continue
        tomas = sorted([t for t in os.listdir(f_dir) if os.path.isdir(os.path.join(f_dir, t))])
        
        for toma in tomas:
            if cfg['filtro'].lower() not in toma.lower():
                continue
            vocal = toma.split('_')[0].upper()
            if vocal not in VOCALES:
                continue
                
            toma_path = os.path.join(f_dir, toma)
            res_toma = procesar_toma_tricanale_riguroso(toma_path)
            if res_toma is not None and len(res_toma[cfg['canal_idx']]) > 0:
                datos_musculos[m_nom][vocal].extend(res_toma[cfg['canal_idx']])

    print("\n--- RECUENTO DE PULSOS EXTRAIDOS ---")
    for cfg in CANALES_CONFIG:
        m = cfg['musculo']
        conteo_str = " | ".join([f"/{v.lower()}/: {len(datos_musculos[m][v])}" for v in VOCALES])
        print(f"[{cfg['id']}] {cfg['grupo']:22s} | {m:20s} -> {conteo_str}")

    N_TRIALS = 30  # 30 ensayos por vocal = 150 fonaciones
    rng = np.random.RandomState(42)

    X_raw_syn = []
    y_syn = []

    for vi, v in enumerate(VOCALES):
        for tr in range(N_TRIALS):
            trial = []
            for mi, cfg in enumerate(CANALES_CONFIG):
                m = cfg['musculo']
                pool = datos_musculos[m][v]
                if len(pool) > 0:
                    p = pool[rng.choice(len(pool))]
                    trial.append(p)
                else:
                    trial.append(np.zeros(target_len))
            X_raw_syn.append(np.array(trial))
            y_syn.append(vi)

    X_raw_syn = np.array(X_raw_syn, dtype=np.float32)  # (150, 9, 20)
    y_syn = np.array(y_syn, dtype=np.int64)            # (150,)
    
    return X_raw_syn, y_syn

# ------------------------------------------------------------------------------
# 6. Ejecución del Pipeline y Comparativa
# ------------------------------------------------------------------------------
def main():
    X_raw_syn, y_syn = extraer_dataset_pseudo_hd(target_len=TARGET_LEN)
    N, n_ch, n_pts = X_raw_syn.shape
    D = n_ch * n_pts
    print(f"\nDimension del Tensor 3D: {X_raw_syn.shape} | Espacio Ambiente: {D}D ({n_ch} canales x {n_pts} puntos)")

    # 1. Matriz Aplanada Directa (Sin Acondicionamiento Reposo/Impedancia)
    X_direct_flat = X_raw_syn.reshape(N, -1)

    # 2. Aplicar Acondicionamiento Reposo Basal e Impedancia P95
    sesiones_dummy = np.zeros(N, dtype=int)  # Ensamble de los 9 canales por ensayo
    X_cond_3d = acondicionar_reposo_impedancia(X_raw_syn, sesiones_dummy, n_canales=n_ch, n_pts_reposo=5)
    X_cond_flat = X_cond_3d.reshape(N, -1)
    print("Acondicionamiento por reposo basal e impedancia P95 completado exitosamente.")

    # --------------------------------------------------------------------------
    # Evaluaciones con Validación Cruzada 5-fold Estratificada
    # --------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # LDA en Espacio Ambiente (180D)
    acc_direct_ambient = cross_val_score(LinearDiscriminantAnalysis(), X_direct_flat, y_syn, cv=cv).mean() * 100.0
    acc_cond_ambient = cross_val_score(LinearDiscriminantAnalysis(), X_cond_flat, y_syn, cv=cv).mean() * 100.0

    # PCA 2D
    pca_raw = PCA(n_components=2)
    Z_pca_raw = pca_raw.fit_transform(X_direct_flat)
    acc_pca_raw = cross_val_score(LinearDiscriminantAnalysis(), Z_pca_raw, y_syn, cv=cv).mean() * 100.0

    pca_cond = PCA(n_components=9)
    Z_pca_cond_full = pca_cond.fit_transform(X_cond_flat)
    Z_pca_cond = Z_pca_cond_full[:, :2]
    evr_cond = pca_cond.explained_variance_ratio_ * 100.0
    acc_pca_cond = cross_val_score(LinearDiscriminantAnalysis(), Z_pca_cond, y_syn, cv=cv).mean() * 100.0

    # Isomap 2D
    iso_raw = Isomap(n_neighbors=15, n_components=2)
    Z_iso_raw = iso_raw.fit_transform(X_direct_flat)
    acc_iso_raw = cross_val_score(LinearDiscriminantAnalysis(), Z_iso_raw, y_syn, cv=cv).mean() * 100.0

    iso_cond = Isomap(n_neighbors=15, n_components=2)
    Z_iso_cond = iso_cond.fit_transform(X_cond_flat)
    acc_iso_cond = cross_val_score(LinearDiscriminantAnalysis(), Z_iso_cond, y_syn, cv=cv).mean() * 100.0

    # GMM No Supervisado sobre Proyecciones Acondicionadas
    def eval_gmm(Z, y_true):
        gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
        y_pred = gmm.fit_predict(Z)
        cm = np.zeros((5, 5), dtype=int)
        for true_val, pred_val in zip(y_true, y_pred):
            cm[true_val, pred_val] += 1
        r_ind, c_ind = linear_sum_assignment(-cm)
        return (cm[r_ind, c_ind].sum() / len(y_true)) * 100.0

    gmm_pca_cond = eval_gmm(Z_pca_cond, y_syn)
    gmm_iso_cond = eval_gmm(Z_iso_cond, y_syn)

    print("\n" + "="*95)
    print(f"RESULTADOS COMPARATIVOS: CANDELA PSEUDO HD-sEMG (9 CANALES, {TARGET_LEN} PTS = {D}D)")
    print("="*95)
    print(f"1. Espacio Ambiente ({D}D Directo):")
    print(f"   - Sin correccion impedancia: {acc_direct_ambient:5.2f}%")
    print(f"   - Con reposo e impedancia:   {acc_cond_ambient:5.2f}%")
    print(f"\n2. Proyeccion PCA 2D:")
    print(f"   - Sin correccion impedancia: {acc_pca_raw:5.2f}%")
    print(f"   - Con reposo e impedancia:   {acc_pca_cond:5.2f}% (GMM No Sup: {gmm_pca_cond:5.2f}%)")
    print(f"   - Varianza PC1: {evr_cond[0]:.2f}% | PC2: {evr_cond[1]:.2f}% | Acumulada 2D: {evr_cond[0]+evr_cond[1]:.2f}%")
    print(f"\n3. Proyeccion Isomap 2D (k=15):")
    print(f"   - Sin correccion impedancia: {acc_iso_raw:5.2f}%")
    print(f"   - Con reposo e impedancia:   {acc_iso_cond:5.2f}% (GMM No Sup: {gmm_iso_cond:5.2f}%)")
    print("="*95)

    # --------------------------------------------------------------------------
    # 7. Generación del Panel Gráfico Profesional
    # --------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # Panel A: PCA 2D Acondicionado
    ax_a = fig.add_subplot(gs[0, 0])
    for vi, v in enumerate(VOCALES):
        idx = np.where(y_syn == vi)[0]
        pts = Z_pca_cond[idx]
        color = COLOR_MAP_HEX[v]
        ax_a.scatter(pts[:, 0], pts[:, 1], c=color, label=f'/{v.lower()}/', s=45, alpha=0.75, edgecolors='none')
        # Centroide y elipse
        mean = np.mean(pts, axis=0)
        ax_a.scatter(mean[0], mean[1], c=color, s=140, marker='D', edgecolors='black', linewidth=1.5, zorder=5)
        cov = np.cov(pts[:, 0], pts[:, 1])
        vals, vecs = np.linalg.eigh(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2.0 * np.sqrt(np.maximum(1e-6, vals))
        ell = Ellipse(xy=mean, width=w, height=h, angle=theta, facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.2)
        ax_a.add_patch(ell)
    ax_a.set_title(f"A: PCA 2D - Reposo Basal e Impedancia P95 (LDA {acc_pca_cond:.1f}% | GMM {gmm_pca_cond:.1f}%)", fontsize=11, fontweight='bold')
    ax_a.set_xlabel(f"PC1 ({evr_cond[0]:.1f}% varianza)", fontsize=10)
    ax_a.set_ylabel(f"PC2 ({evr_cond[1]:.1f}% varianza)", fontsize=10)
    ax_a.legend(loc='best', framealpha=0.9, fontsize=9)
    ax_a.grid(True, linestyle=':', alpha=0.5)

    # Panel B: Isomap 2D Acondicionado
    ax_b = fig.add_subplot(gs[0, 1])
    for vi, v in enumerate(VOCALES):
        idx = np.where(y_syn == vi)[0]
        pts = Z_iso_cond[idx]
        color = COLOR_MAP_HEX[v]
        ax_b.scatter(pts[:, 0], pts[:, 1], c=color, label=f'/{v.lower()}/', s=45, alpha=0.75, edgecolors='none')
        mean = np.mean(pts, axis=0)
        ax_b.scatter(mean[0], mean[1], c=color, s=140, marker='D', edgecolors='black', linewidth=1.5, zorder=5)
        cov = np.cov(pts[:, 0], pts[:, 1])
        vals, vecs = np.linalg.eigh(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2.0 * np.sqrt(np.maximum(1e-6, vals))
        ell = Ellipse(xy=mean, width=w, height=h, angle=theta, facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.2)
        ax_b.add_patch(ell)
    ax_b.set_title(f"B: Isomap 2D - Reposo Basal e Impedancia P95 (LDA {acc_iso_cond:.1f}% | GMM {gmm_iso_cond:.1f}%)", fontsize=11, fontweight='bold')
    ax_b.set_xlabel("Dimension Geodesica 1", fontsize=10)
    ax_b.set_ylabel("Dimension Geodesica 2", fontsize=10)
    ax_b.legend(loc='best', framealpha=0.9, fontsize=9)
    ax_b.grid(True, linestyle=':', alpha=0.5)

    # Panel C: Varianza Explicada PCA (9 componentes)
    ax_c = fig.add_subplot(gs[1, 0])
    comps = np.arange(1, 10)
    ax_c.bar(comps, evr_cond[:9], color='#2B5B84', alpha=0.85, width=0.6, label='Varianza Individual')
    ax_c.plot(comps, np.cumsum(evr_cond[:9]), color='#E63946', marker='o', linewidth=2.0, label='Varianza Acumulada')
    for ci, v_ind in enumerate(evr_cond[:9]):
        ax_c.text(ci+1, v_ind + 0.8, f"{v_ind:.1f}%", ha='center', fontsize=8, fontweight='bold')
    ax_c.set_title("C: Espectro de Varianza Explicada PCA en 9 Componentes", fontsize=11, fontweight='bold')
    ax_c.set_xlabel("Componente Principal", fontsize=10)
    ax_c.set_ylabel("Varianza Explicada (%)", fontsize=10)
    ax_c.set_xticks(comps)
    ax_c.set_ylim(0, max(105, np.sum(evr_cond[:9]) + 5))
    ax_c.legend(loc='center right', framealpha=0.9, fontsize=9)
    ax_c.grid(True, linestyle=':', alpha=0.5)

    # Panel D: Comparativa de Desempeño LDA
    ax_d = fig.add_subplot(gs[1, 1])
    categorias = ['Espacio Ambiente\n(180D)', 'PCA 2D\n(Lineal)', 'Isomap 2D\n(Geodesico)']
    x_pos = np.arange(len(categorias))
    w_bar = 0.35

    bar1 = ax_d.bar(x_pos - w_bar/2, [acc_direct_ambient, acc_pca_raw, acc_iso_raw], w_bar, label='Sin Acondicionamiento Reposo', color='#A0AAB2', alpha=0.85)
    bar2 = ax_d.bar(x_pos + w_bar/2, [acc_cond_ambient, acc_pca_cond, acc_iso_cond], w_bar, label='Con Reposo Basal e Impedancia P95', color='#2A9D8F', alpha=0.95)

    ax_d.axhline(20.0, color='#6C757D', linestyle='--', linewidth=1.2, label='Azar Teorico (20.0%)')

    for b in list(bar1) + list(bar2):
        h = b.get_height()
        ax_d.text(b.get_x() + b.get_width()/2.0, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_d.set_title("D: Exactitud LDA de Decodificacion Vocalica - Impacto del Reposo e Impedancia", fontsize=11, fontweight='bold')
    ax_d.set_ylabel("Exactitud Multiclase (%)", fontsize=10)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(categorias, fontsize=9.5)
    ax_d.set_ylim(0, 105)
    ax_d.legend(loc='lower right', framealpha=0.9, fontsize=8.5)
    ax_d.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle(f"Pseudo HD-sEMG Facial de 9 Canales en Candela - Remuestreo a 20 Puntos y Acondicionamiento Reposo Basal/Impedancia", fontsize=13, fontweight='bold', y=0.98)

    save_path = os.path.join(out_dir, "panel_pseudo_hd_candela_20pts_pca_isomap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n[Guardado] Panel oficial guardado en: {save_path}")

    # Exportar metricas JSON
    metricas = {
        "n_canales": n_ch,
        "puntos_por_canal": TARGET_LEN,
        "dimension_ambiente": D,
        "n_muestras": N,
        "acc_ambiente_sin_acond": float(acc_direct_ambient),
        "acc_ambiente_con_acond": float(acc_cond_ambient),
        "acc_pca_2d_sin_acond": float(acc_pca_raw),
        "acc_pca_2d_con_acond": float(acc_pca_cond),
        "gmm_pca_2d_con_acond": float(gmm_pca_cond),
        "acc_isomap_2d_sin_acond": float(acc_iso_raw),
        "acc_isomap_2d_con_acond": float(acc_iso_cond),
        "gmm_isomap_2d_con_acond": float(gmm_iso_cond),
        "varianza_pc1": float(evr_cond[0]),
        "varianza_pc2": float(evr_cond[1]),
        "varianza_acumulada_2d": float(evr_cond[0] + evr_cond[1])
    }
    json_path = os.path.join(out_dir, "metricas_pseudo_hd_candela_20pts.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=4)
    print(f"[Guardado] Metricas JSON guardadas en: {json_path}")

if __name__ == '__main__':
    main()
