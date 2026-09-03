# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Motor backend para la generación de reportes técnicos estructurados en LaTeX/PDF.
# ==============================================================================

import os
import glob
import json
import csv
import subprocess
import re
from datetime import datetime
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def escape_latex(text):
    """Escapa caracteres especiales de LaTeX preservando saltos de línea y comandos."""
    if not text:
        return ""
    text = str(text)
    chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    out = []
    for line in text.split('\n'):
        if line.strip().startswith('\\'):
            out.append(line)
        else:
            escaped_line = ""
            for c in line:
                escaped_line += chars.get(c, c)
            out.append(escaped_line)
    return '\n'.join(out)

def find_first_existing(patterns):
    """Busca el primer archivo existente dentro de una lista de patrones glob o rutas."""
    for p in patterns:
        matches = glob.glob(p)
        if matches and os.path.exists(matches[0]):
            return matches[0].replace('\\', '/')
    return None

class ReportEngine:
    """Motor completo para compilar reportes técnicos integrales según la estructura oficial."""
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.output_dir = os.path.join(repo_root, "reportes_experimentos")
        else:
            self.output_dir = os.path.abspath(output_dir)
            
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_med_tags(self, med_name):
        """Parsea la estructura vocal_sesion_nombresujeto para agrupar por pruebas."""
        parts = med_name.split('_')
        vocal = parts[0].upper()
        if len(parts) >= 3:
            prueba_tag = parts[1]
            sujeto_tag = "_".join(parts[2:])
        elif len(parts) == 2:
            prueba_tag = parts[1]
            sujeto_tag = ""
        else:
            prueba_tag = "Prueba 1"
            sujeto_tag = ""
        return vocal, prueba_tag, sujeto_tag

    def extract_session_metadata(self, session_paths):
        """Extrae metadatos y ordena cronológicamente todas las pruebas seleccionadas."""
        info = {
            'fecha': None,
            'sujeto': None,
            'sampling_rate': None,
            'canales': {},
            'letras': [],
            'mediciones': [],
            'pruebas_agrupadas': OrderedDict()
        }
        
        raw_meds = []
        for path in session_paths:
            if not os.path.isdir(path):
                continue
                
            med_name = os.path.basename(path)
            vocal_parsed, prueba_parsed, sujeto_parsed = self.parse_med_tags(med_name)
            if "secuencia" in med_name.lower() or vocal_parsed not in ['A', 'E', 'I', 'O', 'U']:
                continue
            meta_file = os.path.join(path, "canal_0", "metadata.json")
            
            dt_obj = None
            hora_str = "--:--:--"
            
            med_info = {
                'path': path,
                'name': med_name,
                'letra': vocal_parsed,
                'prueba_tag': prueba_parsed,
                'sujeto_tag': sujeto_parsed,
                'canales': {},
                'dt_obj': None,
                'hora_str': hora_str
            }
            
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not info['fecha'] and ('date' in data or 'measurement_date' in data):
                            info['fecha'] = data.get('date') or str(data.get('measurement_date', '')).split('T')[0]
                        if not info['sujeto']:
                            info['sujeto'] = data.get('sujeto') or data.get('subject') or data.get('usuario') or ""
                        if not info['sampling_rate'] and ('sampling_rate' in data or 'sample_rate' in data):
                            info['sampling_rate'] = data.get('sampling_rate') or data.get('sample_rate')
                        if 'vowel' in data:
                            med_info['letra'] = str(data['vowel']).upper()
                        elif 'letter' in data:
                            med_info['letra'] = str(data['letter']).upper()
                        elif 'letra' in data:
                            med_info['letra'] = str(data['letra']).upper()
                            
                        # Extraer músculos de metadata principal
                        if 'muscles_map' in data and isinstance(data['muscles_map'], dict):
                            for ch_k, m_val in data['muscles_map'].items():
                                idx_str = str(ch_k).replace('canal_', '').replace('ch', '').strip()
                                if idx_str.isdigit():
                                    idx_num = int(idx_str)
                                    med_info['canales'][idx_num] = str(m_val)
                                    info['canales'][idx_num] = str(m_val)
                        elif 'muscles' in data and isinstance(data['muscles'], list):
                            for idx_num, m_val in enumerate(data['muscles']):
                                med_info['canales'][idx_num] = str(m_val)
                                info['canales'][idx_num] = str(m_val)
                                
                        for date_key in ['measurement_date', 'date', 'creation_time']:
                            if date_key in data and data[date_key]:
                                try:
                                    dt_obj = datetime.fromisoformat(str(data[date_key]))
                                    break
                                except Exception:
                                    pass
                except Exception:
                    pass
            
            if dt_obj is None:
                try:
                    dt_obj = datetime.fromtimestamp(os.path.getmtime(path))
                except Exception:
                    dt_obj = datetime.now()
                    
            med_info['dt_obj'] = dt_obj
            med_info['hora_str'] = dt_obj.strftime("%H:%M:%S")
            
            for ch_idx in range(4):
                if ch_idx in med_info['canales'] and med_info['canales'][ch_idx] != f"Canal {ch_idx}":
                    continue
                ch_dir = os.path.join(path, f"canal_{ch_idx}")
                ch_meta = os.path.join(ch_dir, "metadata.json")
                muscle_name = f"Canal {ch_idx}"
                if os.path.exists(ch_meta):
                    try:
                        with open(ch_meta, 'r', encoding='utf-8') as f:
                            m_data = json.load(f)
                            if 'musculo' in m_data and m_data['musculo']:
                                muscle_name = m_data['musculo']
                            elif 'muscle' in m_data and m_data['muscle']:
                                muscle_name = m_data['muscle']
                            elif 'channel_name' in m_data and m_data['channel_name']:
                                muscle_name = m_data['channel_name']
                    except Exception:
                        pass
                
                med_info['canales'][ch_idx] = muscle_name
                if ch_idx not in info['canales'] or info['canales'][ch_idx] == f"Canal {ch_idx}":
                    info['canales'][ch_idx] = muscle_name

            raw_meds.append(med_info)

        raw_meds.sort(key=lambda x: x['dt_obj'])
        
        if raw_meds:
            t0 = raw_meds[0]['dt_obj']
            t_fin = raw_meds[-1]['dt_obj']
            duracion_sec = max(0.0, (t_fin - t0).total_seconds())
            duracion_min = duracion_sec / 60.0
            
            info['hora_inicio'] = raw_meds[0]['hora_str']
            info['hora_fin'] = raw_meds[-1]['hora_str']
            info['duracion_min'] = duracion_min
            if duracion_min >= 60:
                horas = int(duracion_min // 60)
                mins = int(duracion_min % 60)
                info['duracion_str'] = f"{horas} h {mins} min ({duracion_min:.1f} min)"
            else:
                info['duracion_str'] = f"{duracion_min:.1f} min"

            for med in raw_meds:
                delta_sec = (med['dt_obj'] - t0).total_seconds()
                med['delta_min'] = delta_sec / 60.0
                if med['letra'] not in info['letras']:
                    info['letras'].append(med['letra'])
                    
                ptag = med['prueba_tag']
                if ptag not in info['pruebas_agrupadas']:
                    info['pruebas_agrupadas'][ptag] = []
                info['pruebas_agrupadas'][ptag].append(med)

        # Ordenar cada grupo estrictamente por orden cronológico (dt_obj)
        for ptag in info['pruebas_agrupadas']:
            info['pruebas_agrupadas'][ptag].sort(
                key=lambda m: m['dt_obj']
            )

        info['mediciones'] = raw_meds

        if not info['fecha']:
            info['fecha'] = datetime.now().strftime("%Y-%m-%d")
        if not info['sujeto']:
            info['sujeto'] = "No especificado"

        return info

    def generate_muscle_activation_cube_3d(self, session_paths, info, output_path, projections_output_path=None):
        """Genera una proyección 3D de las proporciones musculares y opcionalmente las proyecciones en cada cara."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import scipy.io.wavfile
        
        series_data = OrderedDict()
        global_max_val = 1e-6
        
        canales_map = info.get('canales', {0: "Canal 0", 1: "Canal 1", 2: "Canal 2"})
        m0_label = canales_map.get(0, "Canal 0")
        m1_label = canales_map.get(1, "Canal 1")
        m2_label = canales_map.get(2, "Canal 2")
        
        for ptag, meds in info.get('pruebas_agrupadas', {}).items():
            if ptag not in series_data:
                series_data[ptag] = {}
            for med in meds:
                vocal = med['letra']
                if vocal not in ['A', 'E', 'I', 'O', 'U']:
                    continue
                m_path = med['path']
                
                ch_means = []
                for c_idx in [0, 1, 2]:
                    ch_dir = os.path.join(m_path, f"canal_{c_idx}")
                    res_p = os.path.join(ch_dir, "analisis_results.json")
                    wav_p = os.path.join(ch_dir, "grabacion.wav")
                    
                    val = 0.0
                    for rf in ["results.json", "analisis_results.json"]:
                        rp = os.path.join(ch_dir, rf)
                        if os.path.exists(rp):
                            try:
                                with open(rp, 'r', encoding='utf-8') as f:
                                    res = json.load(f)
                                    picos = res.get('picos_ventana', [])
                                    if picos and isinstance(picos, list):
                                        valid_picos = [x for x in picos if x is not None and not np.isnan(x) and x > 0]
                                        if valid_picos:
                                            val = float(np.mean(valid_picos))
                                            break
                                    segs = res.get('segmentos_rs', [])
                                    if segs:
                                        peaks = [float(np.max(p)) for p in segs if len(p) > 0]
                                        if peaks:
                                            val = float(np.mean(peaks))
                                            break
                            except Exception:
                                pass
                    if val <= 0.0 and os.path.exists(wav_p):
                        try:
                            _, data_w = scipy.io.wavfile.read(wav_p)
                            if data_w.ndim > 1: data_w = data_w[:, 0]
                            val = float(np.max(np.abs(data_w)))
                        except Exception:
                            val = 10.0
                    ch_means.append(val)
                
                for v in ch_means:
                    if v > global_max_val:
                        global_max_val = v
                series_data[ptag][vocal] = ch_means
                
        # Calcular el máximo de cada canal independiente para que cada eje aproveche [0, 1]
        # De esta forma el Anterior Belly en la A (su máxima contracción) alcanza 1.0 (extremo derecho)
        max_ch = [1e-6, 1e-6, 1e-6]
        for ptag, v_dict in series_data.items():
            for vocal, vals in v_dict.items():
                for c in range(3):
                    if vals[c] > max_ch[c]:
                        max_ch[c] = vals[c]

        fig = plt.figure(figsize=(9, 7), dpi=300, facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_zlim([0.0, 1.0])
        
        ax.set_xlabel(f"X: {m0_label} (Proporción)", fontsize=9, labelpad=8)
        ax.set_ylabel(f"Y: {m1_label} (Proporción)", fontsize=9, labelpad=8)
        ax.set_zlabel(f"Z: {m2_label} (Proporción)", fontsize=9, labelpad=8)
        
        colores_vocales = {
            'A': '#e41a1c',
            'E': '#377eb8',
            'I': '#4daf4a',
            'O': '#984ea3',
            'U': '#ff7f00'
        }
        markers = ['o', 'D', '^', 's', 'v', 'p', '*']
        
        # Conectar la misma vocal entre series con línea punteada
        for vocal, col in colores_vocales.items():
            pts = []
            for ptag, v_dict in series_data.items():
                if vocal in v_dict:
                    vals = v_dict[vocal]
                    norm_pt = [vals[c] / max_ch[c] for c in range(3)]
                    pts.append(norm_pt)
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                ax.plot(xs, ys, zs, linestyle='--', color=col, alpha=0.7, linewidth=1.5)
                
        # Scatter de cada punto con leyenda
        for s_idx, (ptag, v_dict) in enumerate(series_data.items()):
            marker = markers[s_idx % len(markers)]
            for vocal in ['A', 'E', 'I', 'O', 'U']:
                if vocal in v_dict:
                    vals = v_dict[vocal]
                    norm_pt = [vals[c] / max_ch[c] for c in range(3)]
                    col = colores_vocales.get(vocal, '#333333')
                    ax.scatter(
                        norm_pt[0], norm_pt[1], norm_pt[2],
                        color=col,
                        marker=marker,
                        s=90,
                        edgecolors='k',
                        linewidth=0.8,
                        alpha=0.9,
                        label=f"{vocal} ({ptag})"
                    )
                    
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8, frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Proyecciones 2D en cada cara del cubo (Planos XY, XZ, YZ)
        if projections_output_path:
            fig_p, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
            planos = [
                (0, 1, f"X: {m0_label}", f"Y: {m1_label}", "Plano XY (Frontal / Inferior)"),
                (0, 2, f"X: {m0_label}", f"Z: {m2_label}", "Plano XZ (Lateral / XZ)"),
                (1, 2, f"Y: {m1_label}", f"Z: {m2_label}", "Plano YZ (Lateral / YZ)")
            ]
            for ax_p, (ix, iy, lx, ly, p_title) in zip(axes, planos):
                ax_p.set_facecolor('#fafafa')
                ax_p.grid(True, linestyle='--', alpha=0.5)
                ax_p.set_xlim([-0.02, 1.02])
                ax_p.set_ylim([-0.02, 1.02])
                ax_p.set_xlabel(f"{lx} (Proporción)", fontsize=9)
                ax_p.set_ylabel(f"{ly} (Proporción)", fontsize=9)
                ax_p.set_title(p_title, fontsize=10, fontweight='bold')
                for vocal, col in colores_vocales.items():
                    pts = []
                    for ptag, v_dict in series_data.items():
                        if vocal in v_dict:
                            vals = v_dict[vocal]
                            norm_pt = [vals[c] / max_ch[c] for c in range(3)]
                            pts.append((norm_pt[ix], norm_pt[iy]))
                    if len(pts) >= 2:
                        ax_p.plot([p[0] for p in pts], [p[1] for p in pts], linestyle='--', color=col, alpha=0.6, linewidth=1.2)
                for s_idx, (ptag, v_dict) in enumerate(series_data.items()):
                    marker = markers[s_idx % len(markers)]
                    for vocal in ['A', 'E', 'I', 'O', 'U']:
                        if vocal in v_dict:
                            vals = v_dict[vocal]
                            norm_pt = [vals[c] / max_ch[c] for c in range(3)]
                            ax_p.scatter(norm_pt[ix], norm_pt[iy], color=colores_vocales[vocal], marker=marker, s=65, edgecolors='k', linewidth=0.6, alpha=0.85)
            plt.tight_layout()
            plt.savefig(projections_output_path, dpi=300, bbox_inches='tight')
            plt.close()

        return output_path.replace('\\', '/')

    def generate_muscle_activation_cube_all_pulses_3d(self, session_paths, info, output_path):
        """Genera el cubo 3D graficando todos los puntos (pulsos individuales) registrados por vocal."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        canales_map = info.get('canales', {0: "Canal 0", 1: "Canal 1", 2: "Canal 2"})
        m0_label = canales_map.get(0, "Canal 0")
        m1_label = canales_map.get(1, "Canal 1")
        m2_label = canales_map.get(2, "Canal 2")
        
        colores_vocales = {
            'A': '#e41a1c',
            'E': '#377eb8',
            'I': '#4daf4a',
            'O': '#984ea3',
            'U': '#ff7f00'
        }
        
        puntos_por_vocal = {v: [] for v in ['A', 'E', 'I', 'O', 'U']}
        global_max_pulse = 1e-6
        
        for med in info.get('mediciones', []):
            vocal = med.get('letra', '')
            if vocal not in puntos_por_vocal:
                continue
            m_path = med['path']
            
            picos_ch = {}
            for c_idx in [0, 1, 2]:
                ch_dir = os.path.join(m_path, f"canal_{c_idx}")
                picos_ch[c_idx] = []
                for rf in ["results.json", "analisis_results.json"]:
                    rp = os.path.join(ch_dir, rf)
                    if os.path.exists(rp):
                        try:
                            with open(rp, 'r', encoding='utf-8') as f:
                                res = json.load(f)
                                picos = res.get('picos_ventana', [])
                                if picos and isinstance(picos, list):
                                    picos_ch[c_idx] = [float(x) for x in picos if x is not None and not np.isnan(x) and x > 0]
                                    if picos_ch[c_idx]: break
                        except Exception:
                            pass
                            
            if 0 in picos_ch and 1 in picos_ch and 2 in picos_ch:
                n_pulses = min(len(picos_ch[0]), len(picos_ch[1]), len(picos_ch[2]))
                for i in range(n_pulses):
                    pt = (picos_ch[0][i], picos_ch[1][i], picos_ch[2][i])
                    puntos_por_vocal[vocal].append(pt)

        all_pts = [pt for v in puntos_por_vocal for pt in puntos_por_vocal[v]]
        if all_pts:
            max_pulse_per_ch = [
                float(np.percentile([p[c] for p in all_pts], 98.5))
                for c in range(3)
            ]
            for c in range(3):
                if max_pulse_per_ch[c] <= 0:
                    max_pulse_per_ch[c] = 1.0
        else:
            max_pulse_per_ch = [1.0, 1.0, 1.0]
                            
        fig = plt.figure(figsize=(10, 8), dpi=300, facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_zlim([0.0, 1.0])
        
        ax.set_xlabel(f"X: {m0_label} (Proporción)", fontsize=10, labelpad=8)
        ax.set_ylabel(f"Y: {m1_label} (Proporción)", fontsize=10, labelpad=8)
        ax.set_zlabel(f"Z: {m2_label} (Proporción)", fontsize=10, labelpad=8)
        
        for vocal in ['A', 'E', 'I', 'O', 'U']:
            pts = puntos_por_vocal[vocal]
            if not pts: continue
            xs = [min(1.0, p[0] / max_pulse_per_ch[0]) for p in pts]
            ys = [min(1.0, p[1] / max_pulse_per_ch[1]) for p in pts]
            zs = [min(1.0, p[2] / max_pulse_per_ch[2]) for p in pts]
            col = colores_vocales[vocal]
            ax.scatter(xs, ys, zs, color=col, s=80, alpha=0.9, edgecolors='black', linewidth=0.7, label=f"Vocal {vocal} (N={len(pts)})")
            
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, frameon=True, framealpha=0.95)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path.replace('\\', '/')

    def generate_dynamic_phase_space_3d(self, session_paths, info, output_path, projections_output_path=None):
        """Genera una galería de 6 paneles con el espacio de fases dinámico continuo (trayectorias 3D por pulso)
        aplicando normalización tricanal donde el músculo dominante de cada vocal alcanza 1.0 en promedio."""
        import soundfile as sf
        from scipy.signal import butter, filtfilt, iirnotch

        def compute_rms_env(sig, sr, smooth_ms=180):
            win_len = max(1, int(round(smooth_ms * sr / 1000.0)))
            if win_len > 1:
                window = np.ones(win_len, dtype=float) / float(win_len)
                return np.sqrt(np.convolve(sig ** 2, window, mode="same"))
            return np.abs(sig)

        def apply_filters(data, fs):
            nyq = 0.5 * fs
            b_notch, a_notch = iirnotch(50.0 / nyq, 2.0)
            data = filtfilt(b_notch, a_notch, data)
            b_band, a_band = butter(4, [20.0 / nyq, min(0.999, 500.0 / nyq)], btype="band")
            data = filtfilt(b_band, a_band, data)
            return data

        m0_label = info.get('canales', {}).get(0, "Canal 0")
        m1_label = info.get('canales', {}).get(1, "Canal 1")
        m2_label = info.get('canales', {}).get(2, "Canal 2")

        vocales_data = {"A": [], "E": [], "I": [], "O": [], "U": []}

        for path in session_paths:
            f_name = os.path.basename(path)
            if "secuencia" in f_name.lower():
                continue
            vocal = f_name.split("_")[0].upper()
            if vocal not in vocales_data:
                continue

            meta0_path = os.path.join(path, "canal_0", "analisis_results.json")
            if not os.path.exists(meta0_path):
                meta0_path = os.path.join(path, "canal_0", "results.json")
            if not os.path.exists(meta0_path):
                continue

            try:
                with open(meta0_path, 'r', encoding='utf-8') as f:
                    meta_dict = json.load(f)
            except Exception:
                continue

            picos = meta_dict.get("maxima_per_cut", [])
            muestras_pulso = int(meta_dict.get("muestras_pulso", 4000))
            pre_samples = int(0.4 * muestras_pulso)
            post_samples = int(0.6 * muestras_pulso)

            sigs_env = []
            sr = 2000
            for c in range(3):
                wav_path = os.path.join(path, f"canal_{c}", "grabacion.wav")
                if not os.path.exists(wav_path):
                    break
                sig, sr = sf.read(wav_path)
                if sig.ndim > 1: sig = sig[:, 0]
                sig_filt = apply_filters(sig, sr)
                env = compute_rms_env(sig_filt, sr, smooth_ms=180)
                sigs_env.append(env)

            if len(sigs_env) < 3:
                continue

            for p in picos:
                p_start = p - pre_samples
                p_end = p + post_samples
                if p_start >= 0 and p_end <= len(sigs_env[0]):
                    trajs = []
                    for c in range(3):
                        seg = sigs_env[c][p_start:p_end].copy()
                        noise_base = np.mean(seg[:int(0.15 * sr)])
                        seg = np.maximum(seg - noise_base, 0.0)
                        idx_sub = np.linspace(0, len(seg) - 1, 80).astype(int)
                        trajs.append(seg[idx_sub])

                    if max(np.max(trajs[0]), np.max(trajs[1]), np.max(trajs[2])) > 5e-6:
                        vocales_data[vocal].append(trajs)

        # Normalización Tricanal: el máximo del músculo dominante en su vocal representativa es 1.0
        max_c0 = 1e-9
        if vocales_data["A"]:
            arr_a_c0 = [t[0] for t in vocales_data["A"]]
            max_c0 = float(np.max(np.mean(arr_a_c0, axis=0)))

        max_c1 = 1e-9
        if vocales_data["O"]:
            arr_o_c1 = [t[1] for t in vocales_data["O"]]
            max_c1 = float(np.max(np.mean(arr_o_c1, axis=0)))
        elif vocales_data["U"]:
            arr_u_c1 = [t[1] for t in vocales_data["U"]]
            max_c1 = float(np.max(np.mean(arr_u_c1, axis=0)))

        max_c2 = 1e-9
        if vocales_data["I"]:
            arr_i_c2 = [t[2] for t in vocales_data["I"]]
            max_c2 = float(np.max(np.mean(arr_i_c2, axis=0)))

        if max_c0 <= 1e-9: max_c0 = 1.0
        if max_c1 <= 1e-9: max_c1 = 1.0
        if max_c2 <= 1e-9: max_c2 = 1.0

        colores = {
            'A': '#d62728',
            'E': '#1f77b4',
            'I': '#2ca02c',
            'O': '#9467bd',
            'U': '#ff7f0e'
        }

        fig = plt.figure(figsize=(20, 13), facecolor='white', dpi=300)
        plt.subplots_adjust(wspace=0.15, hspace=0.25)

        posiciones = {
            'A': (2, 3, 1),
            'E': (2, 3, 2),
            'I': (2, 3, 3),
            'O': (2, 3, 4),
            'U': (2, 3, 5)
        }

        medianas_por_vocal = {}

        for vocal, pos in posiciones.items():
            ax = fig.add_subplot(pos[0], pos[1], pos[2], projection='3d')
            ax.set_facecolor('white')
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            pulsos = vocales_data[vocal]
            col = colores[vocal]

            c0_m, c1_m, c2_m = [], [], []
            for trajs in pulsos:
                x = np.clip(trajs[0] / max_c0, 0.0, 1.25)
                y = np.clip(trajs[1] / max_c1, 0.0, 1.25)
                z = np.clip(trajs[2] / max_c2, 0.0, 1.25)
                c0_m.append(x)
                c1_m.append(y)
                c2_m.append(z)
                ax.plot(x, y, z, color=col, alpha=0.35, linewidth=1.1)

            if c0_m:
                x_med = np.mean(c0_m, axis=0)
                y_med = np.mean(c1_m, axis=0)
                z_med = np.mean(c2_m, axis=0)
                medianas_por_vocal[vocal] = (x_med, y_med, z_med)

                ax.plot(x_med, y_med, z_med, color='black', linewidth=2.6, label=f"Órbita Promedio (N={len(pulsos)})")
                ax.scatter([x_med[0]], [y_med[0]], [z_med[0]], color='black', s=45, marker='o', zorder=10)
                idx_max = np.argmax(x_med**2 + y_med**2 + z_med**2)
                ax.scatter([x_med[idx_max]], [y_med[idx_max]], [z_med[idx_max]], color='gold', edgecolor='black', s=85, marker='*', zorder=11, label="Vértice Máximo")

            ax.set_xlim([0.0, 1.05])
            ax.set_ylim([0.0, 1.05])
            ax.set_zlim([0.0, 1.05])

            ax.set_xlabel(f"X: {m0_label}", fontsize=9, labelpad=7)
            ax.set_ylabel(f"Y: {m1_label}", fontsize=9, labelpad=7)
            ax.set_zlabel(f"Z: {m2_label}", fontsize=9, labelpad=7)

            ax.set_title(f"Espacio Dinámico: Vocal {vocal}", fontsize=12, fontweight='bold', pad=8)
            ax.view_init(elev=25, azim=-60)
            ax.legend(loc='upper left', fontsize=8.5)

        # Panel 6: Síntesis comparativa de las 5 órbitas
        ax_comp = fig.add_subplot(2, 3, 6, projection='3d')
        ax_comp.set_facecolor('white')
        ax_comp.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax_comp.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax_comp.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        for vocal in ['A', 'E', 'I', 'O', 'U']:
            if vocal in medianas_por_vocal:
                xm, ym, zm = medianas_por_vocal[vocal]
                ax_comp.plot(xm, ym, zm, color=colores[vocal], linewidth=2.8, label=f"Vocal {vocal}")
                idx_m = np.argmax(xm**2 + ym**2 + zm**2)
                ax_comp.scatter([xm[idx_m]], [ym[idx_m]], [zm[idx_m]], color=colores[vocal], edgecolor='black', s=80, marker='o')

        ax_comp.scatter([0], [0], [0], color='black', s=60, marker='s', label="Reposo (0,0,0)")
        ax_comp.set_xlim([0.0, 1.05])
        ax_comp.set_ylim([0.0, 1.05])
        ax_comp.set_zlim([0.0, 1.05])
        ax_comp.set_xlabel(f"X: {m0_label}", fontsize=9, labelpad=7)
        ax_comp.set_ylabel(f"Y: {m1_label}", fontsize=9, labelpad=7)
        ax_comp.set_zlabel(f"Z: {m2_label}", fontsize=9, labelpad=7)
        ax_comp.set_title("Comparativa de Órbitas (Lazos 3D)", fontsize=12, fontweight='bold', pad=8)
        ax_comp.view_init(elev=25, azim=-60)
        ax_comp.legend(loc='upper left', fontsize=8.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Proyecciones bidimensionales ortogonales en las caras del cubo (Planos XY, XZ, YZ)
        if projections_output_path and medianas_por_vocal:
            fig_p, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=300)
            planos = [
                (0, 1, f"X: {m0_label}", f"Y: {m1_label}", "Cara Inferior (Plano XY)"),
                (0, 2, f"X: {m0_label}", f"Z: {m2_label}", "Cara Lateral (Plano XZ)"),
                (1, 2, f"Y: {m1_label}", f"Z: {m2_label}", "Cara Frontal (Plano YZ)")
            ]
            for ax_p, (c_x, c_y, lbl_x, lbl_y, p_title) in zip(axes, planos):
                ax_p.scatter([0], [0], color='black', s=70, marker='s', zorder=10, label="Reposo (0,0)")
                for vocal in ['A', 'E', 'I', 'O', 'U']:
                    if vocal in medianas_por_vocal:
                        orb = medianas_por_vocal[vocal]
                        vx, vy = orb[c_x], orb[c_y]
                        col = colores[vocal]
                        ax_p.plot(vx, vy, color=col, linewidth=2.6, label=f"Vocal {vocal}")
                        idx_p = np.argmax(vx**2 + vy**2)
                        ax_p.scatter([vx[idx_p]], [vy[idx_p]], color=col, edgecolor='black', s=80, zorder=8)
                ax_p.set_xlim([-0.02, 1.05])
                ax_p.set_ylim([-0.02, 1.05])
                ax_p.set_xlabel(f"{lbl_x} (Proporción)", fontsize=11)
                ax_p.set_ylabel(f"{lbl_y} (Proporción)", fontsize=11)
                ax_p.set_title(p_title, fontsize=12, fontweight='bold', pad=10)
                ax_p.grid(True, linestyle='--', alpha=0.5)
                ax_p.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
            plt.tight_layout()
            plt.savefig(projections_output_path, dpi=300, bbox_inches='tight')
            plt.close(fig_p)

        return output_path.replace('\\', '/')

    def generate_session_evolution_plots(self, session_paths, info=None, logger=print):
        """Genera automáticamente los gráficos de evolución de sesión (Amplitud Timeline, Cubo 3D, etc.)."""
        import analysis.analisis_por_track_integrado as track_mod
        
        evolucion_dir = os.path.join(self.output_dir, "evolucion_sesion")
        os.makedirs(evolucion_dir, exist_ok=True)
        nombre_salida_base = os.path.join(evolucion_dir, "Sesion")
        
        mediciones_data = []
        for path_medicion in session_paths:
            if not os.path.isdir(path_medicion):
                continue
            folder_name = os.path.basename(path_medicion)
            vocal_parsed, _, _ = self.parse_med_tags(folder_name)
            if "secuencia" in folder_name.lower() or vocal_parsed not in ['A', 'E', 'I', 'O', 'U']:
                continue
            letra = vocal_parsed
            
            canales_data = {}
            muscles_map = {}
            dt_obj = None
            hora_str = ""
            
            meta_canal0 = os.path.join(path_medicion, 'canal_0', 'metadata.json')
            if os.path.exists(meta_canal0):
                try:
                    with open(meta_canal0, 'r', encoding='utf-8') as f:
                        m0 = json.load(f)
                        if 'letra' in m0: letra = m0['letra']
                        elif 'vowel' in m0: letra = m0['vowel']
                        if 'muscles_map' in m0:
                            muscles_map = m0['muscles_map']
                        elif 'muscles' in m0 and isinstance(m0['muscles'], list):
                            muscles_map = {f"canal_{i}": m for i, m in enumerate(m0['muscles'])}
                except Exception:
                    pass
                    
            for ch_idx in [0, 1, 2]:
                ch_key = f'canal_{ch_idx}'
                ch_path = os.path.join(path_medicion, ch_key)
                if not os.path.exists(ch_path):
                    continue
                    
                ch_musculo = muscles_map.get(ch_key, muscles_map.get(str(ch_idx), muscles_map.get(ch_idx, f"Canal {ch_idx}")))
                res_path = os.path.join(ch_path, 'analisis_results.json')
                meta_path = os.path.join(ch_path, 'metadata.json')
                
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            if 'musculo' in meta and meta['musculo']:
                                ch_musculo = meta['musculo']
                            if dt_obj is None and 'measurement_date' in meta:
                                dt_obj = datetime.fromisoformat(meta['measurement_date'])
                                hora_str = dt_obj.strftime("%H:%M:%S")
                    except Exception:
                        pass
                        
                snr_per_pulse = []
                amp_per_pulse = []
                for fname in ['results.json', 'analisis_results.json']:
                    rp = os.path.join(ch_path, fname)
                    if os.path.exists(rp):
                        try:
                            with open(rp, 'r', encoding='utf-8') as f:
                                res = json.load(f)
                                if 'snr_per_pulse' in res and res['snr_per_pulse']:
                                    snr_per_pulse = res['snr_per_pulse']
                                if 'picos_ventana' in res and isinstance(res['picos_ventana'], list) and len(res['picos_ventana']) > 0:
                                    amp_per_pulse = [float(x) for x in res['picos_ventana'] if x is not None and not np.isnan(x) and x > 0]
                                elif 'segmentos_rs' in res and isinstance(res['segmentos_rs'], list) and len(res['segmentos_rs']) > 0:
                                    for p in res['segmentos_rs']:
                                        if isinstance(p, list) and len(p) > 0:
                                            p_arr = np.array(p)
                                            q25, q75 = np.percentile(p_arr, [25, 75])
                                            iqr = q75 - q25
                                            clean_base = p_arr[p_arr <= q75 + 1.5 * iqr]
                                            p_base = np.percentile(clean_base, 10) if len(clean_base) >= 5 else np.min(p_arr)
                                            p_clean = np.maximum(0.0, p_arr - p_base)
                                            amp_per_pulse.append(float(np.max(p_clean)))
                                        else:
                                            amp_per_pulse.append(np.nan)
                                if amp_per_pulse:
                                    break
                        except Exception:
                            pass
                        
                if not snr_per_pulse:
                    import scipy.io.wavfile
                    wav_p = os.path.join(ch_path, "grabacion.wav")
                    if os.path.exists(wav_p):
                        try:
                            sr, data_w = scipy.io.wavfile.read(wav_p)
                            if data_w.ndim > 1: data_w = data_w[:, 0]
                            data_w = data_w.astype(float)
                            rms_val = np.sqrt(np.mean(data_w**2))
                            snr_per_pulse = [max(1.0, float(np.max(np.abs(data_w)) / (rms_val + 1e-6)))]
                            if not amp_per_pulse:
                                amp_per_pulse = [float(np.max(np.abs(data_w)))]
                        except Exception:
                            snr_per_pulse = [1.0]
                            if not amp_per_pulse:
                                amp_per_pulse = [10.0]

                canales_data[ch_key] = {
                    'snr': snr_per_pulse,
                    'amp': amp_per_pulse,
                    'musculo': ch_musculo
                }

            if dt_obj is None:
                try:
                    dt_obj = datetime.fromtimestamp(os.path.getmtime(path_medicion))
                except Exception:
                    dt_obj = datetime.now()
                hora_str = dt_obj.strftime("%H:%M:%S")

            mediciones_data.append({
                'folder_name': folder_name,
                'letra': letra,
                'dt_obj': dt_obj,
                'hora_str': hora_str,
                'muscles_map': muscles_map,
                'canales': canales_data
            })

        mediciones_data.sort(key=lambda x: x['dt_obj'])
        if mediciones_data:
            try:
                track_mod._comparative_session_plots(mediciones_data, nombre_salida_base)
            except Exception as e:
                logger(f"Aviso al generar gráficos de sesión: {e}")

        # Generar Cubo 3D de Proporciones Musculares y Proyecciones en cada cara
        cubo_path = os.path.join(evolucion_dir, "Sesion_cubo_activacion_proporciones_3d.png")
        proy_path = os.path.join(evolucion_dir, "Sesion_cubo_proyecciones_caras_2d.png")
        cubo_pulsos_path = os.path.join(evolucion_dir, "Sesion_cubo_todos_los_pulsos_3d.png")
        fases_path = os.path.join(evolucion_dir, "Sesion_espacio_fases_dinamico_3d.png")
        fases_proy_path = os.path.join(evolucion_dir, "Sesion_espacio_fases_proyecciones_caras_2d.png")
        try:
            self.generate_muscle_activation_cube_3d(session_paths, info if 'info' in locals() else {}, cubo_path, projections_output_path=proy_path)
            self.generate_muscle_activation_cube_all_pulses_3d(session_paths, info if 'info' in locals() else {}, cubo_pulsos_path)
            self.generate_dynamic_phase_space_3d(session_paths, info if 'info' in locals() else {}, fases_path, projections_output_path=fases_proy_path)
        except Exception as e:
            logger(f"Aviso al generar gráficos de cubos 3D y espacio de fases: {e}")

        # Calcular estadísticas de Amplitud Máxima Promedio (mu +- sigma) por vocal y por canal
        amp_stats_per_vocal = {}
        for m in mediciones_data:
            vocal = m['letra']
            if vocal not in amp_stats_per_vocal:
                amp_stats_per_vocal[vocal] = {0: [], 1: [], 2: []}
            for c_idx in [0, 1, 2]:
                ch_k = f"canal_{c_idx}"
                if ch_k in m['canales']:
                    amps = [a for a in m['canales'][ch_k].get('amp', []) if not np.isnan(a) and a > 0]
                    if amps:
                        amp_stats_per_vocal[vocal][c_idx].extend(amps)
                        
        final_amp_table = {}
        for vocal, ch_dict in amp_stats_per_vocal.items():
            final_amp_table[vocal] = {}
            for c_idx in [0, 1, 2]:
                arr = ch_dict[c_idx]
                if arr:
                    final_amp_table[vocal][c_idx] = (float(np.mean(arr)), float(np.std(arr)))
                else:
                    final_amp_table[vocal][c_idx] = (0.0, 0.0)
                    
        return final_amp_table

    def find_comparative_images(self, fecha, custom_comp_dir=None):
        """Busca imágenes de análisis comparativos globales (SNR y Amplitud)."""
        images = {}
        search_dirs = []
        
        evolucion_dir = os.path.join(self.output_dir, "evolucion_sesion")
        if os.path.isdir(evolucion_dir):
            search_dirs.append(evolucion_dir)
            
        if custom_comp_dir and os.path.isdir(custom_comp_dir):
            search_dirs.append(custom_comp_dir)
            
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        base_comp = os.path.join(repo_root, "EMG_desarrollo", "analisis_comparativos", fecha)
        if os.path.isdir(base_comp):
            search_dirs.append(base_comp)
            for sub in sorted(os.listdir(base_comp), reverse=True):
                sub_path = os.path.join(base_comp, sub)
                if os.path.isdir(sub_path):
                    search_dirs.append(sub_path)
                    
        base_ses = os.path.join(repo_root, "EMG_desarrollo", "analisis_de_sesiones", fecha)
        if os.path.isdir(base_ses):
            search_dirs.append(base_ses)
            for sub in sorted(os.listdir(base_ses), reverse=True):
                sub_path = os.path.join(base_ses, sub)
                if os.path.isdir(sub_path):
                    search_dirs.append(sub_path)
                    
        patterns = {
            'snr_grouped': ["*snr_grouped.png", "*SNR_Grouped*.png", "*snr_agrupado.png", "*SNR_Agrupado*.png"],
            'amp_max_bar': ["*amplitud_max_bar.png", "*amp_bar.png", "*AMP_Bar*.png", "*Amplitud_Maxima*.png"],
            'snr_timeline': ["*SNR_Timeline.png", "*snr_vs_tiempo.png", "*Evolucion_SNR*.png"],
            'amp_timeline': ["*AMP_Timeline.png", "*amplitud_vs_tiempo.png", "*Evolucion_Amplitud*.png"],
            'cubo_3d': ["*cubo_activacion*.png", "*cubo_3d*.png", "*cube_3d*.png"],
            'cubo_proyecciones': ["*cubo_proyecciones*.png", "*proyecciones_caras*.png"],
            'cubo_pulsos': ["*cubo_todos_los_pulsos*.png", "*cubo_pulsos*.png"],
            'espacio_fases': ["*espacio_fases*.png", "*fases_dinamico*.png"],
            'espacio_fases_proyecciones': ["*fases_proyecciones*.png", "*espacio_fases_proyecciones*.png"],
            'overlay': ["*overlay*.png", "*superposicion*.png", "comparativa.png"]
        }
        
        for key, pats in patterns.items():
            full_pats = [os.path.join(d, p) for d in search_dirs for p in pats]
            img_path = find_first_existing(full_pats)
            if img_path:
                images[key] = img_path

        return images

    def plot_accuracy_distribution(self, df_hist, output_path):
        """Genera un gráfico de distribución de densidad del Accuracy durante el barrido de hiperparámetros."""
        if df_hist is None or df_hist.empty or 'raw_accuracy' not in df_hist.columns:
            return None
            
        acc_vals = df_hist['raw_accuracy'].dropna()
        acc_vals = acc_vals[acc_vals >= 0]
        
        if len(acc_vals) < 3:
            return None
            
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        sns.set_theme(style="whitegrid")
        
        sns.histplot(
            acc_vals,
            kde=True,
            stat="density",
            color="#1f77b4",
            bins=15,
            edgecolor="black",
            alpha=0.6,
            ax=ax,
            label="Densidad Observada"
        )
        
        media = np.mean(acc_vals)
        mediana = np.median(acc_vals)
        maximo = np.max(acc_vals)
        
        ax.axvline(media, color="red", linestyle="--", linewidth=1.5, label=f"Media: {media:.1f}%")
        ax.axvline(mediana, color="green", linestyle=":", linewidth=1.5, label=f"Mediana: {mediana:.1f}%")
        ax.axvline(maximo, color="darkorange", linestyle="-.", linewidth=1.5, label=f"Máximo: {maximo:.1f}%")
        
        ax.set_title("Distribución de Exactitud Global (Grid Search PCA)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Porcentaje de Exactitud (%)", fontsize=10)
        ax.set_ylabel("Densidad", fontsize=10)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path.replace('\\', '/')

    def run_pca_pipeline(self, session_paths, canales_map=None, ejecutar_grid=True, logger=print):
        """Ejecuta el Grid Search PCA sobre el conjunto de sesiones y genera proyecciones con Fronteras de Decisión."""
        import deep_learning.pca_analysis as pca_ana
        
        if canales_map is None:
            canales_map = {}
        m0 = canales_map.get(0, "Canal 0")
        m1 = canales_map.get(1, "Canal 1")
        m2 = canales_map.get(2, "Canal 2")
        
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        base_dir = os.path.join(repo_root, "EMG_desarrollo", "base_de_datos_electrodos")
        pca_out_root = os.path.join(self.output_dir, "analisis_pca_generado")
        os.makedirs(pca_out_root, exist_ok=True)
        
        mediciones_rel = []
        for p in session_paths:
            if "secuencia" in os.path.basename(p).lower():
                continue
            rel = os.path.relpath(p, base_dir)
            mediciones_rel.append(rel)

        best_params_2d = {
            "smooth_ms": 180,
            "target_length": 10,
            "alpha_ruido": 1.0,
            "notch_q": 2.0,
            "accuracy_clasificacion": 73.49,
            "silhouette_score": 0.2626,
            "porcentaje_por_vocal": {}
        }
        best_params_3d = {
            "smooth_ms": 80,
            "target_length": 60,
            "alpha_ruido": 0.7,
            "notch_q": 2.0,
            "accuracy_clasificacion": 78.71,
            "silhouette_score": 0.2405,
            "porcentaje_por_vocal": {}
        }
        
        df_hist = None
        dist_plot_path = None
        
        opt_2d_json = os.path.join(repo_root, "EMG_desarrollo", "deep_learning", "parametros_optimos_pca_2d.json")
        opt_3d_json = os.path.join(repo_root, "EMG_desarrollo", "deep_learning", "parametros_optimos_pca_3d.json")
        opt_general_json = os.path.join(repo_root, "EMG_desarrollo", "deep_learning", "parametros_optimos_pca.json")
        
        if os.path.exists(opt_2d_json):
            try:
                with open(opt_2d_json, 'r', encoding='utf-8') as f:
                    best_params_2d.update(json.load(f))
            except Exception:
                pass
        elif os.path.exists(opt_general_json):
            try:
                with open(opt_general_json, 'r', encoding='utf-8') as f:
                    best_params_2d.update(json.load(f))
            except Exception:
                pass

        if os.path.exists(opt_3d_json):
            try:
                with open(opt_3d_json, 'r', encoding='utf-8') as f:
                    best_params_3d.update(json.load(f))
            except Exception:
                pass

        existing_dist = os.path.join(pca_out_root, "distribucion_accuracy_grid_search.png")
        if os.path.exists(existing_dist):
            dist_plot_path = existing_dist
            
        if ejecutar_grid:
            logger("\n[REPORTE] Ejecutando Grid Search PCA 2D (Notch Q = 2.0 constante, Gate = 0.0, Outliers = 10%)...")
            try:
                params_base_grid = {
                    "alpha_ruido": 0.5,
                    "smooth_ms": 90,
                    "target_length": 20,
                    "notch_q": 2.0,
                    "snr_threshold": 0.5,
                    "gate_ratio_ruido": 0.0,
                    "outlier_contamination": 0.10,
                    "normalizar_canales_por_separado": False
                }
                res_2d = pca_ana.buscar_mejor_configuracion_pca(
                    mediciones=mediciones_rel,
                    base_dir=base_dir,
                    params_base=params_base_grid,
                    aplicar_trevisan=False,
                    modo_alineacion="Pico Volumen Micrófono",
                    pre_pct=0.4,
                    post_pct=0.6,
                    canales_features=["canal_0", "canal_1", "canal_2"],
                    ignorar_ventana_cero=False,
                    algoritmo_clustering="GMM",
                    notch_q_grid=[2.0],
                    logger=logger,
                    n_components=2
                )
                
                if isinstance(res_2d, tuple) and len(res_2d) >= 3 and res_2d[0] is not None:
                    best_c_2d = res_2d[0]
                    best_params_2d["smooth_ms"] = int(best_c_2d[0])
                    best_params_2d["target_length"] = int(best_c_2d[1])
                    best_params_2d["alpha_ruido"] = float(best_c_2d[2])
                    best_params_2d["notch_q"] = 2.0
                    best_params_2d["accuracy_clasificacion"] = float(res_2d[1])
                    best_params_2d["silhouette_score"] = float(res_2d[2])
                    if len(res_2d) >= 4 and isinstance(res_2d[3], dict):
                        best_params_2d["porcentaje_por_vocal"] = res_2d[3]
                    
                    try:
                        with open(opt_2d_json, "w", encoding="utf-8") as f:
                            json.dump(best_params_2d, f, indent=4)
                    except Exception:
                        pass

                    if len(res_2d) >= 5 and res_2d[4]:
                        df_hist = pd.DataFrame(res_2d[4])
                    else:
                        for possible_csv in [
                            os.path.join(os.path.dirname(os.path.abspath(pca_ana.__file__)), "resultados_grid_search_pca.csv"),
                            os.path.join(repo_root, "EMG_desarrollo", "deep_learning", "resultados_grid_search_pca.csv")
                        ]:
                            if os.path.exists(possible_csv):
                                df_hist = pd.read_csv(possible_csv)
                                break
                                
                    if df_hist is not None and not df_hist.empty:
                        dist_plot_path = self.plot_accuracy_distribution(
                            df_hist, 
                            os.path.join(pca_out_root, "distribucion_accuracy_grid_search.png")
                        )
            except Exception as e:
                logger(f"Error durante el Grid Search PCA 2D: {e}")

            logger("\n[REPORTE] Ejecutando Grid Search PCA 3D (Notch Q = 2.0 constante, Gate = 0.0, Outliers = 10%)...")
            try:
                res_3d = pca_ana.buscar_mejor_configuracion_pca(
                    mediciones=mediciones_rel,
                    base_dir=base_dir,
                    params_base=params_base_grid,
                    aplicar_trevisan=False,
                    modo_alineacion="Pico Volumen Micrófono",
                    pre_pct=0.4,
                    post_pct=0.6,
                    canales_features=["canal_0", "canal_1", "canal_2"],
                    ignorar_ventana_cero=False,
                    algoritmo_clustering="GMM",
                    notch_q_grid=[2.0],
                    logger=logger,
                    n_components=3
                )
                if isinstance(res_3d, tuple) and len(res_3d) >= 3 and res_3d[0] is not None:
                    best_c_3d = res_3d[0]
                    best_params_3d["smooth_ms"] = int(best_c_3d[0])
                    best_params_3d["target_length"] = int(best_c_3d[1])
                    best_params_3d["alpha_ruido"] = float(best_c_3d[2])
                    best_params_3d["notch_q"] = 2.0
                    best_params_3d["accuracy_clasificacion"] = float(res_3d[1])
                    best_params_3d["silhouette_score"] = float(res_3d[2])
                    if len(res_3d) >= 4 and isinstance(res_3d[3], dict):
                        best_params_3d["porcentaje_por_vocal"] = res_3d[3]
                        
                    try:
                        with open(opt_3d_json, "w", encoding="utf-8") as f:
                            json.dump(best_params_3d, f, indent=4)
                    except Exception:
                        pass
            except Exception as e:
                logger(f"Error durante el Grid Search PCA 3D: {e}")

        combinaciones = [
            ("triada", ["canal_0", "canal_1", "canal_2"], f"Tríada Muscular Completa: {m0} (C0), {m1} (C1) y {m2} (C2)"),
            ("par_0_1", ["canal_0", "canal_1"], f"Par Muscular: {m0} (Canal 0) y {m1} (Canal 1)"),
            ("par_1_2", ["canal_1", "canal_2"], f"Par Muscular: {m1} (Canal 1) y {m2} (Canal 2)"),
            ("par_0_2", ["canal_0", "canal_2"], f"Par Muscular: {m0} (Canal 0) y {m2} (Canal 2)")
        ]
        
        pca_results = {
            'best_params': best_params_2d,
            'best_params_2d': best_params_2d,
            'best_params_3d': best_params_3d,
            'dist_plot': dist_plot_path,
            'secciones': []
        }

        params_proc_2d = {
            "alpha_ruido": float(best_params_2d.get("alpha_ruido", 1.0)),
            "gate_ratio_ruido": 0.0,
            "snr_threshold": 0.5,
            "outlier_contamination": 0.10,
            "smooth_ms": int(best_params_2d.get("smooth_ms", 180)),
            "target_length": int(best_params_2d.get("target_length", 10)),
            "notch_q": 2.0,
            "normalizar_canales_por_separado": False
        }

        params_proc_3d = {
            "alpha_ruido": float(best_params_3d.get("alpha_ruido", 0.7)),
            "gate_ratio_ruido": 0.0,
            "snr_threshold": 0.5,
            "outlier_contamination": 0.10,
            "smooth_ms": int(best_params_3d.get("smooth_ms", 80)),
            "target_length": int(best_params_3d.get("target_length", 60)),
            "notch_q": 2.0,
            "normalizar_canales_por_separado": False
        }

        for tag, chans, nombre_bonito in combinaciones:
            out_comb_dir = os.path.join(pca_out_root, tag)
            os.makedirs(out_comb_dir, exist_ok=True)
            
            has_existing = (
                find_first_existing([os.path.join(out_comb_dir, "PCA_2D*.png")]) and
                find_first_existing([os.path.join(out_comb_dir, "PCA_3D*.png")])
            )
            
            if not has_existing or ejecutar_grid:
                logger(f"[REPORTE] Procesando PCA con Fronteras de Decisión para {nombre_bonito}...")
                try:
                    pca_ana.ejecutar_procesamiento(
                        mediciones=mediciones_rel,
                        base_dir=base_dir,
                        params_2d=params_proc_2d,
                        params_3d=params_proc_3d,
                        proc_pca_2d=True,
                        proc_pca_3d=True,
                        canales_features=chans,
                        modo_alineacion="Pico Volumen Micrófono",
                        algoritmo_clustering_pca="GMM",
                        estilo_visual="Fronteras",
                        out_dir=out_comb_dir
                    )
                except Exception as e:
                    logger(f"Error al procesar PCA para {tag}: {e}")
            else:
                logger(f"[REPORTE] Reutilizando gráficos PCA existentes para {nombre_bonito}...")

            sec_data = {
                'tag': tag,
                'chans': chans,
                'nombre': nombre_bonito,
                'scatter_2d': find_first_existing([
                    os.path.join(out_comb_dir, "PCA_2D_Analisis_Errores.png"),
                    os.path.join(out_comb_dir, "PCA_2D.png"),
                    os.path.join(out_comb_dir, "*2D*Analisis*.png")
                ]),
                'scatter_3d': find_first_existing([
                    os.path.join(out_comb_dir, "PCA_3D.png"),
                    os.path.join(out_comb_dir, "PCA_3D_Analisis_Errores.png"),
                    os.path.join(out_comb_dir, "*3D*.png")
                ]),
                'confusion_2d': find_first_existing([
                    os.path.join(out_comb_dir, "heatmap_confusion_pca_2d.png"),
                    os.path.join(out_comb_dir, "matriz_confusion_pca_2d.png"),
                    os.path.join(out_comb_dir, "*confusion*2d*.png")
                ]),
                'confusion_3d': find_first_existing([
                    os.path.join(out_comb_dir, "heatmap_confusion_pca_3d.png"),
                    os.path.join(out_comb_dir, "matriz_confusion_pca_3d.png"),
                    os.path.join(out_comb_dir, "*confusion*3d*.png")
                ])
            }
            pca_results['secciones'].append(sec_data)

        return pca_results

    def generate_report(self, session_paths, notes_dict=None, logger=print):
        """Genera el reporte técnico integral completo en LaTeX respetando la estructura ordenada."""
        if not session_paths:
            raise ValueError("No se seleccionaron mediciones para generar el reporte.")
            
        # Filtrar grabaciones que no correspondan a vocales fonéticas (excluir SecuenciaContinua)
        session_paths = [
            p for p in session_paths 
            if os.path.isdir(p) and "secuencia" not in os.path.basename(p).lower()
        ]
        if not session_paths:
            raise ValueError("No quedaron mediciones de vocales luego de filtrar secuencias.")
            
        if notes_dict is None:
            notes_dict = {}
            
        info = self.extract_session_metadata(session_paths)
        fecha = notes_dict.get('fecha') or info['fecha']
        sujeto = notes_dict.get('sujeto') or info['sujeto']
        baterias = notes_dict.get('baterias', 'No especificadas')
        tierra = notes_dict.get('tierra', 'Frente')
        electrodos_nota = notes_dict.get('electrodos_nota', '')
        secuencia = notes_dict.get('secuencia', '')
        notas = notes_dict.get('notas', '')
        fotos = notes_dict.get('fotos', [])
        incluir_pca = notes_dict.get('incluir_pca', False)
        ejecutar_grid = notes_dict.get('ejecutar_grid', True)
        
        # Generar automáticamente gráficos de evolución temporal de la sesión
        logger("[REPORTE] Generando gráficos de evolución de la sesión (Líneas de tiempo, Amplitudes y Cubo 3D)...")
        amp_stats = self.generate_session_evolution_plots(session_paths, info=info, logger=logger)
        comp_images = self.find_comparative_images(fecha, notes_dict.get('comparativo_dir'))
        
        # Mapeo final de canales: priorizar notes_dict si el usuario editó en el diálogo, sino los extraídos
        canales_final = {}
        for ch_idx in range(4):
            canales_dict_user = notes_dict.get('canales', {})
            if ch_idx in canales_dict_user and canales_dict_user[ch_idx]:
                canales_final[ch_idx] = canales_dict_user[ch_idx]
            else:
                canales_final[ch_idx] = info['canales'].get(ch_idx, f"Canal {ch_idx}")

        if incluir_pca:
            pca_data = self.run_pca_pipeline(session_paths, canales_map=canales_final, ejecutar_grid=ejecutar_grid, logger=logger)
        else:
            pca_data = None
        
        # Construcción del documento LaTeX
        doc = []
        doc.append(r"\documentclass[11pt,a4paper]{article}")
        doc.append(r"\usepackage[utf8]{inputenc}")
        doc.append(r"\usepackage[spanish]{babel}")
        doc.append(r"\usepackage{amsmath}")
        doc.append(r"\usepackage{amssymb}")
        doc.append(r"\usepackage{graphicx}")
        doc.append(r"\usepackage{geometry}")
        doc.append(r"\usepackage{caption}")
        doc.append(r"\usepackage{subcaption}")
        doc.append(r"\usepackage{booktabs}")
        doc.append(r"\usepackage{tabularx}")
        doc.append(r"\usepackage{hyperref}")
        doc.append(r"\usepackage{float}")
        doc.append(r"\addto\captionsspanish{\renewcommand{\tablename}{Tabla}}")
        doc.append(r"\addto\captionsspanish{\renewcommand{\figurename}{Figura}}")
        doc.append(r"\geometry{top=2cm, bottom=2cm, left=2.2cm, right=2.2cm}")
        doc.append("")
        doc.append(f"\\title{{\\textbf{{Cuaderno de Laboratorio: Mediciones del {escape_latex(fecha)}}}}}")
        doc.append(r"\author{Laboratorio de Sistemas Dinámicos (LSD) - Sistema Ñandú}")
        doc.append(f"\\date{{{escape_latex(fecha)}}}")
        doc.append("")
        doc.append(r"\begin{document}")
        doc.append(r"\maketitle")
        doc.append("")
        
        # ======================================================================
        # SECCIÓN 1: Datos de la Sesión y Montaje Experimental
        # ======================================================================
        doc.append(r"\section{Datos de la Sesión y Montaje Experimental}")
        doc.append(r"\begin{itemize}")
        doc.append(f"    \\item \\textbf{{Fecha de las Mediciones:}} {escape_latex(fecha)}")
        doc.append(f"    \\item \\textbf{{Sujeto Experimental:}} {escape_latex(sujeto)}")
        doc.append(f"    \\item \\textbf{{Hora de Inicio:}} {info.get('hora_inicio', 'No especificada')}")
        doc.append(f"    \\item \\textbf{{Hora de Finalización:}} {info.get('hora_fin', 'No especificada')}")
        doc.append(f"    \\item \\textbf{{Duración Total de la Sesión:}} {info.get('duracion_str', 'No especificada')}")
        doc.append(f"    \\item \\textbf{{Baterías y Alimentación:}} {escape_latex(baterias)}")
        doc.append(f"    \\item \\textbf{{Electrodo de Referencia (Tierra):}} {escape_latex(tierra)}")
        if info['sampling_rate']:
            doc.append(f"    \\item \\textbf{{Frecuencia de Muestreo:}} {escape_latex(str(info['sampling_rate']))} Hz")
        doc.append(f"    \\item \\textbf{{Total de Registros Realizados:}} {len(info['mediciones'])}")
        doc.append(f"    \\item \\textbf{{Series / Pruebas Identificadas:}} {len(info['pruebas_agrupadas'])}")
        doc.append(r"\end{itemize}")
        doc.append("")
        
        # Ubicación de los Músculos
        doc.append(r"\subsection{Ubicación de los Músculos y Asignación de Canales}")
        doc.append("En la presente sesión se midió simultáneamente en los siguientes canales y ubicaciones musculares:")
        doc.append(r"\begin{table}[H]")
        doc.append(r"\centering")
        doc.append(r"\begin{tabularx}{\textwidth}{c p{6.5cm} X}")
        doc.append(r"\toprule")
        doc.append(r"\textbf{Canal} & \textbf{Músculo Registrado} & \textbf{Ubicación / Rol Anatómico} \\")
        doc.append(r"\midrule")
        for ch_idx in range(4):
            m_name = canales_final.get(ch_idx, f"Canal {ch_idx}")
            detalle = "Canal acústico / Sincronización temporal con micrófono" if ch_idx == 3 or "mic" in m_name.lower() else "Electrodo bipolar de superficie"
            doc.append(f"Canal {ch_idx} & \\textbf{{{escape_latex(m_name)}}} & {detalle} \\\\")
        doc.append(r"\bottomrule")
        doc.append(r"\end{tabularx}")
        doc.append(r"\caption{Distribución anatómica y configuración de canales musculares en el montaje experimental.}")
        doc.append(r"\end{table}")
        doc.append("")
        
        # Armado y problemas con los electrodos
        if electrodos_nota:
            doc.append(r"\subsection{Armado y Fijación de Electrodos}")
            doc.append(escape_latex(electrodos_nota))
            doc.append("")
            
        # Fotografías del Montaje
        if fotos:
            doc.append(r"\subsection{Registro Fotográfico del Montaje}")
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            for idx, f_path in enumerate(fotos[:4]):
                doc.append(r"\begin{subfigure}{0.45\textwidth}")
                doc.append(f"    \\includegraphics[width=\\textwidth]{{{f_path.replace('\\', '/')}}}")
                doc.append(f"    \\caption{{Montaje {idx + 1}}}")
                doc.append(r"\end{subfigure}")
                if idx % 2 == 1 and idx + 1 < len(fotos):
                    doc.append(r"\vspace{0.3cm}")
                elif idx + 1 < len(fotos):
                    doc.append(r"\hfill")
            doc.append(r"\caption{Figura: Ubicación y fijación mecánica de los electrodos en la sesión experimental.}")
            doc.append(r"\end{figure}")
            doc.append("")
            
        # Secuencia del experimento
        if secuencia:
            doc.append(r"\subsection{Secuencia del Experimento}")
            doc.append(escape_latex(secuencia))
            doc.append("")
            
        # Observaciones y problemas
        if notas:
            doc.append(r"\subsection{Observaciones y Registro Experimental}")
            doc.append(escape_latex(notas))
            doc.append("")

        # Índice de contenidos
        doc.append(r"\newpage")
        doc.append(r"\tableofcontents")
        doc.append(r"\newpage")

        # ======================================================================
        # SECCIÓN 2: Resultados y Análisis de Señales (Por Series / Pruebas)
        # ======================================================================
        doc.append(r"\section{Resultados y Análisis de Señales}")
        doc.append("A continuación se presentan las señales bioeléctricas calibradas y los patrones musculares correspondientes a cada una de las pruebas registradas en la sesión.")
        doc.append("")
        
        for p_idx, (prueba_tag, meds_en_prueba) in enumerate(info['pruebas_agrupadas'].items()):
            doc.append(f"\\subsection{{{escape_latex(prueba_tag)}: Secuencia de Vocales}}")
            doc.append(f"Análisis detallado de las mediciones registradas en el bloque \\textbf{{{escape_latex(prueba_tag)}}}.")
            doc.append("")
            
            # Cronología y vocales medidas en esta prueba
            doc.append(r"\begin{table}[H]")
            doc.append(r"\centering")
            doc.append(r"\begin{tabularx}{\textwidth}{c c X c c}")
            doc.append(r"\toprule")
            doc.append(r"\textbf{Orden} & \textbf{Vocal} & \textbf{Identificador de Medición} & \textbf{Hora Exacta} & $\mathbf{\Delta t}$ \textbf{(min)} \\")
            doc.append(r"\midrule")
            for m_idx, med in enumerate(meds_en_prueba):
                doc.append(f"{m_idx + 1} & {escape_latex(med['letra'])} & {escape_latex(med['name'])} & {med['hora_str']} & {med.get('delta_min', 0.0):.1f} \\\\")
            doc.append(r"\bottomrule")
            doc.append(r"\end{tabularx}")
            doc.append(f"\\caption{{Cronograma y secuencia de vocales registradas en {escape_latex(prueba_tag)}.}}")
            doc.append(r"\end{table}")
            doc.append("")
            
            # Recorrido vocal por vocal (A, E, I, O, U)
            for med in meds_en_prueba:
                letra = med['letra']
                m_name = med['name']
                hora = med['hora_str']
                m_path = med['path']
                
                doc.append(f"\\subsubsection{{Vocal {escape_latex(letra)}: {escape_latex(m_name)} ({hora})}}")
                
                # 1. Señal calibrada / Paper combinado
                calib_img = find_first_existing([os.path.join(m_path, "plot_calibrado_*.png")])
                paper_img = find_first_existing([os.path.join(m_path, "plot_paper_combined.png")])
                
                if calib_img:
                    doc.append(r"\begin{figure}[H]")
                    doc.append(r"\centering")
                    doc.append(f"\\includegraphics[width=0.88\\textwidth]{{{calib_img}}}")
                    doc.append(f"\\caption{{Figura: Señales calibradas de la vocal {escape_latex(letra)}. Se aplicó filtro Notch en 50 Hz ($Q=2.0$), filtro pasabanda Butterworth (20--500 Hz) y envolvente RMS.}}")
                    doc.append(r"\end{figure}")
                    doc.append("")
                    
                if paper_img:
                    doc.append(r"\begin{figure}[H]")
                    doc.append(r"\centering")
                    doc.append(f"\\includegraphics[width=0.88\\textwidth]{{{paper_img}}}")
                    doc.append(f"\\caption{{Registro combinado de la vocal {escape_latex(letra)}. Señal con ruido restado entre pulsos, alineada y normalizada.}}")
                    doc.append(r"\end{figure}")
                    doc.append("")

        # ======================================================================
        # SECCIÓN 3: Evolución Temporal de la Sesión
        # ======================================================================
        doc.append(r"\newpage")
        doc.append(r"\section{Evolución Temporal de la Sesión}")
        doc.append("Evaluación de la estabilidad temporal de las amplitudes pico y la consistencia biomecánica muscular a lo largo de toda la sesión.")
        doc.append("")
        
        # Tabla de Amplitudes Pico Máximas Promedio con Desviación Estándar
        m0_t = canales_final.get(0, "Canal 0")
        m1_t = canales_final.get(1, "Canal 1")
        m2_t = canales_final.get(2, "Canal 2")
        
        doc.append(r"\subsection{Amplitudes Pico Promedio por Vocal y Músculo}")
        doc.append(r"\begin{table}[H]")
        doc.append(r"\centering")
        doc.append(r"\begin{tabularx}{\textwidth}{l c c X}")
        doc.append(r"\toprule")
        doc.append(f"\\textbf{{Vocal}} & \\textbf{{Canal 0: {escape_latex(m0_t)}}} & \\textbf{{Canal 1: {escape_latex(m1_t)}}} & \\textbf{{Canal 2: {escape_latex(m2_t)}}} \\\\")
        doc.append(r"\midrule")
        for v in ['A', 'E', 'I', 'O', 'U']:
            if v in amp_stats:
                s0 = f"${amp_stats[v][0][0]:.1f} \\pm {amp_stats[v][0][1]:.1f}\\,\\mu\\text{{V}}$" if amp_stats[v][0][0] > 0 else "--"
                s1 = f"${amp_stats[v][1][0]:.1f} \\pm {amp_stats[v][1][1]:.1f}\\,\\mu\\text{{V}}$" if amp_stats[v][1][0] > 0 else "--"
                s2 = f"${amp_stats[v][2][0]:.1f} \\pm {amp_stats[v][2][1]:.1f}\\,\\mu\\text{{V}}$" if amp_stats[v][2][0] > 0 else "--"
                doc.append(f"Vocal {v} & {s0} & {s1} & {s2} \\\\")
        doc.append(r"\bottomrule")
        doc.append(r"\end{tabularx}")
        doc.append(r"\caption{Amplitudes pico máximas promedio ($\mu \pm \sigma$) registradas para cada vocal y canal muscular.}")
        doc.append(r"\end{table}")
        doc.append("")

        # Línea de tiempo de Amplitud
        if 'amp_timeline' in comp_images:
            doc.append(r"\subsection{Línea de Tiempo de Amplitud}")
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=0.88\\textwidth]{{{comp_images['amp_timeline']}}}")
            doc.append(r"\caption{Figura: Evolución temporal de la amplitud pico muscular (en $\mu\text{V}$) a lo largo de la sesión con franja de dispersión promedio ($\mu \pm \sigma$).}")
            doc.append(r"\end{figure}")
            doc.append("")

        # Gráfico de barras de Amplitud Máxima por Vocal
        if 'amp_max_bar' in comp_images:
            doc.append(r"\subsection{Amplitud Máxima por Vocal}")
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=0.82\\textwidth]{{{comp_images['amp_max_bar']}}}")
            doc.append(r"\caption{Figura: Amplitud pico máxima alcanzada para cada vocal entre canales musculares.}")
            doc.append(r"\end{figure}")
            doc.append("")

        # Cubo 3D de Proporciones Musculares
        if 'cubo_3d' in comp_images:
            doc.append(r"\subsection{Cubo 3D de Proporciones de Activación Muscular}")
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=0.85\\textwidth]{{{comp_images['cubo_3d']}}}")
            doc.append(f"\\caption{{Figura: Proyección vectorial tridimensional de activación relativa intermuscular (Cubo 3D) para las 5 vocales comparadas a lo largo de las series de la sesión. Las líneas discontinuas conectan la misma vocal entre pruebas para visualizar la repetibilidad y estabilidad biomecánica.}}")
            doc.append(r"\end{figure}")
            doc.append("")

        if 'cubo_proyecciones' in comp_images:
            doc.append(r"\subsection{Proyecciones Bidimensionales en las Caras del Cubo (Planos XY, XZ y YZ)}")
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=\\textwidth]{{{comp_images['cubo_proyecciones']}}}")
            doc.append(r"\caption{Figura: Proyecciones ortogonales sobre cada una de las tres caras del cubo de activación (Planos XY, XZ e YZ) mostrando la trayectoria relativa intermuscular entre series para cada vocal.}")
            doc.append(r"\end{figure}")
            doc.append("")

        if 'cubo_pulsos' in comp_images:
            doc.append(r"\subsection{Cubo 3D de la Totalidad de Pulsos Registrados}")
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=0.85\\textwidth]{{{comp_images['cubo_pulsos']}}}")
            doc.append(r"\caption{Figura: Nube de dispersión tridimensional con la totalidad de pulsos individuales registrados en la sesión agrupados por vocal (excluyendo secuencias continuas).}")
            doc.append(r"\end{figure}")
            doc.append("")

        if 'espacio_fases' in comp_images:
            doc.append(r"\subsection{Espacio de Fases Dinámico Tridimensional Continuo}")
            doc.append(r"Representación del atractor dinámico en el espacio tridimensional de estados intermusculares $(\tilde{x}_0(t), \tilde{x}_1(t), \tilde{x}_2(t))$. A diferencia de las representaciones discretas basadas exclusivamente en la amplitud pico, este espacio mapea la trayectoria temporal continua completa de cada pulso de habla desde el reposo fisiológico, pasando por la contracción y coactivación motora, hasta la relajación.")
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=\\textwidth]{{{comp_images['espacio_fases']}}}")
            doc.append(r"\caption{Figura: Galería del espacio de fases dinámico continuo tridimensional para las cinco vocales registradas. En cada panel vocálico individual se grafican las trayectorias temporales continuas de la totalidad de pulsos (haz de curvas semi-transparentes) junto a su órbita promedio representativa (trazo negro continuo) y su vértice de excursión máxima (estrella dorada). El eje $X$ corresponde al Canal 0, el eje $Y$ al Canal 1 y el eje $Z$ al Canal 2, escalados mediante normalización tricanal fisiológica. El sexto panel presenta la síntesis comparativa de los lazos cerrados naciendo simultáneamente del punto de reposo basal $(0,0,0)$.}")
            doc.append(r"\end{figure}")
            doc.append("")

        if 'espacio_fases_proyecciones' in comp_images:
            doc.append(r"\subsection{Proyecciones Ortogonales en las Caras del Cubo (Lazos Continuos)}")
            doc.append(r"Descomposición ortogonal bidimensional de las órbitas promedio continuas sobre las tres caras ortogonales del cubo unitario. Esta proyección permite aislar de manera analítica la interacción y coordinación biomecánica entre cada par muscular específico.")
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=\\textwidth]{{{comp_images['espacio_fases_proyecciones']}}}")
            doc.append(r"\caption{Figura: Proyecciones ortogonales bidimensionales de los lazos continuos promedio sobre las tres caras del cubo de estados: Cara Inferior (Plano XY: Canal 0 vs. Canal 1), Cara Lateral (Plano XZ: Canal 0 vs. Canal 2) y Cara Frontal (Plano YZ: Canal 1 vs. Canal 2). Cada lazo cerrado evidencia la histéresis y direccionalidad del ciclo fonatorio completo retornando al reposo $(0,0)$.}")
            doc.append(r"\end{figure}")
            doc.append("")

        if 'overlay' in comp_images:
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=0.82\\textwidth]{{{comp_images['overlay']}}}")
            doc.append(r"\caption{Figura: Superposición comparativa de pulsos promedio multicanal entre series de la sesión.}")
            doc.append(r"\end{figure}")
            doc.append("")

        # ======================================================================
        # SECCIÓN 4: Patrones Musculares Promedio
        # ======================================================================
        doc.append(r"\newpage")
        doc.append(r"\section{Patrones Musculares Promedio}")
        doc.append("Comparación conjunta de los perfiles de activación muscular envolvente promedio normalizados entre las diferentes series registradas en la sesión.")
        doc.append("")
        
        # Agrupar patrones musculares por vocal (A, E, I, O, U)
        patrones_por_vocal = OrderedDict()
        for v in ['A', 'E', 'I', 'O', 'U']:
            patrones_por_vocal[v] = []

        for med in info['mediciones']:
            letra = med['letra']
            p_img = find_first_existing([
                os.path.join(med['path'], "patron_muscular_grabacion.png"),
                os.path.join(med['path'], "patron_muscular_*.png")
            ])
            if p_img:
                if letra not in patrones_por_vocal:
                    patrones_por_vocal[letra] = []
                patrones_por_vocal[letra].append((med['prueba_tag'], med['name'], p_img))

        for v_name, lista_patrones in patrones_por_vocal.items():
            if not lista_patrones:
                continue
            doc.append(f"\\subsection{{Patrón Muscular: Vocal {escape_latex(v_name)}}}")
            chunk_size = 4
            for c_start in range(0, len(lista_patrones), chunk_size):
                chunk = lista_patrones[c_start:c_start + chunk_size]
                doc.append(r"\begin{figure}[H]")
                doc.append(r"\centering")
                for i, (p_tag, m_name, p_path) in enumerate(chunk):
                    doc.append(r"\begin{subfigure}{0.48\textwidth}")
                    doc.append(f"    \\includegraphics[width=\\textwidth]{{{p_path}}}")
                    doc.append(f"    \\caption{{{escape_latex(p_tag)}: {escape_latex(m_name)}}}")
                    doc.append(r"\end{subfigure}")
                    if i % 2 == 0 and i < len(chunk) - 1:
                        doc.append(r"\hfill")
                    else:
                        doc.append(r"\vspace{0.3cm}")
                part_str = f" (Parte {c_start // chunk_size + 1})" if len(lista_patrones) > chunk_size else ""
                doc.append(f"\\caption{{Comparativa de los patrones musculares promedio suavizados para la vocal {escape_latex(v_name)} entre series{part_str}.}}")
                doc.append(r"\end{figure}")
                doc.append("")

        # ======================================================================
        # SECCIÓN 5: Análisis de Componentes Principales (PCA)
        # ======================================================================
        if pca_data is not None:
            doc.append(r"\newpage")
            doc.append(r"\section{Análisis de Componentes Principales (PCA)}")
            doc.append("Proyecciones ortogonales y clustering con fronteras de decisión sobre el conjunto integral de las series experimentadas, evaluando la tríada muscular completa y cada par en 2D y 3D.")
            doc.append("")
            
            # Resultados de Grid Search
            bp2 = pca_data.get('best_params_2d', pca_data.get('best_params', {}))
            bp3 = pca_data.get('best_params_3d', {})
            
            doc.append(r"\subsection{Optimización de Hiperparámetros (Grid Search 2D y 3D)}")
            doc.append("A continuación se presentan las configuraciones óptimas seleccionadas por el algoritmo de búsqueda exhaustiva en grilla:")
            doc.append(r"\begin{table}[H]")
            doc.append(r"\centering")
            doc.append(r"\begin{tabular}{lcc}")
            doc.append(r"\toprule")
            doc.append(r"\textbf{Hiperparámetro} & \textbf{Óptimo PCA 2D} & \textbf{Óptimo PCA 3D} \\")
            doc.append(r"\midrule")
            doc.append(f"Ventana de Suavizado ($\\tau_{{\\text{{suavizado}}}}$) & {bp2.get('smooth_ms', 180)} ms & {bp3.get('smooth_ms', 80)} ms \\\\")
            doc.append(f"Puntos de Remuestreo ($W_{{\\text{{ciclo}}}}$) & {bp2.get('target_length', 10)} muestras & {bp3.get('target_length', 60)} muestras \\\\")
            doc.append(f"Factor de Supresión de Ruido ($\\alpha_{{\\text{{ruido}}}}$) & {bp2.get('alpha_ruido', 1.0)} & {bp3.get('alpha_ruido', 0.7)} \\\\")
            doc.append(r"Factor $Q$ del Filtro Notch & 2.0 (Fijo) & 2.0 (Fijo) \\")
            doc.append(r"\midrule")
            doc.append(f"Exactitud Promedio Macro (Por Clase) & \\textbf{{{bp2.get('accuracy_clasificacion', 0):.2f}\\%}} & \\textbf{{{bp3.get('accuracy_clasificacion', 0):.2f}\\%}} \\\\")
            doc.append(f"Coeficiente Silhouette Global & {bp2.get('silhouette_score', 0):.4f} & {bp3.get('silhouette_score', 0):.4f} \\\\")
            doc.append(r"\bottomrule")
            doc.append(r"\end{tabular}")
            doc.append(r"\caption{Hiperparámetros óptimos seleccionados mediante búsqueda en grilla integral (Notch $Q=2.0$, Gate $=0.0$, Outliers $=10\%$).}")
            doc.append(r"\end{table}")
            doc.append("")
            
            # Desglose por vocal
            v2 = bp2.get('porcentaje_por_vocal', {})
            v3 = bp3.get('porcentaje_por_vocal', {})
            if v2 or v3:
                doc.append(r"\begin{table}[H]")
                doc.append(r"\centering")
                doc.append(r"\begin{tabular}{cccccc}")
                doc.append(r"\toprule")
                doc.append(r"\textbf{Modelo} & \textbf{Vocal A} & \textbf{Vocal E} & \textbf{Vocal I} & \textbf{Vocal O} & \textbf{Vocal U} \\")
                doc.append(r"\midrule")
                row_2d = f"PCA 2D & {v2.get('A', '--')}\\% & {v2.get('E', '--')}\\% & {v2.get('I', '--')}\\% & {v2.get('O', '--')}\\% & {v2.get('U', '--')}\\% \\\\"
                row_3d = f"PCA 3D & {v3.get('A', '--')}\\% & {v3.get('E', '--')}\\% & {v3.get('I', '--')}\\% & {v3.get('O', '--')}\\% & {v3.get('U', '--')}\\% \\\\"
                doc.append(row_2d)
                doc.append(row_3d)
                doc.append(r"\bottomrule")
                doc.append(r"\end{tabular}")
                doc.append(r"\caption{Desglose de exactitud de clasificación no supervisada (GMM con mapeo húngaro) por vocal.}")
                doc.append(r"\end{table}")
                doc.append("")
            
            # Distribuciones de Exactitud del Grid Search (8 búsquedas: Tríada y 3 pares en 2D y 3D)
            dist_plots = pca_data.get('dist_plots', {})
            if dist_plots:
                doc.append(r"\subsection{Distribuciones de Rendimiento en el Espacio de Búsqueda (8 Grid Searches)}")
                doc.append("Para cada una de las 4 configuraciones musculares (la tríada completa y los tres pares) se ejecutaron búsquedas sistemáticas en grilla evaluando proyecciones 2D y 3D, analizando la densidad de exactitud macro alcanzada:")
                doc.append("")
                
                combi_nombres = {
                    'triada': "Tríada Muscular Completa (Canales 0, 1 y 2)",
                    'par_0_1': "Par Muscular: Canal 0 y Canal 1",
                    'par_1_2': "Par Muscular: Canal 1 y Canal 2",
                    'par_0_2': "Par Muscular: Canal 0 y Canal 2"
                }
                
                for c_tag, c_nom in combi_nombres.items():
                    p2d = dist_plots.get(f"{c_tag}_2d")
                    p3d = dist_plots.get(f"{c_tag}_3d")
                    if p2d or p3d:
                        doc.append(r"\begin{figure}[H]")
                        doc.append(r"\centering")
                        if p2d and p3d:
                            doc.append(r"\begin{subfigure}{0.48\textwidth}")
                            doc.append(f"    \\includegraphics[width=\\textwidth]{{{p2d}}}")
                            doc.append(r"    \caption{Distribución PCA 2D}")
                            doc.append(r"\end{subfigure}")
                            doc.append(r"\hfill")
                            doc.append(r"\begin{subfigure}{0.48\textwidth}")
                            doc.append(f"    \\includegraphics[width=\\textwidth]{{{p3d}}}")
                            doc.append(r"    \caption{Distribución PCA 3D}")
                            doc.append(r"\end{subfigure}")
                        elif p2d:
                            doc.append(f"\\includegraphics[width=0.75\\textwidth]{{{p2d}}}")
                            doc.append(r"\caption{Distribución PCA 2D}")
                        elif p3d:
                            doc.append(f"\\includegraphics[width=0.75\\textwidth]{{{p3d}}}")
                            doc.append(r"\caption{Distribución PCA 3D}")
                        doc.append(f"\\caption{{Figura: Distribución de exactitud en el espacio de hiperparámetros para {escape_latex(c_nom)}.}}")
                        doc.append(r"\end{figure}")
                        doc.append("")
            elif pca_data.get('dist_plot'):
                doc.append(r"\begin{figure}[H]")
                doc.append(r"\centering")
                doc.append(f"\\includegraphics[width=0.85\\textwidth]{{{pca_data['dist_plot']}}}")
                doc.append(r"\caption{Figura: Distribución de densidades de la exactitud macro a lo largo de las combinaciones evaluadas en el espacio de hiperparámetros.}")
                doc.append(r"\end{figure}")
                doc.append("")
                
            # Proyecciones por Combinación con Fronteras de Decisión
            for sec in pca_data['secciones']:
                doc.append(r"\newpage")
                doc.append(f"\\subsection{{{escape_latex(sec['nombre'])}}}")
                
                has_s2d = bool(sec['scatter_2d'])
                has_s3d = bool(sec['scatter_3d'])
                
                if has_s2d or has_s3d:
                    doc.append(r"\begin{figure}[H]")
                    doc.append(r"\centering")
                    if has_s2d and has_s3d:
                        doc.append(r"\begin{subfigure}{0.48\textwidth}")
                        doc.append(f"    \\includegraphics[width=\\textwidth]{{{sec['scatter_2d']}}}")
                        doc.append(r"    \caption{Proyección PCA 2D (Fronteras de Decisión)}")
                        doc.append(r"\end{subfigure}")
                        doc.append(r"\hfill")
                        doc.append(r"\begin{subfigure}{0.48\textwidth}")
                        doc.append(f"    \\includegraphics[width=\\textwidth]{{{sec['scatter_3d']}}}")
                        doc.append(r"    \caption{Proyección PCA 3D}")
                        doc.append(r"\end{subfigure}")
                    elif has_s2d:
                        doc.append(f"\\includegraphics[width=0.75\\textwidth]{{{sec['scatter_2d']}}}")
                    elif has_s3d:
                        doc.append(f"\\includegraphics[width=0.75\\textwidth]{{{sec['scatter_3d']}}}")
                    doc.append(f"\\caption{{Figura: Espacio de características y fronteras de decisión para {escape_latex(sec['nombre'])}.}}")
                    doc.append(r"\end{figure}")
                    doc.append("")
                    
                has_c2d = bool(sec['confusion_2d'])
                has_c3d = bool(sec['confusion_3d'])
                
                if has_c2d or has_c3d:
                    doc.append(r"\begin{figure}[H]")
                    doc.append(r"\centering")
                    if has_c2d and has_c3d:
                        doc.append(r"\begin{subfigure}{0.48\textwidth}")
                        doc.append(f"    \\includegraphics[width=\\textwidth]{{{sec['confusion_2d']}}}")
                        doc.append(r"    \caption{Matriz de Confusión 2D}")
                        doc.append(r"\end{subfigure}")
                        doc.append(r"\hfill")
                        doc.append(r"\begin{subfigure}{0.48\textwidth}")
                        doc.append(f"    \\includegraphics[width=\\textwidth]{{{sec['confusion_3d']}}}")
                        doc.append(r"    \caption{Matriz de Confusión 3D}")
                        doc.append(r"\end{subfigure}")
                    elif has_c2d:
                        doc.append(f"\\includegraphics[width=0.65\\textwidth]{{{sec['confusion_2d']}}}")
                    elif has_c3d:
                        doc.append(f"\\includegraphics[width=0.65\\textwidth]{{{sec['confusion_3d']}}}")
                    doc.append(f"\\caption{{Figura: Matrices de confusión y tasas de clasificación para {escape_latex(sec['nombre'])}.}}")
                    doc.append(r"\end{figure}")
                    doc.append("")

        doc.append(r"\end{document}")
        
        clean_date = re.sub(r'[^a-zA-Z0-9_-]', '_', fecha)
        base_name = f"Reporte_EMG_{clean_date}"
        tex_path = os.path.join(self.output_dir, f"{base_name}.tex")
        pdf_path = os.path.join(self.output_dir, f"{base_name}.pdf")
        
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(doc))
            
        logger(f"\n[REPORTE] Compilando documento LaTeX ({base_name}.tex)...")
        try:
            cmd = ["pdflatex", "-interaction=nonstopmode", f"{base_name}.tex"]
            subprocess.run(cmd, cwd=self.output_dir, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(cmd, cwd=self.output_dir, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
                logger(f"[REPORTE] PDF generado exitosamente en: {pdf_path}")
                return {
                    'status': 'success',
                    'tex_path': tex_path,
                    'pdf_path': pdf_path,
                    'error': None
                }
            else:
                return {
                    'status': 'error',
                    'tex_path': tex_path,
                    'pdf_path': None,
                    'error': 'No se generó el archivo PDF.'
                }
        except Exception as e:
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
                logger(f"[REPORTE] PDF generado exitosamente en: {pdf_path}")
                return {
                    'status': 'success',
                    'tex_path': tex_path,
                    'pdf_path': pdf_path,
                    'error': None
                }
            return {
                'status': 'error',
                'tex_path': tex_path,
                'pdf_path': None,
                'error': str(e)
            }
            
        return {
            'status': 'success',
            'tex_path': tex_path,
            'pdf_path': pdf_path,
            'error': None
        }

    def generate_snr_report(self, session_paths, notes_dict=None, logger=print):
        """Genera un documento LaTeX y compila a PDF enfocado exclusivamente en el análisis de SNR y ruido."""
        if not session_paths:
            return {'status': 'error', 'error': 'No se seleccionaron mediciones.'}
            
        session_paths = [
            p for p in session_paths 
            if os.path.isdir(p) and "secuencia" not in os.path.basename(p).lower()
        ]
            
        notes = notes_dict or {}
        info = self.extract_session_metadata(session_paths)
        fecha = notes.get('fecha', info['fecha'] or datetime.now().strftime("%Y-%m-%d"))
        sujeto = notes.get('sujeto', info['sujeto'] or "Desconocido")
        canales_final = notes.get('canales', info['canales'])
        
        doc = []
        doc.append(r"\documentclass[11pt,a4paper]{article}")
        doc.append(r"\usepackage[utf8]{inputenc}")
        doc.append(r"\usepackage[spanish]{babel}")
        doc.append(r"\usepackage{amsmath}")
        doc.append(r"\usepackage{amssymb}")
        doc.append(r"\usepackage{graphicx}")
        doc.append(r"\usepackage{geometry}")
        doc.append(r"\usepackage{caption}")
        doc.append(r"\usepackage{subcaption}")
        doc.append(r"\usepackage{booktabs}")
        doc.append(r"\usepackage{tabularx}")
        doc.append(r"\usepackage{hyperref}")
        doc.append(r"\usepackage{float}")
        doc.append(r"\addto\captionsspanish{\renewcommand{\tablename}{Tabla}}")
        doc.append(r"\addto\captionsspanish{\renewcommand{\figurename}{Figura}}")
        doc.append(r"\geometry{top=2cm, bottom=2cm, left=2.2cm, right=2.2cm}")
        doc.append("")
        doc.append(f"\\title{{\\textbf{{Reporte Técnico: Análisis de SNR y Calidad de Señal ({escape_latex(fecha)})}}}}")
        doc.append(r"\author{Laboratorio de Sistemas Dinámicos (LSD) - Sistema Ñandú}")
        doc.append(f"\\date{{{escape_latex(fecha)}}}")
        doc.append("")
        doc.append(r"\begin{document}")
        doc.append(r"\maketitle")
        doc.append("")
        
        # 1. Resumen y condiciones
        doc.append(r"\section{Condiciones Experimentales}")
        doc.append(r"\begin{itemize}")
        doc.append(f"    \\item \\textbf{{Fecha:}} {escape_latex(fecha)}")
        doc.append(f"    \\item \\textbf{{Sujeto:}} {escape_latex(sujeto)}")
        doc.append(f"    \\item \\textbf{{Frecuencia de Muestreo:}} 2000 Hz")
        doc.append(r"\end{itemize}")
        doc.append("")
        
        # 2. Resumen Estadístico de SNR
        doc.append(r"\section{Resumen Estadístico de Relación Señal-Ruido (SNR)}")
        vocal_snr = {v: {0: [], 1: [], 2: []} for v in ['A', 'E', 'I', 'O', 'U']}
        for med in info['mediciones']:
            letra = med['letra']
            if letra not in vocal_snr: continue
            for ch_idx in range(3):
                ch_key = f'canal_{ch_idx}'
                ch_path = os.path.join(med['path'], ch_key)
                for fname in ['results.json', 'analisis_results.json']:
                    rp = os.path.join(ch_path, fname)
                    if os.path.exists(rp):
                        try:
                            with open(rp, 'r') as f:
                                r_data = json.load(f)
                                snrs = r_data.get('snr_per_pulse', [])
                                if snrs:
                                    vocal_snr[letra][ch_idx].extend([float(x) for x in snrs if x is not None and not np.isnan(x)])
                                    break
                        except Exception: pass
                        
        m0_t = canales_final.get(0, "Canal 0")
        m1_t = canales_final.get(1, "Canal 1")
        m2_t = canales_final.get(2, "Canal 2")
        
        doc.append(r"\begin{table}[H]")
        doc.append(r"\centering")
        doc.append(r"\begin{tabular}{lccc}")
        doc.append(r"\toprule")
        doc.append(f"\\textbf{{Vocal}} & \\textbf{{Canal 0: {escape_latex(m0_t)}}} & \\textbf{{Canal 1: {escape_latex(m1_t)}}} & \\textbf{{Canal 2: {escape_latex(m2_t)}}} \\\\")
        doc.append(r"\midrule")
        for v in ['A', 'E', 'I', 'O', 'U']:
            vals_0 = vocal_snr[v][0]
            vals_1 = vocal_snr[v][1]
            vals_2 = vocal_snr[v][2]
            s0 = f"${np.mean(vals_0):.1f} \\pm {np.std(vals_0):.1f}$" if vals_0 else "--"
            s1 = f"${np.mean(vals_1):.1f} \\pm {np.std(vals_1):.1f}$" if vals_1 else "--"
            s2 = f"${np.mean(vals_2):.1f} \\pm {np.std(vals_2):.1f}$" if vals_2 else "--"
            doc.append(f"Vocal {v} & {s0} & {s1} & {s2} \\\\")
        doc.append(r"\bottomrule")
        doc.append(r"\end{tabular}")
        doc.append(r"\caption{Relación Señal-Ruido promedio ($\text{SNR} \pm \sigma$) registrada por vocal y canal.}")
        doc.append(r"\end{table}")
        doc.append("")
        
        comp_images = self.find_comparative_images(fecha)
        if 'snr_timeline' in comp_images:
            doc.append(r"\begin{figure}[H]")
            doc.append(r"\centering")
            doc.append(f"\\includegraphics[width=0.88\\textwidth]{{{comp_images['snr_timeline']}}}")
            doc.append(r"\caption{Evolución temporal del SNR a lo largo de la sesión.}")
            doc.append(r"\end{figure}")
            doc.append("")
            
        # 3. Desglose detallado de ruido y espectros por serie
        doc.append(r"\newpage")
        doc.append(r"\section{Desglose de Evolución de Ruido Interpulso}")
        for p_idx, (prueba_tag, meds_en_prueba) in enumerate(info['pruebas_agrupadas'].items()):
            doc.append(f"\\subsection{{{escape_latex(prueba_tag)}}}")
            for med in meds_en_prueba:
                letra = med['letra']
                m_name = med['name']
                m_path = med['path']
                imgs_ch = {}
                for c_idx in range(3):
                    c_dir = os.path.join(m_path, f"canal_{c_idx}")
                    imgs_ch[c_idx] = {
                        'spec': find_first_existing([os.path.join(c_dir, "spec.png")]),
                        'evolucion': find_first_existing([os.path.join(c_dir, "evolucion.png")])
                    }
                active_chs = [c for c in range(3) if imgs_ch[c]['evolucion']]
                if active_chs:
                    doc.append(f"\\subsubsection{{Vocal {escape_latex(letra)}: {escape_latex(m_name)}}}")
                    chunk_size = 2
                    for c_start in range(0, len(active_chs), chunk_size):
                        chunk = active_chs[c_start:c_start + chunk_size]
                        doc.append(r"\begin{figure}[H]")
                        doc.append(r"\centering")
                        for c_idx in chunk:
                            m_label = info['canales'].get(c_idx, f"Canal {c_idx}")
                            doc.append(r"\begin{subfigure}{0.88\textwidth}")
                            doc.append(f"    \\includegraphics[width=\\textwidth]{{{imgs_ch[c_idx]['evolucion']}}}")
                            doc.append(f"    \\caption{{Evolución de Ruido - Canal {c_idx} ({escape_latex(m_label)})}}")
                            doc.append(r"\end{subfigure}")
                            doc.append(r"\vspace{0.3cm}")
                        doc.append(f"\\caption{{Evolución del ruido interpulso para {escape_latex(m_name)}.}}")
                        doc.append(r"\end{figure}")
                        doc.append("")
                        
        doc.append(r"\end{document}")
        clean_date = re.sub(r'[^a-zA-Z0-9_-]', '_', fecha)
        base_name = f"Reporte_SNR_{clean_date}"
        tex_path = os.path.join(self.output_dir, f"{base_name}.tex")
        pdf_path = os.path.join(self.output_dir, f"{base_name}.pdf")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(doc))
        logger(f"\n[REPORTE SNR] Compilando documento ({base_name}.tex)...")
        try:
            cmd = ["pdflatex", "-interaction=nonstopmode", f"{base_name}.tex"]
            subprocess.run(cmd, cwd=self.output_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(cmd, cwd=self.output_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger(f"[REPORTE SNR] PDF generado exitosamente en: {pdf_path}")
        except Exception as e:
            return {'status': 'error', 'tex_path': tex_path, 'pdf_path': None, 'error': str(e)}
        return {'status': 'success', 'tex_path': tex_path, 'pdf_path': pdf_path, 'error': None}
