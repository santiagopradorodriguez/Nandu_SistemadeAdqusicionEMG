# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de auditoría ligera y rápida de metadatos anatómicos e inter-día.
#              Opera exclusivamente con bibliotecas estándar (os, json) para evitar
#              bloqueos en el hilo principal de la interfaz gráfica (UI Thread).
# ==============================================================================

import os
import json

# Caché en memoria para evitar lecturas redundantes a disco en eventos rápidos de UI
_meta_cache = {}

def leer_metadata_toma(r):
    """
    Lee metadata.json de una toma (probando canal_0/metadata.json y luego metadata.json).
    Usa caché en memoria validado por fecha de modificación (mtime) para velocidad instantánea.
    """
    meta_file = os.path.join(r, "canal_0", "metadata.json")
    if not os.path.exists(meta_file):
        meta_file = os.path.join(r, "metadata.json")
        if not os.path.exists(meta_file):
            return None, f"No se encontró metadata.json en {os.path.basename(r)}"

    try:
        mtime = os.path.getmtime(meta_file)
        if meta_file in _meta_cache:
            cached_mtime, cached_data = _meta_cache[meta_file]
            if cached_mtime == mtime:
                return cached_data, None

        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        _meta_cache[meta_file] = (mtime, meta)
        return meta, None
    except Exception as e:
        return None, f"Error al leer metadata.json en {os.path.basename(r)}: {e}"


def auditar_metadatos_sesiones(rutas_tomas):
    """
    Inspecciona los archivos metadata.json de cada toma seleccionada.
    Verifica coherencia en:
    1. Músculos asignados por canal (canal_0, canal_1, canal_2).
    2. Tempo de metrónomo (bpm).
    3. Frecuencia de muestreo (sample_rate).
    """
    info_sesiones = {}
    advertencias = []

    musculos_referencia = None
    bpm_referencia = None
    fs_referencia = None
    toma_ref_name = None

    for r in rutas_tomas:
        toma_name = os.path.basename(r)
        meta, err = leer_metadata_toma(r)
        if err:
            advertencias.append(f"[AVISO] {err}" if "No se encontró" in err else f"[ERROR] {err}")
            continue

        bpm = meta.get('bpm', 40)
        fs = meta.get('sample_rate', 2000)
        sujeto = meta.get('sujeto', 'Desconocido')
        m_date_raw = str(meta.get('measurement_date') or meta.get('date') or '').strip()
        fecha_toma = m_date_raw.split('T')[0][:10] if m_date_raw else "Desconocida"

        # Mapeo de músculos
        m_map = meta.get('muscles_map', {})
        if not m_map:
            muscles_list = meta.get('muscles', [])
            if len(muscles_list) >= 3:
                m_map = {f"canal_{i}": muscles_list[i] for i in range(len(muscles_list))}
            else:
                m_map = {"canal_0": "Desconocido", "canal_1": "Desconocido", "canal_2": "Desconocido"}

        info_sesiones[toma_name] = {
            'sujeto': sujeto,
            'bpm': bpm,
            'fs': fs,
            'fecha': fecha_toma,
            'muscles_map': m_map,
            'ruta': r
        }

        # Comprobación de coherencia respecto a la primera toma
        if musculos_referencia is None:
            musculos_referencia = m_map
            bpm_referencia = bpm
            fs_referencia = fs
            toma_ref_name = toma_name
        else:
            # Chequeo de músculos
            for ch in ["canal_0", "canal_1", "canal_2"]:
                m_curr = str(m_map.get(ch, "")).lower()
                m_ref = str(musculos_referencia.get(ch, "")).lower()
                if m_curr and m_ref and m_curr != m_ref:
                    adv = (f"[ALERTA ANATOMICA] Discrepancia en {ch}: '{m_map.get(ch)}' en {toma_name} "
                           f"vs '{musculos_referencia.get(ch)}' en {toma_ref_name}. "
                           f"Podrían estarse mezclando músculos distintos entre sesiones.")
                    if adv not in advertencias:
                        advertencias.append(adv)

            # Chequeo de BPM
            if bpm != bpm_referencia:
                adv_bpm = (f"[ALERTA METRONOMO] BPM inconsistente: {bpm} en {toma_name} "
                           f"vs {bpm_referencia} en {toma_ref_name}. La duración de ventana variará.")
                if adv_bpm not in advertencias:
                    advertencias.append(adv_bpm)

            # Chequeo de Fs
            if fs != fs_referencia:
                adv_fs = (f"[ALERTA FRECUENCIA] Tasa de muestreo dispar: {fs} Hz en {toma_name} "
                           f"vs {fs_referencia} Hz en {toma_ref_name}.")
                if adv_fs not in advertencias:
                    advertencias.append(adv_fs)

    es_compatible = len(advertencias) == 0
    fechas_detectadas = sorted(list(set(info['fecha'] for info in info_sesiones.values() if info['fecha'] != 'Desconocida')))
    return {
        'compatible': es_compatible,
        'advertencias': advertencias,
        'info_sesiones': info_sesiones,
        'musculos_resumen': musculos_referencia or {},
        'bpm_comun': bpm_referencia,
        'fs_comun': fs_referencia,
        'fechas_detectadas': fechas_detectadas
    }
