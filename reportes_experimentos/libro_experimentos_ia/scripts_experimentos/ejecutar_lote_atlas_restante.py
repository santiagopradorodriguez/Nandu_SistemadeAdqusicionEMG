# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Orquestador en Lote del Atlas Oficial sEMG (18 Diapositivas)
#              - Orden cronológico riguroso basado en metadata.json ("measurement_date")
#              - Unifica 2026-08-29, 2026-08-30, 2026-09-23 Config 2 y renombre de 09-29
#              - Preserva al 100% la estética canónica e invarianza visual
# ==============================================================================

import os
import sys
import json
import glob
import subprocess
import shutil

from compilador_maestro_atlas import compilar_diapositiva_maestra

OUT_DIR = '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29'
BASE_DRIVE = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Candela_Organizado_Drive'
BASE_RAW = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos'

# ==============================================================================
# LISTA DE LAS 18 DIAPOSITIVAS CRONOLÓGICAS
# ==============================================================================

CONFIGS = [
    # DIAPOSITIVA 01: 2026-08-28 Digástrico Milohioideo Orbicularis (Archivado 08-29)
    {
        'id': '01_2026-08-29',
        'titulo': "Experimento 2026-08-28: Digastrico Milohioideo Orbicularis",
        'camp': BASE_RAW,
        'subfolder': '2026-08-29',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Anterior Belly', 'musculo_linea2': 'Digástrico Submentoniano', 'funcion': 'Apertura mandibular activa', 'col': '#ef4444', 'foto': f'{OUT_DIR}/01_render_0829_belly_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '19.4 ± 3.5 mm', 'dist_menton': '40.9 mm', 'dist_nariz': '114.1 mm', 'dist_trago': '117.6 mm', 'dist_comisura': '77.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Milohioideo', 'musculo_linea2': 'Piso Bucal (Desplazado)', 'funcion': 'Elevación del suelo bucal', 'col': '#38bdf8', 'foto': f'{OUT_DIR}/02_render_0829_milohioideo_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '23.1 ± 3.5 mm', 'dist_menton': '42.2 mm', 'dist_nariz': '72.3 mm', 'dist_trago': '64.9 mm', 'dist_comisura': '50.0 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/03_render_0829_orbicularis_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '19.4 ± 3.5 mm', 'dist_menton': '39.5 mm', 'dist_nariz': '22.6 mm', 'dist_trago': '65.3 mm', 'dist_comisura': '25.6 mm'}
        ],
        'filtro_pca': lambda d, p: os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_Prueba1_Cande', 'E': 'E_Prueba1_Cande', 'I': 'I_Prueba1_Cande', 'O': 'O_Prueba1_Cande', 'U': 'U_Prueba1_Cande'},
            {'A': 'A_Prueba2_Cande', 'E': 'E_Prueba2_Cande', 'I': 'I_Prueba2_Cande', 'O': 'O_Prueba2_Cande', 'U': 'U_Prueba2_Cande'}
        ],
        'cuaderno': {'sujeto': 'Candela (08-28)', 'horario': '16:29 a 16:39', 'duracion': '10.0 min total', 'fs': '2000 Hz', 'ref': 'Mastoides', 'series_n': '2'}
    },

    # DIAPOSITIVA 02: 2026-08-28 Digástrico Nudo de la Sonrisa Orbicularis (Archivado 08-30)
    {
        'id': '02_2026-08-30',
        'titulo': "Experimento 2026-08-28: Digastrico NudoSonrisa Orbicularis",
        'camp': BASE_RAW,
        'subfolder': '2026-08-30',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Anterior Belly', 'musculo_linea2': 'Digástrico Submentoniano', 'funcion': 'Apertura mandibular activa', 'col': '#ef4444', 'foto': f'{OUT_DIR}/01_render_0829_belly_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '19.4 ± 3.5 mm', 'dist_menton': '40.9 mm', 'dist_nariz': '114.1 mm', 'dist_trago': '117.6 mm', 'dist_comisura': '77.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Nudo de la Sonrisa', 'musculo_linea2': 'Modíolo / Risorio', 'funcion': 'Retracción comisural y sonrisa', 'col': '#9d4edd', 'foto': f'{OUT_DIR}/02_render_0830_nudo_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '20.4 ± 3.5 mm', 'dist_menton': '35.9 mm', 'dist_nariz': '48.7 mm', 'dist_trago': '39.0 mm', 'dist_comisura': '21.4 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/03_render_0829_orbicularis_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '19.4 ± 3.5 mm', 'dist_menton': '39.5 mm', 'dist_nariz': '22.6 mm', 'dist_trago': '65.3 mm', 'dist_comisura': '25.6 mm'}
        ],
        'filtro_pca': lambda d, p: os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_Prueba4_Cande', 'E': 'E_Prueba4_Cande', 'I': 'I_Prueba4_Cande', 'O': 'O_Prueba4_Cande', 'U': 'U_Prueba4_Cande'},
            {'A': 'A_Prueba5_Cande', 'E': 'E_Prueba5_Cande', 'I': 'I_Prueba5_Cande', 'O': 'O_Prueba5_Cande', 'U': 'U_Prueba5_Cande'}
        ],
        'cuaderno': {'sujeto': 'Candela (08-28)', 'horario': '16:52 a 17:00', 'duracion': '8.5 min total', 'fs': '2000 Hz', 'ref': 'Mastoides', 'series_n': '2'}
    },

    # DIAPOSITIVA 07: 2026-09-23 Config 2 Masetero y Orbicularis (Canal 1 Inactivo)
    {
        'id': '07_2026-09-23_config2',
        'titulo': "Experimento 2026-09-23: Config2 Masetero y Orbicularis",
        'camp': os.path.join(BASE_DRIVE, '2026-09-23_Config2_Masetero_y_Orbicularis_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación mandibular y oclusión', 'col': '#f04e23', 'foto': '', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '21.0 ± 3.5 mm', 'dist_menton': '67.5 ± 4.0 mm', 'dist_nariz': '71.5 ± 4.0 mm', 'dist_trago': '44.5 ± 3.5 mm', 'dist_comisura': '41.5 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Inactivo', 'musculo_linea2': 'Sin Conexión', 'funcion': 'Canal auxiliar desconectado', 'col': '#64748b', 'foto': '', 'activo': False},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': '', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '23.0 ± 3.5 mm', 'dist_menton': '46.5 ± 3.5 mm', 'dist_nariz': '16.5 ± 3.5 mm', 'dist_trago': '72.5 ± 4.0 mm', 'dist_comisura': '14.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_Pruebat1_Candela', 'E': 'E_Pruebat1_Candela', 'I': 'I_Pruebat1_Candela', 'O': 'O_Pruebat1_Candela', 'U': 'U_Pruebat1_Candela'},
            {'A': 'A_Prueba2_Candela', 'E': 'E_Prueba2_Candela', 'I': 'I_Prueba2_Candela', 'O': 'O_Prueba2_Candela', 'U': 'U_Prueba2_Candela'},
            {'A': 'A_Prueba3_Candela', 'E': 'E_Prueba3_Candela', 'I': 'I_Prueba3_Candela', 'O': 'O_Prueba3_Candela', 'U': 'U_Prueba3_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (09-23 Config 2)', 'horario': '15:12 a 16:17', 'duracion': '65.0 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '3'}
    },

    # DIAPOSITIVA 08: 2026-09-25 Bloque 1a - Belly 1
    {
        'id': '08_2026-09-25_bloque1a',
        'titulo': "Experimento 2026-09-25: Bloque 1a Belly 1",
        'camp': os.path.join(BASE_DRIVE, '2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación y oclusión', 'col': '#f04e23', 'foto': f'{OUT_DIR}/01_render_0925_bloque1_masetero_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '65.0 ± 4.0 mm', 'dist_nariz': '70.0 ± 4.0 mm', 'dist_trago': '44.0 ± 3.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Anterior Belly', 'musculo_linea2': '', 'funcion': 'Apertura mandibular activa', 'col': '#ef4444', 'foto': f'{OUT_DIR}/02_render_0925_bloque1_digastrico_3x4.png', 'tam_electrodo': '10 mm disco ag/agcl', 'ied': '25.0 ± 3.5 mm', 'dist_menton': '22.0 ± 3.5 mm', 'dist_nariz': 'N/A submentoniano', 'dist_trago': '75.0 ± 4.5 mm', 'dist_comisura': '35.0 ± 3.5 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/03_render_0925_bloque1_orbicular_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '24.0 ± 3.5 mm', 'dist_menton': '45.0 ± 3.5 mm', 'dist_nariz': '12.0 ± 3.0 mm', 'dist_trago': '72.0 ± 4.0 mm', 'dist_comisura': '15.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: ('Digastrico' in d or 'astrico' in d) and os.path.exists(f'{p}/canal_0/metadata.json') and json.load(open(f'{p}/canal_0/metadata.json')).get('muscles_map', {}).get('canal_1') == 'belly 1',
        'ses_curvas': [
            {'A': 'A_PruebaDigastrico_Candela', 'E': 'E_PruebaDigastrico_Candela', 'I': 'I_PruebaDigastrico_Candela', 'O': 'O_PruebaDigastrico_Candela', 'U': 'U_PruebaDigastrico_Candela'},
            {'A': 'A_PruebaDigastrico_2_Candela', 'E': 'E_PruebaDigastrico_2_Candela', 'I': 'I_PruebaDigastrico_2_Candela', 'O': 'O_PruebaDigastrico_2_Candela', 'U': 'U_PruebaDigastrico_2_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (09-25)', 'horario': '14:07 a 14:14', 'duracion': '6.5 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '2'}
    },

    # DIAPOSITIVA 09: 2026-09-25 Bloque 1b - Belly 2
    {
        'id': '09_2026-09-25_bloque1b',
        'titulo': "Experimento 2026-09-25: Bloque 1b Belly 2",
        'camp': os.path.join(BASE_DRIVE, '2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación y oclusión', 'col': '#f04e23', 'foto': f'{OUT_DIR}/01_render_0925_bloque1_masetero_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '65.0 ± 4.0 mm', 'dist_nariz': '70.0 ± 4.0 mm', 'dist_trago': '44.0 ± 3.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Anterior Belly', 'musculo_linea2': 'Horizontal 90°', 'funcion': 'Apertura mandibular activa', 'col': '#ef4444', 'foto': f'{OUT_DIR}/02_render_0925_bloque1b_digastrico_horizontal_curado_3x4.png', 'tam_electrodo': '10 mm disco ag/agcl', 'ied': '23.4 ± 3.5 mm', 'dist_menton': '37.9 ± 3.5 mm', 'dist_nariz': 'N/A submentoniano', 'dist_trago': '78.0 ± 4.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/03_render_0925_bloque1_orbicular_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '24.0 ± 3.5 mm', 'dist_menton': '45.0 ± 3.5 mm', 'dist_nariz': '12.0 ± 3.0 mm', 'dist_trago': '72.0 ± 4.0 mm', 'dist_comisura': '15.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: ('Digastrico' in d or 'astrico' in d) and os.path.exists(f'{p}/canal_0/metadata.json') and json.load(open(f'{p}/canal_0/metadata.json')).get('muscles_map', {}).get('canal_1') == 'belly 2',
        'ses_curvas': [
            {'A': 'A_PruebaDigastrico2_Candela', 'E': 'E_PruebaDigastrico2_Candela', 'I': 'I_PruebaDigastrico2_Candela', 'O': 'O_PruebaDigastrico2_Candela', 'U': 'U_PruebaDigastrico2_Candela'},
            {'A': 'A_PruebaDigastrico3_Candela', 'E': 'E_PruebaDigastrico3_Candela', 'I': 'I_PruebaDigastrico3_Candela', 'O': 'O_PruebaDigastrico3_Candela', 'U': 'U_PruebaDigastrico3_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (09-25)', 'horario': '14:34 a 14:40', 'duracion': '5.8 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '2'}
    },

    # DIAPOSITIVA 10: 2026-09-25 Bloque 2 - Mentoniano
    {
        'id': '10_2026-09-25_bloque2',
        'titulo': "Experimento 2026-09-25: Bloque 2 Mentoniano",
        'camp': os.path.join(BASE_DRIVE, '2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Mentoniano', 'musculo_linea2': 'Eje Vertical Mentón', 'funcion': 'Elevación y eversión mentón', 'col': '#9d4edd', 'foto': f'{OUT_DIR}/render_propuesta_opcion1_mentoniano_frontal_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '18.7 ± 3.5 mm', 'dist_menton': '22.0 ± 3.5 mm', 'dist_nariz': '58.0 ± 4.0 mm', 'dist_trago': '80.0 ± 4.5 mm', 'dist_comisura': '28.0 ± 3.5 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/02_render_0925_bloque2_orbicular_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '27.4 ± 3.5 mm', 'dist_menton': '45.0 ± 3.5 mm', 'dist_nariz': '14.0 ± 3.0 mm', 'dist_trago': '72.0 ± 4.0 mm', 'dist_comisura': '16.0 ± 3.5 mm'},
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación y oclusión', 'col': '#f04e23', 'foto': f'{OUT_DIR}/03_render_0925_bloque2_masetero_foto07_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '65.0 ± 4.0 mm', 'dist_nariz': '70.0 ± 4.0 mm', 'dist_trago': '44.0 ± 3.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: 'Mentoniano' in d and os.path.exists(f'{p}/canal_0/metadata.json'),
        'ses_curvas': [
            {'A': 'A_PruebaMentoniano1_Candela', 'E': 'E_PruebaMentoniano1_Candela', 'I': 'I_PruebaMentoniano1_Candela', 'O': 'O_PruebaMentoniano1_Candela', 'U': 'U_PruebaMentoniano1_Candela'},
            {'A': 'A_PruebaMentoniano2_Candela', 'E': 'E_PruebaMentoniano2_Candela', 'I': 'I_PruebaMentoniano2_Candela', 'O': 'O_PruebaMentoniano2_Candela', 'U': 'U_PruebaMentoniano2_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (09-25)', 'horario': '15:09 a 15:25', 'duracion': '16.0 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '2'}
    },

    # DIAPOSITIVA 11: 2026-09-25 Bloque 3 - Modíolo
    {
        'id': '11_2026-09-25_bloque3',
        'titulo': "Experimento 2026-09-25: Bloque 3 Modiolo",
        'camp': os.path.join(BASE_DRIVE, '2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Modíolo', 'musculo_linea2': 'Nudo Comisural', 'funcion': 'Coordinación perioral', 'col': '#ef4444', 'foto': f'{OUT_DIR}/01_render_0925_bloque3_modiolo_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '19.5 ± 3.5 mm', 'dist_menton': '48.0 ± 4.0 mm', 'dist_nariz': '38.0 ± 3.5 mm', 'dist_trago': '66.0 ± 4.0 mm', 'dist_comisura': '12.0 ± 3.0 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/02_render_0925_bloque2_orbicular_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '27.4 ± 3.5 mm', 'dist_menton': '45.0 ± 3.5 mm', 'dist_nariz': '14.0 ± 3.0 mm', 'dist_trago': '72.0 ± 4.0 mm', 'dist_comisura': '16.0 ± 3.5 mm'},
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación y oclusión', 'col': '#f04e23', 'foto': f'{OUT_DIR}/03_render_0925_bloque3_masetero_foto12_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '65.0 ± 4.0 mm', 'dist_nariz': '70.0 ± 4.0 mm', 'dist_trago': '44.0 ± 3.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: 'Modiolo' in d and os.path.exists(f'{p}/canal_0/metadata.json'),
        'ses_curvas': [
            {'A': 'A_PruebaModiolo1_Candela', 'E': 'E_PruebaModiolo1_Candela', 'I': 'I_PruebaModiolo1_Candela', 'O': 'O_PruebaModiolo1_Candela', 'U': 'U_PruebaModiolo1_Candela'},
            {'A': 'A_PruebaModiolo2_Candela', 'E': 'E_PruebaModiolo2_Candela', 'I': 'I_PruebaModiolo2_Candela', 'O': 'O_PruebaModiolo2_Candela', 'U': 'U_PruebaModiolo2_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (09-25)', 'horario': '16:26 a 16:38', 'duracion': '12.0 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '2'}
    },

    # DIAPOSITIVA 12: 2026-09-25 Bloque 4 - NuevaBelly
    {
        'id': '12_2026-09-25_bloque4',
        'titulo': "Experimento 2026-09-25: Bloque 4 NuevaBelly",
        'camp': os.path.join(BASE_DRIVE, '2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación y oclusión', 'col': '#f04e23', 'foto': f'{OUT_DIR}/01_render_0925_bloque1_masetero_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '65.0 ± 4.0 mm', 'dist_nariz': '70.0 ± 4.0 mm', 'dist_trago': '44.0 ± 3.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Anterior Belly', 'musculo_linea2': 'Nueva Posición Submentoniana', 'funcion': 'Apertura mandibular activa', 'col': '#ef4444', 'foto': f'{OUT_DIR}/02_render_0925_bloque4_nuevabelly_3x4.png', 'tam_electrodo': '10 mm disco ag/agcl', 'ied': '19.8 ± 3.5 mm', 'dist_menton': '22.0 ± 3.5 mm', 'dist_nariz': 'N/A submentoniano', 'dist_trago': '75.0 ± 4.5 mm', 'dist_comisura': '35.0 ± 3.5 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/02_render_0925_bloque2_orbicular_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '27.4 ± 3.5 mm', 'dist_menton': '45.0 ± 3.5 mm', 'dist_nariz': '14.0 ± 3.0 mm', 'dist_trago': '72.0 ± 4.0 mm', 'dist_comisura': '16.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: 'NuevaBelly' in d and os.path.exists(f'{p}/canal_0/metadata.json'),
        'ses_curvas': [
            {'A': 'A_PruebaNuevaBelly_Candela', 'E': 'E_PruebaNuevaBelly_Candela', 'I': 'I_PruebaNuevaBelly_Candela', 'O': 'O_PruebaNuevaBelly_Candela', 'U': 'U_PruebaNuevaBelly_Candela'},
            {'A': 'A_PruebaNuevaBelly_2_Candela', 'E': 'E_PruebaNuevaBelly_2_Candela', 'I': 'I_PruebaNuevaBelly_2_Candela', 'O': 'O_PruebaNuevaBelly_2_Candela', 'U': 'U_PruebaNuevaBelly_2_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (09-25)', 'horario': '16:42 a 16:55', 'duracion': '13.0 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '2'}
    },

    # DIAPOSITIVA 13: 2026-09-29 Modiolo y Masetero (Cable Orbicular dañado)
    {
        'id': '13_2026-09-29',
        'titulo': "Experimento 2026-09-29: Modiolo y Masetero",
        'camp': os.path.join(BASE_DRIVE, '2026-09-29_Modiolo_y_Orbicularis_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Modíolo', 'musculo_linea2': 'Nudo Comisural', 'funcion': 'Coordinación perioral', 'col': '#ef4444', 'foto': f'{OUT_DIR}/01_render_0929_modiolo_comisura_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '46.0 ± 4.0 mm', 'dist_nariz': '38.0 ± 3.5 mm', 'dist_trago': '68.0 ± 4.0 mm', 'dist_comisura': '11.5 ± 3.0 mm'},
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación y oclusión', 'col': '#f04e23', 'foto': f'{OUT_DIR}/foto_perdida_masetero_3x4.png', 'tam_electrodo': '5 x 5 mm ± 0.02 mm', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '65.0 ± 4.0 mm', 'dist_nariz': '70.0 ± 4.0 mm', 'dist_trago': '44.0 ± 3.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Inactivo', 'musculo_linea2': 'Sin Conexión - Cable Dañado', 'funcion': 'Canal auxiliar desconectado', 'col': '#64748b', 'foto': '', 'activo': False}
        ],
        'filtro_pca': lambda d, p: os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_Prueba1_Candela', 'E': 'E_Prueba1_Candela', 'I': 'I_Prueba1_Candela', 'O': 'O_Prueba1_Candela', 'U': 'U_Prueba1_Candela'},
            {'A': 'A_Prueba2_Candela', 'E': 'E_Prueba2_Candela', 'I': 'I_Prueba2_Candela', 'O': 'O_Prueba2_Candela', 'U': 'U_Prueba2_Candela'},
            {'A': 'A_Prueba3_Candela', 'E': 'E_Prueba3_Candela', 'I': 'I_Prueba3_Candela', 'O': 'O_Prueba3_Candela', 'U': 'U_Prueba3_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (09-29)', 'horario': '18:00 a 18:20', 'duracion': '20.0 min total', 'fs': '6000 Hz', 'ref': 'Mastoides', 'series_n': '3'}
    },

    # DIAPOSITIVA 14: 2026-10-20 Exp 1 - Masetero 24 mm
    {
        'id': '14_2026-10-20_exp1',
        'titulo': "Experimento 2026-10-20: Exp1 Masetero 24 mm",
        'camp': os.path.join(BASE_DRIVE, '2026-10-20_Control_Experimental_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{BASE_DRIVE}/2026-10-20_Control_Experimental_Candela/fotos_montaje/02_exp1_frontal_orbicularis_y_modiolo.png', 'tam_electrodo': '15 mm (circular)', 'ied': '15.0 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Modíolo', 'musculo_linea2': 'Nudo Comisural / DAO', 'funcion': 'Coordinación perioral', 'col': '#9d4edd', 'foto': f'{BASE_DRIVE}/2026-10-20_Control_Experimental_Candela/fotos_montaje/01_exp1_lateral_izq_modiolo_masetero.png', 'tam_electrodo': '15 mm (circular)', 'ied': '15.0 ± 3.5 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular (24 mm)', 'funcion': 'Elevación mandibular y oclusión', 'col': '#f04e23', 'foto': f'{BASE_DRIVE}/2026-10-20_Control_Experimental_Candela/fotos_montaje/01_exp1_lateral_izq_modiolo_masetero.png', 'tam_electrodo': '24 mm (recortado)', 'ied': '44.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: 'Exp1' in d and os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_Exp1_prueba1_Candela', 'E': 'E_Exp1_prueba1_Candela', 'I': 'I_Exp1_prueba1_Candela', 'O': 'O_Exp1_prueba1_Candela', 'U': 'U_Exp1_prueba1_Candela'},
            {'A': 'A_Exp1_prueba2_Candela', 'E': 'E_Exp1_prueba2_Candela', 'I': 'I_Exp1_prueba2_Candela', 'O': 'O_Exp1_prueba2_Candela', 'U': 'U_Exp1_prueba2_Candela'},
            {'A': 'A_Exp1_prueba3_Candela', 'E': 'E_Exp1_prueba3_Candela', 'I': 'I_Exp1_prueba3_Candela', 'O': 'O_Exp1_prueba3_Candela', 'U': 'U_Exp1_prueba3_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (10-20 Exp1)', 'horario': '10:59 a 11:15', 'duracion': '16.0 min total', 'fs': '6000 Hz', 'ref': 'Mastoides', 'series_n': '3'}
    },

    # DIAPOSITIVA 15: 2026-10-20 Exp 2 - Masetero 23 mm (Parches comerciales)
    {
        'id': '15_2026-10-20_exp2',
        'titulo': "Experimento 2026-10-20: Exp2 Masetero 23 mm",
        'camp': os.path.join(BASE_DRIVE, '2026-10-20_Control_Experimental_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/03_render_1020_exp2_orbicular_3x4.png', 'tam_electrodo': '10 mm (circular)', 'ied': '15.0 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Modíolo', 'musculo_linea2': 'Nudo Comisural / DAO', 'funcion': 'Coordinación perioral', 'col': '#9d4edd', 'foto': f'{OUT_DIR}/01_render_1020_exp2_modiolo_3x4.png', 'tam_electrodo': '10 mm (circular)', 'ied': '15.0 ± 3.5 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular (23 mm)', 'funcion': 'Elevación mandibular y oclusión', 'col': '#f04e23', 'foto': f'{BASE_DRIVE}/2026-10-20_Control_Experimental_Candela/fotos_montaje/03_exp2_lateral_izq_modiolo_masetero_orbicular.png', 'tam_electrodo': '23 mm (snap comercial)', 'ied': '30.0 ± 3.5 mm', 'dist_trago': '8.0 mm (al lóbulo)', 'dist_ojo': '35.0 mm', 'dist_nariz': '35.0 mm'}
        ],
        'filtro_pca': lambda d, p: 'Exp2' in d and os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_Exp2_prueba1_Candela', 'E': 'E_Exp2_prueba1_Candela', 'I': 'I_Exp2_prueba1_Candela', 'O': 'O_Exp2_prueba1_Candela', 'U': 'U_Exp2_prueba1_Candela'},
            {'A': 'A_Exp2_prueba2_Candela', 'E': 'E_Exp2_prueba2_Candela', 'I': 'I_Exp2_prueba2_Candela', 'O': 'O_Exp2_prueba2_Candela', 'U': 'U_Exp2_prueba2_Candela'},
            {'A': 'A_Exp2_prueba3_Candela', 'E': 'E_Exp2_prueba3_Candela', 'I': 'I_Exp2_prueba3_Candela', 'O': 'O_Exp2_prueba3_Candela', 'U': 'U_Exp2_prueba3_Candela'},
            {'A': 'A_Exp2_prueba4_Candela', 'E': 'E_Exp2_prueba4_Candela', 'I': 'I_Exp2_prueba4_Candela', 'O': 'O_Exp2_prueba4_Candela', 'U': 'U_Exp2_prueba4_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (10-20 Exp2)', 'horario': '12:08 a 12:26', 'duracion': '18.0 min total', 'fs': '6000 Hz', 'ref': 'Mastoides', 'series_n': '4'}
    },

    # DIAPOSITIVA 16: 2026-10-23 Exp1 Masetero Monocanal
    {
        'id': '16_2026-10-23_exp1',
        'titulo': "Experimento 2026-10-23: Exp1 Masetero Monocanal",
        'camp': os.path.join(BASE_DRIVE, '2026-10-23_Exp1_Masetero_Monocanal_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación mandibular y oclusión', 'col': '#f04e23', 'foto': f'{OUT_DIR}/01_render_1023_exp1_masetero_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '66.0 ± 4.0 mm', 'dist_nariz': '73.0 ± 4.0 mm', 'dist_trago': '44.0 ± 3.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Inactivo', 'musculo_linea2': 'Sin Conexión', 'funcion': 'Canal auxiliar desconectado', 'col': '#64748b', 'foto': '', 'activo': False},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Inactivo', 'musculo_linea2': 'Sin Conexión', 'funcion': 'Canal auxiliar desconectado', 'col': '#64748b', 'foto': '', 'activo': False}
        ],
        'filtro_pca': lambda d, p: os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_Prueba1_Candela', 'E': 'E_Prueba1_Candela', 'I': 'I_Prueba1_Candela', 'O': 'O_Prueba1_Candela', 'U': 'U_Prueba1_Candela'},
            {'A': 'A_Prueba2_Candela', 'E': 'E_Prueba2_Candela', 'I': 'I_Prueba2_Candela', 'O': 'O_Prueba2_Candela', 'U': 'U_Prueba2_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (10-23)', 'horario': '11:50 a 12:10', 'duracion': '20.0 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '2'}
    },

    # DIAPOSITIVA 17: 2026-10-23 Exp2 Belly y Orbicularis
    {
        'id': '17_2026-10-23_exp2',
        'titulo': "Experimento 2026-10-23: Exp2 Belly y Orbicularis",
        'camp': os.path.join(BASE_DRIVE, '2026-10-23_Exp2_Belly_y_Orbicularis_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/01_render_1023_exp2_orbi_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '22.0 ± 3.5 mm', 'dist_menton': '45.0 ± 3.5 mm', 'dist_nariz': '17.0 ± 3.5 mm', 'dist_trago': '72.0 ± 4.0 mm', 'dist_comisura': '14.0 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Inactivo', 'musculo_linea2': 'Sin Conexión', 'funcion': 'Canal auxiliar desconectado', 'col': '#64748b', 'foto': '', 'activo': False},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Anterior Belly', 'musculo_linea2': 'Digástrico Submentoniano', 'funcion': 'Apertura mandibular activa', 'col': '#ef4444', 'foto': f'{OUT_DIR}/02_render_1023_exp2_belly_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '17.5 ± 3.5 mm', 'dist_menton': '28.0 ± 3.5 mm', 'dist_nariz': 'N/A (submentoniano)', 'dist_trago': '78.5 ± 4.5 mm', 'dist_comisura': '38.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_PruebaBelly1_Candela', 'E': 'E_PruebaBelly1_Candela', 'I': 'I_PruebaBelly1_Candela', 'O': 'O_PruebaBelly1_Candela', 'U': 'U_PruebaBelly1_Candela'},
            {'A': 'A_PruebaBelly2_Candela', 'E': 'E_PruebaBelly2_Candela', 'I': 'I_PruebaBelly2_Candela', 'O': 'O_PruebaBelly2_Candela', 'U': 'U_PruebaBelly2_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (10-23)', 'horario': '12:33 a 12:55', 'duracion': '22.0 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '2'}
    },

    # DIAPOSITIVA 18: 2026-10-23 Exp3 Triada Masetero Modiolo Orbicularis
    {
        'id': '18_2026-10-23_exp3',
        'titulo': "Experimento 2026-10-23: Exp3 Triada Masetero Modiolo Orbicularis",
        'camp': os.path.join(BASE_DRIVE, '2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela'),
        'subfolder': 'mediciones',
        'canales': [
            {'col_csv': 'Canal 0', 'musculo_linea1': 'Masetero', 'musculo_linea2': 'Cuerpo mandibular', 'funcion': 'Elevación mandibular y oclusión', 'col': '#f04e23', 'foto': f'{OUT_DIR}/01_render_1023_exp3_masetero_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '20.0 ± 3.5 mm', 'dist_menton': '66.0 ± 4.0 mm', 'dist_nariz': '73.0 ± 4.0 mm', 'dist_trago': '44.0 ± 3.5 mm', 'dist_comisura': '42.0 ± 3.5 mm'},
            {'col_csv': 'Canal 1', 'musculo_linea1': 'Modíolo', 'musculo_linea2': 'Nudo Comisural', 'funcion': 'Coordinación perioral', 'col': '#9d4edd', 'foto': f'{OUT_DIR}/02_render_1023_exp3_modiolo_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '19.0 ± 3.5 mm', 'dist_menton': '52.0 ± 4.0 mm', 'dist_nariz': '48.0 ± 4.0 mm', 'dist_trago': '99.0 ± 4.0 mm', 'dist_comisura': '11.0 ± 3.5 mm'},
            {'col_csv': 'Canal 2', 'musculo_linea1': 'Orbicular', 'musculo_linea2': 'Labio Superior', 'funcion': 'Protrusión y esfínter oral', 'col': '#f59e0b', 'foto': f'{OUT_DIR}/03_render_1023_exp3_orbi_3x4.png', 'tam_electrodo': '5 x 5 mm (±0.02 mm)', 'ied': '23.0 ± 3.5 mm', 'dist_menton': '46.0 ± 3.5 mm', 'dist_nariz': '17.0 ± 3.5 mm', 'dist_trago': '72.0 ± 4.0 mm', 'dist_comisura': '14.0 ± 3.5 mm'}
        ],
        'filtro_pca': lambda d, p: os.path.isdir(p) and not 'Secuencia' in d,
        'ses_curvas': [
            {'A': 'A_PruebaTriada1_Candela', 'E': 'E_PruebaTriada1_Candela', 'I': 'I_PruebaTriada1_Candela', 'O': 'O_PruebaTriada1_Candela', 'U': 'U_PruebaTriada1_Candela'},
            {'A': 'A_PruebaTriada2_Candela', 'E': 'E_PruebaTriada2_Candela', 'I': 'I_PruebaTriada2_Candela', 'O': 'O_PruebaTriada2_Candela', 'U': 'U_PruebaTriada2_Candela'},
            {'A': 'A_PruebaTriada3_Candela', 'E': 'E_PruebaTriada3_Candela', 'I': 'I_PruebaTriada3_Candela', 'O': 'O_PruebaTriada3_Candela', 'U': 'U_PruebaTriada3_Candela'},
            {'A': 'A_PruebaTriada4_Candela', 'E': 'E_PruebaTriada4_Candela', 'I': 'I_PruebaTriada4_Candela', 'O': 'O_PruebaTriada4_Candela', 'U': 'U_PruebaTriada4_Candela'}
        ],
        'cuaderno': {'sujeto': 'Candela (10-23)', 'horario': '15:55 a 16:30', 'duracion': '35.0 min total', 'fs': '6000 Hz', 'ref': 'Frente', 'series_n': '4'}
    }
]

def main():
    print(f"Iniciando compilación en lote de {len(CONFIGS)} diapositivas dinámicas...")
    generados_pdf = {}

    for cfg in CONFIGS:
        s_id = cfg['id']
        png_out = os.path.join(OUT_DIR, f"diapositiva_{s_id}.png")
        pdf_out = os.path.join(OUT_DIR, f"diapositiva_{s_id}.pdf")

        compilar_diapositiva_maestra(
            titulo_experimento=cfg['titulo'],
            base_dir_camp=cfg['camp'],
            canales_cfg=cfg['canales'],
            ses_list_curvas=cfg['ses_curvas'],
            filtro_mediciones_pca=cfg['filtro_pca'],
            info_cuaderno=cfg['cuaderno'],
            png_out=png_out,
            pdf_out=pdf_out,
            subfolder_mediciones=cfg.get('subfolder', 'mediciones')
        )
        generados_pdf[s_id] = pdf_out

    print("\nCompilación de diapositivas dinámicas finalizada.")

    # Lista final ordenada de las 18 diapositivas cronológicas
    lista_18_pdfs = [
        os.path.join(OUT_DIR, "diapositiva_01_2026-08-29.pdf"),
        os.path.join(OUT_DIR, "diapositiva_02_2026-08-30.pdf"),
        os.path.join(OUT_DIR, "diapositiva_01_2026-09-01.pdf"),         # Diapo 03 cronológica
        os.path.join(OUT_DIR, "diapositiva_02_2026-09-15.pdf"),         # Diapo 04 cronológica
        os.path.join(OUT_DIR, "diapositiva_03_2026-09-18.pdf"),         # Diapo 05 cronológica
        os.path.join(OUT_DIR, "diapositiva_04_2026-09-23_config1.pdf"), # Diapo 06 cronológica
        os.path.join(OUT_DIR, "diapositiva_07_2026-09-23_config2.pdf"), # Diapo 07 cronológica
        os.path.join(OUT_DIR, "diapositiva_08_2026-09-25_bloque1a.pdf"),# Diapo 08 cronológica
        os.path.join(OUT_DIR, "diapositiva_09_2026-09-25_bloque1b.pdf"),# Diapo 09 cronológica
        os.path.join(OUT_DIR, "diapositiva_10_2026-09-25_bloque2.pdf"), # Diapo 10 cronológica
        os.path.join(OUT_DIR, "diapositiva_11_2026-09-25_bloque3.pdf"), # Diapo 11 cronológica
        os.path.join(OUT_DIR, "diapositiva_12_2026-09-25_bloque4.pdf"), # Diapo 12 cronológica
        os.path.join(OUT_DIR, "diapositiva_13_2026-09-29.pdf"),         # Diapo 13 cronológica
        os.path.join(OUT_DIR, "diapositiva_14_2026-10-20_exp1.pdf"),    # Diapo 14 cronológica
        os.path.join(OUT_DIR, "diapositiva_15_2026-10-20_exp2.pdf"),    # Diapo 15 cronológica
        os.path.join(OUT_DIR, "diapositiva_16_2026-10-23_exp1.pdf"),    # Diapo 16 cronológica
        os.path.join(OUT_DIR, "diapositiva_17_2026-10-23_exp2.pdf"),    # Diapo 17 cronológica
        os.path.join(OUT_DIR, "diapositiva_18_2026-10-23_exp3.pdf")     # Diapo 18 cronológica
    ]

    atlas_out = os.path.join(OUT_DIR, "atlas_oficial_candela_completo_01_al_18.pdf")
    drive_out_18 = os.path.join(BASE_DRIVE, "atlas_oficial_candela_completo_01_al_18.pdf")
    drive_out_master = os.path.join(BASE_DRIVE, "atlas_oficial_candela_completo.pdf")

    cmd_unite = ["pdfunite"] + lista_18_pdfs + [atlas_out]
    print(f"\nUniendo 18 páginas en {atlas_out}...")
    subprocess.run(cmd_unite, check=True)

    shutil.copyfile(atlas_out, drive_out_18)
    shutil.copyfile(atlas_out, drive_out_master)
    print(f"Copia oficial guardada en {drive_out_master}")

if __name__ == '__main__':
    main()
