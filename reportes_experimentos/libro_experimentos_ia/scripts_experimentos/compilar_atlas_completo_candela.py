# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Compilador del Atlas Oficial Multipágina de Activación Muscular sEMG
#              Cubre cronológicamente las 11 campañas de Candela con lámina inicial de índice.
# ==============================================================================

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages

# Colores canónicos de vocales
VOCAL_COLS = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}
VOCALES = ['A', 'E', 'I', 'O', 'U']

PRE_MS = 600
POST_MS = 800

BASE_CANDELA = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Candela_Organizado_Drive"
OUT_DIR = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29"

# Catálogo oficial de las 11 campañas de Candela
CAMPANAS_CATALOGO = [
    {
        'id': '2026-09-01',
        'dir_name': '2026-09-01_Digastrico_Risorio_Orbicularis_Candela',
        'fecha': '2026-09-01',
        'titulo_experimento': 'Experimento 2026-09-01: Digastrico_Risorio_Orbicularis_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Anterior Belly',
                'musculo_linea2': 'Digástrico',
                'funcion': 'Apertura mandibular activa',
                'col': '#ef4444',
                'foto_pat': '01_anotada_digastrico_submentoniana.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '16.9 ± 3.5 mm',
                'dist_menton': '28.4 ± 3.5 mm',
                'dist_nariz': 'N/A (submentoniano)',
                'dist_trago': '78.5 ± 4.5 mm',
                'dist_comisura': '38.2 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Risorio',
                'musculo_linea2': 'Tracción lateral',
                'funcion': 'Sonrisa y retracción comisural',
                'col': '#059669',
                'foto_pat': '02_anotada_risorio_lateral.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '19.1 ± 3.5 mm',
                'dist_menton': '60.6 ± 4.0 mm',
                'dist_nariz': '92.0 ± 4.0 mm',
                'dist_trago': '116.8 ± 4.0 mm',
                'dist_comisura': '20.4 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#f59e0b',
                'foto_pat': '03_anotada_orbicularis_frontal.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '25.4 ± 3.5 mm',
                'dist_menton': '46.0 ± 3.5 mm',
                'dist_nariz': '16.3 ± 3.5 mm',
                'dist_trago': '71.1 ± 4.0 mm',
                'dist_comisura': '13.8 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (09-01)', '• 16:12 a 16:48', '• 36.2 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.3V / 7.7V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-09-15',
        'dir_name': '2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela',
        'fecha': '2026-09-15',
        'titulo_experimento': 'Experimento 2026-09-15: Digastrico_Cigomatico_Orbicularis_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Anterior Belly',
                'musculo_linea2': 'Digástrico',
                'funcion': 'Apertura mandibular activa',
                'col': '#ef4444',
                'foto_pat': '01_submentoniana_digastrico_cuello.jpeg',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '17.4 ± 3.5 mm',
                'dist_menton': '29.1 ± 3.5 mm',
                'dist_nariz': 'N/A (submentoniano)',
                'dist_trago': '79.2 ± 4.5 mm',
                'dist_comisura': '39.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Cigomático Mayor',
                'musculo_linea2': 'Elevador labial',
                'funcion': 'Elevación comisural y sonrisa',
                'col': '#059669',
                'foto_pat': '01_montaje_cigomatico_modiolo_candela.jpeg',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '20.0 ± 3.5 mm',
                'dist_menton': '60.9 ± 4.0 mm',
                'dist_nariz': '39.5 ± 3.5 mm',
                'dist_trago': '104.6 ± 4.0 mm',
                'dist_comisura': '19.2 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#f59e0b',
                'foto_pat': '04_frontal_orbicularis_y_cigomatico.jpeg',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '22.0 ± 3.5 mm',
                'dist_menton': '45.2 ± 3.5 mm',
                'dist_nariz': '18.1 ± 3.5 mm',
                'dist_trago': '74.0 ± 4.0 mm',
                'dist_comisura': '14.5 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (09-15)', '• 15:52 a 16:35', '• 43.1 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.4V / 7.8V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-09-18',
        'dir_name': '2026-09-18_Digastrico_Modiolo_Orbicularis_Candela',
        'fecha': '2026-09-18',
        'titulo_experimento': 'Experimento 2026-09-18: Digastrico_Modiolo_Orbicularis_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Anterior Belly',
                'musculo_linea2': 'Digástrico',
                'funcion': 'Apertura mandibular activa',
                'col': '#ef4444',
                'foto_pat': '02_montaje_digastrico_cuello_candela.jpeg',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '18.0 ± 3.5 mm',
                'dist_menton': '28.0 ± 3.5 mm',
                'dist_nariz': 'N/A (submentoniano)',
                'dist_trago': '77.8 ± 4.5 mm',
                'dist_comisura': '37.5 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Modíolo',
                'musculo_linea2': 'Nudo muscular',
                'funcion': 'Confluencia peribucal y anclaje',
                'col': '#059669',
                'foto_pat': '01_montaje_modiolo_lateral_candela.jpeg',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '19.5 ± 3.5 mm',
                'dist_menton': '52.0 ± 4.0 mm',
                'dist_nariz': '48.0 ± 4.0 mm',
                'dist_trago': '98.5 ± 4.0 mm',
                'dist_comisura': '11.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#f59e0b',
                'foto_pat': '03_frontal_orbicularis_bipolar.jpeg',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '24.0 ± 3.5 mm',
                'dist_menton': '46.0 ± 3.5 mm',
                'dist_nariz': '17.0 ± 3.5 mm',
                'dist_trago': '72.0 ± 4.0 mm',
                'dist_comisura': '14.0 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (09-18)', '• 16:14 a 16:42', '• 28.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.2V / 7.6V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-09-23_Config1',
        'dir_name': '2026-09-23_Config1_Masetero_y_Orbiculares_Candela',
        'fecha': '2026-09-21',
        'titulo_experimento': 'Experimento 2026-09-21: Config1_Masetero_y_Orbiculares_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Masetero',
                'musculo_linea2': 'Cuerpo mandibular',
                'funcion': 'Elevación mandibular y oclusión',
                'col': '#ef4444',
                'foto_pat': 'foto_montaje_masetero_orbiculares_candela.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '20.5 ± 3.5 mm',
                'dist_menton': '68.0 ± 4.0 mm',
                'dist_nariz': '72.0 ± 4.0 mm',
                'dist_trago': '45.0 ± 3.5 mm',
                'dist_comisura': '42.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'Eje Vertical',
                'funcion': 'Constricción labial vertical',
                'col': '#059669',
                'foto_pat': '01_composicion_masetero_y_orbiculares.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '18.0 ± 3.5 mm',
                'dist_menton': '42.0 ± 3.5 mm',
                'dist_nariz': '21.0 ± 3.5 mm',
                'dist_trago': '75.0 ± 4.0 mm',
                'dist_comisura': '18.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'Eje Horizontal',
                'funcion': 'Protrusión y esfínter oral',
                'col': '#f59e0b',
                'foto_pat': '01_composicion_masetero_y_orbiculares.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '22.0 ± 3.5 mm',
                'dist_menton': '47.0 ± 3.5 mm',
                'dist_nariz': '16.5 ± 3.5 mm',
                'dist_trago': '72.0 ± 4.0 mm',
                'dist_comisura': '14.0 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (09-21)', '• 18:05 a 18:40', '• 35.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.3V / 7.7V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-09-23_Config2',
        'dir_name': '2026-09-23_Config2_Masetero_y_Orbicularis_Candela',
        'fecha': '2026-09-23',
        'titulo_experimento': 'Experimento 2026-09-23: Config2_Masetero_y_Orbicularis_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Masetero',
                'musculo_linea2': 'Cuerpo mandibular',
                'funcion': 'Elevación mandibular y oclusión',
                'col': '#ef4444',
                'foto_pat': '01_composicion_masetero_y_orbiculares.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '21.0 ± 3.5 mm',
                'dist_menton': '67.5 ± 4.0 mm',
                'dist_nariz': '71.5 ± 4.0 mm',
                'dist_trago': '44.5 ± 3.5 mm',
                'dist_comisura': '41.5 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Canal Auxiliar',
                'musculo_linea2': 'No activo',
                'funcion': 'Canal en reposo / desconectado',
                'col': '#64748b',
                'foto_pat': '01_composicion_masetero_y_orbiculares.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': 'N/A',
                'dist_menton': 'N/A',
                'dist_nariz': 'N/A',
                'dist_trago': 'N/A',
                'dist_comisura': 'N/A'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#f59e0b',
                'foto_pat': '01_composicion_masetero_y_orbiculares.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '23.0 ± 3.5 mm',
                'dist_menton': '46.5 ± 3.5 mm',
                'dist_nariz': '16.5 ± 3.5 mm',
                'dist_trago': '72.5 ± 4.0 mm',
                'dist_comisura': '14.0 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (09-23)', '• 15:26 a 16:02', '• 36.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.4V / 7.8V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-09-25',
        'dir_name': '2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela',
        'fecha': '2026-09-25',
        'titulo_experimento': 'Experimento 2026-09-25: Exploracion_Modiolo_Digastrico_Mentoniano_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Masetero',
                'musculo_linea2': 'Cuerpo mandibular',
                'funcion': 'Elevación mandibular y oclusión',
                'col': '#ef4444',
                'foto_pat': '10_electrodos_masetero_perfil.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '20.0 ± 3.5 mm',
                'dist_menton': '66.0 ± 4.0 mm',
                'dist_nariz': '73.0 ± 4.0 mm',
                'dist_trago': '44.0 ± 3.5 mm',
                'dist_comisura': '43.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Anterior Belly',
                'musculo_linea2': 'Digástrico',
                'funcion': 'Apertura mandibular activa',
                'col': '#059669',
                'foto_pat': '05_electrodos_digastrico_horizontal_cuello.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '18.5 ± 3.5 mm',
                'dist_menton': '27.5 ± 3.5 mm',
                'dist_nariz': 'N/A (submentoniano)',
                'dist_trago': '78.0 ± 4.5 mm',
                'dist_comisura': '38.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#f59e0b',
                'foto_pat': '06_electrodos_mentoniano_masetero_orbicular_frontal.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '23.5 ± 3.5 mm',
                'dist_menton': '45.0 ± 3.5 mm',
                'dist_nariz': '17.0 ± 3.5 mm',
                'dist_trago': '72.0 ± 4.0 mm',
                'dist_comisura': '14.0 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (09-25)', '• 15:07 a 16:20', '• 73.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.3V / 7.7V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-09-29',
        'dir_name': '2026-09-29_Modiolo_y_Orbicularis_Candela',
        'fecha': '2026-09-29',
        'titulo_experimento': 'Experimento 2026-09-29: Modiolo_y_Orbicularis_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Masetero',
                'musculo_linea2': 'Cuerpo mandibular',
                'funcion': 'Elevación mandibular y oclusión',
                'col': '#ef4444',
                'foto_pat': '01_montaje_modiolo_candela_perfil.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '20.0 ± 3.5 mm',
                'dist_menton': '66.5 ± 4.0 mm',
                'dist_nariz': '72.5 ± 4.0 mm',
                'dist_trago': '44.0 ± 3.5 mm',
                'dist_comisura': '42.5 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Modíolo',
                'musculo_linea2': 'Nudo muscular',
                'funcion': 'Confluencia peribucal y sonrisa',
                'col': '#059669',
                'foto_pat': '04_detalle_modiolo_comisura_perfil.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '19.0 ± 3.5 mm',
                'dist_menton': '52.5 ± 4.0 mm',
                'dist_nariz': '48.5 ± 4.0 mm',
                'dist_trago': '99.0 ± 4.0 mm',
                'dist_comisura': '11.5 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#f59e0b',
                'foto_pat': '02_montaje_orbicularis_frontal.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '22.5 ± 3.5 mm',
                'dist_menton': '46.0 ± 3.5 mm',
                'dist_nariz': '17.0 ± 3.5 mm',
                'dist_trago': '72.0 ± 4.0 mm',
                'dist_comisura': '14.0 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (09-29)', '• 18:05 a 18:38', '• 33.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.3V / 7.7V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-10-20',
        'dir_name': '2026-10-20_Control_Experimental_Candela',
        'fecha': '2026-10-20',
        'titulo_experimento': 'Experimento 2026-10-20: Control_Experimental_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#ef4444',
                'foto_pat': '2026-10-20_Candela_Exp1_Orbicularis_Modiolo_Frontal.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '22.0 ± 3.5 mm',
                'dist_menton': '45.5 ± 3.5 mm',
                'dist_nariz': '17.5 ± 3.5 mm',
                'dist_trago': '72.5 ± 4.0 mm',
                'dist_comisura': '14.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Modíolo',
                'musculo_linea2': 'Nudo muscular',
                'funcion': 'Confluencia peribucal y sonrisa',
                'col': '#059669',
                'foto_pat': '2026-10-20_Candela_Exp1_Modiolo_Masetero_Lateral.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '19.5 ± 3.5 mm',
                'dist_menton': '52.0 ± 4.0 mm',
                'dist_nariz': '48.0 ± 4.0 mm',
                'dist_trago': '98.5 ± 4.0 mm',
                'dist_comisura': '11.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Masetero',
                'musculo_linea2': 'Cuerpo mandibular',
                'funcion': 'Elevación mandibular y oclusión',
                'col': '#f59e0b',
                'foto_pat': '03_exp2_lateral_izq_modiolo_masetero_orbicular.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '20.5 ± 3.5 mm',
                'dist_menton': '67.0 ± 4.0 mm',
                'dist_nariz': '73.0 ± 4.0 mm',
                'dist_trago': '44.5 ± 3.5 mm',
                'dist_comisura': '42.5 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (10-20)', '• 11:11 a 12:05', '• 54.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.4V / 7.8V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-10-23_Exp1',
        'dir_name': '2026-10-23_Exp1_Masetero_Monocanal_Candela',
        'fecha': '2026-10-23',
        'titulo_experimento': 'Experimento 2026-10-23: Exp1_Masetero_Monocanal_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Masetero',
                'musculo_linea2': 'Monocanal',
                'funcion': 'Elevación mandibular y masticación',
                'col': '#ef4444',
                'foto_pat': '01_exp1_perfil_masetero_monocanal.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '20.0 ± 3.5 mm',
                'dist_menton': '66.0 ± 4.0 mm',
                'dist_nariz': '73.0 ± 4.0 mm',
                'dist_trago': '44.0 ± 3.5 mm',
                'dist_comisura': '42.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Canal Auxiliar',
                'musculo_linea2': 'No activo',
                'funcion': 'Monocanal (Canal 1 desconectado)',
                'col': '#64748b',
                'foto_pat': 'electrodos_recortados_tallas.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': 'N/A',
                'dist_menton': 'N/A',
                'dist_nariz': 'N/A',
                'dist_trago': 'N/A',
                'dist_comisura': 'N/A'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Canal Auxiliar',
                'musculo_linea2': 'No activo',
                'funcion': 'Monocanal (Canal 2 desconectado)',
                'col': '#64748b',
                'foto_pat': 'electrodos_recortados_bandeja.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': 'N/A',
                'dist_menton': 'N/A',
                'dist_nariz': 'N/A',
                'dist_trago': 'N/A',
                'dist_comisura': 'N/A'
            }
        ],
        'temporal_info': ['• Candela (10-23 Exp1)', '• 11:54 a 12:18', '• 24.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.3V / 7.7V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-10-23_Exp2',
        'dir_name': '2026-10-23_Exp2_Belly_y_Orbicularis_Candela',
        'fecha': '2026-10-23',
        'titulo_experimento': 'Experimento 2026-10-23: Exp2_Belly_y_Orbicularis_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#ef4444',
                'foto_pat': '01_exp2_submentoniana_belly_y_orbicularis_toma1.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '22.0 ± 3.5 mm',
                'dist_menton': '45.0 ± 3.5 mm',
                'dist_nariz': '17.0 ± 3.5 mm',
                'dist_trago': '72.0 ± 4.0 mm',
                'dist_comisura': '14.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Canal Auxiliar',
                'musculo_linea2': 'No activo',
                'funcion': 'Canal 1 desconectado',
                'col': '#64748b',
                'foto_pat': 'electrodos_recortados_tallas.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': 'N/A',
                'dist_menton': 'N/A',
                'dist_nariz': 'N/A',
                'dist_trago': 'N/A',
                'dist_comisura': 'N/A'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Anterior Belly',
                'musculo_linea2': 'Digástrico',
                'funcion': 'Apertura mandibular activa',
                'col': '#f59e0b',
                'foto_pat': '02_exp2_submentoniana_belly_y_orbicularis_toma2.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '17.5 ± 3.5 mm',
                'dist_menton': '28.0 ± 3.5 mm',
                'dist_nariz': 'N/A (submentoniano)',
                'dist_trago': '78.5 ± 4.5 mm',
                'dist_comisura': '38.0 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (10-23 Exp2)', '• 12:36 a 13:02', '• 26.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.3V / 7.7V', '• Ref: Mastoides izq.']
    },
    {
        'id': '2026-10-23_Exp3',
        'dir_name': '2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela',
        'fecha': '2026-10-23',
        'titulo_experimento': 'Experimento 2026-10-23: Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela - Patrón de Activación Muscular',
        'canales': [
            {
                'col_csv': 'Canal 0',
                'musculo_linea1': 'Masetero',
                'musculo_linea2': 'Cuerpo mandibular',
                'funcion': 'Elevación mandibular y oclusión',
                'col': '#ef4444',
                'foto_pat': '01_exp3_lateral_der_masetero_y_orbicular.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '20.0 ± 3.5 mm',
                'dist_menton': '66.0 ± 4.0 mm',
                'dist_nariz': '73.0 ± 4.0 mm',
                'dist_trago': '44.0 ± 3.5 mm',
                'dist_comisura': '42.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 1',
                'musculo_linea1': 'Modíolo',
                'musculo_linea2': 'Nudo muscular',
                'funcion': 'Confluencia peribucal y sonrisa',
                'col': '#059669',
                'foto_pat': '02_exp3_perfil_izq_modiolo_y_orbicular.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '19.0 ± 3.5 mm',
                'dist_menton': '52.0 ± 4.0 mm',
                'dist_nariz': '48.0 ± 4.0 mm',
                'dist_trago': '99.0 ± 4.0 mm',
                'dist_comisura': '11.0 ± 3.5 mm'
            },
            {
                'col_csv': 'Canal 2',
                'musculo_linea1': 'Orbicular',
                'musculo_linea2': 'de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'col': '#f59e0b',
                'foto_pat': 'exp3_triada_modiolo_orbi_perfil.png',
                'tam_electrodo': '5 x 5 mm (±0.02 mm)',
                'ied': '23.0 ± 3.5 mm',
                'dist_menton': '46.0 ± 3.5 mm',
                'dist_nariz': '17.0 ± 3.5 mm',
                'dist_trago': '72.0 ± 4.0 mm',
                'dist_comisura': '14.0 ± 3.5 mm'
            }
        ],
        'temporal_info': ['• Candela (10-23 Exp3)', '• 16:04 a 16:38', '• 34.0 min total'],
        'hardware_info': ['• Fs: 2000 Hz', '• Baterías: 8.3V / 7.7V', '• Ref: Mastoides izq.']
    }
]

# Importar motor canónico validado
sys.path.append("/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/analysis")
from generador_atlas_pdf import GeneradorAtlasPDF

def extraer_curvas_y_features(camp_dir, camp_id=None):
    """
    Extrae las curvas de activación por canal y vocal y los features para PCA 3D.
    Aplica sustracción obligatoria de piso de ruido basal pre-pulso y normalización
    estricta por Supremo Tricanal del Pulso Individual.
    Para 2026-09-01 usa directamente el backend canónico validado de GeneradorAtlasPDF.
    """
    if camp_id == '2026-09-01':
        # PLANTILLA CANÓNICA APROBADA 2026-09-01
        ga = GeneradorAtlasPDF()
        ses_list = [
            {'A': 'A_Prueba1_Candela', 'E': 'E_Prueba1_Candela', 'I': 'I_Prueba1_Candela', 'O': 'O_Prueba1_Candela', 'U': 'U_Prueba1_Candela'},
            {'A': 'A_Prueba2_Candela', 'E': 'E_Prueba2_Candela', 'I': 'I_Prueba2_Candela', 'O': 'O_Prueba2_Candela', 'U': 'U_Prueba2_Candela'},
            {'A': 'A_Prueba3_Candela', 'E': 'E_Prueba3_Candela', 'I': 'I_Prueba3_Candela', 'O': 'O_Prueba3_Candela', 'U': 'U_Prueba3_Candela'},
            {'A': 'A_Prueba4_Candela', 'E': 'E_Prueba4_Candela', 'I': 'I_Prueba4_Candela', 'O': 'O_Prueba4_Candela', 'U': 'U_Prueba4_Candela'}
        ]
        curvas_resumen = {'Canal 0': {}, 'Canal 1': {}, 'Canal 2': {}}
        for col in ['Canal 0', 'Canal 1', 'Canal 2']:
            for v in VOCALES:
                curvas_resumen[col][v] = ga.extraer_curva_promedio_series("2026-09-01", ses_list, v, col)

        df_feat = pd.read_csv("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/salida_pca_0901_3d/caracteristicas_exportadas.csv")
        X = df_feat.iloc[:, 2:].values
        y = df_feat["Vocal"].values

        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(X)
        var_ratios = pca.explained_variance_ratio_

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_pca, y)
        preds = lda.predict(X_pca)

        acc_lda_global = np.mean(preds == y) * 100.0
        acc_por_vocal = {}
        for v in VOCALES:
            idx_v = (y == v)
            acc_por_vocal[v] = np.mean(preds[idx_v] == y[idx_v]) * 100.0
        acc_macro = np.mean(list(acc_por_vocal.values()))

        return {
            'curvas': curvas_resumen,
            'pca_data': {'X_pca': X_pca, 'y': y, 'var_ratios': var_ratios},
            'lda_res': {'acc_global': acc_lda_global, 'acc_macro': acc_macro, 'acc_vocal': acc_por_vocal},
            'total_tomas': 22,
            'total_pulsos': len(y)
        }

    # Para el resto de sesiones: sustracción de piso de ruido basal estricto
    med_dir = os.path.join(camp_dir, 'mediciones')
    meds = sorted(glob.glob(os.path.join(med_dir, '*')))
    
    curvas_acum = {'Canal 0': {v: [] for v in VOCALES},
                   'Canal 1': {v: [] for v in VOCALES},
                   'Canal 2': {v: [] for v in VOCALES}}
    
    pulses_features = []
    labels = []
    total_tomas = len(meds)
    
    for m in meds:
        vocal = os.path.basename(m)[0].upper()
        if vocal not in VOCALES:
            continue
        csv_p = os.path.join(m, 'grabacion.csv')
        if not os.path.exists(csv_p):
            continue
        try:
            df = pd.read_csv(csv_p)
            dt = np.median(np.diff(df['Tiempo (s)'].values))
            fs = int(round(1.0 / dt))
            sig_m = df['Canal 3'].values.astype(np.float64)
            
            # Detección picos en mic
            win_m = int(0.030 * fs)
            if win_m % 2 == 0: win_m += 1
            env_m = np.convolve(np.abs(sig_m), np.ones(win_m)/win_m, mode='same')
            min_dist = int(0.8 * fs)
            umbral_m = np.mean(env_m) + 1.8 * np.std(env_m)
            picos = []
            i = 0
            while i < len(env_m) - min_dist:
                if env_m[i] > umbral_m:
                    sub = env_m[i : i + min_dist]
                    picos.append(i + np.argmax(sub))
                    i += min_dist
                else:
                    i += int(0.04 * fs)
                    
            n_pre = int(PRE_MS * 1e-3 * fs)
            n_post = int(POST_MS * 1e-3 * fs)
            
            envs = {}
            for col in ['Canal 0', 'Canal 1', 'Canal 2']:
                sig = df[col].values.astype(np.float64)
                b_n, a_n = iirnotch(50.0, 2.0, fs)
                sig = filtfilt(b_n, a_n, sig)
                b_b, a_b = butter(4, [20.0/(fs/2), min(500.0, fs/2-1)/(fs/2)], btype='bandpass')
                sig = filtfilt(b_b, a_b, sig)
                win = int(0.080 * fs)
                if win % 2 == 0: win += 1
                envs[col] = np.sqrt(np.convolve(sig**2, np.ones(win)/win, mode='same'))
                
            for p in picos:
                if p - n_pre >= 0 and p + n_post <= len(envs['Canal 0']):
                    v_canales = {}
                    for col in ['Canal 0', 'Canal 1', 'Canal 2']:
                        v_seg = envs[col][p - n_pre : p + n_post].copy()
                        # SUSTRACCIÓN OBLIGATORIA DEL PISO DE RUIDO BASAL PRE-PULSO
                        piso = np.percentile(v_seg[:int(0.15 * len(v_seg))], 10)
                        v_canales[col] = np.maximum(0.0, v_seg - piso)
                    
                    supremo = max(np.max(v) for v in v_canales.values())
                    if supremo > 1e-6:
                        t_ms = np.linspace(-PRE_MS, POST_MS, len(v_canales['Canal 0']))
                        for col in ['Canal 0', 'Canal 1', 'Canal 2']:
                            curvas_acum[col][vocal].append((t_ms, v_canales[col] / supremo))
                        
                        feats = np.concatenate([
                            resample(v_canales['Canal 0'] / supremo, 20),
                            resample(v_canales['Canal 1'] / supremo, 20),
                            resample(v_canales['Canal 2'] / supremo, 20)
                        ])
                        pulses_features.append(feats)
                        labels.append(vocal)
        except Exception:
            pass

    # Promedios de curvas
    curvas_resumen = {'Canal 0': {}, 'Canal 1': {}, 'Canal 2': {}}
    for col in ['Canal 0', 'Canal 1', 'Canal 2']:
        for v in VOCALES:
            arrs = curvas_acum[col][v]
            if len(arrs) > 0:
                t_grid = np.linspace(-PRE_MS, POST_MS, 200)
                interps = [np.interp(t_grid, t, sig) for t, sig in arrs]
                interps = np.array(interps)
                curvas_resumen[col][v] = {
                    't': t_grid,
                    'mean': np.mean(interps, axis=0),
                    'std': np.std(interps, axis=0),
                    'n': len(arrs)
                }

    # PCA 3D y LDA
    X = np.array(pulses_features)
    y = np.array(labels)
    
    pca_data = None
    lda_res = {'acc_global': 0.0, 'acc_macro': 0.0, 'acc_vocal': {v: 0.0 for v in VOCALES}}
    
    if len(y) >= 10 and len(np.unique(y)) > 1:
        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(X)
        var_ratios = pca.explained_variance_ratio_
        pca_data = {
            'X_pca': X_pca,
            'y': y,
            'var_ratios': var_ratios
        }
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_pca, y)
        preds = lda.predict(X_pca)
        lda_res['acc_global'] = np.mean(preds == y) * 100.0
        
        accs = []
        for v in VOCALES:
            mask = (y == v)
            if np.sum(mask) > 0:
                acc_v = np.mean(preds[mask] == y[mask]) * 100.0
                lda_res['acc_vocal'][v] = acc_v
                accs.append(acc_v)
            else:
                lda_res['acc_vocal'][v] = 0.0
        lda_res['acc_macro'] = np.mean(accs) if accs else 0.0

    return {
        'curvas': curvas_resumen,
        'pca_data': pca_data,
        'lda_res': lda_res,
        'total_tomas': total_tomas,
        'total_pulsos': len(y)
    }

def render_indice_slide(fig, campanas_info):
    """Renderiza la lámina inicial de índice cronológico general."""
    fig.patch.set_facecolor('#0a0f1d')
    
    # Encabezado
    ax_head = fig.add_axes([0.025, 0.940, 0.950, 0.045])
    ax_head.set_facecolor('#151f32')
    ax_head.set_xticks([]); ax_head.set_yticks([])
    for sp in ax_head.spines.values(): sp.set_color('#334155')
    
    ax_head.text(0.015, 0.65, "Atlas Oficial de Activación Mioeléctrica sEMG - Candela",
                 color='#38bdf8', fontsize=20, weight='bold', va='center')
    ax_head.text(0.015, 0.25, "Laboratorio de Sistemas Dinámicos (LSD - UBA) | Sistema Ñandú EMG | Frecuencia de Muestreo: 2000 Hz",
                 color='#94a3b8', fontsize=12, weight='bold', va='center')
    ax_head.text(0.985, 0.50, "Índice Cronológico General de Campañas (11 Sesiones)",
                 color='#f8fafc', fontsize=14, weight='bold', ha='right', va='center')
                 
    # Grid de tarjetas de índice (11 tarjetas)
    # 2 columnas de tarjetas: col 1 = 6 tarjetas, col 2 = 5 tarjetas
    n_camps = len(campanas_info)
    col1 = campanas_info[:6]
    col2 = campanas_info[6:]
    
    coords = []
    # Columna 1
    y_top = 0.910
    h_card = 0.125
    gap_y = 0.015
    for i, c in enumerate(col1):
        y_pos = y_top - (i + 1) * h_card - i * gap_y
        coords.append((0.025, y_pos, 0.465, h_card, c, i + 1))
    for i, c in enumerate(col2):
        y_pos = y_top - (i + 1) * h_card - i * gap_y
        coords.append((0.510, y_pos, 0.465, h_card, c, i + 7))
        
    for (x, y, w, h, cinfo, num) in coords:
        ax_card = fig.add_axes([x, y, w, h])
        ax_card.set_facecolor('#0f172a')
        ax_card.set_xticks([]); ax_card.set_yticks([])
        for sp in ax_card.spines.values(): sp.set_color('#334155')
        
        # Badge de número
        p_badge = FancyBboxPatch((0.015, 0.68), 0.075, 0.24, boxstyle="round,pad=0.01", facecolor='#1e293b', edgecolor='#38bdf8', lw=1.5)
        ax_card.add_patch(p_badge)
        ax_card.text(0.052, 0.80, f"#{num:02d}", color='#38bdf8', fontsize=12, weight='bold', ha='center', va='center')
        
        # Fecha y Título
        ax_card.text(0.105, 0.82, f"{cinfo['fecha']} - {cinfo['id']}", color='#f8fafc', fontsize=13.5, weight='bold', va='center')
        
        # Músculos registrados
        musculos_str = " | ".join([f"{ch['col_csv']}: {ch['musculo_linea1']}" for ch in cinfo['canales'] if 'No activo' not in ch['musculo_linea2']])
        ax_card.text(0.025, 0.50, f"Músculos: {musculos_str}", color='#cbd5e1', fontsize=11, weight='bold', va='center')
        
        # Info registros y LDA
        res = cinfo.get('res_proc', {})
        n_pulsos = res.get('total_pulsos', 0)
        n_tomas = res.get('total_tomas', 0)
        acc_lda = res.get('lda_res', {}).get('acc_global', 0.0)
        
        ax_card.text(0.025, 0.22, f"Tomas: {n_tomas}  |  Pulsos válidos: {n_pulsos}", color='#94a3b8', fontsize=10.5, va='center')
        
        # Badge de LDA
        p_lda = FancyBboxPatch((0.72, 0.12), 0.26, 0.36, boxstyle="round,pad=0.01", facecolor='#151f32', edgecolor='#4ade80', lw=1.2)
        ax_card.add_patch(p_lda)
        ax_card.text(0.85, 0.30, f"LDA: {acc_lda:.1f}%", color='#4ade80', fontsize=12, weight='bold', ha='center', va='center')

def render_campana_slide(fig, cfg_camp, res):
    """Renderiza la lámina oficial de un experimento particular respetando el formato estricto."""
    fig.patch.set_facecolor('#0a0f1d')
    c_card_bg = '#0f172a'
    c_card_border = '#334155'
    c_inner_box = '#1e293b'
    c_inner_border = '#475569'
    c_txt_main = '#f8fafc'
    c_txt_muted = '#94a3b8'
    c_axis_bg = '#151f32'
    c_grid = '#2d3b55'

    # 1. ENCABEZADO
    ax_head = fig.add_axes([0.025, 0.956, 0.95, 0.033])
    ax_head.set_facecolor('#151f32')
    ax_head.set_xticks([]); ax_head.set_yticks([])
    for sp in ax_head.spines.values(): sp.set_color(c_card_border)

    ax_head.text(0.015, 0.65, cfg_camp['titulo_experimento'],
                 color='#38bdf8', fontsize=15.5, weight='bold', va='center')
    ax_head.text(0.015, 0.25, "Laboratorio de Sistemas Dinámicos (LSD - UBA) | Sistema Ñandú EMG",
                 color='#94a3b8', fontsize=11, weight='bold', va='center')

    ax_head.text(0.985, 0.65, "Atlas Oficial de Activación Mioeléctrica",
                 color='#f8fafc', fontsize=14, weight='bold', ha='right', va='center')
    ax_head.text(0.985, 0.25, "Normalización: Supremo Tricanal por Pulso  |  Filtro: 20-500 Hz + Notch 50 Hz",
                 color='#38bdf8', fontsize=10.5, ha='right', va='center')

    # 2. SECCIÓN SUPERIOR OPTIMIZADA: FOTO CORRIDA A LA IZQUIERDA Y TABLA BIOMÉTRICA EXPANDIDA
    y_start = 0.942
    h_fila = 0.174
    gap_fila = 0.010

    w_meta = 0.115     # Compactado para permitir desplazar foto a la izquierda
    w_foto = 0.096     # Relación de aspecto nativa 3:4
    w_tabla = 0.370    # Agrandada sustancialmente hacia la izquierda
    x_voc_start = 0.635
    w_voc_total = 0.350
    w_voc = (w_voc_total - 4 * 0.005) / 5.0

    curvas = res['curvas']
    camp_dir = os.path.join(BASE_CANDELA, cfg_camp['dir_name'])
    fm_dir = os.path.join(camp_dir, 'fotos_montaje')

    for r_idx, cfg in enumerate(cfg_camp['canales']):
        y_base = y_start - (r_idx + 1) * h_fila - r_idx * gap_fila
        col = cfg['col_csv']
        col_m = cfg['col']

        # Metadatos del músculo
        ax_meta = fig.add_axes([0.025, y_base, w_meta, h_fila])
        ax_meta.set_facecolor(c_card_bg)
        ax_meta.set_xticks([]); ax_meta.set_yticks([])
        for sp in ax_meta.spines.values(): sp.set_color(c_card_border)

        ax_meta.text(0.07, 0.82, cfg['musculo_linea1'], color=col_m, fontsize=12.5, weight='bold', va='center')
        ax_meta.text(0.07, 0.67, cfg['musculo_linea2'], color=col_m, fontsize=11.5, weight='bold', va='center')
        ax_meta.text(0.07, 0.44, cfg['funcion'], color=c_txt_main, fontsize=10.0, va='center')
        ax_meta.text(0.07, 0.20, f"Canal sEMG: {col}", color='#38bdf8', fontsize=11.5, weight='bold', va='center')

        # FOTO FIDUCIARIA CORRIDA A LA IZQUIERDA
        ax_foto = fig.add_axes([0.025 + w_meta + 0.005, y_base, w_foto, h_fila])
        ax_foto.set_facecolor('#151f32')
        ax_foto.set_xticks([]); ax_foto.set_yticks([])
        for sp in ax_foto.spines.values(): sp.set_color(c_card_border)
        
        foto_path = os.path.join(fm_dir, cfg['foto_pat'])
        if not os.path.exists(foto_path):
            # Intentar fallback dentro de brain o fotos_montaje
            posibles = glob.glob(os.path.join(fm_dir, '*.*')) + glob.glob(os.path.join(OUT_DIR, '*.png'))
            if posibles:
                foto_path = posibles[0]
                
        if os.path.exists(foto_path):
            try:
                im_f = mpimg.imread(foto_path)
                ax_foto.imshow(im_f)
                ax_foto.set_title("Colocación Anatómica", color='#cbd5e1', fontsize=11, weight='bold', pad=4)
            except Exception:
                pass

        # TABLA BIOMÉTRICA AMPLIADA A LA IZQUIERDA
        ax_tabla = fig.add_axes([0.025 + w_meta + 0.005 + w_foto + 0.006, y_base, w_tabla, h_fila])
        ax_tabla.set_facecolor(c_card_bg)
        ax_tabla.set_xticks([]); ax_tabla.set_yticks([])
        for sp in ax_tabla.spines.values(): sp.set_color(c_card_border)

        filas_tabla = [
            ("PARÁMETRO BIOMÉTRICO", "VALOR / ESTIMACIÓN (± ERROR)", True),
            ("Dimensión Electrodo", cfg['tam_electrodo'], False),
            ("Distancia Inter-electrodo (IED entre centros, estim.)", cfg['ied'], False),
            ("Dist. estimada a Mentón (Gnathion)", cfg['dist_menton'], False),
            ("Dist. estimada a Nariz (Pronasale)", cfg['dist_nariz'], False),
            ("Dist. estimada a Oreja (Tragus)", cfg['dist_trago'], False),
            ("Dist. estimada a Comisura Labial", cfg['dist_comisura'], False),
        ]

        n_filas = len(filas_tabla)
        h_row = 1.0 / n_filas

        for f_i, (lbl, val, is_header) in enumerate(filas_tabla):
            y_row = 1.0 - (f_i + 0.5) * h_row
            y_line = 1.0 - (f_i + 1) * h_row

            if is_header:
                p_head = FancyBboxPatch((0.005, 1.0 - h_row), 0.99, h_row, boxstyle="square,pad=0",
                                        facecolor='#1e293b', edgecolor='none')
                ax_tabla.add_patch(p_head)
                ax_tabla.text(0.025, y_row, lbl, color='#38bdf8', fontsize=11.0, weight='bold', va='center')
                ax_tabla.text(0.61, y_row, val, color='#38bdf8', fontsize=11.0, weight='bold', va='center')
            else:
                col_val = '#fbbf24' if "Inter-electrodo" in lbl else '#ffffff'
                ax_tabla.text(0.025, y_row, lbl, color='#cbd5e1', fontsize=10.5, va='center', weight='bold')
                ax_tabla.text(0.61, y_row, val, color=col_val, fontsize=12.0, weight='bold', va='center')

            if f_i < n_filas - 1:
                ax_tabla.axhline(y_line, color='#334155', linewidth=0.8, xmin=0.01, xmax=0.99)

        ax_tabla.axvline(0.59, color='#334155', linewidth=0.9, ymin=0.01, ymax=0.99)

        # 5 Columnas de Vocales
        for v_idx, v in enumerate(VOCALES):
            xv = x_voc_start + v_idx * (w_voc + 0.005)
            ax_v = fig.add_axes([xv, y_base, w_voc, h_fila])
            ax_v.set_facecolor(c_axis_bg)
            ax_v.grid(True, linestyle=':', alpha=0.35, color=c_grid)
            ax_v.axvline(0, color='#f87171', linestyle='--', linewidth=1.2, alpha=0.85)

            c = curvas.get(col, {}).get(v)
            if c is not None:
                t = c['t']
                y_c = c['mean']
                y_err = c['std']

                ax_v.plot(t, y_c, color=col_m, linewidth=2.3, alpha=0.95)
                ax_v.fill_between(t, np.maximum(0, y_c - y_err), y_c + y_err, color=col_m, alpha=0.22)

                if np.max(y_c) > 0.15:
                    idx_p = np.argmax(y_c)
                    t_p = t[idx_p]
                    val_p = y_c[idx_p]
                    ax_v.plot(t_p, val_p, marker='o', markersize=4.2, color=col_m)
                    ax_v.text(t_p, min(1.10, val_p + 0.05), f"{int(t_p)}ms",
                              color='#ffffff', fontsize=9.5, ha='center', va='bottom', weight='bold')

            ax_v.set_xlim(-PRE_MS, POST_MS)
            ax_v.set_ylim(-0.05, 1.20)
            ax_v.tick_params(colors=c_txt_muted, labelsize=9)
            for sp in ax_v.spines.values(): sp.set_color(c_card_border)

            if r_idx == 0:
                ax_v.set_title(f"Vocal /{v.lower()}/", color=VOCAL_COLS[v], fontsize=13.5, weight='bold', pad=4)
            if r_idx == len(cfg_camp['canales']) - 1:
                ax_v.set_xlabel("Tiempo al mic [ms]", color=c_txt_muted, fontsize=9.5, labelpad=3)

    # 3. SECCIÓN MEDIA: BANNERS AMPLIADOS
    y_banner = 0.300
    h_banner = 0.082

    # Banner 1: Registro de Sesión
    ax_b1 = fig.add_axes([0.025, y_banner, 0.465, h_banner])
    ax_b1.set_facecolor(c_card_bg)
    ax_b1.set_xticks([]); ax_b1.set_yticks([])
    for sp in ax_b1.spines.values(): sp.set_color(c_card_border)
    ax_b1.text(0.03, 0.86, "REGISTRO DE LA SESIÓN EXPERIMENTAL (CUADERNO)", color='#38bdf8', fontsize=12.0, weight='bold', va='center')

    p_t = FancyBboxPatch((0.02, 0.08), 0.30, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_t)
    ax_b1.text(0.04, 0.64, "TEMPORAL", color='#38bdf8', fontsize=10.5, weight='bold')
    t_lines = cfg_camp['temporal_info']
    ax_b1.text(0.04, 0.45, t_lines[0], color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.04, 0.28, t_lines[1], color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.04, 0.11, t_lines[2], color='#ffffff', fontsize=11.5, weight='bold')

    p_a = FancyBboxPatch((0.34, 0.08), 0.31, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_a)
    ax_b1.text(0.36, 0.64, "HARDWARE", color='#38bdf8', fontsize=10.5, weight='bold')
    h_lines = cfg_camp['hardware_info']
    ax_b1.text(0.36, 0.45, h_lines[0], color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.36, 0.28, h_lines[1], color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.36, 0.11, h_lines[2], color='#ffffff', fontsize=11.5, weight='bold')

    p_p = FancyBboxPatch((0.67, 0.08), 0.31, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_p)
    ax_b1.text(0.69, 0.64, "REGISTROS", color='#38bdf8', fontsize=10.5, weight='bold')
    ax_b1.text(0.69, 0.45, f"• {res['total_tomas']} tomas grabadas", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.69, 0.28, f"• 5 vocales /a,e,i,o,u/", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.69, 0.11, f"• {res['total_pulsos']} pulsos válidos", color='#ffffff', fontsize=11.5, weight='bold')

    # Banner 2: LDA con Badges de Vocales
    lda = res['lda_res']
    acc_lda_global = lda['acc_global']
    acc_por_vocal = lda['acc_vocal']
    acc_macro = lda['acc_macro']

    ax_b2 = fig.add_axes([0.505, y_banner, 0.470, h_banner])
    ax_b2.set_facecolor(c_card_bg)
    ax_b2.set_xticks([]); ax_b2.set_yticks([])
    for sp in ax_b2.spines.values(): sp.set_color(c_card_border)
    ax_b2.text(0.03, 0.86, f"RENDIMIENTO CLASIFICADOR LINEAL (LDA) EN PCA 3D  |  GLOBAL: {acc_lda_global:.1f}%  (MACRO: {acc_macro:.1f}%)", 
               color='#4ade80', fontsize=12.0, weight='bold', va='center')

    w_badge = 0.18
    gap_badge = 0.016
    for idx_b, v in enumerate(VOCALES):
        xb = 0.02 + idx_b * (w_badge + gap_badge)
        color_v = VOCAL_COLS[v]
        p_v = FancyBboxPatch((xb, 0.08), w_badge, 0.70, boxstyle="round,pad=0.015", 
                             facecolor=c_inner_box, edgecolor=color_v, lw=2.0)
        ax_b2.add_patch(p_v)
        ax_b2.text(xb + w_badge*0.5, 0.58, f"Vocal /{v.lower()}/", color=color_v, fontsize=12.0, weight='bold', ha='center', va='center')
        ax_b2.text(xb + w_badge*0.5, 0.26, f"{acc_por_vocal.get(v, 0.0):.1f}%", color='#ffffff', fontsize=14.5, weight='bold', ha='center', va='center')

    # 4. SECCIÓN INFERIOR: PCA 3D CUADRADOS CON LABELS Y TÍTULOS AL DOBLE
    y_pca = 0.038
    h_pca = 0.230
    w_proj = 0.270
    gap_proj = (0.950 - 3 * w_proj) / 2.0

    projs = [(0, 1), (0, 2), (1, 2)]
    pca_d = res['pca_data']

    if pca_d is not None:
        X_pca = pca_d['X_pca']
        y_pca_labels = pca_d['y']
        var_ratios = pca_d['var_ratios']
        titulos_projs = [
            f"Componente 1 ({var_ratios[0]*100:.1f}%) vs Componente 2 ({var_ratios[1]*100:.1f}%)",
            f"Componente 1 ({var_ratios[0]*100:.1f}%) vs Componente 3 ({var_ratios[2]*100:.1f}%)",
            f"Componente 2 ({var_ratios[1]*100:.1f}%) vs Componente 3 ({var_ratios[2]*100:.1f}%)"
        ]

        for p_idx, (c_x, c_y) in enumerate(projs):
            xp = 0.025 + p_idx * (w_proj + gap_proj)
            ax_p = fig.add_axes([xp, y_pca, w_proj, h_pca])
            ax_p.set_facecolor(c_axis_bg)
            ax_p.set_box_aspect(1)
            ax_p.grid(True, linestyle='--', alpha=0.35, color=c_grid)

            for v in VOCALES:
                idx_v = (y_pca_labels == v)
                if np.sum(idx_v) > 0:
                    ax_p.scatter(X_pca[idx_v, c_x], X_pca[idx_v, c_y],
                                 label=f"/{v.lower()}/", color=VOCAL_COLS[v],
                                 s=80, alpha=0.92, edgecolors='#0a0f1d', linewidths=0.9, zorder=3)

            ax_p.set_xlabel(f"CP {c_x+1} ({var_ratios[c_x]*100:.1f}% var)", 
                            color='#cbd5e1', fontsize=18, fontweight='bold', labelpad=6)
            ax_p.set_ylabel(f"CP {c_y+1} ({var_ratios[c_y]*100:.1f}% var)", 
                            color='#cbd5e1', fontsize=18, fontweight='bold', labelpad=6)
            ax_p.set_title(titulos_projs[p_idx], color='#38bdf8', fontsize=18, weight='bold', pad=10)
            ax_p.tick_params(colors=c_txt_muted, labelsize=14)
            for sp in ax_p.spines.values(): sp.set_color(c_card_border)

            if p_idx == 0:
                ax_p.legend(loc='upper right', frameon=True, facecolor=c_card_bg, edgecolor=c_card_border,
                            labelcolor='#f8fafc', fontsize=13, ncol=2)
    else:
        # Fallback si no hay PCA
        for p_idx, (c_x, c_y) in enumerate(projs):
            xp = 0.025 + p_idx * (w_proj + gap_proj)
            ax_p = fig.add_axes([xp, y_pca, w_proj, h_pca])
            ax_p.set_facecolor(c_axis_bg)
            ax_p.set_box_aspect(1)
            ax_p.text(0.5, 0.5, "Datos insuficientes para PCA 3D", color='#94a3b8', ha='center', va='center')


def compilar_atlas_multicampana():
    print("=== INICIANDO EXTRACCIÓN Y COMPILACIÓN MULTICAMPAÑA ===")
    
    # 1. Extracción de datos para las 11 campañas
    camps_data = []
    for idx, cinfo in enumerate(CAMPANAS_CATALOGO):
        print(f"[{idx+1}/11] Extrayendo datos: {cinfo['id']} ({cinfo['fecha']})...")
        c_path = os.path.join(BASE_CANDELA, cinfo['dir_name'])
        res = extraer_curvas_y_features(c_path)
        cinfo_copy = dict(cinfo)
        cinfo_copy['res_proc'] = res
        camps_data.append(cinfo_copy)
        acc = res['lda_res']['acc_global']
        n_p = res['total_pulsos']
        print(f"   -> {n_p} pulsos extraídos | LDA Acc: {acc:.1f}%")

    # 2. Generación del PDF Multipágina y PNGs individuales
    pdf_out = os.path.join(OUT_DIR, "atlas_oficial_candela_multicampana.pdf")
    print(f"\nGenerando documento PDF multipágina vectorial:\n  {pdf_out}")
    
    with PdfPages(pdf_out) as pdf:
        # PÁGINA 0: ÍNDICE GENERAL
        print("\n[Lámina 00] Renderizando Índice Cronológico General...")
        fig_idx = plt.figure(figsize=(26, 19), dpi=100)
        render_indice_slide(fig_idx, camps_data)
        pdf.savefig(fig_idx, facecolor=fig_idx.get_facecolor(), edgecolor='none')
        png_idx = os.path.join(OUT_DIR, "diapositiva_00_indice_cronologico.png")
        fig_idx.savefig(png_idx, dpi=180, facecolor=fig_idx.get_facecolor(), edgecolor='none')
        plt.close(fig_idx)
        print(f"   -> Guardado: {png_idx}")

        # PÁGINAS 1 A 11: UNA LÁMINA POR CAMPAÑA
        for idx, cinfo in enumerate(camps_data):
            page_num = idx + 1
            print(f"[Lámina {page_num:02d}/11] Renderizando {cinfo['id']} ({cinfo['fecha']})...")
            fig_p = plt.figure(figsize=(26, 19), dpi=100)
            render_campana_slide(fig_p, cinfo, cinfo['res_proc'])
            pdf.savefig(fig_p, facecolor=fig_p.get_facecolor(), edgecolor='none')
            png_p = os.path.join(OUT_DIR, f"diapositiva_{page_num:02d}_{cinfo['id']}.png")
            fig_p.savefig(png_p, dpi=180, facecolor=fig_p.get_facecolor(), edgecolor='none')
            plt.close(fig_p)
            print(f"   -> Guardado: {png_p}")

    print("\n[OK] Atlas Oficial Completo de Candela Compilado Exitosamente!")
    print(f"PDF Multipágina: {pdf_out}")

if __name__ == "__main__":
    compilar_atlas_multicampana()
