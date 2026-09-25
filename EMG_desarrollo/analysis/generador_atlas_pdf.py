# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Generador del Atlas Oficial de Activación Muscular sEMG en PDF.
#              - Lectura directa y exclusiva desde grabacion.csv con pandas.
#              - Promedio multiserie consolidado por músculo y día.
#              - Normalización estricta por Supremo Tricanal del Pulso Individual.
#              - Opción de fotografía lateral de colocación de electrodos.
#              - Documento PDF multipágina vectorial mediante PdfPages.
#              - Soporte de temas: Publicación Científica (Blanco) y Modo Oscuro.
# ==============================================================================

import os
import sys
import argparse
import textwrap
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# Ventana temporal fisiológica (-600 ms a +800 ms respecto al micrófono = 1400 ms)
PRE_MS = 600
POST_MS = 800

# Catálogo consolidado oficial de mediciones
CATALOGO_SUJETOS = [
    # ========================== 1. SUJETO: CANDELA ==========================
    {
        'sujeto': 'Candela',
        'registros': [
            {
                'musculo': 'Digástrico Anterior',
                'funcion': 'Apertura mandibular activa',
                'detalle': '09-01 (Pruebas 1 a 4)',
                'col_csv': 'Canal 0',
                'col': '#ef4444',
                'fecha': '2026-09-01',
                'ses_list': [
                    {'A': 'A_Prueba1_Candela', 'E': 'E_Prueba1_Candela', 'I': 'I_Prueba1_Candela', 'O': 'O_Prueba1_Candela', 'U': 'U_Prueba1_Candela'},
                    {'A': 'A_Prueba2_Candela', 'E': 'E_Prueba2_Candela', 'I': 'I_Prueba2_Candela', 'O': 'O_Prueba2_Candela', 'U': 'U_Prueba2_Candela'},
                    {'A': 'A_Prueba3_Candela', 'E': 'E_Prueba3_Candela', 'I': 'I_Prueba3_Candela', 'O': 'O_Prueba3_Candela', 'U': 'U_Prueba3_Candela'},
                    {'A': 'A_Prueba4_Candela', 'E': 'E_Prueba4_Candela', 'I': 'I_Prueba4_Candela', 'O': 'O_Prueba4_Candela', 'U': 'U_Prueba4_Candela'}
                ]
            },
            {
                'musculo': 'Digástrico Anterior',
                'funcion': 'Apertura mandibular activa',
                'detalle': '09-16 (Series 1 a 4)',
                'col_csv': 'Canal 0',
                'col': '#dc2626',
                'fecha': '2026-09-16',
                'foto_override': 'EMG_desarrollo/fotos/WhatsApp Image 2026-09-16 at 12.52.46.jpeg',
                'ses_list': [
                    {'A': 'A_Serie1_Candela', 'E': 'E_Serie1_Candela', 'I': 'I_Serie1_Candela', 'O': 'O_Serie1_Candela', 'U': 'U_Serie1_Candela'},
                    {'A': 'A_Serie2_Candela', 'E': 'E_Serie2_Candela', 'I': 'I_Serie2_Candela', 'O': 'O_Serie2_Candela', 'U': 'U_Serie2_Candela'},
                    {'A': 'A_Serie3_Candela', 'E': 'E_Serie3_Candela', 'I': 'I_Serie3_Candela', 'O': 'O_Serie3_Candela', 'U': 'U_Serie3_Candela'},
                    {'A': 'A_Serie4_Candela', 'E': 'E_Serie4_Candela', 'I': 'I_Serie4_Candela', 'O': 'O_Serie4_Candela', 'U': 'U_Serie4_Candela'}
                ]
            },
            {
                'musculo': 'Milohioideo (Desplazado)',
                'funcion': 'Piso bucal (desplazamiento sensor)',
                'detalle': '08-29 (Pruebas 1 y 2)',
                'col_csv': 'Canal 1',
                'col': '#f87171',
                'fecha': '2026-08-29',
                'ses_list': [
                    {'A': 'A_Prueba1_Cande', 'E': 'E_Prueba1_Cande', 'I': 'I_Prueba1_Cande', 'O': 'O_Prueba1_Cande', 'U': 'U_Prueba1_Cande'},
                    {'A': 'A_Prueba2_Cande', 'E': 'E_Prueba2_Cande', 'I': 'I_Prueba2_Cande', 'O': 'O_Prueba2_Cande', 'U': 'U_Prueba2_Cande'}
                ]
            },
            {
                'musculo': 'Cigomático Mayor',
                'funcion': 'Retracción comisural y elevación',
                'detalle': '09-16 (Series 1 a 4)',
                'col_csv': 'Canal 2',
                'col': '#10b981',
                'fecha': '2026-09-16',
                'foto_override': 'EMG_desarrollo/fotos/WhatsApp Image 2026-09-16 at 12.52.45.jpeg',
                'ses_list': [
                    {'A': 'A_Serie1_Candela', 'E': 'E_Serie1_Candela', 'I': 'I_Serie1_Candela', 'O': 'O_Serie1_Candela', 'U': 'U_Serie1_Candela'},
                    {'A': 'A_Serie2_Candela', 'E': 'E_Serie2_Candela', 'I': 'I_Serie2_Candela', 'O': 'O_Serie2_Candela', 'U': 'U_Serie2_Candela'},
                    {'A': 'A_Serie3_Candela', 'E': 'E_Serie3_Candela', 'I': 'I_Serie3_Candela', 'O': 'O_Serie3_Candela', 'U': 'U_Serie3_Candela'},
                    {'A': 'A_Serie4_Candela', 'E': 'E_Serie4_Candela', 'I': 'I_Serie4_Candela', 'O': 'O_Serie4_Candela', 'U': 'U_Serie4_Candela'}
                ]
            },
            {
                'musculo': 'Risorio (Tracción lateral)',
                'funcion': 'Sonrisa y tracción transversal',
                'detalle': '09-01 (Pruebas 1 a 4)',
                'col_csv': 'Canal 1',
                'col': '#059669',
                'fecha': '2026-09-01',
                'ses_list': [
                    {'A': 'A_Prueba1_Candela', 'E': 'E_Prueba1_Candela', 'I': 'I_Prueba1_Candela', 'O': 'O_Prueba1_Candela', 'U': 'U_Prueba1_Candela'},
                    {'A': 'A_Prueba2_Candela', 'E': 'E_Prueba2_Candela', 'I': 'I_Prueba2_Candela', 'O': 'O_Prueba2_Candela', 'U': 'U_Prueba2_Candela'},
                    {'A': 'A_Prueba3_Candela', 'E': 'E_Prueba3_Candela', 'I': 'I_Prueba3_Candela', 'O': 'O_Prueba3_Candela', 'U': 'U_Prueba3_Candela'},
                    {'A': 'A_Prueba4_Candela', 'E': 'E_Prueba4_Candela', 'I': 'I_Prueba4_Candela', 'O': 'O_Prueba4_Candela', 'U': 'U_Prueba4_Candela'}
                ]
            },
            {
                'musculo': 'Modíolo (Nudo Sonrisa)',
                'funcion': 'Confluencia muscular peribucal',
                'detalle': '08-30 (Pruebas 4 y 5)',
                'col_csv': 'Canal 1',
                'col': '#34d399',
                'fecha': '2026-08-30',
                'ses_list': [
                    {'A': 'A_Prueba4_Cande', 'E': 'E_Prueba4_Cande', 'I': 'I_Prueba4_Cande', 'O': 'O_Prueba4_Cande', 'U': 'U_Prueba4_Cande'},
                    {'A': 'A_Prueba5_Cande', 'E': 'E_Prueba5_Cande', 'I': 'I_Prueba5_Cande', 'O': 'O_Prueba5_Cande', 'U': 'U_Prueba4_Cande'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'detalle': '08-25 (Día 1, Prueba 1)',
                'col_csv': 'Canal 0',
                'col': '#f59e0b',
                'fecha': '2026-08-25',
                'ses_list': [
                    {'A': 'A_Prueba1_Candela', 'E': 'E_Prueba1_Candela', 'I': 'I_Prueba1_Candela', 'O': 'O_Prueba1_Candela', 'U': 'U_Prueba1_Candela'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'detalle': '08-29 (Día 2, Pruebas 1 y 2)',
                'col_csv': 'Canal 2',
                'col': '#d97706',
                'fecha': '2026-08-29',
                'ses_list': [
                    {'A': 'A_Prueba1_Cande', 'E': 'E_Prueba1_Cande', 'I': 'I_Prueba1_Cande', 'O': 'O_Prueba1_Cande', 'U': 'U_Prueba1_Cande'},
                    {'A': 'A_Prueba2_Cande', 'E': 'E_Prueba2_Cande', 'I': 'I_Prueba2_Cande', 'O': 'O_Prueba2_Cande', 'U': 'U_Prueba2_Cande'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'detalle': '08-30 (Día 3, Pruebas 4 y 5)',
                'col_csv': 'Canal 2',
                'col': '#b45309',
                'fecha': '2026-08-30',
                'ses_list': [
                    {'A': 'A_Prueba4_Cande', 'E': 'E_Prueba4_Cande', 'I': 'I_Prueba4_Cande', 'O': 'O_Prueba4_Cande', 'U': 'U_Prueba4_Cande'},
                    {'A': 'A_Prueba5_Cande', 'E': 'E_Prueba5_Cande', 'I': 'I_Prueba5_Cande', 'O': 'O_Prueba5_Cande', 'U': 'U_Prueba5_Cande'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'detalle': '09-01 (Día 4, Pruebas 1 a 4)',
                'col_csv': 'Canal 2',
                'col': '#92400e',
                'fecha': '2026-09-01',
                'ses_list': [
                    {'A': 'A_Prueba1_Candela', 'E': 'E_Prueba1_Candela', 'I': 'I_Prueba1_Candela', 'O': 'O_Prueba1_Candela', 'U': 'U_Prueba1_Candela'},
                    {'A': 'A_Prueba2_Candela', 'E': 'E_Prueba2_Candela', 'I': 'I_Prueba2_Candela', 'O': 'O_Prueba2_Candela', 'U': 'U_Prueba2_Candela'},
                    {'A': 'A_Prueba3_Candela', 'E': 'E_Prueba3_Candela', 'I': 'I_Prueba3_Candela', 'O': 'O_Prueba3_Candela', 'U': 'U_Prueba3_Candela'},
                    {'A': 'A_Prueba4_Candela', 'E': 'E_Prueba4_Candela', 'I': 'I_Prueba4_Candela', 'O': 'O_Prueba4_Candela', 'U': 'U_Prueba4_Candela'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción y protrusión labial',
                'detalle': '09-16 (Día 5, Series 1 a 4)',
                'col_csv': 'Canal 1',
                'col': '#78350f',
                'fecha': '2026-09-16',
                'foto_override': 'EMG_desarrollo/fotos/orbicularis_oris_candela_frontal.jpeg',
                'ses_list': [
                    {'A': 'A_Serie1_Candela', 'E': 'E_Serie1_Candela', 'I': 'I_Serie1_Candela', 'O': 'O_Serie1_Candela', 'U': 'U_Serie1_Candela'},
                    {'A': 'A_Serie2_Candela', 'E': 'E_Serie2_Candela', 'I': 'I_Serie2_Candela', 'O': 'O_Serie2_Candela', 'U': 'U_Serie2_Candela'},
                    {'A': 'A_Serie3_Candela', 'E': 'E_Serie3_Candela', 'I': 'I_Serie3_Candela', 'O': 'O_Serie3_Candela', 'U': 'U_Serie3_Candela'},
                    {'A': 'A_Serie4_Candela', 'E': 'E_Serie4_Candela', 'I': 'I_Serie4_Candela', 'O': 'O_Serie4_Candela', 'U': 'U_Serie4_Candela'}
                ]
            }
        ]
    },

    # ========================== 2. SUJETO: LUCAS ==========================
    {
        'sujeto': 'Lucas',
        'registros': [
            {
                'musculo': 'Submentoniano (Belly/Milo)',
                'funcion': 'Apertura mandibular y descenso lingual',
                'detalle': '07-10 (Tomas T1 a T7)',
                'col_csv': 'Canal 0',
                'col': '#dc2626',
                'fecha': '2026-07-10',
                'ses_list': [
                    {'A': 'A_T1_Lucas', 'E': 'E_T1_Lucas', 'I': 'I_T1_Lucas', 'O': 'O_T1_Lucas', 'U': 'U_T1_Lucas'},
                    {'A': 'A_T2_Lucas', 'E': 'E_T2_Lucas', 'I': 'I_T2_Lucas', 'O': 'O_T2_Lucas', 'U': 'U_T2_Lucas'},
                    {'A': 'A_T3_Lucas', 'E': 'E_T3_Lucas', 'I': 'I_T3_Lucas', 'O': 'O_T3_Lucas', 'U': 'U_T3_Lucas'},
                    {'A': 'A_T4_Lucas', 'E': 'E_T4_Lucas', 'I': 'I_T4_Lucas', 'O': 'O_T4_Lucas', 'U': 'U_T4_Lucas'},
                    {'A': 'A_T5_Lucas', 'E': 'E_T5_Lucas', 'I': 'I_T5_Lucas', 'O': 'O_T5_Lucas', 'U': 'U_T5_Lucas'},
                    {'A': 'A_T6_Lucas', 'E': 'E_T6_Lucas', 'I': 'I_T6_Lucas', 'O': 'O_T6_Lucas', 'U': 'U_T6_Lucas'},
                    {'A': 'A_T7_Lucas', 'E': 'E_T7_Lucas', 'I': 'I_T7_Lucas', 'O': 'O_T7_Lucas', 'U': 'U_T7_Lucas'}
                ]
            },
            {
                'musculo': 'Depresor Ángulo Oral (DAO)',
                'funcion': 'Depresión comisural labial',
                'detalle': '07-10 (Tomas T1 a T7)',
                'col_csv': 'Canal 1',
                'col': '#0284c7',
                'fecha': '2026-07-10',
                'ses_list': [
                    {'A': 'A_T1_Lucas', 'E': 'E_T1_Lucas', 'I': 'I_T1_Lucas', 'O': 'O_T1_Lucas', 'U': 'U_T1_Lucas'},
                    {'A': 'A_T2_Lucas', 'E': 'E_T2_Lucas', 'I': 'I_T2_Lucas', 'O': 'O_T2_Lucas', 'U': 'U_T2_Lucas'},
                    {'A': 'A_T3_Lucas', 'E': 'E_T3_Lucas', 'I': 'I_T3_Lucas', 'O': 'O_T3_Lucas', 'U': 'U_T3_Lucas'},
                    {'A': 'A_T4_Lucas', 'E': 'E_T4_Lucas', 'I': 'I_T4_Lucas', 'O': 'O_T4_Lucas', 'U': 'U_T4_Lucas'},
                    {'A': 'A_T5_Lucas', 'E': 'E_T5_Lucas', 'I': 'I_T5_Lucas', 'O': 'O_T5_Lucas', 'U': 'U_T5_Lucas'},
                    {'A': 'A_T6_Lucas', 'E': 'E_T6_Lucas', 'I': 'I_T6_Lucas', 'O': 'O_T6_Lucas', 'U': 'U_T6_Lucas'},
                    {'A': 'A_T7_Lucas', 'E': 'E_T7_Lucas', 'I': 'I_T7_Lucas', 'O': 'O_T7_Lucas', 'U': 'U_T7_Lucas'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción y redondeo labial',
                'detalle': 'Día 1: 06-01 (Toma 1)',
                'col_csv': 'Canal 2',
                'col': '#f59e0b',
                'fecha': '2026-07-10',
                'ses_list': [
                    {'A': 'A_T1_Lucas', 'E': 'E_T1_Lucas', 'I': 'I_T1_Lucas', 'O': 'O_T1_Lucas', 'U': 'U_T1_Lucas'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción y redondeo labial',
                'detalle': 'Día 2: 06-03 (Tomas 2 y 3)',
                'col_csv': 'Canal 2',
                'col': '#d97706',
                'fecha': '2026-07-10',
                'ses_list': [
                    {'A': 'A_T2_Lucas', 'E': 'E_T2_Lucas', 'I': 'I_T2_Lucas', 'O': 'O_T2_Lucas', 'U': 'U_T2_Lucas'},
                    {'A': 'A_T3_Lucas', 'E': 'E_T3_Lucas', 'I': 'I_T3_Lucas', 'O': 'O_T3_Lucas', 'U': 'U_T3_Lucas'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción y redondeo labial',
                'detalle': 'Día 3: 06-10 (Tomas 4 a 7)',
                'col_csv': 'Canal 2',
                'col': '#b45309',
                'fecha': '2026-07-10',
                'ses_list': [
                    {'A': 'A_T4_Lucas', 'E': 'E_T4_Lucas', 'I': 'I_T4_Lucas', 'O': 'O_T4_Lucas', 'U': 'U_T4_Lucas'},
                    {'A': 'A_T5_Lucas', 'E': 'E_T5_Lucas', 'I': 'I_T5_Lucas', 'O': 'O_T5_Lucas', 'U': 'U_T5_Lucas'},
                    {'A': 'A_T6_Lucas', 'E': 'E_T6_Lucas', 'I': 'I_T6_Lucas', 'O': 'O_T6_Lucas', 'U': 'U_T6_Lucas'},
                    {'A': 'A_T7_Lucas', 'E': 'E_T7_Lucas', 'I': 'I_T7_Lucas', 'O': 'O_T7_Lucas', 'U': 'U_T7_Lucas'}
                ]
            }
        ]
    },

    # ========================== 3. SUJETO: SANTI ==========================
    {
        'sujeto': 'Santi',
        'registros': [
            {
                'musculo': 'Milohioideo (Piso bucal)',
                'funcion': 'Elevación del piso bucal',
                'detalle': '06-22 (Prueba 1)',
                'col_csv': 'Canal 0',
                'col': '#dc2626',
                'fecha': '2026-06-22',
                'ses_list': [
                    {'A': 'A_Prueba1_SANTI', 'E': 'E_Prueba1_SANTI', 'I': 'I_Prueba1_SANTI', 'O': 'O_Prueba1_SANTI', 'U': 'U_Prueba1_SANTI'}
                ]
            },
            {
                'musculo': 'Depresor Ángulo (DAO)',
                'funcion': 'Depresión comisural',
                'detalle': '06-22 (Prueba 1)',
                'col_csv': 'Canal 1',
                'col': '#0284c7',
                'fecha': '2026-06-22',
                'ses_list': [
                    {'A': 'A_Prueba1_SANTI', 'E': 'E_Prueba1_SANTI', 'I': 'I_Prueba1_SANTI', 'O': 'O_Prueba1_SANTI', 'U': 'U_Prueba1_SANTI'}
                ]
            },
            {
                'musculo': 'Orbicular de los Labios',
                'funcion': 'Constricción peribucal',
                'detalle': '06-22 (Prueba 1)',
                'col_csv': 'Canal 2',
                'col': '#f59e0b',
                'fecha': '2026-06-22',
                'ses_list': [
                    {'A': 'A_Prueba1_SANTI', 'E': 'E_Prueba1_SANTI', 'I': 'I_Prueba1_SANTI', 'O': 'O_Prueba1_SANTI', 'U': 'U_Prueba1_SANTI'}
                ]
            }
        ]
    },

    # ========================== 4. SUJETO: PETRA ==========================
    {
        'sujeto': 'Petra',
        'registros': [
            {
                'musculo': 'Digástrico Anterior',
                'funcion': 'Apertura mandibular activa (Silicona med1)',
                'detalle': '08-28 (Silicona med1)',
                'col_csv': 'Canal 0',
                'col': '#dc2626',
                'fecha': '2026-08-28',
                'ses_list': [
                    {'A': 'A_med1_clase24_Petra', 'E': 'E_med1_clase24_Petra', 'I': 'I_med1_clase24_Petra', 'O': 'O_med1_clase24_Petra', 'U': 'U_med1_clase24_Petra'}
                ]
            },
            {
                'musculo': 'Platisma (Cuello)',
                'funcion': 'Tensión cutánea cervical (Ag/AgCl med3)',
                'detalle': '08-27 (Ag/AgCl med3)',
                'col_csv': 'Canal 1',
                'col': '#0ea5e9',
                'fecha': '2026-08-27',
                'ses_list': [
                    {'A': 'A_med3_clase4_Petra', 'E': 'E_med3_clase4_Petra', 'I': 'I_med3_clase4_Petra', 'O': 'O_med3_clase4_Petra', 'U': 'U_med3_clase4_Petra'}
                ]
            },
            {
                'musculo': 'Cigomático (Silicona/IED)',
                'funcion': 'Elevación comisural lateral (Silicona med1)',
                'detalle': '08-28 (Silicona med1)',
                'col_csv': 'Canal 2',
                'col': '#10b981',
                'fecha': '2026-08-28',
                'ses_list': [
                    {'A': 'A_med1_clase24_Petra', 'E': 'E_med1_clase24_Petra', 'I': 'I_med1_clase24_Petra', 'O': 'O_med1_clase24_Petra', 'U': 'U_med1_clase24_Petra'}
                ]
            }
        ]
    }
]

class GeneradorAtlasPDF:
    """Motor backend para compilar el Atlas de Activación Muscular sEMG en formato PDF."""
    
    def __init__(self, base_db=None, output_dir=None):
        if base_db is None:
            possible_paths = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'base_de_datos_electrodos')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'EMG_desarrollo', 'base_de_datos_electrodos')),
                os.path.abspath('EMG_desarrollo/base_de_datos_electrodos')
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    self.base_db = p
                    break
            else:
                self.base_db = possible_paths[0]
        else:
            self.base_db = os.path.abspath(base_db)

        if output_dir is None:
            self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resultados'))
        else:
            self.output_dir = os.path.abspath(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        self.vocales = ['A', 'E', 'I', 'O', 'U']
        self.vocal_cols = {
            'A': '#dc2626',
            'E': '#84cc16',
            'I': '#16a34a',
            'O': '#f59e0b',
            'U': '#7c3aed'
        }

    def procesar_csv_toma(self, fecha, ses_name):
        """
        Lee directamente grabacion.csv de la toma.
        Aplica filtros estándar (Notch 50 Hz, Pasa-banda 20-450 Hz, RMS 80 ms).
        Detecta los picos acústicos en Canal 3 (Micrófono).
        Normaliza cada contracción por el Supremo Tricanal del Pulso Individual (M_supremo,pulso).
        """
        csv_path = os.path.join(self.base_db, fecha, ses_name, 'grabacion.csv')
        if not os.path.exists(csv_path):
            return None

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return None

        cols_requeridas = ['Tiempo (s)', 'Canal 0', 'Canal 1', 'Canal 2', 'Canal 3']
        if not all(c in df.columns for c in cols_requeridas):
            return None

        t_arr = df['Tiempo (s)'].values
        if len(t_arr) < 100:
            return None

        dt = np.median(np.diff(t_arr))
        if dt <= 0:
            return None
        fs = int(round(1.0 / dt))

        sig_m = df['Canal 3'].values.astype(np.float64)

        # Envolvente para detección de inicio acústico
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

        if len(picos) == 0:
            return None

        n_pre = int(PRE_MS * 1e-3 * fs)
        n_post = int(POST_MS * 1e-3 * fs)

        envs_musculares = {}
        for col in ['Canal 0', 'Canal 1', 'Canal 2']:
            sig_c = df[col].values.astype(np.float64)

            # Filtro Notch 50 Hz
            b_n, a_n = iirnotch(50.0, 2.0, fs)
            sig_c = filtfilt(b_n, a_n, sig_c)

            # Filtro pasa-banda 20-450 Hz
            b_b, a_b = butter(4, [20.0 / (fs/2), min(450.0, fs/2 - 1) / (fs/2)], btype='bandpass')
            sig_c = filtfilt(b_b, a_b, sig_c)

            # Envolvente RMS (80 ms)
            win = int(0.080 * fs)
            if win % 2 == 0: win += 1
            env_c = np.sqrt(np.convolve(sig_c**2, np.ones(win)/win, mode='same'))
            envs_musculares[col] = env_c

        pulsos_por_canal = {'Canal 0': [], 'Canal 1': [], 'Canal 2': []}

        for p in picos:
            valido = True
            v_canales = {}
            for col in ['Canal 0', 'Canal 1', 'Canal 2']:
                if p - n_pre >= 0 and p + n_post <= len(envs_musculares[col]):
                    v = envs_musculares[col][p - n_pre : p + n_post].copy()
                    piso = np.percentile(v[:int(0.15 * len(v))], 10)
                    v = np.maximum(0.0, v - piso)
                    v_canales[col] = v
                else:
                    valido = False
                    break

            if not valido or len(v_canales) == 0:
                continue

            # Supremo Tricanal individual del pulso
            m_supremo = max(np.max(v) for v in v_canales.values())
            if m_supremo < 1e-6:
                m_supremo = 1.0

            for col, v in v_canales.items():
                pulsos_por_canal[col].append(v / m_supremo)

        return pulsos_por_canal

    def extraer_curva_promedio_series(self, fecha, ses_list, vocal, col_target):
        """Acumula y promedia todos los pulsos normalizados de todas las series del día."""
        todos_pulsos = []
        for ses_dict in ses_list:
            ses_name = ses_dict.get(vocal)
            if not ses_name:
                continue
            res_toma = self.procesar_csv_toma(fecha, ses_name)
            if res_toma is not None and col_target in res_toma:
                todos_pulsos.extend(res_toma[col_target])

        if len(todos_pulsos) == 0:
            return None

        arr = np.array(todos_pulsos)
        mean_v = np.mean(arr, axis=0)
        std_v = np.std(arr, axis=0) / np.sqrt(len(arr))
        n_pts = len(mean_v)
        t_ms = np.linspace(-PRE_MS, POST_MS, n_pts)
        return {'t': t_ms, 'mean': mean_v, 'std': std_v, 'n_pulsos': len(arr)}

    def buscar_foto_registro(self, fecha, ses_list, reg=None):
        """Busca la fotografía fiduciaria de colocación de electrodos."""
        if reg and 'foto_override' in reg:
            p_override = reg['foto_override']
            if not os.path.isabs(p_override):
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                cand_paths = [
                    os.path.join(repo_root, p_override),
                    os.path.join(self.base_db, '..', p_override),
                    os.path.join(os.path.dirname(__file__), '..', p_override)
                ]
                for cp in cand_paths:
                    if os.path.exists(cp):
                        return cp
            elif os.path.exists(p_override):
                return p_override

        for ses_dict in ses_list:
            for v, sname in ses_dict.items():
                p = os.path.join(self.base_db, fecha, sname, 'photo.png')
                if os.path.exists(p):
                    return p
        p_direct = os.path.join(self.base_db, fecha, 'photo.png')
        if os.path.exists(p_direct):
            return p_direct
        return None

    def generar_atlas(self, ruta_pdf=None, incluir_foto=True, tema='publicacion',
                      sujetos_filtro=None, filas_por_pagina=2, progress_cb=None):
        """
        Compila el Atlas completo en PDF.
        Argumentos:
            ruta_pdf: Ruta de salida del archivo PDF.
            incluir_foto: Si es True, renderiza la fotografía lateral si está disponible.
            tema: 'publicacion' (fondo blanco) o 'oscuro' (fondo azul noche).
            sujetos_filtro: Lista de nombres de sujetos a incluir (ej: ['Candela', 'Lucas']).
            filas_por_pagina: Número de filas musculares por página (recomendado: 2 o 3).
            progress_cb: Función de callback(texto, porcentaje_entero).
        """
        if ruta_pdf is None:
            nombre_arch = f"Atlas_Activacion_sEMG_{tema}.pdf"
            ruta_pdf = os.path.join(self.output_dir, nombre_arch)
        else:
            ruta_pdf = os.path.abspath(ruta_pdf)

        # Filtrar catálogo por sujetos si se especificó
        if sujetos_filtro:
            sujetos_norm = [s.strip().lower() for s in sujetos_filtro]
            catalogo = [b for b in CATALOGO_SUJETOS if b['sujeto'].strip().lower() in sujetos_norm]
        else:
            catalogo = CATALOGO_SUJETOS

        filas_totales = sum(len(b['registros']) for b in catalogo)
        if filas_totales == 0:
            if progress_cb: progress_cb("No se encontraron registros para los sujetos seleccionados.", 0)
            return None

        # Configuración de estilos según tema
        if tema == 'oscuro':
            c_bg_page = '#0a0f1d'
            c_bg_axis = '#151f32'
            c_grid = '#2d3b55'
            c_txt_main = '#f8fafc'
            c_txt_muted = '#94a3b8'
            c_spines = '#3b4b66'
            c_header_bg = '#1e293b'
            c_badge_bg = '#0f172a'
        else:
            c_bg_page = '#ffffff'
            c_bg_axis = '#f8fafc'
            c_grid = '#e2e8f0'
            c_txt_main = '#0f172a'
            c_txt_muted = '#64748b'
            c_spines = '#cbd5e1'
            c_header_bg = '#f1f5f9'
            c_badge_bg = '#e2e8f0'

        # Extracción y cálculo de curvas para cada registro
        datos_procesados = []
        contador_prog = 0

        for b in catalogo:
            suj_nombre = b['sujeto']
            for reg in b['registros']:
                contador_prog += 1
                pct = int((contador_prog / filas_totales) * 60)
                msg = f"Extrayendo [{contador_prog}/{filas_totales}] {suj_nombre} - {reg['musculo']} ({reg['detalle']})"
                if progress_cb: progress_cb(msg, pct)
                print(msg)

                fila_vocales = {}
                for v in self.vocales:
                    c = self.extraer_curva_promedio_series(reg['fecha'], reg['ses_list'], v, reg['col_csv'])
                    fila_vocales[v] = c

                foto_path = self.buscar_foto_registro(reg['fecha'], reg['ses_list'], reg=reg) if incluir_foto else None

                datos_procesados.append({
                    'sujeto': suj_nombre,
                    'cfg': reg,
                    'curvas': fila_vocales,
                    'foto_path': foto_path
                })

        # Agrupar registros en páginas según filas_por_pagina
        paginas = []
        for b in catalogo:
            suj_nombre = b['sujeto']
            regs_sujeto = [item for item in datos_procesados if item['sujeto'] == suj_nombre]
            for i in range(0, len(regs_sujeto), filas_por_pagina):
                paginas.append({
                    'sujeto': suj_nombre,
                    'registros': regs_sujeto[i : i + filas_por_pagina]
                })

        total_paginas = len(paginas)
        if progress_cb: progress_cb("Generando documento PDF vectorial...", 65)

        fig_w = 14.0
        fig_h = 8.8

        with PdfPages(ruta_pdf) as pdf:
            for num_pag, pag in enumerate(paginas, start=1):
                pct = 65 + int((num_pag / total_paginas) * 33)
                msg = f"Renderizando página {num_pag}/{total_paginas} (Sujeto: {pag['sujeto']})"
                if progress_cb: progress_cb(msg, pct)
                print(msg)

                fig = plt.figure(figsize=(fig_w, fig_h), facecolor=c_bg_page)

                # Encabezado institucional superior
                ax_head = fig.add_axes([0.04, 0.90, 0.92, 0.08])
                ax_head.set_facecolor(c_header_bg)
                ax_head.set_xticks([])
                ax_head.set_yticks([])
                for sp in ax_head.spines.values():
                    sp.set_color(c_spines)

                ax_head.text(0.015, 0.65, "ATLAS DE ACTIVACIÓN MIOELÉCTRICA sEMG",
                            color=c_txt_main, fontsize=12, weight='bold', va='center')
                ax_head.text(0.015, 0.25, f"Sujeto: {pag['sujeto']}  |  Proyecto Ñandú (LSD - UBA)",
                            color=c_txt_muted, fontsize=9.5, weight='bold', va='center')

                ax_head.text(0.985, 0.65, f"Página {num_pag} de {total_paginas}",
                            color=c_txt_main, fontsize=10, weight='bold', ha='right', va='center')
                ax_head.text(0.985, 0.25, "Normalización: Supremo Tricanal por Pulso  |  Ventana: -600 ms a +800 ms",
                            color=c_txt_muted, fontsize=8.5, ha='right', va='center')

                regs_en_pag = pag['registros']
                n_regs = len(regs_en_pag)

                top_y = 0.88
                bottom_y = 0.05
                h_usable = top_y - bottom_y
                gap_y = 0.03
                h_fila = (h_usable - (n_regs - 1) * gap_y) / filas_por_pagina

                for r_idx, item in enumerate(regs_en_pag):
                    cfg = item['cfg']
                    curvas = item['curvas']
                    foto_path = item['foto_path']
                    color_musculo = cfg['col']

                    y_base = top_y - (r_idx + 1) * h_fila - r_idx * gap_y

                    if incluir_foto:
                        x_meta = 0.04
                        w_meta = 0.16
                        x_foto = 0.21
                        w_foto = 0.14
                        x_vocs_start = 0.36
                        w_voc_total = 0.60
                    else:
                        x_meta = 0.04
                        w_meta = 0.18
                        x_foto = None
                        w_foto = 0.0
                        x_vocs_start = 0.23
                        w_voc_total = 0.73

                    w_voc = (w_voc_total - 4 * 0.012) / 5.0

                    # 1. Panel de Metadatos
                    ax_meta = fig.add_axes([x_meta, y_base, w_meta, h_fila])
                    ax_meta.set_facecolor(c_badge_bg)
                    ax_meta.set_xticks([])
                    ax_meta.set_yticks([])
                    for sp in ax_meta.spines.values():
                        sp.set_color(c_spines)

                    func_wrapped = textwrap.fill(cfg['funcion'], width=22)
                    det_wrapped = textwrap.fill(cfg['detalle'], width=24)

                    ax_meta.text(0.06, 0.85, cfg['musculo'], color=color_musculo,
                                 fontsize=9.5, weight='bold', va='center')
                    ax_meta.text(0.06, 0.65, func_wrapped, color=c_txt_main,
                                 fontsize=7.5, va='center')
                    ax_meta.text(0.06, 0.42, f"Fecha: {cfg['fecha']}", color=c_txt_muted,
                                 fontsize=7.5, va='center')
                    ax_meta.text(0.06, 0.25, det_wrapped, color=c_txt_muted,
                                 fontsize=7.0, va='center')
                    ax_meta.text(0.06, 0.08, f"Canal sEMG: {cfg['col_csv']}", color=c_txt_main,
                                 fontsize=7.5, weight='bold', va='center')

                    # 2. Panel de Fotografía Fiduciaria
                    if incluir_foto:
                        ax_foto = fig.add_axes([x_foto, y_base, w_foto, h_fila])
                        ax_foto.set_facecolor(c_bg_axis)
                        ax_foto.set_xticks([])
                        ax_foto.set_yticks([])
                        for sp in ax_foto.spines.values():
                            sp.set_color(c_spines)

                        if foto_path and os.path.exists(foto_path):
                            try:
                                img = mpimg.imread(foto_path)
                                ax_foto.imshow(img)
                                ax_foto.set_title("Colocación de electrodos", color=c_txt_muted, fontsize=7.5, pad=3)
                            except Exception:
                                ax_foto.text(0.5, 0.5, "Error al cargar foto", color=c_txt_muted,
                                             fontsize=7.5, ha='center', va='center')
                        else:
                            ax_foto.text(0.5, 0.5, "Sin fotografía\nfiduciaria", color=c_txt_muted,
                                         fontsize=7.5, ha='center', va='center', multialignment='center')

                    # 3. Paneles de Vocales (/A/, /E/, /I/, /O/, /U/)
                    for v_idx, v in enumerate(self.vocales):
                        xv = x_vocs_start + v_idx * (w_voc + 0.012)
                        ax_v = fig.add_axes([xv, y_base, w_voc, h_fila])
                        ax_v.set_facecolor(c_bg_axis)
                        ax_v.grid(True, linestyle=':', alpha=0.4, color=c_grid)

                        # Línea de referencia acústica en t = 0 ms
                        ax_v.axvline(0, color='#f87171', linestyle='--', linewidth=1.1, alpha=0.85)

                        c = curvas.get(v)
                        if c is not None:
                            t = c['t']
                            y = c['mean']
                            y_err = c['std']

                            ax_v.plot(t, y, color=color_musculo, linewidth=1.8, alpha=0.95)
                            ax_v.fill_between(t, np.maximum(0, y - y_err), y + y_err,
                                              color=color_musculo, alpha=0.20)

                            if np.max(y) > 0.20:
                                idx_p = np.argmax(y)
                                t_p = t[idx_p]
                                val_p = y[idx_p]
                                ax_v.plot(t_p, val_p, marker='o', markersize=3.5, color=color_musculo)
                                ax_v.text(t_p, min(1.10, val_p + 0.05), f"{int(t_p)}ms",
                                          color=c_txt_main, fontsize=7, ha='center', va='bottom', weight='bold')
                        else:
                            ax_v.text(0, 0.5, "Sin dato", color=c_txt_muted, fontsize=7.5, ha='center', va='center')

                        ax_v.set_xlim(-PRE_MS, POST_MS)
                        ax_v.set_ylim(-0.05, 1.20)
                        ax_v.tick_params(colors=c_txt_muted, labelsize=7)
                        for sp in ax_v.spines.values():
                            sp.set_color(c_spines)

                        # Encabezado de columna de vocal solo en la primera fila de la página
                        if r_idx == 0:
                            ax_v.set_title(f"Vocal /{v}/", color=self.vocal_cols[v], fontsize=11, weight='bold', pad=4)

                        # Rótulo del eje X en la última fila
                        if r_idx == n_regs - 1:
                            ax_v.set_xlabel("Tiempo al mic [ms]", color=c_txt_muted, fontsize=7.5, labelpad=2)

                pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none', dpi=220)
                plt.close(fig)

        if progress_cb: progress_cb(f"Atlas PDF generado exitosamente: {ruta_pdf}", 100)
        print(f"\n[Atlas PDF Exitoso] Guardado en: {ruta_pdf}")
        return ruta_pdf

def main():
    parser = argparse.ArgumentParser(description="Generador del Atlas Oficial sEMG en formato PDF")
    parser.add_argument("--output", "-o", default=None, help="Ruta de salida del archivo PDF")
    parser.add_argument("--sin-foto", action="store_true", help="Desactiva la inclusión de fotos laterales")
    parser.add_argument("--tema", choices=['publicacion', 'oscuro'], default='publicacion',
                        help="Tema visual: 'publicacion' (blanco) u 'oscuro' (cyberpunk)")
    parser.add_argument("--filas", type=int, default=2, help="Número de filas musculares por página (default: 2)")
    parser.add_argument("--sujetos", nargs="+", default=None, help="Filtrar por nombres de sujetos")
    args = parser.parse_args()

    generador = GeneradorAtlasPDF()
    generador.generar_atlas(
        ruta_pdf=args.output,
        incluir_foto=not args.sin_foto,
        tema=args.tema,
        sujetos_filtro=args.sujetos,
        filas_por_pagina=args.filas
    )

if __name__ == '__main__':
    main()
