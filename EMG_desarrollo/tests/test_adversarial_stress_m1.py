# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Pruebas de estrés y casos límite adversariales para R1 y R3.
# ==============================================================================

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EMG_DESARROLLO = os.path.abspath(os.path.join(TESTS_DIR, '..'))
if EMG_DESARROLLO not in sys.path:
    sys.path.insert(0, EMG_DESARROLLO)
PCA_UMAP_DIR = os.path.join(EMG_DESARROLLO, 'deep_learning', 'pca_umap_clustering')
if PCA_UMAP_DIR not in sys.path:
    sys.path.insert(0, PCA_UMAP_DIR)
DL_DIR = os.path.join(EMG_DESARROLLO, 'deep_learning')
if DL_DIR not in sys.path:
    sys.path.insert(0, DL_DIR)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QImage

import generador_pca_umap as gpu
import pca_analysis as pa
from gui_app.main_app import ReaperStyleHub, ZoomableImageWidget


class TestAdversarialStressM1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['stress_harness'])

    def test_stress_gallery_corrupted_and_unusual_files(self):
        """Stress test gallery loading against corrupted CSVs, malformed JSON, invalid UTF-8 TXT, and 0-byte images."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            hub = ReaperStyleHub()
            hub.cmb_resultados.clear()

            # 1. 0-byte PNG file
            p_zero_png = os.path.join(tmp_dir, "zero.png")
            with open(p_zero_png, 'wb') as f: pass
            hub.cmb_resultados.addItem("zero.png", p_zero_png)
            hub._cargar_imagen_visor(0)
            # Should not crash; pixmap will be null
            self.assertTrue(hub.img_viewer._pixmap is None or hub.img_viewer._pixmap.isNull())

            # 2. Corrupted CSV file
            p_bad_csv = os.path.join(tmp_dir, "corrupt.csv")
            with open(p_bad_csv, 'w') as f:
                f.write("A,B,C\n1,2\n3,4,5,6,7,8\n\"unclosed quote")
            hub.cmb_resultados.addItem("corrupt.csv", p_bad_csv)
            hub._cargar_imagen_visor(1)
            # Should fall back to txt_metricas_visor error message without crash
            self.assertFalse(hub.txt_metricas_visor.isHidden())

            # 3. Malformed JSON file
            p_bad_json = os.path.join(tmp_dir, "bad.json")
            with open(p_bad_json, 'w') as f:
                f.write("{ invalid_json: [1, 2, }")
            hub.cmb_resultados.addItem("bad.json", p_bad_json)
            hub._cargar_imagen_visor(2)
            self.assertIn("invalid_json", hub.txt_metricas_visor.toPlainText())

            # 4. Invalid UTF-8 text file (raw binary)
            p_bin_txt = os.path.join(tmp_dir, "binary.txt")
            with open(p_bin_txt, 'wb') as f:
                f.write(b"\xFF\xFE\xFD\x80\x81\x82 Non UTF-8 bytes")
            hub.cmb_resultados.addItem("binary.txt", p_bin_txt)
            hub._cargar_imagen_visor(3)
            # errors='replace' must handle without raising UnicodeDecodeError
            self.assertTrue(len(hub.txt_metricas_visor.toPlainText()) > 0)

    def test_stress_3d_pca_extreme_geometry(self):
        """Stress test 3D scatter plots with planar data (0-variance in Z), extreme coordinates, and single point."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Case 1: Planar data in 3D (z = 0 for all points)
            X_flat = np.array([
                [1.0, 2.0, 0.0],
                [-1.0, 2.0, 0.0],
                [0.0, -2.0, 0.0],
                [2.0, -1.0, 0.0]
            ])
            Y_flat = np.array(['A', 'A', 'E', 'E'])
            out_flat = os.path.join(tmp_dir, "flat_3d.png")
            gpu.plot_scatter(X_flat, Y_flat, "Planar 3D Scatter", out_flat, is_3d=True)
            self.assertTrue(os.path.exists(out_flat))

            # Case 2: Extreme magnitude coordinates (+/- 1e6)
            X_extreme = np.array([
                [1e6, 2e6, -1e6],
                [1.1e6, 1.9e6, -1.1e6],
                [-2e6, -3e6, 4e6],
                [-2.1e6, -2.9e6, 4.1e6]
            ])
            Y_extreme = np.array(['A', 'A', 'E', 'E'])
            out_extreme = os.path.join(tmp_dir, "extreme_3d.png")
            gpu.plot_scatter(X_extreme, Y_extreme, "Extreme Coords", out_extreme, is_3d=True)
            self.assertTrue(os.path.exists(out_extreme))

            # Case 3: Single point cluster in plot_analisis_errores_3d
            X_single = np.array([
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [-1.0, -1.0, -1.0],
                [-2.0, -2.0, -2.0]
            ])
            Y_single = np.array(['A', 'A', 'E', 'E'])
            Tomas_single = ['T1', 'T2', 'T3', 'T4']
            out_single = os.path.join(tmp_dir, "single_clust.png")
            gpu.plot_analisis_errores_3d(X_single, Y_single, Tomas_single, "Small Cluster", out_single)
            self.assertTrue(os.path.exists(out_single))

    def test_stress_table_with_special_characters_and_large_dimensions(self):
        """Stress test table image generator with Matplotlib special chars, math symbols, and large shape."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Special chars: $, %, _, &, #, <, >, /, \
            cols = [
                r"Ratio_SNR_>_15dB_&_Q=50",
                r"Mean_Error_%_[\sigma=3]",
                r"Param_\alpha_Ruido_#1"
            ]
            rows = [
                r"Condición_#1_Vocal_A",
                r"Condición_#2_Vocal_E_&_I",
                r"Condición_#3_Vocal_O_/_U"
            ]
            data = [
                [12.345, 0.012, 0.999],
                [45.678, 0.054, 0.888],
                [78.910, 0.120, 0.777]
            ]
            df_special = pd.DataFrame(data, index=rows, columns=cols)
            out_special = os.path.join(tmp_dir, "table_special_chars.png")
            gpu.guardar_tabla_imagen(df_special, "Tabla con Caracteres Especiales", out_special)
            self.assertTrue(os.path.exists(out_special))
            self.assertGreater(os.path.getsize(out_special), 15000)

            # Large matrix (15 rows x 8 columns)
            df_large = pd.DataFrame(
                np.random.rand(15, 8),
                index=[f"Medicion_Registro_Sesion_{i+1:02d}" for i in range(15)],
                columns=[f"Feature_Canal_{c+1}" for c in range(8)]
            )
            out_large = os.path.join(tmp_dir, "table_large.png")
            gpu.guardar_tabla_imagen(df_large, "Tabla Dimensional Grande (15x8)", out_large)
            self.assertTrue(os.path.exists(out_large))
            with Image.open(out_large) as img:
                self.assertGreaterEqual(img.width, 2000)
                self.assertGreaterEqual(img.height, 1500)

    def test_stress_confusion_matrix_edge_cases(self):
        """Stress test confusion matrix heatmap with single class, identical predictions, and zero matrix."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # All zeros
            df_zero = pd.DataFrame(
                np.zeros((5, 5)),
                index=['A', 'E', 'I', 'O', 'U'],
                columns=['A', 'E', 'I', 'O', 'U']
            )
            out_zero = os.path.join(tmp_dir, "cm_zero.png")
            gpu.plot_confusion_matrix_heatmap(df_zero, "Confusion Matrix - All Zeros", out_zero)
            self.assertTrue(os.path.exists(out_zero))

            # 100% perfect classification
            df_perfect = pd.DataFrame(
                np.diag([100.0] * 5),
                index=['A', 'E', 'I', 'O', 'U'],
                columns=['A', 'E', 'I', 'O', 'U']
            )
            out_perfect = os.path.join(tmp_dir, "cm_perfect.png")
            gpu.plot_confusion_matrix_heatmap(df_perfect, "Confusion Matrix - 100% Accuracy", out_perfect)
            self.assertTrue(os.path.exists(out_perfect))


if __name__ == '__main__':
    unittest.main()
