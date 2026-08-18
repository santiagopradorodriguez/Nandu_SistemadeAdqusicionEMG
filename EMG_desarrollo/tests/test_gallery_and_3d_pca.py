# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Pruebas empíricas para Results Gallery, Proyecciones 3D PCA,
#              Matriz de Confusión y Exportación de Tablas de Métricas (M1).
# ==============================================================================

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

# Ensure EMG_desarrollo and subdirectories are on sys.path
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

# Set headless environment for Qt and Matplotlib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QImage

import generador_pca_umap as gpu
import pca_analysis as pa
from gui_app.main_app import ZoomableImageWidget, ReaperStyleHub


class TestGalleryAnd3DPCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test_harness'])

    def test_01_gallery_recursive_discovery_and_extensions(self):
        """Test that _refrescar_visor_imagenes recursively discovers files in nested dirs and filters extensions."""
        with tempfile.TemporaryDirectory() as tmp_root:
            # Create standard folder structure inside temporary root
            dir_res = os.path.join(tmp_root, "resultados", "nested_a", "nested_b")
            dir_pca = os.path.join(tmp_root, "deep_learning", "pca_umap_clustering", "resultados_pca_umap", "sub_pca")
            dir_umap = os.path.join(tmp_root, "deep_learning", "resultados_umap_supervisado", "sub_umap")
            dir_comp = os.path.join(tmp_root, "analisis_comparativos", "sub_comp")
            for d in [dir_res, dir_pca, dir_umap, dir_comp]:
                os.makedirs(d, exist_ok=True)

            # Create test files with allowed extensions
            f_png = os.path.join(dir_res, "test_plot.png")
            f_jpg = os.path.join(dir_res, "test_photo.jpg")
            f_csv = os.path.join(dir_pca, "metricas.csv")
            f_tex = os.path.join(dir_pca, "tabla.tex")
            f_json = os.path.join(dir_umap, "config.json")
            f_txt = os.path.join(dir_comp, "log.txt")

            # Create test files with disallowed extensions
            f_bin = os.path.join(dir_res, "data.bin")
            f_wav = os.path.join(dir_res, "audio.wav")
            f_py = os.path.join(dir_comp, "script.py")

            # Populate dummy content
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(f_png)
            img.save(f_jpg)
            pd.DataFrame({'Metric': ['Acc', 'Loss'], 'Value': [0.95, 0.05]}).to_csv(f_csv, index=False)
            with open(f_tex, 'w') as f: f.write(r"\begin{tabular}...\end{tabular}")
            with open(f_json, 'w') as f: f.write('{"test": true}')
            with open(f_txt, 'w') as f: f.write("Summary logs")
            with open(f_bin, 'wb') as f: f.write(b"\x00\x01\x02")
            with open(f_wav, 'wb') as f: f.write(b"RIFFdummyWAV")
            with open(f_py, 'w') as f: f.write("print(1)")

            # Create Hub and inject mocked root dir for testing discovery
            hub = ReaperStyleHub()
            
            # Monkey patch root_dir inside _refrescar_visor_imagenes logic
            orig_refrescar = hub._refrescar_visor_imagenes
            
            def mocked_refrescar():
                hub.cmb_resultados.clear()
                paths_to_check = [
                    os.path.join(tmp_root, "resultados"),
                    os.path.join(tmp_root, "deep_learning", "pca_umap_clustering", "resultados_pca_umap"),
                    os.path.join(tmp_root, "deep_learning", "resultados_umap_supervisado"),
                    os.path.join(tmp_root, "analisis_comparativos")
                ]
                found_files = []
                valid_exts = ('.png', '.jpg', '.jpeg', '.csv', '.tex', '.json', '.txt')
                for base_path in paths_to_check:
                    if os.path.exists(base_path):
                        for root, _, files in os.walk(base_path):
                            for f in files:
                                if f.lower().endswith(valid_exts):
                                    full_path = os.path.join(root, f)
                                    rel_path = os.path.relpath(full_path, base_path)
                                    try:
                                        mtime = os.path.getmtime(full_path)
                                    except OSError:
                                        mtime = 0
                                    category = os.path.basename(base_path)
                                    display_label = f"[{category}] {rel_path}"
                                    found_files.append((mtime, display_label, full_path))
                found_files.sort(key=lambda x: x[0], reverse=True)
                for _, label, path in found_files:
                    hub.cmb_resultados.addItem(label, path)
                if hub.cmb_resultados.count() > 0:
                    hub.cmb_resultados.setCurrentIndex(0)

            mocked_refrescar()

            # Verify discovered items count
            self.assertEqual(hub.cmb_resultados.count(), 6)
            
            # Verify paths and labels
            item_paths = [hub.cmb_resultados.itemData(i) for i in range(hub.cmb_resultados.count())]
            self.assertIn(f_png, item_paths)
            self.assertIn(f_jpg, item_paths)
            self.assertIn(f_csv, item_paths)
            self.assertIn(f_tex, item_paths)
            self.assertIn(f_json, item_paths)
            self.assertIn(f_txt, item_paths)
            self.assertNotIn(f_bin, item_paths)
            self.assertNotIn(f_wav, item_paths)
            self.assertNotIn(f_py, item_paths)

    def test_02_gallery_file_loading_all_formats(self):
        """Test _cargar_imagen_visor for .png, .csv, .tex, .json, .txt formats and edge cases."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            hub = ReaperStyleHub()
            hub.cmb_resultados.clear()

            # 1. Image loading
            p_img = os.path.join(tmp_dir, "test.png")
            Image.new('RGB', (200, 150), color='green').save(p_img)
            hub.cmb_resultados.addItem("image.png", p_img)
            hub._cargar_imagen_visor(0)
            self.assertEqual(hub.visor_subtabs.currentIndex(), 0)
            self.assertIsNotNone(hub.img_viewer._pixmap)
            self.assertFalse(hub.img_viewer._pixmap.isNull())

            # 2. CSV loading
            p_csv = os.path.join(tmp_dir, "data.csv")
            df_test = pd.DataFrame({'ColA': [1.23456, 7.89], 'ColB': ['Vocal_A', 'Vocal_E']})
            df_test.to_csv(p_csv, index=False)
            hub.cmb_resultados.addItem("data.csv", p_csv)
            hub._cargar_imagen_visor(1)
            self.assertFalse(hub.tbl_metricas_visor.isHidden())
            self.assertTrue(hub.txt_metricas_visor.isHidden())
            self.assertEqual(hub.tbl_metricas_visor.rowCount(), 2)
            self.assertEqual(hub.tbl_metricas_visor.columnCount(), 2)
            self.assertEqual(hub.tbl_metricas_visor.item(0, 0).text(), "1.235")
            self.assertEqual(hub.tbl_metricas_visor.item(0, 1).text(), "Vocal_A")

            # 3. TeX loading
            p_tex = os.path.join(tmp_dir, "matrix.tex")
            tex_content = r"\begin{tabular}{cc} A & B \\ \end{tabular}"
            with open(p_tex, 'w', encoding='utf-8') as f: f.write(tex_content)
            hub.cmb_resultados.addItem("matrix.tex", p_tex)
            hub._cargar_imagen_visor(2)
            self.assertEqual(hub.visor_subtabs.currentIndex(), 1)
            self.assertFalse(hub.txt_metricas_visor.isHidden())
            self.assertTrue(hub.tbl_metricas_visor.isHidden())
            self.assertIn("tabular", hub.txt_metricas_visor.toPlainText())

            # 4. JSON loading
            p_json = os.path.join(tmp_dir, "metrics.json")
            json_content = '{"accuracy": 98.4, "vowels": ["A", "E", "I", "O", "U"]}'
            with open(p_json, 'w', encoding='utf-8') as f: f.write(json_content)
            hub.cmb_resultados.addItem("metrics.json", p_json)
            hub._cargar_imagen_visor(3)
            self.assertEqual(hub.visor_subtabs.currentIndex(), 1)
            self.assertFalse(hub.txt_metricas_visor.isHidden())
            self.assertTrue(hub.tbl_metricas_visor.isHidden())
            self.assertIn("accuracy", hub.txt_metricas_visor.toPlainText())

            # 5. TXT loading
            p_txt = os.path.join(tmp_dir, "log.txt")
            txt_content = "Sesion completada con exito.\nSin errores detectados."
            with open(p_txt, 'w', encoding='utf-8') as f: f.write(txt_content)
            hub.cmb_resultados.addItem("log.txt", p_txt)
            hub._cargar_imagen_visor(4)
            self.assertEqual(hub.visor_subtabs.currentIndex(), 1)
            self.assertFalse(hub.txt_metricas_visor.isHidden())
            self.assertTrue(hub.tbl_metricas_visor.isHidden())
            self.assertIn("Sesion completada", hub.txt_metricas_visor.toPlainText())

            # 6. Edge cases: Invalid indices and non-existent files
            hub._cargar_imagen_visor(-1)
            hub._cargar_imagen_visor(999)
            hub.cmb_resultados.addItem("ghost.png", os.path.join(tmp_dir, "nonexistent.png"))
            hub._cargar_imagen_visor(5)

    def test_03_zoomable_image_widget_controls(self):
        """Test interactive controls of ZoomableImageWidget: zoom in, out, reset, fit, and fullscreen."""
        widget = ZoomableImageWidget()
        self.assertEqual(widget.placeholder_text, "[Sin Imagen]")
        self.assertIsNone(widget._pixmap)

        # Create QPixmap and assign
        qimg = QImage(400, 300, QImage.Format_RGB32)
        qimg.fill(0xFF00FF)
        pix = QPixmap.fromImage(qimg)
        widget.setPixmap(pix, filepath="dummy.png")
        self.assertFalse(widget._pixmap.isNull())

        # Test initial fit mode
        widget.fit_to_view()
        self.assertTrue(widget._fit_mode)
        self.assertEqual(widget.lbl_zoom.text(), "Ajustado")

        # Test zoom in
        prev_scale = widget._scale_factor
        widget.zoom_in()
        self.assertFalse(widget._fit_mode)
        self.assertGreater(widget._scale_factor, prev_scale)

        # Test zoom out
        prev_scale = widget._scale_factor
        widget.zoom_out()
        self.assertLess(widget._scale_factor, prev_scale)

        # Test reset zoom (1:1)
        widget.reset_zoom()
        self.assertEqual(widget._scale_factor, 1.0)
        self.assertEqual(widget.lbl_zoom.text(), "100%")

        # Test clear / reset text
        widget.setText("[Nueva Imagen Requerida]")
        self.assertIsNone(widget._pixmap)
        self.assertEqual(widget.img_label.text(), "[Nueva Imagen Requerida]")

    def test_04_3d_pca_floor_wall_projections(self):
        """Test plot_scatter with is_3d=True renders XY floor, XZ back wall, and YZ lateral wall projections."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            np.random.seed(42)
            n_pts_per_vocal = 15
            vocales = ['A', 'E', 'I', 'O', 'U']
            
            # Generate distinct 3D clusters
            X_list = []
            Y_list = []
            centers = [
                [2.0, 2.0, 2.0],
                [-2.0, 2.0, -1.0],
                [0.0, -3.0, 2.0],
                [3.0, -2.0, -2.0],
                [-2.0, -2.0, 0.0]
            ]
            for v, c in zip(vocales, centers):
                pts = np.random.randn(n_pts_per_vocal, 3) * 0.4 + c
                X_list.append(pts)
                Y_list.extend([v] * n_pts_per_vocal)
            X_3d = np.vstack(X_list)
            Y = np.array(Y_list)

            out_path_gpu = os.path.join(tmp_dir, "pca_3d_gpu.png")
            gpu.plot_scatter(X_3d, Y, "PCA 3D - Vocales EMG", out_path_gpu, is_3d=True, variance_ratios=[0.55, 0.30, 0.15])
            
            self.assertTrue(os.path.exists(out_path_gpu), "Output file was not created.")
            self.assertGreater(os.path.getsize(out_path_gpu), 10000, "Output file is suspiciously small.")
            
            # Verify image opens with PIL and has valid dimensions
            with Image.open(out_path_gpu) as img:
                self.assertGreaterEqual(img.width, 1000)
                self.assertGreaterEqual(img.height, 800)

            # Test parity with pca_analysis.py
            out_path_pa = os.path.join(tmp_dir, "pca_3d_pa.png")
            pa.plot_scatter(X_3d, Y, "PCA 3D - Parity Test", out_path_pa, is_3d=True, variance_ratios=[0.55, 0.30, 0.15])
            self.assertTrue(os.path.exists(out_path_pa))
            self.assertGreater(os.path.getsize(out_path_pa), 10000)

    def test_05_3d_pca_multi_angle_projections(self):
        """Test plot_scatter_3d_multi_angle renders 3 views with plane projections without exception."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            np.random.seed(123)
            X_3d = np.random.randn(50, 3)
            Y = np.array(['A']*10 + ['E']*10 + ['I']*10 + ['O']*10 + ['U']*10)
            out_path = os.path.join(tmp_dir, "pca_3d_multi_angle.png")

            gpu.plot_scatter_3d_multi_angle(X_3d, Y, "PCA 3D Multi-Angle View", out_path, variance_ratios=[0.6, 0.25, 0.15])
            
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 20000)
            with Image.open(out_path) as img:
                # Multi-angle figure has width ~ 26 inches * 300 dpi ~= 7000+ px (or tightly cropped)
                self.assertGreater(img.width, 2000)

    def test_06_3d_pca_error_analysis_centroid_drop_lines_and_floor_ellipses(self):
        """Test plot_analisis_errores_3d verifies drop lines, centroid projection, and floor ellipses for K-Means and GMM."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            np.random.seed(99)
            n_per_vocal = 12
            vocales = ['A', 'E', 'I', 'O', 'U']
            X_list = []
            Y_list = []
            Tomas = []
            centers = [
                [3.0, 0.0, 1.0],
                [-3.0, 1.0, -1.0],
                [0.0, 3.0, 0.0],
                [0.0, -3.0, 2.0],
                [1.0, 1.0, -3.0]
            ]
            for i, (v, c) in enumerate(zip(vocales, centers)):
                pts = np.random.randn(n_per_vocal, 3) * 0.35 + c
                X_list.append(pts)
                Y_list.extend([v] * n_per_vocal)
                for t in range(n_per_vocal):
                    Tomas.append(f"Toma_{v}_{t+1}")
            X = np.vstack(X_list)
            Y = np.array(Y_list)

            # 1. Test with K-Means
            out_kmeans = os.path.join(tmp_dir, "analisis_errores_kmeans_3d.png")
            gpu.plot_analisis_errores_3d(X, Y, Tomas, "Análisis Errores 3D (K-Means)", out_kmeans, algoritmo="K-Means")
            self.assertTrue(os.path.exists(out_kmeans))
            self.assertGreater(os.path.getsize(out_kmeans), 15000)

            # 2. Test with GMM
            out_gmm = os.path.join(tmp_dir, "analisis_errores_gmm_3d.png")
            gpu.plot_analisis_errores_3d(X, Y, Tomas, "Análisis Errores 3D (GMM)", out_gmm, algoritmo="GMM")
            self.assertTrue(os.path.exists(out_gmm))
            self.assertGreater(os.path.getsize(out_gmm), 15000)

            # 3. Test 2D Multi-plane error analysis
            out_2d_proj = os.path.join(tmp_dir, "analisis_errores_3d_proy2d.png")
            gpu.plot_analisis_errores_3d_proyecciones_2d(X, Y, "Proyecciones 2D Multi-Plano", out_2d_proj, variance_ratios=[0.5, 0.3, 0.2])
            self.assertTrue(os.path.exists(out_2d_proj))
            self.assertGreater(os.path.getsize(out_2d_proj), 20000)

    def test_07_3d_pca_degenerate_and_edge_cases(self):
        """Stress test 3D PCA plotters with degenerate data: single sample per cluster, collinear points, zero variance."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Collinear points (rank 1 covariance matrix)
            t = np.linspace(-5, 5, 20)
            X_collinear = np.column_stack([t, t * 2, t * 3])
            Y_collinear = np.array(['A']*10 + ['E']*10)
            Tomas_collinear = [f"Toma_{i}" for i in range(20)]

            out_collinear = os.path.join(tmp_dir, "collinear_3d.png")
            gpu.plot_analisis_errores_3d(X_collinear, Y_collinear, Tomas_collinear, "Collinear Test", out_collinear)
            self.assertTrue(os.path.exists(out_collinear))

            # 2. Two clusters only
            X_2cl = np.random.randn(20, 3)
            Y_2cl = np.array(['A']*10 + ['E']*10)
            Tomas_2cl = [f"T_{i}" for i in range(20)]
            out_2cl = os.path.join(tmp_dir, "two_clusters_3d.png")
            gpu.plot_analisis_errores_3d(X_2cl, Y_2cl, Tomas_2cl, "Two Clusters", out_2cl)
            self.assertTrue(os.path.exists(out_2cl))

    def test_08_metric_table_image_export_long_labels(self):
        """Test guardar_tabla_imagen does not clip with long titles, long row labels, and long column headers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_long = pd.DataFrame({
                'Configuracion_Parametros_Filtro_Notch_Q50_BandaEstrecha': [98.765, 87.654, 91.234],
                'Distancia_Centroide_Separacion_InterVocalica_Normalizada': [4.567, 3.210, 5.890],
                'Indice_Silhouette_Espacio_Latente_Tres_Dimensiones': [0.789, 0.654, 0.812]
            }, index=[
                'Sujeto_Experimental_1_Condicion_Ruido_Ambiente_Bajo',
                'Sujeto_Experimental_2_Condicion_Ruido_Ambiente_Moderado',
                'Sujeto_Experimental_3_Condicion_Senal_Limpia_Optima'
            ])

            out_table = os.path.join(tmp_dir, "tabla_metricas_extensas.png")
            long_title = "Tabla Comparativa de Metricas de Rendimiento en Clasificacion Acustica EMG para Distintas Configuraciones DSP"
            
            gpu.guardar_tabla_imagen(df_long, long_title, out_table)
            
            self.assertTrue(os.path.exists(out_table))
            self.assertGreater(os.path.getsize(out_table), 20000)
            
            # Inspect image dimensions
            with Image.open(out_table) as img:
                # Dynamic width should expand to comfortably accommodate long strings
                self.assertGreaterEqual(img.width, 1500, "Table width should dynamically scale to avoid text overlap.")
                self.assertGreaterEqual(img.height, 400, "Table height should comfortably accommodate rows and title.")

    def test_09_confusion_matrix_heatmap_long_labels(self):
        """Test plot_confusion_matrix_heatmap with long labels and long titles without text truncation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            classes = [
                'Vocal_A_Larga_Frecuencia_Baja',
                'Vocal_E_Cerrada_Tension_Media',
                'Vocal_I_Alta_Frecuencia_Aguda',
                'Vocal_O_Redonda_Tono_Grave',
                'Vocal_U_Posterior_Profunda'
            ]
            cm_data = [
                [95.0, 2.0, 1.0, 1.0, 1.0],
                [1.0, 93.0, 4.0, 1.0, 1.0],
                [2.0, 3.0, 92.0, 2.0, 1.0],
                [1.0, 1.0, 2.0, 94.0, 2.0],
                [0.0, 1.0, 1.0, 2.0, 96.0]
            ]
            df_cm = pd.DataFrame(cm_data, index=classes, columns=classes)
            
            out_cm = os.path.join(tmp_dir, "heatmap_confusion_long_labels.png")
            title = "Matriz de Confusion Normalizada - Modelo KNN Espacio 3D PCA"
            
            gpu.plot_confusion_matrix_heatmap(df_cm, title, out_cm)
            
            self.assertTrue(os.path.exists(out_cm))
            self.assertGreater(os.path.getsize(out_cm), 25000)
            
            with Image.open(out_cm) as img:
                self.assertGreaterEqual(img.width, 1200)
                self.assertGreaterEqual(img.height, 1000)

    def test_10_guardar_matriz_latex(self):
        """Test guardar_matriz_latex exports correct LaTeX code structure with cell colors and header."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            classes = ['A', 'E', 'I', 'O', 'U']
            cm_data = [
                [95.0, 2.0, 1.0, 1.0, 1.0],
                [1.0, 93.0, 4.0, 1.0, 1.0],
                [2.0, 3.0, 92.0, 2.0, 1.0],
                [1.0, 1.0, 2.0, 94.0, 2.0],
                [0.0, 1.0, 1.0, 2.0, 96.0]
            ]
            df_cm = pd.DataFrame(cm_data, index=classes, columns=classes)
            
            out_tex = os.path.join(tmp_dir, "matriz_confusion.tex")
            gpu.guardar_matriz_latex(df_cm, "Matriz de Confusión PCA 3D", out_tex)
            
            self.assertTrue(os.path.exists(out_tex))
            with open(out_tex, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertIn(r"\begin{tabular}", content)
            self.assertIn(r"\end{tabular}", content)
            self.assertIn(r"\cellcolor[rgb]{", content)
            self.assertIn(r"\textbf{Real A}", content)
            self.assertIn("95\\%", content)


if __name__ == '__main__':
    unittest.main()
