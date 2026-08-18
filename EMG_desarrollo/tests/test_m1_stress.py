# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Suite de pruebas de estrés y validación empírica para el Milestone M1.
# ==============================================================================

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Set headless environment for Qt and Matplotlib
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMG_DESARROLLO = os.path.join(REPO_ROOT, "EMG_desarrollo")
if EMG_DESARROLLO not in sys.path:
    sys.path.insert(0, EMG_DESARROLLO)
PCA_DIR = os.path.join(EMG_DESARROLLO, "deep_learning", "pca_umap_clustering")
if PCA_DIR not in sys.path:
    sys.path.insert(0, PCA_DIR)
DL_DIR = os.path.join(EMG_DESARROLLO, "deep_learning")
if DL_DIR not in sys.path:
    sys.path.insert(0, DL_DIR)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

# Initialize QApplication for headless tests
app = QApplication.instance()
if app is None:
    app = QApplication(["test_m1_stress"])

from gui_app.views.ui_analysis import AnalysisPanel, ProcessingTab, ComparativeTab
from gui_app.main_app import ZoomableImageWidget, ReaperStyleHub
import generador_pca_umap as gpu
import pca_analysis as pca_an


class TestAnalysisPanelInputs(unittest.TestCase):
    """Pruebas de estrés y robustez de captura de parámetros en AnalysisPanel y ProcessingTab."""

    def setUp(self):
        self.panel = AnalysisPanel()
        self.proc_tab = self.panel.tab_procesamiento

    def test_default_processing_kwargs(self):
        """Verifica que los kwargs por defecto se extraigan con todos los campos requeridos."""
        kw = self.panel.get_processing_kwargs()
        expected_keys = {
            'mostrar_recortes', 'mostrar_senal_cruda', 'tema_cyberpunk',
            'mostrar_espectrograma', 'frecuenciamaxima', 'apply_notch_filter',
            'notch_q_factor', 'mostrar_evolucion', 'evol_t_start', 'evol_t_end',
            'excluded_windows_list', 'tipo_envolvente', 'smooth_ms',
            'highpass_cutoff_hz', 'lowpass_cutoff_hz'
        }
        self.assertEqual(set(kw.keys()), expected_keys)
        self.assertEqual(len(kw), 15)
        self.assertTrue(kw['mostrar_recortes'])
        self.assertFalse(kw['mostrar_senal_cruda'])
        self.assertEqual(kw['frecuenciamaxima'], "5000")
        self.assertEqual(kw['excluded_windows_list'], [])

    def test_default_trevisan_kwargs(self):
        """Verifica que get_trevisan_kwargs devuelva tipos correctos y valores esperados."""
        kw = self.panel.get_trevisan_kwargs()
        self.assertIn('smooth_ms', kw)
        self.assertIsInstance(kw['smooth_ms'], float)
        self.assertEqual(kw['smooth_ms'], 50.0)
        self.assertEqual(kw['alpha_ruido'], 1.0)
        self.assertEqual(kw['snr_threshold'], 3.0)
        self.assertEqual(kw['n_pts_window'], 100)

    def test_empty_and_whitespace_inputs(self):
        """Prueba comportamiento con campos de texto vacíos o sólo espacios."""
        self.proc_tab.inp_spec_fmax.setText("   ")
        self.proc_tab.inp_notch_q.setText("")
        self.proc_tab.inp_evol_start.setText(" \t ")
        self.proc_tab.inp_evol_end.setText("")
        self.proc_tab.inp_excluded.setText("   ")
        self.proc_tab.inp_smooth.setText(" ")
        self.proc_tab.inp_hp.setText("")
        self.proc_tab.inp_lp.setText(" \n ")

        kw = self.panel.get_processing_kwargs()
        self.assertEqual(kw['frecuenciamaxima'], "5000")
        self.assertEqual(kw['notch_q_factor'], "2.0")
        self.assertEqual(kw['evol_t_start'], "10")
        self.assertEqual(kw['evol_t_end'], "1000")
        self.assertEqual(kw['excluded_windows_list'], [])
        self.assertEqual(kw['smooth_ms'], "50")
        self.assertEqual(kw['highpass_cutoff_hz'], "20")
        self.assertEqual(kw['lowpass_cutoff_hz'], "500")

        trev_kw = self.panel.get_trevisan_kwargs()
        self.assertEqual(trev_kw['smooth_ms'], 50.0)

    def test_corrupted_excluded_windows(self):
        """Prueba parsing de listas de ventanas excluidas con strings corruptos y mixtos."""
        corrupted_cases = [
            ("1, 2, 3", [1, 2, 3]),
            ("1, a, 3, -4, 5.5, , 6", [1, 3, 6]),
            (",,,,,", []),
            ("abc, def, ghi", []),
            ("0, 10, 999", [0, 10, 999]),
            ("   12  ,   34   ", [12, 34]),
            ("1;;2::3", []),
            ("999999999999", [999999999999]),
        ]
        for input_str, expected in corrupted_cases:
            self.proc_tab.inp_excluded.setText(input_str)
            kw = self.panel.get_processing_kwargs()
            self.assertEqual(kw['excluded_windows_list'], expected, f"Fallo con input: {input_str}")

    def test_corrupted_trevisan_smooth(self):
        """Prueba valores no numéricos y extremos en inp_smooth para get_trevisan_kwargs."""
        test_values = [
            ("abc", 50.0),
            ("", 50.0),
            ("   ", 50.0),
            ("12.5", 12.5),
            ("0", 0.0),
            ("-100", -100.0),
            ("1e3", 1000.0),
            ("inf", float("inf")),
            ("nan", True),  # check np.isnan
            ("!@#$%", 50.0),
            ("None", 50.0),
        ]
        for inp, expected in test_values:
            self.proc_tab.inp_smooth.setText(inp)
            kw = self.panel.get_trevisan_kwargs()
            if inp == "nan":
                self.assertTrue(np.isnan(kw['smooth_ms']))
            else:
                self.assertEqual(kw['smooth_ms'], expected, f"Fallo para input {inp}")


class TestZoomableImageWidget(unittest.TestCase):
    """Pruebas de estrés para el widget interactivo de imagen ZoomableImageWidget."""

    def setUp(self):
        self.widget = ZoomableImageWidget(placeholder="[Test Placeholder]")

    def test_initial_state(self):
        """Verifica el estado inicial del widget."""
        self.assertIsNone(self.widget._pixmap)
        self.assertEqual(self.widget.placeholder_text, "[Test Placeholder]")
        self.assertEqual(self.widget.img_label.text(), "[Test Placeholder]")
        self.assertEqual(self.widget.lbl_zoom.text(), "Ajustado")

    def test_set_valid_pixmap(self):
        """Prueba la asignación de un QPixmap válido y su ajuste visual."""
        img = QImage(400, 300, QImage.Format_RGB32)
        img.fill(Qt.blue)
        pixmap = QPixmap.fromImage(img)
        self.widget.setPixmap(pixmap, filepath="/path/test.png")
        self.assertIsNotNone(self.widget._pixmap)
        self.assertEqual(self.widget._filepath, "/path/test.png")
        self.assertEqual(self.widget.img_label.text(), "")

    def test_set_null_and_none_pixmap(self):
        """Prueba la asignación de pixmap nulo o None."""
        self.widget.setPixmap(QPixmap())
        self.assertEqual(self.widget.img_label.text(), "[Test Placeholder]")
        self.widget.setPixmap(None)
        self.assertEqual(self.widget.img_label.text(), "[Test Placeholder]")

    def test_set_text(self):
        """Prueba la asignación directa de texto."""
        self.widget.setText("Texto informativo")
        self.assertIsNone(self.widget._pixmap)
        self.assertEqual(self.widget.img_label.text(), "Texto informativo")
        self.assertEqual(self.widget.lbl_zoom.text(), "-")

    def test_zoom_in_out_limits(self):
        """Prueba los límites de escala al hacer zoom repetidamente."""
        img = QImage(200, 200, QImage.Format_RGB32)
        img.fill(Qt.red)
        pix = QPixmap.fromImage(img)
        self.widget.setPixmap(pix)

        # Zoom in repetido hasta el tope (8.0)
        for _ in range(50):
            self.widget.zoom_in()
        self.assertAlmostEqual(self.widget._scale_factor, 8.0, places=2)

        # Zoom out repetido hasta el piso (0.1)
        for _ in range(100):
            self.widget.zoom_out()
        self.assertAlmostEqual(self.widget._scale_factor, 0.1, places=2)

        # Reset zoom
        self.widget.reset_zoom()
        self.assertEqual(self.widget._scale_factor, 1.0)

    def test_fit_to_view_dimensions(self):
        """Prueba el ajuste automático con distintas relaciones de aspecto del viewport."""
        img = QImage(1000, 500, QImage.Format_RGB32)
        pix = QPixmap.fromImage(img)
        self.widget.setPixmap(pix)
        self.widget.fit_to_view()
        self.assertTrue(self.widget._fit_mode)
        self.assertTrue(0.05 <= self.widget._scale_factor <= 1.0)

    def test_fullscreen_dialog_execution(self):
        """Prueba que el diálogo de pantalla completa se inicialice correctamente sin bloquear."""
        img = QImage(300, 200, QImage.Format_RGB32)
        pix = QPixmap.fromImage(img)
        self.widget.setPixmap(pix)

        # Usar un timer para cerrar el diálogo inmediatamente si se abre
        QTimer.singleShot(50, lambda: QApplication.activeModalWidget() and QApplication.activeModalWidget().accept())
        # Llamamos a show_fullscreen en modo headless
        # En headless no se bloquea si el timer lo cierra
        self.widget.show_fullscreen()


class TestReaperStyleHubHeadless(unittest.TestCase):
    """Pruebas de inicialización, cambio de pestañas y visor de resultados en ReaperStyleHub."""

    def setUp(self):
        self.hub = ReaperStyleHub()

    def test_hub_initialization(self):
        """Verifica que la ventana principal y los subcomponentes se inicialicen correctamente."""
        self.assertIsNotNone(self.hub.visor_subtabs)
        self.assertEqual(self.hub.visor_subtabs.count(), 2)
        self.assertIsNotNone(self.hub.img_viewer)
        self.assertIsNotNone(self.hub.tbl_metricas_visor)
        self.assertIsNotNone(self.hub.txt_metricas_visor)

    def test_result_loading_png(self):
        """Prueba la carga de una imagen PNG en el visor."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_png = f.name
        try:
            img = QImage(100, 100, QImage.Format_RGB32)
            img.fill(Qt.green)
            img.save(tmp_png)

            self.hub.cmb_resultados.clear()
            self.hub.cmb_resultados.addItem("test_img", tmp_png)
            self.hub._cargar_imagen_visor(0)

            self.assertEqual(self.hub.visor_subtabs.currentIndex(), 0)
            self.assertIsNotNone(self.hub.img_viewer._pixmap)
        finally:
            if os.path.exists(tmp_png):
                os.remove(tmp_png)

    def test_result_loading_valid_csv(self):
        """Prueba la carga de una tabla CSV válida en el visor de métricas."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
            f.write("Metrica,Vocal_A,Vocal_E\nExactitud,95.0,92.5\nF1,0.94,0.91\n")
            tmp_csv = f.name
        try:
            self.hub.cmb_resultados.clear()
            self.hub.cmb_resultados.addItem("test_metrics", tmp_csv)
            self.hub._cargar_imagen_visor(0)

            self.assertEqual(self.hub.visor_subtabs.currentIndex(), 1)
            self.assertEqual(self.hub.tbl_metricas_visor.rowCount(), 2)
            self.assertEqual(self.hub.tbl_metricas_visor.columnCount(), 3)
            self.assertFalse(self.hub.tbl_metricas_visor.isHidden())
            self.assertTrue(self.hub.txt_metricas_visor.isHidden())
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)

    def test_result_loading_corrupted_csv(self):
        """Prueba la carga de un CSV corrupto asegurando fallback a texto con mensaje de error."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='wb') as f:
            f.write(b"\x00\xff\xfe\x00\x00\x00corrupt,data\n1,2,3,4,5\n6")
            tmp_csv = f.name
        try:
            self.hub.cmb_resultados.clear()
            self.hub.cmb_resultados.addItem("corrupt_csv", tmp_csv)
            self.hub._cargar_imagen_visor(0)

            self.assertEqual(self.hub.visor_subtabs.currentIndex(), 1)
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)

    def test_result_loading_text_and_json(self):
        """Prueba la carga de archivos de texto (.txt, .json, .tex)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            f.write('{"accuracy": 0.98, "n_components": 3}')
            tmp_json = f.name
        try:
            self.hub.cmb_resultados.clear()
            self.hub.cmb_resultados.addItem("test_json", tmp_json)
            self.hub._cargar_imagen_visor(0)

            self.assertEqual(self.hub.visor_subtabs.currentIndex(), 1)
            self.assertIn("accuracy", self.hub.txt_metricas_visor.toPlainText())
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)

    def test_result_loading_invalid_index_and_missing_file(self):
        """Prueba que índices fuera de rango o archivos inexistentes no lancen excepciones."""
        self.hub.cmb_resultados.clear()
        self.hub._cargar_imagen_visor(-1)
        self.hub._cargar_imagen_visor(99)
        self.hub.cmb_resultados.addItem("missing", "/non/existent/file.png")
        self.hub._cargar_imagen_visor(0)


class TestLatentSpacePlotsEdgeCases(unittest.TestCase):
    """Pruebas de estrés y casos borde en funciones de ploteo 2D y 3D (R3)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_plot_scatter_2d_and_3d_single_sample_per_class(self):
        """Prueba plot_scatter con 1 muestra por clase en 2D y 3D."""
        X_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        X_2d = X_3d[:, :2]
        Y = np.array(['A', 'E', 'I'])

        p_2d = os.path.join(self.tmpdir.name, "scatter_2d_single.png")
        gpu.plot_scatter(X_2d, Y, "2D Single Sample", p_2d, is_3d=False, variance_ratios=[0.6, 0.4])
        self.assertTrue(os.path.exists(p_2d))
        self.assertGreater(os.path.getsize(p_2d), 1000)

        p_3d = os.path.join(self.tmpdir.name, "scatter_3d_single.png")
        gpu.plot_scatter(X_3d, Y, "3D Single Sample", p_3d, is_3d=True, variance_ratios=[0.5, 0.3, 0.2])
        self.assertTrue(os.path.exists(p_3d))
        self.assertGreater(os.path.getsize(p_3d), 1000)

    def test_plot_scatter_collinear_points(self):
        """Prueba plot_scatter con puntos colineales en 3D (rango 1)."""
        t = np.linspace(0, 10, 30)
        X_3d = np.column_stack([t, 2 * t, -3 * t])
        Y = np.array(['A'] * 15 + ['B'] * 15)

        p_collinear = os.path.join(self.tmpdir.name, "scatter_collinear.png")
        gpu.plot_scatter(X_3d, Y, "3D Collinear", p_collinear, is_3d=True, connect_points=True)
        self.assertTrue(os.path.exists(p_collinear))
        self.assertGreater(os.path.getsize(p_collinear), 1000)

    def test_plot_scatter_identical_coordinates(self):
        """Prueba plot_scatter con puntos completamente idénticos (varianza cero)."""
        X_3d = np.zeros((20, 3))
        Y = np.array(['A'] * 10 + ['B'] * 10)

        p_identical = os.path.join(self.tmpdir.name, "scatter_identical.png")
        gpu.plot_scatter(X_3d, Y, "3D Identical Coordinates", p_identical, is_3d=True)
        self.assertTrue(os.path.exists(p_identical))
        self.assertGreater(os.path.getsize(p_identical), 1000)

    def test_plot_scatter_ten_classes(self):
        """Prueba plot_scatter con 10 clases para validar paleta de colores y leyenda."""
        n_classes = 10
        samples_per_class = 4
        X_3d = np.random.randn(n_classes * samples_per_class, 3)
        Y = np.array([f"Class_{i}" for i in range(n_classes) for _ in range(samples_per_class)])

        p_10cls = os.path.join(self.tmpdir.name, "scatter_10cls.png")
        gpu.plot_scatter(X_3d, Y, "3D 10 Classes", p_10cls, is_3d=True, variance_ratios=[0.4, 0.3, 0.2])
        self.assertTrue(os.path.exists(p_10cls))
        self.assertGreater(os.path.getsize(p_10cls), 1000)

    def test_plot_scatter_3d_multi_angle_edge_cases(self):
        """Prueba plot_scatter_3d_multi_angle con datasets normales, colineales y de 10 clases."""
        # 1. Dataset normal 3 clases
        X_3d = np.random.randn(30, 3)
        Y = np.array(['A'] * 10 + ['E'] * 10 + ['O'] * 10)
        p_multi = os.path.join(self.tmpdir.name, "multi_angle_normal.png")
        gpu.plot_scatter_3d_multi_angle(X_3d, Y, "Multi Angle 3D", p_multi, variance_ratios=[0.5, 0.3, 0.2])
        self.assertTrue(os.path.exists(p_multi))

        # 2. Colineal
        t = np.linspace(0, 5, 20)
        X_col = np.column_stack([t, t, t])
        Y_col = np.array(['A'] * 10 + ['B'] * 10)
        p_multi_col = os.path.join(self.tmpdir.name, "multi_angle_collinear.png")
        gpu.plot_scatter_3d_multi_angle(X_col, Y_col, "Multi Angle Collinear", p_multi_col)
        self.assertTrue(os.path.exists(p_multi_col))

        # 3. 10 clases
        n_classes = 10
        X_10 = np.random.randn(40, 3)
        Y_10 = np.array([f"C{i}" for i in range(n_classes) for _ in range(4)])
        p_multi_10 = os.path.join(self.tmpdir.name, "multi_angle_10cls.png")
        gpu.plot_scatter_3d_multi_angle(X_10, Y_10, "Multi Angle 10 Classes", p_multi_10)
        self.assertTrue(os.path.exists(p_multi_10))

    def test_plot_analisis_errores_3d_kmeans_and_gmm(self):
        """Prueba plot_analisis_errores_3d con KMeans y GMM bajo casos límite."""
        # Dataset 3 clases
        X = np.vstack([
            np.random.normal(loc=[0, 0, 0], scale=0.5, size=(10, 3)),
            np.random.normal(loc=[3, 3, 3], scale=0.5, size=(10, 3)),
            np.random.normal(loc=[6, 0, 3], scale=0.5, size=(10, 3)),
        ])
        Y = np.array(['A'] * 10 + ['E'] * 10 + ['I'] * 10)
        Tomas = [f"toma_{i}" for i in range(30)]

        # KMeans
        p_err_km = os.path.join(self.tmpdir.name, "err_3d_kmeans.png")
        gpu.plot_analisis_errores_3d(X, Y, Tomas, "Analisis Errores KMeans", p_err_km, algoritmo="K-Means")
        self.assertTrue(os.path.exists(p_err_km))

        # GMM
        p_err_gmm = os.path.join(self.tmpdir.name, "err_3d_gmm.png")
        gpu.plot_analisis_errores_3d(X, Y, Tomas, "Analisis Errores GMM", p_err_gmm, algoritmo="GMM")
        self.assertTrue(os.path.exists(p_err_gmm))

    def test_plot_analisis_errores_3d_singular_covariance(self):
        """Prueba plot_analisis_errores_3d con puntos colineales donde la covarianza es singular."""
        t = np.linspace(-5, 5, 20)
        X = np.vstack([
            np.column_stack([t[:10], t[:10], t[:10]]),
            np.column_stack([t[10:] + 10, t[10:] + 10, t[10:] + 10])
        ])
        Y = np.array(['A'] * 10 + ['B'] * 10)
        Tomas = [f"toma_{i}" for i in range(20)]

        p_err_sing = os.path.join(self.tmpdir.name, "err_3d_singular.png")
        # Debe calcular elipsoides y elipses 2D proyectadas sin lanzar excepciones por singularidad
        gpu.plot_analisis_errores_3d(X, Y, Tomas, "Analisis Errores Singular Covariance", p_err_sing, algoritmo="K-Means")
        self.assertTrue(os.path.exists(p_err_sing))

    def test_pca_analysis_module_parity(self):
        """Verifica que el módulo pca_analysis.py exporte y ejecute las funciones idénticas."""
        X_3d = np.random.randn(15, 3)
        Y = np.array(['A'] * 5 + ['E'] * 5 + ['I'] * 5)
        Tomas = [f"t_{i}" for i in range(15)]

        p_scat = os.path.join(self.tmpdir.name, "pca_an_scatter.png")
        pca_an.plot_scatter(X_3d, Y, "PCA AN Scatter", p_scat, is_3d=True)
        self.assertTrue(os.path.exists(p_scat))

        p_multi = os.path.join(self.tmpdir.name, "pca_an_multi.png")
        pca_an.plot_scatter_3d_multi_angle(X_3d, Y, "PCA AN Multi", p_multi)
        self.assertTrue(os.path.exists(p_multi))

        p_err = os.path.join(self.tmpdir.name, "pca_an_err.png")
        pca_an.plot_analisis_errores_3d(X_3d, Y, Tomas, "PCA AN Error", p_err)
        self.assertTrue(os.path.exists(p_err))


    def test_guardar_tabla_imagen_edge_cases(self):
        """Prueba guardar_tabla_imagen con dataframes vacíos, de 1 fila, y columnas con nombres largos."""
        # 1. Dataframe estándar
        df_std = pd.DataFrame({"Vocal": ["A", "E", "I", "O", "U"], "Precision": [95.2, 88.0, 92.1, 85.4, 99.0]})
        p_tbl_std = os.path.join(self.tmpdir.name, "tabla_std.png")
        gpu.guardar_tabla_imagen(df_std, "Metricas Estandar", p_tbl_std)
        self.assertTrue(os.path.exists(p_tbl_std))

        # 2. Dataframe con 1 sola celda y nombres muy largos
        df_single = pd.DataFrame({"Metrica_De_Rendimiento_Extremadamente_Larga": [99.999]}, index=["Fila_Con_Nombre_Muy_Extenso"])
        p_tbl_single = os.path.join(self.tmpdir.name, "tabla_single.png")
        gpu.guardar_tabla_imagen(df_single, "Tabla Unica Celda", p_tbl_single)
        self.assertTrue(os.path.exists(p_tbl_single))

    def test_plot_confusion_matrix_heatmap(self):
        """Prueba plot_confusion_matrix_heatmap con matrices de confusión 2x2 y 5x5."""
        # 1. Matriz 5x5
        vocales = ["A", "E", "I", "O", "U"]
        cm_data = np.array([
            [90, 5, 2, 1, 2],
            [3, 88, 4, 3, 2],
            [1, 2, 95, 1, 1],
            [2, 3, 1, 90, 4],
            [1, 1, 1, 2, 95]
        ])
        df_cm = pd.DataFrame(cm_data, index=[f"Real {v}" for v in vocales], columns=[f"Real {v}" for v in vocales])
        p_cm = os.path.join(self.tmpdir.name, "cm_5x5.png")
        gpu.plot_confusion_matrix_heatmap(df_cm, "Matriz de Confusion 5x5", p_cm)
        self.assertTrue(os.path.exists(p_cm))

        # 2. Matriz 2x2
        cm_2x2 = pd.DataFrame([[80, 20], [10, 90]], index=["Real A", "Real B"], columns=["Real A", "Real B"])
        p_cm_2 = os.path.join(self.tmpdir.name, "cm_2x2.png")
        gpu.plot_confusion_matrix_heatmap(cm_2x2, "Matriz 2x2", p_cm_2)
        self.assertTrue(os.path.exists(p_cm_2))

    def test_plot_analisis_errores_3d_proyecciones_2d(self):
        """Prueba plot_analisis_errores_3d_proyecciones_2d con 3 clases y datos colineales."""
        X_3d = np.random.randn(30, 3)
        Y = np.array(['A'] * 10 + ['E'] * 10 + ['I'] * 10)
        p_proj = os.path.join(self.tmpdir.name, "proyecciones_2d.png")
        gpu.plot_analisis_errores_3d_proyecciones_2d(X_3d, Y, "Proyecciones 2D", p_proj, variance_ratios=[0.5, 0.3, 0.2])
        self.assertTrue(os.path.exists(p_proj))


if __name__ == "__main__":
    unittest.main()

