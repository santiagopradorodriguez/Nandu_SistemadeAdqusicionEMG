# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Test Integral: Visor CSV, Galería de Fotometría 3D y Build PyInstaller (.spec)
# ==============================================================================

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

# Configurar entorno Qt sin pantalla
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EMG_DESARROLLO = os.path.abspath(os.path.join(TESTS_DIR, '..'))
REPO_ROOT = os.path.abspath(os.path.join(EMG_DESARROLLO, '..'))

for p in (REPO_ROOT, EMG_DESARROLLO, os.path.join(EMG_DESARROLLO, 'gui_app'), os.path.join(EMG_DESARROLLO, 'herramientas_build')):
    if p not in sys.path:
        sys.path.insert(0, p)

from PySide6.QtWidgets import QApplication
from gui_app.views.csv_viewer_widget import CsvViewerWidget, PhotometryGalleryDialog
import herramientas_build.crear_spec_ejecutable as spec_builder


class TestCsvViewerPhotometryAndSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test_runner'])

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.med_dir = os.path.join(self.tmp_dir.name, "Sujeto1_Prueba_2026-10-24")
        os.makedirs(self.med_dir, exist_ok=True)

        # Crear estructura de subcarpetas
        self.ch0_dir = os.path.join(self.med_dir, "canal_0")
        self.ch1_dir = os.path.join(self.med_dir, "canal_1")
        self.foto_dir = os.path.join(self.med_dir, "fotometria")
        self.capturas_dir = os.path.join(self.foto_dir, "capturas_toma")
        os.makedirs(self.ch0_dir, exist_ok=True)
        os.makedirs(self.ch1_dir, exist_ok=True)
        os.makedirs(self.capturas_dir, exist_ok=True)

        # 1. metadata.json raíz con canales no contiguos (ai0 y ai2)
        self.root_meta = {
            "sujeto": "Sujeto1",
            "prueba": "Prueba_O_U",
            "comentario": "Notas iniciales del experimento en laboratorio",
            "notas_experimento": "Notas iniciales del experimento en laboratorio",
            "channels": ["ai0", "ai2"],
            "muscles": ["Cigomático mayor", "Depresor del ángulo bucal"],
            "muscles_map": {
                "canal_0": "Cigomático mayor",
                "canal_1": "Depresor del ángulo bucal"
            },
            "separabilidad_o_u": {
                "delta_apertura": "0.142",
                "d_cohen": "2.85",
                "distincion": "Alta"
            }
        }
        with open(os.path.join(self.med_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.root_meta, f, indent=2)

        # 2. metadata.json de canal_0 y canal_1
        ch0_meta = {
            "canal": "canal_0",
            "physical_channel": "ai0",
            "musculo": "Cigomático mayor",
            "resistencia_ohm": 100.0,
            "comentario": self.root_meta["comentario"]
        }
        with open(os.path.join(self.ch0_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(ch0_meta, f, indent=2)

        ch1_meta = {
            "canal": "canal_1",
            "physical_channel": "ai2",
            "musculo": "Depresor del ángulo bucal",
            "resistencia_ohm": 100.0,
            "comentario": self.root_meta["comentario"]
        }
        with open(os.path.join(self.ch1_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(ch1_meta, f, indent=2)

        # 3. calibracion_espacial.json
        self.cal_espacial = {
            "exito": True,
            "metadatos_calibracion_espacial": {
                "distancia_interpupilar_px": 86.4,
                "distancia_interpupilar_3d": 0.0632,
                "electrodos": [
                    {
                        "canal": "canal_0",
                        "canal_fisico_ai": "ai0",
                        "musculo_asignado": "Cigomático mayor",
                        "centroide_2d_px": [128.5, 256.0],
                        "vertice_malla_3d": 205,
                        "vector_3d_relativo_normalizado": [0.1245, -0.3421, 0.5678],
                        "distancia_3d_normalizada": 0.6789
                    },
                    {
                        "canal": "canal_1",
                        "canal_fisico_ai": "ai2",
                        "musculo_asignado": "Depresor del ángulo bucal",
                        "centroide_2d_px": [140.0, 310.5],
                        "vertice_malla_3d": 379,
                        "vector_3d_relativo_normalizado": [-0.1523, -0.4120, 0.4981],
                        "distancia_3d_normalizada": 0.6654
                    }
                ]
            }
        }
        with open(os.path.join(self.foto_dir, "calibracion_espacial.json"), "w", encoding="utf-8") as f:
            json.dump(self.cal_espacial, f, indent=2)

        # 4. fotometria_telemetria.csv
        telem_data = {
            "idx": [0, 1, 2, 3],
            "t_daq_s": [0.5, 1.0, 1.5, 2.0],
            "fase": ["evento", "evento", "evento", "evento"],
            "palabra": ["/O/", "/O/", "/U/", "/U/"],
            "pulso": [1, 2, 3, 4],
            "archivo_imagen": [
                "000_evento_p01_O.jpg",
                "001_evento_p02_O.jpg",
                "002_evento_p03_U.jpg",
                "003_evento_p04_U.jpg"
            ],
            "yaw_deg": [1.5, 1.8, -0.5, -0.8],
            "pitch_deg": [-2.1, -2.0, -1.9, -1.8],
            "roll_deg": [0.4, 0.5, 0.3, 0.2],
            "apertura_labial_norm": [0.380, 0.395, 0.220, 0.235]
        }
        pd.DataFrame(telem_data).to_csv(os.path.join(self.foto_dir, "fotometria_telemetria.csv"), index=False)

        # 5. Imágenes de prueba (.png y .jpg)
        img_dummy = Image.new("RGB", (160, 120), color=(0, 200, 150))
        img_dummy.save(os.path.join(self.foto_dir, "ref_frontal_triangulada.png"))
        img_dummy.save(os.path.join(self.foto_dir, "ref_perfil_45.jpg"))
        for fname in telem_data["archivo_imagen"]:
            img_dummy.save(os.path.join(self.capturas_dir, fname))

        # 6. grabacion.csv con canales no contiguos
        t = np.linspace(0, 3.0, 300)
        s0 = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.random.randn(len(t))
        s1 = 0.03 * np.cos(2 * np.pi * 30 * t) + 0.01 * np.random.randn(len(t))
        df_sig = pd.DataFrame({
            "Tiempo (s)": t,
            "Canal 0": s0,
            "Canal 1": s1
        })
        self.csv_path = os.path.join(self.med_dir, "grabacion.csv")
        df_sig.to_csv(self.csv_path, index=False)

        # 7. calibracion.csv
        self.cal_csv_path = os.path.join(self.med_dir, "calibracion.csv")
        df_sig.to_csv(self.cal_csv_path, index=False)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_01_crear_spec_ejecutable(self):
        """Verifica que crear_spec_ejecutable.py genere el archivo EMG_Studio.spec con todas las dependencias."""
        spec_builder.generar_spec()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(spec_builder.__file__)))
        build_dir = os.path.join(base_dir, "EMG_Ejecutable_Build")
        spec_file = os.path.join(build_dir, "EMG_Studio.spec")

        self.assertTrue(os.path.isfile(spec_file), f"No se encontró {spec_file}")

        with open(spec_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Validar dependencias clave requeridas en el objetivo
        self.assertIn("calibracion_espacial_electrodos", content)
        self.assertIn("fotometria_webcam", content)
        self.assertIn("mediapipe", content)
        self.assertIn("collect_data_files('mediapipe')", content)
        self.assertIn("cv2", content)
        self.assertIn("face_landmarker.task", content)

    def test_02_csv_viewer_banner_and_non_contiguous_channels(self):
        """Verifica que el visor cargue notas, separabilidad, fotometría y resuelva canales no contiguos."""
        viewer = CsvViewerWidget()
        viewer.load_csv(self.csv_path)

        # Cargar directamente sincrónico para validar el procesamiento completo de canales y UI
        df = pd.read_csv(self.csv_path)
        viewer._on_csv_loaded(df, canales=["Canal 0", "Canal 1"], time_col="Tiempo (s)", max_y=1.0)

        # 1. Banner superior
        viewer.show()
        self.assertFalse(viewer.lbl_notas_banner.isHidden())
        self.assertTrue(viewer.lbl_notas_banner.isVisible())
        banner_text = viewer.lbl_notas_banner.text()
        self.assertIn("Notas iniciales del experimento", banner_text)
        self.assertIn("Separabilidad /O/ vs /U/", banner_text)
        self.assertIn("Calibración 3D", banner_text)

        # 2. Edición de notas
        self.assertEqual(viewer.txt_notas_sesion.toPlainText().strip(), "Notas iniciales del experimento en laboratorio")

        # 3. Canales no contiguos (ai0 y ai2 mapeados a músculos respectivos)
        self.assertIn("Canal 0", viewer.channel_checkboxes)
        self.assertIn("Canal 1", viewer.channel_checkboxes)

        chk0 = viewer.channel_checkboxes["Canal 0"]
        chk1 = viewer.channel_checkboxes["Canal 1"]

        self.assertIn("Cigomático mayor", chk0.text())
        self.assertIn("ai0", chk0.text())
        self.assertIn("Depresor del ángulo bucal", chk1.text())
        self.assertIn("ai2", chk1.text())

        # 4. Botón de Fotometría habilitado y visible
        self.assertTrue(viewer.btn_ver_fotometria.isVisible())
        self.assertTrue(viewer.btn_ver_fotometria.isEnabled())

    def test_03_guardar_notas_actualiza_archivos_y_banner(self):
        """Verifica que guardar notas actualice metadata.json (raíz y canales), notas_experimento.txt y banner."""
        viewer = CsvViewerWidget()
        viewer.load_csv(self.csv_path)

        nuevas_notas = "Notas editadas por el operador: Sujeto descansado, electrodo CH1 reposicionado."
        viewer.txt_notas_sesion.setPlainText(nuevas_notas)
        viewer._save_session_notes()

        # Verificar archivo raíz
        with open(os.path.join(self.med_dir, "metadata.json"), "r", encoding="utf-8") as f:
            md_r = json.load(f)
        self.assertEqual(md_r.get("notas_experimento"), nuevas_notas)
        self.assertEqual(md_r.get("comentario"), nuevas_notas)

        # Verificar archivo canal_0 y canal_1
        with open(os.path.join(self.ch0_dir, "metadata.json"), "r", encoding="utf-8") as f:
            md_0 = json.load(f)
        self.assertEqual(md_0.get("notas_experimento"), nuevas_notas)

        with open(os.path.join(self.ch1_dir, "metadata.json"), "r", encoding="utf-8") as f:
            md_1 = json.load(f)
        self.assertEqual(md_1.get("notas_experimento"), nuevas_notas)

        # Verificar notas_experimento.txt
        with open(os.path.join(self.med_dir, "notas_experimento.txt"), "r", encoding="utf-8") as f:
            txt_content = f.read().strip()
        self.assertEqual(txt_content, nuevas_notas)

        # Verificar actualización del banner en UI
        self.assertIn(nuevas_notas, viewer.lbl_notas_banner.text())

    def test_04_dialogo_fotometria_galeria_y_telemetria(self):
        """Verifica que PhotometryGalleryDialog cargue imágenes, tabla 3D y telemetría facial con sincronización."""
        dlg = PhotometryGalleryDialog(self.med_dir)

        # 1. Galería de imágenes cargadas (referencias y capturas de pulsos)
        count_img = dlg.list_images.count()
        self.assertGreaterEqual(count_img, 6) # 2 ref + 4 capturas

        # 2. Tabla de electrodos 3D
        self.assertEqual(dlg.tbl_electrodos.rowCount(), 2)
        # Comprobar columnas: Canal, Músculo, Centroide, Vértice, Vector 3D, Distancia
        item_ch0 = dlg.tbl_electrodos.item(0, 0)
        item_musc0 = dlg.tbl_electrodos.item(0, 1)
        self.assertIn("ai0", item_ch0.text())
        self.assertIn("Cigomático mayor", item_musc0.text())

        # 3. Tabla de telemetría
        self.assertEqual(dlg.tbl_telemetria.rowCount(), 4)

        # 4. Sincronización interactiva: al seleccionar fila en tabla de telemetría se selecciona la imagen
        dlg.tbl_telemetria.selectRow(2) # Correspondiente a 002_evento_p03_U.jpg
        selected_item = dlg.list_images.currentItem()
        self.assertIsNotNone(selected_item)
        self.assertIn("002_evento_p03_U.jpg", selected_item.text())

        dlg.close()

    def test_05_cargar_calibracion_csv(self):
        """Verifica que el visor cargue correctamente calibracion.csv activando banner, notas y fotometría."""
        viewer = CsvViewerWidget()
        viewer.load_csv(self.cal_csv_path)

        viewer.show()
        self.assertEqual(viewer.current_medicion_dir, self.med_dir)
        self.assertFalse(viewer.btn_ver_fotometria.isHidden())
        self.assertTrue(viewer.btn_ver_fotometria.isEnabled())
        self.assertFalse(viewer.lbl_notas_banner.isHidden())
        self.assertTrue(viewer.lbl_notas_banner.isVisible())


if __name__ == "__main__":
    unittest.main()
