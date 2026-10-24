import os
import sys
import tempfile
import json
import unittest
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6 import QtWidgets, QtCore

class TestNonContiguousMusclesCVM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if not cls.app:
            cls.app = QtWidgets.QApplication([])

    def test_01_config_manager_channel_resolution(self):
        from EMG_desarrollo.utils.config_manager import ConfigManager, get_muscle_color
        mgr = ConfigManager()

        # Test index=3 resolution without double Canal
        cfg3 = mgr.get_channel_config(3)
        self.assertIn("musculo", cfg3)
        cfg_str3 = mgr.get_channel_config("Canal 3")
        self.assertEqual(cfg3["musculo"], cfg_str3["musculo"])

        # Test updating canal 3 to Risorio
        mgr.update_channel_muscle(3, "Risorio", save=False)
        cfg_risorio = mgr.get_channel_config(3)
        self.assertEqual(cfg_risorio["musculo"], "Risorio")
        # Risorio color should be green (#00cc44), NOT forced red (#ff0000)
        self.assertEqual(cfg_risorio["color_hex"].lower(), "#00cc44")

        # Re-assign back to Micrófono
        mgr.update_channel_muscle(3, "Micrófono", save=False)
        cfg_mic = mgr.get_channel_config(3)
        self.assertEqual(cfg_mic["color_hex"].lower(), "#ff0000")

    def test_02_plotter_non_contiguous_channels(self):
        from EMG_desarrollo.acquisition.autoforge_daq_experimental import RealTimePlotter
        plotter = RealTimePlotter()

        # Simulate selecting non-contiguous channels: ai1 and ai3
        self.assertGreaterEqual(len(plotter.channel_checkboxes), 4)
        plotter.channel_checkboxes[0].setChecked(False)
        plotter.channel_checkboxes[1].setChecked(True)
        plotter.channel_checkboxes[2].setChecked(False)
        plotter.channel_checkboxes[3].setChecked(True)

        plotter._actualizar_canales_activos(update_gui=True)

        self.assertEqual(plotter.active_channel_indices, [1, 3])
        self.assertEqual(plotter.NUM_CANALES, 2)
        self.assertEqual(len(plotter.curvas), 2)
        
        # Check curve names contain ai1 and ai3
        c0_name = plotter.curvas[0].name()
        c1_name = plotter.curvas[1].name()
        self.assertTrue(c0_name.startswith("ai1"), f"Expected ai1 prefix, got {c0_name}")
        self.assertTrue(c1_name.startswith("ai3"), f"Expected ai3 prefix, got {c1_name}")

    def test_03_muscle_selection_dialog_runtime_update(self):
        from EMG_desarrollo.acquisition.autoforge_daq_experimental import RealTimePlotter, MuscleSelectionDialog
        plotter = RealTimePlotter()
        plotter.channel_checkboxes[0].setChecked(False)
        plotter.channel_checkboxes[1].setChecked(True)
        plotter.channel_checkboxes[2].setChecked(False)
        plotter.channel_checkboxes[3].setChecked(True)
        plotter._actualizar_canales_activos(update_gui=True)

        dialog = MuscleSelectionDialog(plotter)
        self.assertEqual(dialog.active_hw_channels, [1, 3])
        self.assertIn(1, dialog.combos)
        self.assertIn(3, dialog.combos)

        # Set muscles
        dialog.combos[1].setCurrentText("Zygomaticus Major")
        dialog.combos[3].setCurrentText("Risorio")
        dialog.guardar_y_cerrar()

        # Verify plotter and config were updated
        self.assertEqual(plotter.nombres_musculos_hw[1], "Zygomaticus Major")
        self.assertEqual(plotter.nombres_musculos_hw[3], "Risorio")
        self.assertEqual(plotter.nombres_musculos, ["Zygomaticus Major", "Risorio"])

        # Check curves updated
        self.assertIn("Zygomaticus Major", plotter.curvas[0].name())
        self.assertIn("Risorio", plotter.curvas[1].name())

    def test_04_cvm_vowel_triangle_a_i_u(self):
        from EMG_desarrollo.acquisition.autoforge_daq_experimental import RealTimePlotter, AutoForgeDialog
        plotter = RealTimePlotter()
        plotter.channel_checkboxes[0].setChecked(True)
        plotter.channel_checkboxes[1].setChecked(True)
        plotter._actualizar_canales_activos(update_gui=False)

        # AutoForgeDialog CVM checkbox text
        dlg = AutoForgeDialog(plotter)
        self.assertIn("A, I, U", dlg.chk_calibrar_cvm.text())

        # Start CVM
        plotter.is_autoforge_running = True
        plotter.estado_iniciar_calibracion_cvm(lambda: None)
        self.assertEqual(len(plotter.cvm_sequence), 9)
        vowels = [item[0] for item in plotter.cvm_sequence]
        self.assertEqual(vowels, ['A', 'A', 'A', 'I', 'I', 'I', 'U', 'U', 'U'])
        self.assertNotIn('O', vowels)

        # Mic channel exclusion in supremo calculation
        plotter.cvm_env_max_per_channel = [150.0, 5000.0] # Let ch1 be mic
        plotter.nombres_musculos = ["Risorio", "Micrófono"]
        plotter.active_channel_indices = [0, 3]
        plotter.estado_finalizar_calibracion_cvm()
        # Ch 1 is mic, so supremo should be 150.0 (not 5000.0)
        self.assertEqual(plotter.supremo_global_uV, 150.0)

    def test_05_final_noise_metadata_propagation(self):
        from EMG_desarrollo.acquisition.autoforge_daq_experimental import RealTimePlotter
        plotter = RealTimePlotter()
        plotter.is_autoforge_running = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = os.path.join(tmp_dir, "A_Prueba1_Sujeto1")
            ch0_dir = os.path.join(session_dir, "canal_0")
            os.makedirs(ch0_dir, exist_ok=True)

            root_meta = {"ruido_basal": {"inicial": {"mean_uV": [10.0]}}}
            with open(os.path.join(session_dir, "metadata.json"), 'w') as f:
                json.dump(root_meta, f)
            with open(os.path.join(ch0_dir, "metadata.json"), 'w') as f:
                json.dump(root_meta, f)

            plotter.autoforge_session_dirs = [session_dir]
            plotter.final_raw_noise_accumulated = [[np.zeros(100)]]
            plotter.NUM_CANALES = 1

            plotter.estado_finalizar_ruido_final()

            # Verify final noise was written to metadata.json
            with open(os.path.join(session_dir, "metadata.json"), 'r') as f:
                saved_root = json.load(f)
            self.assertIn("final", saved_root["ruido_basal"])
            self.assertIn("mean_uV", saved_root["ruido_basal"]["final"])

            with open(os.path.join(ch0_dir, "metadata.json"), 'r') as f:
                saved_ch0 = json.load(f)
            self.assertIn("final", saved_ch0["ruido_basal"])

    def test_06_autoforge_token_and_reset(self):
        from EMG_desarrollo.acquisition.autoforge_daq_experimental import RealTimePlotter
        plotter = RealTimePlotter()
        plotter.is_autoforge_running = True
        plotter.is_cvm_active = True
        token_before = getattr(plotter, '_autoforge_token', 0)

        fired = []
        plotter._safe_autoforge_single_shot(10, lambda: fired.append(True))

        plotter.reset_autoforge_buttons()
        self.assertFalse(plotter.is_autoforge_running)
        self.assertFalse(plotter.is_cvm_active)
        self.assertGreater(getattr(plotter, '_autoforge_token', 0), token_before)

        # Process Qt events to see if timer fires
        QtWidgets.QApplication.processEvents()
        self.assertEqual(fired, [], "Timer callback should have been canceled by token change")

if __name__ == '__main__':
    unittest.main()
