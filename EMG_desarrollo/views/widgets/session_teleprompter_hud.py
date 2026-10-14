# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Archivo: EMG_desarrollo/views/widgets/session_teleprompter_hud.py
# Descripción: HUD Teleprompter con carrusel de palabras, medidor % CVM y
#              botón interactivo de Pausa/Reanudación de sesión.
# ==============================================================================

from __future__ import annotations
from PySide6 import QtWidgets, QtCore


class SessionTeleprompterHUD(QtWidgets.QFrame):
    pause_toggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(135)
        self.setStyleSheet("""
            QFrame {
                background-color: #0B0C10;
                border: 2px solid #45A29E;
                border-radius: 8px;
            }
        """)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(20)

        # 1. Columna Izquierda: Estado, Botón Pausa y Palabra Anterior
        left_col = QtWidgets.QVBoxLayout()
        self.lbl_phase_badge = QtWidgets.QLabel("FASE: EN ESPERA")
        self.lbl_phase_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_phase_badge.setStyleSheet("""
            background-color: #1F2833; color: #66FCF1; font-weight: bold;
            font-size: 12px; padding: 4px 10px; border-radius: 4px; border: 1px solid #66FCF1;
        """)

        self.btn_pause_session = QtWidgets.QPushButton("⏸ PAUSAR SESIÓN")
        self.btn_pause_session.setCheckable(True)
        self.btn_pause_session.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_pause_session.setStyleSheet("""
            QPushButton {
                background-color: #221A00; color: #FFCC00; font-weight: bold;
                font-size: 13px; padding: 6px 12px; border: 2px solid #FFCC00; border-radius: 4px;
            }
            QPushButton:hover { background-color: #332800; }
            QPushButton:checked {
                background-color: #FFCC00; color: #000000; border: 2px solid #FFFFFF;
            }
        """)
        self.btn_pause_session.toggled.connect(self._on_pause_btn_toggled)

        self.lbl_prev_word = QtWidgets.QLabel("ANTERIOR: --")
        self.lbl_prev_word.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_prev_word.setStyleSheet("color: #6C7A89; font-size: 15px; font-weight: bold; border: none;")

        left_col.addWidget(self.lbl_phase_badge)
        left_col.addWidget(self.btn_pause_session)
        left_col.addStretch()
        left_col.addWidget(self.lbl_prev_word)
        layout.addLayout(left_col, stretch=2)

        # 2. Columna Central: Palabra Objetivo Gigante + Barra de Pulso
        center_col = QtWidgets.QVBoxLayout()
        self.lbl_current_word = QtWidgets.QLabel("PREPARANDO")
        self.lbl_current_word.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_current_word.setStyleSheet("""
            color: #00FFCC;
            font-family: 'Segoe UI', 'Inter', sans-serif;
            font-size: 56px;
            font-weight: 900;
            border: none;
        """)

        self.bar_rep_progress = QtWidgets.QProgressBar()
        self.bar_rep_progress.setRange(0, 100)
        self.bar_rep_progress.setValue(0)
        self.bar_rep_progress.setFormat("Pulso %v / %m")
        self.bar_rep_progress.setFixedHeight(16)
        self.bar_rep_progress.setStyleSheet("""
            QProgressBar {
                background-color: #1F2833; color: #FFFFFF; font-weight: bold;
                border: 1px solid #45A29E; border-radius: 4px; text-align: center; font-size: 11px;
            }
            QProgressBar::chunk { background-color: #00FFCC; border-radius: 3px; }
        """)

        center_col.addWidget(self.lbl_current_word, stretch=1)
        center_col.addWidget(self.bar_rep_progress)
        layout.addLayout(center_col, stretch=5)

        # 3. Columna Derecha: Siguiente Palabra y Medidor de Fuerza (% CVM)
        right_col = QtWidgets.QVBoxLayout()
        self.lbl_next_word = QtWidgets.QLabel("SIGUIENTE: --")
        self.lbl_next_word.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_next_word.setStyleSheet("color: #FFAA00; font-size: 18px; font-weight: bold; border: none;")

        cvm_box = QtWidgets.QHBoxLayout()
        lbl_cvm_title = QtWidgets.QLabel("Esfuerzo (% CVM):")
        lbl_cvm_title.setStyleSheet("color: #C5C6C7; font-size: 11px; font-weight: bold; border: none;")
        self.bar_cvm_force = QtWidgets.QProgressBar()
        self.bar_cvm_force.setRange(0, 100)
        self.bar_cvm_force.setValue(0)
        self.bar_cvm_force.setFixedHeight(16)
        self.bar_cvm_force.setStyleSheet("""
            QProgressBar {
                background-color: #1F2833; color: #FFFFFF; font-weight: bold;
                border: 1px solid #555; border-radius: 4px; text-align: center; font-size: 10px;
            }}
            QProgressBar::chunk { background-color: #39FF14; }
        """)
        cvm_box.addWidget(lbl_cvm_title)
        cvm_box.addWidget(self.bar_cvm_force)

        right_col.addWidget(self.lbl_next_word)
        right_col.addStretch()
        right_col.addLayout(cvm_box)
        layout.addLayout(right_col, stretch=2)

    def _on_pause_btn_toggled(self, paused: bool) -> None:
        if paused:
            self.btn_pause_session.setText("▶ REANUDAR SESIÓN")
            self.lbl_phase_badge.setText("⏸ SESIÓN EN PAUSA")
        else:
            self.btn_pause_session.setText("⏸ PAUSAR SESIÓN")
            self.lbl_phase_badge.setText("FASE: REANUDADA")
        self.pause_toggled.emit(paused)

    def set_paused_state(self, paused: bool) -> None:
        self.btn_pause_session.blockSignals(True)
        self.btn_pause_session.setChecked(paused)
        self.btn_pause_session.setText("▶ REANUDAR SESIÓN" if paused else "⏸ PAUSAR SESIÓN")
        self.btn_pause_session.blockSignals(False)

    def update_teleprompter(
        self,
        phase_text: str,
        current_word: str,
        prev_word: str = "--",
        next_word: str = "--",
        current_rep: int = 0,
        total_reps: int = 1,
        accent_color: str = "#00FFCC",
    ) -> None:
        self.lbl_phase_badge.setText(phase_text.upper())
        self.lbl_current_word.setText(current_word.upper())
        self.lbl_current_word.setStyleSheet(
            f"color: {accent_color}; font-family: 'Segoe UI', sans-serif; font-size: 56px; font-weight: 900; border: none;"
        )
        self.lbl_prev_word.setText(f"ANTERIOR: {prev_word.upper()}")
        self.lbl_next_word.setText(f"SIGUIENTE: {next_word.upper()}")
        self.bar_rep_progress.setMaximum(max(1, total_reps))
        self.bar_rep_progress.setValue(min(current_rep, total_reps))

    def update_cvm_effort(self, current_env_peak_uV: float, cvm_supremo_p95_uV: float) -> None:
        if cvm_supremo_p95_uV <= 1e-3:
            return
        pct = int(min(100.0, max(0.0, (current_env_peak_uV / cvm_supremo_p95_uV) * 100.0)))
        self.bar_cvm_force.setValue(pct)
        chunk_col = "#FFAA00" if pct < 20 else ("#39FF14" if pct <= 85 else "#FF0055")
        self.bar_cvm_force.setStyleSheet(f"""
            QProgressBar {{
                background-color: #1F2833; color: #FFFFFF; font-weight: bold;
                border: 1px solid #555; border-radius: 4px; text-align: center; font-size: 10px;
            }}
            QProgressBar::chunk {{ background-color: {chunk_col}; }}
        """)
