# ==============================================================================
# Archivo: EMG_desarrollo/views/widgets/reaper_oscilloscope.py
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Descripción: Osciloscopio conmutable (Superpuesto vs Multi-Pista REAPER),
#              controles Mute/Solo, sombreado de Pre-Roll y botón de Pausa.
# ==============================================================================

from __future__ import annotations
from typing import List, Optional
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg


class TrackControlHeader(QtWidgets.QFrame):
    """Cabezal compacto de pista estilo REAPER con nombre de músculo, Mute, Solo y Vp-p."""
    mute_toggled = QtCore.Signal(int, bool)
    solo_toggled = QtCore.Signal(int, bool)

    def __init__(self, ch_idx: int, name: str, color_hex: str, parent=None):
        super().__init__(parent)
        self.ch_idx = ch_idx
        self.color_hex = color_hex
        self.setFixedHeight(32)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #12161F;
                border-left: 4px solid {color_hex};
                border-bottom: 1px solid #232A38;
                border-radius: 3px;
            }}
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        self.lbl_name = QtWidgets.QLabel(f"CH{ch_idx}: {name}")
        self.lbl_name.setStyleSheet(f"color: {color_hex}; font-weight: bold; font-size: 12px; border: none;")
        layout.addWidget(self.lbl_name)

        self.btn_mute = QtWidgets.QPushButton("M")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setFixedSize(22, 22)
        self.btn_mute.setToolTip("Mute / Ocultar canal")
        self.btn_mute.setStyleSheet("""
            QPushButton { background-color: #1F2833; color: #8892B0; font-weight: bold; border: 1px solid #333; border-radius: 3px; }
            QPushButton:checked { background-color: #FF0055; color: #FFFFFF; border: 1px solid #FF5588; }
        """)
        self.btn_mute.toggled.connect(lambda checked: self.mute_toggled.emit(self.ch_idx, checked))

        self.btn_solo = QtWidgets.QPushButton("S")
        self.btn_solo.setCheckable(True)
        self.btn_solo.setFixedSize(22, 22)
        self.btn_solo.setToolTip("Solo (aislar este canal)")
        self.btn_solo.setStyleSheet("""
            QPushButton { background-color: #1F2833; color: #8892B0; font-weight: bold; border: 1px solid #333; border-radius: 3px; }
            QPushButton:checked { background-color: #FFCC00; color: #000000; border: 1px solid #FFEE55; }
        """)
        self.btn_solo.toggled.connect(lambda checked: self.solo_toggled.emit(self.ch_idx, checked))

        self.lbl_metrics = QtWidgets.QLabel("-- µVp-p | -- µVrms")
        self.lbl_metrics.setStyleSheet("color: #C5C6C7; font-family: 'Consolas', monospace; font-size: 11px; border: none;")

        layout.addWidget(self.btn_mute)
        layout.addWidget(self.btn_solo)
        layout.addSpacing(8)
        layout.addWidget(self.lbl_metrics)
        layout.addStretch()


class ReaperOscilloscopeWidget(QtWidgets.QWidget):
    """
    Osciloscopio avanzado conmutable entre Vista Superpuesta y Pistas Separadas (REAPER),
    con Pre-Roll Circular sombreado y botón de Pausa integrado.
    """
    preroll_changed = QtCore.Signal(float)
    pause_requested = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.num_channels = 0
        self.channel_names: List[str] = []
        self.channel_colors: List[str] = []
        self.is_stacked_mode = False
        self.preroll_seconds = 1.0
        self.plot_duration_s = 10.0

        self.track_headers: List[TrackControlHeader] = []
        self.plots: List[pg.PlotItem] = []
        self.curves: List[pg.PlotDataItem] = []
        self.preroll_regions: List[pg.LinearRegionItem] = []

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        toolbar = QtWidgets.QFrame()
        toolbar.setStyleSheet("background-color: #0B0C10; border: 1px solid #1F2833; border-radius: 4px;")
        tb_layout = QtWidgets.QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 4, 8, 4)

        self.btn_pause = QtWidgets.QPushButton("⏸ PAUSAR")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setStyleSheet("""
            QPushButton {
                background-color: #1F2833; color: #FFCC00; font-weight: bold;
                padding: 4px 14px; border: 2px solid #FFCC00; border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #FFCC00; color: #000000; border: 2px solid #FFFFFF;
            }
        """)
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        tb_layout.addWidget(self.btn_pause)

        tb_layout.addSpacing(10)
        self.btn_toggle_layout = QtWidgets.QPushButton("Vista: Pistas Separadas (REAPER)")
        self.btn_toggle_layout.setCheckable(True)
        self.btn_toggle_layout.setStyleSheet("""
            QPushButton {
                background-color: #1F2833; color: #66FCF1; font-weight: bold;
                padding: 4px 12px; border: 1px solid #45A29E; border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #66FCF1; color: #0B0C10;
            }
        """)
        self.btn_toggle_layout.toggled.connect(self._on_layout_mode_toggled)
        tb_layout.addWidget(self.btn_toggle_layout)

        tb_layout.addSpacing(15)
        lbl_pre = QtWidgets.QLabel("Pre-Roll Circular:")
        lbl_pre.setStyleSheet("color: #66FCF1; font-weight: bold; font-size: 12px;")
        tb_layout.addWidget(lbl_pre)

        self.spin_preroll = QtWidgets.QDoubleSpinBox()
        self.spin_preroll.setRange(0.0, 5.0)
        self.spin_preroll.setSingleStep(0.5)
        self.spin_preroll.setValue(1.0)
        self.spin_preroll.setSuffix(" s")
        self.spin_preroll.setStyleSheet("background-color: #1F2833; color: #FFFFFF; font-weight: bold; padding: 2px 6px;")
        self.spin_preroll.valueChanged.connect(self._on_preroll_spin_changed)
        tb_layout.addWidget(self.spin_preroll)

        self.chk_show_preroll_zone = QtWidgets.QCheckBox("Sombrear Zona Pre-Roll")
        self.chk_show_preroll_zone.setChecked(True)
        self.chk_show_preroll_zone.setStyleSheet("color: #C5C6C7; font-size: 11px;")
        self.chk_show_preroll_zone.toggled.connect(self._update_preroll_regions)
        tb_layout.addWidget(self.chk_show_preroll_zone)

        tb_layout.addStretch()
        main_layout.addWidget(toolbar)

        self.headers_container = QtWidgets.QWidget()
        self.headers_layout = QtWidgets.QHBoxLayout(self.headers_container)
        self.headers_layout.setContentsMargins(0, 0, 0, 0)
        self.headers_layout.setSpacing(4)
        main_layout.addWidget(self.headers_container)

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground("#08090C")
        main_layout.addWidget(self.graphics_widget, stretch=1)

    def _on_pause_toggled(self, paused: bool) -> None:
        self.btn_pause.setText("▶ REANUDAR" if paused else "⏸ PAUSAR")
        self.pause_requested.emit(paused)

    def set_pause_button_state(self, paused: bool) -> None:
        self.btn_pause.blockSignals(True)
        self.btn_pause.setChecked(paused)
        self.btn_pause.setText("▶ REANUDAR" if paused else "⏸ PAUSAR")
        self.btn_pause.blockSignals(False)

    def setup_channels(self, names: List[str], colors: List[str], plot_duration_s: float) -> None:
        self.channel_names = names
        self.channel_colors = colors
        self.num_channels = len(names)
        self.plot_duration_s = plot_duration_s

        while self.headers_layout.count():
            item = self.headers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.track_headers.clear()

        for i in range(self.num_channels):
            hdr = TrackControlHeader(i, names[i], colors[i])
            hdr.mute_toggled.connect(self._refresh_visibility)
            hdr.solo_toggled.connect(self._refresh_visibility)
            self.headers_layout.addWidget(hdr)
            self.track_headers.append(hdr)

        self._rebuild_plots()

    def _rebuild_plots(self) -> None:
        self.graphics_widget.clear()
        self.plots.clear()
        self.curves.clear()
        self.preroll_regions.clear()

        if self.num_channels == 0:
            return

        if not self.is_stacked_mode:
            p = self.graphics_widget.addPlot(row=0, col=0)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("bottom", "Tiempo (s)")
            p.setLabel("left", "Amplitud (µV)")
            p.getAxis("left").setWidth(60)
            p.setXRange(-self.plot_duration_s, 0, padding=0)

            reg = pg.LinearRegionItem(
                values=[-self.preroll_seconds, 0.0],
                orientation="vertical",
                brush=pg.mkBrush(0, 255, 204, 28),
                pen=pg.mkPen(0, 255, 204, 120, width=1, style=QtCore.Qt.DashLine),
                movable=False,
            )
            p.addItem(reg)
            self.preroll_regions.append(reg)
            self.plots.append(p)

            for i in range(self.num_channels):
                c = p.plot(pen=pg.mkPen(self.channel_colors[i], width=1.4), name=self.channel_names[i])
                self.curves.append(c)
        else:
            first_plot = None
            for i in range(self.num_channels):
                p = self.graphics_widget.addPlot(row=i, col=0)
                p.showGrid(x=True, y=True, alpha=0.25)
                p.setLabel("left", f"CH{i}\n(µV)", color=self.channel_colors[i])
                p.getAxis("left").setWidth(60)
                p.setXRange(-self.plot_duration_s, 0, padding=0)

                if first_plot is not None:
                    p.setXLink(first_plot)
                else:
                    first_plot = p

                if i < self.num_channels - 1:
                    p.hideAxis("bottom")
                else:
                    p.setLabel("bottom", "Tiempo (s)")

                reg = pg.LinearRegionItem(
                    values=[-self.preroll_seconds, 0.0],
                    orientation="vertical",
                    brush=pg.mkBrush(0, 255, 204, 25),
                    pen=pg.mkPen(0, 255, 204, 100, width=1, style=QtCore.Qt.DashLine),
                    movable=False,
                )
                p.addItem(reg)
                self.preroll_regions.append(reg)

                c = p.plot(pen=pg.mkPen(self.channel_colors[i], width=1.5))
                self.plots.append(p)
                self.curves.append(c)

        self._update_preroll_regions()
        self._refresh_visibility()

    def _on_layout_mode_toggled(self, checked: bool) -> None:
        self.is_stacked_mode = checked
        self.btn_toggle_layout.setText(
            "Vista: Superpuesta (Clásica)" if checked else "Vista: Pistas Separadas (REAPER)"
        )
        self._rebuild_plots()

    def _on_preroll_spin_changed(self, val: float) -> None:
        self.preroll_seconds = val
        self._update_preroll_regions()
        self.preroll_changed.emit(val)

    def _update_preroll_regions(self) -> None:
        show = self.chk_show_preroll_zone.isChecked() and (self.preroll_seconds > 0.0)
        for reg in self.preroll_regions:
            reg.setRegion([-self.preroll_seconds, 0.0])
            reg.setVisible(show)

    def _refresh_visibility(self) -> None:
        any_solo = any(h.btn_solo.isChecked() for h in self.track_headers)
        for i, hdr in enumerate(self.track_headers):
            if i >= len(self.curves):
                continue
            if any_solo:
                visible = hdr.btn_solo.isChecked() and not hdr.btn_mute.isChecked()
            else:
                visible = not hdr.btn_mute.isChecked()
            self.curves[i].setVisible(visible)
            if self.is_stacked_mode and i < len(self.plots):
                self.plots[i].setVisible(visible)

    def update_channel_metrics(self, ch_idx: int, vpp_uV: float, rms_uV: float, snr: Optional[float] = None) -> None:
        if 0 <= ch_idx < len(self.track_headers):
            txt = f"{vpp_uV:5.1f} µVp-p | {rms_uV:4.1f} µVrms"
            if snr is not None:
                txt += f" | SNR: {snr:.1f}"
            self.track_headers[ch_idx].lbl_metrics.setText(txt)
