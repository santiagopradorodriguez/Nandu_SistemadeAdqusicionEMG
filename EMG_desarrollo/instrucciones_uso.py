# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de interfaz gráfica que muestra las instrucciones de uso del sistema.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de interfaz gráfica que muestra las instrucciones de uso del sistema.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
instrucciones_uso.py - v4.0

Ventana de instrucciones nativa en PySide6 con estética profesional (Cyberpunk UI).
Detalla todo el funcionamiento de la nueva arquitectura de EMG Studio.
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QTextBrowser, QLabel, QPushButton
)
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import Qt

INSTRUCTIONS_HTML = """
<html>
<head>
<style>
    body {
        background-color: #050505;
        color: #dddddd;
        font-family: 'Segoe UI', Arial, sans-serif;
        margin: 20px;
        line-height: 1.6;
    }
    h1 { color: #ff003c; border-bottom: 2px solid #ff003c; padding-bottom: 10px; }
    h2 { color: #00ffcc; margin-top: 30px; }
    h3 { color: #ffaa00; }
    p { font-size: 14px; }
    ul { font-size: 14px; }
    li { margin-bottom: 10px; }
    b { color: #ffffff; }
    .footer {
        margin-top: 50px;
        padding: 20px;
        background-color: #111111;
        border-top: 1px solid #333333;
        text-align: center;
        font-size: 12px;
        color: #777777;
    }
    .highlight { color: #00ffcc; font-weight: bold; }
</style>
</head>
<body>

    <h1><center>EMG Studio v4.x - Guía de Operación</center></h1>
    <p>Bienvenido al Sistema Avanzado de Adquisición y Análisis de Señales Electromiográficas (sEMG).</p>

    <h2>PASO 1: Adquisición de Señales (Tab 1)</h2>
    <p>El hub principal permite lanzar dos modalidades de adquisición:</p>
    <ul>
        <li><b>Adquisición Manual:</b> Ideal para pruebas libres. Inicia una grabación continua con semáforo y metrónomo visual. Genera archivos <code>.wav</code> y <code>.csv</code>.</li>
        <li><b>Auto-Forge (Auto-Grabado):</b> Rutina estricta y automatizada. Graba secuencias exactas de reposo y contracción. Diseñado para estandarizar bases de datos de Deep Learning.
            <br><i>*Nota: Para activar el Modo de Envolvente RMS en Tiempo Real, marca la casilla correspondiente en la interfaz de AutoForge antes de comenzar a grabar.</i>
        </li>
    </ul>

    <h2>PASO 2: Análisis Individual y Curación (Tab 2)</h2>
    <p>Desde el panel izquierdo (Gestor de Sesiones), selecciona una o múltiples mediciones.</p>
    <ul>
        <li><b>Procesamiento de Pulsos:</b> Aplica filtros Notch (50Hz) y Pasa-Banda (20-500Hz), calcula la envolvente RMS y aísla los pulsos usando las ventanas del metrónomo.</li>
        <li><b>Curación:</b> En modo interactivo, puedes descartar pulsos ruidosos o anómalos. Los resultados se guardan en <code>analisis_results.json</code> y se exportan gráficos de alta calidad (<code>pulses.png</code>).</li>
    </ul>

    <h2>PASO 3: Visualización Integrada (Tab 3)</h2>
    <p>Explora tus datos crudos y procesados sin salir del programa.</p>
    <ul>
        <li><b>Visor CSV (Natívo PyQtGraph):</b> Permite hacer zoom interactivo, downsampling automático y filtrado en tiempo real de cualquier señal en bruto.</li>
        <li><b>Visor de Electrodos:</b> Muestra una grilla con las miniaturas de todos los canales procesados para una revisión rápida.</li>
    </ul>

    <h2>PASO 4: Análisis Comparativo (Tab 4)</h2>
    <p>Selecciona varias mediciones en el gestor y presiona "Lanzar Análisis Comparativo".</p>
    <ul>
        <li><b>Master-Slave Alignment:</b> Utiliza el canal 0 como referencia temporal para alinear los pulsos de los canales adyacentes mediante <i>Cross-Correlation</i>, manteniendo intacta la sinergia muscular.</li>
    </ul>

    <h2>Herramientas Secundarias (Barra Superior)</h2>
    <ul>
        <li><span class="highlight">Reproductor de Audios (Mini-DAW):</span> Selecciona el Reproductor desde la barra de herramientas para emitir estímulos auditivos directamente a través del Canal 3 del DAQ.</li>
        <li><span class="highlight">Configuración General (NUEVO):</span> Menú centralizado para personalizar todos los aspectos del programa. Permite ajustar el mapeo de músculos por canal, elegir colores hexadecilmales, setear parámetros por defecto para la DAQ (Sample Rate, Filtro, Canales activos) y guardar estas preferencias para futuros usos.</li>
        <li><span class="highlight">Extractor de Datos (Deep Learning):</span> Recolecta todos los pulsos procesados, aplica un resampling a 500 puntos (Nyquist estandarizado), normaliza vía Min-Max, y exporta tensores listos en formato <code>.npy</code> para PyTorch.</li>
        <li><span class="highlight">Editor de Mediciones:</span> Utilidad para renombrar formalmente las carpetas de adquisición y re-rutear sus metadatos internos de manera segura.</li>
    </ul>

    <div class="footer">
        <b>ACERCA DE Y CRÉDITOS</b><br><br>
        Desarrollado integralmente para el Laboratorio de Sistemas Dinámicos (LSD).<br>
        Facultad de Ciencias Exactas y Naturales (FCEyN), Universidad de Buenos Aires (UBA).<br><br>
        <b>Autores e Investigadores Principales:</b><br>
        Santiago Prado & Lucas Braunstein<br><br>
        &copy; 2026. Todos los derechos reservados. Proyecto Ñandú LSD.
    </div>

</body>
</html>
"""

class InstructionsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ñandú LSD - Manual de Instrucciones - EMG Studio")
        self.setGeometry(200, 100, 900, 750)
        self.setStyleSheet("background-color: #000000;")
        
        # Intentar cargar icono
        self.setWindowIcon(QIcon("icono.ico"))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Título superior
        lbl_header = QLabel("MANUAL DEL USUARIO")
        lbl_header.setAlignment(Qt.AlignCenter)
        lbl_header.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 24px;
            font-weight: bold;
            color: #000000;
            background-color: #ff003c;
            padding: 10px;
            border: 2px solid #ff003c;
        """)
        layout.addWidget(lbl_header)
        
        # Visor de HTML
        self.browser = QTextBrowser()
        self.browser.setHtml(INSTRUCTIONS_HTML)
        self.browser.setOpenExternalLinks(True)
        self.browser.setStyleSheet("""
            QTextBrowser {
                background-color: #0a0a0a;
                border: 2px solid #333333;
                border-radius: 5px;
                padding: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background: #111;
                width: 14px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #ff003c;
                min-height: 20px;
                border-radius: 7px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        layout.addWidget(self.browser)
        
        # Botón de cierre
        btn_close = QPushButton("ENTENDIDO")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("""
            QPushButton {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 16px;
                font-weight: bold;
                background-color: transparent;
                color: #00ffcc;
                border: 2px solid #00ffcc;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #00ffcc;
                color: #000000;
            }
            QPushButton:pressed {
                background-color: #00aa88;
                border: 2px solid #00aa88;
            }
        """)
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

def main():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    window = InstructionsWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
