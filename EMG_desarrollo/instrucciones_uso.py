# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de interfaz gráfica que muestra las instrucciones de uso del sistema.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
instrucciones_uso.py - v6.0

Ventana de instrucciones nativa en PySide6 con estetica profesional (Cyberpunk UI).
Detalla el funcionamiento integral de la plataforma Nandu EMG v6.0.
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
    h2 { color: #00ffcc; margin-top: 30px; border-bottom: 1px solid #222; padding-bottom: 5px; }
    h3 { color: #ffaa00; margin-top: 15px; }
    p { font-size: 14px; }
    ul { font-size: 14px; }
    li { margin-bottom: 8px; }
    b { color: #ffffff; }
    code { color: #00ffcc; background-color: #111111; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', monospace; }
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
    .box { background-color: #0d0d0d; border-left: 4px solid #ff003c; padding: 10px; margin: 15px 0; }
</style>
</head>
<body>

    <h1><center>Nandu EMG v6.0 - Manual de Instrucciones y Guia de Operacion</center></h1>
    <p>Bienvenido a la plataforma cientifica integral de Adquisicion, Procesamiento Digital de Senales (DSP), Machine Learning y Deep Learning para Electromiografia de Superficie (sEMG).</p>

    <div class="box">
        <b>Regla de Oro de la Base de Datos:</b> Toda sesion se estructura deterministicamente bajo el patron <code>base_de_datos_electrodos/&lt;Fecha&gt;/&lt;Sesion&gt;/canal_0..3/</code>. El archivo <code>metadata.json</code> principal se localiza siempre en <code>canal_0/</code>.
    </div>

    <h2>PESTANA 1: Inicio y Adquisicion</h2>
    <p>El modulo de captura permite dos modalidades de operacion:</p>
    <ul>
        <li><b>Adquisicion Manual (Libre):</b> Grabacion continua multicanal con filtrado dinámico en tiempo real (Notch 50 Hz y Pasabanda 20-500 Hz), autoescala Peak-Hold y exportacion simultanea en formato WAV y CSV.</li>
        <li><b>AutoForge DAQ (Automatizado):</b> Protocolo riguroso de guiado experimental basado en maquinas de estados. Lee un diccionario de fonemas (<code>palabras.txt</code>) y ejecuta automaticamente:
            <ul>
                <li><i>Calibracion Dinamica de Ruido Basal:</i> Muestreo silencioso previo para determinar umbrales estadisticos adaptativos.</li>
                <li><i>Metronomo Audiovisual Esclavo:</i> Pauta temporal visual gigante y senales sonoras para fijar ventanas de preparacion, contraccion y reposo.</li>
                <li><i>Evaluacion de Calidad SNR:</i> Estimacion en tiempo real de la relacion senal-ruido inter-pulso para descarte preventivo de ensayos contaminados.</li>
                <li><i>Modo Secuencia Continua:</i> Grabacion ciclica del diccionario completo con autogeneracion del vector de metadatos <code>valid_words</code>.</li>
                <li><i>Modo Envolvente RMS en Vivo:</i> Visualizacion dinamica del esfuerzo muscular vectorizado.</li>
            </ul>
        </li>
    </ul>

    <h2>PESTANA 2: Visualizacion</h2>
    <p>Herramientas de exploracion grafica de alto rendimiento sin salir del entorno:</p>
    <ul>
        <li><b>Explorador de Senales (CSV):</b> Graficador de alto desempeno basado en PyQtGraph con soporte para zoom bidireccional, downsampling inteligente y aplicacion de filtros en caliente sobre senales continuas masivas.</li>
        <li><b>Historial Graficos Musculares:</b> Visualizacion de biopotenciales calibrados y filtrados.</li>
        <li><b>Visor de Electrodos (Grilla):</b> Muestra una matriz comparativa simultanea de los 4 canales fisicos para evaluar la respuesta global de los grupos musculares.</li>
        <li><b>Historial Patron Muscular:</b> Analisis topologico y perfiles de activacion por sesion.</li>
    </ul>

    <h2>PESTANA 3: Analisis y Extraccion</h2>
    <p>Procesamiento avanzado de senales y preservacion biomecanica:</p>
    <ul>
        <li><b>Procesamiento de Pulsos (Interactivo y Rapido):</b> Segmentacion de ventanas temporales, filtrado de fase cero (filtfilt) y aislamiento de activaciones mioelectricas.</li>
        <li><b>Alineacion Master-Slave:</b> Emplea el Canal 0 como referencia temporal para alinear los canales adyacentes mediante correlacion cruzada (<i>Cross-Correlation</i>), manteniendo estrictamente intactos los desfases fisiologicos y las sinergias inter-musculares.</li>
        <li><b>Analisis de Sesion y Estadisticas:</b> Extraccion de amplitudes maximas calibradas (Volts / Ohms), generacion de histogramas y exportacion del archivo estructurado <code>analisis_results.json</code>.</li>
    </ul>

    <h2>PESTANA 4: Machine Learning y Deep Learning</h2>
    <p>Pipeline completo de modelado inteligente y clasificacion:</p>
    <ul>
        <li><b>Reduccion Dimensional (PCA y UMAP):</b> Analisis de Componentes Principales lineal y proyecciones no lineales UMAP (tanto no supervisado como supervisado) para visualizacion de clusters fonatorios.</li>
        <li><b>Autoencoders Convolucionales 1D (PyTorch):</b> Redes profundas para compresion no lineal de senales, analisis de espacios latentes 2D/3D y reconstruccion temporal de gestos fonatorios.</li>
        <li><b>Clasificador XGBoost:</b> Entrenamiento y evaluacion supervisada de modelos de ensamble sobre matrices de caracteristicas mioelectricas.</li>
        <li><b>Binarizacion y Decodificacion (Metodo Trevisan):</b> Analisis biofisico de patrones discretos de disparo de unidades motoras y decodificacion continua.</li>
        <li><b>Galeria de Resultados Integrada:</b> Visor unificado con zoom interactivo para figuras complejas y visor tabular para matrices y tablas de metricas (CSV, JSON, LaTeX, TXT).</li>
    </ul>

    <h2>PESTANA 5: Historial de Resultados</h2>
    <p>Navegacion estructurada del archivo cientifico:</p>
    <ul>
        <li><b>Historial de Comparativas:</b> Acceso directo a los reportes consolidados generados en <code>analisis_comparativos/</code>.</li>
        <li><b>Historial de Sesion:</b> Exploracion cronologica y lectura de metadatos de las sesiones procesadas en <code>analisis_de_sesiones/</code>.</li>
    </ul>

    <h2>Herramientas Auxiliares y Menu Superior</h2>
    <ul>
        <li><span class="highlight">Reproductor de Audios (Mini-DAW):</span> Modulo para emitir estimulos auditivos sincronizados a traves del Canal 3 del DAQ durante protocolos experimentales.</li>
        <li><span class="highlight">Configuracion General:</span> Panel centralizado para definir parametros del DAQ (frecuencia de muestreo, canales activos, filtros), mapeo anatomico de grupos musculares (Orbicularis Oris, Depressor Anguli Oris, Mylohyoid) y preferencias visuales con persistencia JSON.</li>
        <li><span class="highlight">Extractor de Datos Tensoriales (<code>dl_data_pipeline.py</code>):</span> Procesa lotes de grabaciones, aplica remuestreo a 500 muestras, normalizacion Min-Max [0, 1] y genera tensores binarios <code>.npy</code> compatibles con <code>torch.utils.data.Dataset</code>.</li>
        <li><span class="highlight">Editor y Migrador de Mediciones:</span> Herramientas para renombrar, curar metadatos y asegurar la organizacion jerarquica por fechas de la base de datos.</li>
    </ul>

    <div class="footer">
        <b>LABORATORIO DE SISTEMAS DINAMICOS (LSD)</b><br>
        Facultad de Ciencias Exactas y Naturales (FCEyN), Universidad de Buenos Aires (UBA).<br><br>
        <b>Investigadores y Autores Principales:</b> Santiago Prado & Lucas Braunstein<br>
        Codigos preliminares: Tomas Mininni y Roman Rolla.<br><br>
        Version 6.0 — Todos los derechos reservados. Proyecto Nandu LSD.
    </div>

</body>
</html>
"""

class InstructionsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nandu LSD - Manual de Instrucciones - EMG Studio v6.0")
        self.setGeometry(200, 100, 950, 800)
        self.setStyleSheet("background-color: #000000;")
        
        # Intentar cargar icono
        self.setWindowIcon(QIcon("icono.ico"))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Titulo superior
        lbl_header = QLabel("MANUAL DEL USUARIO - VERSION 6.0")
        lbl_header.setAlignment(Qt.AlignCenter)
        lbl_header.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 22px;
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
        
        # Boton de cierre
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
