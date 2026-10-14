# ==============================================================================
# Archivo: actualizar_botones_cyberpunk.py
# Descripción: Eleva la estética de los botones de autoforge_daq_experimental.py
#              a un nivel ultra profesional conservando la identidad Cyberpunk
#              (fondos oscuros, bordes de neón cian/magenta/verde/ámbar y estados LED).
# ==============================================================================

import os
import re
import sys
import py_compile

CANDIDATES = [
    os.path.join("acquisition", "autoforge_daq_experimental.py"),
    os.path.join("EMG_desarrollo", "acquisition", "autoforge_daq_experimental.py"),
    "autoforge_daq_experimental.py",
]

TARGET_PATH = next((os.path.abspath(c) for c in CANDIDATES if os.path.exists(c)), None)
if not TARGET_PATH:
    print("Error: No se encontró autoforge_daq_experimental.py")
    sys.exit(1)

CYBERPUNK_PRO_STYLES = '''
# ==============================================================================
# --- ESTILOS VISUALES CYBERPUNK PRO (FABFILTER / TEKTRONIX / REAPER) ---
# ==============================================================================

CYBERPUNK_QSS = """
QWidget {
    background-color: #050508;
    color: #00FFCC;
    font-family: 'Courier New', Courier, monospace;
    font-size: 12px;
}

/* Agrupadores tipo Rack de Laboratorio */
QGroupBox {
    background-color: #080C14;
    border: 1px solid #00FFFF;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #00FFFF;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #080C14;
    color: #00FFFF;
}

/* Etiquetas y Checkboxes */
QLabel {
    color: #00FFCC;
    font-weight: bold;
    background-color: transparent;
}
QCheckBox {
    color: #00FFCC;
    font-weight: bold;
    spacing: 6px;
    background-color: transparent;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #00FFFF;
    background-color: #0B0F19;
}
QCheckBox::indicator:hover {
    border: 1px solid #00FFCC;
    background-color: #112233;
}
QCheckBox::indicator:checked {
    background-color: #00FFFF;
    border: 1px solid #00FFFF;
}

/* Campos de entrada, Combobox y Spinbox analógicos */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, SpinBox {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1px solid #00FFFF;
    border-radius: 4px;
    padding: 3px 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-weight: bold;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover, SpinBox:hover {
    border: 1px solid #39FF14;
    color: #39FF14;
}
QComboBox QAbstractItemView {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1px solid #00FFFF;
    selection-background-color: #003344;
    selection-color: #39FF14;
}

/* Botones Estándar Cyberpunk Pro */
QPushButton {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1.5px solid #00FFFF;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: bold;
    font-family: 'Courier New', Courier, monospace;
}
QPushButton:hover {
    background-color: #00FFFF;
    color: #050508;
    border: 1.5px solid #FFFFFF;
}
QPushButton:pressed {
    background-color: #0088AA;
    color: #FFFFFF;
}
QPushButton:disabled {
    background-color: #06080C;
    color: #334455;
    border: 1.5px solid #1A2233;
}
"""

# Botones de Transporte y Comandos Maestros
BTN_START_CYBER = """
QPushButton {
    background-color: rgba(0, 255, 128, 0.12);
    color: #39FF14;
    border: 2px solid #39FF14;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 900;
    padding: 8px 16px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #39FF14;
    color: #050508;
}
"""

BTN_STOP_CYBER = """
QPushButton {
    background-color: rgba(255, 0, 85, 0.15);
    color: #FF0055;
    border: 2px solid #FF0055;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 900;
    padding: 8px 16px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #FF0055;
    color: #FFFFFF;
}
"""

BTN_REC_START_CYBER = """
QPushButton {
    background-color: #0B0F19;
    color: #00FFFF;
    border: 2px solid #00FFFF;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
}
QPushButton:hover {
    background-color: #00FFFF;
    color: #050508;
}
"""

BTN_REC_STOP_CYBER = """
QPushButton {
    background-color: #FF0055;
    color: #FFFFFF;
    border: 2px solid #FFAACC;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
}
QPushButton:hover {
    background-color: #CC0044;
}
"""

BTN_PAUSE_CYBER = """
QPushButton {
    background-color: rgba(255, 204, 0, 0.12);
    color: #FFCC00;
    border: 2px solid #FFCC00;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: rgba(255, 204, 0, 0.25);
}
QPushButton:checked {
    background-color: #FFCC00;
    color: #050508;
    border: 2px solid #FFFFFF;
}
QPushButton:disabled {
    background-color: #06080C;
    color: #333333;
    border: 1.5px solid #1A2233;
}
"""

BTN_AUTOFORGE_DISC_CYBER = """
QPushButton {
    background-color: rgba(255, 0, 255, 0.12);
    color: #FF00FF;
    border: 2px solid #FF00FF;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #FF00FF;
    color: #050508;
}
"""

BTN_AUTOFORGE_CONT_CYBER = """
QPushButton {
    background-color: rgba(170, 0, 255, 0.14);
    color: #CC44FF;
    border: 2px solid #AA00FF;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #AA00FF;
    color: #FFFFFF;
}
"""
'''


def upgrade_buttons(path: str):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Inyectar estilos Cyberpunk Pro si no están
    if "CYBERPUNK_QSS" not in code:
        code = code.replace(
            "from pyqtgraph import SpinBox, InfiniteLine",
            "from pyqtgraph import SpinBox, InfiniteLine\n" + CYBERPUNK_PRO_STYLES,
            1,
        )

    # Actualizar constantes de botones en __init__
    code = code.replace("self.BTN_START_STYLE = BTN_DAQ_START_PRO", "self.BTN_START_STYLE = BTN_START_CYBER")
    code = code.replace("self.BTN_STOP_STYLE = BTN_DAQ_STOP_PRO", "self.BTN_STOP_STYLE = BTN_STOP_CYBER")
    code = code.replace("self.BTN_REC_START_STYLE = BTN_REC_START_PRO", "self.BTN_REC_START_STYLE = BTN_REC_START_CYBER")
    code = code.replace("self.BTN_REC_STOP_STYLE = BTN_REC_STOP_PRO", "self.BTN_REC_STOP_STYLE = BTN_REC_STOP_CYBER")

    # Actualizar estilos en _setup_ui_controls de forma segura
    code = re.sub(
        r'self\.btn_pause\.setStyleSheet\("""[\s\S]*?"""\)',
        'self.btn_pause.setStyleSheet(BTN_PAUSE_CYBER)',
        code,
    )
    code = re.sub(
        r'self\.btn_autoforge\.setStyleSheet\("""[\s\S]*?"""\)',
        'self.btn_autoforge.setStyleSheet(BTN_AUTOFORGE_DISC_CYBER)',
        code,
    )
    code = re.sub(
        r'self\.btn_autoforge_continuo\.setStyleSheet\("""[\s\S]*?"""\)',
        'self.btn_autoforge_continuo.setStyleSheet(BTN_AUTOFORGE_CONT_CYBER)',
        code,
    )

    # Estilizar el botón de Terminal y Verificar DAQ
    code = re.sub(
        r'self\.btn_terminal\.setStyleSheet\([\s\S]*?\n\s*\)',
        'self.btn_terminal.setStyleSheet("QPushButton { background-color: #0B0F19; color: #39FF14; border: 1.5px solid #39FF14; padding: 7px 12px; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #39FF14; color: #050508; }")',
        code,
    )
    code = re.sub(
        r'self\.btn_check_daq\.setStyleSheet\([\s\S]*?\n\s*\)',
        'self.btn_check_daq.setStyleSheet("QPushButton { background-color: #0B0F19; color: #00FFFF; border: 1.5px solid #00FFFF; padding: 6px 10px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #00FFFF; color: #000000; }")',
        code,
    )

    # Reemplazar la hoja de estilos global por CYBERPUNK_QSS
    old_qss = re.compile(r'app\.setStyleSheet\("""\s*QWidget\s*\{[\s\S]*?\}\s*"""\)', re.DOTALL)
    if old_qss.search(code):
        code = old_qss.sub("app.setStyleSheet(CYBERPUNK_QSS)", code, count=1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    py_compile.compile(path, doraise=True)
    print("======================================================================")
    print(f" [OK] Estética de botones Cyberpunk Pro aplicada en: {path}")
    print(" [OK] Compilación verificada con py_compile sin errores")
    print("======================================================================")


if __name__ == "__main__":
    upgrade_buttons(TARGET_PATH)
