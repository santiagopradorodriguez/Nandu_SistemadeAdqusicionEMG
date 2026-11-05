# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institucion: Laboratorio de Sistemas Dinamicos (LSD) - FCEyN, UBA
# Descripcion: Sistema de Design Tokens y Tema Visual Clinico (Neon Clinical Slate
#              y variantes Azul Cian / Rojo Neon para Main Hub y AutoForge DAQ).
# ==============================================================================

T = {
    "bg0": "#050508",
    "bg1": "#080C14",
    "bg2": "#0B0F19",
    "line": "#1E293B",
    "text": "#E2E8F0",
    "text_dim": "#94A3B8",
    "white": "#FFFFFF",
    "accent": "#00FFFF",
    "ok": "#39FF14",
    "warn": "#FFCC00",
    "err": "#FF0055",
    "err_bg": "#240A10",
    "purple": "#CC44FF",
}

THEME = T

CHANNEL_PALETTE = [
    "#FF8A4C",  # Canal 0: Naranja
    "#2DD4BF",  # Canal 1: Turquesa
    "#4ADE80",  # Canal 2: Verde
    "#F43F5E",  # Canal 3: Rojo / Microfono
    "#A855F7",  # Extra 1
    "#38BDF8",  # Extra 2
    "#FACC15",  # Extra 3
    "#FB7185",  # Extra 4
]

PLOT_ANTIALIAS = False


def load_fonts():
    """Carga fuentes opcionales del sistema si estan disponibles."""
    pass


def style_label_value(color: str = None, size: int = 12) -> str:
    c = color or T["accent"]
    return f"color: {c}; font-family: 'Consolas', 'Courier New', monospace; font-size: {size}px; font-weight: bold; background: transparent;"


def style_badge(fg: str = None, bg: str = None, border: str = None) -> str:
    c_fg = fg or T["accent"]
    c_bg = bg or T["bg1"]
    c_bd = border or c_fg
    return (
        f"color: {c_fg}; background-color: {c_bg}; border: 1px solid {c_bd}; "
        f"border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 11px;"
    )


def style_panel_card(border_color: str = None, bg_color: str = None) -> str:
    c_bd = border_color or T["line"]
    c_bg = bg_color or T["bg1"]
    return f"QFrame {{ background-color: {c_bg}; border: 1px solid {c_bd}; border-radius: 6px; }}"


def style_button_variant(variant: str = "primary") -> str:
    v = (variant or "primary").lower()
    if v == "success":
        return """
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
    if v == "danger":
        return """
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
    if v == "warning":
        return """
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
}
"""
    if v == "purple":
        return """
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
    if v == "tool":
        return """
QPushButton {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1.5px solid #00FFFF;
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: bold;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #00FFFF;
    color: #050508;
}
"""
    return """
QPushButton {
    background-color: #0B0F19;
    color: #00FFFF;
    border: 2px solid #00FFFF;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #00FFFF;
    color: #050508;
}
"""


def build_qss(theme_dict: dict = None) -> str:
    return """
QWidget {
    background-color: #050508;
    color: #00FFCC;
    font-family: 'Courier New', Courier, monospace;
    font-size: 12px;
}

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

SpinBox {
    background-color: #0B0F19;
    border: 1.5px solid #00FFFF;
    border-radius: 4px;
    min-height: 22px;
    max-height: 24px;
    padding: 0px 2px;
    margin: 1px 0px;
}
SpinBox:hover {
    border: 1.5px solid #39FF14;
}
SpinBox QLineEdit {
    border: none;
    background: transparent;
    color: #00FFCC;
    padding: 0px 3px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    font-weight: bold;
    min-height: 18px;
}
SpinBox:hover QLineEdit {
    color: #39FF14;
}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1.5px solid #00FFFF;
    border-radius: 4px;
    padding: 1px 5px;
    margin: 1px 0px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    font-weight: bold;
    min-height: 22px;
    max-height: 24px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border: 1.5px solid #39FF14;
    color: #39FF14;
}
QComboBox QAbstractItemView {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1px solid #00FFFF;
    selection-background-color: #003344;
    selection-color: #39FF14;
}

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


def get_main_hub_palette(variant: str = "azul") -> dict:
    is_red = "roj" in str(variant).lower()
    if is_red:
        return {
            "label": "Tema: Rojo Neón",
            "bg_main": "#07090e",
            "bg_card": "#0d121b",
            "bg_header": "#0b0f17",
            "border_card": "#ef4444",
            "accent": "#ef4444",
            "accent_bright": "#f87171",
            "accent_title": "#f87171",
            "bullet_inner": "#ef4444",
            "btn_daq_bg0": "#1f0910",
            "btn_daq_bg1": "#3b0a18",
            "btn_daq_border": "#ef4444",
            "btn_daq_fg": "#fca5a5",
            "btn_daq_glow": "#ef4444",
            "btn_auto_bg0": "#0b192c",
            "btn_auto_bg1": "#112a46",
            "btn_auto_border": "#38bdf8",
            "btn_auto_fg": "#7dd3fc",
            "btn_exp_bg0": "#1a0f2e",
            "btn_exp_bg1": "#2d1650",
            "btn_exp_border": "#a855f7",
            "btn_exp_fg": "#d8b4fe",
            "btn_dec_bg0": "#07221b",
            "btn_dec_bg1": "#0c3b2e",
            "btn_dec_border": "#10b981",
            "btn_dec_fg": "#6ee7b7",
        }
    return {
        "label": "Tema: Azul Cian",
        "bg_main": "#060a12",
        "bg_card": "#0b1220",
        "bg_header": "#090f1a",
        "border_card": "#00d2ff",
        "accent": "#00d2ff",
        "accent_bright": "#38bdf8",
        "accent_title": "#38bdf8",
        "bullet_inner": "#00ffcc",
        "btn_daq_bg0": "#06202e",
        "btn_daq_bg1": "#0a364d",
        "btn_daq_border": "#00ffcc",
        "btn_daq_fg": "#00ffcc",
        "btn_daq_glow": "#00ffcc",
        "btn_auto_bg0": "#0b192c",
        "btn_auto_bg1": "#132c4c",
        "btn_auto_border": "#38bdf8",
        "btn_auto_fg": "#7dd3fc",
        "btn_exp_bg0": "#19102c",
        "btn_exp_bg1": "#2a174b",
        "btn_exp_border": "#c084fc",
        "btn_exp_fg": "#e9d5ff",
        "btn_dec_bg0": "#1f0b18",
        "btn_dec_bg1": "#38112b",
        "btn_dec_border": "#f43f5e",
        "btn_dec_fg": "#fda4af",
    }


def build_main_hub_qss(variant: str = "azul") -> str:
    P = get_main_hub_palette(variant)
    return f"""
QMainWindow, QWidget {{
    background-color: {P['bg_main']};
    color: #e2e8f0;
    font-family: 'Segoe UI', 'Inter', 'Ubuntu', sans-serif;
}}
QToolBar#mainHeaderBar {{
    background-color: {P['bg_header']};
    border-bottom: 1.5px solid {P['border_card']};
    padding: 4px 12px;
    spacing: 8px;
}}
QFrame#welcomeCard {{
    background-color: {P['bg_card']};
    border: 1.5px solid {P['border_card']};
    border-radius: 14px;
}}
QFrame#consoleCard {{
    background-color: {P['bg_card']};
    border: 1px solid #1e293b;
    border-radius: 10px;
}}
QWidget#consoleHeaderBar {{
    background-color: #0f172a;
    border-top-left-radius: 9px;
    border-top-right-radius: 9px;
    border-bottom: 1px solid #1e293b;
}}
QTextEdit#orchestratorConsole {{
    background-color: #05080f;
    color: #38bdf8;
    border: none;
    border-bottom-left-radius: 9px;
    border-bottom-right-radius: 9px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 6px 10px;
}}
QPushButton#themeSwitchBtn, QToolButton#headerNavBtn {{
    background-color: #111c2d;
    color: #e2e8f0;
    border: 1px solid {P['border_card']};
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12.5px;
    font-weight: 600;
}}
QPushButton#themeSwitchBtn:hover, QToolButton#headerNavBtn:hover {{
    background-color: {P['border_card']};
    color: #05080f;
}}
QTabWidget::pane {{
    border: 1px solid #1e293b;
    background-color: {P['bg_main']};
}}
QTabBar::tab {{
    background-color: #0d1524;
    color: #94a3b8;
    border: 1px solid #1e293b;
    padding: 8px 16px;
    font-weight: 700;
    font-size: 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {P['bg_card']};
    color: {P['accent_bright']};
    border-bottom: 2px solid {P['border_card']};
}}
"""
