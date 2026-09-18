import os
os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
import numpy as np
import matplotlib.pyplot as plt

def generar_graficos():
    output_dir = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/documentacion_hardware/imagenes'
    output_dir_md = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/archivos_md/imagenes'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_md, exist_ok=True)

    # -------------------------------------------------------------
    # Gráfico 1: Curvas Teóricas según Fabricante (Homólogo a Figura 18)
    # -------------------------------------------------------------
    f = np.logspace(1, 5, 500) # 10 Hz a 100 kHz
    w = 2 * np.pi * f

    # Parámetros del filtro de entrada (33 nF + 100 kOhm)
    R_in = 100e3
    C_in = 33e-9
    tau_in = R_in * C_in # 3.3 ms
    f_L_teor = 1.0 / (2 * np.pi * tau_in) # 48.23 Hz
    H_in = (1j * w * tau_in) / (1 + 1j * w * tau_in)

    # Configuración de resistencias Rg según Figura 18
    rg_configs = [
        {"rg": 33,  "color": "#4A154B", "label": r"$33\ \Omega$"},
        {"rg": 47,  "color": "#3B528B", "label": r"$47\ \Omega$"},
        {"rg": 52,  "color": "#21908C", "label": r"$52\ \Omega$"},
        {"rg": 100, "color": "#5DC863", "label": r"$100\ \Omega$"},
        {"rg": 220, "color": "#CDDC39", "label": r"$220\ \Omega$"},
    ]

    plt.figure(figsize=(9, 5.5), dpi=300)
    
    for cfg in rg_configs:
        rg = cfg["rg"]
        G_dc = 1.0 + (49400.0 / rg)
        # Polo de alta frecuencia del AD620 (ancho de banda según datasheet)
        # Para G >= 100, BW ~ 12 MHz / G (a G=100 BW=120kHz, a G=1000 BW=12kHz)
        f_H = 1.2e7 / G_dc
        H_ad620 = G_dc / (1.0 + 1j * (f / f_H))
        
        H_total = H_in * H_ad620
        mag = np.abs(H_total)

        # Curva de respuesta teórica
        plt.semilogx(f, mag, color=cfg["color"], linewidth=2.2, label=cfg["label"])
        # Línea punteada de valor teórico en banda media
        plt.axhline(G_dc, color=cfg["color"], linestyle="--", alpha=0.6, linewidth=1.2)

    plt.xlabel("Frecuencia (Hz)", fontsize=12, fontweight="bold")
    plt.ylabel(r"$V_{\mathrm{out}} / V_{\mathrm{in}}$", fontsize=13, fontweight="bold")
    plt.title("Respuesta en Frecuencia Teórica del Front-End sEMG: AD620 y Filtro RC", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.xlim(10, 100000)
    plt.ylim(-50, 2700)
    plt.legend(loc="upper left", frameon=True, fontsize=10, title=r"Resistencia $R_G$")
    plt.tight_layout()

    fig1_path = os.path.join(output_dir, "curva_respuesta_frecuencia_teorica.png")
    fig1_path_md = os.path.join(output_dir_md, "curva_respuesta_frecuencia_teorica.png")
    plt.savefig(fig1_path, dpi=300)
    plt.savefig(fig1_path_md, dpi=300)
    plt.close()
    print(f"Guardado: {fig1_path}")

    # -------------------------------------------------------------
    # Gráfico 2: Comparación Circuito Original vs Modificación Propuesta
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=300, sharex=True)

    # Subplot 1: Ganancia |H(f)| y recuperación de banda sEMG facial (20-450 Hz)
    # Original: Rg=100 ohm (G=495), Cin=33nF, Rin=100k
    G_nom = 1.0 + 49400.0 / 100.0
    f_H_100 = 1.2e7 / G_nom
    H_orig = ((1j * w * tau_in) / (1 + 1j * w * tau_in)) * (G_nom / (1 + 1j * (f / f_H_100)))

    # Modificado: Entradas directas (Cin puenteado), C_G = 100 uF en serie con Rg=100 ohm
    # Ganancia en función de frecuencia: G(s) = 1 + 49.4k / (Rg + 1/(s*C_G))
    # Para s -> 0: G(0) = 1.0. Para s -> inf: G_ac = 495
    # f_c_gain = 1 / (2*pi * Rg * C_G) = 1 / (2*pi * 100 * 100uF) = 15.91 Hz
    tau_G = 100.0 * 100e-6
    Z_G = 100.0 + 1.0 / (1j * w * 100e-6)
    G_mod_s = 1.0 + 49400.0 / Z_G
    H_mod = G_mod_s * (1.0 / (1 + 1j * (f / f_H_100)))

    ax1.semilogx(f, np.abs(H_orig), color="#D32F2F", linewidth=2.2, label=r"Original: $C_{\mathrm{in}}=33\ \mathrm{nF},\ R=100\ \mathrm{k}\Omega\ (f_c \approx 48.2\ \mathrm{Hz})$")
    ax1.semilogx(f, np.abs(H_mod), color="#1976D2", linewidth=2.2, linestyle="-", label=r"Modificado: Entradas directas $+ C_G=100\ \mu\mathrm{F}\ (f_c \approx 15.9\ \mathrm{Hz})$")
    
    # Banda de sEMG facial resaltada
    ax1.axvspan(20, 450, color="#4CAF50", alpha=0.15, label="Banda sEMG Facial (20 a 450 Hz)")
    ax1.axvline(20, color="#388E3C", linestyle=":", linewidth=1.2)
    ax1.axvline(450, color="#388E3C", linestyle=":", linewidth=1.2)
    ax1.axvline(48.23, color="#D32F2F", linestyle="--", linewidth=1.0, alpha=0.7, label=r"Corte original $f_c = 48.2\ \mathrm{Hz}$")
    ax1.axvline(15.91, color="#1976D2", linestyle="--", linewidth=1.0, alpha=0.7, label=r"Corte modificado $f_c = 15.9\ \mathrm{Hz}$")

    ax1.set_ylabel(r"Ganancia $|V_{\mathrm{out}} / V_{\mathrm{in}}|$", fontsize=11, fontweight="bold")
    ax1.set_title("Respuesta en Frecuencia: Original vs Modificado", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax1.set_ylim(-20, 560)

    # Subplot 2: Impedancia de entrada |Z_in(f)|
    # Original: Z_in = sqrt(R^2 + (1/(w*C))^2) con R=100k, C=33nF
    Z_in_orig = np.sqrt(R_in**2 + (1.0 / (w * C_in))**2)
    # Modificado: Z_in = 10 MOhm (resistencias de 10M a masa, entradas directas)
    Z_in_mod = np.ones_like(f) * 10e6

    ax2.loglog(f, Z_in_orig, color="#D32F2F", linewidth=2.2, label=r"Original: $Z_{\mathrm{in}} \approx 100\ \mathrm{k}\Omega$")
    ax2.loglog(f, Z_in_mod, color="#1976D2", linewidth=2.2, label=r"Modificado: $R_{\mathrm{bias}} = 10\ \mathrm{M}\Omega\ (Z_{\mathrm{in}} \geq 10\ \mathrm{M}\Omega)$")
    
    # Línea de degradación de electrodo (100 kOhm)
    ax2.axhline(100e3, color="#FF9800", linestyle="--", linewidth=1.5, label=r"Contacto degradado ($Z_e \approx 100\ \mathrm{k}\Omega$)")
    ax2.axhline(10e3, color="#9C27B0", linestyle=":", linewidth=1.5, label=r"Contacto óptimo ($Z_e \approx 10\ \mathrm{k}\Omega$)")

    ax2.set_xlabel("Frecuencia (Hz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Impedancia de Entrada $|Z_{\mathrm{in}}|\ (\Omega)$", fontsize=11, fontweight="bold")
    ax2.set_title("Impedancia de Entrada frente a Impedancia de Contacto de Electrodos", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True)
    ax2.set_xlim(10, 100000)
    ax2.set_ylim(1e4, 5e7)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "comparacion_respuesta_frecuencia_impedancia.png")
    fig2_path_md = os.path.join(output_dir_md, "comparacion_respuesta_frecuencia_impedancia.png")
    plt.savefig(fig2_path, dpi=300)
    plt.savefig(fig2_path_md, dpi=300)
    plt.close()
    print(f"Guardado: {fig2_path}")

if __name__ == "__main__":
    generar_graficos()
