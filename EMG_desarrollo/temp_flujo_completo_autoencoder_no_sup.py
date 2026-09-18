
import json
import sys
import os
from datetime import datetime

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'/tmp/tmpiloph7uh.json', 'r') as f:
    kwargs = json.load(f)

base_dir = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos'
mediciones = ['2026-09-01/A_Prueba1_Candela', '2026-09-01/A_Prueba2_Candela', '2026-09-01/A_Prueba3_Candela', '2026-09-01/A_Prueba4_Candela', '2026-09-01/E_Prueba1_Candela', '2026-09-01/E_Prueba2_Candela', '2026-09-01/E_Prueba3_Candela', '2026-09-01/E_Prueba4_Candela', '2026-09-01/E_Prueba5_Candela', '2026-09-01/I_Prueba1_Candela', '2026-09-01/I_Prueba2_Candela', '2026-09-01/I_Prueba3_Candela', '2026-09-01/I_Prueba4_Candela', '2026-09-01/O_Prueba1_Candela', '2026-09-01/O_Prueba2_Candela', '2026-09-01/O_Prueba3_Candela', '2026-09-01/O_Prueba4_Candela', '2026-09-01/O_Prueba5_Candela', '2026-09-01/SecuenciaContinua_Prueba1_Candela', '2026-09-01/SecuenciaContinua_Prueba2_Candela', '2026-09-01/SecuenciaContinua_Prueba3_Candela', '2026-09-01/U_Prueba1_Candela', '2026-09-01/U_Prueba2_Candela', '2026-09-01/U_Prueba3_Candela', '2026-09-01/U_Prueba4_Candela']
rutas = [os.path.join(base_dir, m) for m in mediciones]

import deep_learning.motor_autoencoder_unificado as motor

modalidad = kwargs.get('modalidad', 'envolvente')
latent_dim = kwargs.get('latent_dim', 2)
epochs = kwargs.get('epochs', 150)
batch_size = kwargs.get('batch_size', 32)
lr = kwargs.get('lr', 0.002)
tipo_perdida = kwargs.get('tipo_perdida', 'mse')
gamma_sdtw = kwargs.get('gamma_sdtw', 1.0)
lambda_orto = kwargs.get('lambda_orto', 0.0)
algoritmo_clustering = kwargs.get('algoritmo_clustering', 'gmm')
p95 = kwargs.get('usar_calibracion_p95', True)
modo_align = kwargs.get('modo_alineacion', 'Pico Volumen Micrófono')

ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
carpeta_exp = os.path.join(project_root, "resultados", "resultados_autoencoder", f"procesamiento_{ts_str}_{modalidad}_{latent_dim}d")
os.makedirs(carpeta_exp, exist_ok=True)

orto_txt = f" | Ortogonalidad Latente: {lambda_orto}" if lambda_orto > 0 else ""
print("\n" + "="*70)
print(f"  FLUJO COMPLETO: AUTOENCODER NO SUPERVISADO ({modalidad.upper()} - {latent_dim}D)")
print(f"  Pérdida: {tipo_perdida.upper()} (gamma: {gamma_sdtw}){orto_txt} | Clustering: {algoritmo_clustering.upper()}")
print(f"  Carpeta de procesamiento dedicada: {carpeta_exp}")
print("="*70 + "\n")

tipo_env = kwargs.get('tipo_envolvente', 'rms')
smooth_ms = kwargs.get('smooth_ms', 100)

npz_path, n_pulsos = motor.extraer_dataset_unificado(
    rutas,
    usar_calibracion_p95=p95,
    modo_alineacion=modo_align,
    carpeta_salida=carpeta_exp,
    tipo_envolvente=tipo_env,
    smooth_ms=smooth_ms,
    alpha_ruido=kwargs.get('alpha_ruido', 1.0),
    target_len=kwargs.get('target_len', 100),
    highpass_cutoff_hz=kwargs.get('highpass_cutoff_hz', 20.0),
    lowpass_cutoff_hz=kwargs.get('lowpass_cutoff_hz', 450.0),
    notch_q=kwargs.get('notch_q', 2.0),
    outlier_contamination=kwargs.get('outlier_contamination', 0.10),
    w_canales=kwargs.get('w_canales', [1.0, 1.0, 1.0]),
    tipo_filtro_linea=kwargs.get('tipo_filtro_linea', 'adaptativo'),
    callback_log=print
)

modelo, pth = motor.entrenar_autoencoder(
    archivo_npz=npz_path,
    modalidad=modalidad,
    latent_dim=latent_dim,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    tipo_perdida=tipo_perdida,
    gamma_sdtw=gamma_sdtw,
    lambda_orto=lambda_orto,
    carpeta_salida=carpeta_exp,
    usar_custom_arch=kwargs.get('usar_custom_arch', False),
    codigo_custom_arch=kwargs.get('codigo_custom_arch', None),
    callback_log=print
)

metricas = motor.evaluar_espacio_latente(
    archivo_npz=npz_path,
    modelo=modelo,
    modalidad=modalidad,
    latent_dim=latent_dim,
    carpeta_salida=carpeta_exp,
    usar_custom_arch=kwargs.get('usar_custom_arch', False),
    codigo_custom_arch=kwargs.get('codigo_custom_arch', None),
    mostrar_grafico=True,
    algoritmo_clustering=algoritmo_clustering,
    callback_log=print
)
print(f"\nFlujo completo culminado con exito!")
print(f"Exactitud {metricas.get('algoritmo_clustering', 'Clustering')} Global: {metricas['cluster_acc']:.2f}%")
print(f"Informe grafico: {metricas['fig_path']}")
print(f"Todos los archivos organizados en: {carpeta_exp}")
if metricas and metricas.get('fig_path') and os.path.exists(metricas['fig_path']):
    motor.abrir_imagen_en_visor(metricas['fig_path'])
