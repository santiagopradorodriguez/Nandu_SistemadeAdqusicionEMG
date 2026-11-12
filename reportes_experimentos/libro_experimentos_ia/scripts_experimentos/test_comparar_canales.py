import pandas as pd
import numpy as np

# Cargar df_tray de la sesion
df_tray = pd.read_csv("EMG_desarrollo/base_de_datos_electrodos/2026-11-09/SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1/trayectoria_puntos.csv")

for col in ["apertura_vertical_mm", "ancho_comisuras_mm", "descenso_menton_mm"]:
    print(f"Columna {col}: min={df_tray[col].min():.2f}, max={df_tray[col].max():.2f}, std={df_tray[col].std():.2f}")
