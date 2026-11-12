# Instrucciones de Entrenamiento para Argon: Nuevo Dataset con Pose 3D y Desacople de Ruido

Fecha: 2026-11-09
Sesión de Entrenamiento: `SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1` (2026-11-09)
Sesión Previa de Comparación: `SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02` (2026-11-08)

---

## 1. Diagnóstico del Fenómeno Físico y Hallazgo Clave
Al evaluar la sesión de 250 pulsos a 40 BPM se obtuvieron los siguientes resultados:
- **Espacio 3D (6 canales):** 59.1% LDA global (K-Means 43.0%).
- **Espacio 4D (6 canales):** **63.0% LDA global** (K-Means 40.9%).
  - /a/: 71.7%
  - /e/: 52.2%
  - /i/: 65.2%
  - /o/: 54.3%
  - /u/: 71.7%

**Hallazgo:** Los objetivos motores extremos (/a/ apertura máxima y /u/ constricción esfinteriana máxima) se mantienen arriba en 71.7%. Sin embargo, al aumentar la velocidad articular (40 BPM con transiciones rápidas), las vocales intermedias (/e/, /o/) caen al 52–54% debido a que el sujeto co-articula más rápido y delega la sintonía fina a la lengua dentro de la cavidad bucal.

---

## 2. Nuevas Variables Registradas para Cancelar Ruido de Movimiento
En `trayectoria_puntos.csv` de la sesión del 2026-11-09 se registraron **31 nuevas columnas** que deben usarse para limpiar el ruido residual:
1. **Canales ya compensados por marco óseo craneal:**
   - `ap_rigido_mm`, `com_rigido_mm`, `mej_rigido_mm`, `mal_rigido_mm`, `mas_rigido_mm`.
   - Comparar contra los canales brutos (`ap_bruto_mm`, `com_bruto_mm`, etc.) para verificar la cancelación del cabeceo.
2. **Ángulos de orientación de la cabeza (Head Pose):**
   - `yaw_deg` (giro lateral), `pitch_deg` (cabeceo), `roll_deg` (inclinación lateral).
   - **Instrucción para la red:** Usar `yaw_deg`, `pitch_deg` y `roll_deg` como variables exógenas de desagregación o regularización para que el autoencoder proyecte a un espacio ortogonal a las rotaciones del cráneo.
3. **Puntos fiduciarios óseos directos:**
   - Pronasale (`nariz_x_px`, `nariz_y_px`, `nariz_z_rel`), puente nasal, glabela, cantos oculares y tragos auriculares.

---

## 3. Requerimientos para la Red Neuronal (Autoencoder / Clasificador)
1. **Entrenamiento en Espacio 4D:**
   - Como se demostró empíricamente, el componente PC4 aporta 4.9% de varianza ortogonal neta que desacopla la confusión entre /e/ y /o/. Configurar la dimensión latente `latent_dim = 4` (o comparar 3D vs 4D).
2. **Entrenamiento Multiescala con Residuos:**
   - Entrenar con los 6 canales en modo ratio relativo ($\Delta d / d_{\text{reposo}}$) para invarianza de escala.
   - Incluir aumentación con jitter temporal ($\pm 50$ a $\pm 120\,\text{ms}$) y variaciones de cadencia ($\pm 20\%$).
3. **Clase Explícita de Reposo / Silencio (`'---'`):**
   - Cuando la norma de energía articular se encuentre en reposo (menor a $U_{\text{bajo}}$), la red debe mapear al centroide de reposo $(0, 0, 0, 0)$ y predecir `'---'`, sin forzar la salida hacia /I/.
4. **Formato de Entrega de Modelos:**
   - Guardar checkpoints en formato `.npz` y `.pt` listos para inferencia en tiempo real en `realtime_facial_decoder_engine.py`, exportando la tabla de proyecciones latentes en `.csv`.
