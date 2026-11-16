# Resultados Completos de Auditoría, Ablación P95 y Análisis Biomecánico — Sesión 2026-11-13

**Sujeto:** Sujeto experimental (`Sujeto1`)  
**Cobertura:** 1.225 pulsos auditados, estandarización a 15 pulsos limpios por vocal y ablación de reposo basal P95.  
**Artefactos fuente:** `paquete_auditado_limpio_2026-11-13.zip`, `auditoria_pulsos_2026-11-13.csv`.

---

## 1. Hallazgos de Auditoría de Código y Carpetas

1. **`analisis_trevisan.py`:** Busca `sample_inicio/fin_ventana` en lugar de `sample_ventana_ini/fin`, desbordando $+400\,\text{ms}$ al slot siguiente y duplicando `A_Win2==Win3` e `I_Win1==Win2` en `09_F2_ProtoA`.
2. **`get_interpulse_noise`:** Aplica un corte abrupto en `5.0 * initial_noise` que genera la meseta en `Win24` de `12_F2_ProtoD`.
3. **`F1_ProtoA_Continua_Recup`:** Tenía cinco subcarpetas sin `_Rec_` que duplicaban 25 pulsos (bloqueadas en los nuevos `excluded_windows.json`).

---

## 2. Tablas Completas de Rendimiento y $K$-Vocal (Separadas Con y Sin Corrección de Impedancia P95)

### Tabla 1 — CON Corrección de Reposo Basal e Impedancia P95 (`correccion_impedancia=True`)

| Corrida / Sesión (`id_analisis`) | $N$ Crudo $\to$ Limpio | GMM 3D Crudo $\to$ Limpio (%) | LDA 3D **Train** Crudo $\to$ Limpio (%) | LDA 3D **5-Fold CV** Crudo $\to$ Limpio (%) | Dist. Media / Mín. | $K$-Vocal **GMM 3D** Limpio (`A / E / I / O / U` %) | $K$-Vocal **LDA Train** Limpio (`A / E / I / O / U` %) | $K$-Vocal **LDA 5-Fold CV** Limpio (`A / E / I / O / U` %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_F1_ProtoA_Continua_NatMet` | 128 $\to$ **75** | 48,6 $\to$ **50,7** | 60,2 $\to$ **73,3** | 57,8 $\to$ **61,3** | 2,10 / 0,34 | 80,0 / 60,0 / 13,3 / 100,0 / 0,0 | 86,7 / 46,7 / 40,0 / 93,3 / 100,0 | 66,7 / 40,0 / 13,3 / 86,7 / 100,0 |
| `02_F1_ProtoB_Continua_NatJit` | 124 $\to$ **75** | 56,3 $\to$ **49,3** | 64,5 $\to$ **73,3** | 62,1 $\to$ **73,3** | 2,03 / 0,27 | 60,0 / 53,3 / 33,3 / 100,0 / 0,0 | 66,7 / 40,0 / 73,3 / 93,3 / 93,3 | 66,7 / 40,0 / 73,3 / 93,3 / 93,3 |
| `03_F1_ProtoC_Continua_ForzMet` | 97 $\to$ **75** | 49,3 $\to$ **60,0** | 66,0 $\to$ **74,7** | 64,9 $\to$ **68,0** | 2,10 / 0,41 | 0,0 / 100,0 / 60,0 / 40,0 / 100,0 | 33,3 / 86,7 / 66,7 / 86,7 / 100,0 | 26,7 / 80,0 / 60,0 / 73,3 / 100,0 |
| `04_F1_ProtoD_Continua_ForzJit` | 84 $\to$ **70** | 54,8 $\to$ **53,6** | 85,7 $\to$ **85,7** | 81,0 $\to$ **85,7** | 2,61 / 0,80 | 73,3 / 85,7 / 9,1 / 100,0 / 0,0 | 80,0 / 100,0 / 54,5 / 86,7 / 100,0 | 73,3 / 92,9 / 54,5 / 100,0 / 100,0 |
| `05_F1_ProtoA_Auto_NatMet` | 73 $\to$ **73** | 51,0 $\to$ **51,0** | 76,7 $\to$ **76,7** | 74,0 $\to$ **74,0** | 2,17 / 0,37 | 35,7 / 100,0 / 0,0 / 92,9 / 26,7 | 50,0 / 80,0 / 66,7 / 85,7 / 100,0 | 50,0 / 80,0 / 53,3 / 85,7 / 100,0 |
| `06_F1_ProtoB_Auto_NatJit` | 85 $\to$ **72** | 50,0 $\to$ **60,2** | 70,6 $\to$ **79,2** | 68,2 $\to$ **76,4** | 2,23 / 0,28 | 35,7 / 78,6 / 73,3 / 13,3 / 100,0 | 64,3 / 64,3 / 80,0 / 86,7 / 100,0 | 57,1 / 64,3 / 73,3 / 93,3 / 92,9 |
| `07_F1_ProtoC_Auto_ForzMet` | 70 $\to$ **70** | 67,1 $\to$ **67,1** | 77,1 $\to$ **77,1** | 77,1 $\to$ **77,1** | 2,45 / 0,64 | 92,9 / 7,1 / 100,0 / 50,0 / 85,7 | 92,9 / 71,4 / 78,6 / 57,1 / 85,7 | 92,9 / 71,4 / 78,6 / 64,3 / 78,6 |
| `08_F1_ProtoD_Auto_ForzJit` | 72 $\to$ **72** | 60,0 $\to$ **60,0** | 84,7 $\to$ **84,7** | 80,6 $\to$ **80,6** | 2,59 / 0,34 | 92,9 / 0,0 / 100,0 / 100,0 / 7,1 | 100,0 / 78,6 / 66,7 / 93,3 / 85,7 | 100,0 / 71,4 / 73,3 / 73,3 / 85,7 |
| `09_F2_ProtoA_Continua_NatMet` | 92 $\to$ **75** | 44,8 $\to$ **44,0** | 59,8 $\to$ **69,3** | 56,5 $\to$ **62,7** | 2,04 / 0,45 | 20,0 / 86,7 / 13,3 / 100,0 / 0,0 | 46,7 / 66,7 / 40,0 / 93,3 / 100,0 | 26,7 / 73,3 / 20,0 / 93,3 / 100,0 |
| `10_F2_ProtoB_Continua_NatSinRitmo` | 124 $\to$ **75** | 46,4 $\to$ **49,3** | 64,5 $\to$ **72,0** | 60,5 $\to$ **68,0** | 2,14 / 0,19 | 73,3 / 20,0 / 53,3 / 100,0 / 0,0 | 66,7 / 66,7 / 46,7 / 93,3 / 86,7 | 60,0 / 60,0 / 40,0 / 93,3 / 86,7 |
| `11_F2_ProtoC_Continua_ForzMet` | 122 $\to$ **75** | 55,4 $\to$ **69,3** | 84,4 $\to$ **90,7** | 82,8 $\to$ **90,7** | 2,68 / 0,55 | 73,3 / 73,3 / 100,0 / 100,0 / 0,0 | 100,0 / 93,3 / 93,3 / 93,3 / 73,3 | 100,0 / 93,3 / 93,3 / 93,3 / 73,3 |
| `12_F2_ProtoD_Continua_ForzSinRitmo` | 125 $\to$ **75** | 59,2 $\to$ **66,7** | 82,4 $\to$ **96,0** | 80,0 $\to$ **93,3** | 2,72 / 0,66 | 100,0 / 66,7 / 66,7 / 100,0 / 0,0 | 100,0 / 93,3 / 93,3 / 100,0 / 93,3 | 100,0 / 86,7 / 93,3 / 93,3 / 93,3 |
| `13_Global_Fase1_Manana` | 733 $\to$ **582** | 45,7 $\to$ **55,2** | 65,2 $\to$ **68,7** | 62,5 $\to$ **67,4** | 1,90 / 0,38 | 36,2 / 99,1 / 1,7 / 54,2 / 84,6 | 48,3 / 66,4 / 57,4 / 78,8 / 92,3 | 47,4 / 64,7 / 53,9 / 78,0 / 92,3 |
| `14_Global_Fase2_Tarde` | 463 $\to$ **300** | 47,3 $\to$ **50,7** | 65,4 $\to$ **70,7** | 64,8 $\to$ **70,0** | 1,99 / 0,29 | 50,0 / 98,3 / 3,3 / 50,0 / 51,7 | 68,3 / 55,0 / 53,3 / 88,3 / 88,3 | 66,7 / 55,0 / 53,3 / 88,3 / 86,7 |
| `15_PCA_Maestro_Global_Calibrado` | 1196 $\to$ **882** | 48,1 $\to$ **45,5** | 62,3 $\to$ **65,6** | 62,0 $\to$ **65,4** | 1,90 / 0,37 | 40,9 / 97,7 / 1,1 / 57,3 / 30,5 | 47,7 / 58,0 / 49,7 / 81,5 / 91,0 | 47,7 / 58,0 / 48,0 / 82,0 / 91,0 |

### Tabla 2 — SIN Corrección de Reposo Basal e Impedancia P95 (`correccion_impedancia=False`)

| Corrida / Sesión (`id_analisis`) | $N$ Crudo $\to$ Limpio | GMM 3D Crudo $\to$ Limpio (%) | LDA 3D **Train** Crudo $\to$ Limpio (%) | LDA 3D **5-Fold CV** Crudo $\to$ Limpio (%) | Dist. Media / Mín. | $K$-Vocal **GMM 3D** Limpio (`A / E / I / O / U` %) | $K$-Vocal **LDA Train** Limpio (`A / E / I / O / U` %) | $K$-Vocal **LDA 5-Fold CV** Limpio (`A / E / I / O / U` %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_F1_ProtoA_Continua_NatMet` | 128 $\to$ **75** | 47,7 $\to$ **54,7** | 55,5 $\to$ **70,7** | 49,2 $\to$ **58,7** | 1,15 / 0,16 | 100,0 / 33,3 / 40,0 / 100,0 / 0,0 | 100,0 / 60,0 / 40,0 / 73,3 / 80,0 | 80,0 / 53,3 / 20,0 / 73,3 / 66,7 |
| `02_F1_ProtoB_Continua_NatJit` | 124 $\to$ **75** | 47,3 $\to$ **44,0** | 59,7 $\to$ **57,3** | 54,8 $\to$ **52,0** | 1,10 / 0,14 | 60,0 / 26,7 / 33,3 / 100,0 / 0,0 | 40,0 / 53,3 / 26,7 / 93,3 / 73,3 | 40,0 / 40,0 / 13,3 / 93,3 / 73,3 |
| `03_F1_ProtoC_Continua_ForzMet` | 97 $\to$ **75** | 54,2 $\to$ **53,3** | 64,9 $\to$ **73,3** | 63,9 $\to$ **70,7** | 1,24 / 0,19 | 66,7 / 0,0 / 73,3 / 26,7 / 100,0 | 53,3 / 80,0 / 73,3 / 60,0 / 100,0 | 46,7 / 80,0 / 66,7 / 60,0 / 100,0 |
| `04_F1_ProtoD_Continua_ForzJit` | 84 $\to$ **70** | 50,5 $\to$ **55,0** | 82,1 $\to$ **87,1** | 79,8 $\to$ **85,7** | 1,40 / 0,43 | 80,0 / 85,7 / 9,1 / 100,0 / 0,0 | 80,0 / 92,9 / 72,7 / 86,7 / 100,0 | 73,3 / 85,7 / 72,7 / 93,3 / 100,0 |
| `05_F1_ProtoA_Auto_NatMet` | 73 $\to$ **73** | 52,2 $\to$ **52,2** | 76,7 $\to$ **76,7** | 74,0 $\to$ **74,0** | 1,08 / 0,29 | 14,3 / 53,3 / 93,3 / 0,0 / 100,0 | 64,3 / 60,0 / 80,0 / 78,6 / 100,0 | 64,3 / 60,0 / 80,0 / 71,4 / 93,3 |
| `06_F1_ProtoB_Auto_NatJit` | 85 $\to$ **72** | 55,5 $\to$ **58,8** | 69,4 $\to$ **75,0** | 61,2 $\to$ **72,2** | 1,10 / 0,12 | 78,6 / 28,6 / 73,3 / 13,3 / 100,0 | 57,1 / 57,1 / 73,3 / 86,7 / 100,0 | 50,0 / 57,1 / 73,3 / 86,7 / 92,9 |
| `07_F1_ProtoC_Auto_ForzMet` | 70 $\to$ **70** | 67,1 $\to$ **67,1** | 75,7 $\to$ **75,7** | 70,0 $\to$ **70,0** | 1,25 / 0,20 | 92,9 / 7,1 / 100,0 / 50,0 / 85,7 | 92,9 / 64,3 / 78,6 / 57,1 / 85,7 | 92,9 / 57,1 / 64,3 / 57,1 / 78,6 |
| `08_F1_ProtoD_Auto_ForzJit` | 72 $\to$ **72** | 60,0 $\to$ **60,0** | 77,8 $\to$ **77,8** | 72,2 $\to$ **72,2** | 1,37 / 0,07 | 92,9 / 0,0 / 100,0 / 100,0 / 7,1 | 100,0 / 85,7 / 73,3 / 66,7 / 64,3 | 100,0 / 85,7 / 73,3 / 60,0 / 42,9 |
| `09_F2_ProtoA_Continua_NatMet` | 92 $\to$ **75** | 47,4 $\to$ **48,0** | 59,8 $\to$ **66,7** | 55,4 $\to$ **66,7** | 1,06 / 0,24 | 26,7 / 73,3 / 40,0 / 100,0 / 0,0 | 46,7 / 53,3 / 46,7 / 86,7 / 100,0 | 40,0 / 66,7 / 46,7 / 80,0 / 100,0 |
| `10_F2_ProtoB_Continua_NatSinRitmo` | 124 $\to$ **75** | 45,8 $\to$ **50,7** | 61,3 $\to$ **69,3** | 58,1 $\to$ **70,7** | 1,14 / 0,11 | 40,0 / 80,0 / 33,3 / 100,0 / 0,0 | 66,7 / 60,0 / 40,0 / 93,3 / 86,7 | 66,7 / 60,0 / 46,7 / 93,3 / 86,7 |
| `11_F2_ProtoC_Continua_ForzMet` | 122 $\to$ **75** | 61,9 $\to$ **57,3** | 77,0 $\to$ **89,3** | 75,4 $\to$ **89,3** | 1,32 / 0,26 | 46,7 / 20,0 / 100,0 / 73,3 / 46,7 | 100,0 / 93,3 / 86,7 / 86,7 / 80,0 | 100,0 / 93,3 / 86,7 / 86,7 / 80,0 |
| `12_F2_ProtoD_Continua_ForzSinRitmo` | 125 $\to$ **75** | 54,4 $\to$ **66,7** | 80,8 $\to$ **92,0** | 80,0 $\to$ **89,3** | 1,45 / 0,32 | 100,0 / 66,7 / 66,7 / 100,0 / 0,0 | 100,0 / 93,3 / 86,7 / 86,7 / 93,3 | 100,0 / 93,3 / 73,3 / 86,7 / 93,3 |
| `13_Global_Fase1_Manana` | 733 $\to$ **582** | 53,0 $\to$ **51,4** | 59,9 $\to$ **62,9** | 59,3 $\to$ **62,5** | 1,03 / 0,17 | 36,2 / 45,7 / 63,5 / 26,3 / 85,5 | 54,3 / 63,8 / 43,5 / 59,3 / 93,2 | 54,3 / 64,7 / 40,9 / 59,3 / 93,2 |
| `14_Global_Fase2_Tarde` | 463 $\to$ **300** | 46,1 $\to$ **47,7** | 66,3 $\to$ **74,3** | 65,4 $\to$ **73,3** | 1,14 / 0,21 | 50,0 / 6,7 / 76,7 / 48,3 / 56,7 | 75,0 / 50,0 / 61,7 / 91,7 / 93,3 | 75,0 / 46,7 / 60,0 / 91,7 / 93,3 |
| `15_PCA_Maestro_Global_Calibrado` | 1196 $\to$ **882** | 47,4 $\to$ **51,9** | 62,9 $\to$ **63,8** | 61,4 $\to$ **63,7** | 1,05 / 0,13 | 38,1 / 46,6 / 63,4 / 20,2 / 91,0 | 62,5 / 52,3 / 54,9 / 60,1 / 89,3 | 61,4 / 51,1 / 56,0 / 61,2 / 88,7 |

---

## 3. Desglose Biomecánico Fino en `{/A/, /E/, /I/}` (Picos por Canal y Confusiones)

`/O/` y `/U/` jamás se confunden con `{/A/, /E/, /I/}` (Cigomático apagado $\approx 0{,}00\text{--}0{,}16$ y Orbicular al máximo $\approx 0{,}92\text{--}0{,}96$). El 100 % del colapso ocurre dentro de `{/A/, /E/, /I/}`:

| Protocolo (`id_analisis`) | Vocal | Pico Medio Sin P95 `[Ch0, Ch1, Ch2]` | Pico Medio Con P95 `[Ch0, Ch1, Ch2]` | Dispersión ($\sigma$) | Predicción LDA Train ($N=15$) | Predicción LDA 5-Fold CV ($N=15$) |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `01_F1_ProtoA` (Nat + Met) | `/A/` | `[0,32, 0,94, 0,29]` | `[0,79, 1,11, 0,22]` | 0,10 | 13 A, 2 I (86,7 %) | 10 A, 3 I, 2 E (66,7 %) |
| `01_F1_ProtoA` (Nat + Met) | `/E/` | `[0,41, 0,90, 0,63]` | `[1,09, 1,14, 0,64]` | 0,14 | 7 E, 4 A, 4 I (46,7 %) | 6 E, 5 I, 4 A (40,0 %) |
| `01_F1_ProtoA` (Nat + Met) | `/I/` | `[0,47, 0,89, 0,57]` | `[1,20, 1,14, 0,55]` | 0,15 | 6 I, 5 A, 4 E (40,0 %) | 2 I, 7 A, 6 E (13,3 %) |
| `02_F1_ProtoB` (Nat + Jitter) | `/A/` | `[0,25, 0,94, 0,36]` | `[0,74, 1,13, 0,34]` | 0,06 | 10 A, 4 E, 1 I (66,7 %) | 10 A, 4 E, 1 I (66,7 %) |
| `02_F1_ProtoB` (Nat + Jitter) | `/E/` | `[0,28, 0,92, 0,51]` | `[0,83, 1,17, 0,52]` | 0,09 | 6 E, 6 A, 3 I (40,0 %) | 6 E, 6 A, 3 I (40,0 %) |
| `02_F1_ProtoB` (Nat + Jitter) | `/I/` | `[0,44, 0,91, 0,48]` | `[1,40, 1,11, 0,49]` | 0,10 | 11 I, 3 E, 1 A (73,3 %) | 11 I, 3 E, 1 A (73,3 %) |
| `03_F1_ProtoC` (Forz + Met) | `/A/` | `[0,47, 0,91, 0,49]` | `[0,97, 1,02, 0,39]` | 0,10 | 5 A, 5 E, 5 I (33,3 %) | 4 A, 6 E, 5 I (26,7 %) |
| `03_F1_ProtoC` (Forz + Met) | `/E/` | `[0,43, 0,94, 0,42]` | `[0,87, 1,21, 0,32]` | 0,07 | 13 E, 2 A (86,7 %) | 12 E, 3 A (80,0 %) |
| `03_F1_ProtoC` (Forz + Met) | `/I/` | `[0,56, 0,91, 0,69]` | `[1,12, 1,10, 0,64]` | 0,09 | 10 I, 5 A (66,7 %) | 9 I, 6 A (60,0 %) |
| `04_F1_ProtoD` (Forz + Jitter) | `/A/` | `[0,80, 0,78, 0,49]` | `[1,53, 0,87, 0,22]` | 0,16 | 12 A, 2 E, 1 I (80,0 %) | 11 A, 3 E, 1 I (73,3 %) |
| `04_F1_ProtoD` (Forz + Jitter) | `/E/` | `[0,42, 0,92, 0,41]` | `[0,59, 1,34, 0,15]` | 0,16 | 14 E (100,0 %, N=14) | 13 E, 1 I (92,9 %) |
| `04_F1_ProtoD` (Forz + Jitter) | `/I/` | `[0,55, 0,88, 0,55]` | `[0,88, 1,08, 0,34]` | 0,18 | 6 I, 3 E, 2 A (54,5 %, N=11) | 6 I, 3 E, 2 A (54,5 %) |
| `09_F2_ProtoA` (Nat + Met) | `/A/` | `[0,89, 0,64, 0,54]` | `[1,16, 1,14, 0,51]` | 0,28 | 7 A, 5 I, 3 E (46,7 %) | 4 A, 6 E, 4 I, 1 U (26,7 %) |
| `09_F2_ProtoA` (Nat + Met) | `/E/` | `[0,87, 0,69, 0,59]` | `[1,07, 1,16, 0,66]` | 0,12 | 10 E, 5 I (66,7 %) | 11 E, 4 I (73,3 %) |
| `09_F2_ProtoA` (Nat + Met) | `/I/` | `[0,86, 0,66, 0,60]` | `[1,16, 1,10, 0,67]` | 0,20 | 6 I, 7 E, 2 A (40,0 %) | 3 I, 10 E, 2 A (20,0 %) |
| `10_F2_ProtoB` (Nat SinRitmo) | `/A/` | `[0,85, 0,75, 0,67]` | `[1,05, 1,29, 0,73]` | 0,11 | 10 A, 3 E, 2 I (66,7 %) | 9 A, 4 E, 2 I (60,0 %) |
| `10_F2_ProtoB` (Nat SinRitmo) | `/E/` | `[0,89, 0,67, 0,59]` | `[1,19, 1,10, 0,71]` | 0,11 | 10 E, 3 A, 2 I (66,7 %) | 9 E, 3 A, 3 I (60,0 %) |
| `10_F2_ProtoB` (Nat SinRitmo) | `/I/` | `[0,88, 0,62, 0,72]` | `[1,13, 1,02, 0,79]` | 0,08 | 7 I, 5 E, 3 A (46,7 %) | 6 I, 5 E, 4 A (40,0 %) |
| `11_F2_ProtoC` (Forz + Met) | `/A/` | `[0,94, 0,13, 0,32]` | `[1,08, 0,14, 0,10]` | 0,08 | 15 A (100,0 %) | 15 A (100,0 %) |
| `11_F2_ProtoC` (Forz + Met) | `/E/` | `[0,94, 0,48, 0,42]` | `[1,15, 1,27, 0,28]` | 0,10 | 14 E, 1 I (93,3 %) | 14 E, 1 I (93,3 %) |
| `11_F2_ProtoC` (Forz + Met) | `/I/` | `[0,93, 0,50, 0,66]` | `[1,06, 1,34, 0,53]` | 0,10 | 14 I, 1 E (93,3 %) | 14 I, 1 E (93,3 %) |
| `12_F2_ProtoD` (Forz SinRitmo) | `/A/` | `[0,95, 0,32, 0,35]` | `[1,11, 0,38, 0,15]` | 0,09 | 15 A (100,0 %) | 15 A (100,0 %) |
| `12_F2_ProtoD` (Forz SinRitmo) | `/E/` | `[0,92, 0,80, 0,53]` | `[1,19, 1,54, 0,33]` | 0,10 | 14 E, 1 I (93,3 %) | 13 E, 2 I (86,7 %) |
| `12_F2_ProtoD` (Forz SinRitmo) | `/I/` | `[0,94, 0,67, 0,53]` | `[1,10, 1,14, 0,36]` | 0,10 | 14 I, 1 A (93,3 %) | 14 I, 1 A (93,3 %) |

---

## 4. Separabilidad de `/O/` y `/U/` (Habla Natural vs. Forzada)

- **Aislamiento total respecto a `{/A/, /E/, /I/}`:** 0 confusiones en las 12 sesiones individuales.
- **Separación mutua `/O/` vs. `/U/` en Habla Natural:** Alcanza **85,7 % a 100,0 %** en Train y CV. Como `/O/` es vocal media ($F_1 = 571\,\text{Hz}$) y `/U/` cerrada ($F_1 = 445\,\text{Hz}$), `/O/` recluta el doble de apertura mandibular en Digástrico (`Ch0`) que `/U/` (`1,05 vs 0,50` en `01`; `0,56 vs 0,27` en `09`; `0,55 vs 0,19` en `10`).
- **Rol decisivo de P95:** Sin P95, `Ch2` (Orbicular) domina la norma y aplasta `Ch0`, haciendo caer $d(O,U)$ de `0,93` a `0,17` en `01_F1_ProtoA` y de `0,61` a `0,07` en `08_F1_ProtoD_Auto`.

---

## 5. Efecto del Cambio de Metrónomo: 4 Sub-Beeps (Fase 1) vs. 1 Solo Beep / Sin Ritmo (Fase 2)

1. **Fase 1 (4 sub-beeps por ciclo):** Rampa de preparación motora anticipatoria que reduce desfases temporales ($\sigma = 0{,}06\text{--}0{,}10$), pero en Habla Forzada periódica (`03_F1_ProtoC`) mantiene co-contracción tónica del Cigomático (`0,91` en `/A/`), colapsando `/A/` al `33,3 %` Train / `26,7 %` CV. El Jitter (`04_F1_ProtoD`) rompe el automatismo y permite desacoplar Digástrico (`1,53` en `/A/` vs `0,59` en `/E/`), subiendo `/A/` a `80,0 %` Train / `73,3 %` CV.
2. **Fase 2 (1 solo beep seco en `09`/`11` vs. Sin Ritmo en `10`/`12`):** El sujeto pasa de anticipación a reacción auditivo-motora ($\sim 200\text{--}400\,\text{ms}$), concentrando desfases de ventana (`19` en `09_ProtoA`, `13` en `11_ProtoC`) y duplicando la dispersión en Habla Natural (`σ = 0,28` en `/A/`). Al desaparecer la co-contracción tónica de los sub-beeps en Habla Forzada, `/A/` alcanza `100,0 %` (`11_ProtoC` y `12_ProtoD`), logrando en **`12_F2_ProtoD` (Forzada Sin Ritmo)** el máximo rendimiento de la campaña (**96,0 % Train / 93,3 % CV**).
