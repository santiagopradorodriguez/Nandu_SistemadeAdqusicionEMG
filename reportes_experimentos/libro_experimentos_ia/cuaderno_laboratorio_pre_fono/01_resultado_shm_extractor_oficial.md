# Resultados Solla-Hopfield-Mindlin con Función Extractora Oficial (2026-09-15)

Se integró de forma completa la función oficial `extraer_y_filtrar` en el script [metodo_solla_hopfield_mindlin.py](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/libro_experimentos_ia/scripts_experimentos/metodo_solla_hopfield_mindlin.py).

## Parámetros Canónicos Oficiales Integrados
* **Función:** `extraer_y_filtrar` con `aplicar_trevisan=False` y alineación por *Pico Volumen Micrófono* (pre 40%, post 60%).
* **Filtros:** Notch IIR $Q=30$ en 50 Hz + Pasaaltos 20 Hz + Pasabajos 500 Hz.
* **Envolvente:** Suave 125 ms.
* **Longitud normalizada:** `target_length = 20` puntos por canal (dimensión $20 \times 3 = 60$).
* **Purga de artefactos:** *Isolation Forest* al 10% de contaminación.
* **Total de pulsos limpios extraídos:** exactamente **203 pulsos** (coincidencia idéntica con el Atlas).

---

## Comparativa de Separabilidad y Exactitud

| Métrica | Exactitud (%) | Notas |
| :--- | :---: | :--- |
| **PCA Clásico Oficial (Atlas sEMG)** | **85.22%** | LDA 3D sobre los 203 pulsos limpios oficiales (Varianza explicada: 91.74%). /a/=87.8%, /e/=78.1%, /i/=100%, /o/=80.5%, /u/=80.5%. |
| **PCA Clásico (Validación Cruzada 5-Fold)** | **74.91%** | Evaluación con partición ciega 80/20 sobre los mismos 203 pulsos. |
| **Solla–Mindlin (SVD + Lévy + CoM)** | **82.76%** | Espacio sinérgico motor con rotación de fase causal. |
| **Solla–Hopfield–Mindlin (+Procrustes SO(3))** | **82.78%** | Variedad alineada inter-serie + atractores de Hopfield modernos. |

---

## Figura Comparativa Oficial (6 Paneles)

![Panel Comparativo Solla-Hopfield-Mindlin](/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/Solla_Hopfield_Mindlin_6Paneles_Oficial_2026_09_15.png)

* El Panel (B) ahora muestra explícitamente el **85.2% oficial del Atlas** junto con el resultado de validación cruzada.
* Todos los puntos respetan la paleta canónica (/a/ rojo `#E63946`, /e/ azul `#1F77B4`, /i/ verde `#2CA02C`, /o/ morado `#9D4EDD`, /u/ amarillo `#E7A61A`).
