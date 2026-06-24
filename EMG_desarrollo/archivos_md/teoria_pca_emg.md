# Guía Teórica y Práctica: Análisis de Componentes Principales (PCA) en Señales EMG

Esta guía documenta los fundamentos del Análisis de Componentes Principales (PCA), cómo se inserta en nuestro flujo de trabajo de procesamiento de señales mioeléctricas, y cómo deben interpretarse sus métricas y gráficos asociados.

---

## 1. ¿Qué es la Reducción de Dimensiones?

Cuando extraemos los pulsos musculares de los canales (ej. Mylohyoid, Depressor Anguli Oris, Orbicularis Oris), obtenemos vectores de alta dimensionalidad. Si cada canal aporta 100 muestras temporales, un solo pulso vocalizado genera un punto en un **espacio de 300 dimensiones**.

El cerebro humano no puede visualizar ni razonar intuitivamente en más de 3 dimensiones espaciales. Además, los algoritmos de Machine Learning sufren lo que se conoce como *la maldición de la dimensionalidad*: a medida que aumentan las variables, la distancia relativa entre todos los puntos tiende a igualarse, dificultando la clasificación.

La **reducción de dimensiones** es el proceso matemático mediante el cual tomamos este espacio de 300 variables y lo "aplastamos" hacia un espacio mucho menor (2D, 3D o 10D), perdiendo la menor cantidad de información relevante posible. 

## 2. Teoría detrás del PCA (Principal Component Analysis)

PCA es un algoritmo algebraico que busca encontrar nuevos ejes (Componentes Principales) que maximicen la **varianza** de los datos. 

1. **Estandarización y Matriz de Covarianza**: El algoritmo analiza cómo covarían todas las 300 variables entre sí. Por ejemplo, si el canal 1 y el canal 2 suelen subir al mismo tiempo durante la vocal 'A', están altamente correlacionados.
2. **Autovectores y Autovalores (Eigenvectors/Eigenvalues)**: A través de la descomposición matemática, PCA encuentra "direcciones" en el espacio de 300 dimensiones donde los datos se esparcen más.
   - El **Primer Componente Principal (PC1)** es la dirección que explica la mayor cantidad de variabilidad pura.
   - El **Segundo Componente (PC2)** es la dirección ortogonal (perpendicular) a PC1 que explica la segunda mayor variabilidad, y así sucesivamente.

### Interpretación de Distancias en PCA
A diferencia de algoritmos topológicos como UMAP o t-SNE, **PCA preserva estrictamente las distancias globales reales**.
- Si dos puntos están muy cerca en el gráfico PCA, significa que las morfologías de sus señales musculares (la forma de la onda) son **algebraicamente casi idénticas**.
- Si dos nubes de puntos (ej. Vocal 'A' vs Vocal 'I') se solapan, significa que la activación mioeléctrica bruta no es lo suficientemente distinta como para separarse linealmente.
- **La Distancia Euclidiana manda**: La lejanía en los ejes PC1 y PC2 corresponde linealmente a diferencias de magnitud y fase en la señal original.

## 3. ¿Qué significa "Supervisado" vs "No Supervisado"?

### PCA es inherentemente No Supervisado
Matemáticamente, PCA **no mira las etiquetas** (no sabe qué pulso es 'A', 'E', 'I', 'O', 'U'). Sólo observa números y busca las direcciones de mayor varianza. Por lo tanto, si las vocales se agrupan en clusters distintos en el gráfico, es una prueba contundente de que la biología muscular subyacente es naturalmente diferente para cada vocal, sin que ningún algoritmo haya forzado esa separación.

### El rol del flag "Supervisado" en la interfaz
Cuando en nuestra interfaz elegís la opción "Supervisado" para PCA, lo que cambia **no es el algoritmo matemático de proyección** (que sigue siendo ciego), sino el **flujo de análisis posterior**:
1. **Coloración y Auditoría**: El sistema utiliza las etiquetas reales para colorear las nubes de puntos, permitiendo auditar visualmente cómo la varianza natural del músculo se correlaciona con la intención vocal del paciente.
2. **Nomenclatura**: Se añade el flag `_Sup` al nombre de la carpeta para denotar que este lote de extracción se utilizó para entrenar modelos clasificadores que sí ven las etiquetas.
*(Nota: Si enviamos estos mismos datos luego a UMAP, UMAP sí posee un modo matemáticamente supervisado donde deforma el espacio para alejar clases distintas usando la etiqueta como fuerza de gravedad).*

## 4. PCA 2D vs PCA 3D

El sistema genera gráficas tanto en dos como en tres dimensiones. ¿Cuándo usar cada una?

- **PCA 2D (PC1 vs PC2)**:
  - **Ventaja**: Es fácil de leer en papel y reportes.
  - **Uso**: Útil cuando el porcentaje de varianza explicada acumulada por PC1+PC2 es alto (ej. > 60%). Si es así, significa que mirar un gráfico 2D es casi tan bueno como mirar los 300 puntos.
  - **Riesgo**: Si la varianza retenida es baja (ej. 25%), puede parecer que dos vocales están solapadas (chocando), cuando en realidad en dimensiones superiores están totalmente separadas. Es como ver la sombra de un cilindro contra la pared: parece un rectángulo plano.

- **PCA 3D (PC1 vs PC2 vs PC3)**:
  - **Ventaja**: Añade el tercer eje de máxima varianza. Generalmente destraba clusters que en 2D parecían fusionados.
  - **Uso**: Ideal para análisis interactivo en computadora para rotar el gráfico y verificar si los grupos musculares tienen "profundidad" de separación que no se capta de frente.

## 5. El Flujo de Trabajo (Workflow) en Nuestro Sistema

El rol de PCA en el orquestador actual sigue estos pasos estrictos:

1. **Extracción y Filtrado (Corte Físico)**: Se barren los archivos buscando picos en el micrófono. Por cada pico válido que supere el umbral SNR (Signal-to-Noise Ratio), se recorta exactamente una ventana de tiempo en el Mylohyoid, Depressor Anguli Oris y Orbicularis Oris.
2. **Aplanamiento**: Cada ventana recortada se resamplea a 100 puntos y se "aplana", pegando los tres canales uno tras otro hasta formar un vector maestro de 300 columnas.
3. **Ajuste PCA**: Se introduce la inmensa matriz `[Pulsos Totales x 300]` en el motor PCA.
4. **Cálculo de Componentes**: El motor extrae las variables que dictan la varianza, calculando la "Varianza Explicada" (cuánta información real conservamos).
5. **Auditoría Vectorial**: Si se detectan anomalías o puntos sueltos ("outliers") muy lejos del clúster central en la gráfica PCA, podemos abrir el **Auditor Vectorial PCA**. Al hacer clic en un pulso específico, el auditor desanda las 300 dimensiones y nos dibuja la forma de onda temporal exacta que generó ese punto, revelando si fue un artefacto eléctrico, un ruido, o una vocalización atípica.
6. **Exportación**: El vector reducido (ya sea 10D, 20D) o el completo 300D se exporta en CSV para alimentar la siguiente etapa de Machine Learning.
