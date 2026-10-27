#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera documentación completa y explicativa en formato README.md para la carpeta
de Google Drive 'Candela_Organizado_Drive/':
1. Un README.md maestro en la raíz con la cronología, evolución de tríadas y tabla de resultados.
2. Un README.md individual dentro de cada una de las 11 carpetas de experimentos detallando
   fecha, músculos, canales, objetivos, observaciones y contenido.
"""

import os

DRIVE_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Candela_Organizado_Drive"

EXPERIMENTOS_INFO = [
    {
        "carpeta": "2026-09-01_Digastrico_Risorio_Orbicularis_Candela",
        "fecha": "2026-09-01",
        "titulo": "Experimento 2026-09-01 - Tríada Inicial: Digástrico, Zona Risorio y Orbicularis",
        "canales": [
            ("Canal 0", "Digástrico (Anterior Belly)", "Submentoniano (Apertura mandibular pura para /a/)"),
            ("Canal 1", "Zona Risorio", "Comisura labial / mejilla baja (Retracción lateral para sonrisas /e/, /i/)"),
            ("Canal 2", "Orbicularis Oris", "Esfínter peribucal (Constricción y protrusión labial para /o/, /u/)"),
            ("Canal 3", "Micrófono", "Canal acústico de sincronización temporal fonatoria")
        ],
        "objetivo": "Primera campaña de registro sistemático en el sujeto Candela probando la viabilidad de registrar la retracción de la comisura labial mediante electrodos bipolares situados sobre el trayecto del músculo risorio.",
        "hallazgos": "Se logró desacoplar claramente la apertura (/a/) de la constricción (/u/), pero el risorio evidenció un ángulo de tracción muy horizontal bajo que introdujo dispersión angular entre /e/ e /i/ a 75.1 grados.",
        "archivos_principales": "Reporte_EMG_2026-09-01.pdf, subcarpeta fotos_montaje_y_figuras/ y 22 tomas en mediciones/ (A_Prueba1_Candela a U_Prueba4_Candela)."
    },
    {
        "carpeta": "2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela",
        "fecha": "2026-09-15",
        "titulo": "Experimento 2026-09-15 - Tríada Canónica: Digástrico, Cigomático Mayor y Orbicularis",
        "canales": [
            ("Canal 0", "Digástrico (Anterior Belly)", "Submentoniano (Descenso de mandíbula)"),
            ("Canal 1", "Cigomático Mayor (Zygomaticus Major)", "Puente malar alto a 45 grados (Elevación y tracción oblicua para /e/, /i/)"),
            ("Canal 2", "Orbicularis Oris", "Borde labial peribucal (Protrusión y esfínter)"),
            ("Canal 3", "Micrófono", "Sincronismo acústico")
        ],
        "objetivo": "Sustitución del sensor de comisura baja por el Cigomático Mayor sobre el pómulo para corregir la dispersión angular del risorio y evaluar la calidad y relación señal-a-ruido (SNR) de la tríada.",
        "hallazgos": "Éxito biomecánico rotundo: el Cigomático Mayor a 45 grados colapsó la divergencia de /e/ a solo 6.9 grados respecto al modelo de referencia, con correlación canónica rho_1 = 0.9824. Definición de la Tríada Canónica del proyecto.",
        "archivos_principales": "Reporte_de_Experimento_2026-09-15.pdf, Reporte_SNR_Y_Calidad_2026-09-15.pdf y 20 tomas en mediciones/ (Series 1 a 4)."
    },
    {
        "carpeta": "2026-09-18_Digastrico_Modiolo_Orbicularis_Candela",
        "fecha": "2026-09-18",
        "titulo": "Experimento 2026-09-18 - Exploración de Nodo Muscular: Modíolo Comisural",
        "canales": [
            ("Canal 0", "Digástrico (Anterior Belly)", "Submentoniano"),
            ("Canal 1", "Modíolo Comisural", "Nodo lateral peribucal de confluencia muscular"),
            ("Canal 2", "Orbicularis Oris", "Esfínter labial"),
            ("Canal 3", "Micrófono", "Canal acústico")
        ],
        "objetivo": "Evaluar la colocación del electrodo bipolar del Canal 1 directamente sobre el modíolo para captar simultáneamente componentes de retracción y elevación en un único nodo anatómico.",
        "hallazgos": "El modíolo demostró compartir la cinemática horizontal con correlación muy alta (rho_1 = 0.9966), permitiendo separar las 5 vocales y capturar secuencias continuas de fonación.",
        "archivos_principales": "Reporte_EMG_2026-09-18.pdf y 22 tomas en mediciones/ (incluyendo SecuenciaContinua_Prueba1_Candela)."
    },
    {
        "carpeta": "2026-09-23_Config1_Masetero_y_Orbiculares_Candela",
        "fecha": "2026-09-23 (Tarde - Serie S1 a S4)",
        "titulo": "Experimento 2026-09-23 Configuración 1 - Tríada: Masetero y Doble Sensor Labial Ortogonal",
        "canales": [
            ("Canal 0", "Masetero (Cuerpo y borde anterior)", "Puente submalar alto (3.0 a 3.5 cm debajo del canto externo del ojo)"),
            ("Canal 1", "Orbicularis Oris Vertical", "Fibras verticales del labio superior para apertura/tensión"),
            ("Canal 2", "Orbicularis Oris Horizontal", "Fibras peribucales transversales para constricción/protrusión"),
            ("Canal 3", "Micrófono", "Sincronismo acústico")
        ],
        "objetivo": "Evaluar el reemplazo del Digástrico submentoniano por el Masetero sobre la mejilla, combinado con dos dipolos ortogonales (vertical y horizontal) en el labio sin solapamiento.",
        "hallazgos": "Récord histórico de separabilidad en Candela: PCA 3D alcanzó 84.48% de exactitud macro con 100% en /I/, 93.3% en /A/ y 93.3% en /E/. La foto del montaje oficial quedó registrada en la presentación de avances.",
        "archivos_principales": "Reporte_EMG_2026-09-23_Config1_Masetero_Orbiculares.pdf, foto_montaje_2026_09_21_s1_s4.png y 20 tomas en mediciones/ (A_PruebaS1_Candela a U_PruebaS4_Candela)."
    },
    {
        "carpeta": "2026-09-23_Config2_Masetero_y_Orbicularis_Candela",
        "fecha": "2026-09-23 (Tarde - Pruebat1 y Prueba2 a 5)",
        "titulo": "Experimento 2026-09-23 Configuración 2 - Sistema Minimalista Bicanal: Masetero y Orbicularis",
        "canales": [
            ("Canal 0", "Masetero", "Segunda posición anatómica sobre mejilla media"),
            ("Canal 1", "Inactivo / Desconectado", "Canal sin electrodo"),
            ("Canal 2", "Orbicularis Oris", "Esfínter labial peribucal"),
            ("Canal 3", "Micrófono", "Canal acústico")
        ],
        "objetivo": "Comprobar si una configuración minimalista de únicamente 2 canales activos (Masetero + Orbicular) es suficiente para separar las 5 vocales del castellano.",
        "hallazgos": "Rendimiento sobresaliente de 82.70% en PCA 3D (100% en /A/, 100% en /E/ y 87.5% en /I/). El análisis monocanal reveló que el Masetero en esta posición alcanza por sí solo 77.14% en 3D.",
        "archivos_principales": "Reporte_EMG_2026-09-23_Config2_Masetero_Orbi.pdf y 19 tomas en mediciones/ (A_Pruebat1_Candela, Prueba2 a 5)."
    },
    {
        "carpeta": "2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela",
        "fecha": "2026-09-25",
        "titulo": "Experimento 2026-09-25 - Exploración Multi-Sitio con Electrodos Recortados Pequeños",
        "canales": [
            ("Config Modiolo", "Modíolo + Orbicular", "Electrodos recortados en nodo comisural"),
            ("Config Mentoniano", "Mentoniano + Orbicular", "Depresor labio inferior / borla del mentón"),
            ("Config Digástrico", "Digástrico (Nueva Belly) + Orbicular", "Reposicionamiento submentoniano"),
            ("Canal 3", "Micrófono", "Sincronismo acústico")
        ],
        "objetivo": "Barrido anatómico fino utilizando parches recortados manualmente para aislar pequeños vientres musculares faciales sin interferencia de crosstalk.",
        "hallazgos": "Se generó un registro fotográfico completo de 14 tomas del rostro con las marcas anatómicas y se mapearon las trayectorias de activación de cada subsitio.",
        "archivos_principales": "Reporte_Control_Experimental_2026-09-25.pdf, fotos_montaje_originales/ (14 fotos), fotos_montaje_y_figuras/ y 56 tomas en mediciones/."
    },
    {
        "carpeta": "2026-09-29_Modiolo_y_Orbicularis_Candela",
        "fecha": "2026-09-29",
        "titulo": "Experimento 2026-09-29 - Configuración Bicanal Optimizada: Modíolo Lateral y Orbicularis",
        "canales": [
            ("Canal 0", "Orbicularis Oris", "Filtrum superior peribucal"),
            ("Canal 1", "Modíolo Lateral", "Comisura hacia afuera/atrás sobre borde de mejilla"),
            ("Canal 2", "Inactivo / Desconectado", "Sin electrodo"),
            ("Canal 3", "Micrófono", "Sincronismo acústico")
        ],
        "objetivo": "Medición con parches Medi-Trace recortados a ~8 mm con estricta separación física para evitar solapamientos mecánicos, evaluando separación no supervisada por PCA.",
        "hallazgos": "La orientación lateral del modíolo permitió alcanzar 71.32% de exactitud macro en PCA 3D (87.0% en /A/, 84.0% en /E/ y 79.2% en /I/), sirviendo como patrón de referencia de modíolo libre.",
        "archivos_principales": "Reporte_EMG_2026-09-29.pdf (con sección PCA integrada), fotos_montaje_y_figuras/ y 15 tomas en mediciones/ (A_Prueba1_Candela a U_Prueba3_Candela)."
    },
    {
        "carpeta": "2026-10-20_Control_Experimental_Candela",
        "fecha": "2026-10-20",
        "titulo": "Experimento 2026-10-20 - Control Experimental Multi-Sujeto y Repetibilidad",
        "canales": [
            ("Exp 1", "Tríada Masetero + Modiolo + Orbicular", "Sesión control de Candela"),
            ("Exp 2", "Tríada Repetibilidad Inter-Día", "Pruebas cruzadas"),
            ("Canal 3", "Micrófono", "Sincronismo acústico")
        ],
        "objetivo": "Sesión de control experimental estricto midiendo simultáneamente a Candela y Santi para auditar derivas de impedancia, calidad de la tierra y repetibilidad.",
        "hallazgos": "Se verificó la estabilidad del piso de ruido tisular y se compilaron tanto el reporte de control experimental como la síntesis técnica general.",
        "archivos_principales": "reporte_control_experimental_2026_10_20.pdf, sintesis_experimento_2026_10_20.pdf, fotos_montaje_y_figuras/ y tomas de Candela en mediciones/."
    },
    {
        "carpeta": "2026-10-23_Exp1_Masetero_Monocanal_Candela",
        "fecha": "2026-10-23 (Mañana - 11:50 a 11:57 hs)",
        "titulo": "Experimento 2026-10-23 Exp 1 - Búsqueda Exploratoria Monocanal de Masetero",
        "canales": [
            ("Canal 0", "Masetero", "Parches semi-grandes recortados a la mitad (24 mm, IED = 2.9 cm)"),
            ("Canal 1", "Inactivo / Desconectado", "Sin electrodo"),
            ("Canal 2", "Inactivo / Desconectado", "Sin electrodo"),
            ("Canal 3", "Micrófono", "Sincronismo acústico (Tierra en clavícula)")
        ],
        "objetivo": "Barrido de colocación sobre el vientre del masetero hasta hallar la posición que capture la apertura mandibular pura de /a/ sin canibalizar las sonrisas.",
        "hallazgos": "El canal monocanal de masetero alcanzó 62.51% en PCA 2D y 58.90% en PCA 3D, aislando /I/ al 100%, /O/ al 83.33% y /U/ al 60.0%.",
        "archivos_principales": "Reporte_EMG_2026-10-23_Exp1_Masetero_Monocanal.pdf, registro_orden_autograbado_discreto_Prueba1_Sujeto1.txt, fotos_montaje_originales/ y 13 tomas en mediciones/."
    },
    {
        "carpeta": "2026-10-23_Exp2_Belly_y_Orbicularis_Candela",
        "fecha": "2026-10-23 (Mediodía - 12:00 a 12:15 hs)",
        "titulo": "Experimento 2026-10-23 Exp 2 - Comparativa Bicanal: Anterior Belly y Orbicularis",
        "canales": [
            ("Canal 0", "Orbicularis Oris", "Labio superior peribucal"),
            ("Canal 1", "Inactivo / Desconectado", "Sin electrodo"),
            ("Canal 2", "Digástrico (Anterior Belly)", "Submentoniano recortado (amplificación x2.5 por software)"),
            ("Canal 3", "Micrófono", "Sincronismo acústico")
        ],
        "objetivo": "Prueba comparativa directa de contraste entre el Digástrico submentoniano y el Orbicular peribucal utilizando parches recortados a 24 mm.",
        "hallazgos": "La configuración bicanal alcanzó 50.00% en PCA 2D y 3D, confirmando la necesidad de incorporar un tercer canal de comisura/masetero para romper la ambigüedad entre /e/ y /i/.",
        "archivos_principales": "Reporte_EMG_2026-10-23_Exp2_Belly_Orbi_x25.pdf, registro_orden_autograbado_discreto_pruebabelly_Sujeto1.txt, fotos_montaje_originales/ y 10 tomas en mediciones/."
    },
    {
        "carpeta": "2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela",
        "fecha": "2026-10-23 (Tarde - 15:43 a 15:59 hs)",
        "titulo": "Experimento 2026-10-23 Exp 3 - Tríada Fijada a 6000 Hz: Masetero, Modíolo y Orbicularis",
        "canales": [
            ("Canal 0", "Masetero", "Parches recortados (24 mm, IED = 2.9 cm, superior a 2.5 cm del lóbulo)"),
            ("Canal 1", "Modíolo", "Electrodos de 1 cm con presión manual continua para evitar pliegue"),
            ("Canal 2", "Orbicularis Oris", "Borde peribucal en esfínter labial"),
            ("Canal 3", "Micrófono", "Sincronismo acústico a 6000 Hz (Tierra en clavícula)")
        ],
        "objetivo": "Ensayo definitivo de adquisición tricanal a alta tasa de muestreo (6000 Hz) con 20 tomas (4 series completas de las 5 vocales).",
        "hallazgos": "PCA 3D alcanzó 53.37%. La auditoría fotográfica forense demostró que el modíolo se desplazó hacia el labio inferior/mentón encintado junto con el orbicular (colinealidad r = 0.903) y la tira blanca de cinta puenteó mecánicamente masetero con orbicular, reduciendo la ortogonalidad frente a septiembre.",
        "archivos_principales": "Reporte_EMG_2026-10-23_Exp3_Triada_Masetero_x25.pdf, registro_orden_autograbado_discreto_Pruebatricanal_Sujeto1.txt, fotos_montaje_originales/ y 20 tomas en mediciones/."
    }
]

def main():
    print("Generando READMEs explicativos...")

    # 1. README maestro en la raíz
    readme_root_path = os.path.join(DRIVE_DIR, "README.md")
    with open(readme_root_path, "w", encoding="utf-8") as f:
        f.write("# Repositorio Oficial de Experimentos y Campañas sEMG - Sujeto Candela\n\n")
        f.write("Este repositorio contiene todas las campañas de adquisición electromiográfica facial (sEMG) realizadas con el sujeto **Candela**, organizadas cronológicamente para Google Drive.\n\n")
        f.write("Cada carpeta representa una campaña experimental independiente y contiene:\n")
        f.write("- **Reporte oficial en PDF:** Documento técnico con los datos del montaje, formas de onda calibradas, espectrogramas multimodales y análisis no supervisado de clustering (PCA 2D y 3D con algoritmo húngaro).\n")
        f.write("- **Registro fotográfico:** Fotos de los electrodos en el rostro del sujeto, marcas anatómicas y figuras del reporte.\n")
        f.write("- **Carpeta `mediciones/`:** Tomas individuales en formato estándar del proyecto (`<Vocal>_<Prueba>_Candela`), conteniendo los archivos brutos de señal (`canal_0/`, `canal_1/`, `canal_2/`, `canal_3/`) y `metadata.json` con fecha y parámetros.\n")
        f.write("- **`README.md` individual:** Resumen específico del montaje, objetivos y resultados de esa sesión.\n\n")
        
        f.write("## Cronología y Resumen de Campañas\n\n")
        f.write("| Fecha | Carpeta de Campaña | Montaje Muscular | Exactitud PCA 3D | Reporte PDF |\n")
        f.write("| :--- | :--- | :--- | :---: | :--- |\n")
        for exp in EXPERIMENTOS_INFO:
            rep = exp["archivos_principales"].split(",")[0]
            pca_str = "-"
            if "84.48%" in exp["hallazgos"]:
                pca_str = "**84.48%**"
            elif "82.70%" in exp["hallazgos"]:
                pca_str = "**82.70%**"
            elif "71.32%" in exp["hallazgos"]:
                pca_str = "71.32%"
            elif "62.51%" in exp["hallazgos"]:
                pca_str = "58.90%"
            elif "53.37%" in exp["hallazgos"]:
                pca_str = "53.37%"
            elif "50.00%" in exp["hallazgos"]:
                pca_str = "50.00%"
            f.write(f"| {exp['fecha']} | [`{exp['carpeta']}`](./{exp['carpeta']}) | {exp['canales'][0][1]} + {exp['canales'][1][1]} | {pca_str} | {rep} |\n")
        
        f.write("\n---\n\n")
        f.write("## Evolución Biomecánica de las Tríadas Musculares\n\n")
        f.write("1. **Septiembre 01 (Risorio):** Se inició con Digástrico anterior (apertura) + Risorio en comisura baja + Orbicular. El risorio tiraba muy horizontal e introducía confusión angular entre /e/ e /i/.\n")
        f.write("2. **Septiembre 15 (Cigomático Mayor):** Se sustituyó el risorio por el Cigomático Mayor sobre el pómulo (tracción a 45 grados). Las sonrisas colapsaron a su posición correcta, consolidando la Tríada Canónica.\n")
        f.write("3. **Septiembre 18 a 29 (Modíolo lateral):** Se exploró el modíolo como nodo concentrador. Cuando el electrodo se colocó libre con orientación lateral hacia la mejilla (29 de septiembre), alcanzó 71.32% de exactitud con electrodos recortados.\n")
        f.write("4. **Septiembre 23 (Masetero y Orbiculares):** Se exploró el Masetero sobre la mejilla alta. Con 3 canales alcanzó el **récord absoluto de 84.48% en 3D**, e incluso en configuración minimalista bicanal (Masetero + Orbicular) alcanzó **82.70%**.\n")
        f.write("5. **Octubre 23 (Tríada Masetero-Modíolo-Orbicular a 6000 Hz):** Se ensayó la tríada completa a 6000 Hz. El análisis forense documentó cómo el encintado conjunto del modíolo hacia el mentón acopló los canales (r = 0.903), explicando la caída a 53.37% frente a septiembre y confirmando la necesidad de independencia mecánica en los parches.\n\n")
        f.write("## Especificación Técnica de Canales\n\n")
        f.write("- **Hardware:** Amplificador de instrumentación AD620 con resistencia de ganancia soldada fija $R_G = 100\\,\\Omega \\implies G \\approx 495\\,\\text{V/V}$.\n")
        f.write("- **Canal 0:** Músculo mandibular / elevador principal.\n")
        f.write("- **Canal 1:** Músculo comisural / retractor lateral.\n")
        f.write("- **Canal 2:** Músculo esfínter labial (Orbicularis Oris).\n")
        f.write("- **Canal 3:** Micrófono de referencia fonatoria para segmentación y sincronismo.\n")
        f.write("- **Normalización:** Supremo Tricanal por pulso individual $\\max_{c} (\\max_t |x_c(t)|)$ con supresión de ruido dinámico interpulso.\n")

    print("[OK] README maestro generado en:", readme_root_path)

    # 2. README individual en cada carpeta
    for exp in EXPERIMENTOS_INFO:
        exp_dir = os.path.join(DRIVE_DIR, exp["carpeta"])
        if not os.path.isdir(exp_dir):
            continue
        readme_exp_path = os.path.join(exp_dir, "README.md")
        with open(readme_exp_path, "w", encoding="utf-8") as f:
            f.write(f"# {exp['titulo']}\n\n")
            f.write(f"- **Fecha de Medición:** {exp['fecha']}\n")
            f.write(f"- **Sujeto Experimental:** Candela\n")
            f.write(f"- **Carpeta Oficial:** `{exp['carpeta']}`\n\n")
            f.write("## 1. Configuración de Canales y Músculos Registrados\n\n")
            f.write("| Canal | Músculo Registrado | Rol Anatómico y Función |\n")
            f.write("| :--- | :--- | :--- |\n")
            for c_num, c_musc, c_rol in exp["canales"]:
                f.write(f"| **{c_num}** | **{c_musc}** | {c_rol} |\n")
            f.write("\n## 2. Objetivo del Experimento\n\n")
            f.write(f"{exp['objetivo']}\n\n")
            f.write("## 3. Principales Hallazgos y Resultados\n\n")
            f.write(f"{exp['hallazgos']}\n\n")
            f.write("## 4. Contenido de esta Carpeta\n\n")
            f.write(f"- **Reporte Técnico PDF:** Ver archivos `{exp['archivos_principales'].split(',')[0]}`.\n")
            f.write("- **Imágenes y Fotos de Montaje:** Subcarpeta `fotos_montaje_y_figuras/` (o `fotos_montaje_originales/`).\n")
            f.write("- **Tomas Bioeléctricas sEMG:** Subcarpeta `mediciones/` con todas las tomas en nomenclatura estándar `<Vocal>_<Prueba>_Candela`.\n")
            f.write("- **Descripción Textual:** Archivo `DESCRIPCION_EXPERIMENTO.txt` con la metadata inicial extraída del reporte.\n")

        print(f"[OK] README individual generado para: {exp['carpeta']}")

    print("\nTodos los READMEs fueron generados exitosamente.")

if __name__ == "__main__":
    main()
