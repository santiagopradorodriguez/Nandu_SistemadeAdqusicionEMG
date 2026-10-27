"""
Proyecto NANDU LSD - Sistema de Adquisición sEMG
Renombrado Sistemático de Fotos de Montaje por Músculos Reales y Vistas Anatómicas.
Actualiza archivos en disco, electrodos_anotados_agente.csv y registro_auditoria_detecciones.md.
"""

import os
import sys
import csv
import shutil
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
BASE_DRIVE = os.path.join(REPO_ROOT, "Candela_Organizado_Drive")
CSV_PATH = os.path.join(REPO_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/electrodos_anotados_agente.csv")
AUDIT_MD_PATH = os.path.join(REPO_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/registro_auditoria_detecciones.md")

# Mapeo manual exacto y transparente de nombres por campaña
# (nombre_actual_en_disco_o_csv -> nuevo_nombre_descriptivo_por_musculo)
MAPEO_RENOMBRES = {
    # --- 2026-09-01: Digástrico, Risorio, Orbicular ---
    ("2026-09-01_Digastrico_Risorio_Orbicularis_Candela", "01_montaje_digastrico_risorio_candela.jpeg"): "01_lateral_izq_risorio_y_mastoides.jpeg",
    ("2026-09-01_Digastrico_Risorio_Orbicularis_Candela", "01_montaje_risorio_orbicularis_lateral_candela.jpeg"): "01_lateral_izq_risorio_y_mastoides.jpeg",
    ("2026-09-01_Digastrico_Risorio_Orbicularis_Candela", "02_montaje_detalle_comisura_candela.jpeg"): "02_submentoniana_digastrico_cuello.jpeg",
    ("2026-09-01_Digastrico_Risorio_Orbicularis_Candela", "02_montaje_digastrico_cuello_candela.jpeg"): "02_submentoniana_digastrico_cuello.jpeg",
    ("2026-09-01_Digastrico_Risorio_Orbicularis_Candela", "03_montaje_orbicularis_candela.jpeg"): "03_frontal_orbicularis_bipolar.jpeg",
    ("2026-09-01_Digastrico_Risorio_Orbicularis_Candela", "04_montaje_vista_general_candela.jpeg"): "04_vista_general_orbicularis.jpeg",

    # --- 2026-09-15: Digástrico, Cigomático Mayor, Orbicular ---
    ("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", "01_montaje_cigomatico_modiolo_candela.jpeg"): "01_submentoniana_digastrico_cuello.jpeg",
    ("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", "01_montaje_digastrico_cuello_candela.jpeg"): "01_submentoniana_digastrico_cuello.jpeg",
    ("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", "02_montaje_orbicularis_lateral_candela.jpeg"): "02_oblicua_orbicularis_y_cigomatico.jpeg",
    ("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", "02_montaje_cigomatico_orbicularis_candela.jpeg"): "02_oblicua_orbicularis_y_cigomatico.jpeg",
    ("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", "03_montaje_detalle_electrodos_candela.jpeg"): "03_detalle_referencia_mastoides.jpeg",
    ("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", "03_montaje_electrodo_referencia_mastoides_candela.jpeg"): "03_detalle_referencia_mastoides.jpeg",
    ("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", "04_montaje_orbicularis_frontal_candela.jpeg"): "04_frontal_orbicularis_y_cigomatico.jpeg",
    ("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", "04_montaje_cigomatico_orbicularis_frontal_candela.jpeg"): "04_frontal_orbicularis_y_cigomatico.jpeg",

    # --- 2026-09-18: Digástrico, Modíolo, Orbicular ---
    ("2026-09-18_Digastrico_Modiolo_Orbicularis_Candela", "01_montaje_modiolo_lateral_candela.jpeg"): "01_submentoniana_triada_digastrico_modiolo_orbicular.jpeg",
    ("2026-09-18_Digastrico_Modiolo_Orbicularis_Candela", "01_montaje_digastrico_cuello_candela.jpeg"): "01_submentoniana_triada_digastrico_modiolo_orbicular.jpeg",
    ("2026-09-18_Digastrico_Modiolo_Orbicularis_Candela", "02_montaje_modiolo_orbicularis_lateral_candela.jpeg"): "02_lateral_modiolo_y_orbicular.jpeg",
    ("2026-09-18_Digastrico_Modiolo_Orbicularis_Candela", "03_montaje_orbicularis_peribucal_candela.jpeg"): "03_frontal_orbicularis_bipolar.jpeg",
    ("2026-09-18_Digastrico_Modiolo_Orbicularis_Candela", "03_montaje_orbicularis_frontal_candela.jpeg"): "03_frontal_orbicularis_bipolar.jpeg",

    # --- 2026-09-23: Masetero y Orbiculares ---
    ("2026-09-23_Config1_Masetero_y_Orbiculares_Candela", "foto_montaje_masetero_orbiculares_candela.png"): "01_composicion_masetero_y_orbiculares.png",

    # --- 2026-09-25: Exploración Modíolo, Digástrico, Mentoniano ---
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "01_masetero_modiolo_marcas.png"): "01_marcas_masetero_y_orbicular_lateral.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "02_digastrico_marca_inicial_y_corregida.png"): "02_marcas_digastrico_cuello.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "02_digastrico_marcas_cuello.png"): "02_marcas_digastrico_cuello.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "03_digastrico_detalle_cuello.png"): "03_detalle_marcas_digastrico_cuello.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "03_digastrico_marca_alargada_cuello.png"): "03_detalle_marcas_digastrico_cuello.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "04_masetero_orbicular_electrodos_perfil.png"): "04_electrodos_masetero_y_orbicular_perfil.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "05_digastrico_cable_chico_electrodos.png"): "05_electrodos_digastrico_horizontal_cuello.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "06_mentoniano_electrodos_chicos_frontal.png"): "06_electrodos_mentoniano_masetero_orbicular_frontal.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "07_mentoniano_perfil.png"): "07_electrodos_mentoniano_masetero_perfil.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "08_mentoniano_posicion_inicial.png"): "08_marcas_verdes_mentoniano_orbicular_frontal.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "08_mentoniano_marcas_verdes_frontal.png"): "08_marcas_verdes_mentoniano_orbicular_frontal.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "09_mentoniano_movido_1_5cm.png"): "09_encintado_mentoniano_movido_1_5cm.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "09_mentoniano_encintado_frontal.png"): "09_encintado_mentoniano_movido_1_5cm.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "10_masetero_config_modiolo.png"): "10_electrodos_masetero_perfil.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "11_orbicular_config_modiolo.png"): "11_encintado_modiolo_y_orbicular_lateral.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "11_modiolo_orbicular_encintado.png"): "11_encintado_modiolo_y_orbicular_lateral.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "12_masetero_config_final_buena.png"): "12_electrodos_masetero_config_final.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "13_masetero_marcas_bloque1.png"): "13_marcas_masetero_bloque1.png",
    ("2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela", "14_orbicular_marcas_bloque1.png"): "14_marcas_orbicular_bloque1.png",

    # --- 2026-09-29: Modíolo y Orbicular ---
    ("2026-09-29_Modiolo_y_Orbicularis_Candela", "01_montaje_modiolo_candela_perfil.png"): "01_perfil_modiolo.png",
    ("2026-09-29_Modiolo_y_Orbicularis_Candela", "02_montaje_orbicularis_frontal.png"): "02_frontal_orbicularis.png",
    ("2026-09-29_Modiolo_y_Orbicularis_Candela", "03_marcas_orbicularis_frontal_candela.png"): "02_frontal_orbicularis.png",
    ("2026-09-29_Modiolo_y_Orbicularis_Candela", "03_montaje_vista_general_lateral.png"): "03_vista_general_lateral.png",
    ("2026-09-29_Modiolo_y_Orbicularis_Candela", "04_montaje_detalle_electrodos_recortados.png"): "04_detalle_modiolo_comisura_perfil.png",
    ("2026-09-29_Modiolo_y_Orbicularis_Candela", "04_marcas_modiolo_comisura_perfil_candela.png"): "04_detalle_modiolo_comisura_perfil.png",
    ("2026-09-29_Modiolo_y_Orbicularis_Candela", "05_montaje_comisura_primer_plano.png"): "05_primer_plano_comisura_referencia.png",
    ("2026-09-29_Modiolo_y_Orbicularis_Candela", "05_electrodo_referencia_mastoides_candela.png"): "05_primer_plano_comisura_referencia.png",

    # --- 2026-10-20: Control Experimental Modíolo, Masetero, Orbicular ---
    ("2026-10-20_Control_Experimental_Candela", "2026-10-20_Candela_Exp1_Modiolo_Masetero_Lateral.png"): "01_exp1_lateral_izq_modiolo_masetero.png",
    ("2026-10-20_Control_Experimental_Candela", "2026-10-20_Candela_Exp1_Orbicularis_Modiolo_Frontal.png"): "02_exp1_frontal_orbicularis_y_modiolo.png",
    ("2026-10-20_Control_Experimental_Candela", "2026-10-20_Candela_Exp2_Modiolo_Masetero_Lateral.png"): "03_exp2_lateral_izq_modiolo_masetero_orbicular.png",

    # --- 2026-10-23: Experimentos 1, 2 y 3 ---
    ("2026-10-23_Exp1_Masetero_Monocanal_Candela", "exp1_masetero_perfil.png"): "01_exp1_perfil_masetero_monocanal.png",
    ("2026-10-23_Exp2_Belly_y_Orbicularis_Candela", "exp2_belly_orbicularis_01.png"): "01_exp2_submentoniana_belly_y_orbicularis_toma1.png",
    ("2026-10-23_Exp2_Belly_y_Orbicularis_Candela", "exp2_belly_orbicularis_02.png"): "02_exp2_submentoniana_belly_y_orbicularis_toma2.png",
    ("2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela", "exp3_triada_masetero_camara.png"): "01_exp3_lateral_der_masetero_y_orbicular.png",
    ("2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela", "exp3_triada_modiolo_orbi_perfil.png"): "02_exp3_perfil_izq_modiolo_y_orbicular.png",
}


def ejecutar_renombrado() -> None:
    print("Iniciando renombrado descriptivo por musculos...")

    # 1. Renombrar en disco (copiando/renombrando con seguridad)
    for (camp, orig_name), new_name in MAPEO_RENOMBRES.items():
        montaje_dir = os.path.join(BASE_DRIVE, camp, "fotos_montaje")
        if not os.path.exists(montaje_dir):
            continue

        p_orig = os.path.join(montaje_dir, orig_name)
        p_dest = os.path.join(montaje_dir, new_name)

        if os.path.exists(p_orig) and p_orig != p_dest:
            # Crear copia con nuevo nombre si no existe
            if not os.path.exists(p_dest):
                shutil.copy2(p_orig, p_dest)
                print(f"  [DISCO] {camp.split('_')[0]}: {orig_name} -> {new_name}")

    # 2. Actualizar electrodos_anotados_agente.csv
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            filas = list(reader)
            fieldnames = reader.fieldnames

        actualizadas = 0
        for r in filas:
            camp = r["Campaña"]
            foto = r["Archivo_Foto"]
            key = (camp, foto)
            if key in MAPEO_RENOMBRES:
                nuevo_nom = MAPEO_RENOMBRES[key]
                if r["Archivo_Foto"] != nuevo_nom:
                    r["Archivo_Foto"] = nuevo_nom
                    actualizadas += 1

        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filas)
        print(f"  [CSV] {actualizadas} filas actualizadas en electrodos_anotados_agente.csv")

    # 3. Actualizar registro_auditoria_detecciones.md
    if os.path.exists(AUDIT_MD_PATH):
        with open(AUDIT_MD_PATH, "r", encoding="utf-8") as f:
            lineas = f.readlines()

        nuevas_lineas = []
        for l in lineas:
            l_mod = l
            for (camp, orig_name), new_name in MAPEO_RENOMBRES.items():
                if orig_name in l_mod:
                    l_mod = l_mod.replace(orig_name, new_name)
            nuevas_lineas.append(l_mod)

        with open(AUDIT_MD_PATH, "w", encoding="utf-8") as f:
            f.writelines(nuevas_lineas)
        print(f"  [AUDITORIA] Nombres actualizados en registro_auditoria_detecciones.md")

    print("[OK] Renombrado completado con exito.")


if __name__ == "__main__":
    ejecutar_renombrado()
