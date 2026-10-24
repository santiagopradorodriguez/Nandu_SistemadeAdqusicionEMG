# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Gestor de cola programable para procesamiento por lotes de reportes.
# ==============================================================================

import os
import sys
import json
import time
from datetime import datetime
from collections import OrderedDict

# Asegurar path de importación
script_dir = os.path.dirname(os.path.abspath(__file__))
emg_root = os.path.dirname(script_dir)
if emg_root not in sys.path:
    sys.path.insert(0, emg_root)

from analysis.report_engine import ReportEngine


class ReportTask:
    """Representa una tarea individual programada de generación de reporte."""

    def __init__(self, nombre, sesion, tipo_reporte="pca", notas_dict=None, fotos=None, opciones_analisis=None):
        self.nombre = str(nombre)
        self.sesion = sesion  # Puede ser ruta absoluta, carpeta de fecha o filtro de prueba
        self.tipo_reporte = str(tipo_reporte).lower()  # 'base', 'pca', 'snr', 'espectral', 'multimodal', 'todos'
        self.notas_dict = dict(notas_dict or {})
        self.fotos = list(fotos or [])
        self.opciones_analisis = dict(opciones_analisis or {})
        self.estado = "pendiente"  # 'pendiente', 'en_proceso', 'completado', 'error'
        self.resultado = {}

    def to_dict(self):
        """Serializa la tarea a un diccionario JSON."""
        return {
            "nombre": self.nombre,
            "sesion": self.sesion,
            "tipo_reporte": self.tipo_reporte,
            "notas_dict": self.notas_dict,
            "fotos": self.fotos,
            "opciones_analisis": self.opciones_analisis,
            "estado": self.estado,
            "resultado": self.resultado
        }

    @classmethod
    def from_dict(cls, data):
        """Crea una tarea desde un diccionario serializado."""
        task = cls(
            nombre=data.get("nombre", "Tarea sin nombre"),
            sesion=data.get("sesion", ""),
            tipo_reporte=data.get("tipo_reporte", "pca"),
            notas_dict=data.get("notas_dict", {}),
            fotos=data.get("fotos", []),
            opciones_analisis=data.get("opciones_analisis", {})
        )
        task.estado = data.get("estado", "pendiente")
        task.resultado = data.get("resultado", {})
        return task

    def resolver_rutas_tomas(self, base_dir=None):
        """
        Resuelve 'self.sesion' en una lista de rutas absolutas de tomas individuales
        que contienen la carpeta 'canal_0'.
        """
        if base_dir is None:
            repo_root = os.path.abspath(os.path.join(emg_root, ".."))
            base_dir = os.path.join(emg_root, "base_de_datos_electrodos")

        # Si ya es una lista explícita de rutas
        if isinstance(self.sesion, (list, tuple)):
            rutas_validas = []
            for item in self.sesion:
                p = os.path.abspath(item)
                if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                    rutas_validas.append(p)
            return sorted(rutas_validas)

        sesion_str = str(self.sesion).strip()
        filtro_subcadena = self.opciones_analisis.get("filtro_prueba", "")

        # Caso 1: Ruta absoluta directa
        if os.path.isabs(sesion_str) and os.path.isdir(sesion_str):
            target_dir = sesion_str
        else:
            # Caso 2: Ruta relativa respecto a base_dir o dentro de fecha
            target_dir = os.path.join(base_dir, sesion_str)
            if not os.path.isdir(target_dir):
                # Probar si es 'fecha/filtro'
                partes = sesion_str.split(os.sep)
                if len(partes) >= 2:
                    fecha_candidate = os.path.join(base_dir, partes[0])
                    if os.path.isdir(fecha_candidate):
                        target_dir = fecha_candidate
                        if not filtro_subcadena:
                            filtro_subcadena = "_".join(partes[1:])

        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"No se encontró el directorio de sesión: {target_dir}")

        # Si el directorio mismo es una toma individual
        if os.path.isdir(os.path.join(target_dir, "canal_0")):
            return [os.path.abspath(target_dir)]

        # Explorar carpetas hijas en target_dir
        tomas_encontradas = []
        for entry in sorted(os.listdir(target_dir)):
            full_p = os.path.join(target_dir, entry)
            if not os.path.isdir(full_p) or entry.startswith('.'):
                continue
            if "secuencia" in entry.lower() or "calibracion" in entry.lower() or entry.lower().endswith(".csv"):
                continue
            if filtro_subcadena:
                if isinstance(filtro_subcadena, (list, tuple)):
                    if not any(str(f).lower() in entry.lower() for f in filtro_subcadena):
                        continue
                elif str(filtro_subcadena).lower() not in entry.lower():
                    continue
            if os.path.isdir(os.path.join(full_p, "canal_0")):
                tomas_encontradas.append(os.path.abspath(full_p))

        return sorted(tomas_encontradas)


class ReportQueueManager:
    """Administrador y motor de ejecución secuencial de la cola de reportes."""

    def __init__(self, output_dir=None):
        self.tasks = []
        if output_dir is None:
            repo_root = os.path.abspath(os.path.join(emg_root, ".."))
            self.output_dir = os.path.join(repo_root, "reportes_experimentos")
        else:
            self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def add_task(self, task):
        """Añade una tarea a la cola."""
        if not isinstance(task, ReportTask):
            raise TypeError("La tarea debe ser una instancia de ReportTask.")
        self.tasks.append(task)

    def remove_task(self, index):
        """Elimina una tarea por índice."""
        if 0 <= index < len(self.tasks):
            return self.tasks.pop(index)
        return None

    def clear(self):
        """Limpia todas las tareas de la cola."""
        self.tasks.clear()

    def count(self):
        """Retorna la cantidad de tareas encoladas."""
        return len(self.tasks)

    def save_to_json(self, filepath):
        """Guarda la cola de tareas en un archivo JSON."""
        data = {
            "version": "1.0",
            "descripcion": "Cola de tareas programables de reportes - Sistema Ñandú EMG",
            "fecha_guardado": datetime.now().isoformat(),
            "output_dir": self.output_dir,
            "tareas": [t.to_dict() for t in self.tasks]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_from_json(self, filepath):
        """Carga tareas desde un archivo JSON a la cola."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo de cola no encontrado: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tareas_data = data.get("tareas", [])
        cargadas = []
        for t_dict in tareas_data:
            t = ReportTask.from_dict(t_dict)
            self.add_task(t)
            cargadas.append(t)
        return cargadas

    def ejecutar_tarea(self, task, logger=print):
        """Ejecuta una tarea individual respetando sus parámetros específicos."""
        task.estado = "en_proceso"
        t_start = time.time()
        
        logger(f"\n=======================================================")
        logger(f"EJECUTANDO TAREA: {task.nombre}")
        logger(f"Tipo de Reporte: {task.tipo_reporte.upper()}")
        logger(f"Sesión/Filtro: {task.sesion}")
        logger(f"=======================================================")

        try:
            session_paths = task.resolver_rutas_tomas()
            if not session_paths:
                raise ValueError(f"No se encontraron tomas para '{task.sesion}' con los filtros especificados.")
            
            logger(f"Tomas resueltas ({len(session_paths)} en total):")
            for p in session_paths[:5]:
                logger(f"  - {os.path.basename(p)}")
            if len(session_paths) > 5:
                logger(f"  ... y {len(session_paths) - 5} tomas más.")

            # Preparar diccionario de notas y opciones
            notes_dict = dict(task.notas_dict)
            notes_dict['fotos'] = list(task.fotos)
            
            # Integrar opciones de análisis directamente en notes_dict para que ReportEngine las absorba
            opciones = dict(task.opciones_analisis)
            for k_opt in [
                'canales_features', 'canales_pca', 'pesos_canales',
                'lowpass_cutoff_hz', 'highpass_cutoff_hz',
                'correccion_impedancia', 'aplicar_correccion_intersesion',
                'ejecutar_grid', 'report_tag'
            ]:
                if k_opt in opciones:
                    notes_dict[k_opt] = opciones[k_opt]
            if 'canales_features' in opciones and 'canales_pca' not in notes_dict:
                notes_dict['canales_pca'] = opciones['canales_features']

            salida_dir_tarea = opciones.get('salida_dir') or self.output_dir
            engine = ReportEngine(output_dir=salida_dir_tarea)

            archivos_generados = []
            res_detalle = {}

            tipo = task.tipo_reporte.lower()

            if tipo == 'base':
                notes_dict['incluir_pca'] = False
                res = engine.generate_report(session_paths, notes_dict=notes_dict, logger=logger)
                res_detalle = res
                if res.get('status') == 'success' and res.get('pdf_path'):
                    archivos_generados.append(res['pdf_path'])

            elif tipo == 'pca':
                notes_dict['incluir_pca'] = True
                res = engine.generate_report(session_paths, notes_dict=notes_dict, logger=logger)
                res_detalle = res
                if res.get('status') == 'success' and res.get('pdf_path'):
                    archivos_generados.append(res['pdf_path'])

            elif tipo == 'snr':
                res = engine.generate_snr_report(session_paths, notes_dict=notes_dict, logger=logger)
                res_detalle = res
                if res.get('status') == 'success' and res.get('pdf_path'):
                    archivos_generados.append(res['pdf_path'])

            elif tipo == 'espectral':
                res = engine.generate_spectral_report(session_paths, notes_dict=notes_dict, logger=logger)
                res_detalle = res
                if res.get('status') == 'success' and res.get('pdf_path'):
                    archivos_generados.append(res['pdf_path'])

            elif tipo == 'multimodal':
                gen_list = engine.generate_multimodal_paper_figures(session_paths, logger=logger)
                res_detalle = {'status': 'success', 'figuras': gen_list}
                archivos_generados.extend(gen_list)

            elif tipo == 'todos':
                # Generar figuras multimodales primero
                logger("\n[1/4] Generando figuras multimodales paper...")
                gen_list = engine.generate_multimodal_paper_figures(session_paths, logger=logger)
                
                # Reporte con PCA
                logger("\n[2/4] Generando Reporte Técnico con PCA...")
                notes_dict['incluir_pca'] = True
                res_pca = engine.generate_report(session_paths, notes_dict=notes_dict, logger=logger)
                if res_pca.get('status') == 'success' and res_pca.get('pdf_path'):
                    archivos_generados.append(res_pca['pdf_path'])
                    
                # Reporte SNR
                logger("\n[3/4] Generando Reporte de SNR y Calidad...")
                res_snr = engine.generate_snr_report(session_paths, notes_dict=notes_dict, logger=logger)
                if res_snr.get('status') == 'success' and res_snr.get('pdf_path'):
                    archivos_generados.append(res_snr['pdf_path'])

                # Reporte Espectral
                logger("\n[4/4] Generando Reporte Espectral y PSD...")
                res_esp = engine.generate_spectral_report(session_paths, notes_dict=notes_dict, logger=logger)
                if res_esp.get('status') == 'success' and res_esp.get('pdf_path'):
                    archivos_generados.append(res_esp['pdf_path'])

                res_detalle = {
                    'status': 'success',
                    'reportes_generados': archivos_generados,
                    'figuras_multimodales': len(gen_list)
                }

            else:
                raise ValueError(f"Tipo de reporte desconocido: '{task.tipo_reporte}'")

            duracion = time.time() - t_start
            task.estado = "completado"
            task.resultado = {
                "status": "success",
                "duracion_seg": round(duracion, 2),
                "archivos": archivos_generados,
                "detalle": res_detalle,
                "error": None
            }
            logger(f"Tarea finalizada exitosamente en {duracion:.1f} s.")
            return task.resultado

        except Exception as e:
            duracion = time.time() - t_start
            task.estado = "error"
            task.resultado = {
                "status": "error",
                "duracion_seg": round(duracion, 2),
                "archivos": [],
                "error": str(e)
            }
            logger(f"ERROR en la tarea '{task.nombre}': {e}")
            return task.resultado

    def execute_queue(self, logger=print, callback_task_start=None, callback_task_finish=None, continuar_en_error=True):
        """
        Ejecuta todas las tareas encoladas en orden secuencial.
        Retorna un informe consolidado de ejecución.
        """
        total = len(self.tasks)
        if total == 0:
            logger("La cola de tareas está vacía.")
            return {"total": 0, "exitosas": 0, "fallidas": 0, "detalles": []}

        t_global_start = time.time()
        logger(f"\n=======================================================")
        logger(f"INICIANDO EJECUCIÓN DE COLA ({total} TAREAS PROGRAMADAS)")
        logger(f"Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger(f"=======================================================\n")

        exitosas = 0
        fallidas = 0
        detalles = []

        for idx, task in enumerate(self.tasks):
            logger(f"\n>>> [TAREA {idx + 1}/{total}] {task.nombre} <<<")
            if callback_task_start:
                callback_task_start(idx, task)

            res = self.ejecutar_tarea(task, logger=logger)
            detalles.append({"indice": idx, "nombre": task.nombre, "resultado": res})

            if res.get("status") == "success":
                exitosas += 1
            else:
                fallidas += 1
                if not continuar_en_error:
                    logger(f"Deteniendo ejecución por error en tarea {idx + 1}.")
                    break

            if callback_task_finish:
                callback_task_finish(idx, task, res)

        duracion_total = time.time() - t_global_start
        logger(f"\n=======================================================")
        logger(f"RESUMEN FINAL DE COLA DE REPORTES")
        logger(f"Tareas Totales: {total} | Exitosas: {exitosas} | Fallidas: {fallidas}")
        logger(f"Tiempo Total: {duracion_total / 60.0:.2f} min ({duracion_total:.1f} s)")
        logger(f"=======================================================\n")

        summary = {
            "fecha_ejecucion": datetime.now().isoformat(),
            "total": total,
            "exitosas": exitosas,
            "fallidas": fallidas,
            "duracion_total_seg": round(duracion_total, 2),
            "detalles": detalles
        }

        # Guardar log de resumen en output_dir
        summary_path = os.path.join(self.output_dir, f"resumen_cola_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            logger(f"Resumen de ejecución guardado en: {summary_path}")
        except Exception as e:
            logger(f"Aviso al guardar resumen JSON: {e}")

        return summary
