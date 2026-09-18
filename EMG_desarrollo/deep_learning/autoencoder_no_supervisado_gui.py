# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Interfaz Gráfica Unificada de Autoencoders No Supervisados (Cero Etiquetas)
#              - Selección de mediciones con auditoría estricta de metadatos inter-día.
#              - Alertas anatómicas prominentes si se mezclan músculos o BPM dispares.
#              - Soporte de 3 modalidades fisiológicas:
#                1. Envolvente 1D (RMS / Filtrada a 200 Hz).
#                2. Señal Cruda 1D (2000 Hz / Micro-dinámica GAP+GMP).
#                3. Espectrograma 2D / 3D (STFT Calibrada en dB).
#              - Entrenamiento 100% No Supervisado (Zero-Labels en función de pérdida).
#              - Evaluación Canónica (Alineación /a/ en +Y, sonrisa en +X) y métricas GMM.
# ==============================================================================

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
emg_desarrollo_dir = os.path.abspath(os.path.join(script_dir, ".."))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if emg_desarrollo_dir not in sys.path:
    sys.path.insert(0, emg_desarrollo_dir)

import motor_autoencoder_unificado as motor

class AutoencoderNoSupervisadoGUI:
    def __init__(self, root, rutas_preseleccionadas=None):
        self.root = root
        self.root.title("NANDU LSD - Autoencoder No Supervisado (Cero Etiquetas)")
        self.root.geometry("980x920")
        self.root.minsize(850, 750)
        
        # Paleta Cyberpunk LSD
        self.bg_dark = "#0B0C10"
        self.bg_panel = "#1F2833"
        self.cyan_neon = "#66FCF1"
        self.cyan_dim = "#45A29E"
        self.fg_text = "#C5C6C7"
        self.green_neon = "#00FF00"
        self.red_alert = "#FF0055"
        self.yellow_warn = "#FFE600"
        
        self.root.configure(bg=self.bg_dark)
        self.base_dir = os.path.join(emg_desarrollo_dir, "base_de_datos_electrodos")
        self.archivo_npz_actual = os.path.join(emg_desarrollo_dir, "cache_datos", "dataset_autoencoder_unificado.npz")
        self.modelo_actual = None
        self.rutas_preseleccionadas = rutas_preseleccionadas or []
        
        self.setup_ui()
        self.cargar_arbol_sesiones()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background=self.bg_panel, foreground=self.fg_text, font=("Arial", 10))
        style.configure("TFrame", background=self.bg_panel)
        style.configure("TButton", font=("Arial", 10, "bold"), background=self.cyan_dim, foreground="black")

        main_frame = tk.Frame(self.root, bg=self.bg_dark, padx=15, pady=10)
        main_frame.pack(fill="both", expand=True)

        # --- ENCABEZADO ---
        lbl_titulo = tk.Label(
            main_frame, text="ESTUDIO DE AUTOENCODERS NO SUPERVISADOS (CERO ETIQUETAS)",
            bg=self.bg_dark, fg=self.cyan_neon, font=("Arial", 15, "bold")
        )
        lbl_titulo.pack(pady=(0, 2))
        lbl_sub = tk.Label(
            main_frame, text="Descubrimiento de Variedades Bioeléctricas sin Supervisión (Envolventes | Cruda | Espectrogramas)",
            bg=self.bg_dark, fg=self.cyan_dim, font=("Arial", 10)
        )
        lbl_sub.pack(pady=(0, 8))

        # --- PANEL 1: EXPLORADOR DE SESIONES Y AUDITORÍA DE METADATOS ---
        frame_ses = tk.LabelFrame(
            main_frame, text=" 1. Selección de Mediciones y Auditoría Inter-Día ",
            bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 10, "bold"), padx=8, pady=6
        )
        frame_ses.pack(fill="x", pady=4)

        bar_sel = tk.Frame(frame_ses, bg=self.bg_panel)
        bar_sel.pack(fill="x", pady=(0, 4))
        tk.Button(bar_sel, text="Seleccionar Todo", bg="#111", fg=self.fg_text, font=("Arial", 8), command=self.seleccionar_todo).pack(side="left", padx=2)
        tk.Button(bar_sel, text="Solo Lucas (2026-07-10)", bg="#111", fg=self.cyan_neon, font=("Arial", 8), command=lambda: self.filtrar_sesion("Lucas")).pack(side="left", padx=2)
        tk.Button(bar_sel, text="Solo Candela (2026-09-01)", bg="#111", fg=self.cyan_neon, font=("Arial", 8), command=lambda: self.filtrar_sesion("Candela")).pack(side="left", padx=2)
        tk.Button(bar_sel, text="Desmarcar Todo", bg="#111", fg=self.fg_text, font=("Arial", 8), command=self.desmarcar_todo).pack(side="left", padx=2)
        tk.Button(bar_sel, text="Auditar Metadatos Ahora", bg="#2b2600", fg=self.yellow_warn, font=("Arial", 8, "bold"), command=self.ejecutar_auditoria_manual).pack(side="right", padx=2)

        f_list = tk.Frame(frame_ses, bg=self.bg_panel)
        f_list.pack(fill="x")
        scr = tk.Scrollbar(f_list)
        scr.pack(side="right", fill="y")
        self.listbox_ses = tk.Listbox(
            f_list, selectmode=tk.EXTENDED, height=5, bg="#0e1117", fg="white",
            selectbackground=self.cyan_dim, yscrollcommand=scr.set, font=("Consolas", 9)
        )
        self.listbox_ses.pack(side="left", fill="x", expand=True)
        scr.config(command=self.listbox_ses.yview)
        self.listbox_ses.bind('<<ListboxSelect>>', lambda e: self.actualizar_auditoria_automatica())

        self.lbl_alerta_meta = tk.Label(
            frame_ses, text="[AUDITORIA]: Seleccione tomas para inspeccionar coherencia anatómica y BPM.",
            bg="#111111", fg=self.cyan_dim, font=("Consolas", 9), anchor="w", padx=6, pady=4
        )
        self.lbl_alerta_meta.pack(fill="x", pady=(4, 0))

        # --- PANEL 2: MODALIDAD FISIOLÓGICA DE ENTRADA ---
        frame_mod = tk.LabelFrame(
            main_frame, text=" 2. Modalidad Fisiológica de Entrada ",
            bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 10, "bold"), padx=8, pady=6
        )
        frame_mod.pack(fill="x", pady=4)

        self.var_modalidad = tk.StringVar(value="envolvente")
        mods = [
            ("Envolvente 1D (RMS a 200 Hz) [Línea Base: 56.1%]", "envolvente"),
            ("Señal Cruda 1D (2000 Hz / GAP+GMP)", "cruda"),
            ("Espectrograma 2D (STFT Calibrado en dB)", "espectrograma_2d"),
            ("Espectrograma 3D (STFT en Espacio 3D)", "espectrograma_3d")
        ]
        grid_mod = tk.Frame(frame_mod, bg=self.bg_panel)
        grid_mod.pack(fill="x")
        for i, (txt, val) in enumerate(mods):
            r = tk.Radiobutton(
                grid_mod, text=txt, variable=self.var_modalidad, value=val,
                bg=self.bg_panel, fg="white", selectcolor=self.bg_dark,
                activebackground=self.bg_panel, activeforeground=self.cyan_neon,
                font=("Arial", 9, "bold" if "56.1%" in txt else "normal")
            )
            r.grid(row=i//2, column=i%2, sticky="w", padx=10, pady=2)

        # --- PANEL 3: PARÁMETROS DE ENTRENAMIENTO ---
        frame_par = tk.LabelFrame(
            main_frame, text=" 3. Parámetros de Optimización y Calibración ",
            bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 10, "bold"), padx=8, pady=6
        )
        frame_par.pack(fill="x", pady=4)

        f_inputs = tk.Frame(frame_par, bg=self.bg_panel)
        f_inputs.pack(fill="x")

        tk.Label(f_inputs, text="Épocas:", bg=self.bg_panel, fg=self.fg_text).grid(row=0, column=0, sticky="e", padx=4)
        self.ent_epochs = tk.Entry(f_inputs, width=6, bg="#0e1117", fg="white")
        self.ent_epochs.insert(0, "150")
        self.ent_epochs.grid(row=0, column=1, sticky="w", padx=4)

        tk.Label(f_inputs, text="Batch Size:", bg=self.bg_panel, fg=self.fg_text).grid(row=0, column=2, sticky="e", padx=4)
        self.ent_batch = tk.Entry(f_inputs, width=6, bg="#0e1117", fg="white")
        self.ent_batch.insert(0, "32")
        self.ent_batch.grid(row=0, column=3, sticky="w", padx=4)

        tk.Label(f_inputs, text="Learning Rate:", bg=self.bg_panel, fg=self.fg_text).grid(row=0, column=4, sticky="e", padx=4)
        self.ent_lr = tk.Entry(f_inputs, width=8, bg="#0e1117", fg="white")
        self.ent_lr.insert(0, "0.002")
        self.ent_lr.grid(row=0, column=5, sticky="w", padx=4)

        self.var_p95 = tk.BooleanVar(value=True)
        chk_p95 = tk.Checkbutton(
            f_inputs, text="Calibración Fisiológica P95 Cruzada (Balance Inter-Electrodo)",
            variable=self.var_p95, bg=self.bg_panel, fg=self.cyan_neon, selectcolor=self.bg_dark,
            font=("Arial", 9, "bold")
        )
        chk_p95.grid(row=1, column=0, columnspan=6, sticky="w", padx=4, pady=(4, 0))

        # --- PANEL 4: BOTONES DE ACCION ---
        frame_btns = tk.Frame(main_frame, bg=self.bg_dark)
        frame_btns.pack(fill="x", pady=6)

        self.btn_extraer = tk.Button(
            frame_btns, text="1. EXTRAER DATASET",
            bg=self.cyan_dim, fg="black", font=("Arial", 10, "bold"),
            command=self.ejecutar_extraccion
        )
        self.btn_extraer.pack(side="left", fill="x", expand=True, padx=2, ipady=6)

        self.btn_entrenar = tk.Button(
            frame_btns, text="2. ENTRENAR AUTOENCODER",
            bg=self.cyan_neon, fg="black", font=("Arial", 10, "bold"),
            command=self.ejecutar_entrenamiento
        )
        self.btn_entrenar.pack(side="left", fill="x", expand=True, padx=2, ipady=6)

        self.btn_plotear = tk.Button(
            frame_btns, text="3. PLOTEAR ESPACIO LATENTE",
            bg=self.green_neon, fg="black", font=("Arial", 10, "bold"),
            command=self.ejecutar_evaluacion
        )
        self.btn_plotear.pack(side="left", fill="x", expand=True, padx=2, ipady=6)

        self.btn_todo = tk.Button(
            frame_btns, text="FLUJO COMPLETO (1 CLICK)",
            bg="#ff8800", fg="black", font=("Arial", 10, "bold"),
            command=self.ejecutar_todo_en_cadena
        )
        self.btn_todo.pack(side="left", fill="x", expand=True, padx=2, ipady=6)

        # --- PANEL 5: CONSOLA DE REGISTRO EN VIVO ---
        lbl_consola = tk.Label(main_frame, text="Registro de Actividad y Métricas:", bg=self.bg_dark, fg=self.fg_text, font=("Arial", 9, "bold"))
        lbl_consola.pack(anchor="w", pady=(4, 1))

        self.txt_log = tk.Text(main_frame, height=10, bg="#050505", fg="#00FF88", font=("Consolas", 9), state="disabled")
        self.txt_log.pack(fill="both", expand=True, pady=(0, 4))

        f_aux = tk.Frame(main_frame, bg=self.bg_dark)
        f_aux.pack(fill="x")
        tk.Button(f_aux, text="Abrir Carpeta de Resultados", bg="#111", fg=self.yellow_warn, font=("Arial", 9), command=self.abrir_carpeta_resultados).pack(side="left", padx=2)
        tk.Button(f_aux, text="Limpiar Consola", bg="#111", fg=self.fg_text, font=("Arial", 9), command=self.limpiar_consola).pack(side="right", padx=2)

    def log(self, mensaje):
        def _append():
            self.txt_log.config(state="normal")
            self.txt_log.insert(tk.END, str(mensaje) + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state="disabled")
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.root.after(0, _append)

    def limpiar_consola(self):
        self.txt_log.config(state="normal")
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.config(state="disabled")

    def toggle_botones(self, estado):
        st = "normal" if estado else "disabled"
        self.btn_extraer.config(state=st)
        self.btn_entrenar.config(state=st)
        self.btn_plotear.config(state=st)
        self.btn_todo.config(state=st)

    def cargar_arbol_sesiones(self):
        if not os.path.exists(self.base_dir):
            self.log(f"[ERROR] No existe la base de datos en: {self.base_dir}")
            return

        self.listbox_ses.delete(0, tk.END)
        self.rutas_dict = {}

        fechas = sorted(os.listdir(self.base_dir))
        for f in fechas:
            f_path = os.path.join(self.base_dir, f)
            if not os.path.isdir(f_path):
                continue
            tomas = sorted([
                t for t in os.listdir(f_path)
                if os.path.isdir(os.path.join(f_path, t)) and t.split('_')[0].upper() in ['A', 'E', 'I', 'O', 'U']
            ])
            for t in tomas:
                item_label = f"[{f}] {t}"
                self.listbox_ses.insert(tk.END, item_label)
                self.rutas_dict[item_label] = os.path.join(f_path, t)

        if self.rutas_preseleccionadas:
            rutas_norm = [os.path.normpath(r) for r in self.rutas_preseleccionadas]
            self.listbox_ses.selection_clear(0, tk.END)
            encontrados = 0
            for idx in range(self.listbox_ses.size()):
                label = self.listbox_ses.get(idx)
                ruta = os.path.normpath(self.rutas_dict.get(label, ""))
                for r_sel in rutas_norm:
                    if ruta == r_sel or os.path.basename(ruta) == os.path.basename(r_sel):
                        self.listbox_ses.selection_set(idx)
                        encontrados += 1
                        break
            self.actualizar_auditoria_automatica()
            self.log(f"[INICIO] Se sincronizaron {encontrados} mediciones preseleccionadas desde el Gestor de Sesiones.")
        else:
            self.filtrar_sesion("2026-07-10")

    def seleccionar_todo(self):
        self.listbox_ses.select_set(0, tk.END)
        self.actualizar_auditoria_automatica()

    def desmarcar_todo(self):
        self.listbox_ses.selection_clear(0, tk.END)
        self.actualizar_auditoria_automatica()

    def filtrar_sesion(self, termino):
        self.listbox_ses.selection_clear(0, tk.END)
        for i in range(self.listbox_ses.size()):
            txt = self.listbox_ses.get(i)
            if termino.lower() in txt.lower():
                self.listbox_ses.selection_set(i)
        self.actualizar_auditoria_automatica()

    def get_rutas_seleccionadas(self):
        indices = self.listbox_ses.curselection()
        return [self.rutas_dict[self.listbox_ses.get(i)] for i in indices if self.listbox_ses.get(i) in self.rutas_dict]

    def actualizar_auditoria_automatica(self):
        rutas = self.get_rutas_seleccionadas()
        if not rutas:
            self.lbl_alerta_meta.config(
                text="[AUDITORIA]: No hay sesiones seleccionadas.",
                bg="#111", fg=self.fg_text
            )
            return

        res = motor.auditar_metadatos_sesiones(rutas)
        if not res['compatible']:
            adv_txt = " | ".join(res['advertencias'][:2])
            self.lbl_alerta_meta.config(
                text=f"{adv_txt}",
                bg=self.red_alert, fg="white"
            )
        else:
            m_res = res['musculos_resumen']
            m0 = m_res.get('canal_0', 'Ch0')
            m1 = m_res.get('canal_1', 'Ch1')
            m2 = m_res.get('canal_2', 'Ch2')
            n_tomas = len(rutas)
            self.lbl_alerta_meta.config(
                text=f"[OK - {n_tomas} tomas coherentes] Ch0: {m0} | Ch1: {m1} | Ch2: {m2}",
                bg="#003311", fg=self.green_neon
            )

    def ejecutar_auditoria_manual(self):
        rutas = self.get_rutas_seleccionadas()
        if not rutas:
            messagebox.showwarning("Auditoría", "Seleccione al menos una sesión para auditar.")
            return
        
        self.log("=" * 60)
        self.log("AUDITORIA DETALLADA DE METADATOS INTER-DIA")
        self.log("=" * 60)
        res = motor.auditar_metadatos_sesiones(rutas)
        
        for adv in res['advertencias']:
            self.log(adv)
            
        if res['compatible']:
            self.log("[AUDITORIA EXITOSA]: Todas las tomas seleccionadas comparten músculos idénticos por canal, metrónomo y fs.")
        else:
            self.log("[ADVERTENCIA]: Existen discrepancias anatómicas o de metrónomo entre las tomas seleccionadas.")
            messagebox.showwarning("Alerta de Auditoría", "Se detectaron discrepancias entre las tomas seleccionadas. Revise la consola.")

    def ejecutar_extraccion(self):
        rutas = self.get_rutas_seleccionadas()
        if not rutas:
            messagebox.showwarning("Extracción", "Seleccione al menos una sesión de la lista.")
            return

        p95 = self.var_p95.get()
        self.toggle_botones(False)
        self.log("\n>>> INICIANDO EXTRACCION DE DATASET...")

        def _hilo():
            try:
                npz_path, n_pulsos = motor.extraer_dataset_unificado(
                    rutas, usar_calibracion_p95=p95, callback_log=self.log
                )
                self.archivo_npz_actual = npz_path
                self.log(f">>> Extracción completada exitosamente. Pulsos listos: {n_pulsos}\n")
            except Exception as e:
                self.log(f"[ERROR EN EXTRACCION]: {e}")
                messagebox.showerror("Error de Extracción", str(e))
            finally:
                self.root.after(0, lambda: self.toggle_botones(True))

        threading.Thread(target=_hilo, daemon=True).start()

    def ejecutar_entrenamiento(self):
        if not os.path.exists(self.archivo_npz_actual):
            messagebox.showwarning("Entrenamiento", "Primero debe extraer el dataset (Paso 1).")
            return

        mod_sel = self.var_modalidad.get()
        if mod_sel == "espectrograma_3d":
            modalidad = "espectrograma"
            latent_dim = 3
        elif mod_sel == "espectrograma_2d":
            modalidad = "espectrograma"
            latent_dim = 2
        else:
            modalidad = mod_sel
            latent_dim = 2

        try:
            epochs = int(self.ent_epochs.get())
            batch_size = int(self.ent_batch.get())
            lr = float(self.ent_lr.get())
        except ValueError:
            messagebox.showerror("Error", "Parámetros numéricos inválidos en épocas, batch o lr.")
            return

        self.toggle_botones(False)
        self.log(f"\n>>> INICIANDO ENTRENAMIENTO CERO SUPERVISADO ({mod_sel.upper()})...")

        def _hilo():
            try:
                mod, pth = motor.entrenar_autoencoder(
                    self.archivo_npz_actual, modalidad=modalidad, latent_dim=latent_dim,
                    epochs=epochs, batch_size=batch_size, lr=lr, callback_log=self.log
                )
                self.modelo_actual = mod
                self.log(">>> Entrenamiento finalizado con éxito.\n")
            except Exception as e:
                self.log(f"[ERROR EN ENTRENAMIENTO]: {e}")
                messagebox.showerror("Error de Entrenamiento", str(e))
            finally:
                self.root.after(0, lambda: self.toggle_botones(True))

        threading.Thread(target=_hilo, daemon=True).start()

    def ejecutar_evaluacion(self):
        if not os.path.exists(self.archivo_npz_actual):
            messagebox.showwarning("Ploteo", "No se encontró el dataset en caché.")
            return

        mod_sel = self.var_modalidad.get()
        if mod_sel == "espectrograma_3d":
            modalidad = "espectrograma"
            latent_dim = 3
        elif mod_sel == "espectrograma_2d":
            modalidad = "espectrograma"
            latent_dim = 2
        else:
            modalidad = mod_sel
            latent_dim = 2

        pth_modelo = os.path.join(emg_desarrollo_dir, "resultados", "resultados_autoencoder", f"autoencoder_{modalidad}_{latent_dim}d.pth")
        if self.modelo_actual is None:
            if not os.path.exists(pth_modelo):
                messagebox.showwarning("Ploteo", "No se encontró el modelo entrenado. Ejecute el Paso 2 primero.")
                return
            if modalidad == "envolvente":
                self.modelo_actual = motor.AutoencoderEnvolvente1D(latent_dim=latent_dim)
            elif modalidad == "cruda":
                self.modelo_actual = motor.AutoencoderCruda1D(latent_dim=latent_dim)
            else:
                self.modelo_actual = motor.AutoencoderEspectrograma2D(latent_dim=latent_dim)
            self.modelo_actual.load_state_dict(torch.load(pth_modelo, map_location='cpu'))

        self.toggle_botones(False)
        self.log(f"\n>>> EVALUANDO Y PLOTEANDO ESPACIO LATENTE CANONICO ({mod_sel.upper()})...")

        def _hilo():
            try:
                metricas = motor.evaluar_espacio_latente(
                    self.archivo_npz_actual, self.modelo_actual, modalidad=modalidad,
                    latent_dim=latent_dim, callback_log=self.log
                )
                self.log(f">>> Exactitud GMM Final: {metricas['gmm_acc']:.2f}% | Silueta: {metricas['silhouette']:+.3f}")
                self.abrir_archivo(metricas['fig_path'])
            except Exception as e:
                self.log(f"[ERROR EN PLOTEO]: {e}")
                messagebox.showerror("Error de Ploteo", str(e))
            finally:
                self.root.after(0, lambda: self.toggle_botones(True))

        threading.Thread(target=_hilo, daemon=True).start()

    def ejecutar_todo_en_cadena(self):
        rutas = self.get_rutas_seleccionadas()
        if not rutas:
            messagebox.showwarning("Flujo Completo", "Seleccione al menos una sesión.")
            return

        mod_sel = self.var_modalidad.get()
        if mod_sel == "espectrograma_3d":
            modalidad = "espectrograma"
            latent_dim = 3
        elif mod_sel == "espectrograma_2d":
            modalidad = "espectrograma"
            latent_dim = 2
        else:
            modalidad = mod_sel
            latent_dim = 2

        p95 = self.var_p95.get()
        try:
            epochs = int(self.ent_epochs.get())
            batch_size = int(self.ent_batch.get())
            lr = float(self.ent_lr.get())
        except ValueError:
            messagebox.showerror("Error", "Parámetros inválidos.")
            return

        self.toggle_botones(False)
        self.log(f"\n>>> EJECUTANDO FLUJO COMPLETO: EXTRACCION -> ENTRENAMIENTO -> EVALUACION ({mod_sel.upper()})...")

        def _hilo_cadena():
            try:
                npz_path, n_pulsos = motor.extraer_dataset_unificado(
                    rutas, usar_calibracion_p95=p95, callback_log=self.log
                )
                self.archivo_npz_actual = npz_path

                mod, pth = motor.entrenar_autoencoder(
                    npz_path, modalidad=modalidad, latent_dim=latent_dim,
                    epochs=epochs, batch_size=batch_size, lr=lr, callback_log=self.log
                )
                self.modelo_actual = mod

                metricas = motor.evaluar_espacio_latente(
                    npz_path, mod, modalidad=modalidad,
                    latent_dim=latent_dim, callback_log=self.log
                )
                self.log(f"\n>>> FLUJO COMPLETO FINALIZADO CON EXITO. Exactitud GMM: {metricas['gmm_acc']:.2f}%\n")
                self.abrir_archivo(metricas['fig_path'])

            except Exception as e:
                self.log(f"[ERROR EN FLUJO COMPLETO]: {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self.root.after(0, lambda: self.toggle_botones(True))

        threading.Thread(target=_hilo_cadena, daemon=True).start()

    def abrir_carpeta_resultados(self):
        res_dir = os.path.join(emg_desarrollo_dir, "resultados", "resultados_autoencoder")
        if sys.platform == "win32":
            os.startfile(res_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", res_dir])
        else:
            subprocess.Popen(["xdg-open", res_dir])

    def abrir_archivo(self, filepath):
        if not os.path.exists(filepath):
            return
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", filepath])
        else:
            subprocess.Popen(["xdg-open", filepath])

def main():
    root = tk.Tk()
    rutas = sys.argv[1:] if len(sys.argv) > 1 else None
    app = AutoencoderNoSupervisadoGUI(root, rutas_preseleccionadas=rutas)
    root.mainloop()

if __name__ == "__main__":
    main()
