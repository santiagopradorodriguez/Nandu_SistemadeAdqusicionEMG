#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: daq_bridge_server.py
# Descripción: Servidor Puente DAQ para Windows con Interfaz Gráfica (GUI) de
#              1 clic (Tkinter nativo, sin dependencias pesadas) y modo CLI
#              (--headless). Lee la tarjeta NI USB-6210 en Windows y transmite
#              en tiempo real por ZeroMQ hacia Linux.
# ==============================================================================

import argparse
import json
import socket
import struct
import sys
import threading
import time
import numpy as np

try:
    import zmq
except ImportError:
    zmq = None

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    NIDAQMX_AVAILABLE = True
except ImportError:
    nidaqmx = None
    AcquisitionType = None
    TerminalConfiguration = None
    NIDAQMX_AVAILABLE = False

MAGIC_HEADER = b"NDAQ"


def get_local_ips():
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_primary = s.getsockname()[0]
        s.close()
        if ip_primary and ip_primary not in ips:
            ips.insert(0, ip_primary)
    except Exception:
        pass
    return ips or ["127.0.0.1"]


def list_ni_devices():
    if not NIDAQMX_AVAILABLE:
        return ["Dev1"]
    try:
        system = nidaqmx.system.System.local()
        devs = [d.name for d in system.devices]
        return devs if devs else ["Dev1"]
    except Exception:
        return ["Dev1"]


class DAQBridgeServerEngine:
    """Motor de adquisición NI-DAQmx / Simulador y transmisión ZeroMQ."""

    def __init__(
        self,
        device="Dev1",
        channels=4,
        rate=2000.0,
        chunk=200,
        port=5555,
        mode="DIFF",
        vmin=-5.0,
        vmax=5.0,
        simulate=False,
        on_log=print,
        on_stats=None,
    ):
        self.device = device
        self.channels = int(channels)
        self.rate = float(rate)
        self.chunk = int(chunk)
        self.port = int(port)
        self.mode = str(mode).upper()
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.simulate = bool(simulate) or (not NIDAQMX_AVAILABLE)
        self.on_log = on_log
        self.on_stats = on_stats

        self._stop_event = threading.Event()
        self._thread = None
        self.seq_counter = 0
        self.total_samples = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_loop(self):
        if zmq is None:
            self.on_log("[ERROR] pyzmq no está instalado. Ejecuta: pip install pyzmq numpy nidaqmx")
            return

        ctx = zmq.Context()
        pub_sock = ctx.socket(zmq.PUB)
        pub_sock.setsockopt(zmq.SNDHWM, 120)
        pub_sock.setsockopt(zmq.LINGER, 0)
        pub_sock.bind(f"tcp://0.0.0.0:{self.port}")

        rep_sock = ctx.socket(zmq.REP)
        rep_sock.setsockopt(zmq.LINGER, 0)
        rep_sock.bind(f"tcp://0.0.0.0:{self.port + 1}")
        poller = zmq.Poller()
        poller.register(rep_sock, zmq.POLLIN)

        ips_str = ", ".join(get_local_ips())
        self.on_log(f"[OK] Servidor ZeroMQ activo en {ips_str} (Datos: {self.port} | Control: {self.port + 1})")

        task = None
        use_sim = self.simulate
        if not use_sim:
            try:
                term_map = {
                    "DIFF": TerminalConfiguration.DIFF,
                    "RSE": TerminalConfiguration.RSE,
                    "NRSE": TerminalConfiguration.NRSE,
                }
                term_cfg = term_map.get(self.mode, TerminalConfiguration.DIFF)
                task = nidaqmx.Task()
                for ch_idx in range(self.channels):
                    task.ai_channels.add_ai_voltage_chan(
                        f"{self.device}/ai{ch_idx}",
                        terminal_config=term_cfg,
                        min_val=self.vmin,
                        max_val=self.vmax,
                    )
                task.timing.cfg_samp_clk_timing(
                    rate=self.rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=max(self.chunk * 10, 4096),
                )
                task.start()
                self.on_log(
                    f"[DAQ] Adquiriendo de {self.device}/ai0:ai{self.channels - 1} @ {self.rate:.0f} Hz ({self.mode})"
                )
            except Exception as e:
                self.on_log(f"[AVISO] No se pudo abrir {self.device} ({e}). Pasando a modo Simulador EMG.")
                if task is not None:
                    try:
                        task.close()
                    except Exception:
                        pass
                    task = None
                use_sim = True

        self.seq_counter = 0
        self.total_samples = 0
        t0 = time.perf_counter()
        last_stat_t = t0

        try:
            while not self._stop_event.is_set():
                # 1. Atender pings/consultas desde Linux en puerto port+1
                socks = dict(poller.poll(timeout=0))
                if rep_sock in socks:
                    try:
                        _ = rep_sock.recv(zmq.NOBLOCK)
                        meta = {
                            "status": "ok",
                            "device": "Simulador" if use_sim else self.device,
                            "num_channels": self.channels,
                            "sample_rate": self.rate,
                            "chunk_size": self.chunk,
                            "terminal_config": self.mode,
                            "seq": self.seq_counter,
                        }
                        rep_sock.send_string(json.dumps(meta))
                    except Exception:
                        pass

                # 2. Leer bloque de muestras
                if not use_sim and task is not None:
                    raw = task.read(number_of_samples_per_channel=self.chunk, timeout=2.0)
                    arr = np.asarray(raw, dtype=np.float64)
                    if arr.ndim == 1:
                        arr = arr[np.newaxis, :]
                else:
                    dt = self.chunk / self.rate
                    time.sleep(max(0.002, dt * 0.92))
                    t_vec = (self.total_samples + np.arange(self.chunk)) / self.rate
                    arr = np.zeros((self.channels, self.chunk), dtype=np.float64)
                    for c in range(self.channels):
                        env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.5 * t_vec - c * 0.8)
                        carrier = np.sin(2.0 * np.pi * (85.0 + 35.0 * c) * t_vec)
                        arr[c, :] = (0.00018 * (env ** 3) * carrier) + np.random.normal(0.0, 0.000015, self.chunk)

                arr = np.ascontiguousarray(arr, dtype=np.float64)
                n_ch, n_s = arr.shape
                header = struct.pack(
                    "<4sQdIII",
                    MAGIC_HEADER,
                    self.seq_counter,
                    time.time(),
                    n_ch,
                    n_s,
                    int(round(self.rate)),
                )
                pub_sock.send_multipart([header, arr.tobytes()], flags=zmq.NOBLOCK)

                self.seq_counter += 1
                self.total_samples += n_s

                now = time.perf_counter()
                if self.on_stats and (now - last_stat_t) >= 0.4:
                    last_stat_t = now
                    peak_uv = float(np.max(np.abs(arr[:, -min(50, n_s):]))) * 1e3  # aprox uV con ganancia 1000
                    self.on_stats(self.seq_counter, self.total_samples, peak_uv, use_sim)
        except Exception as e:
            self.on_log(f"[ERROR BUCLE DAQ] {e}")
        finally:
            if task is not None:
                try:
                    task.stop()
                    task.close()
                except Exception:
                    pass
            try:
                pub_sock.close()
                rep_sock.close()
                ctx.term()
            except Exception:
                pass
            self.on_log("[DETENIDO] Puente DAQ cerrado limpiamente.")


def launch_windows_gui(default_args):
    """Interfaz gráfica sencilla en Tkinter para cualquier usuario en Windows."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Ñandú LSD — Puente DAQ Windows (NI USB-6210 -> Linux)")
    root.geometry("620x480")
    root.configure(bg="#0b0f19")

    engine_holder = {"engine": None}
    ips = get_local_ips()
    ip_principal = ips[0]

    # Encabezado con la IP lista para copiar en Linux
    hdr = tk.Frame(root, bg="#111827", bd=1, relief="solid", padx=12, pady=10)
    hdr.pack(fill="x", padx=12, pady=(12, 6))

    tk.Label(
        hdr,
        text="PUENTE NI-DAQmx PARA WINDOWS / VM",
        bg="#111827",
        fg="#00ffcc",
        font=("Segoe UI", 13, "bold"),
    ).pack(anchor="w")

    tk.Label(
        hdr,
        text=f"IP de esta máquina Windows:  {ip_principal}:{default_args.port}   (En Linux puedes dejar 'auto:5555')",
        bg="#111827",
        fg="#ffcc00",
        font=("Consolas", 10, "bold"),
    ).pack(anchor="w", pady=(4, 0))

    # Formulario de parámetros sencillos
    frm = tk.Frame(root, bg="#0b0f19", padx=12, pady=6)
    frm.pack(fill="x")

    tk.Label(frm, text="Dispositivo NI:", bg="#0b0f19", fg="#e5e7eb", font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=4)
    var_dev = tk.StringVar(value=default_args.device)
    cmb_dev = ttk.Combobox(frm, textvariable=var_dev, values=list_ni_devices(), width=14)
    cmb_dev.grid(row=0, column=1, sticky="w", padx=6)

    tk.Label(frm, text="Canales (ai0..aiN):", bg="#0b0f19", fg="#e5e7eb", font=("Segoe UI", 10)).grid(row=0, column=2, sticky="w", padx=(14, 4))
    var_ch = tk.IntVar(value=default_args.channels)
    spn_ch = tk.Spinbox(frm, from_=1, to=8, textvariable=var_ch, width=6, bg="#1f2937", fg="#00ffcc")
    spn_ch.grid(row=0, column=3, sticky="w")

    tk.Label(frm, text="Frecuencia (Hz):", bg="#0b0f19", fg="#e5e7eb", font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=4)
    var_rate = tk.StringVar(value=str(int(default_args.rate)))
    cmb_rate = ttk.Combobox(frm, textvariable=var_rate, values=["1000", "2000", "5000", "10000"], width=14)
    cmb_rate.grid(row=1, column=1, sticky="w", padx=6)

    tk.Label(frm, text="Modo Terminal:", bg="#0b0f19", fg="#e5e7eb", font=("Segoe UI", 10)).grid(row=1, column=2, sticky="w", pady=4, padx=(14, 4))
    var_mode = tk.StringVar(value=default_args.mode)
    cmb_mode = ttk.Combobox(frm, textvariable=var_mode, values=["DIFF", "RSE", "NRSE"], width=8)
    cmb_mode.grid(row=1, column=3, sticky="w")

    var_sim = tk.BooleanVar(value=default_args.simulate or (not NIDAQMX_AVAILABLE))
    chk_sim = tk.Checkbutton(
        frm,
        text="Modo Simulador (probar red sin DAQ conectada)",
        variable=var_sim,
        bg="#0b0f19",
        fg="#ff9900",
        selectcolor="#1f2937",
        activebackground="#0b0f19",
        font=("Segoe UI", 9, "bold"),
    )
    chk_sim.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 2))

    # Indicador de estado y paquetes
    lbl_stats = tk.Label(
        root,
        text="Estado: LISTO PARA INICIAR | Paquetes enviados: 0",
        bg="#111827",
        fg="#39ff14",
        font=("Consolas", 10, "bold"),
        pady=6,
    )
    lbl_stats.pack(fill="x", padx=12, pady=4)

    # Botón principal de 1 clic
    btn_toggle = tk.Button(
        root,
        text="▶ INICIAR PUENTE DAQ",
        bg="#065f46",
        fg="#ffffff",
        activebackground="#059669",
        font=("Segoe UI", 12, "bold"),
        pady=10,
        cursor="hand2",
    )
    btn_toggle.pack(fill="x", padx=12, pady=6)

    # Consola de eventos
    txt_log = tk.Text(root, height=10, bg="#050508", fg="#00ffcc", font=("Consolas", 9))
    txt_log.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def append_log(msg):
        def _do():
            txt_log.insert("end", str(msg) + "\n")
            txt_log.see("end")
        root.after(0, _do)

    def update_stats(seq, total_samps, peak_uv, is_sim):
        modo_str = "SIMULADOR" if is_sim else f"NI-DAQ ({var_dev.get()})"
        def _do():
            lbl_stats.config(
                text=f"ACTIVO [{modo_str}] | Paquetes: {seq} | Muestras: {total_samps:,} | Pico: {peak_uv:.1f} µV"
            )
        root.after(0, _do)

    def toggle_server():
        if engine_holder["engine"] is None:
            eng = DAQBridgeServerEngine(
                device=var_dev.get().strip() or "Dev1",
                channels=int(var_ch.get()),
                rate=float(var_rate.get()),
                chunk=default_args.chunk,
                port=default_args.port,
                mode=var_mode.get(),
                simulate=bool(var_sim.get()),
                on_log=append_log,
                on_stats=update_stats,
            )
            engine_holder["engine"] = eng
            eng.start()
            btn_toggle.config(text="■ DETENER PUENTE DAQ", bg="#991b1b", activebackground="#dc2626")
        else:
            engine_holder["engine"].stop()
            engine_holder["engine"] = None
            btn_toggle.config(text="▶ INICIAR PUENTE DAQ", bg="#065f46", activebackground="#059669")
            lbl_stats.config(text="Estado: DETENIDO")

    btn_toggle.config(command=toggle_server)

    # Auto-iniciar al abrir la ventana para que el usuario no tenga ni que tocar botones si no quiere
    root.after(300, toggle_server)

    def on_close():
        if engine_holder["engine"] is not None:
            engine_holder["engine"].stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Servidor Puente NI-DAQmx para Windows (GUI + CLI)")
    parser.add_argument("--device", default="Dev1", help="Nombre del dispositivo NI-DAQmx (ej. Dev1)")
    parser.add_argument("--channels", type=int, default=4, help="Cantidad de canales analógicos (ai0..aiN)")
    parser.add_argument("--rate", type=float, default=2000.0, help="Frecuencia de muestreo en Hz")
    parser.add_argument("--chunk", type=int, default=100, help="Muestras por paquete")
    parser.add_argument("--port", type=int, default=5555, help="Puerto TCP ZeroMQ")
    parser.add_argument("--mode", default="DIFF", choices=["DIFF", "RSE", "NRSE"], help="Modo de terminal analógica")
    parser.add_argument("--simulate", action="store_true", help="Simular señales EMG sin hardware NI conectado")
    parser.add_argument("--headless", action="store_true", help="Ejecutar en consola pura sin abrir ventana GUI")
    args = parser.parse_args()

    if args.headless:
        eng = DAQBridgeServerEngine(
            device=args.device,
            channels=args.channels,
            rate=args.rate,
            chunk=args.chunk,
            port=args.port,
            mode=args.mode,
            simulate=args.simulate,
        )
        eng.start()
        print("Presiona Ctrl+C para detener el servidor...")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            eng.stop()
    else:
        try:
            launch_windows_gui(args)
        except Exception as e:
            print(f"[Aviso] No se pudo abrir GUI Tkinter ({e}), ejecutando en modo consola...")
            eng = DAQBridgeServerEngine(
                device=args.device,
                channels=args.channels,
                rate=args.rate,
                chunk=args.chunk,
                port=args.port,
                mode=args.mode,
                simulate=args.simulate,
            )
            eng.start()
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                eng.stop()


if __name__ == "__main__":
    main()
