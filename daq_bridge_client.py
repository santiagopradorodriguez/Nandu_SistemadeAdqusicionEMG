#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daq_bridge_client.py
====================
Cliente ZeroMQ en Tiempo Real para Pop!_OS / Linux (Receptor de la placa NI USB-6259).

Características principales:
----------------------------
1. Conexión mediante ZeroMQ (`SUB` o `PULL`) hacia la IP de la máquina virtual Windows
   en la interfaz virtual `virbr0` (ej. `192.168.122.x:5555`), con auto-detección de IP
   de la VM si se especifica `host="auto"`.
2. Generador / iterador `read_block()` (disponible tanto como método de `DAQBridgeClient`
   como función generadora independiente) que:
   - Recibe los 2 frames multipart (`[frame_json, frame_bytes]`).
   - Reconstruye el array de NumPy sin copias innecesarias mediante `np.frombuffer()`
     respetando el `dtype` (`float64` o `float32`) y el `shape` original `(canales, muestras)`.
   - Verifica la continuidad de bloques usando el índice de secuencia `seq_idx` y contabiliza
     cualquier pérdida de paquetes.
3. Reconexión automática transparente ante reinicios de la VM o cortes momentáneos de red.
4. Canal de control remoto (`configure_remote_server` y `ping_server` en puerto `5556`)
   para que `autoforge_daq_experimental.py` pueda cambiar canales, Sample Rate y modo
   `RSE` / `DIFF` directamente desde la interfaz gráfica en Pop!_OS.
5. Bucle de demostración en tiempo real (`__main__`) que calcula e imprime el valor RMS,
   la media por canal, el retardo entre bloques y la latencia de red.

Instalación en Pop!_OS / Linux:
-------------------------------
    pip install pyzmq numpy

Ejecución de prueba en Pop!_OS:
-------------------------------
    python3 daq_bridge_client.py --host auto --port 5555
    # o especificando la IP de la VM en virbr0:
    python3 daq_bridge_client.py --host 192.168.122.100 --port 5555
"""

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import zmq


def detect_vm_ip(port: int = 5555, timeout_s: float = 0.08) -> str:
    """
    Intenta descubrir automáticamente la IP de la VM Windows sobre `virbr0` (192.168.122.x)
    en Pop!_OS / Linux usando:
      1. Variable de entorno `DAQ_VM_IP` si existe.
      2. Concesiones DHCP de libvirt (`/var/lib/libvirt/dnsmasq/*.status`).
      3. Tabla ARP de `virbr0` (`ip neigh show dev virbr0`).
      4. `virsh domifaddr` si libvirt está instalado.
      5. Escaneo rápido de sockets TCP al puerto 5555 en candidatos activos.
    Si no detecta ninguna VM remota, retorna `127.0.0.1` (útil para pruebas locales o port-forwarding).
    """
    env_ip = os.environ.get("DAQ_VM_IP", "").strip()
    if env_ip:
        return env_ip

    candidates: List[str] = []

    # 1. Revisar leases de dnsmasq de libvirt
    for status_file in glob.glob("/var/lib/libvirt/dnsmasq/*.status"):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                ip = entry.get("ip-address")
                if ip and ip not in candidates:
                    candidates.append(ip)
        except Exception:
            pass

    # 2. Revisar tabla ARP de virbr0
    try:
        out = subprocess.check_output(
            ["ip", "neigh", "show", "dev", "virbr0"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.5,
        )
        for line in out.splitlines():
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
            if m:
                ip = m.group(1)
                if ip not in candidates and not ip.endswith(".1"):
                    candidates.append(ip)
    except Exception:
        pass

    # 3. Revisar subredes de interfaces virtuales (virbr0 / vmnet / vboxnet)
    try:
        out = subprocess.check_output(
            ["ip", "-o", "-4", "addr", "show"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.5,
        )
        for line in out.splitlines():
            if any(ifname in line for ifname in ("virbr", "vnet", "vboxnet", "vmnet")):
                m = re.search(r"inet\s+(\d+\.\d+\.\d+)\.\d+/", line)
                if m:
                    prefix = m.group(1)
                    # Probar las IPs más habituales asignadas por DHCP en libvirt/KVM
                    for host_id in (100, 101, 2, 10, 50, 150, 200):
                        ip_cand = f"{prefix}.{host_id}"
                        if ip_cand not in candidates:
                            candidates.append(ip_cand)
    except Exception:
        pass

    # Probar si alguno de los candidatos tiene el puerto del servidor DAQ abierto
    for ip in ["127.0.0.1"] + candidates:
        try:
            with socket.create_connection((ip, int(port)), timeout=timeout_s):
                return ip
        except Exception:
            continue

    if candidates:
        return candidates[0]
    return "192.168.122.100"


def parse_host_port(endpoint_str: str, default_port: int = 5555) -> Tuple[str, int]:
    """Convierte '192.168.122.100:5555' o 'auto:5555' en (host, port)."""
    raw = str(endpoint_str or "").strip()
    if raw.startswith("tcp://"):
        raw = raw[len("tcp://") :]
    if ":" in raw:
        h, p_str = raw.rsplit(":", 1)
        try:
            port = int(p_str)
        except ValueError:
            port = default_port
    else:
        h = raw
        port = default_port

    h = h.strip() or "auto"
    if h.lower() == "auto":
        h = detect_vm_ip(port=port)
    return h, port


class DAQBridgeClient:
    """
    Cliente receptor ZeroMQ con generador `read_block()`, verificación de pérdida de
    secuencia, reconexión automática ante caídas de la VM y control remoto del servidor.
    """

    def __init__(
        self,
        host: str = "auto",
        port: int = 5555,
        control_port: Optional[int] = None,
        socket_mode: str = "SUB",
        recv_timeout_ms: int = 1500,
        reconnect_Wait_s: float = 1.0,
    ):
        resolved_host, resolved_port = parse_host_port(f"{host}:{port}", default_port=port)
        self.raw_host = host
        self.host = resolved_host
        self.port = int(resolved_port)
        self.control_port = int(control_port) if control_port is not None else (self.port + 1)
        self.socket_mode = socket_mode.upper().strip()
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.reconnect_wait_s = float(reconnect_Wait_s)

        self.zmq_ctx: Optional[zmq.Context] = None
        self.sock: Optional[zmq.Socket] = None
        self.poller: Optional[zmq.Poller] = None

        self.expected_seq_idx: Optional[int] = None
        self.total_dropped_blocks: int = 0
        self.total_received_blocks: int = 0
        self.last_block_perf_time: Optional[float] = None
        self.connected: bool = False

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    @property
    def control_endpoint(self) -> str:
        return f"tcp://{self.host}:{self.control_port}"

    def connect(self) -> None:
        """Abre o recrea el socket ZeroMQ hacia la VM Windows."""
        self.close_socket()
        if str(self.raw_host).lower() == "auto":
            self.host = detect_vm_ip(port=self.port)

        if self.zmq_ctx is None or self.zmq_ctx.closed:
            self.zmq_ctx = zmq.Context()

        sock_type = zmq.PULL if self.socket_mode == "PULL" else zmq.SUB
        self.sock = self.zmq_ctx.socket(sock_type)
        self.sock.setsockopt(zmq.RCVHWM, 256)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RECONNECT_IVL, 250)
        self.sock.setsockopt(zmq.RECONNECT_IVL_MAX, 2000)
        if sock_type == zmq.SUB:
            self.sock.setsockopt(zmq.SUBSCRIBE, b"")

        self.sock.connect(self.endpoint)
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)
        self.connected = True

    def close_socket(self) -> None:
        """Cierra únicamente el socket de datos (usado durante reconexiones)."""
        if self.sock is not None:
            try:
                if self.poller is not None:
                    self.poller.unregister(self.sock)
            except Exception:
                pass
            try:
                self.sock.close(0)
            except Exception:
                pass
            self.sock = None
        self.connected = False

    def close(self) -> None:
        """Cierra el socket y termina el contexto ZeroMQ."""
        self.close_socket()
        if self.zmq_ctx is not None:
            try:
                self.zmq_ctx.term()
            except Exception:
                pass
            self.zmq_ctx = None

    def ping_server(self, timeout_ms: int = 400) -> Optional[Dict[str, Any]]:
        """
        Consulta el estado del servidor en la VM Windows mediante el puerto de control (5556).
        Retorna el diccionario de respuesta o `None` si la VM aún no responde.
        """
        if str(self.raw_host).lower() == "auto":
            self.host = detect_vm_ip(port=self.port)

        ctx = self.zmq_ctx if (self.zmq_ctx is not None and not self.zmq_ctx.closed) else zmq.Context()
        own_ctx = ctx is not self.zmq_ctx
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        req.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        try:
            req.connect(self.control_endpoint)
            req.send_json({"cmd": "PING"})
            reply = req.recv_json()
            return reply
        except Exception:
            return None
        finally:
            try:
                req.close(0)
            except Exception:
                pass
            if own_ctx:
                try:
                    ctx.term()
                except Exception:
                    pass

    def configure_remote_server(
        self,
        device: Optional[str] = None,
        channels: Optional[List[str]] = None,
        terminal_config: Optional[str] = None,
        sample_rate_hz: Optional[float] = None,
        chunk_size: Optional[int] = None,
        min_val: float = -10.0,
        max_val: float = 10.0,
        dtype: str = "float64",
        timeout_ms: int = 800,
    ) -> Optional[Dict[str, Any]]:
        """
        Envía una orden de reconfiguración en caliente al servidor `daq_bridge_server.py`
        corriendo en la VM Windows.
        """
        ctx = self.zmq_ctx if (self.zmq_ctx is not None and not self.zmq_ctx.closed) else zmq.Context()
        own_ctx = ctx is not self.zmq_ctx
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        req.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        payload: Dict[str, Any] = {
            "cmd": "CONFIG",
            "min_val": min_val,
            "max_val": max_val,
            "dtype": dtype,
        }
        if device is not None:
            payload["device"] = device
        if channels is not None:
            payload["channels"] = channels
        if terminal_config is not None:
            payload["terminal_config"] = terminal_config
        if sample_rate_hz is not None:
            payload["sample_rate_hz"] = float(sample_rate_hz)
        if chunk_size is not None:
            payload["chunk_size"] = int(chunk_size)

        try:
            req.connect(self.control_endpoint)
            req.send_json(payload)
            return req.recv_json()
        except Exception:
            return None
        finally:
            try:
                req.close(0)
            except Exception:
                pass
            if own_ctx:
                try:
                    ctx.term()
                except Exception:
                    pass

    def read_block(
        self, stop_event: Optional[Any] = None
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """
        Generador continuo que recibe los 2 frames multipart desde ZeroMQ,
        reconstruye la matriz NumPy `(canales, muestras)` mediante `np.frombuffer()`,
        verifica que no se hayan perdido bloques usando `seq_idx` y reconecta
        automáticamente si la VM se reinicia o se corta momentáneamente la red.

        Yields:
            Tuple[np.ndarray, Dict[str, Any]]: `(data_array, metadata_dict)`
        """
        if not self.connected or self.sock is None:
            self.connect()

        while stop_event is None or not stop_event.is_set():
            try:
                if self.sock is None or self.poller is None:
                    self.connect()

                events = dict(self.poller.poll(timeout=self.recv_timeout_ms))
                if self.sock not in events:
                    if stop_event is not None and stop_event.is_set():
                        break
                    print(
                        f"[ZMQ BRIDGE] Sin datos tras {self.recv_timeout_ms} ms desde {self.endpoint}. "
                        "Verificando/reconectando con la VM Windows..."
                    )
                    self.connect()
                    continue

                frames = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                if len(frames) != 2:
                    continue

                frame_meta, frame_raw = frames
                meta: Dict[str, Any] = json.loads(frame_meta.decode("utf-8"))

                # 1. Reconstruir el array binario con np.frombuffer respetando dtype y shape (canales, muestras)
                dtype_str = str(meta.get("dtype", "float64"))
                np_dtype = np.float32 if "32" in dtype_str else np.float64

                shape = meta.get("shape")
                if shape and len(shape) == 2:
                    n_ch, n_samps = int(shape[0]), int(shape[1])
                else:
                    n_ch = int(meta.get("num_channels", len(meta.get("channels", []))))
                    n_samps = int(meta.get("num_samples", 0))

                arr = np.frombuffer(frame_raw, dtype=np_dtype).reshape((n_ch, n_samps)).copy()

                # 2. Verificar índice de secuencia para detectar bloques perdidos o reinicio del servidor
                seq_idx = int(meta.get("seq_idx", 0))
                dropped_now = 0
                if self.expected_seq_idx is not None:
                    if seq_idx > self.expected_seq_idx:
                        dropped_now = seq_idx - self.expected_seq_idx
                        self.total_dropped_blocks += dropped_now
                        print(
                            f"[WARN ZMQ] ¡Se detectaron {dropped_now} bloque(s) perdido(s)! "
                            f"(Esperado={self.expected_seq_idx}, Recibido={seq_idx}, "
                            f"Total perdidos={self.total_dropped_blocks})"
                        )
                    elif seq_idx < self.expected_seq_idx:
                        print(
                            f"[INFO ZMQ] Reinicio de secuencia detectado en la VM "
                            f"(seq_idx volvió a {seq_idx}). Sincronizando contador."
                        )
                self.expected_seq_idx = seq_idx + 1
                self.total_received_blocks += 1

                # 3. Calcular intervalo entre bloques recibidos
                now_perf = time.perf_counter()
                dt_ms = (
                    (now_perf - self.last_block_perf_time) * 1000.0
                    if self.last_block_perf_time is not None
                    else 0.0
                )
                self.last_block_perf_time = now_perf

                meta["dropped_blocks_delta"] = dropped_now
                meta["total_dropped_blocks"] = self.total_dropped_blocks
                meta["inter_block_dt_ms"] = dt_ms
                meta["network_latency_ms"] = max(0.0, (time.time() - float(meta.get("timestamp", time.time()))) * 1000.0)

                yield arr, meta

            except zmq.ZMQError as zerr:
                if stop_event is not None and stop_event.is_set():
                    break
                print(f"[ZMQ ERROR] Error de socket ({zerr}). Reintentando en {self.reconnect_wait_s:.1f}s...")
                time.sleep(self.reconnect_wait_s)
                self.connect()
            except Exception as exc:
                if stop_event is not None and stop_event.is_set():
                    break
                print(f"[CLIENT WARN] Error procesando bloque: {exc}")
                time.sleep(0.2)


def read_block(
    host: str = "auto",
    port: int = 5555,
    socket_mode: str = "SUB",
    recv_timeout_ms: int = 1500,
) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
    """
    Función generadora directa que envuelve `DAQBridgeClient.read_block()`
    para consumo rápido en scripts externos.
    """
    client = DAQBridgeClient(
        host=host,
        port=port,
        socket_mode=socket_mode,
        recv_timeout_ms=recv_timeout_ms,
    )
    try:
        yield from client.read_block()
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cliente ZeroMQ en Pop!_OS para lectura en tiempo real desde la VM Windows (NI USB-6259)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="auto",
        help="IP de la VM en virbr0 (ej: 192.168.122.100 o 'auto' para autodetección)",
    )
    parser.add_argument("--port", type=int, default=5555, help="Puerto TCP ZeroMQ de datos (por defecto 5555)")
    parser.add_argument(
        "--mode",
        type=str,
        default="SUB",
        choices=["SUB", "PULL"],
        help="Patrón de socket ZeroMQ (SUB o PULL, por defecto SUB)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Imprimir resumen en consola cada N bloques recibidos (por defecto 5)",
    )

    args = parser.parse_args()

    client = DAQBridgeClient(host=args.host, port=args.port, socket_mode=args.mode)
    print("=" * 78)
    print(" [NI-DAQmx ZeroMQ Bridge Client - Pop!_OS]")
    print(f"  • Conectando a VM Windows : {client.endpoint} (modo {args.mode})")
    print(f"  • Canal de Control        : {client.control_endpoint}")
    print("  • Presione Ctrl+C para detener la recepción.")
    print("=" * 78)

    ping_info = client.ping_server(timeout_ms=500)
    if ping_info:
        print(
            f" [PING OK] Dispositivo={ping_info.get('device')} | "
            f"Canales={ping_info.get('channels')} | "
            f"SR={ping_info.get('sample_rate_hz')} Hz | "
            f"Modo={ping_info.get('terminal_config')}"
        )

    try:
        for block_arr, meta in client.read_block():
            seq_idx = meta["seq_idx"]
            if seq_idx % max(1, args.print_every) == 0:
                rms_per_ch = np.sqrt(np.mean(np.square(block_arr), axis=1))
                mean_per_ch = np.mean(block_arr, axis=1)
                channels = meta.get("channels", [f"ch{i}" for i in range(block_arr.shape[0])])

                ch_stats = " | ".join(
                    f"{ch}: RMS={rms_per_ch[i]:6.4f}V, Media={mean_per_ch[i]:+6.4f}V"
                    for i, ch in enumerate(channels)
                )
                print(
                    f"[Bloque #{seq_idx:06d}] shape={block_arr.shape} | "
                    f"dt={meta['inter_block_dt_ms']:5.1f}ms | "
                    f"perdidos={meta['total_dropped_blocks']} | {ch_stats}"
                )
    except KeyboardInterrupt:
        print("\n[CLIENTE] Recepción finalizada por el usuario (Ctrl+C).")
    finally:
        client.close()


if __name__ == "__main__":
    main()