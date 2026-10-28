# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Servidor HTTP de estímulos para el display remoto (celu/tablet).
#              Expone el estado actual (palabra, metrónomo, sesión) y la página
#              que la pantalla remota muestra en pantalla completa.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
Servidor de estímulos del display remoto (solo biblioteca estándar).

Diseño:
  - Corre en un hilo daemon (clase ServidorEstimulos) y puede embeberse en la
    aplicación principal (AutoForge DAQ) o lanzarse standalone en modo demo.
  - El estado se publica con set_estado(**kw); cada llamada incrementa
    `secuencia`, que el cliente usa para detectar cambios por polling.
  - El cliente HTML se lee del disco en cada request (sin cache), así se puede
    editar en caliente. Si todavía no existe, se devuelve un aviso simple.

Endpoints:
  GET /                    -> cliente_estimulos.html (o aviso si falta).
  GET /estado?t=<token>    -> JSON con el estado completo + ts_epoch (ms).
  GET /sync?c=<ms_cliente> -> {"c": <eco>, "s": <epoch_ms>} para estimar offset.

Uso standalone:
  python display_remoto_servidor.py --demo --puerto=8765
  python display_remoto_servidor.py --demo=1.5 --puerto=9000 --token=secreto

Nota: el token, si se configura, se exige únicamente en /estado. La página y
/sync quedan abiertos para que el handshake de reloj sea barato.
"""

import argparse
import hmac
import html
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# --- Rutas del proyecto (multiplataforma) -------------------------------------
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
RUTA_CLIENTE = os.path.join(DIR_SCRIPT, "display_remoto", "cliente_estimulos.html")
RUTA_PALABRAS = os.path.join(DIR_SCRIPT, "palabras.txt")

# Palabras de respaldo por si palabras.txt no existe o está vacío.
PALABRAS_RESPALDO = ["A", "E", "I", "O", "U"]

# Claves válidas del estado publicado.
CLAVES_ESTADO = (
    "palabra",
    "bpm",
    "subdiv",
    "sesion",
    "grabando",
    "t0_epoch",
    "t0_perf_counter",
)


def _epoch_ms():
    """Devuelve el reloj de pared en milisegundos desde la época Unix."""
    return int(time.time() * 1000)


def _leer_palabras(ruta=RUTA_PALABRAS):
    """Lee palabras.txt (UTF-8, una por línea). Si falla, usa vocales."""
    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            palabras = [linea.strip() for linea in archivo]
        palabras = [p for p in palabras if p and not p.startswith("#")]
        if palabras:
            return palabras
    except OSError:
        pass
    return list(PALABRAS_RESPALDO)


def _ip_local():
    """Obtiene la IP de la interfaz usada para salir a la red (sin enviar datos)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # connect() en UDP no manda paquetes; solo resuelve la interfaz de salida.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class ServidorEstimulos(threading.Thread):
    """Servidor HTTP en hilo daemon con el estado del estímulo.

    Ejemplo de uso embebido:
        servidor = ServidorEstimulos(puerto=8765, token="abc")
        servidor.start()
        servidor.set_estado(palabra="A", bpm=60, grabando=True)
        ...
        servidor.detener()
    """

    def __init__(self, puerto=8765, token=None, host="0.0.0.0",
                 ruta_cliente=RUTA_CLIENTE, log_accesos=False):
        super().__init__(name="ServidorEstimulos", daemon=True)
        self.puerto = int(puerto)
        self.token = token if token else None
        self.host = host
        self.ruta_cliente = ruta_cliente
        self.log_accesos = bool(log_accesos)

        self._lock = threading.RLock()
        self._secuencia = 0
        self._estado = {
            "palabra": "",
            "bpm": 0.0,
            "subdiv": 4,
            "sesion": None,
            "grabando": False,
            "t0_epoch": None,
            "t0_perf_counter": None,
        }

        self._httpd = None
        self._listo = threading.Event()
        self._parar = threading.Event()
        self.error = None

    # --- Estado ----------------------------------------------------------
    @property
    def secuencia(self):
        """Último valor del contador de cambios (thread-safe)."""
        with self._lock:
            return self._secuencia

    def set_estado(self, **kw):
        """Actualiza el estado de forma thread-safe.

        Solo se actualizan las claves pasadas; las desconocidas se ignoran.
        Cada llamada (aunque no traiga claves) incrementa `secuencia` para que
        el cliente detecte el cambio al hacer polling. Devuelve la secuencia.
        """
        with self._lock:
            for clave, valor in kw.items():
                if clave in CLAVES_ESTADO:
                    self._estado[clave] = valor
            self._secuencia += 1
            return self._secuencia

    def obtener_estado(self):
        """Copia del estado actual (sin ts_epoch; lo agrega el endpoint)."""
        with self._lock:
            estado = dict(self._estado)
            estado["secuencia"] = self._secuencia
        return estado

    # --- Ciclo de vida ---------------------------------------------------
    def run(self):
        try:
            httpd = ThreadingHTTPServer((self.host, self.puerto), _ManejadorHTTP)
            httpd.daemon_threads = True
            httpd.estimulos = self
            httpd.log_accesos = self.log_accesos
            self._httpd = httpd
            # Si se pidió puerto 0, este es el puerto real asignado por el SO.
            self.puerto = httpd.server_address[1]
            self._listo.set()
            httpd.serve_forever(poll_interval=0.2)
        except OSError as exc:
            self.error = exc
            self._listo.set()
            print("[display_remoto] No se pudo abrir el puerto {}: {}".format(
                self.puerto, exc), file=sys.stderr)
        finally:
            if self._httpd is not None:
                try:
                    self._httpd.server_close()
                except OSError:
                    pass

    def esperar_listo(self, timeout=5.0):
        """Espera a que el socket esté escuchando. True si arrancó bien."""
        self._listo.wait(timeout)
        return self.error is None and self._httpd is not None and self.is_alive()

    def activo(self):
        """True si el hilo está vivo y escuchando."""
        return self.is_alive() and self.error is None and self._httpd is not None

    def detener(self, timeout=3.0):
        """Cierre limpio: corta serve_forever, cierra el socket y hace join."""
        self._parar.set()
        if self._httpd is not None:
            self._listo.wait(timeout)
            try:
                self._httpd.shutdown()
            except Exception:
                pass
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)
        if self._httpd is not None:
            try:
                self._httpd.server_close()
            except OSError:
                pass

    # Alias explícito para quien prefiera la API de socketserver.
    shutdown = detener


class _ManejadorHTTP(BaseHTTPRequestHandler):
    """Handler de los endpoints del display remoto."""

    server_version = "NanduDisplayRemoto/1.0"
    protocol_version = "HTTP/1.1"

    # --- Utilidades de respuesta -----------------------------------------
    def log_message(self, fmt, *args):
        # Silencioso por defecto; el servidor puede pedir log mínimo.
        if getattr(self.server, "log_accesos", False):
            super().log_message(fmt, *args)

    def _enviar(self, codigo, cuerpo, content_type):
        if isinstance(cuerpo, str):
            cuerpo = cuerpo.encode("utf-8")
        try:
            self.send_response(codigo)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(cuerpo)
        except (BrokenPipeError, ConnectionResetError):
            # El cliente cerró la conexión (p. ej. polling cancelado).
            pass

    def _json(self, codigo, datos):
        texto = json.dumps(datos, ensure_ascii=False)
        self._enviar(codigo, texto, "application/json; charset=utf-8")

    def _html(self, codigo, texto):
        self._enviar(codigo, texto, "text/html; charset=utf-8")

    def _aviso_cliente(self):
        ruta = self.server.estimulos.ruta_cliente
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"es\">\n<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>NANDU LSD - Display remoto</title>\n"
            "<style>body{background:#050505;color:#00FFFF;font-family:sans-serif;"
            "display:flex;align-items:center;justify-content:center;"
            "height:100vh;margin:0;text-align:center}h1{font-size:5vh}"
            "p{font-size:2.2vh;color:#888}</style>\n"
            "</head>\n<body>\n<div>\n"
            "<h1>Cliente no disponible todavía</h1>\n"
            "<p>Falta el archivo:</p>\n"
            "<p><code>__RUTA__</code></p>\n"
            "</div>\n</body>\n</html>\n"
        ).replace("__RUTA__", html.escape(ruta))

    # --- Endpoints --------------------------------------------------------
    def do_GET(self):
        try:
            self._procesar_get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # robustez: nunca tumbar el handler
            try:
                self._json(500, {"error": "{}: {}".format(
                    type(exc).__name__, exc)})
            except Exception:
                pass

    # HEAD se responde igual que GET pero sin cuerpo (el _enviar lo contempla).
    do_HEAD = do_GET

    def _procesar_get(self):
        partes = urlparse(self.path)
        ruta = partes.path
        consulta = parse_qs(partes.query)

        if ruta in ("/", "/index.html"):
            self._servir_cliente()
        elif ruta == "/estado":
            self._servir_estado(consulta)
        elif ruta == "/sync":
            self._servir_sync(consulta)
        elif ruta == "/favicon.ico":
            self._enviar(204, b"", "image/x-icon")
        else:
            self._json(404, {"error": "ruta no encontrada", "ruta": ruta})

    def _servir_cliente(self):
        servidor = self.server.estimulos
        try:
            # Lectura en cada request: permite editar el HTML sin reiniciar.
            with open(servidor.ruta_cliente, "rb") as archivo:
                contenido = archivo.read()
        except OSError:
            self._html(200, self._aviso_cliente())
            return
        self._enviar(200, contenido, "text/html; charset=utf-8")

    def _servir_estado(self, consulta):
        servidor = self.server.estimulos
        token = servidor.token
        if token:
            recibido = consulta.get("t", [""])[0]
            # Comparación en tiempo constante solo si ambos son ASCII.
            try:
                valido = hmac.compare_digest(str(recibido), str(token))
            except TypeError:
                valido = recibido == token
            if not valido:
                self._json(403, {"error": "token inválido"})
                return

        estado = servidor.obtener_estado()
        estado["ts_epoch"] = _epoch_ms()
        self._json(200, estado)

    def _servir_sync(self, consulta):
        # Eco del reloj del cliente para estimar el offset (NTP-lite).
        cliente = consulta.get("c", [None])[0]
        self._json(200, {"c": cliente, "s": _epoch_ms()})


def _ciclo_demo(servidor, velocidad):
    """Cicla las palabras cada `velocidad` segundos hasta Ctrl+C."""
    palabras = _leer_palabras()
    t0_epoch = time.time()
    t0_perf = time.perf_counter()
    servidor.set_estado(
        palabra=palabras[0],
        bpm=60,
        subdiv=4,
        sesion="demo",
        grabando=True,
        t0_epoch=t0_epoch,
        t0_perf_counter=t0_perf,
    )
    print("[display_remoto] Demo: {} palabra(s), ciclo de {:.2f} s".format(
        len(palabras), velocidad))
    indice = 1
    parar = servidor._parar
    try:
        while not parar.is_set():
            if parar.wait(velocidad):
                break
            servidor.set_estado(palabra=palabras[indice % len(palabras)])
            indice += 1
    finally:
        servidor.set_estado(grabando=False)


def _imprimir_urls(servidor):
    puerto = servidor.puerto
    ip = _ip_local()
    print("=" * 62)
    print(" Servidor de estímulos NANDU LSD (display remoto)")
    print("=" * 62)
    print("  Local : http://localhost:{}/".format(puerto))
    print("  Red   : http://{}:{}/".format(ip, puerto))
    if servidor.token:
        print("  Token : requerido en /estado (usar ?t=<token>)")
    else:
        print("  Token : no configurado (acceso libre)")
    print("  Estado: http://localhost:{}/estado".format(puerto))
    print("=" * 62)
    print(" Ctrl+C para detener.")
    sys.stdout.flush()


def _parsear_argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="Servidor HTTP de estímulos para el display remoto NANDU LSD.")
    parser.add_argument(
        "--puerto", type=int, default=8765,
        help="Puerto TCP de escucha (default: 8765).")
    parser.add_argument(
        "--token", default=None,
        help="Token opcional exigido por /estado.")
    parser.add_argument(
        "--demo", nargs="?", const=2.0, default=None, type=float,
        metavar="SEGUNDOS",
        help="Modo demo: cicla palabras.txt cada N segundos (default: 2.0).")
    parser.add_argument(
        "--log", action="store_true",
        help="Mostrar access log de HTTP (por defecto silencioso).")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parsear_argumentos(argv)
    servidor = ServidorEstimulos(
        puerto=args.puerto,
        token=args.token,
        log_accesos=args.log,
    )
    servidor.start()
    if not servidor.esperar_listo(timeout=5.0):
        print("[display_remoto] El servidor no pudo iniciar.", file=sys.stderr)
        servidor.detener()
        return 1

    _imprimir_urls(servidor)

    try:
        if args.demo is not None:
            if args.demo <= 0:
                print("[display_remoto] --demo debe ser > 0.", file=sys.stderr)
                return 2
            _ciclo_demo(servidor, args.demo)
        else:
            while servidor.is_alive():
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[display_remoto] Deteniendo servidor...")
    finally:
        servidor.detener()
        print("[display_remoto] Servidor detenido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
