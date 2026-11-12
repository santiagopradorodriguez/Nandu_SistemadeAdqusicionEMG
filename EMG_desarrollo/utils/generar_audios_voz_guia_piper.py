#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Genera los audios de voz guía de AutoForge (assets/voice_cues/)
#              usando la misma voz Piper del asistente TARS, en WAV PCM16 mono
#              con el silencio inicial recortado para que el ataque coincida
#              exactamente con el pulso del metrónomo.
#
# Uso típico:
#   venv/bin/python generar_audios_voz_guia.py            # genera lo que falte
#   venv/bin/python generar_audios_voz_guia.py --force    # regenera todo
#   venv/bin/python generar_audios_voz_guia.py --dry-run  # muestra el plan
#   venv/bin/python generar_audios_voz_guia.py --outdir /ruta/voice_cues
#
# El reproductor (VoiceCuePlayer) busca los WAV en:
#   1. EMG_desarrollo/acquisition/assets/voice_cues
#   2. <raíz del proyecto>/assets/voice_cues   <-- destino por defecto
#   3. <raíz del proyecto>/voice_cues
# ==============================================================================

import argparse
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import wave

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy es dependencia dura del proyecto
    sys.exit("ERROR: falta numpy. Usar el venv del proyecto: venv/bin/python generar_audios_voz_guia.py")


# ------------------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------------------

RAIZ_SCRIPT = pathlib.Path(__file__).resolve().parent
# El script vive en la raíz del repo; la app vive en EMG_desarrollo/. Si algún día
# se mueve adentro, cualquiera de las dos ubicaciones resuelve la ruta correcta.
if (RAIZ_SCRIPT / "EMG_desarrollo").is_dir():
    RAIZ_PROYECTO = RAIZ_SCRIPT / "EMG_desarrollo"
elif RAIZ_SCRIPT.name == "EMG_desarrollo":
    RAIZ_PROYECTO = RAIZ_SCRIPT
else:
    RAIZ_PROYECTO = RAIZ_SCRIPT / "EMG_desarrollo"

DIR_SALIDA_DEFECTO = RAIZ_PROYECTO / "assets" / "voice_cues"

PIPER_DEFECTO = pathlib.Path("~/.local/bin/piper/piper").expanduser()
MODELO_DEFECTO = pathlib.Path("~/.local/share/piper-voices/es_ES-sharvard-medium.onnx").expanduser()

# Piper sintetiza a 22050 Hz PCM16 mono: se deja el rate nativo (sin remuestreo)
# para no introducir artefactos. VoiceCuePlayer reproduce cada WAV con su propio
# sample rate, así que no hace falta forzar 24/44.1 kHz.
SAMPLE_RATE_NATIVO = 22050

LEAD_MS = 5.0          # silencio inicial conservado (la orden pide < 10 ms)
TAIL_MS = 40.0         # cola de silencio para evitar clicks al cortar
FADE_MS = 1.5          # ramp de entrada/salida sobre el audio recortado
OBJETIVO_DBFS = -1.0   # pico normalizado
UMBRAL_SILENCIO = 0.01  # fracción del pico para detectar inicio/fin de voz
MAX_VOCAL_MS = 350.0   # límite duro para las vocales A/E/I/O/U
LENGTH_SCALE_DEFECTO = 0.85  # vocales cortas y claras (Piper 1.0 = neutro)

# clave -> (texto que se dice, duración máxima en ms o None, length_scale)
# Las claves son las que usa VoiceCuePlayer.say() en autoforge_daq_experimental.py.
GUIONES = {
    "haz_silencio": ("Haz silencio", None, 1.0),
    "preparate": ("Prepárate", None, 1.0),
    "contrae_ahora": ("Contrae ahora", None, 1.0),
    "descanso": ("Descanso", None, 1.0),
    "ruido_final": ("Haz silencio, ruido final", None, 1.0),
    "a": ("A", MAX_VOCAL_MS, LENGTH_SCALE_DEFECTO),
    "e": ("E", MAX_VOCAL_MS, LENGTH_SCALE_DEFECTO),
    "i": ("I", MAX_VOCAL_MS, LENGTH_SCALE_DEFECTO),
    "o": ("O", MAX_VOCAL_MS, LENGTH_SCALE_DEFECTO),
    "u": ("U", MAX_VOCAL_MS, LENGTH_SCALE_DEFECTO),
}


# ------------------------------------------------------------------------------
# Utilidades de audio (stdlib wave + numpy)
# ------------------------------------------------------------------------------

def leer_wav_pcm16(ruta: pathlib.Path):
    """Devuelve (muestras float32 en [-1, 1], sample_rate) de un WAV PCM16."""
    with wave.open(str(ruta), "rb") as wf:
        canales = wf.getnchannels()
        ancho = wf.getsampwidth()
        sr = wf.getframerate()
        crudo = wf.readframes(wf.getnframes())
    if ancho != 2:
        raise ValueError(f"{ruta.name}: se esperaba PCM16, ancho de muestra = {ancho}")
    datos = np.frombuffer(crudo, dtype="<i2").astype(np.float32) / 32768.0
    if canales > 1:
        datos = datos.reshape(-1, canales).mean(axis=1)
    return datos, int(sr)


def escribir_wav_pcm16(ruta: pathlib.Path, muestras: np.ndarray, sr: int) -> None:
    """Escribe un WAV PCM16 mono, con pico recortado para no saturar."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = np.clip(np.asarray(muestras, dtype=np.float32), -1.0, 1.0)
    enteros = np.rint(datos * 32767.0).astype("<i2")
    with wave.open(str(ruta), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(enteros.tobytes())


def _rampa(n: int) -> np.ndarray:
    """Rampa de medio coseno (sin clicks) de n muestras."""
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float32)
    x = np.linspace(0.0, math.pi, n, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(x)).astype(np.float32)


def recortar(muestras: np.ndarray, sr: int) -> np.ndarray:
    """Recorta silencio inicial (dejando LEAD_MS) y final (dejando TAIL_MS)."""
    if muestras.size == 0:
        return muestras
    pico = float(np.max(np.abs(muestras)))
    if pico <= 0.0:
        return np.zeros(int(sr * (LEAD_MS + TAIL_MS) / 1000.0), dtype=np.float32)
    umbral = max(1e-4, UMBRAL_SILENCIO * pico)
    activos = np.nonzero(np.abs(muestras) > umbral)[0]
    if activos.size == 0:
        return np.zeros(int(sr * (LEAD_MS + TAIL_MS) / 1000.0), dtype=np.float32)
    lead = int(round(LEAD_MS * sr / 1000.0))
    cola = int(round(TAIL_MS * sr / 1000.0))
    inicio = max(0, int(activos[0]) - lead)
    fin = min(muestras.size, int(activos[-1]) + 1 + cola)
    recorte = muestras[inicio:fin].copy()
    # Ramp corto sobre el bloque recortado (no sobre el silencio conservado):
    # si el corte cayó en medio de una oscilación, evita el click.
    n_rampa = min(int(round(FADE_MS * sr / 1000.0)), int(activos[0]) - inicio, recorte.size)
    recorte[:n_rampa] *= _rampa(n_rampa)
    return recorte


def limitar_duracion(muestras: np.ndarray, sr: int, max_ms: float):
    """Trunca a max_ms con fade-out de 10 ms. Devuelve (audio, truncado)."""
    maximo = int(round(max_ms * sr / 1000.0))
    if muestras.size <= maximo:
        return muestras, False
    recorte = muestras[:maximo].copy()
    n_fade = min(int(round(10.0 * sr / 1000.0)), recorte.size)
    if n_fade > 1:
        recorte[-n_fade:] *= _rampa(n_fade)[::-1]
    return recorte, True


def normalizar(muestras: np.ndarray) -> np.ndarray:
    """Normaliza el pico a OBJETIVO_DBFS."""
    pico = float(np.max(np.abs(muestras))) if muestras.size else 0.0
    if pico <= 0.0:
        return muestras
    objetivo = 10.0 ** (OBJETIVO_DBFS / 20.0)
    return (muestras * (objetivo / pico)).astype(np.float32)


def metricas(muestras: np.ndarray, sr: int) -> dict:
    pico = float(np.max(np.abs(muestras))) if muestras.size else 0.0
    umbral = max(1e-4, UMBRAL_SILENCIO * pico) if pico > 0 else 1.0
    activos = np.nonzero(np.abs(muestras) > umbral)[0]
    lead_ms = (float(activos[0]) / sr * 1000.0) if activos.size else 0.0
    return {
        "duracion_ms": muestras.size / sr * 1000.0,
        "lead_ms": lead_ms,
        "peak_dbfs": 20.0 * math.log10(pico) if pico > 0 else -math.inf,
    }


# ------------------------------------------------------------------------------
# Piper
# ------------------------------------------------------------------------------

def resolver_binario_piper(ruta_cli: str = None) -> pathlib.Path:
    """Busca el binario de Piper: --piper, ruta conocida, PATH y ~/.local/bin."""
    candidatos = []
    if ruta_cli:
        candidatos.append(pathlib.Path(ruta_cli).expanduser())
    candidatos.append(PIPER_DEFECTO)
    en_path = shutil.which("piper")
    if en_path:
        candidatos.append(pathlib.Path(en_path))
    for candidato in candidatos:
        if candidato.is_file() and os.access(str(candidato), os.X_OK):
            return candidato
    return None


def resolver_modelo_piper(ruta_cli: str = None) -> pathlib.Path:
    candidatos = []
    if ruta_cli:
        candidatos.append(pathlib.Path(ruta_cli).expanduser())
    candidatos.append(MODELO_DEFECTO)
    for candidato in candidatos:
        if candidato.is_file():
            return candidato
    return None


def sintetizar(piper: pathlib.Path, modelo: pathlib.Path, texto: str,
               destino: pathlib.Path, length_scale: float) -> None:
    """Sintetiza texto a un WAV temporal con Piper (stdin -> --output_file)."""
    entorno = dict(os.environ)
    entorno.setdefault("LC_ALL", "C.utf8")
    entorno.setdefault("LANG", "C.utf8")
    comando = [
        str(piper), "--model", str(modelo), "--output_file", str(destino),
        "--length_scale", f"{float(length_scale):.2f}",
    ]
    resultado = subprocess.run(
        comando,
        input=str(texto).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        env=entorno,
    )
    if resultado.returncode != 0 or not destino.is_file() or destino.stat().st_size == 0:
        detalle = (resultado.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"Piper falló (código {resultado.returncode}): {detalle[:300]}")


# ------------------------------------------------------------------------------
# Generación
# ------------------------------------------------------------------------------

def generar_cue(clave: str, texto: str, max_ms, length_scale: float,
                piper: pathlib.Path, modelo: pathlib.Path,
                dir_salida: pathlib.Path, sample_rate: int = None) -> dict:
    """Sintetiza, recorta, normaliza y guarda un cue. Devuelve métricas."""
    with tempfile.TemporaryDirectory(prefix="voz_guia_") as tmp:
        crudo = pathlib.Path(tmp) / f"{clave}_piper.wav"
        sintetizar(piper, modelo, texto, crudo, length_scale)
        muestras, sr = leer_wav_pcm16(crudo)

    recorte = recortar(muestras, sr)
    truncado = False
    if max_ms:
        recorte, truncado = limitar_duracion(recorte, sr, max_ms)
    recorte = normalizar(recorte)

    if sample_rate and int(sample_rate) != sr:
        recorte = remuestrear(recorte, sr, int(sample_rate))
        sr = int(sample_rate)

    destino = dir_salida / f"{clave}.wav"
    escribir_wav_pcm16(destino, recorte, sr)

    info = metricas(recorte, sr)
    info.update({
        "clave": clave,
        "texto": texto,
        "ruta": destino,
        "sr": sr,
        "truncado": truncado,
        "bytes": destino.stat().st_size,
    })
    return info


def remuestrear(muestras: np.ndarray, sr_origen: int, sr_destino: int) -> np.ndarray:
    """Remuestreo lineal simple (solo si el usuario fuerza --rate)."""
    if sr_origen == sr_destino or muestras.size == 0:
        return muestras
    n_destino = max(1, int(round(muestras.size * sr_destino / sr_origen)))
    x_origen = np.arange(muestras.size, dtype=np.float64)
    x_destino = np.linspace(0.0, muestras.size - 1, n_destino, dtype=np.float64)
    return np.interp(x_destino, x_origen, muestras).astype(np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera los audios de voz guía de AutoForge con la voz Piper del asistente.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir", default=str(DIR_SALIDA_DEFECTO),
                        help="Carpeta de salida de los .wav")
    parser.add_argument("--piper", default=None, help="Ruta al binario de Piper")
    parser.add_argument("--model", default=None, help="Ruta al modelo .onnx de Piper")
    parser.add_argument("--rate", type=int, default=None,
                        help="Fuerza un sample rate de salida (por defecto el nativo de Piper)")
    parser.add_argument("--length-scale", type=float, default=None,
                        help="Sobrescribe el length_scale de todos los guiones")
    parser.add_argument("--force", action="store_true",
                        help="Regenera aunque el .wav ya exista")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra qué se generaría sin escribir nada")
    parser.add_argument("--solo", default=None,
                        help="Genera solo estas claves (separadas por coma)")
    args = parser.parse_args(argv)

    dir_salida = pathlib.Path(args.outdir).expanduser().resolve()
    claves = list(GUIONES.keys())
    if args.solo:
        pedidas = [c.strip().lower() for c in args.solo.split(",") if c.strip()]
        desconocidas = [c for c in pedidas if c not in GUIONES]
        if desconocidas:
            print(f"ERROR: claves desconocidas: {', '.join(desconocidas)}")
            print(f"Claves válidas: {', '.join(GUIONES)}")
            return 2
        claves = pedidas

    pendientes = [c for c in claves if args.force or not (dir_salida / f"{c}.wav").is_file()]
    salteadas = [c for c in claves if c not in pendientes]

    print(f"Proyecto      : {RAIZ_PROYECTO}")
    print(f"Destino       : {dir_salida}")
    print(f"Guiones       : {len(claves)}  (a generar: {len(pendientes)}, ya existen: {len(salteadas)})")
    if salteadas:
        print(f"Ya existen    : {', '.join(salteadas)}  (usar --force para regenerar)")

    if args.dry_run:
        print("\n[dry-run] No se escribe nada. Plan:")
        for clave in pendientes:
            texto, max_ms, ls = GUIONES[clave]
            limite = f", máx {max_ms:.0f} ms" if max_ms else ""
            print(f"  {clave:<15} \"{texto}\"  (length_scale {ls:.2f}{limite})")
        return 0

    if not pendientes:
        print("\nNada para hacer: todos los audios ya existen.")
        return 0

    piper = resolver_binario_piper(args.piper)
    modelo = resolver_modelo_piper(args.model)
    if piper is None or modelo is None:
        print("\nERROR: no se encontró Piper o el modelo de voz, así que no se pueden generar los WAV.")
        print(f"  Binario Piper : {piper or 'NO ENCONTRADO'}")
        print(f"  Modelo .onnx  : {modelo or 'NO ENCONTRADO'}")
        print("\nOpciones:")
        print("  1. Instalar Piper 1.2 con la voz del asistente (es_ES-sharvard-medium).")
        print("  2. Pasar rutas explícitas: --piper /ruta/piper --model /ruta/voz.onnx")
        print("  3. Copiar a mano WAVs PCM16 mono con los nombres de las claves.")
        print("\nLa app sigue funcionando sin estos archivos: VoiceCuePlayer simplemente no reproduce nada.")
        return 1

    print(f"Piper         : {piper}")
    print(f"Modelo        : {modelo}\n")

    dir_salida.mkdir(parents=True, exist_ok=True)
    fallos = []
    generados = []

    for clave in pendientes:
        texto, max_ms, ls = GUIONES[clave]
        if args.length_scale is not None:
            ls = float(args.length_scale)
        try:
            info = generar_cue(clave, texto, max_ms, ls, piper, modelo,
                               dir_salida, sample_rate=args.rate)
        except Exception as exc:  # se informa y se sigue con el resto
            fallos.append((clave, str(exc)))
            print(f"  [FALLO] {clave:<15} {exc}")
            continue
        generados.append(info)
        aviso = ""
        if info["truncado"]:
            aviso = f"  <- truncado a {max_ms:.0f} ms"
        print(f"  [OK]    {clave:<15} {info['duracion_ms']:6.0f} ms  "
              f"lead={info['lead_ms']:4.1f} ms  pico={info['peak_dbfs']:5.1f} dBFS  "
              f"{info['bytes']:>6} bytes{aviso}")

    print("\nVerificación")
    print(f"  Archivos en {dir_salida}: {len(list(dir_salida.glob('*.wav')))}")
    lentas = [g for g in generados if g["clave"] in "aeiou" and g["duracion_ms"] > MAX_VOCAL_MS + 1.0]
    for g in lentas:
        print(f"  ADVERTENCIA: la vocal {g['clave']} dura {g['duracion_ms']:.0f} ms "
              f"(límite {MAX_VOCAL_MS:.0f} ms)")
    desfasadas = [g for g in generados if g["lead_ms"] > 10.0]
    for g in desfasadas:
        print(f"  ADVERTENCIA: {g['clave']} arranca a {g['lead_ms']:.1f} ms (se pedía < 10 ms)")

    if fallos:
        print(f"\n{len(fallos)} audio(s) no se pudieron generar.")
        return 1

    print("\nListo. En el ejecutable congelado, copiar la carpeta assets/voice_cues")
    print("junto al .exe (o dentro de la raíz del proyecto) para que la app la encuentre.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
