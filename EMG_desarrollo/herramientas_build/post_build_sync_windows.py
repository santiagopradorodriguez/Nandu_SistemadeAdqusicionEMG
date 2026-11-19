#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: herramientas_build/post_build_sync_windows.py
# Descripción: Post-procesamiento y sincronización para el empaquetado de Windows.
#              Garantiza que build_windows/NanduLsd cuente con:
#              a) _tcl_data y _tk_data (en _internal y root)
#              b) matplotlib/mpl-data completo
#              c) cv2 completo con soporte VideoCapture ffmpeg
#              d) .dist-info de nitypes y nidaqmx
#              e) Scripts empaquetados en _internal y en la raíz
#              f) Voice cues naturales en todas las rutas requeridas
#              g) NanduLsd.exe (Launcher C#) y NanduLsd_Core.exe
# ==============================================================================

import os
import sys
import shutil
import subprocess
import glob


def sync_directory(src, dst):
    if not os.path.exists(src):
        return False
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_dir = os.path.join(dst, rel)
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            sf = os.path.join(root, f)
            df = os.path.join(target_dir, f)
            if not os.path.exists(df) or os.path.getsize(sf) != os.path.getsize(df):
                try:
                    shutil.copy2(sf, df)
                except Exception as e:
                    print(f"[WARN] Error copiando {sf} -> {df}: {e}")
    return True


def post_build_sync(dist_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    emg_dir = os.path.dirname(script_dir)
    repo_root = os.path.dirname(emg_dir)

    if dist_dir is None:
        dist_dir = os.path.join(repo_root, "build_windows", "NanduLsd")

    internal_dir = os.path.join(dist_dir, "_internal")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(internal_dir, exist_ok=True)

    print(f"[POST-BUILD] Sincronizando distribucion en: {dist_dir}")

    # 1. TCL y TK Data
    py_prefix = sys.prefix
    tcl_sys = os.path.join(py_prefix, "tcl")
    if os.path.exists(tcl_sys):
        tcl86_src = os.path.join(tcl_sys, "tcl8.6")
        tk86_src = os.path.join(tcl_sys, "tk8.6")
        if os.path.exists(tcl86_src):
            sync_directory(tcl86_src, os.path.join(internal_dir, "_tcl_data"))
            sync_directory(tcl86_src, os.path.join(dist_dir, "_tcl_data"))
            print("[OK] _tcl_data sincronizado.")
        if os.path.exists(tk86_src):
            sync_directory(tk86_src, os.path.join(internal_dir, "_tk_data"))
            sync_directory(tk86_src, os.path.join(dist_dir, "_tk_data"))
            print("[OK] _tk_data sincronizado.")

    # 2. Matplotlib mpl-data
    try:
        import matplotlib
        mpl_data_src = matplotlib.get_data_path()
        if os.path.exists(mpl_data_src):
            mpl_dst = os.path.join(internal_dir, "matplotlib", "mpl-data")
            sync_directory(mpl_data_src, mpl_dst)
            print("[OK] matplotlib mpl-data sincronizado.")
    except Exception as e:
        print(f"[WARN] Error resolviendo matplotlib data: {e}")

    # 3. OpenCV cv2 completo y ffmpeg videoio
    try:
        import cv2
        cv2_src = os.path.dirname(cv2.__file__)
        cv2_dst = os.path.join(internal_dir, "cv2")
        sync_directory(cv2_src, cv2_dst)
        # Asegurar DLLs ffmpeg de OpenCV en _internal directamente
        for ffmpeg_dll in glob.glob(os.path.join(cv2_src, "*videoio_ffmpeg*.dll")):
            shutil.copy2(ffmpeg_dll, internal_dir)
            shutil.copy2(ffmpeg_dll, cv2_dst)
        print("[OK] cv2 y opencv_videoio sincronizado.")
    except Exception as e:
        print(f"[WARN] Error resolviendo cv2: {e}")

    # 4. nitypes y nidaqmx dist-info
    try:
        import importlib.metadata
        for pkg_name in ("nitypes", "nidaqmx"):
            try:
                dist = importlib.metadata.distribution(pkg_name)
                info_path = getattr(dist, "_path", None)
                if info_path and os.path.exists(str(info_path)):
                    dst_info = os.path.join(internal_dir, os.path.basename(str(info_path)))
                    sync_directory(str(info_path), dst_info)
                    print(f"[OK] dist-info {pkg_name} sincronizado en _internal.")
            except Exception as e:
                print(f"[WARN] dist-info {pkg_name}: {e}")
    except Exception as e:
        print(f"[WARN] metadata check: {e}")

    # 5. Sincronizar scripts y carpetas de código en _internal y en la raíz
    # Priorizar EMG_Ejecutable_Build para preservar los parches de PyInstaller
    build_env_dir = os.path.join(emg_dir, "EMG_Ejecutable_Build")
    code_root = build_env_dir if os.path.exists(build_env_dir) else emg_dir
    modulos_codigo = [
        "acquisition",
        "analysis",
        "deep_learning",
        "gui_app",
        "utils",
        "assets",
        "DataConfig",
    ]
    for mod in modulos_codigo:
        src_cand = os.path.join(code_root, mod)
        if not os.path.exists(src_cand):
            src_cand = os.path.join(emg_dir, mod)
        if not os.path.exists(src_cand):
            src_cand = os.path.join(repo_root, mod)
        if os.path.exists(src_cand):
            # En _internal
            sync_directory(src_cand, os.path.join(internal_dir, mod))
            # En raiz
            sync_directory(src_cand, os.path.join(dist_dir, mod))
            print(f"[OK] Modulo {mod} sincronizado.")

    # Scripts individuales y configs
    archivos_raiz = [
        ("main.py", code_root),
        ("autoforge_daq_experimental.py", os.path.join(code_root, "acquisition")),
        ("config_general.json", code_root),
        ("metronome_config.json", code_root),
        ("palabras.txt", code_root),
        ("logo_nandu_lsd.png", repo_root),
        ("icono.ico", emg_dir),
    ]
    for arc, folder in archivos_raiz:
        src_f = os.path.join(folder, arc)
        if not os.path.exists(src_f):
            src_f = os.path.join(repo_root, arc)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(dist_dir, arc))
            shutil.copy2(src_f, os.path.join(internal_dir, arc))

    # 6. Garantizar que todos los voice cues de 44.1kHz estén en todos los destinos
    voice_src = os.path.join(emg_dir, "assets", "voice_cues")
    if not os.path.exists(voice_src):
        voice_src = os.path.join(repo_root, "assets", "voice_cues")
    if os.path.exists(voice_src):
        for v_dest in [
            os.path.join(dist_dir, "assets", "voice_cues"),
            os.path.join(dist_dir, "acquisition", "assets", "voice_cues"),
            os.path.join(internal_dir, "assets", "voice_cues"),
            os.path.join(internal_dir, "acquisition", "assets", "voice_cues"),
        ]:
            sync_directory(voice_src, v_dest)
        print("[OK] Voice cues sincronizados en todas las rutas.")

    # 7. Compilar o verificar Launcher C# (NanduLsd.exe)
    launcher_cs = os.path.join(script_dir, "launcher.cs")
    nandu_exe = os.path.join(dist_dir, "NanduLsd.exe")
    core_exe = os.path.join(dist_dir, "NanduLsd_Core.exe")
    icon_path = os.path.join(emg_dir, "icono.ico")

    csc_candidates = [
        r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
        r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe",
    ]
    csc_bin = None
    for c in csc_candidates:
        if os.path.exists(c):
            csc_bin = c
            break

    if csc_bin and os.path.exists(launcher_cs):
        cmd = [
            csc_bin, "/nologo", "/target:winexe",
            f"/out:{nandu_exe}",
        ]
        if os.path.exists(icon_path):
            cmd.append(f"/win32icon:{icon_path}")
        cmd.append(launcher_cs)
        try:
            subprocess.run(cmd, check=True)
            print("[OK] Launcher C# NanduLsd.exe compilado exitosamente.")
        except Exception as e:
            print(f"[WARN] Error compilando Launcher C#: {e}")
            if os.path.exists(core_exe) and not os.path.exists(nandu_exe):
                shutil.copy2(core_exe, nandu_exe)
    elif os.path.exists(core_exe) and not os.path.exists(nandu_exe):
        shutil.copy2(core_exe, nandu_exe)
        print("[OK] NanduLsd_Core.exe duplicado como NanduLsd.exe.")

    # 8. Desactivar icuuc.dll incompatible si existe
    for root, dirs, files in os.walk(dist_dir):
        for f in files:
            if f.lower() == "icuuc.dll":
                p = os.path.join(root, f)
                try:
                    os.rename(p, p + ".disabled")
                    print(f"[OK] icuuc.dll desactivado en: {p}")
                except Exception:
                    pass

    # 9. Copiar DLLs CRT (vcruntime140 / msvcp140) de _internal a la raíz para compatibilidad en Win10
    for crt_dll in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "MSVCP140_ATOMIC_WAIT.dll", "msvcp140_codecvt_ids.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        src_crt = os.path.join(internal_dir, crt_dll)
        dst_crt = os.path.join(dist_dir, crt_dll)
        if os.path.exists(src_crt):
            try:
                shutil.copy2(src_crt, dst_crt)
            except Exception:
                pass
    print("[OK] DLLs CRT sincronizadas en la raiz.")

    print("[SUCCESS] Post-build sincronizacion completada exitosamente.")
    return True


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None
    post_build_sync(out_dir)
