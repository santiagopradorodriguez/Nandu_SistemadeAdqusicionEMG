#!/usr/bin/env python3
"""
Auditor de Consumo de Tokens para Antigravity / Gemini.
Analiza los archivos transcript_full.jsonl en ~/.gemini/antigravity/brain/
e informa el consumo aproximado de tokens por prompt y total de la sesión.
"""

import os
import glob
import json
import sys

BRAIN_DIR = os.path.expanduser("~/.gemini/antigravity/brain")

def encontrar_sesiones_recientes():
    if not os.path.exists(BRAIN_DIR):
        print(f"No se encontro el directorio: {BRAIN_DIR}")
        return []
    
    subdirs = [os.path.join(BRAIN_DIR, d) for d in os.listdir(BRAIN_DIR) if os.path.isdir(os.path.join(BRAIN_DIR, d))]
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return subdirs

def auditar_sesion(conv_dir):
    log_file = os.path.join(conv_dir, ".system_generated", "logs", "transcript_full.jsonl")
    if not os.path.exists(log_file):
        log_file = os.path.join(conv_dir, ".system_generated", "logs", "transcript.jsonl")
    
    if not os.path.exists(log_file):
        print(f"No hay logs en: {conv_dir}")
        return
    
    conv_id = os.path.basename(conv_dir)
    print(f"\n=======================================================")
    print(f"Auditoria de Sesion: {conv_id}")
    print(f"Archivo de log: {log_file}")
    print(f"=======================================================\n")
    
    turnos = []
    total_in_chars = 0
    total_out_chars = 0
    
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                step = json.loads(line)
            except Exception:
                continue
            
            source = step.get("source", "")
            step_type = step.get("type", "")
            content = step.get("content", "") or ""
            thinking = step.get("thinking", "") or ""
            tool_calls = step.get("tool_calls", [])
            
            chars = len(content) + len(thinking) + len(json.dumps(tool_calls))
            
            if source in ["USER_EXPLICIT", "SYSTEM"]:
                total_in_chars += chars
                turnos.append({
                    "step": step.get("step_index", 0),
                    "rol": "USER/SISTEMA",
                    "chars": chars,
                    "tokens_est": int(chars / 3.5),
                    "preview": content[:60].replace("\n", " ")
                })
            elif source == "MODEL":
                total_out_chars += chars
                turnos.append({
                    "step": step.get("step_index", 0),
                    "rol": "AGENTE",
                    "chars": chars,
                    "tokens_est": int(chars / 3.5),
                    "preview": (thinking[:40] if thinking else content[:40]).replace("\n", " ")
                })

    tokens_in_tot = int(total_in_chars / 3.5)
    tokens_out_tot = int(total_out_chars / 3.5)
    tokens_gran_total = tokens_in_tot + tokens_out_tot
    
    print(f"{'Paso':<6} | {'Rol':<12} | {'Tokens Est.':<12} | {'Resumen'}")
    print("-" * 65)
    for t in turnos:
        print(f"{t['step']:<6} | {t['rol']:<12} | {t['tokens_est']:<12} | {t['preview']}")
    
    print("-" * 65)
    print(f"Tokens Entrada Estimados : {tokens_in_tot:>10,}")
    print(f"Tokens Salida Estimados  : {tokens_out_tot:>10,}")
    print(f"TOTAL Estimado Sesion    : {tokens_gran_total:>10,}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    sesiones = encontrar_sesiones_recientes()
    if not sesiones:
        sys.exit(0)
    
    # Auditar la sesion mas reciente (la actual)
    auditar_sesion(sesiones[0])
