#!/usr/bin/env python3
"""
Hook de notificación de consumo de tokens para Antigravity.
Calcula el consumo del último turno y el acumulado de la sesión,
mostrando una notificación emergente en el escritorio de Linux vía notify-send.
"""

import sys
import json
import os
import subprocess

def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            return
        
        payload = json.loads(raw_input)
        transcript_path = payload.get("transcriptPath")
        if not transcript_path or not os.path.exists(transcript_path):
            return

        total_chars = 0
        turnos = [] # cada elemento: {"user_prompt": str, "chars": int}
        current_turn_chars = 0
        current_prompt = "Prompt del usuario"

        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
                total_chars += chars

                if step_type == "USER_INPUT":
                    if current_turn_chars > 0:
                        turnos.append({"prompt": current_prompt, "chars": current_turn_chars})
                    current_turn_chars = chars
                    limpio = content.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "")
                    limpio = limpio.split("<ADDITIONAL_METADATA>")[0].strip()
                    current_prompt = (limpio[:45] + "...") if len(limpio) > 45 else limpio
                else:
                    current_turn_chars += chars

        if current_turn_chars > 0:
            turnos.append({"prompt": current_prompt, "chars": current_turn_chars})

        if not turnos:
            return

        ultimo_turno = turnos[-1]
        tokens_turno = int(ultimo_turno["chars"] / 3.5)
        tokens_total = int(total_chars / 3.5)
        prompt_txt = ultimo_turno["prompt"]

        titulo = "Antigravity • Consumo de Tokens"
        mensaje = f"Turno: ~{tokens_turno:,} tokens\nTotal sesión: ~{tokens_total:,} tokens\nPrompt: \"{prompt_txt}\""

        subprocess.run(
            ["notify-send", "-a", "Antigravity", "-t", "4000", titulo, mensaje],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    except Exception:
        pass

if __name__ == "__main__":
    main()
