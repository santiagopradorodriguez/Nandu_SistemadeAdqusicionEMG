#!/usr/bin/env bash
set -euo pipefail

ANTIGRAVITY_BIN="${ANTIGRAVITY_BIN:-$HOME/AppImages/Antigravity-x64/antigravity}"

if [[ ! -x "$ANTIGRAVITY_BIN" ]]; then
    echo "No se encontró el ejecutable de Antigravity: $ANTIGRAVITY_BIN" >&2
    exit 1
fi

# Chromium/Electron no puede crear su sandbox SUID en algunos entornos Linux.
# Además, algunas combinaciones de drivers X11/Vulkan provocan una violación
# de segmento durante el arranque. Forzamos renderizado por software y
# desactivamos las rutas gráficas conflictivas.
export LIBGL_ALWAYS_SOFTWARE=1
export ELECTRON_DISABLE_SANDBOX=1

exec "$ANTIGRAVITY_BIN" \
    --no-sandbox \
    --disable-gpu \
    --disable-gpu-compositing \
    --disable-gpu-sandbox \
    --disable-features=Vulkan,UseSkiaRenderer \
    "$@"
