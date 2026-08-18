# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Test de higiene de caracteres y detección estricta de emojis.
# ==============================================================================

import os
import unicodedata
import unittest


class TestRepoEmojiHygiene(unittest.TestCase):
    """Verifica que el repositorio esté 100% libre de emojis y caracteres unicode no conformes."""

    def test_zero_emojis_in_source_code_and_docs(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(tests_dir, "..", ".."))

        target_exts = {".py", ".md", ".sh", ".bat", ".json", ".tex"}
        exclude_dirs = {".git", ".agents", "venv", "__pycache__", ".pytest_cache", ".gemini", "build", "dist", "build_linux", "build_windows", "EMG_Ejecutable_Build", "base_de_datos_electrodos", "base_de_datos_letras", "archivos_temporales", "papers", "resultados", "resultados_pca_umap", "resultados_umap_supervisado", "analisis_comparativos"}

        allowed_special = set("±°µ≤≥≠≈∞√∫ªºáéíóúÁÉÍÓÚñÑüÜ¿¡—–“”«»’‘█─│├└┌┐┬┴┼")
        violations = []

        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in target_exts:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_root)

                    try:
                        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                            for line_num, line in enumerate(f, 1):
                                for col_num, ch in enumerate(line, 1):
                                    code = ord(ch)
                                    cat = unicodedata.category(ch)

                                    is_emoji = (
                                        0x1F300 <= code <= 0x1FAFF or
                                        0x2600 <= code <= 0x27BF or
                                        0x1F600 <= code <= 0x1F64F or
                                        0x1F680 <= code <= 0x1F6FF or
                                        0x2700 <= code <= 0x27BF or
                                        0xFE00 <= code <= 0xFE0F or
                                        (cat in ("So", "Sk") and code > 127 and ch not in allowed_special)
                                    )
                                    if is_emoji:
                                        violations.append(
                                            f"{rel_path}:{line_num}:{col_num} [{hex(code)} {unicodedata.name(ch, 'UNKNOWN')}] '{ch}' -> {line.strip()}"
                                        )
                    except Exception as e:
                        violations.append(f"Error leyendo {rel_path}: {e}")

        error_msg = "\n".join(violations)
        self.assertEqual(len(violations), 0, f"Se detectaron {len(violations)} emojis o caracteres no conformes:\n{error_msg}")


if __name__ == "__main__":
    unittest.main()
