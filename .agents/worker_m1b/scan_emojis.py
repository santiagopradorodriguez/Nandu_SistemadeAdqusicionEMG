import os
import unicodedata
import json

repo_dir = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
target_exts = {".py", ".md", ".sh", ".bat", ".json", ".txt", ".yml", ".yaml"}
exclude_dirs = {".git", "venv", "__pycache__", ".pytest_cache", ".gemini"}

findings = []

for root, dirs, files in os.walk(repo_dir):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in target_exts:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, repo_dir)
            
            # Skip agent reports/briefings from previous runs if they are quoting audit reports, but check code thoroughly!
            # Actually let's scan EVERYTHING in repo_dir
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        for col_num, ch in enumerate(line, 1):
                            cat = unicodedata.category(ch)
                            code = ord(ch)
                            
                            # Emoji ranges
                            is_emoji = (
                                0x1F300 <= code <= 0x1FAFF or
                                0x2600 <= code <= 0x27BF or
                                0x1F600 <= code <= 0x1F64F or
                                0x1F680 <= code <= 0x1F6FF or
                                0x2700 <= code <= 0x27BF or
                                0xFE00 <= code <= 0xFE0F or
                                (cat in ("So", "Sk") and code > 127 and ch not in "±°µ≤≥≠≈∞√∫ªº")
                            )
                            if is_emoji:
                                findings.append({
                                    "file": rel_path,
                                    "line": line_num,
                                    "col": col_num,
                                    "code_hex": hex(code),
                                    "name": unicodedata.name(ch, "UNKNOWN"),
                                    "char": ch,
                                    "line_text": line.strip()
                                })
            except Exception as e:
                print(f"Error reading {rel_path}: {e}")

output_json = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/worker_m1b/emoji_findings.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2, ensure_ascii=False)

print(f"TOTAL_FINDINGS={len(findings)}")
for item in findings:
    print(f"{item['file']}:{item['line']} [{item['code_hex']} - {item['name']}] ({item['char']}) -> {item['line_text']}")
