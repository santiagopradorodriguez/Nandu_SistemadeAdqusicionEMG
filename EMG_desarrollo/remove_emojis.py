import os
import re

def remove_emojis(text):
    # Rango común de emojis en unicode
    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])|"  # flags (iOS)
        u"[\U00010000-\U0010ffff]|"
        u"([✏️⏹✅❌⚡🔥↗↘])",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

files_to_clean = [
    "acquisition/autoforge_daq.py",
    "acquisition/autoforge_daq_experimental.py",
    "gui_app/main_app.py",
    "utils/logger.py"
]

for filepath in files_to_clean:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove emojis and any double spaces left behind
        cleaned_content = remove_emojis(content)
        cleaned_content = cleaned_content.replace('  ', ' ')
        
        if cleaned_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Limpio: {filepath}")
        else:
            print(f"Sin cambios: {filepath}")

