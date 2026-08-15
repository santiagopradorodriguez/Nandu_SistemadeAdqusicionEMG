import re

with open("sections/01_hardware.tex", "r") as f:
    content = f.read()

# Locate the subsection
section_title = r"\\subsubsection\{Comparación de ruidos iniciales en los últimos experimentos\}"
parts = re.split(section_title, content)

if len(parts) == 2:
    # Replace [htbp] with [H] only in the second part (the last subsection)
    modified_second_part = parts[1].replace(r"\begin{figure}[htbp]", r"\begin{figure}[H]")
    new_content = parts[0] + section_title + modified_second_part
    with open("sections/01_hardware.tex", "w") as f:
        f.write(new_content)
    print("Fixed floats successfully.")
else:
    print("Could not find the section.")
