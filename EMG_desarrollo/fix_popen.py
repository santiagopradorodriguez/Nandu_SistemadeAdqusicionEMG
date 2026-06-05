import os

files = [
    r"c:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\acquisition\autoforge_daq.py",
    r"c:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\acquisition\autoforge_daq_experimental.py"
]

for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find where text=True is and append creationflags
    # Be careful not to replace it multiple times
    if "creationflags" not in content:
        content = content.replace(
            "text=True # Para enviar texto en lugar de bytes",
            "text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000) # Evitar consola"
        )
        content = content.replace(
            "text=True",
            "text=True,\n                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Modificado {filepath}")
    else:
        print(f"Ya estaba modificado {filepath}")
