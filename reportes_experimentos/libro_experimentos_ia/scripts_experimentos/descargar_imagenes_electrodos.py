"""
Descarga y acondiciona imágenes de electrodos para las diapositivas de Beamer:
1. Meditrace (Ag/AgCl estándar pre-gelado descartable)
2. Premium (Ambu BlueSensor Ag/AgCl wet-gel)
3. Copa de oro (Gold Cup reutilizable)
"""
import os
import urllib.request
from PIL import Image

OUT_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/presentacion/imagenes"
os.makedirs(OUT_DIR, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

urls = {
    "electrodo_meditrace.jpg": "https://tigermedical.com/cdn/shop/files/COV31078135_c63cc7b3-7c3a-4e1c-b268-64d749a5993d_868x700.jpg?v=1688973911",
    "electrodo_premium.jpg": "https://d1jhm577bx9zey.cloudfront.net/admin/public/getimage.ashx?Crop=0&Image=/Files/Images/ambu/products/cardiology/ecg-electrodes/BlueSensor_N/BlueSensor-N-front-back-4.png&Format=jpg&Width=600&Height=&Quality=90",
    "electrodo_copa_oro.jpg": "https://lifesync.com/cdn/shop/articles/lfc1.jpg?v=1748978512&width=768"
}

for name, url in urls.items():
    dest = os.path.join(OUT_DIR, name)
    print(f"Descargando {name} desde {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp, open(dest, 'wb') as out_file:
            out_file.write(resp.read())
        # Verificar que sea una imagen válida con Pillow
        with Image.open(dest) as img:
            img.verify()
        print(f"  OK: {dest} ({os.path.getsize(dest)} bytes)")
    except Exception as e:
        print(f"  ERROR al descargar {name}: {e}")
