import zipfile, os, struct
import xml.etree.ElementTree as ET

doc_dir = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
src_docx = os.path.join(doc_dir, "Cuaderno Tesis.docx")
out_docx = os.path.join(doc_dir, "Cuaderno Tesis_actualizado.docx")

base_dir = os.path.join(doc_dir, "EMG_desarrollo/base_de_datos_electrodos/2026-08-25")

images_to_add = []

# Mosaic
caption_mosaic = "Figura: Comparativa de los patrones musculares promedio suavizados para las 5 vocales (A, E, I, O, U)."
mosaic_path = os.path.join(doc_dir, "patrones_musculares_comparativa_5vocales.png")
if os.path.exists(mosaic_path):
    images_to_add.append({"path": mosaic_path, "caption": caption_mosaic, "max_cx": 5486400})

# Vocal A
caption_paper_a = "Figura: Registro combinado de la vocal A. Señal con ruido restado entre pulsos, alineada y normalizada."
paper_a = os.path.join(base_dir, "a_Prueba1_Candela", "plot_paper_combined.png")
if os.path.exists(paper_a):
    images_to_add.append({"path": paper_a, "caption": caption_paper_a, "max_cx": 5486400})

caption_calib_a = "Figura: Señales calibradas de la vocal A. Se aplicó filtro notch, filtro pasabanda (20-500 Hz) y envolvente RMS de 75 ms."
calib_a = os.path.join(base_dir, "a_Prueba1_Candela", "plot_calibrado_2026-08-25_a_Prueba1_Candela.png")
if os.path.exists(calib_a):
    images_to_add.append({"path": calib_a, "caption": caption_calib_a, "max_cx": 4200000})

# Vocales E, I
caption_paper_ei = "Figura: Registro combinado de las vocales E e I. Señales con ruido restado entre pulsos, alineadas y normalizadas."
paper_ei = os.path.join(doc_dir, "plot_paper_par_ei.png")
if os.path.exists(paper_ei):
    images_to_add.append({"path": paper_ei, "caption": caption_paper_ei, "max_cx": 5943600})

caption_calib_ei = "Figura: Señales calibradas de las vocales E e I. Se aplicó filtro notch, filtro pasabanda (20-500 Hz) y envolvente RMS de 75 ms."
par_ei = os.path.join(doc_dir, "plot_calibrado_par_ei.png")
if os.path.exists(par_ei):
    images_to_add.append({"path": par_ei, "caption": caption_calib_ei, "max_cx": 5943600})

# Vocales O, U
caption_paper_ou = "Figura: Registro combinado de las vocales O y U. Señales con ruido restado entre pulsos, alineadas y normalizadas."
paper_ou = os.path.join(doc_dir, "plot_paper_par_ou.png")
if os.path.exists(paper_ou):
    images_to_add.append({"path": paper_ou, "caption": caption_paper_ou, "max_cx": 5943600})

caption_calib_ou = "Figura: Señales calibradas de las vocales O y U. Se aplicó filtro notch, filtro pasabanda (20-500 Hz) y envolvente RMS de 75 ms."
par_ou = os.path.join(doc_dir, "plot_calibrado_par_ou.png")
if os.path.exists(par_ou):
    images_to_add.append({"path": par_ou, "caption": caption_calib_ou, "max_cx": 5943600})

with zipfile.ZipFile(src_docx, "r") as zin:
    files = {name: zin.read(name) for name in zin.namelist()}

rels_xml = files["word/_rels/document.xml.rels"].decode("utf-8")
rel_entries = ""

for i, img in enumerate(images_to_add):
    r_id = f"rId{41 + i}"
    doc_id = 34 + i
    img_target = f"media/image{doc_id}.png"
    img["r_id"] = r_id
    img["doc_id"] = doc_id
    
    with open(img["path"], "rb") as fimg:
        img_bytes = fimg.read()
        files[f"word/{img_target}"] = img_bytes
        w, h = struct.unpack(">LL", img_bytes[16:24])
        img["cx"] = img["max_cx"]
        img["cy"] = int(img["max_cx"] * h / w)
        
    rel_entries += f'<Relationship Id="{r_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{img_target}"/>'

rels_xml = rels_xml.replace("</Relationships>", f"{rel_entries}</Relationships>")
files["word/_rels/document.xml.rels"] = rels_xml.encode("utf-8")

doc_xml = files["word/document.xml"].decode("utf-8")
root = ET.fromstring(doc_xml)
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
body = root.find("w:body", ns)
children = list(body)

start_cut_idx = None
for i, child in enumerate(children):
    texts = "".join([n.text for n in child.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t") if n.text])
    if "Baterias 8,30" in texts:
        start_cut_idx = i
        break

for child in children[start_cut_idx:-1]:
    body.remove(child)

def create_p_xml(text="", bold=False, italic=False, center=False, style=None):
    p_elem = ET.Element("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")
    pPr = ET.SubElement(p_elem, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
    if style:
        pStyle = ET.SubElement(pPr, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pStyle")
        pStyle.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", style)
    if center:
        jc = ET.SubElement(pPr, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}jc")
        jc.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "center")
    if text:
        r = ET.SubElement(p_elem, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r")
        rPr = ET.SubElement(r, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr")
        if bold:
            ET.SubElement(rPr, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}b")
        if italic:
            i_elem = ET.SubElement(rPr, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}i")
            i_elem.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "1")
        t = ET.SubElement(r, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
    return p_elem

def create_img_p_xml(r_id, doc_id, filename, cx, cy):
    p_elem = ET.Element("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")
    pPr = ET.SubElement(p_elem, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
    jc = ET.SubElement(pPr, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}jc")
    jc.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "center")
    r = ET.SubElement(p_elem, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r")
    drawing = ET.SubElement(r, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing")
    inline = ET.SubElement(drawing, "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline", {"distT": "114300", "distB": "114300", "distL": "114300", "distR": "114300"})
    ET.SubElement(inline, "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent", {"cx": str(cx), "cy": str(cy)})
    ET.SubElement(inline, "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}effectExtent", {"l": "0", "t": "0", "r": "0", "b": "0"})
    ET.SubElement(inline, "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr", {"id": str(doc_id), "name": filename})
    graphic = ET.SubElement(inline, "{http://schemas.openxmlformats.org/drawingml/2006/main}graphic")
    graphicData = ET.SubElement(graphic, "{http://schemas.openxmlformats.org/drawingml/2006/main}graphicData", {"uri": "http://schemas.openxmlformats.org/drawingml/2006/picture"})
    pic = ET.SubElement(graphicData, "{http://schemas.openxmlformats.org/drawingml/2006/picture}pic")
    nvPicPr = ET.SubElement(pic, "{http://schemas.openxmlformats.org/drawingml/2006/picture}nvPicPr")
    ET.SubElement(nvPicPr, "{http://schemas.openxmlformats.org/drawingml/2006/picture}cNvPr", {"id": "0", "name": filename})
    ET.SubElement(nvPicPr, "{http://schemas.openxmlformats.org/drawingml/2006/picture}cNvPicPr", {"preferRelativeResize": "0"})
    blipFill = ET.SubElement(pic, "{http://schemas.openxmlformats.org/drawingml/2006/picture}blipFill")
    ET.SubElement(blipFill, "{http://schemas.openxmlformats.org/drawingml/2006/main}blip", {"{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed": r_id})
    ET.SubElement(blipFill, "{http://schemas.openxmlformats.org/drawingml/2006/main}srcRect", {"l": "0", "t": "0", "r": "0", "b": "0"})
    stretch = ET.SubElement(blipFill, "{http://schemas.openxmlformats.org/drawingml/2006/main}stretch")
    ET.SubElement(stretch, "{http://schemas.openxmlformats.org/drawingml/2006/main}fillRect")
    spPr = ET.SubElement(pic, "{http://schemas.openxmlformats.org/drawingml/2006/picture}spPr")
    xfrm = ET.SubElement(spPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm")
    ET.SubElement(xfrm, "{http://schemas.openxmlformats.org/drawingml/2006/main}off", {"x": "0", "y": "0"})
    ET.SubElement(xfrm, "{http://schemas.openxmlformats.org/drawingml/2006/main}ext", {"cx": str(cx), "cy": str(cy)})
    prstGeom = ET.SubElement(spPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}prstGeom", {"prst": "rect"})
    ET.SubElement(prstGeom, "{http://schemas.openxmlformats.org/drawingml/2006/main}avLst")
    return p_elem

new_elements = [
    create_p_xml("Mediciones del 25 de agosto de 2026", bold=True, style="Heading1"),
    create_p_xml("Baterías:", bold=True),
    create_p_xml("8.30 V y 8.20 V."),
    create_p_xml("Tierra:", bold=True),
    create_p_xml("El electrodo de tierra lo pusimos en la frente."),
    create_p_xml(),
    create_p_xml("Armado y problemas con los electrodos:", bold=True),
    create_p_xml("Usamos 7 electrodos nuevos, pero los modificamos: los cortamos para que queden muy chicos. Les sacamos casi todo el borde adhesivo, dejando solo la chapa de metal, y les pusimos una planchita de gel abajo."),
    create_p_xml("El problema fue que el gel era muy difícil de controlar y los electrodos se caían muy fácil. Tuvimos que pegarlos con cinta y hacerles un refuerzo doble para que se queden en su lugar. Seguramente esto metió bastante ruido y artefactos en la señal por los falsos contactos o tirones de la cinta al hablar."),
    create_p_xml(),
    create_p_xml("Ubicación de los músculos:", bold=True),
    create_p_xml("Medimos en tres músculos a la vez:"),
    create_p_xml("- Canal 0 (Orbicular): En el labio."),
    create_p_xml("- Canal 1 (Milohioideo): En vez de ponerlo bien al centro debajo de la pera, lo pusimos desplazado hacia la derecha, casi cerca de la amígdala. Pensamos que por ahí podíamos separar mejor la E de la I. El detalle es que este canal quedó muy ruidoso."),
    create_p_xml("- Canal 2 (Vientre anterior del digástrico): Este sí lo pusimos centrado debajo de la pera."),
    create_p_xml(),
    create_p_xml("Secuencia del experimento:", bold=True),
    create_p_xml("El objetivo principal era medir especialmente la O y la U."),
    create_p_xml("Las tareas que hicimos fueron:"),
    create_p_xml("1. \"OU\" 10 veces."),
    create_p_xml("2. \"UO\" 10 veces."),
    create_p_xml("3. Secuencia \"AEIOU\"."),
    create_p_xml("4. Secuencia \"IEAU\"."),
    create_p_xml(),
    create_p_xml("Problema con el software:", bold=True),
    create_p_xml("Hubo un bug en el autograbado: cuando dije \"OU\" por segunda vez, el programa me sobrescribió los archivos de la primera vez en vez de crear datos nuevos. Es algo que tengo que arreglar en el código, pero ya estoy trabajando en eso."),
    create_p_xml(),
    create_p_xml("Gráficos y señales:", bold=True)
]

for img in images_to_add:
    new_elements.append(create_p_xml())
    new_elements.append(create_img_p_xml(img["r_id"], img["doc_id"], f"image{img['doc_id']}.png", img["cx"], img["cy"]))
    new_elements.append(create_p_xml(img["caption"], italic=True, center=True))
    new_elements.append(create_p_xml())

for elem in new_elements:
    body.insert(len(body)-1, elem)

files["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

with zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as zout:
    for name, content in files.items():
        zout.writestr(name, content)

print("Successfully regenerated:", out_docx)
