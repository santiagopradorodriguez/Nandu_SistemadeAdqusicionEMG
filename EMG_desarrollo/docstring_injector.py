# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo docstring_injector.py del sistema NANDU LSD.
# ==============================================================================

import ast
import os

files_to_process = [
    'gui_app/main_app.py',
    'acquisition/autoforge_daq.py',
    'acquisition/manual_daq.py',
    'analysis/plotter_calibrado.py',
    'gui_app/views/config_dialog.py',
    'views/config_dialog.py'
]

def generate_docstring(node):
    if isinstance(node, ast.ClassDef):
        return f'"""\nClase {node.name}.\n\nRepresenta y gestiona las operaciones relacionadas con {node.name}.\n"""'
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [arg.arg for arg in node.args.args if arg.arg != 'self']
        doc = f'"""\nEjecuta la funcionalidad de {node.name}.\n\n'
        if args:
            doc += "Args:\n"
            for a in args:
                doc += f"    {a} (Any): Argumento posicional {a}.\n"
            doc += "\n"
        doc += 'Returns:\n    Any: Resultado de la ejecución de la función.\n"""'
        return doc
    return ""

for path in files_to_process:
    if not os.path.exists(path):
        print(f"Skipping {path}, not found.")
        continue
        
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()
        
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in {path}: {e}")
        continue
        
    lines = source.split('\n')
    insertions = []
    
    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            if not ast.get_docstring(node):
                insertions.append((node.body[0].lineno - 1, generate_docstring(node), node.col_offset))
            self.generic_visit(node)
            
        def visit_FunctionDef(self, node):
            if not ast.get_docstring(node):
                insertions.append((node.body[0].lineno - 1, generate_docstring(node), node.col_offset))
            self.generic_visit(node)
            
        def visit_AsyncFunctionDef(self, node):
            if not ast.get_docstring(node):
                insertions.append((node.body[0].lineno - 1, generate_docstring(node), node.col_offset))
            self.generic_visit(node)
            
    Visitor().visit(tree)
    
    insertions.sort(key=lambda x: x[0], reverse=True)
    for lineno, doc, col_offset in insertions:
        indent = " " * (col_offset + 4)
        doc_indented = "\n".join([(indent + line if line else "") for line in doc.split('\n')])
        lines.insert(lineno, doc_indented)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Added docstrings to {path}")
