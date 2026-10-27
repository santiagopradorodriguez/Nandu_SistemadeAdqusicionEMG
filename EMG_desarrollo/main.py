#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto de entrada para la suite Ñandú EMG Studio desde EMG_desarrollo/.
Laboratorio de Sistemas Dinámicos.
"""

import os
import sys

EMG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EMG_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if EMG_DIR not in sys.path:
    sys.path.insert(0, EMG_DIR)

from EMG_desarrollo.gui_app.main_app import main

if __name__ == "__main__":
    main()
