import sys, os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
import pyqtgraph as pg

app = QApplication(sys.argv)
plot = pg.PlotWidget()
plot.plot([1,2,3], [1000,2000,3000])

ax_left = plot.getAxis('left')
ax_bottom = plot.getAxis('bottom')

ax_left.setLabel('Amplitud', units='V')
ax_bottom.setLabel('Tiempo', units='s')

# Change fonts
font = QFont("Arial", 20, QFont.Bold)
ax_left.setTickFont(font)
ax_left.setLabel('Amplitud', units='V', **{'font-size': '24pt'})
ax_bottom.setTickFont(font)
ax_bottom.setLabel('Tiempo', units='s', **{'font-size': '24pt'})

# Force layout update?
app.processEvents()

exporter = pg.exporters.ImageExporter(plot.scene())
exporter.export('test_pg_export.png')
