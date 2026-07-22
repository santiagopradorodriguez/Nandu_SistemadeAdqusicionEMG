import sys
import os

# Append project dir so pyqtgraph can be loaded if it's installed globally or in .venv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pyqtgraph as pg
except ImportError:
    # Try using the .venv python directly if possible
    pass

app = pg.mkQApp()
plot = pg.PlotWidget()
legend = plot.addLegend()
plot.plot([1,2], [3,4], name='Muscle 1')

for sample, label in legend.items:
    print("Has text attribute?", hasattr(label, 'text'))
    if hasattr(label, 'text'):
        print("label.text:", repr(label.text))
    print("label.opts keys:", list(label.opts.keys()))
    print("label.opts text?", label.opts.get('text', 'NOT FOUND'))
