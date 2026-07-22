import pyqtgraph as pg
app = pg.mkQApp()
plot = pg.PlotWidget()
ax = plot.getAxis('bottom')
print(hasattr(ax, 'pen'))
print(hasattr(ax, 'textPen'))
