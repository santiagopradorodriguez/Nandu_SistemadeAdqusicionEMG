import pyqtgraph as pg
import sys
import PyQt5.QtWidgets as qw

app = qw.QApplication(sys.argv)
plot = pg.PlotWidget()
legend = plot.addLegend()
plot.plot([1,2], [3,4], name='Test 1')
plot.plot([2,3], [1,5], name='Test 2')

# Let's inspect the legend object
print("Has setLabelTextColor?", hasattr(legend, 'setLabelTextColor'))
if hasattr(legend, 'setLabelTextColor'):
    legend.setLabelTextColor(pg.mkColor('k'))

# Let's see the labels
for sample, label in legend.items:
    print("Label text:", label.text)
    # What methods does label have?
    # print(dir(label))
    label.setText(label.text, color='k')
