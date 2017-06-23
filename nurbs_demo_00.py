# -*- coding: utf-8 -*-

from nurbs import Curve as ns
from nurbs import utilities as utils
from nurbs import factories as fact
from matplotlib import pyplot as plt

# Create a NURBS curve instance
curve = fact.from_file('data_demo/curve_00.json')
curve = fact.from_file('data_demo/curve_02.json')

# Calculate curve points
curve.evaluate_rational()

# Arrange control points for plotting
ctrlpts_x = [pt[0] for pt in curve.ctrlpts]
ctrlpts_y = [pt[1] for pt in curve.ctrlpts]


# Arrange curve points for plotting
curvepts_x = [pt[0] for pt in curve.curvepts]
curvepts_y = [pt[1] for pt in curve.curvepts]


# Plot using Matplotlib
plt.figure(figsize=(10.67, 8), dpi=96)
cppolygon, = plt.plot(ctrlpts_x, ctrlpts_y, "k.")
curveplt, = plt.plot(curvepts_x, curvepts_y, "r-")
plt.legend([cppolygon, curveplt], ["Control Points Polygon", "Evaluated Curve"])
plt.show()
