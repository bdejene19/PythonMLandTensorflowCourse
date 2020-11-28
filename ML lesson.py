
import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 2.5, 3, 4] # x-coordinates for graph
y = [1, 4, 7, 9, 15] # y-coordinates
plt.axis([0, 6, 0, 20]) # creating my x and y axis range. 0-6 is x, 0-20 is y

plt.plot(x, y, 'ro')
# can see graph has a linear correspondence, therefore, can use linear regression that cn give us good predictions
# can create a line of best fit --> don't entirely understand syntax for line of best fit
# np.polyfit takes in x and y values, then the number of points (or connections) you want for your LBF
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()

