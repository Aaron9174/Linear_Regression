# |_____________________________|
# | Linear Regression Algorithm |
# | Author: LeMi11ion           |
# | Cred: sentdex, YT Channel   |
# |_____________________________|

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use("fivethirtyeight")

# Mock data
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val-=step
        xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


# Least square slope formula
#     mean(x) * mean(y) - mean(x*y)
# m = -----------------------------
#     [mean(x)]^2 - mean(x^2)
#
def best_fit_helper(xs,ys):
    xMean = sum(xs) / len(xs)
    yMean = sum(ys) / len(ys)

    products = [a * b for a, b in zip(xs, ys)]
    prodMean = sum(products) / len(products)

    numerator = xMean * yMean - prodMean

    xSquared = [ a * b for a,b in zip(xs, xs)]
    xSquaredMean = sum(xSquared) / len(xSquared)

    denomenator = xMean ** 2 - xSquaredMean

    # Slope of the regression line
    m = numerator / denomenator

    # Find the intercept
    b = yMean - m * xMean

    return m, b

# Define the squared error
def squared_error(ys_input, ys_output):
    return sum((ys_output - ys_input)**2)

# Determines how well the linear regression fits the data set
#       SE * Y_hat
# r^2 = ------------
#       SE * Y_Mean
#
def coefficient_of_determination(ys_input, ys_output):
    y_mean_line = [mean(ys_input) for y in ys_input]
    squared_error_regr = squared_error(ys_input, ys_output)
    squared_error_mean = squared_error(ys_input, y_mean_line)
    return 1 - (squared_error_regr/squared_error_mean)

xs, ys = create_dataset(40, 80, 2, correlation=False)

# Get intercept and slope
m, b = best_fit_helper(xs, ys)
regression_output = [ (m*x)+b for x in xs ]

# To be used later
r_squared = coefficient_of_determination(ys, regression_output)
print(r_squared)
# Plotting
plt.scatter(xs, ys)
plt.plot(xs, regression_output)
plt.show()