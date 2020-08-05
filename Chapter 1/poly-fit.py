import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("web_traffic.tsv", delimiter = "\t")

x = data[:, 0]
y = data[:, 1]

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

def error(f, x, y):
    return np.sum((f(x) - y)**2)

# Fit data with a line

f2p = np.polyfit(x,y,3)
f = np.poly1d(f2p)
print(error(f, x, y))

# Predict when the model function reaches 100000

from scipy.optimize import fsolve
reach = fsolve(f-100000, x[-1])/(7*24)
print(reach)

# Separate data at an inflection point

inflection = np.int(3.5*7*24)
xa = x[:inflection]
ya = y[:inflection]

xb = x[inflection:]
yb = y[inflection:]

fa = np.poly1d(np.polyfit(xa, ya, 1))
fb = np.poly1d(np.polyfit(xb, yb, 1))

print(error(fa, xa, ya), error(fb, xb, yb))

# Plotting for inflection point

plt.scatter(x,y)
plt.xlabel("time")
plt.ylabel("hits per hour")
values_a = np.linspace(xa[0], xa[-1], 1000)
values_b = np.linspace(xb[0], xb[-1], 1000)
plt.plot(values_a, fa(values_a), linewidth=1, color="red")
plt.plot(values_b, fb(values_b), linewidth=1, color="green")
plt.show()

# Separate data for training and testing

train_length = np.int(len(x) * 80 /100)
x_train = x[:train_length]
y_train = y[:train_length]

x_test = x[train_length:]
y_test = x[train_length:]

err = []
for i in range(1, 10, 1):
    f = np.poly1d(np.polyfit(x_train, y_train, i))
    err.append(error(f, x_test, y_test))

print(err.index(min(err)))
