import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Xdata = pd.read_csv("transistor_counts.csv")
data = Xdata.to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))


# the transformation will be the normalization of the number of transistors

def normalize_data(X):
    transistors = X[:, 1]
    transistors_average = np.average(transistors)
    transistors_std = np.std(transistors)

    normalized_transistors = (transistors - transistors_average) / transistors_std
    ndata = np.array([X[:, 0], normalized_transistors]).T
    return ndata, transistors_average, transistors_std


def cost_computation(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * alpha * (X.T.dot((prediction - y)))
        cost_history[it] = cost_computation(theta, X, y)
    return theta, cost_history


ndata, transistors_average, transistors_std = normalize_data(data)

X = ndata[:, 0]
y = ndata[:, 1]
X = X.reshape((X.shape[0], 1))
y = y.reshape((y.shape[0], 1))
alpha = 11.9e-11

iterations = 7000

theta = np.random.randn(2, 1)
x_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history = gradient_descent(x_b, y, theta, alpha, iterations)

axes[0].set_ylabel('J(Theta)')
axes[0].set_xlabel('Iterations')
axes[0].set_title('Transformed')
_ = axes[0].plot(range(iterations), cost_history, 'b.')

X = data[:, 0]
y = data[:, 1]
X = X.reshape((X.shape[0], 1))
y = y.reshape((y.shape[0], 1))
theta = np.random.randn(2, 1)
x_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history = gradient_descent(x_b, y, theta, alpha, iterations)

axes[1].set_ylabel('J(Theta)')
axes[1].set_xlabel('Iterations')
axes[1].set_title('Untransformed')
_ = axes[1].plot(range(iterations), cost_history, 'b.')

print("Predicted number of transistors on cpu's made in year 2022 is: ", (theta[0][0] + 2022 * theta[1][0]))

plt.show()
