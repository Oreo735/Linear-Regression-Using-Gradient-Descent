import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('faithful.txt', delimiter=' ')

X = data[:, 0]
y = data[:, 1]

X = X.reshape((X.shape[0], 1))
y = y.reshape((y.shape[0], 1))

fig, axes = plt.subplots(3, 1, figsize=(8, 15))
fig.tight_layout(pad=6.0)
# <Part 1 - סעיף א>
axes[0].plot(X, y, 'xr')
axes[0].set_title("Raw Data")
axes[0].set(xlabel='Duration of minutes of the eruption', ylabel='Time to next eruption')


# </Part 1 - סעיף א>


def hypothesis(m, b, x):
    h = (m * x + b)
    return h


def cost_computation(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * alpha * (X.T.dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cost_computation(theta, X, y)
    return theta, cost_history, theta_history


alpha = 0.0002

iterations = 2000
theta = np.random.randn(2, 1)
x_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history, theta_history = gradient_descent(x_b, y, theta, alpha, iterations)
Y_line = X * theta[1][0] + theta[0][0]
axes[1].plot(X, y, 'xr')
axes[1].plot(X, Y_line)
axes[1].set_title("Raw Data with Regression Line")

axes[2].set_title("J(theta) Conversion alpha = 0.0002")
axes[2].set_ylabel('J(Theta)')
axes[2].set_xlabel('Iterations')
_ = axes[2].plot(range(iterations), cost_history, 'b.')

axes[2].set(xlabel='Iterations', ylabel='J(Theta)')
plt.show()


def predict(minutes):
    return hypothesis(theta[1][0], theta[0][0], minutes)


print("\n\nPredicted Next Eruption for 2.5 minutes: ", predict(2.5))
print("Predicted Next Eruption for 3.7 minutes: ", predict(3.7))
print("Predicted Next Eruption for 5 minutes: ", predict(5))
