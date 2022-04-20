import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Xdata = pd.read_csv("kleibers_law_data.csv")
data = Xdata.to_numpy()
mass = data[:, 0]
daily_kJoul = data[:, 1]

log_mass = np.log(mass)
log_rate = np.log(daily_kJoul)

fig, ax = plt.subplots(figsize=(18, 6))

ax.set_xlabel('log of mass')
ax.set_ylabel('log of metabolic rate')
ax.scatter(log_mass, log_rate, color='black')

plt.show()


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


X = log_mass
y = log_rate

X = X.reshape((X.shape[0], 1))
y = y.reshape((y.shape[0], 1))

alpha = 0.0005

iterations = 10000
theta = np.random.randn(2, 1)
x_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history = gradient_descent(x_b, y, theta, alpha, iterations)
print(theta)


def mass_to_rate(theta, mass):
    log_y = theta[1][0] * np.log(mass) + theta[0][0]
    return np.exp(log_y)


def rate_to_mass(theta, rate):
    log_x = (np.log(rate) - theta[0][0]) / theta[1][0]
    return np.exp(log_x)


fig, ax = plt.subplots(figsize=(12, 8))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
_ = ax.plot(range(iterations), cost_history, 'b.')

print("Predicted daily calories consumption by 25 kg mammal:  ", mass_to_rate(theta, 25) / 4.18)
print("Predicted weight of mammal that has rate of 2.5kJoul/day:  ", rate_to_mass(theta, 2.5))
plt.show()
