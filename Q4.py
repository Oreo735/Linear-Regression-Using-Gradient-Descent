import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('houses.txt', delimiter=',')


def data_normalization(X):
    areas = X[:, 0]
    bedrooms = X[:, 1]
    prices = X[:, 2]

    areas_average = np.average(areas)
    bedrooms_average = np.average(bedrooms)
    prices_average = np.average(prices)

    areas_std = np.std(areas)
    bedrooms_std = np.std(bedrooms)
    price_std = np.std(prices)

    normalized_areas = (areas - areas_average) / areas_std
    normalized_bedrooms = (bedrooms - bedrooms_average) / bedrooms_std
    normalized_prices = (prices - prices_average) / price_std
    ndata = [normalized_areas, normalized_bedrooms, normalized_prices]
    averages = [areas_average, bedrooms_average, prices_average]
    deviations = [areas_std, bedrooms_std, price_std]
    return ndata, averages, deviations


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


ndata, averages, deviations = data_normalization(data)
areas = np.array(ndata[0])
bedrooms = np.array(ndata[1])
X = np.array([areas, bedrooms]).T

y = np.array(ndata[2])

y = y.reshape((y.shape[0], 1))

alpha = 0.0005

iterations = 10000
theta = np.random.randn(3, 1)
x_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history = gradient_descent(x_b, y, theta, alpha, iterations)

theta0 = theta[0]
theta1 = theta[1]
theta2 = theta[2]
h = theta0 + theta1 * 1800 + theta2 * 7

print("Predicted price for 1800 ft² and 7 bedroom house is: ", h[0], "thousand dollars")

theta_unnormalized = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
h = theta_unnormalized[0][0] * 1800 + theta_unnormalized[1][0] * 7
print("Predicted price for same house with normal equations: ", h, "thousand dollars")

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
_ = ax.plot(range(iterations), cost_history, 'b.')

plt.show()
