import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()


# <Part 1 - סעיף א>
def drawCricketData():
    data = np.load('Cricket.npz')['arr_0']

    chirps_per_sec = data[:, 0]
    temperatures = data[:, 1]

    plt.scatter(temperatures, chirps_per_sec)
    plt.ylabel("Temperature (°F)")
    plt.xlabel("Chirps/Second")
    plt.show()


# </Part 1 - סעיף א>


# <Part 2 - סעיף ב>

table = np.load("Cricket.npz")
data = table['arr_0']
X = data[:, 1]
y = data[:, 0]

X = X.reshape((X.shape[0], 1))
y = y.reshape((y.shape[0], 1))


def hypothesis(m, b, x):
    h = (m * x + b)
    return h


def compute_cost(theta, X, y):
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
        cost_history[it] = compute_cost(theta, X, y)
    return theta, cost_history, theta_history


# alpha = 0.01
# iterations 15000 - Thetas become zeros

# alpha = 0.001
# iterations 15000 - Thetas become zeros

# alpha = 0.0001
# iterations 15000 - too much iterations

# alpha = 0.0001
# iterations 1500 - alpha too big

# alpha = 0.00001
# iterations 1500 - not bad, we can do better

# alpha = 0.000001
# iterations 1500 - much better but too many iterations

# alpha = 0.000001
# iterations 150     - alpha too small

# alpha = 0.000005
# iterations 150     - PERFECT!

alpha = 0.000005

iterations = 150
theta = np.random.randn(2, 1)
x_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history, theta_history = gradient_descent(x_b, y, theta, alpha, iterations)

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
_ = ax.plot(range(iterations), cost_history, 'b.')

plt.show()

print("Theta0:", theta[0][0])
print("Theta1:", theta[1][0])


# </Part 2 - סעיף ב>

# <Part 3 and 4 - סעיף ג/ד>
def predict_chirps(temperature):
    return hypothesis(theta[1][0], theta[0][0], temperature)


print("\n\nPredicted Chirps/Sec for 91 degrees: ", predict_chirps(91))
print("Predicted Chirps/Sec for 77 degrees: ", predict_chirps(77))
print("Predicted Chirps/Sec for 35 degrees: ", predict_chirps(35))
# </Part 3 and 4 - סעיף ג/ד>
