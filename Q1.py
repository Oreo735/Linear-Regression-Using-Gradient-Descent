import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()

a = 2.5
b = 10
x = 50 * np.random.rand(500)
y = a * x + b + np.random.normal(0, 5)
plt.scatter(x, y)

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 50, 10000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
print("Model slope a1 = ", model.coef_[0])
print("Model intercept a0 = ", model.intercept_)
