import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Nairobi Office Price Ex.csv'
data = pd.read_csv(file_path)

x = data['SIZE'].values
y = data['PRICE'].values

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * x + c
    dm = (-2 / n) * np.sum(x * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)

    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

m = np.random.randn()
c = np.random.randn()
learning_rate = 0.0001
epochs = 10

mse_history = []

for epoch in range(epochs):
    # Gradient descent
    m, c = gradient_descent(x, y, m, c, learning_rate)

    # Calculate mean squared error for the current epoch
    y_pred = m * x + c
    mse = mean_squared_error(y, y_pred)
    mse_history.append(mse)

    # Error at each epoch
    print(f"Epoch {epoch + 1}, MSE: {mse:.4f}")

plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Line of best fit')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.title('Graph of Office Size in sq. ft. vs. Price')
plt.legend()
plt.show()


print(f"Final slope (m): {m}")
print(f"Final intercept (c): {c}")


predicted_price = m * 100 + c
print(f"The predicted office price for a 100 sq. ft office is: {predicted_price:.2f}")