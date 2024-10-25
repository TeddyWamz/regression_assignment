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
