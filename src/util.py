import numpy as np

# cumpute accuracy of classification algorithm
def accuracy(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return np.sum(y_true == y_pred) / len(y_true)

# compute algorithm error using mean squared error
def mse(y_true:np.ndarray, y_preds:np.ndarray) -> float:
    return np.mean((y_true - y_preds)**2)