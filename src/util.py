import numpy as np

# cumpute accuracy of classification algorithm
def accuracy(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return np.sum(y_true == y_pred) / len(y_true)

# compute algorithm error using mean squared error
def mse(y_true:np.ndarray, y_preds:np.ndarray) -> float:
    return np.mean((y_true - y_preds)**2)

# train test pipeline for regression algorithms
def regression_pipeline(
    regression:object,
    X_train:np.ndarray,
    X_test:np.ndarray,
    y_train:np.ndarray,
    y_test:np.ndarray
):
    # fit algorithm using train data
    regression.fit(X_train, y_train)
    # predict y using test data
    y_pred = regression.predict(X_test)
    # compute error of algorithm
    error = mse(y_test, y_pred)
    print(f"{error:.4f} - error - {regression.name}")
    
    return error

# train test pipeline for classification algorithms
def classification_pipeline(
    classifier:object,
    X_train:np.ndarray,
    X_test:np.ndarray,
    y_train:np.ndarray,
    y_test:np.ndarray
):
    # fit algorithm using train data
    classifier.fit(X_train, y_train)
    # predict y using test data
    y_pred = classifier.predict(X_test)
    # compute accuracy of algorithm
    acc = accuracy(y_test, y_pred)
    print(f"{acc:.4f} - accuracy - {classifier.name}")
    
    return acc