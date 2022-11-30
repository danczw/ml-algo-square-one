import numpy as np

# print colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
) -> float:
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
) -> float:
    # fit algorithm using train data
    classifier.fit(X_train, y_train)
    # predict y using test data
    y_pred = classifier.predict(X_test)
    # compute accuracy of algorithm
    acc = accuracy(y_test, y_pred)
    print(f"{acc:.4f} - accuracy - {classifier.name}")
    
    return acc

# train test pipeline for dimensionality reduction algorithms
def dim_reduction_pipeline(
    dim_reduction:object,
    X_train:np.ndarray,
    X_test:np.ndarray
) -> np.ndarray:
    # fit algorithm using train data
    dim_reduction.fit(X_train)
    # predict y using test data
    X_projected = dim_reduction.transform(X_test)
    # show reduced X shape  
    print(f"{X_projected.shape} - components - {dim_reduction.name}")
    
    return X_projected

# train test pipeline for clustering algorithms
def k_clustering_pipeline(
    clustering:object,
    X_train:np.ndarray,
    y_train:np.ndarray
) -> np.ndarray:
    # fit algorithm using train data
    k = clustering.predict(X_train)
    # show number of predicted and real clusters
    print(f"{len(np.unique(k))} - clusters - {clustering.name}")
    clustering.plot()