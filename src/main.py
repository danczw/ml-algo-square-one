from algo.knn import Knn
from algo.linear_reg import LinearRegression, mse
from algo.logistic_reg import LogisticRegression
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

#--- load data ----------------------------------------------------------------#

# load classification data
iris = datasets.load_iris()
X_class, y_class = iris.data, iris.target
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class,
    y_class,
    test_size=0.2
)

# load regression data
X_reg, y_reg = datasets.make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2
)

#--- def algorithm pipelines --------------------------------------------------#

# knn classification pipeline
def run_knn(
    X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray
) -> None:
    # apply knn algorithm
    clf = Knn(k=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    # compute algorithm performance
    acc = np.sum(predictions == y_test) / len(y_test)
    print(f"{acc:.4f} - accuracy - {clf.name}")

# logistic regression pipeline
def run_logistic_reg(
    X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray
) -> None:
    # apply logistic regression algorithm
    clf = LogisticRegression(lr=0.01, n_iters=1000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    # compute algorithm performance
    acc = np.sum(predictions == y_test) / len(y_test)
    print(f"{acc:.4f} - accuracy - {clf.name}")

# linear regression pipeline
def run_linear_reg(
    X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray
) -> None:
    # apply linear regression algorithm
    reg = LinearRegression(lr=0.01, n_iters=1000)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    
    # compute algorithm performance
    error = mse(y_test, predictions)
    print(f"{error:.4f} - error - {reg.name}")


#--- run algorithm pipelines --------------------------------------------------#

if __name__ == '__main__':
    run_knn(X_class_train, X_class_test, y_class_train, y_class_test)
    run_logistic_reg(X_class_train, X_class_test, y_class_train, y_class_test)
    run_linear_reg(X_reg_train, X_reg_test, y_reg_train, y_reg_test)