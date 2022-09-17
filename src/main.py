from algo.knn import Knn
from algo.decision_tree import DecisionTree
from algo.linear_reg import LinearRegression
from algo.logistic_reg import LogisticRegression
from algo.naive_bayes import NaiveBayes
from algo.random_forest import RandomForest
import numpy as np
from algo.random_forest import RandomForest
import util
from sklearn import datasets
from sklearn.model_selection import train_test_split

#--- load data ----------------------------------------------------------------#

# load multi class classification data
iris = datasets.load_iris()
X_multi_class, y_muli_class = iris.data, iris.target
X_multic_train, X_multic_test, y_multic_train, y_multic_test = train_test_split(
    X_multi_class,
    y_muli_class,
    test_size=0.2
)

# load binary class classification data
bc = datasets.load_breast_cancer()
X_bin_class, y_bin_class = bc.data, bc.target
X_binc_train, X_binc_test, y_binc_train, y_binc_test = train_test_split(
    X_bin_class,
    y_bin_class,
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
    acc = util.accuracy(y_test, predictions)
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
    acc = util.accuracy(y_test, predictions)
    print(f"{acc:.4f} - accuracy - {clf.name}")

def run_decision_tree(
    X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray
) -> None:
    # apply decision tree algorithm
    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    # compute algorithm performance
    acc = util.accuracy(y_test, predictions)
    print(f"{acc:.4f} - accuracy - {clf.name}")

def run_random_forest(
    X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray
) -> None:
    # apply decision tree algorithm
    clf = RandomForest(n_trees=10, min_samples_split=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    # compute algorithm performance
    acc = util.accuracy(y_test, predictions)
    print(f"{acc:.4f} - accuracy - {clf.name}")

def run_naive_bayes(
    X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray
) -> None:
    # apply naive bayes algorithm
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    # compute algorithm performance
    acc = util.accuracy(y_test, predictions)
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
    error = util.mse(y_test, predictions)
    print(f"{error:.4f} - error - {reg.name}")


#--- run algorithm pipelines --------------------------------------------------#

if __name__ == '__main__':
    run_knn(X_multic_train, X_multic_test, y_multic_train, y_multic_test)
    run_logistic_reg(X_binc_train, X_binc_test, y_binc_train, y_binc_test)
    run_decision_tree(X_binc_train, X_binc_test, y_binc_train, y_binc_test)
    run_random_forest(X_binc_train, X_binc_test, y_binc_train, y_binc_test)
    run_naive_bayes(X_binc_train, X_binc_test, y_binc_train, y_binc_test)
    run_linear_reg(X_reg_train, X_reg_test, y_reg_train, y_reg_test)