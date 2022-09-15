from algo.knn import knn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def run_knn():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    clf = knn(k=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    acc = np.sum(predictions == y_test) / len(y_test)
    print(f"{acc:.4f} - accuracy {clf.name}")

if __name__ == '__main__':
    run_knn()