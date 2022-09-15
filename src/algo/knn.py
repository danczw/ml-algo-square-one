from collections import Counter
import numpy as np

def euclidean_distance(x1:int, x2:int) -> float:
    # compute Euclidean distance between two points
    return np.sqrt(np.sum((x1-x2)**2))

class Knn:
    def __init__(self, k=3):
        self.name = "knn"
        self.k = k
    
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        self.X_train = X_train
        self.y_train = y_train
    
    # predict y using test data
    def predict(self, X_test:np.ndarray) -> list:
        y_preds = [self._predict(x) for x in X_test]
        return y_preds
    
    # helper function to predict y using test data
    def _predict(self, x:int) -> int:
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get clostest k points
        k_nearest_indeces = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indeces]
        
        # majority vote
        y_pred = Counter(k_nearest_labels).most_common()[0][0]
        return y_pred