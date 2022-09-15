import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.name = "linear regression"
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    # fit algorithm using train data
    def fit(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        # init weights and bias
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for itr in range(self.n_iters):
            # get predictions
            y_preds = np.dot(X_train, self.weights) + self.bias
            
            # gradient descent
            # compute gradients of weights and bias
            dweights = (1/n_samples) * np.dot(X_train.T, (y_preds - y_train))
            dbias = (1/n_samples) * np.sum(y_preds - y_train)
            
            # update weights and bias
            self.weights = self.weights - self.lr * dweights
            self.bias = self.bias - self.lr * dbias
    
    # predict y using test data
    def predict(self, X_test:np.ndarray) -> list:
        y_preds = np.dot(X_test, self.weights) + self.bias
        return y_preds